"""VOA Controller."""
import logging
import math
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

from src.alice.voa.voa_hardware import VOAHardwareDriver
from src.alice.voa.voa_simulator import VOASimulator
from src.utils.data_structures import DecoyState, DecoyInfo
from src.alice.qrng import QRNGSimulator, QRNGHardware, OperationMode
from .voa_base import BaseVOADriver


class DecoyInfoExtended(DecoyInfo):
    """Extended DecoyInfo class with helper methods for state-based access."""
    
    def get_intensity(self, state: DecoyState) -> float:
        """Get the intensity for a given decoy state."""
        state_str = str(state)  # This converts SIGNAL -> "signal", etc.
        if state_str in self.intensities:
            return self.intensities[state_str]
        else:
            raise ValueError(f"Unknown decoy state: {state}")
    
    def get_probability(self, state: DecoyState) -> float:
        """Get the probability for a given decoy state."""
        state_str = str(state)  # This converts SIGNAL -> "signal", etc.
        if state_str in self.probabilities:
            return self.probabilities[state_str]
        else:
            raise ValueError(f"Unknown decoy state: {state}")
    def set_intensity(self, state: DecoyState, intensity: float) -> None:
        """Set the intensity for a given decoy state."""
        state_str = str(state)
        self.intensities[state_str] = intensity

    def set_probability(self, state: DecoyState, probability: float, validate: bool = False) -> None:
        """Set the probability for a given decoy state."""
        state_str = str(state)
        self.probabilities[state_str] = probability
        if validate and abs(sum(self.probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("Probabilities must sum to 1.0 after modification")

    def set_intensities(self, signal: float, weak: float, vacuum: float) -> None:
        """Set all three decoy-state intensities at once."""
        self.intensities = {
            str(DecoyState.SIGNAL): signal,
            str(DecoyState.WEAK): weak,
            str(DecoyState.VACUUM): vacuum
        }

    def set_probabilities(self, signal: float, weak: float, vacuum: float, validate: bool = False) -> None:
        """Set all three decoy-state probabilities at once."""
        self.probabilities = {
            str(DecoyState.SIGNAL): signal,
            str(DecoyState.WEAK): weak,
            str(DecoyState.VACUUM): vacuum
        }
        if validate and abs(sum(self.probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("Decoy state probabilities must sum to 1.0")

@dataclass
class VOAOutput:
    """Output of the VOA controller containing pulse type and attenuation."""
    pulse_type: DecoyState
    attenuation_db: float
    target_intensity: float


class VOAController:
    """Controls the VOA with QRNG-based state selection and intensity management."""
    
    def __init__(self, 
                 driver: Union[BaseVOADriver, VOASimulator, VOAHardwareDriver],
                 physical: bool = False,
                 qrng_driver: Union[QRNGSimulator, QRNGHardware] = None,
                 decoy_info: Optional[DecoyInfoExtended] = None):
        """
        Initialize VOA controller.
        
        Args:
            driver: The driver to use for controlling the VOA (e.g., VOASimulator or VOAHardwareDriver)
            physical: Whether to use physical hardware or simulator (DEPRECATED)
            decoy_info: DecoyInfo configuration with intensities and probabilities
            n_pulse_initial: Initial number of photons per pulse (for attenuation calculation)
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.physical = physical
        self.n_pulse_initial = 1000
        self.current_attenuation = 0.0
        self.current_state = DecoyState.SIGNAL
        self.is_on = False  # Track if the controller is initialized
        
        # Handle configuration setup
        if decoy_info is not None:
            if not isinstance(decoy_info, DecoyInfoExtended):
                raise TypeError("decoy_info must be an instance of DecoyInfoExtended")
            self.decoy_info = decoy_info

        else:
            # Create default configuration
            self.decoy_info = DecoyInfoExtended()
            self.logger.warning("No DecoyInfo provided, using default configuration.")
        
        self.intensities_defined = True  # DecoyInfo always has default intensities
        self.probabilities_defined = True  # DecoyInfo always has default probabilities
        
        # # Initialize VOA hardware/simulator
        # if self.physical:
        #     self.voa = VOAHardware()
        #     self.qrng = QRNGHardware()
        # else:
        #     self.voa = VOASimulator()
        #     self.qrng = QRNGSimulator()
        self.voa_driver = driver
        self.qrng_driver = qrng_driver if qrng_driver else QRNGSimulator()
        
        # self.logger.info(f"VOA controller initialized in {'physical' if physical else 'simulation'} mode.")
        self.logger.info(f"Intensities: {self.decoy_info.intensities}")
        self.logger.info(f"Probabilities: {self.decoy_info.probabilities}")

    def set_decoy_info(self, decoy_info: DecoyInfoExtended) -> None:
        """Set the complete decoy information."""
        self.decoy_info = decoy_info
        logging.info(f"Decoy info updated: intensities={decoy_info.intensities}, probabilities={decoy_info.probabilities}")

    def update_intensity(self, state: DecoyState, intensity: float) -> None:
        """Update the intensity for a specific decoy state."""
        self.decoy_info.set_intensity(state, intensity)
        logging.info(f"Updated {state} intensity to {intensity}")

    def update_probability(self, state: DecoyState, probability: float) -> None:
        """Update the probability for a specific decoy state."""
        self.decoy_info.set_probability(state, probability, validate=True)
        logging.info(f"Updated {state} probability to {probability}")

    def update_intensities(self, signal: float = None, weak: float = None, vacuum: float = None) -> None:
        """Update multiple intensities at once."""
        if signal is not None and signal >= 0 and weak is not None and weak >= 0 and vacuum is not None and vacuum >= 0:
            self.decoy_info.set_intensities(signal, weak, vacuum)
            logging.info(f"Updated intensities: {self.decoy_info.intensities}")

    def update_probabilities(self, signal: float = None, weak: float = None, vacuum: float = None) -> None:
        """Update multiple probabilities at once."""
        if signal is not None and signal >= 0 and weak is not None and weak >= 0 and vacuum is not None and vacuum >= 0:
            self.decoy_info.set_probabilities(signal, weak, vacuum, validate=True)
            logging.info(f"Updated probabilities: {self.decoy_info.probabilities}")



    def get_random_state_by_probability(self, optional_probs: dict[DecoyState, float] = None) -> DecoyState:
        """
        Select a decoy state based on configured probabilities using biased QRNG bits.
        
        First, decide signal vs. decoy with a single biased bit (bias=p_signal).
        If decoy is chosen, decide weak vs. vacuum with another biased bit
        (bias = p_weak / (p_weak + p_vacuum)).
        
        Args:
            optional_probs: Optional dictionary of probabilities for each decoy state.
            If not provided, uses the probabilities from internal decoy_info.

        Returns:
            Selected DecoyState based on probabilities
        """
        # fetch probabilities
        if optional_probs:
            p_signal = optional_probs.get(DecoyState.SIGNAL, 0.0)
            p_weak = optional_probs.get(DecoyState.WEAK, 0.0)
            p_vacuum = optional_probs.get(DecoyState.VACUUM, 0.0)
        else:
            p_signal = self.decoy_info.get_probability(DecoyState.SIGNAL)
            p_weak = self.decoy_info.get_probability(DecoyState.WEAK)
            p_vacuum = self.decoy_info.get_probability(DecoyState.VACUUM)

        # decide signal vs. decoy
        bit = self.qrng_driver.get_random_bits_biased(mode=OperationMode.STREAMING, bias=p_signal)
        if bit == 0:
            selected_state = DecoyState.SIGNAL
        elif bit == 1:
            # decide weak vs. vacuum among decoys
            total_decoy = p_weak + p_vacuum
            if total_decoy <= 0:
                # fallback if no decoy probability
                selected_state = DecoyState.VACUUM
            else:
                bias_weak = p_weak / total_decoy
                bit2 = self.qrng_driver.get_random_bits_biased(mode=OperationMode.STREAMING, bias=bias_weak)
                selected_state = DecoyState.WEAK if bit2 == 0 else DecoyState.VACUUM
        else:
            raise ValueError(f"QRNG returned an unexpected value of bit (should be 0 or 1): {bit}")

        self.set_state(selected_state)
        logging.debug(f"Biased bits -> {selected_state} (p_signal={p_signal:.3f}, "
                      f"p_weak={p_weak:.3f}, p_vac={p_vacuum:.3f})")
        return selected_state

    def get_state_from_bits(self, bit1: int, bit2: int) -> DecoyState:
        """
        Convert two random bits to a decoy state.
        
        Mapping:
        00 -> SIGNAL (0)
        01 -> SIGNAL (0) - fallback to signal for simplicity
        10 -> WEAK (1) 
        11 -> VACUUM (2)
        
        Args:
            bit1: First random bit (0 or 1)
            bit2: Second random bit (0 or 1)
            
        Returns:
            Selected DecoyState
        """
        state_map = {
            (0, 0): DecoyState.SIGNAL,
            (0, 1): DecoyState.SIGNAL,
            (1, 0): DecoyState.WEAK,
            (1, 1): DecoyState.VACUUM
        }
        
        selected_state = state_map[(bit1, bit2)]
        self.set_state(selected_state)
        logging.debug(f"Bits ({bit1},{bit2}) -> {selected_state}")
        return selected_state

    def get_random_state_from_random_bits(self) -> DecoyState:
        """Select a random decoy state using QRNG."""
        # Get two random bits from QRNG
        bit1 = self.qrng_driver.get_random_bit(mode=OperationMode.STREAMING)
        bit2 = self.qrng_driver.get_random_bit(mode=OperationMode.STREAMING)

        # Convert to state
        state = self.get_state_from_bits(bit1, bit2)
        
        logging.debug(f"Random state selected: {state}")
        return state

    def calculate_attenuation_for_intensity(self, target_mu: float, nb_pulse: int = None) -> float:
        """
        Calculate attenuation needed to achieve target mean photon number.
        
        Formula: A_dB = 10 * log10(N_pulse / μ)
        
        Args:
            target_mu: Target mean photon number
            nb_pulse: Number of pulses to use for calculation.
                      If None, uses the initial n_pulse_initial (default 1000).

        Returns:
            Attenuation in dB
        """
        if target_mu <= 0:
            # For vacuum state (μ=0), use maximum attenuation
            return 100.0  # High attenuation for vacuum
        
        if nb_pulse is not None:
            if nb_pulse <= 0:
                raise ValueError("Number of pulses must be positive")
            self.n_pulse_initial = nb_pulse
        else:
            if self.n_pulse_initial <= 0:
                raise ValueError("Initial pulse photon number must be positive")
        
        attenuation_db = 10 * math.log10(self.n_pulse_initial / target_mu)
        
        # Ensure reasonable bounds
        attenuation_db = max(0.0, min(attenuation_db, 100.0))
        
        logging.debug(f"Calculated attenuation: {attenuation_db:.2f} dB for μ={target_mu}")
        return attenuation_db

    def calculate_attenuation_for_state(self, state: DecoyState) -> float:
        """Calculate the attenuation for a given decoy state."""        
        target_intensity = self.decoy_info.get_intensity(state)
        return self.calculate_attenuation_for_intensity(target_intensity)

    def generate_pulse_with_state_selection(self, use_probabilities: bool = True, optional_probs: dict[DecoyState, float] = None) -> VOAOutput:
        """
        Generate a pulse with random state selection and corresponding attenuation.
        
        Args:
            use_probabilities: If True, use probability-based selection.
                              If False, use uniform random selection with bit mapping.
            optional_probs: Optional dictionary of probabilities for each decoy state.
                            If not provided, uses the probabilities from internal decoy_info.

        Returns:
            VOAOutput containing pulse type, attenuation, and target intensity
        """
        # Select random state using appropriate method
        if use_probabilities:
            selected_state = self.get_random_state_by_probability(optional_probs)
        else:
            selected_state = self.get_random_state_from_random_bits()
        
        # Calculate attenuation for the selected state
        target_intensity = self.decoy_info.get_intensity(selected_state)
        attenuation = self.calculate_attenuation_for_intensity(target_intensity)
        
        # Set the attenuation on the VOA
        self.set_attenuation(attenuation)
        
        output = VOAOutput(
            pulse_type=selected_state,
            attenuation_db=attenuation,
            target_intensity=target_intensity
        )
        
        logging.debug(f"Generated pulse: Type={selected_state}, "
                     f"Attenuation={attenuation:.2f}dB, μ={target_intensity}")
        
        return output


    def set_attenuation(self, attenuation_db: float) -> None:
        """Set the VOA attenuation to the specified value in the driver simulator/hardware."""
        self.voa_driver.set_attenuation(attenuation_db)
        self.current_attenuation = attenuation_db
        logging.debug(f"VOA attenuation set to {attenuation_db:.2f} dB")

    def get_attenuation(self) -> float:
        """Get the current VOA attenuation."""
        return self.current_attenuation

    def set_state(self, state: DecoyState) -> None:
        """Manually select a specific VOA state."""
        self.current_state = state
        
        attenuation = self.calculate_attenuation_for_state(state)
        self.set_attenuation(attenuation)
        
        logging.debug(f"State manually selected: {state}")

    def get_current_state(self) -> DecoyState:
        """Get the current VOA state."""
        return self.current_state
    
    def get_current_attenuation(self) -> float:
        """Get the current VOA attenuation."""
        return self.current_attenuation

    def get_current_target_intensity(self) -> float:
        """Get the current target intensity."""
        return self.decoy_info.get_intensity(self.current_state)

    def initialize(self) -> None:
        """Initialize the VOA and QRNG."""
        # VOA initialization would happen in the hardware/simulator constructors
        self.is_on = True
        self.voa_driver.initialize()
        logging.info("VOA controller initialized")

    def shutdown(self) -> None:
        """Shutdown the VOA and QRNG."""
        self.is_on = False
        self.voa_driver.shutdown()
        logging.info("VOA controller shutdown")

    def reset(self) -> None:
        """Reset the VOA to its default state."""
        self.current_attenuation = 0.0
        self.current_state = DecoyState.SIGNAL
        self.set_attenuation(0.0)
        logging.info("VOA controller reset")


