import random
import numpy as np
from enum import Enum, auto
from typing import Tuple, Optional, Union

class PulseType(Enum):
    """Enumeration for different pulse types in WCP BB84 protocol"""
    # SIGNAL = "signal"
    # DECOY = "decoy" 
    # VACUUM = "vacuum"
    SIGNAL = auto()
    DECOY = auto()
    VACUUM = auto()

    def __str__(self):
        return self.name.lower()

class WCPPulse:
    """
    A class to represent a Weak Coherent Pulse (WCP) for BB84 protocol,
    managing Poisson-distributed photon numbers and different pulse intensities.
    """
    
    def __init__(self, bit=None, base=0, pulse_type=PulseType.SIGNAL, intensity=0.5):
        """
        Initialize the WCP pulse with a given bit, base, pulse type and intensity.
        
        :param bit: Optional, the initial bit of the pulse (0 or 1)
        :param base: The polarization base of the pulse (0 or 1)
        :param pulse_type: The type of pulse (signal, decoy, vacuum)
        :param intensity: The mean photon number (μ) for Poisson distribution
        """
        self.bit = bit
        self.base = base
        self.pulse_type = pulse_type
        self.intensity = intensity
        self.photon_number = None
        self.photon_number_float = None
        self._measured = False

        # Map pulse types to indices for encoding
        self.pulse_type_map = {
            PulseType.SIGNAL: 0,
            PulseType.DECOY: 1,
            PulseType.VACUUM: 2
        }

        # Generate photon number from Poisson distribution
        self._generate_photon_number()
        
    def _generate_photon_number(self):
        """Generate photon number from Poisson distribution based on intensity"""
        if self.pulse_type == PulseType.VACUUM:
            self.photon_number = 0
        else:
            self.photon_number = np.random.poisson(self.intensity)
    
    @staticmethod
    def pulse_info_to_byte(bit: int, base: int, pulse_type_idx: int) -> int:
        """
        Encode pulse information into a single byte for transmission. This will use bitwise operations to include
        
        :param bit: The bit value (0 or 1) in the least significant bit
        :param base: The base value (0 or 1) in the second least significant bit
        :param pulse_type_idx: Index of pulse type (0=signal, 1=decoy, 2=vacuum) in the third and fourth least significant bits
        :return: Encoded byte value (0-7) containing all information
            
            The resulting byte has the format: [unused][unused][unused][unused][pulse_type1][pulse_type0][base][bit]
            where the most significant bits are unused and left as zeros.
        """
        # Encode: 2 bits for pulse_type, 1 bit for base, 1 bit for bit
        # Format: [unused][unused][unused][unused][pulse_type1][pulse_type0][base][bit]
        return (pulse_type_idx << 2) | (base << 1) | bit
    
    @staticmethod
    def byte_to_pulse_info(byte: int) -> Tuple[int, int, int]:
        """
        Decode pulse information from a byte into the 3 components: bit, base, and pulse type index.
        
        :param byte: The encoded byte
        :return: Tuple of (bit, base, pulse_type_idx)
            - bit: Least significant bit (0 or 1)
            - base: Second least significant bit (0 or 1)
            - pulse_type_idx: Index of pulse type (0=signal, 1=decoy, 2=vacuum)
        """
        bit = byte & 0x01
        base = (byte >> 1) & 0x01
        pulse_type_idx = (byte >> 2) & 0x03
        return bit, base, pulse_type_idx
    
    @staticmethod
    def from_byte_and_intensity(byte: int, intensities: Union[dict, list]) -> 'WCPPulse':
        """
        Create a WCP pulse from encoded byte and intensity information.
        
        :param byte: Encoded pulse information
        :param intensities: Dictionary with intensities for each pulse type i.e. {'signal': μs, 'decoy': μd, 'vacuum': μv}
            Or a list with intensities in the order of [μs, μd, μv].
        :type intensities: dict or list
        :return: New WCPPulse instance
        """
        bit, base, pulse_type_idx = WCPPulse.byte_to_pulse_info(byte)
        
        pulse_types = [PulseType.SIGNAL, PulseType.DECOY, PulseType.VACUUM]
        pulse_type = pulse_types[pulse_type_idx]

        if isinstance(intensities, dict):
            intensity_map = {
                PulseType.SIGNAL: intensities.get('signal', 0.5),
                PulseType.DECOY: intensities.get('decoy', 0.1),
                PulseType.VACUUM: intensities.get('vacuum', 0.0)
            }
        elif isinstance(intensities, list):
            # Map list indices to pulse types
            intensity_map = {
                PulseType.SIGNAL: intensities[0] if len(intensities) > 0 else 0.5,
                PulseType.DECOY: intensities[1] if len(intensities) > 1 else 0.1,
                PulseType.VACUUM: intensities[2] if len(intensities) > 2 else 0.0
            }
        else:
            raise ValueError("Intensities must be a dictionary or a list.")
        
        intensity = intensity_map[pulse_type]
        return WCPPulse(bit, base, pulse_type, intensity)

    @staticmethod
    def from_parameters_and_intensity(bit: int, base: int, pulse_type: PulseType, intensities: Union[dict, list]) -> 'WCPPulse':
        """
        Create a WCP pulse from individual parameters and intensity information.
        
        :param bit: The bit value (0 or 1)
        :param base: The base value (0 or 1)
        :param pulse_type: The pulse type (PulseType enum)
        :param intensities: Dictionary with intensities for each pulse type i.e. {'signal': μs, 'decoy': μd, 'vacuum': μv}
            Or a list with intensities in the order of [μs, μd, μv].
        :type intensities: dict or list
        :return: New WCPPulse instance
        """
        if isinstance(intensities, dict):
            intensity_map = {
                PulseType.SIGNAL: intensities.get('signal', 0.5),
                PulseType.DECOY: intensities.get('decoy', 0.1),
                PulseType.VACUUM: intensities.get('vacuum', 0.0)
            }
        elif isinstance(intensities, list):
            # Map list indices to pulse types
            intensity_map = {
                PulseType.SIGNAL: intensities[0] if len(intensities) > 0 else 0.5,
                PulseType.DECOY: intensities[1] if len(intensities) > 1 else 0.1,
                PulseType.VACUUM: intensities[2] if len(intensities) > 2 else 0.0
            }
        else:
            raise ValueError("Intensities must be a dictionary or a list.")
        
        intensity = intensity_map[pulse_type]
        return WCPPulse(bit, base, pulse_type, intensity)
    
    def get_byte(self) -> int:
        """Get the byte representation of the pulse information"""

        pulse_type_idx = self.pulse_type_map[self.pulse_type]
        return self.pulse_info_to_byte(self.bit, self.base, pulse_type_idx)
    
    def get_type_byte(self) -> int:
        """Get the byte representation of the pulse type only (ignoring bit and base)
        Returns the pulse type index as 0, 1, or 2 for signal, decoy, or vacuum respectively.
        """
        return self.pulse_type_map[self.pulse_type]

    def measure(self, measurement_base: int, detector_efficiency: float = 0.1, dark_count_rate: float = 1e-6) -> Tuple[bool, Optional[int]]:
        """
        Simulate measurement of the WCP pulse including realistic detector effects.
        
        :param measurement_base: The base to measure in (0 or 1)
        :param detector_efficiency: Detector efficiency (η)
        :param dark_count_rate: Dark count probability
        :return: Tuple of (detected, measured_bit) where detected indicates if pulse was detected
        """
        if self._measured:
            raise Exception("Pulse already measured!")
        
        self._measured = True
        
        print(f"Starting measurement with parameters:")
        print(f"Measurement base: {measurement_base}, Detector efficiency: {detector_efficiency}, Dark count rate: {dark_count_rate}")
        print(f"Pulse photon number: {self.photon_number}, Pulse base: {self.base}, Pulse bit: {self.bit}")
        
        # Check if any photons are detected (including dark counts)
        detection_prob = 0.0

        if self.photon_number > 0:
            # Probability of detecting at least one photon
            detection_prob = 1 - (1 - detector_efficiency) ** self.photon_number
            print(f"Detection probability from photons: {detection_prob}")
        
        # Add dark count probability
        detection_prob = detection_prob + dark_count_rate * (1 - detection_prob)
        print(f"Total detection probability (including dark counts): {detection_prob}")
        
        detected = random.random() < detection_prob
        print(f"Detection result: {'Detected' if detected else 'Not detected'}")
        
        if not detected:
            return False, None
        
        # If detected, determine the measured bit
        if self.photon_number == 0:  # Dark count case
            measured_bit = random.choice([0, 1])
            print(f"Dark count case: Randomly chosen measured bit: {measured_bit}")
        elif self.base == measurement_base:
            # Correct basis measurement
            measured_bit = self.bit
            print(f"Correct basis measurement: Measured bit matches pulse bit: {measured_bit}")
        else:
            # Wrong basis measurement - random result
            measured_bit = random.choice([0, 1])
            print(f"Wrong basis measurement: Randomly chosen measured bit: {measured_bit}")
        
        return True, measured_bit
    
    @staticmethod
    def measure_byte(byte, measurement_base: int, intensities: dict, detector_efficiency: float = 0.1, dark_count_rate: float = 1e-6) -> Tuple[bool, Optional[int]]:
        """
        Static method to simulate measurement directly from byte representation.
        
        :param byte: Encoded pulse byte
        :param measurement_base: The base to measure in
        :param intensities: Dictionary of intensities for each pulse type
        :param detector_efficiency: Detector efficiency
        :param dark_count_rate: Dark count probability
        :return: Tuple of (detected, measured_bit)
        """
        pulse = WCPPulse.from_byte_and_intensity(byte, intensities)
        return pulse.measure(measurement_base, detector_efficiency, dark_count_rate)
    
    def apply_channel_loss(self, transmission_efficiency: float) -> None:
        """
        Apply channel loss by reducing the effective photon number.
        
        :param transmission_efficiency: Channel transmission efficiency (0-1)
        """
        if self.photon_number > 0:
            # Each photon has a probability of being transmitted
            transmitted_photons = 0
            for _ in range(self.photon_number):
                if random.random() < transmission_efficiency:
                    transmitted_photons += 1
            self.photon_number = transmitted_photons
    
    def get_intensity_type_index(self) -> int:
        """Get the index of the pulse type for categorization.
        
        Returns 0 for signal, 1 for decoy, 2 for vacuum."""
        return self.pulse_type_map[self.pulse_type]
    
    def __repr__(self):
        """String representation of the WCP pulse"""
        return (f"WCPPulse(bit={self.bit}, base={self.base}, "
                f"type={self.pulse_type.value}, intensity={self.intensity}, "
                f"photon_number={self.photon_number})")


class WCPIntensityManager:
    """
    Manages intensity selection and distribution for WCP BB84 protocol.
    """
    
    def __init__(self, signal_intensity=0.5, decoy_intensity=0.1, vacuum_intensity=0.0,
                 signal_prob=0.7, decoy_prob=0.25, vacuum_prob=0.05):
        """
        Initialize intensity manager with pulse intensities and probabilities.
        
        :param signal_intensity: Mean photon number for signal pulses (μs)
        :param decoy_intensity: Mean photon number for decoy pulses (μd)
        :param vacuum_intensity: Mean photon number for vacuum pulses (μv)
        :param signal_prob: Probability of sending signal pulse
        :param decoy_prob: Probability of sending decoy pulse
        :param vacuum_prob: Probability of sending vacuum pulse

        WARNING: Probabilities should sum to 1. If not, they will be normalized.
        """

        # Initialize intensities
        if signal_intensity < 0 or decoy_intensity < 0 or vacuum_intensity < 0:
            raise ValueError("Intensities must be non-negative. Please provide valid intensities.")
        # self.intensities_map = {
        #     'signal': signal_intensity,
        #     'decoy': decoy_intensity,
        #     'vacuum': vacuum_intensity
        # }
        self.intensities_map = {
            PulseType.SIGNAL: signal_intensity,
            PulseType.DECOY: decoy_intensity,
            PulseType.VACUUM: vacuum_intensity
        }
        
        # Initialize probabilities
        if signal_prob < 0 or decoy_prob < 0 or vacuum_prob < 0:
            raise ValueError("Probabilities must be non-negative. Please provide valid probabilities.")
        # Normalize probabilities
        total_prob = signal_prob + decoy_prob + vacuum_prob # This should be 1.0 ideally
        if total_prob <= 0:
            raise ValueError("Total probabilities cannot be zero or negative. Please provide valid probabilities.")
        if total_prob > 1.0: # Warning: This will normalize probabilities
            print("Warning: Probabilities exceed 1 ({}). Normalizing them to sum to 1.".format(total_prob))
        self.probabilities_map = {
            PulseType.SIGNAL: signal_prob / total_prob,
            PulseType.DECOY: decoy_prob / total_prob,
            PulseType.VACUUM: vacuum_prob / total_prob
        }
    
    def select_pulse_type(self) -> PulseType:
        """Randomly select a pulse type based on configured probabilities"""
        rand_val = random.random()
        cumulative_prob = 0.0
        
        # for pulse_type, prob in self.probabilities_map.items():
        #     cumulative_prob += prob
        #     if rand_val <= cumulative_prob:
        #         return pulse_type
        pulse_types = [PulseType.SIGNAL, PulseType.DECOY, PulseType.VACUUM]
        weights = [self.probabilities_map[pt] for pt in pulse_types]
        rand_pulse = random.choices(pulse_types, weights=weights)[0]

        if sum(weights) == 0:
            print("Warning: All probabilities are zero, returning SIGNAL as fallback.")
            return PulseType.SIGNAL
        if rand_pulse is None:
            print("Warning: Random pulse selection failed, returning SIGNAL as fallback.")
            return PulseType.SIGNAL

        return rand_pulse

        # If no type was selected (should not happen), return signal as fallback
        print("Warning: No pulse type selected, returning SIGNAL as fallback.")
        print(f"Random value: {rand_val}, Cumulative probabilities: {cumulative_prob}")
        print(f"Probabilities map: {self.probabilities_map}")
        return PulseType.SIGNAL  # Fallback
    
    def get_intensity(self, pulse_type: PulseType) -> float:
        """Get intensity for a given pulse type"""
        if pulse_type not in self.intensities_map:
            raise ValueError(f"Invalid pulse type: {pulse_type}. Available types are: {list(self.intensities_map.keys())}")
        return self.intensities_map[pulse_type]

    def create_pulse(self, bit, base) -> WCPPulse:
        """Create a new WCP pulse with randomly selected type and appropriate intensity"""
        pulse_type = self.select_pulse_type()
        intensity = self.get_intensity(pulse_type)
        return WCPPulse(bit, base, pulse_type, intensity)
