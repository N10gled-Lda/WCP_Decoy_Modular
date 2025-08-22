"""
Simple Quantum Channel - Minimal Implementation.

Provides a straightforward channel that can either:
1. Pass pulses through unchanged
2. Apply simple attenuation in dB
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Union
from ..utils.data_structures import Pulse


@dataclass
class SimpleChannelConfig:
    """Simple configuration for quantum channel."""
    # Basic attenuation
    attenuation_db: float = 0.0  # Total attenuation in dB
    
    # Simple mode flags
    pass_through_mode: bool = True  # If True, no modifications applied
    apply_attenuation: bool = False  # If True, apply attenuation_db
    
    # Optional random loss
    random_loss_probability: float = 0.0  # Probability of pulse loss (0.0 to 1.0)


class SimpleQuantumChannel:
    """
    Simplified quantum channel implementation.
    
    Modes of operation:
    1. Pass-through: Pulse goes through unchanged
    2. Attenuation: Apply fixed attenuation in dB
    3. Random loss: Some pulses are randomly lost
    """
    
    def __init__(self, config: SimpleChannelConfig):
        """
        Initialize simple quantum channel.
        
        Args:
            config: Channel configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.logger.info("Simple quantum channel initialized")
        self.logger.info(f"Pass-through mode: {config.pass_through_mode}")
        if config.apply_attenuation:
            self.logger.info(f"Attenuation: {config.attenuation_db} dB")
        if config.random_loss_probability > 0:
            self.logger.info(f"Random loss probability: {config.random_loss_probability}")

    def transmit_pulse(self, pulse: Pulse) -> Optional[Pulse]:
        """
        Transmit a single pulse through the channel.
        
        Args:
            pulse: Input pulse
            
        Returns:
            Output pulse (None if lost)
        """
        # Pass-through mode - no changes
        if self.config.pass_through_mode:
            return pulse
        
        # Check for random loss first
        if self.config.random_loss_probability > 0:
            if np.random.random() < self.config.random_loss_probability:
                self.logger.debug(f"Pulse {pulse.pulse_id} randomly lost")
                return None
        
        # Apply attenuation if enabled
        if self.config.apply_attenuation and self.config.attenuation_db > 0:
            # Convert dB to linear attenuation factor
            attenuation_factor = 10 ** (-self.config.attenuation_db / 10)
            
            # Create new pulse with reduced intensity
            attenuated_pulse = Pulse(
                pulse_id=pulse.pulse_id,
                timestamp=pulse.timestamp,
                basis=pulse.basis,
                bit=pulse.bit,
                intensity=pulse.intensity * attenuation_factor,
                wavelength_nm=pulse.wavelength_nm,
                duration_ns=pulse.duration_ns
            )
            
            self.logger.debug(f"Pulse {pulse.pulse_id} attenuated by {self.config.attenuation_db} dB")
            return attenuated_pulse
        
        # No modifications applied
        return pulse

    def transmit_pulses(self, pulses: List[Pulse]) -> List[Pulse]:
        """
        Transmit a batch of pulses through the channel.
        
        Args:
            pulses: List of input pulses
            
        Returns:
            List of output pulses (excluding lost pulses)
        """
        output_pulses = []
        
        for pulse in pulses:
            transmitted_pulse = self.transmit_pulse(pulse)
            if transmitted_pulse is not None:
                output_pulses.append(transmitted_pulse)
        
        self.logger.debug(f"Transmitted {len(output_pulses)}/{len(pulses)} pulses")
        return output_pulses

    def set_attenuation(self, attenuation_db: float) -> None:
        """Set channel attenuation in dB."""
        self.config.attenuation_db = attenuation_db
        self.config.apply_attenuation = True
        self.config.pass_through_mode = False
        self.logger.info(f"Channel attenuation set to {attenuation_db} dB")

    def set_pass_through_mode(self, enable: bool = True) -> None:
        """Enable or disable pass-through mode."""
        self.config.pass_through_mode = enable
        if enable:
            self.config.apply_attenuation = False
        self.logger.info(f"Pass-through mode: {'enabled' if enable else 'disabled'}")

    def set_random_loss(self, probability: float) -> None:
        """Set random loss probability (0.0 to 1.0)."""
        self.config.random_loss_probability = max(0.0, min(1.0, probability))
        if probability > 0:
            self.config.pass_through_mode = False
        self.logger.info(f"Random loss probability set to {self.config.random_loss_probability}")

    def get_transmission_efficiency(self) -> float:
        """
        Calculate overall transmission efficiency.
        
        Returns:
            Transmission efficiency (0.0 to 1.0)
        """
        efficiency = 1.0
        
        # Account for attenuation
        if self.config.apply_attenuation:
            efficiency *= 10 ** (-self.config.attenuation_db / 10)
        
        # Account for random loss
        efficiency *= (1.0 - self.config.random_loss_probability)
        
        return efficiency

    def reset_to_pass_through(self) -> None:
        """Reset channel to pass-through mode."""
        self.config.pass_through_mode = True
        self.config.apply_attenuation = False
        self.config.attenuation_db = 0.0
        self.config.random_loss_probability = 0.0
        self.logger.info("Channel reset to pass-through mode")
