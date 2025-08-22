"""
Simple Optical Table - Minimal Polarization Measurement.

Provides straightforward polarization measurement that can either:
1. Pass polarization unchanged
2. Apply angular deviation
3. Perfect measurement (no errors)
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum

from ...utils.data_structures import Pulse, Basis, Bit


class SimpleDetectorId(Enum):
    """Simple detector identifiers."""
    H = 0  # Horizontal
    V = 1  # Vertical  
    D = 2  # Diagonal (+45°)
    A = 3  # Anti-diagonal (-45°)


@dataclass
class SimpleOpticalConfig:
    """Simple configuration for optical table."""
    # Basic operation modes
    perfect_measurement: bool = True  # Perfect polarization measurement
    apply_angular_deviation: bool = False  # Apply angular deviations
    
    # Angular deviation parameters (in degrees)
    angular_deviation_degrees: float = 0.0  # Fixed angular deviation
    random_angular_deviation: bool = False  # Random deviations
    max_random_deviation_degrees: float = 5.0  # Max random deviation
    
    # Measurement basis alignment
    basis_alignment_error_degrees: float = 0.0  # Systematic basis misalignment


class SimpleOpticalTable:
    """
    Simplified optical table for polarization measurement.
    
    Key simplifications:
    1. Direct polarization to detector mapping
    2. Malus law (cos²θ) for intensity calculation
    3. Simple angular deviations
    4. Perfect or imperfect measurement modes
    """
    
    def __init__(self, config: SimpleOpticalConfig):
        """
        Initialize simple optical table.
        
        Args:
            config: Optical table configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Current measurement basis
        self.current_basis = Basis.Z
        
        self.logger.info("Simple optical table initialized")
        self.logger.info(f"Perfect measurement: {config.perfect_measurement}")
        if config.apply_angular_deviation:
            self.logger.info(f"Angular deviation: {config.angular_deviation_degrees}°")

    def set_measurement_basis(self, basis: Basis) -> bool:
        """
        Set the measurement basis.
        
        Args:
            basis: Measurement basis (Z or X)
            
        Returns:
            True if successful
        """
        self.current_basis = basis
        self.logger.debug(f"Measurement basis set to {basis.value}")
        return True

    def measure_polarization(self, pulse: Pulse) -> Tuple[SimpleDetectorId, float]:
        """
        Measure polarization and determine which detector fires.
        
        Args:
            pulse: Input pulse
            
        Returns:
            Tuple of (detector_id, detection_probability)
        """
        # Get the angle between pulse polarization and measurement basis
        angle_diff = self._calculate_polarization_angle_difference(pulse.basis, self.current_basis)
        
        # Apply angular deviations if enabled
        if self.config.apply_angular_deviation:
            angle_diff += self._get_angular_deviation()
        
        # Apply basis alignment error
        angle_diff += np.radians(self.config.basis_alignment_error_degrees)
        
        # Determine which detector should fire based on polarization
        detector_id = self._determine_detector(pulse.basis, self.current_basis, angle_diff)
        
        # Calculate detection probability using Malus law
        detection_probability = self._calculate_detection_probability(angle_diff, pulse.intensity)
        
        self.logger.debug(f"Pulse {pulse.pulse_id}: basis {pulse.basis.value} → detector {detector_id.name}, P={detection_probability:.3f}")
        
        return detector_id, detection_probability

    def _calculate_polarization_angle_difference(self, pulse_basis: Basis, measurement_basis: Basis) -> float:
        """
        Calculate angle difference between pulse and measurement polarizations.
        
        Args:
            pulse_basis: Basis of the incoming pulse
            measurement_basis: Current measurement basis
            
        Returns:
            Angle difference in radians
        """
        # Define basis angles (in radians)
        basis_angles = {
            Basis.Z: 0.0,      # Z basis: 0° (H) and 90° (V)  
            Basis.X: np.pi/4   # X basis: 45° (D) and 135° (A)
        }
        
        pulse_angle = basis_angles[pulse_basis]
        measurement_angle = basis_angles[measurement_basis]
        
        return abs(pulse_angle - measurement_angle)

    def _get_angular_deviation(self) -> float:
        """Get angular deviation to apply."""
        if self.config.random_angular_deviation:
            # Random deviation within specified range
            max_dev = np.radians(self.config.max_random_deviation_degrees)
            return np.random.uniform(-max_dev, max_dev)
        else:
            # Fixed deviation
            return np.radians(self.config.angular_deviation_degrees)

    def _determine_detector(self, pulse_basis: Basis, measurement_basis: Basis, angle_diff: float) -> SimpleDetectorId:
        """
        Determine which detector should fire based on polarization.
        
        Args:
            pulse_basis: Original pulse basis
            measurement_basis: Current measurement basis  
            angle_diff: Angular difference
            
        Returns:
            Detector that should fire
        """
        if self.config.perfect_measurement:
            # Perfect measurement - deterministic detector selection
            if measurement_basis == Basis.Z:
                # Z basis measurement
                if pulse_basis == Basis.Z:
                    # Same basis - choose H or V based on bit value
                    return SimpleDetectorId.H if np.random.random() < 0.5 else SimpleDetectorId.V
                else:
                    # Different basis - random between H and V
                    return SimpleDetectorId.H if np.random.random() < 0.5 else SimpleDetectorId.V
            else:
                # X basis measurement  
                if pulse_basis == Basis.X:
                    # Same basis - choose D or A based on bit value
                    return SimpleDetectorId.D if np.random.random() < 0.5 else SimpleDetectorId.A
                else:
                    # Different basis - random between D and A
                    return SimpleDetectorId.D if np.random.random() < 0.5 else SimpleDetectorId.A
        else:
            # Imperfect measurement - use Malus law probability
            if measurement_basis == Basis.Z:
                # Measuring in Z basis
                prob_h = np.cos(angle_diff) ** 2
                return SimpleDetectorId.H if np.random.random() < prob_h else SimpleDetectorId.V
            else:
                # Measuring in X basis
                prob_d = np.cos(angle_diff) ** 2  
                return SimpleDetectorId.D if np.random.random() < prob_d else SimpleDetectorId.A

    def _calculate_detection_probability(self, angle_diff: float, intensity: float) -> float:
        """
        Calculate detection probability using Malus law.
        
        Args:
            angle_diff: Angular difference in radians
            intensity: Pulse intensity
            
        Returns:
            Detection probability (0.0 to 1.0)
        """
        # Malus law: I = I₀ * cos²(θ)
        malus_factor = np.cos(angle_diff) ** 2
        
        # Scale by pulse intensity
        probability = intensity * malus_factor
        
        # Ensure probability is between 0 and 1
        return max(0.0, min(1.0, probability))

    def detector_to_bit(self, detector_id: SimpleDetectorId, measurement_basis: Basis) -> Optional[Bit]:
        """
        Convert detector ID to bit value based on measurement basis.
        
        Args:
            detector_id: Which detector fired
            measurement_basis: Basis used for measurement
            
        Returns:
            Measured bit value (or None if ambiguous)
        """
        if measurement_basis == Basis.Z:
            # Z basis: H=0, V=1
            if detector_id == SimpleDetectorId.H:
                return Bit.ZERO
            elif detector_id == SimpleDetectorId.V:
                return Bit.ONE
        else:
            # X basis: D=0, A=1  
            if detector_id == SimpleDetectorId.D:
                return Bit.ZERO
            elif detector_id == SimpleDetectorId.A:
                return Bit.ONE
        
        return None

    def set_perfect_measurement(self, enable: bool = True) -> None:
        """Enable or disable perfect measurement mode."""
        self.config.perfect_measurement = enable
        self.logger.info(f"Perfect measurement: {'enabled' if enable else 'disabled'}")

    def set_angular_deviation(self, deviation_degrees: float, random: bool = False) -> None:
        """
        Set angular deviation parameters.
        
        Args:
            deviation_degrees: Fixed deviation or max random deviation
            random: Whether to use random deviations
        """
        self.config.angular_deviation_degrees = deviation_degrees
        self.config.random_angular_deviation = random
        self.config.apply_angular_deviation = True
        
        if random:
            self.config.max_random_deviation_degrees = deviation_degrees
            self.logger.info(f"Random angular deviation set: ±{deviation_degrees}°")
        else:
            self.logger.info(f"Fixed angular deviation set: {deviation_degrees}°")

    def set_basis_alignment_error(self, error_degrees: float) -> None:
        """Set systematic basis alignment error."""
        self.config.basis_alignment_error_degrees = error_degrees
        self.logger.info(f"Basis alignment error set to {error_degrees}°")

    def reset_to_perfect(self) -> None:
        """Reset to perfect measurement mode."""
        self.config.perfect_measurement = True
        self.config.apply_angular_deviation = False
        self.config.angular_deviation_degrees = 0.0
        self.config.basis_alignment_error_degrees = 0.0
        self.config.random_angular_deviation = False
        self.logger.info("Optical table reset to perfect measurement mode")

    def get_current_basis(self) -> Basis:
        """Get the current measurement basis."""
        return self.current_basis
