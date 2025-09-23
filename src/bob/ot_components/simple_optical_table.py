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
            pulse: Input pulse with polarization angle (degrees) and photon count
            
        Returns:
            Tuple of (detector_id, detection_probability)
        """
        # Convert polarization angle to determine which detector should fire
        detector_id = self._determine_detector_from_angle(pulse.polarization)
        
        # Apply angular deviations if enabled
        effective_angle = pulse.polarization
        if self.config.apply_angular_deviation:
            effective_angle += np.degrees(self._get_angular_deviation())
        
        # Apply basis alignment error
        effective_angle += self.config.basis_alignment_error_degrees
        
        # Calculate detection probability using Malus law and photon count
        detection_probability = self._calculate_detection_probability(effective_angle, pulse.photons)
        
        self.logger.debug(f"Pulse {pulse.polarization}°: → detector {detector_id.name}, P={detection_probability:.3f}")
        
        return detector_id, detection_probability

    def _determine_detector_from_angle(self, polarization_degrees: float) -> SimpleDetectorId:
        """
        Determine which detector should fire based on polarization angle.
        
        Args:
            polarization_degrees: Polarization angle in degrees
            
        Returns:
            Detector that should fire
        """
        # Normalize angle to 0-180° range
        angle = polarization_degrees % 180
        
        # Define detector angles
        detector_angles = {
            SimpleDetectorId.H: 0.0,    # Horizontal (0°)
            SimpleDetectorId.V: 90.0,   # Vertical (90°)
            SimpleDetectorId.D: 45.0,   # Diagonal (45°)
            SimpleDetectorId.A: 135.0   # Anti-diagonal (135°)
        }
        
        # If perfect measurement, choose deterministically based on measurement basis
        if self.config.perfect_measurement:
            if self.current_basis == Basis.Z:
                # Z basis: choose closest between H(0°) and V(90°)
                return SimpleDetectorId.H if abs(angle - 0) < abs(angle - 90) else SimpleDetectorId.V
            else:
                # X basis: choose closest between D(45°) and A(135°)
                return SimpleDetectorId.D if abs(angle - 45) < abs(angle - 135) else SimpleDetectorId.A
        else:
            # Imperfect measurement: probabilistic based on Malus law
            if self.current_basis == Basis.Z:
                # Z basis measurement
                angle_diff_h = abs(angle - 0)
                if angle_diff_h > 90:
                    angle_diff_h = 180 - angle_diff_h
                prob_h = np.cos(np.radians(angle_diff_h)) ** 2
                return SimpleDetectorId.H if np.random.random() < prob_h else SimpleDetectorId.V
            else:
                # X basis measurement
                angle_diff_d = abs(angle - 45)
                if angle_diff_d > 90:
                    angle_diff_d = 180 - angle_diff_d
                prob_d = np.cos(np.radians(angle_diff_d)) ** 2
                return SimpleDetectorId.D if np.random.random() < prob_d else SimpleDetectorId.A

    def _calculate_detection_probability(self, effective_angle: float, photon_count: int) -> float:
        """
        Calculate detection probability using Malus law.
        
        Args:
            effective_angle: Effective polarization angle in degrees (after deviations)
            photon_count: Number of photons in the pulse
            
        Returns:
            Detection probability (0.0 to 1.0)
        """
        if photon_count == 0:
            return 0.0
        
        # Normalize angle
        angle = effective_angle % 180
        
        # Calculate angle difference from measurement basis
        if self.current_basis == Basis.Z:
            # For Z basis, calculate distance from H(0°) or V(90°)
            angle_diff = min(abs(angle - 0), abs(angle - 90))
        else:
            # For X basis, calculate distance from D(45°) or A(135°)
            angle_diff = min(abs(angle - 45), abs(angle - 135))
        
        # Ensure angle difference is within 0-90°
        if angle_diff > 90:
            angle_diff = 180 - angle_diff
        
        # Malus law: I = I₀ * cos²(θ)
        malus_factor = np.cos(np.radians(angle_diff)) ** 2
        
        # Scale by photon count (assume linear relationship)
        probability = min(1.0, photon_count * malus_factor / 10.0)  # Normalize to reasonable scale
        
        return max(0.0, probability)

    def _get_angular_deviation(self) -> float:
        """Get angular deviation to apply (in radians)."""
        if self.config.random_angular_deviation:
            # Random deviation within specified range
            max_dev = np.radians(self.config.max_random_deviation_degrees)
            return np.random.uniform(-max_dev, max_dev)
        else:
            # Fixed deviation
            return np.radians(self.config.angular_deviation_degrees)

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
