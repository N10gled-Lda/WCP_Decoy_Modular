"""
Simple Detector - Direct Pulse to Detector ID Mapping.

Takes a Pulse (with polarization angle and photon count) and returns which 
detector (0-3) would fire, based on:
1. Polarization angle analysis
2. Malus law for detection probability
3. Quantum efficiency and noise effects
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import IntEnum

from ...utils.data_structures import Pulse, Basis


class DetectorId(IntEnum):
    """Detector identifiers mapped to polarization angles."""
    H = 0  # Horizontal (0°)
    V = 1  # Vertical (90°)  
    D = 2  # Diagonal (45°)
    A = 3  # Anti-diagonal (135°)


@dataclass
class SimpleDetectorConfig:
    """Configuration for simple detector."""
    # Detection parameters
    quantum_efficiency: float = 0.8  # Probability of detecting a photon
    dark_count_rate_hz: float = 100.0  # Dark counts per second
    detection_threshold: int = 1  # Minimum photons for detection
    
    # Measurement basis
    measurement_basis: Basis = Basis.Z  # Current measurement basis (Z or X)
    
    # Physics parameters
    apply_malus_law: bool = True  # Apply cos²θ intensity dependence
    perfect_detection: bool = False  # Perfect detection (no noise/losses)
    
    # Angular tolerance
    angular_tolerance_degrees: float = 22.5  # ±22.5° around ideal angles


class SimpleDetector:
    """
    Simplified detector that maps pulse polarization to detector ID.
    
    Process:
    1. Analyze pulse polarization angle
    2. Determine which detector should fire based on measurement basis
    3. Apply quantum efficiency and Malus law
    4. Return detector ID (0-3) or None if no detection
    
    Detector mapping:
    - Z basis: H(0°) → DetectorId.H(0), V(90°) → DetectorId.V(1)
    - X basis: D(45°) → DetectorId.D(2), A(135°) → DetectorId.A(3)
    """
    
    def __init__(self, config: SimpleDetectorConfig = None):
        """Initialize simple detector."""
        self.config = config or SimpleDetectorConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Detection statistics
        self.total_pulses = 0
        self.detected_pulses = 0
        self.detection_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        
        self.logger.info(f"Simple detector initialized")
        self.logger.info(f"Measurement basis: {self.config.measurement_basis.value}")
        self.logger.info(f"Perfect detection: {self.config.perfect_detection}")

    def detect_pulse(self, pulse: Pulse) -> Optional[int]:
        """
        Detect a pulse and return detector ID.
        
        Args:
            pulse: Input pulse with polarization angle (degrees) and photon count
            
        Returns:
            Detector ID (0-3) or None if no detection
        """
        self.total_pulses += 1
        
        # Check if we have enough photons
        if pulse.photons < self.config.detection_threshold:
            self.logger.debug(f"Pulse rejected: {pulse.photons} photons < threshold {self.config.detection_threshold}")
            return None
        
        # Determine which detector should fire based on polarization
        detector_id = self._determine_detector_from_polarization(pulse.polarization)
        
        if detector_id is None:
            self.logger.debug(f"No detector found for polarization {pulse.polarization}°")
            return None
        
        # Perfect detection mode - always detect
        if self.config.perfect_detection:
            self._record_detection(detector_id)
            return detector_id
        
        # Apply quantum efficiency
        if np.random.random() > self.config.quantum_efficiency:
            self.logger.debug(f"Pulse lost to quantum efficiency")
            return None
        
        # Apply Malus law if enabled
        if self.config.apply_malus_law:
            detection_probability = self._calculate_malus_probability(pulse.polarization, detector_id)
            if np.random.random() > detection_probability:
                self.logger.debug(f"Pulse lost to Malus law (P={detection_probability:.3f})")
                return None
        
        # Detection successful
        self._record_detection(detector_id)
        self.logger.debug(f"Pulse detected: {pulse.polarization}° → detector {detector_id}")
        return detector_id

    def _determine_detector_from_polarization(self, polarization_degrees: float) -> Optional[int]:
        """
        Determine which detector fires based on polarization angle.
        
        Args:
            polarization_degrees: Polarization angle in degrees
            
        Returns:
            Detector ID (0-3) or None if no photons
        """
        # Normalize angle to 0-180° range
        angle = polarization_degrees % 180
        
        # Define detector angles and tolerance
        tolerance = self.config.angular_tolerance_degrees
        
        # Detector angle mappings
        detector_angles = {
            0: 0.0,    # H detector (horizontal)
            1: 90.0,   # V detector (vertical)
            2: 45.0,   # D detector (diagonal)
            3: 135.0   # A detector (anti-diagonal)
        }
        
        # BB84 Protocol Logic:
        # 1. If pulse matches measurement basis → deterministic detection
        # 2. If pulse doesn't match measurement basis → random detection (50/50)
        
        if self.config.measurement_basis == Basis.Z:
            # Measuring in Z basis (H/V detectors)
            
            # Check if pulse is close to Z basis angles (0° or 90°)
            close_to_h = abs(angle - 0) <= tolerance or abs(angle - 180) <= tolerance
            close_to_v = abs(angle - 90) <= tolerance
            
            if close_to_h:
                return 0  # H detector (same basis - deterministic)
            elif close_to_v:
                return 1  # V detector (same basis - deterministic)
            else:
                # Different basis (e.g., D/A pulse measured in Z basis)
                # Random detection between H and V (50/50)
                return 0 if np.random.random() < 0.5 else 1
                
        elif self.config.measurement_basis == Basis.X:
            # Measuring in X basis (D/A detectors)
            
            # Check if pulse is close to X basis angles (45° or 135°)
            close_to_d = abs(angle - 45) <= tolerance
            close_to_a = abs(angle - 135) <= tolerance
            
            if close_to_d:
                return 2  # D detector (same basis - deterministic)
            elif close_to_a:
                return 3  # A detector (same basis - deterministic)
            else:
                # Different basis (e.g., H/V pulse measured in X basis)
                # Random detection between D and A (50/50)
                return 2 if np.random.random() < 0.5 else 3
        
        # Should never reach here with valid measurement basis
        return None

    def _calculate_malus_probability(self, polarization_degrees: float, detector_id: int) -> float:
        """
        Calculate detection probability using Malus law.
        
        Args:
            polarization_degrees: Pulse polarization angle
            detector_id: Target detector ID
            
        Returns:
            Detection probability (0.0 to 1.0)
        """
        # Normalize angle
        angle = polarization_degrees % 180
        
        # For BB84, the Malus law depends on the basis alignment, not the specific detector
        # Same basis → high probability (close to 1.0)
        # Different basis → medium probability (~0.5)
        
        if self.config.measurement_basis == Basis.Z:
            # Measuring in Z basis
            # Check if pulse is aligned with Z basis (0° or 90°)
            distance_to_z_basis = min(abs(angle - 0), abs(angle - 90), abs(angle - 180))
        else:
            # Measuring in X basis  
            # Check if pulse is aligned with X basis (45° or 135°)
            distance_to_x_basis = min(abs(angle - 45), abs(angle - 135))
            distance_to_z_basis = distance_to_x_basis
        
        # Apply Malus law: I = I₀ * cos²(θ)
        # For same basis: angle difference ≈ 0°, cos²(0°) = 1.0
        # For different basis: angle difference ≈ 45°, cos²(45°) = 0.5
        angle_diff_rad = np.radians(distance_to_z_basis)
        probability = np.cos(angle_diff_rad) ** 2
        
        # Ensure minimum detection probability (quantum noise, etc.)
        return max(0.1, min(1.0, probability))

    def _record_detection(self, detector_id: int) -> None:
        """Record a successful detection."""
        self.detected_pulses += 1
        self.detection_counts[detector_id] += 1

    def set_measurement_basis(self, basis: Basis) -> None:
        """Set the measurement basis."""
        self.config.measurement_basis = basis
        self.logger.info(f"Measurement basis set to {basis.value}")

    def set_perfect_detection(self, enable: bool = True) -> None:
        """Enable/disable perfect detection mode."""
        self.config.perfect_detection = enable
        self.logger.info(f"Perfect detection: {'enabled' if enable else 'disabled'}")

    def set_quantum_efficiency(self, efficiency: float) -> None:
        """Set quantum efficiency (0.0 to 1.0)."""
        self.config.quantum_efficiency = max(0.0, min(1.0, efficiency))
        self.logger.info(f"Quantum efficiency set to {self.config.quantum_efficiency}")

    def get_detection_rate(self) -> float:
        """Get overall detection rate."""
        return self.detected_pulses / max(1, self.total_pulses)

    def get_detector_statistics(self) -> dict:
        """Get detection statistics."""
        total_detected = max(1, self.detected_pulses)
        return {
            "total_pulses": self.total_pulses,
            "detected_pulses": self.detected_pulses,
            "detection_rate": self.get_detection_rate(),
            "detector_counts": self.detection_counts.copy(),
            "detector_rates": {
                detector: count / total_detected 
                for detector, count in self.detection_counts.items()
            }
        }

    def reset_statistics(self) -> None:
        """Reset detection statistics."""
        self.total_pulses = 0
        self.detected_pulses = 0
        self.detection_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.logger.info("Detection statistics reset")

    def detector_name(self, detector_id: int) -> str:
        """Get detector name from ID."""
        names = {0: "H", 1: "V", 2: "D", 3: "A"}
        return names.get(detector_id, f"Unknown({detector_id})")

    def get_measurement_basis(self) -> Basis:
        """Get current measurement basis."""
        return self.config.measurement_basis


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create detector with default config
    detector = SimpleDetector()
    
    # Test pulses with different polarizations
    test_pulses = [
        Pulse(polarization=0.0, photons=1),    # Should hit H detector (0)
        Pulse(polarization=90.0, photons=1),   # Should hit V detector (1)
        Pulse(polarization=45.0, photons=1),   # Should hit D detector (2) if X basis
        Pulse(polarization=135.0, photons=1),  # Should hit A detector (3) if X basis
        Pulse(polarization=22.5, photons=1),   # Between H and D
        Pulse(polarization=0.0, photons=0),    # No photons - should not detect
    ]
    
    print("Testing Z basis measurement:")
    detector.set_measurement_basis(Basis.Z)
    for i, pulse in enumerate(test_pulses):
        result = detector.detect_pulse(pulse)
        detector_name = detector.detector_name(result) if result is not None else "None"
        print(f"Pulse {i}: {pulse.polarization}° → detector {result} ({detector_name})")
    
    print("\nTesting X basis measurement:")
    detector.set_measurement_basis(Basis.X)
    detector.reset_statistics()
    for i, pulse in enumerate(test_pulses):
        result = detector.detect_pulse(pulse)
        detector_name = detector.detector_name(result) if result is not None else "None"
        print(f"Pulse {i}: {pulse.polarization}° → detector {result} ({detector_name})")
    
    print(f"\nDetection statistics:")
    stats = detector.get_detector_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")