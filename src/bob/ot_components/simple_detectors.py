"""
Simple Detector System - Photon Count to Detector Number Mapping.

Converts polarization measurements to detector numbers using:
1. Photon count based detection
2. Malus law (sin²θ or cos²θ) for angle-dependent detection
3. Simple detector numbering (0=H, 1=V, 2=D, 3=A)
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

from src.utils.data_structures import Pulse, Basis, Bit


@dataclass
class SimpleDetectorConfig:
    """Simple configuration for detector system."""
    # Detection efficiency  
    quantum_efficiency: float = 0.8  # Probability of detecting a photon
    
    # Dark count parameters
    dark_count_rate_hz: float = 100.0  # Dark counts per second
    
    # Detection threshold
    photon_threshold: int = 1  # Minimum photons needed for detection
    
    # Timing parameters
    dead_time_ns: float = 50.0  # Dead time in nanoseconds
    
    # Angle-dependent detection (Malus law)
    apply_malus_law: bool = True  # Apply cos²θ or sin²θ dependence
    perfect_detection: bool = False  # Perfect detection (no noise/efficiency)


class SimpleDetectorSystem:
    """
    Simplified detector system for QKD.
    
    Converts polarization measurements to detector events:
    - Input: Pulse with basis and bit information
    - Output: Detector number (0-3) and detection confidence
    
    Detector mapping:
    - 0: H detector (horizontal polarization)
    - 1: V detector (vertical polarization)  
    - 2: D detector (diagonal +45°)
    - 3: A detector (anti-diagonal -45°)
    """
    
    def __init__(self, config: SimpleDetectorConfig):
        """
        Initialize simple detector system.
        
        Args:
            config: Detector configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Detection statistics
        self.detection_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.total_detections = 0
        self.dark_counts = 0
        
        # Timing for dark count simulation
        self._last_detection_time = 0.0
        
        self.logger.info("Simple detector system initialized")
        self.logger.info(f"Quantum efficiency: {config.quantum_efficiency}")
        self.logger.info(f"Perfect detection: {config.perfect_detection}")

    def detect_pulse(self, pulse: Pulse, measurement_basis: Basis) -> Optional[int]:
        """
        Detect a pulse and return detector number.
        
        Args:
            pulse: Input pulse with polarization angle (degrees) and photon count
            measurement_basis: Current measurement basis
            
        Returns:
            Detector number (0-3) or None if no detection
        """
        # Check photon threshold first
        if pulse.photons < self.config.photon_threshold:
            self.logger.debug(f"Pulse below threshold ({pulse.photons} photons)")
            return None
        
        # Determine which detector should fire based on polarization angle
        detector_number = self._determine_detector_from_angle(pulse.polarization, measurement_basis)
        
        if detector_number is None:
            return None
        
        # Perfect detection mode
        if self.config.perfect_detection:
            self._record_detection(detector_number)
            return detector_number
        
        # Apply quantum efficiency
        if np.random.random() > self.config.quantum_efficiency:
            self.logger.debug(f"Pulse not detected (quantum efficiency)")
            return None
        
        # Apply Malus law if enabled
        if self.config.apply_malus_law:
            detection_probability = self._calculate_malus_probability(pulse.polarization, detector_number, measurement_basis)
            if np.random.random() > detection_probability:
                self.logger.debug(f"Pulse lost to Malus law (P={detection_probability:.3f})")
                return None
        
        # Detection successful
        self._record_detection(detector_number)
        self.logger.debug(f"Pulse detected on detector {detector_number}")
        return detector_number

    def detect_pulse_batch(self, pulses: List[Pulse], measurement_bases: List[Basis]) -> List[Optional[int]]:
        """
        Detect a batch of pulses.
        
        Args:
            pulses: List of input pulses
            measurement_bases: List of measurement bases for each pulse
            
        Returns:
            List of detector numbers (None for no detection)
        """
        detections = []
        
        for pulse, basis in zip(pulses, measurement_bases):
            detection = self.detect_pulse(pulse, basis)
            detections.append(detection)
        
        return detections

    def _determine_detector_from_angle(self, polarization_degrees: float, measurement_basis: Basis) -> Optional[int]:
        """
        Determine which detector should fire based on polarization angle.
        
        Args:
            polarization_degrees: Polarization angle in degrees
            measurement_basis: Current measurement basis
            
        Returns:
            Detector number (0-3) or None if no clear match
        """
        # Normalize angle to 0-180° range
        angle = polarization_degrees % 180
        
        # Define angular tolerance for detector selection
        tolerance = 22.5  # degrees
        
        # Detector mappings
        if measurement_basis == Basis.Z:
            # Z basis: H(0°) and V(90°)
            if abs(angle - 0) <= tolerance or abs(angle - 180) <= tolerance:
                return 0  # H detector
            elif abs(angle - 90) <= tolerance:
                return 1  # V detector
        else:
            # X basis: D(45°) and A(135°)
            if abs(angle - 45) <= tolerance:
                return 2  # D detector
            elif abs(angle - 135) <= tolerance:
                return 3  # A detector
        
        return None  # No clear detector match

    def _calculate_malus_probability(self, polarization_degrees: float, detector_number: int, measurement_basis: Basis) -> float:
        """
        Calculate detection probability using Malus law.
        
        Args:
            polarization_degrees: Pulse polarization angle
            detector_number: Target detector number
            measurement_basis: Current measurement basis
            
        Returns:
            Detection probability (0.0 to 1.0)
        """
        # Detector angles
        detector_angles = {0: 0.0, 1: 90.0, 2: 45.0, 3: 135.0}
        detector_angle = detector_angles[detector_number]
        
        # Calculate angle difference
        angle_diff = abs(polarization_degrees - detector_angle)
        if angle_diff > 90:
            angle_diff = 180 - angle_diff
        
        # Malus law: I = I₀ * cos²(θ)
        angle_diff_rad = np.radians(angle_diff)
        probability = np.cos(angle_diff_rad) ** 2
        
        return probability

    def _record_detection(self, detector_number: int) -> None:
        """Record a detection event."""
        self.detection_counts[detector_number] += 1
        self.total_detections += 1
        self._last_detection_time = np.random.random() * 1e-6  # Random microsecond timestamp

    def generate_dark_counts(self, time_window_s: float) -> List[int]:
        """
        Generate dark count events for a time window.
        
        Args:
            time_window_s: Time window in seconds
            
        Returns:
            List of detector numbers with dark counts
        """
        if self.config.perfect_detection:
            return []
        
        # Calculate expected dark counts
        expected_counts = self.config.dark_count_rate_hz * time_window_s
        actual_counts = np.random.poisson(expected_counts)
        
        # Randomly distribute among detectors
        dark_count_detectors = []
        for _ in range(actual_counts):
            detector = np.random.randint(0, 4)  # 0-3
            dark_count_detectors.append(detector)
            self.dark_counts += 1
        
        return dark_count_detectors

    def convert_detections_to_bits(self, detections: List[Optional[int]], 
                                  measurement_bases: List[Basis]) -> List[Optional[Bit]]:
        """
        Convert detector numbers to bit values.
        
        Args:
            detections: List of detector numbers (None for no detection)
            measurement_bases: Measurement basis for each detection
            
        Returns:
            List of bit values (None for no detection or wrong basis)
        """
        bits = []
        
        for detection, basis in zip(detections, measurement_bases):
            if detection is None:
                bits.append(None)
                continue
            
            # Convert detector number to bit based on measurement basis
            if basis == Basis.Z:
                # Z basis: H=0, V=1
                if detection == 0:  # H detector
                    bits.append(Bit.ZERO)
                elif detection == 1:  # V detector
                    bits.append(Bit.ONE)
                else:
                    bits.append(None)  # Wrong basis detectors
            else:
                # X basis: D=0, A=1
                if detection == 2:  # D detector
                    bits.append(Bit.ZERO)
                elif detection == 3:  # A detector
                    bits.append(Bit.ONE)
                else:
                    bits.append(None)  # Wrong basis detectors
        
        return bits

    def get_detection_statistics(self) -> Dict[str, float]:
        """Get detection statistics."""
        total = max(1, self.total_detections)  # Avoid division by zero
        
        return {
            "total_detections": self.total_detections,
            "dark_counts": self.dark_counts,
            "detection_rates": {
                f"detector_{i}": count / total for i, count in self.detection_counts.items()
            },
            "average_detection_rate": total / max(1, self._last_detection_time)
        }

    def reset_statistics(self) -> None:
        """Reset detection statistics."""
        self.detection_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.total_detections = 0
        self.dark_counts = 0
        self._last_detection_time = 0.0
        self.logger.info("Detection statistics reset")

    def set_perfect_detection(self, enable: bool = True) -> None:
        """Enable or disable perfect detection mode."""
        self.config.perfect_detection = enable
        self.logger.info(f"Perfect detection: {'enabled' if enable else 'disabled'}")

    def set_quantum_efficiency(self, efficiency: float) -> None:
        """Set detector quantum efficiency (0.0 to 1.0)."""
        self.config.quantum_efficiency = max(0.0, min(1.0, efficiency))
        self.logger.info(f"Quantum efficiency set to {self.config.quantum_efficiency}")

    def set_dark_count_rate(self, rate_hz: float) -> None:
        """Set dark count rate in Hz."""
        self.config.dark_count_rate_hz = max(0.0, rate_hz)
        self.logger.info(f"Dark count rate set to {rate_hz} Hz")

    def detector_number_to_name(self, detector_number: int) -> str:
        """Convert detector number to name."""
        names = {0: "H", 1: "V", 2: "D", 3: "A"}
        return names.get(detector_number, f"Unknown({detector_number})")

    def detector_name_to_number(self, detector_name: str) -> Optional[int]:
        """Convert detector name to number."""
        names = {"H": 0, "V": 1, "D": 2, "A": 3}
        return names.get(detector_name.upper())
