"""Detectors Simulator - Realistic Photon Detection."""
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import time

from ...utils.data_structures import Pulse


class DetectorType(Enum):
    """Types of photon detectors."""
    SPAD = "single_photon_avalanche_diode"
    PMT = "photomultiplier_tube"
    SNSPD = "superconducting_nanowire"
    APD = "avalanche_photodiode"


@dataclass
class DetectorConfig:
    """Configuration for photon detectors."""
    # Basic parameters
    detector_type: DetectorType = DetectorType.SPAD
    quantum_efficiency: float = 0.75
    dark_count_rate_hz: float = 100.0
    
    # Timing parameters
    dead_time_ns: float = 25.0
    timing_jitter_ps: float = 50.0
    gate_width_ns: float = 10.0
    
    # Nonlinearity and afterpulsing
    afterpulsing_probability: float = 0.01
    afterpulsing_time_constant_ns: float = 100.0
    
    # Temperature effects
    temperature_coefficient_per_k: float = -0.02
    operating_temperature_k: float = 273.15 + 20  # 20°C
    
    # Optical parameters
    active_area_diameter_um: float = 25.0
    numerical_aperture: float = 0.22
    
    # Count rate effects
    enable_saturation: bool = True
    saturation_count_rate_hz: float = 1e6


@dataclass
class DetectorStatistics:
    """Statistics for detector performance."""
    total_photons_incident: int = 0
    total_detections: int = 0
    dark_counts: int = 0
    afterpulse_counts: int = 0
    saturated_periods: int = 0
    dead_time_losses: int = 0
    
    # Timing statistics
    detection_times: List[float] = field(default_factory=list)
    timing_jitter_values: List[float] = field(default_factory=list)
    
    # Performance metrics
    current_count_rate_hz: float = 0.0
    detection_efficiency: float = 0.0


class PhotonDetectorSimulator:
    """
    Comprehensive photon detector simulator.
    
    Simulates realistic detector physics including:
    - Quantum efficiency and dark counts
    - Dead time and afterpulsing
    - Timing jitter and gate effects
    - Temperature dependencies
    - Count rate saturation
    - Various detector types (SPAD, PMT, SNSPD, APD)
    """
    
    def __init__(self, config: DetectorConfig, detector_id: str = "Det0"):
        """
        Initialize photon detector simulator.

        Args:
            config: Detector configuration
            detector_id: Unique identifier for this detector
        """
        self.config = config
        self.detector_id = detector_id
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}_{detector_id}")
        self.stats = DetectorStatistics()
        
        # State variables
        self.last_detection_time = 0.0
        self.afterpulse_queue: List[float] = []
        self.is_gated = False
        self.gate_start_time = 0.0
        
        # Count rate tracking
        self.recent_detections: List[float] = []
        self.count_window_s = 1.0  # 1 second window for count rate
        
        self.logger.info(f"Detector {detector_id} initialized: {config.detector_type.value}")
        self.logger.info(f"QE: {config.quantum_efficiency:.3f}, Dark rate: {config.dark_count_rate_hz} Hz")

    def detect_photon(self, pulse: Optional[Pulse] = None, detection_time: Optional[float] = None) -> Tuple[bool, float]:
        """
        Simulate photon detection.

        Args:
            pulse: Incident photon pulse (None for dark count simulation)
            detection_time: Time of detection attempt (if None, use current time)

        Returns:
            Tuple of (detection occurred, actual detection time)
        """
        current_time = detection_time if detection_time is not None else time.time()
        
        # Check if detector is in dead time
        if self._is_in_dead_time(current_time):
            self.stats.dead_time_losses += 1
            return False, current_time
        
        # Check gating
        if self.is_gated and not self._is_gate_open(current_time):
            return False, current_time
        
        # Check for saturation
        if self._is_saturated(current_time):
            self.stats.saturated_periods += 1
            return False, current_time
        
        detection_occurred = False
        detection_type = "none"
        
        if pulse is not None and pulse.photons > 0:
            # Actual photon detection
            detection_occurred = self._detect_signal_photon(pulse, current_time)
            if detection_occurred:
                detection_type = "signal"
                self.stats.total_photons_incident += pulse.photons
        else:
            # Check for dark count
            detection_occurred = self._detect_dark_count(current_time)
            if detection_occurred:
                detection_type = "dark"
        
        # Check for afterpulsing from previous detections
        afterpulse_detected = self._check_afterpulsing(current_time)
        if afterpulse_detected and not detection_occurred:
            detection_occurred = True
            detection_type = "afterpulse"
        
        if detection_occurred:
            actual_detection_time = self._apply_timing_jitter(current_time)
            self._record_detection(actual_detection_time, detection_type)
            return True, actual_detection_time
        
        return False, current_time

    def _detect_signal_photon(self, pulse: Pulse, current_time: float) -> bool:
        """Simulate detection of signal photons."""
        # Apply quantum efficiency
        effective_qe = self._get_effective_quantum_efficiency()
        
        # Each photon has independent detection probability
        for _ in range(pulse.photons):
            if np.random.random() < effective_qe:
                return True  # At least one photon detected
        
        return False

    def _detect_dark_count(self, current_time: float) -> bool:
        """Simulate dark count detection."""
        # Calculate probability of dark count in a small time window
        time_window = 1e-9  # 1 ns window
        dark_prob = self.config.dark_count_rate_hz * time_window
        
        return np.random.random() < dark_prob

    def _check_afterpulsing(self, current_time: float) -> bool:
        """Check for afterpulsing from previous detections."""
        # Remove expired afterpulse events
        cutoff_time = current_time - 10 * self.config.afterpulsing_time_constant_ns * 1e-9
        self.afterpulse_queue = [t for t in self.afterpulse_queue if t > cutoff_time]
        
        # Check probability of afterpulsing
        for afterpulse_time in self.afterpulse_queue:
            time_since_detection = current_time - afterpulse_time
            decay_factor = np.exp(-time_since_detection * 1e9 / self.config.afterpulsing_time_constant_ns)
            afterpulse_prob = self.config.afterpulsing_probability * decay_factor
            
            if np.random.random() < afterpulse_prob:
                return True
        
        return False

    def _is_in_dead_time(self, current_time: float) -> bool:
        """Check if detector is in dead time."""
        if self.last_detection_time == 0.0:
            return False
        
        dead_time_s = self.config.dead_time_ns * 1e-9
        return (current_time - self.last_detection_time) < dead_time_s

    def _is_gate_open(self, current_time: float) -> bool:
        """Check if detection gate is open."""
        if not self.is_gated:
            return True
        
        gate_duration = self.config.gate_width_ns * 1e-9
        return (current_time - self.gate_start_time) < gate_duration

    def _is_saturated(self, current_time: float) -> bool:
        """Check if detector is saturated."""
        if not self.config.enable_saturation:
            return False
        
        # Update recent detection rate
        self._update_count_rate(current_time)
        
        return self.stats.current_count_rate_hz > self.config.saturation_count_rate_hz

    def _get_effective_quantum_efficiency(self) -> float:
        """Calculate effective quantum efficiency including temperature effects."""
        # Temperature dependence
        temp_diff = self.config.operating_temperature_k - 293.15  # Relative to 20°C
        temp_factor = 1.0 + self.config.temperature_coefficient_per_k * temp_diff
        
        effective_qe = self.config.quantum_efficiency * temp_factor
        
        # Ensure QE stays within physical bounds
        return max(0.0, min(1.0, effective_qe))

    def _apply_timing_jitter(self, ideal_time: float) -> float:
        """Apply timing jitter to detection time."""
        jitter_s = np.random.normal(0, self.config.timing_jitter_ps * 1e-12)
        self.stats.timing_jitter_values.append(jitter_s * 1e12)  # Store in ps
        
        return ideal_time + jitter_s

    def _record_detection(self, detection_time: float, detection_type: str) -> None:
        """Record a detection event."""
        self.last_detection_time = detection_time
        self.stats.detection_times.append(detection_time)
        self.stats.total_detections += 1
        
        # Record specific detection types
        if detection_type == "dark":
            self.stats.dark_counts += 1
        elif detection_type == "afterpulse":
            self.stats.afterpulse_counts += 1
        
        # Add to afterpulse queue for future events
        self.afterpulse_queue.append(detection_time)
        
        # Update detection efficiency
        if self.stats.total_photons_incident > 0:
            signal_detections = self.stats.total_detections - self.stats.dark_counts - self.stats.afterpulse_counts
            self.stats.detection_efficiency = signal_detections / self.stats.total_photons_incident
        
        self.logger.debug(f"Detection at {detection_time:.9f}s ({detection_type})")

    def _update_count_rate(self, current_time: float) -> None:
        """Update current count rate based on recent detections."""
        # Keep only recent detections within the count window
        cutoff_time = current_time - self.count_window_s
        self.recent_detections = [t for t in self.recent_detections if t > cutoff_time]
        
        # Add current time if detection occurred
        self.recent_detections.append(current_time)
        
        # Calculate count rate
        self.stats.current_count_rate_hz = len(self.recent_detections) / self.count_window_s

    def open_gate(self, gate_start_time: Optional[float] = None) -> None:
        """Open detection gate for gated operation."""
        self.is_gated = True
        self.gate_start_time = gate_start_time if gate_start_time is not None else time.time()
        self.logger.debug(f"Detection gate opened at {self.gate_start_time:.9f}s")

    def close_gate(self) -> None:
        """Close detection gate."""
        self.is_gated = False
        self.logger.debug("Detection gate closed")

    def set_temperature(self, temperature_k: float) -> None:
        """Set detector operating temperature."""
        self.config.operating_temperature_k = temperature_k
        temp_c = temperature_k - 273.15
        self.logger.info(f"Detector temperature set to {temp_c:.1f}°C")

    def reset_detector(self) -> None:
        """Reset detector state (clear dead time, afterpulses, etc.)."""
        self.last_detection_time = 0.0
        self.afterpulse_queue.clear()
        self.recent_detections.clear()
        self.is_gated = False
        self.logger.info("Detector state reset")

    def get_statistics(self) -> DetectorStatistics:
        """Get detector performance statistics."""
        return self.stats

    def get_detector_info(self) -> Dict[str, Any]:
        """Get comprehensive detector information."""
        # Calculate derived metrics
        signal_detections = self.stats.total_detections - self.stats.dark_counts - self.stats.afterpulse_counts
        dark_count_ratio = self.stats.dark_counts / max(self.stats.total_detections, 1)
        afterpulse_ratio = self.stats.afterpulse_counts / max(self.stats.total_detections, 1)
        
        # Timing statistics
        avg_jitter_ps = np.mean(self.stats.timing_jitter_values) if self.stats.timing_jitter_values else 0.0
        rms_jitter_ps = np.std(self.stats.timing_jitter_values) if self.stats.timing_jitter_values else 0.0
        
        return {
            "detector_id": self.detector_id,
            "config": {
                "type": self.config.detector_type.value,
                "quantum_efficiency": self.config.quantum_efficiency,
                "dark_count_rate_hz": self.config.dark_count_rate_hz,
                "dead_time_ns": self.config.dead_time_ns,
                "timing_jitter_ps": self.config.timing_jitter_ps
            },
            "current_state": {
                "is_gated": self.is_gated,
                "in_dead_time": self._is_in_dead_time(time.time()),
                "current_count_rate_hz": self.stats.current_count_rate_hz,
                "operating_temperature_k": self.config.operating_temperature_k
            },
            "performance": {
                "total_detections": self.stats.total_detections,
                "signal_detections": signal_detections,
                "dark_count_ratio": dark_count_ratio,
                "afterpulse_ratio": afterpulse_ratio,
                "detection_efficiency": self.stats.detection_efficiency,
                "dead_time_losses": self.stats.dead_time_losses
            },
            "timing": {
                "average_jitter_ps": avg_jitter_ps,
                "rms_jitter_ps": rms_jitter_ps,
                "total_timing_measurements": len(self.stats.timing_jitter_values)
            }
        }

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.stats = DetectorStatistics()
        self.recent_detections.clear()
        self.logger.info("Detector statistics reset")

    def calibrate_detector(self) -> Dict[str, float]:
        """Perform detector calibration measurements."""
        self.logger.info(f"Performing detector {self.detector_id} calibration...")
        
        # Measure dark count rate
        dark_count_time_s = 10.0
        start_time = time.time()
        dark_counts = 0
        
        while time.time() - start_time < dark_count_time_s:
            if self.detect_photon(None)[0]:  # No pulse = dark count test
                dark_counts += 1
            time.sleep(0.001)  # 1 ms sampling
        
        measured_dark_rate = dark_counts / dark_count_time_s
        
        # Measure quantum efficiency with known photon source
        test_photons = 1000
        detected_count = 0
        
        for _ in range(test_photons):
            test_pulse = Pulse(polarization=0.0, photons=1)
            if self.detect_photon(test_pulse)[0]:
                detected_count += 1
        
        measured_qe = detected_count / test_photons
        
        calibration_results = {
            "theoretical_dark_rate_hz": self.config.dark_count_rate_hz,
            "measured_dark_rate_hz": measured_dark_rate,
            "theoretical_quantum_efficiency": self.config.quantum_efficiency,
            "measured_quantum_efficiency": measured_qe,
            "calibration_accuracy_dark": abs(measured_dark_rate - self.config.dark_count_rate_hz) / self.config.dark_count_rate_hz,
            "calibration_accuracy_qe": abs(measured_qe - self.config.quantum_efficiency) / self.config.quantum_efficiency
        }
        
        self.logger.info(f"Calibration complete - Dark rate: {measured_dark_rate:.1f} Hz, QE: {measured_qe:.3f}")
        
        return calibration_results


# Legacy class for backward compatibility
class DetectorsSimulator:
    """Legacy detector simulator for backward compatibility."""
    def __init__(self):
        config = DetectorConfig()
        self.detector = PhotonDetectorSimulator(config, "legacy")
        logging.info("Legacy detectors simulator initialized.")

    def detect_photon(self) -> bool:
        """Legacy method for backward compatibility."""
        return self.detector.detect_photon()[0]
