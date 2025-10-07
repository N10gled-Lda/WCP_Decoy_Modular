"""Time Tagger Controller - Unified interface for hardware and simulation with QKD detection."""
import logging
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time

from .timetagger_base import BaseTimeTaggerDriver, TimeTaggerConfig, ChannelConfig


class TimeTaggerDriverType(Enum):
    """Types of time tagger drivers."""
    HARDWARE = "hardware"
    SIMULATOR = "simulator"


class DetectionBasis(Enum):
    """Measurement bases for BB84 protocol."""
    Z = "Z"  # Rectilinear basis (H/V)
    X = "X"  # Diagonal basis (D/A)


class DetectionResult(Enum):
    """Possible detection outcomes."""
    NO_DETECTION = "no_detection"
    BIT_0 = "0" 
    BIT_1 = "1"
    AMBIGUOUS = "ambiguous"  # Multiple detectors fired


@dataclass
class QKDDetection:
    """A single QKD detection event."""
    measurement_basis: DetectionBasis
    bit_value: Optional[str]  # "0" or "1" or None
    detection_result: DetectionResult
    detector_counts: Dict[int, int]
    timestamp_ps: int
    detection_confidence: float  # 0.0 to 1.0


@dataclass
class TimeTaggerControllerConfig:
    """Configuration for the TimeTagger controller with QKD detection."""
    use_hardware: bool = False
    driver_type: TimeTaggerDriverType = TimeTaggerDriverType.SIMULATOR
    detector_channels: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    measurement_basis_channels: Dict[str, List[int]] = field(default_factory=lambda: {
        'Z': [1, 2],  # H/V detectors
        'X': [3, 4]   # D/A detectors
    })
    enable_gating: bool = True
    gate_begin_channel: int = 21
    gate_end_channel: int = -21
    gate_length_ps: int = 1000000
    measurement_duration_s: float = 60.0
    quantum_efficiency: float = 0.8
    dark_count_rate_hz: float = 100.0
    detection_jitter_ps: int = 1000
    
    # QKD Detection parameters
    detector_mapping: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        'Z': {'0': 1, '1': 2},  # H->bit 0, V->bit 1
        'X': {'0': 3, '1': 4}   # D->bit 0, A->bit 1
    })
    min_detection_threshold: int = 1
    max_dark_count_threshold: int = 10


class TimeTaggerController:
    """
    Unified controller for TimeTagger hardware and simulation.
    Similar to Alice's LaserController - takes driver in constructor.
    """
    
    def __init__(self, driver: BaseTimeTaggerDriver, config: TimeTaggerControllerConfig = None):
        """
        Initialize the TimeTagger controller with a specific driver.
        
        Args:
            driver: The TimeTagger driver (hardware or simulator)
            config: Configuration for QKD detection behavior
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config = config or TimeTaggerControllerConfig()
        self.driver = driver
        self._initialized = False
        self._current_basis = 'Z'
        self.last_detection_counts: Dict[int, int] = {}
        
        self.logger.info(f"TimeTagger controller initialized with {type(driver).__name__}")

    def initialize(self) -> bool:
        """Initialize the TimeTagger controller."""
        try:
            if self.driver and self.driver.initialize():
                self._initialized = True
                self.logger.info("TimeTagger controller initialized")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error initializing: {e}")
            return False

    def get_single_gate_counts(self) -> Dict[int, int]:
        """Get counts for a single gate window."""
        if not self._initialized:
            return {}
        try:
            counts = self.driver.get_single_gate_counts()
            self.last_detection_counts = counts.copy()
            return counts
        except Exception as e:
            self.logger.error(f"Failed to get counts: {e}")
            return {}

    def set_measurement_basis(self, basis: str) -> None:
        """Set the measurement basis (Z or X)."""
        if basis not in ['Z', 'X']:
            raise ValueError("Basis must be 'Z' or 'X'")
        self._current_basis = basis

    def get_active_detector_channels(self) -> List[int]:
        """Get detector channels for current basis."""
        return self.config.measurement_basis_channels.get(self._current_basis, [])

    def convert_counts_to_detection_result(self, counts: Dict[int, int]) -> Optional[int]:
        """Convert detection counts to a result."""
        if not counts:
            return None
        active_channels = self.get_active_detector_channels()
        fired_detectors = [ch for ch in active_channels if counts.get(ch, 0) > 0]
        
        if len(fired_detectors) == 1:
            return fired_detectors[0]
        elif len(fired_detectors) > 1:
            import random
            return random.choice(fired_detectors)
        return None

    def add_simulated_pulse(self, arrival_time_ps: int, polarization_degrees: float, 
                          photon_count: int = 1) -> None:
        """Add a simulated pulse."""
        if hasattr(self.driver, 'add_input_pulse'):
            self.driver.add_input_pulse(arrival_time_ps, polarization_degrees, photon_count)

    def get_detector_statistics(self) -> Dict[str, Any]:
        """Get detector statistics from the driver."""
        if not self._initialized or not self.driver:
            return {}
        
        try:
            # Get basic statistics from the driver
            if hasattr(self.driver, 'get_statistics'):
                stats = self.driver.get_statistics()  # Now returns Dict[str, Any]
                return {
                    'count_rates_hz': stats.get('count_rates_hz', {}),
                    'total_events': stats.get('total_events', 0),
                    'buffer_overflow_count': stats.get('buffer_overflows', 0),
                    'measurement_time_s': stats.get('measurement_time_s', 0.0)
                }
            else:
                # Return basic statistics for compatibility
                return {
                    'count_rates_hz': {},
                    'total_events': 0,
                    'buffer_overflow_count': 0,
                    'measurement_time_s': 0.0
                }
        except Exception as e:
            self.logger.error(f"Failed to get detector statistics: {e}")
            return {}

    def perform_qkd_detection(self, basis: DetectionBasis) -> QKDDetection:
        """
        Perform a QKD detection in the specified basis.
        This is the main method for QKD measurements.
        
        Args:
            basis: Measurement basis (Z or X)
            
        Returns:
            QKDDetection: Detection result with bit value and metadata
        """
        self.set_measurement_basis(basis.value)
        
        # Get detection counts from driver
        counts = self.get_single_gate_counts()
        timestamp_ps = int(time.time() * 1e12)
        
        # Process counts into QKD detection
        return self._process_detection_counts(counts, timestamp_ps, basis)

    def _process_detection_counts(self, counts: Dict[int, int], timestamp_ps: int, basis: DetectionBasis) -> QKDDetection:
        """Process raw detector counts into QKD detection result."""
        basis_str = basis.value
        
        # Get detector mapping for current basis
        detector_map = self.config.detector_mapping[basis_str]
        
        # Find which detectors fired above threshold
        fired_detectors = []
        for bit_val, channel in detector_map.items():
            if counts.get(channel, 0) >= self.config.min_detection_threshold:
                fired_detectors.append((bit_val, channel, counts[channel]))
        
        # Determine detection result
        if len(fired_detectors) == 0:
            # No detection
            bit_value = None
            result = DetectionResult.NO_DETECTION
            confidence = 0.0
            
        elif len(fired_detectors) == 1:
            # Clear detection
            bit_value = fired_detectors[0][0]
            result = DetectionResult.BIT_0 if bit_value == "0" else DetectionResult.BIT_1
            confidence = self._calculate_detection_confidence(counts, fired_detectors[0][1])
            
        else:
            # Multiple detectors fired - ambiguous
            # Choose the detector with highest count
            best_detector = max(fired_detectors, key=lambda x: x[2])
            bit_value = best_detector[0]
            result = DetectionResult.AMBIGUOUS
            confidence = 0.5  # Reduced confidence due to ambiguity
            
            self.logger.debug(f"Ambiguous detection: {fired_detectors}, chose {best_detector}")
        
        return QKDDetection(
            measurement_basis=basis,
            bit_value=bit_value,
            detection_result=result,
            detector_counts=counts.copy(),
            timestamp_ps=timestamp_ps,
            detection_confidence=confidence
        )

    def _calculate_detection_confidence(self, counts: Dict[int, int], fired_channel: int) -> float:
        """Calculate confidence score for a detection."""
        total_counts = sum(counts.values())
        if total_counts == 0:
            return 0.0
        
        fired_counts = counts.get(fired_channel, 0)
        
        # Confidence based on signal-to-noise ratio
        # Higher confidence when fired detector dominates
        confidence = fired_counts / total_counts
        
        # Apply quantum efficiency and dark count penalties
        if fired_counts <= self.config.max_dark_count_threshold:
            confidence *= 0.5  # Penalty for potentially dark counts
        
        return min(1.0, confidence)

    def shutdown(self) -> None:
        """Shutdown the controller."""
        if self.driver:
            self.driver.shutdown()
            self.driver = None
        self._initialized = False
