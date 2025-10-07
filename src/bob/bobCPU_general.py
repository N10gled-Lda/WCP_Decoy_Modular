"""
Bob CPU - Main Controller for QKD Receiver.

Orchestrates all Bob-side components including optical table, detectors,
and basis selection to implement BB84 quantum key distribution protocol.
"""

import time
import logging
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from queue import Queue, Empty
from enum import Enum

from ..utils.data_structures import Pulse, Basis, Bit
from ..quantum_channel import FreeSpaceChannelSimulator, ChannelConfig
from .ot_components import (
    OpticalTableSimulator, OpticalTableConfig, MeasurementOutcome,
    PhotonDetectorSimulator, DetectorConfig, DetectorType
)
from .timetagger import (
    TimeTaggerController, TimeTaggerControllerConfig, TimeTaggerConfig,
    ChannelConfig as TTChannelConfig, TimeStamp
)


class BobMode(Enum):
    """Operation modes for Bob CPU."""
    PASSIVE = "passive"      # Random basis selection
    ACTIVE = "active"        # Coordinated basis selection
    CALIBRATION = "calibration"  # System calibration mode


@dataclass
class BobConfig:
    """Configuration for Bob CPU."""
    # General operation
    mode: BobMode = BobMode.PASSIVE
    measurement_duration_s: float = 60.0
    basis_selection_seed: Optional[int] = None
    
    # Channel configuration
    channel_config: ChannelConfig = field(default_factory=ChannelConfig)
    
    # Optical table configuration  
    optical_table_config: OpticalTableConfig = field(default_factory=OpticalTableConfig)
    
    # Detector configurations (4 detectors for BB84: H, V, D, A)
    detector_configs: Dict[str, DetectorConfig] = field(default_factory=lambda: {
        "H": DetectorConfig(detector_type=DetectorType.SPAD, quantum_efficiency=0.75),
        "V": DetectorConfig(detector_type=DetectorType.SPAD, quantum_efficiency=0.75), 
        "D": DetectorConfig(detector_type=DetectorType.SPAD, quantum_efficiency=0.75),
        "A": DetectorConfig(detector_type=DetectorType.SPAD, quantum_efficiency=0.75)
    })
    
    # Time tagger configuration
    timetagger_config: TimeTaggerControllerConfig = field(default_factory=lambda: 
        TimeTaggerControllerConfig(
            use_hardware=False,  # Default to simulator
            timetagger_config=TimeTaggerConfig(
                resolution_ps=1000,  # 1 ps resolution
                buffer_size=100000,
                measurement_duration_s= 1,
                channels={
                    0: TTChannelConfig(channel_id=0, enabled=True),  # H detector
                    1: TTChannelConfig(channel_id=1, enabled=True),  # V detector  
                    2: TTChannelConfig(channel_id=2, enabled=True),  # D detector
                    3: TTChannelConfig(channel_id=3, enabled=True),  # A detector
                }
            )
        )
    )
    
    # Basis selection probabilities
    basis_z_probability: float = 0.5  # Probability of selecting Z basis
    
    # Gating and timing
    enable_gated_detection: bool = True
    gate_width_ns: float = 10.0
    gate_delay_ns: float = 0.0


@dataclass
class BobStatistics:
    """Statistics for Bob's operation."""
    total_measurements: int = 0
    successful_detections: int = 0
    basis_selections: Dict[str, int] = field(default_factory=lambda: {"Z": 0, "X": 0})
    detector_counts: Dict[str, int] = field(default_factory=lambda: {"H": 0, "V": 0, "D": 0, "A": 0})
    coincidences: int = 0
    measurement_times: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Time tagger statistics
    timetag_events: int = 0
    average_count_rate_hz: float = 0.0
    coincidence_rate_hz: float = 0.0
    timing_precision_ps: float = 0.0


@dataclass
class BobData:
    """Bob's measurement data for post-processing."""
    measurement_times: List[float] = field(default_factory=list)
    measurement_bases: List[Basis] = field(default_factory=list)
    measurement_outcomes: List[MeasurementOutcome] = field(default_factory=list)
    measured_bits: List[Optional[Bit]] = field(default_factory=list)
    detection_probabilities: List[float] = field(default_factory=list)
    pulse_ids: List[int] = field(default_factory=list)
    
    # Time tagger data
    timestamps: List[TimeStamp] = field(default_factory=list)
    coincidence_events: List[Tuple[TimeStamp, TimeStamp]] = field(default_factory=list)
    channel_mappings: Dict[int, str] = field(default_factory=lambda: {0: "H", 1: "V", 2: "D", 3: "A"})


class BobCPU:
    """
    Bob's main controller for QKD reception.
    
    Manages the complete receiver system including:
    - Quantum channel simulation
    - Optical table for polarization measurement 
    - Four-detector setup for BB84 protocol
    - Random basis selection
    - Data collection and statistics
    """

    def __init__(self, config: BobConfig):
        """
        Initialize Bob's CPU.

        Args:
            config: Bob's configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # System state
        self.stats = BobStatistics()
        self.bob_data = BobData()
        self._running = False
        self._paused = False
        self._shutdown_event = threading.Event()
        self._pause_event = threading.Event()
        
        # Pulse processing
        self._pulse_queue = Queue(maxsize=10000)
        self._measurement_thread: Optional[threading.Thread] = None
        
        # Random number generator for basis selection
        if config.basis_selection_seed is not None:
            self._rng = np.random.default_rng(config.basis_selection_seed)
        else:
            self._rng = np.random.default_rng()
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info(f"Bob CPU initialized in {config.mode.value} mode")

    def _initialize_components(self) -> None:
        """Initialize all Bob-side components."""
        
        # Initialize quantum channel
        self.quantum_channel = FreeSpaceChannelSimulator(self.config.channel_config)
        
        # Initialize optical table
        self.optical_table = OpticalTableSimulator(self.config.optical_table_config)
        
        # Initialize detectors
        self.detectors = {}
        for detector_name, detector_config in self.config.detector_configs.items():
            self.detectors[detector_name] = PhotonDetectorSimulator(
                detector_config, 
                detector_id=detector_name
            )
        
        # Initialize time tagger
        self.timetagger = TimeTaggerController(self.config.timetagger_config)
        
        self.logger.info("All Bob components initialized")
        self.logger.info(f"Channel: {self.config.channel_config.distance_km} km")
        self.logger.info(f"Detectors: {list(self.detectors.keys())}")
        self.logger.info(f"TimeTagger: {'Hardware' if self.timetagger.is_using_hardware else 'Simulator'}")
        self.logger.info(f"TimeTagger Channels: {list(self.config.timetagger_config.timetagger_config.channels.keys())}")

    def start_measurement(self) -> bool:
        """
        Start QKD measurement process.
        
        Returns:
            bool: True if measurement started successfully
        """
        if self._running:
            self.logger.warning("Measurement already running")
            return False
        
        # Start time tagger measurement first
        if not self.timetagger.start_measurement():
            self.logger.error("Failed to start time tagger measurement")
            return False
        
        self._running = True
        self._shutdown_event.clear()
        self._pause_event.clear()
        
        # Start measurement thread
        self._measurement_thread = threading.Thread(
            target=self._measurement_loop,
            name="BobMeasurement"
        )
        self._measurement_thread.start()
        
        self.logger.info("QKD measurement started (including time tagger)")
        return True

    def stop_measurement(self) -> None:
        """Stop QKD measurement process."""
        if not self._running:
            self.logger.warning("Measurement not running")
            return
        
        self.logger.info("Stopping QKD measurement...")
        self._running = False
        self._shutdown_event.set()
        
        # Wait for measurement thread to finish
        if self._measurement_thread and self._measurement_thread.is_alive():
            self._measurement_thread.join(timeout=5.0)
        
        # Stop time tagger measurement
        self.timetagger.stop_measurement()
        
        self.logger.info("QKD measurement stopped (including time tagger)")

    def receive_pulse(self, pulse: Pulse) -> None:
        """
        Receive a pulse from Alice through the quantum channel.
        
        Args:
            pulse: Pulse from Alice
        """
        try:
            self._pulse_queue.put_nowait(pulse)
        except:
            self.logger.warning("Pulse queue full, dropping pulse")

    def _measurement_loop(self) -> None:
        """Main measurement loop running in separate thread."""
        start_time = time.time()
        measurement_count = 0
        
        try:
            while self._running and (time.time() - start_time) < self.config.measurement_duration_s:
                # Check for pause
                if self._paused:
                    self._pause_event.wait()
                    continue
                
                # Check for shutdown
                if self._shutdown_event.is_set():
                    break
                
                try:
                    # Get pulse from queue (with timeout)
                    pulse = self._pulse_queue.get(timeout=0.1)
                    
                    # Process the pulse
                    self._process_pulse(pulse, measurement_count)
                    measurement_count += 1
                    
                except Empty:
                    # No pulse received, continue
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing pulse {measurement_count}: {e}")
                    self.stats.errors.append(f"Pulse {measurement_count}: {e}")
                    continue
            
            total_time = time.time() - start_time
            self.logger.info(f"Measurement completed: {measurement_count} pulses in {total_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Critical error in measurement loop: {e}")
            self.stats.errors.append(f"Critical measurement error: {e}")
        finally:
            self._running = False

    def _process_pulse(self, pulse: Pulse, pulse_id: int) -> None:
        """Process a single pulse through the complete measurement system."""
        measurement_time = time.time()
        
        # 1. Transmit through quantum channel
        received_pulse = self.quantum_channel.transmit_pulse(pulse)
        
        # 2. Select random measurement basis
        measurement_basis = self._select_measurement_basis()
        
        # 3. Set optical table basis
        basis_set_success = self.optical_table.set_measurement_basis(measurement_basis)
        if not basis_set_success:
            self.logger.warning(f"Failed to set basis {measurement_basis.value} for pulse {pulse_id}")
            return
        
        # 4. Perform polarization measurement
        outcome, detection_prob = self.optical_table.measure_polarization(
            received_pulse, measurement_basis
        )
        
        # 5. Convert outcome to bit value
        measured_bit = self.optical_table.outcome_to_bit(outcome, measurement_basis)
        
        # 6. Simulate detector response and generate timing events
        detection_occurred = self._simulate_detector_response(outcome, received_pulse)
        
        # 7. Generate time tag events based on detection
        if detection_occurred:
            self._generate_timetag_events(outcome, measurement_time)
        
        # 8. Record measurement data
        self._record_measurement_data(
            measurement_time, measurement_basis, outcome, measured_bit,
            detection_prob, pulse_id, detection_occurred
        )
        
        # 9. Update statistics
        self._update_statistics(measurement_basis, outcome, detection_occurred)

    def _select_measurement_basis(self) -> Basis:
        """Randomly select measurement basis."""
        if self._rng.random() < self.config.basis_z_probability:
            return Basis.Z
        else:
            return Basis.X

    def _simulate_detector_response(self, outcome: MeasurementOutcome, pulse: Pulse) -> bool:
        """Simulate response of appropriate detector based on measurement outcome."""
        if outcome == MeasurementOutcome.NO_DETECTION:
            return False
        
        # Map outcome to detector
        detector_map = {
            MeasurementOutcome.H_DETECTION: "H",
            MeasurementOutcome.V_DETECTION: "V", 
            MeasurementOutcome.D_DETECTION: "D",
            MeasurementOutcome.A_DETECTION: "A"
        }
        
        detector_name = detector_map.get(outcome)
        if detector_name and detector_name in self.detectors:
            detector = self.detectors[detector_name]
            
            # Open gate if gated operation
            if self.config.enable_gated_detection:
                detector.open_gate()
            
            # Simulate detection
            detected, detection_time = detector.detect_photon(pulse)
            
            # Close gate
            if self.config.enable_gated_detection:
                detector.close_gate()
            
            return detected
        
        return False

    def _generate_timetag_events(self, outcome: MeasurementOutcome, measurement_time: float) -> None:
        """Generate time tag events for the detected outcome."""
        # Map measurement outcome to time tagger channel
        channel_map = {
            MeasurementOutcome.H_DETECTION: 0,  # Channel 0 for H detector
            MeasurementOutcome.V_DETECTION: 1,  # Channel 1 for V detector
            MeasurementOutcome.D_DETECTION: 2,  # Channel 2 for D detector  
            MeasurementOutcome.A_DETECTION: 3,  # Channel 3 for A detector
        }
        
        channel_id = channel_map.get(outcome)
        if channel_id is not None:
            # For simulator, inject a test event at the appropriate time
            if hasattr(self.timetagger.driver, 'inject_test_event'):
                # Convert measurement time to picoseconds and inject
                time_ps = int(measurement_time * 1e12)
                success = self.timetagger.driver.inject_test_event(channel_id, time_ps)
                if success:
                    self.logger.debug(f"Injected time tag event on channel {channel_id} at {time_ps} ps")
                else:
                    self.logger.warning(f"Failed to inject time tag event on channel {channel_id}")

    def _record_measurement_data(self, measurement_time: float, basis: Basis, 
                               outcome: MeasurementOutcome, bit: Optional[Bit],
                               detection_prob: float, pulse_id: int, 
                               detection_occurred: bool) -> None:
        """Record measurement data for analysis."""
        self.bob_data.measurement_times.append(measurement_time)
        self.bob_data.measurement_bases.append(basis)
        self.bob_data.measurement_outcomes.append(outcome)
        self.bob_data.measured_bits.append(bit)
        self.bob_data.detection_probabilities.append(detection_prob)
        self.bob_data.pulse_ids.append(pulse_id)

    def _update_statistics(self, basis: Basis, outcome: MeasurementOutcome, 
                         detection_occurred: bool) -> None:
        """Update measurement statistics."""
        self.stats.total_measurements += 1
        self.stats.basis_selections[basis.value] += 1
        
        if detection_occurred:
            self.stats.successful_detections += 1
        
        if outcome != MeasurementOutcome.NO_DETECTION:
            outcome_key = outcome.value.split('_')[0]  # Extract H, V, D, A
            if outcome_key in self.stats.detector_counts:
                self.stats.detector_counts[outcome_key] += 1

    def pause_measurement(self) -> None:
        """Pause measurement process."""
        if not self._running:
            self.logger.warning("Measurement not running")
            return
        
        self._paused = True
        self._pause_event.set()
        self.logger.info("QKD measurement paused")

    def resume_measurement(self) -> None:
        """Resume measurement process."""
        if not self._paused:
            self.logger.warning("Measurement not paused")
            return
        
        self._paused = False
        self._pause_event.clear()
        self.logger.info("QKD measurement resumed")

    def get_statistics(self) -> BobStatistics:
        """Get current measurement statistics."""
        return self.stats

    def get_measurement_data(self) -> BobData:
        """Get collected measurement data."""
        return self.bob_data

    def is_running(self) -> bool:
        """Check if measurement is currently running."""
        return self._running

    def is_paused(self) -> bool:
        """Check if measurement is currently paused."""
        return self._paused

    def get_detection_efficiency(self) -> float:
        """Calculate overall detection efficiency."""
        if self.stats.total_measurements == 0:
            return 0.0
        return self.stats.successful_detections / self.stats.total_measurements

    def get_component_info(self) -> Dict[str, Any]:
        """Get information about all components."""
        detector_info = {}
        for name, detector in self.detectors.items():
            detector_info[name] = detector.get_detector_info()
        
        return {
            "quantum_channel": self.quantum_channel.get_channel_info(),
            "optical_table": self.optical_table.get_table_info(),
            "detectors": detector_info,
            "configuration": {
                "mode": self.config.mode.value,
                "basis_z_probability": self.config.basis_z_probability,
                "measurement_duration_s": self.config.measurement_duration_s
            }
        }

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.stats = BobStatistics()
        self.bob_data = BobData()
        
        # Reset component statistics
        self.quantum_channel.reset_statistics()
        self.optical_table.reset_statistics()
        for detector in self.detectors.values():
            detector.reset_statistics()
        
        self.logger.info("All statistics reset")

    def calibrate_system(self) -> Dict[str, Any]:
        """Perform complete system calibration."""
        self.logger.info("Starting Bob system calibration...")
        
        calibration_results = {}
        
        try:
            # Calibrate quantum channel
            calibration_results["channel"] = self.quantum_channel.calibrate_channel()
            
            # Calibrate optical table
            calibration_results["optical_table"] = self.optical_table.calibrate_table()
            
            # Calibrate each detector
            detector_calibrations = {}
            for name, detector in self.detectors.items():
                detector_calibrations[name] = detector.calibrate_detector()
            calibration_results["detectors"] = detector_calibrations
            
            self.logger.info("Bob system calibration completed successfully")
            
        except Exception as e:
            self.logger.error(f"Calibration failed: {e}")
            calibration_results["error"] = str(e)
        
        return calibration_results

    def set_channel_conditions(self, weather: str = "clear", 
                             enable_eavesdropper: bool = False,
                             eavesdropper_strength: float = 0.1) -> None:
        """Set quantum channel conditions for testing."""
        self.quantum_channel.set_weather_condition(weather)
        
        if enable_eavesdropper:
            self.quantum_channel.enable_eavesdropper(eavesdropper_strength)
        else:
            self.quantum_channel.disable_eavesdropper()
        
        self.logger.info(f"Channel conditions set: weather={weather}, eavesdropper={enable_eavesdropper}")

    def get_timetag_data(self, max_events: Optional[int] = None) -> List[TimeStamp]:
        """
        Retrieve time tag data from the time tagger.
        
        Args:
            max_events: Maximum number of events to retrieve
            
        Returns:
            List of TimeStamp events
        """
        timestamps = self.timetagger.get_timestamps(max_events)
        
        # Store in Bob data for analysis
        self.bob_data.timestamps.extend(timestamps)
        
        # Update statistics
        self.stats.timetag_events += len(timestamps)
        
        return timestamps

    def analyze_coincidences(self, time_window_ps: float = 1000.0) -> List[Tuple[TimeStamp, TimeStamp]]:
        """
        Analyze time tag data for coincidence events.
        
        Args:
            time_window_ps: Coincidence time window in picoseconds
            
        Returns:
            List of coincidence pairs
        """
        coincidences = []
        timestamps = self.bob_data.timestamps
        
        for i, ts1 in enumerate(timestamps):
            for j, ts2 in enumerate(timestamps[i+1:], i+1):
                time_diff = abs(ts2.time_ps - ts1.time_ps)
                if time_diff <= time_window_ps and ts1.channel != ts2.channel:
                    coincidences.append((ts1, ts2))
        
        # Store coincidences
        self.bob_data.coincidence_events.extend(coincidences)
        self.stats.coincidences += len(coincidences)
        
        # Update coincidence rate
        if self.stats.measurement_times:
            measurement_duration = max(self.stats.measurement_times) - min(self.stats.measurement_times)
            if measurement_duration > 0:
                self.stats.coincidence_rate_hz = len(coincidences) / measurement_duration
        
        return coincidences

    def get_count_rates(self) -> Dict[str, float]:
        """Get count rates for all detector channels."""
        # Get timetagger count rates
        tt_rates = self.timetagger.get_count_rates()
        
        # Map to detector names
        detector_rates = {}
        for channel_id, rate in tt_rates.items():
            detector_name = self.bob_data.channel_mappings.get(channel_id, f"Ch{channel_id}")
            detector_rates[detector_name] = rate
        
        return detector_rates

    def get_timetagger_status(self) -> Dict[str, Any]:
        """Get comprehensive time tagger status and statistics."""
        return {
            "device_info": self.timetagger.get_device_info(),
            "is_measuring": self.timetagger.is_measuring(),
            "using_hardware": self.timetagger.is_using_hardware,
            "statistics": {
                "total_events": self.stats.timetag_events,
                "coincidences": self.stats.coincidences,
                "coincidence_rate_hz": self.stats.coincidence_rate_hz,
                "count_rates": self.get_count_rates()
            }
        }

    def reset_timetagger(self) -> bool:
        """Reset the time tagger system."""
        success = self.timetagger.reset()
        if success:
            # Clear related data
            self.bob_data.timestamps.clear()
            self.bob_data.coincidence_events.clear()
            self.stats.timetag_events = 0
            self.stats.coincidences = 0
            self.stats.coincidence_rate_hz = 0.0
            
            self.logger.info("Time tagger reset successfully")
        return success

    def __del__(self):
        """Cleanup when object is destroyed."""
        if self._running:
            self.stop_measurement()
        
        # Cleanup timetagger
        if hasattr(self, 'timetagger'):
            self.timetagger.cleanup()
