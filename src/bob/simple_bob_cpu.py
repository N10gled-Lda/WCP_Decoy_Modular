"""
Simple Bob CPU - Streamlined QKD Receiver.

Uses simplified components for straightforward QKD reception:
- Simple quantum channel (pass-through or attenuation)
- Simple optical table (perfect or with angular deviations)  
- Simple detectors (photon count to detector number mapping)
"""

import logging
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from queue import Queue, Empty
from enum import Enum

from src.utils.data_structures import Pulse, Basis, Bit
from src.quantum_channel.simple_channel import SimpleQuantumChannel, SimpleChannelConfig
from src.bob.ot_components.simple_optical_table import SimpleOpticalTable, SimpleOpticalConfig, SimpleDetectorId
from src.bob.ot_components.simple_detectors import SimpleDetectorSystem, SimpleDetectorConfig


class SimpleBobMode(Enum):
    """Operation modes for Simple Bob CPU."""
    PASSIVE = "passive"      # Random basis selection
    PERFECT = "perfect"      # Perfect measurements (no errors)
    REALISTIC = "realistic"  # Realistic measurements with errors


@dataclass
class SimpleBobConfig:
    """Configuration for Simple Bob CPU."""
    # General operation
    mode: SimpleBobMode = SimpleBobMode.PASSIVE
    measurement_duration_s: float = 60.0
    basis_selection_seed: Optional[int] = None
    
    # Quantum channel configuration
    channel_config: SimpleChannelConfig = field(default_factory=SimpleChannelConfig)
    
    # Optical table configuration
    optical_config: SimpleOpticalConfig = field(default_factory=SimpleOpticalConfig)
    
    # Detector configuration
    detector_config: SimpleDetectorConfig = field(default_factory=SimpleDetectorConfig)
    
    # Basis selection probability
    basis_z_probability: float = 0.5  # Probability of selecting Z basis


@dataclass
class SimpleBobStatistics:
    """Statistics for Simple Bob's operation."""
    total_pulses_received: int = 0
    total_detections: int = 0
    basis_selections: Dict[str, int] = field(default_factory=lambda: {"Z": 0, "X": 0})
    detector_events: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 0})
    successful_measurements: int = 0
    detection_efficiency: float = 0.0


@dataclass 
class SimpleBobData:
    """Simple Bob's measurement data."""
    received_pulses: List[Pulse] = field(default_factory=list)
    measurement_bases: List[Basis] = field(default_factory=list)
    detector_numbers: List[Optional[int]] = field(default_factory=list)
    measured_bits: List[Optional[Bit]] = field(default_factory=list)
    detection_probabilities: List[float] = field(default_factory=list)
    measurement_times: List[float] = field(default_factory=list)


class SimpleBobCPU:
    """
    Simplified Bob CPU for QKD reception.
    
    Processing pipeline:
    1. Receive pulse from Alice
    2. Transmit through simple quantum channel  
    3. Select random measurement basis
    4. Measure polarization with simple optical table
    5. Detect with simple detector system
    6. Convert detector number to bit value
    7. Record measurement data
    """
    
    def __init__(self, config: SimpleBobConfig):
        """
        Initialize Simple Bob CPU.
        
        Args:
            config: Bob's configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # System state
        self.stats = SimpleBobStatistics()
        self.data = SimpleBobData()
        self._running = False
        self._shutdown_event = threading.Event()
        
        # Pulse processing
        self._pulse_queue = Queue(maxsize=10000)
        self._measurement_thread: Optional[threading.Thread] = None
        
        # Random number generator
        if config.basis_selection_seed is not None:
            self._rng = np.random.default_rng(config.basis_selection_seed)
        else:
            self._rng = np.random.default_rng()
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info(f"Simple Bob CPU initialized in {config.mode.value} mode")

    def _initialize_components(self) -> None:
        """Initialize all Bob-side components."""
        # Configure components based on mode
        if self.config.mode == SimpleBobMode.PERFECT:
            self._configure_perfect_mode()
        elif self.config.mode == SimpleBobMode.REALISTIC:
            self._configure_realistic_mode()
        
        # Initialize quantum channel
        self.quantum_channel = SimpleQuantumChannel(self.config.channel_config)
        
        # Initialize optical table
        self.optical_table = SimpleOpticalTable(self.config.optical_config)
        
        # Initialize detector system
        self.detector_system = SimpleDetectorSystem(self.config.detector_config)
        
        self.logger.info("All Simple Bob components initialized")
        self.logger.info(f"Channel mode: {'pass-through' if self.config.channel_config.pass_through_mode else 'attenuating'}")
        self.logger.info(f"Optical mode: {'perfect' if self.config.optical_config.perfect_measurement else 'realistic'}")
        self.logger.info(f"Detector mode: {'perfect' if self.config.detector_config.perfect_detection else 'realistic'}")

    def _configure_perfect_mode(self) -> None:
        """Configure all components for perfect operation."""
        self.config.channel_config.pass_through_mode = True
        self.config.optical_config.perfect_measurement = True
        self.config.detector_config.perfect_detection = True

    def _configure_realistic_mode(self) -> None:
        """Configure all components for realistic operation."""
        self.config.channel_config.pass_through_mode = False
        self.config.channel_config.apply_attenuation = True
        self.config.channel_config.attenuation_db = 3.0  # 3 dB loss
        
        self.config.optical_config.perfect_measurement = False
        self.config.optical_config.apply_angular_deviation = True
        self.config.optical_config.angular_deviation_degrees = 2.0
        
        self.config.detector_config.perfect_detection = False
        self.config.detector_config.quantum_efficiency = 0.8

    def receive_pulse(self, pulse: Pulse) -> None:
        """
        Receive a pulse from Alice for processing.
        
        Args:
            pulse: Incoming pulse from Alice
        """
        try:
            self._pulse_queue.put_nowait(pulse)
            self.logger.debug(f"Received pulse {pulse.pulse_id}")
        except:
            self.logger.warning(f"Pulse queue full, dropping pulse {pulse.pulse_id}")

    def start_measurement(self) -> bool:
        """Start QKD measurement process."""
        if self._running:
            self.logger.warning("Measurement already running")
            return False
        
        self._running = True
        self._shutdown_event.clear()
        
        # Start measurement thread
        self._measurement_thread = threading.Thread(
            target=self._measurement_loop,
            name="SimpleBobMeasurement"
        )
        self._measurement_thread.start()
        
        self.logger.info("Simple Bob measurement started")
        return True

    def stop_measurement(self) -> None:
        """Stop QKD measurement process."""
        if not self._running:
            return
        
        self.logger.info("Stopping Simple Bob measurement...")
        self._running = False
        self._shutdown_event.set()
        
        # Wait for measurement thread
        if self._measurement_thread and self._measurement_thread.is_alive():
            self._measurement_thread.join(timeout=5.0)
        
        self.logger.info("Simple Bob measurement stopped")

    def _measurement_loop(self) -> None:
        """Main measurement processing loop."""
        self.logger.debug("Simple Bob measurement loop started")
        
        try:
            while self._running and not self._shutdown_event.is_set():
                try:
                    # Get pulse from queue (with timeout)
                    pulse = self._pulse_queue.get(timeout=0.1)
                    self._process_pulse(pulse)
                    
                except Empty:
                    continue  # No pulse available, continue loop
                except Exception as e:
                    self.logger.error(f"Error processing pulse: {e}")
                    
        except Exception as e:
            self.logger.error(f"Critical error in measurement loop: {e}")
        finally:
            self._running = False
            self.logger.debug("Simple Bob measurement loop ended")

    def _process_pulse(self, pulse: Pulse) -> None:
        """
        Process a single pulse through the measurement system.
        
        Args:
            pulse: Pulse to process
        """
        measurement_time = time.time()
        
        # 1. Transmit through quantum channel
        transmitted_pulse = self.quantum_channel.transmit_pulse(pulse)
        if transmitted_pulse is None:
            self.logger.debug(f"Pulse {pulse.pulse_id} lost in channel")
            return
        
        # 2. Select random measurement basis
        measurement_basis = self._select_measurement_basis()
        
        # 3. Set optical table basis
        self.optical_table.set_measurement_basis(measurement_basis)
        
        # 4. Measure polarization
        detector_id, detection_prob = self.optical_table.measure_polarization(transmitted_pulse)
        
        # 5. Detect with detector system
        detector_number = self.detector_system.detect_pulse(
            transmitted_pulse, measurement_basis, (detector_id, detection_prob)
        )
        
        # 6. Convert to bit value
        measured_bit = None
        if detector_number is not None:
            bits = self.detector_system.convert_detections_to_bits([detector_number], [measurement_basis])
            measured_bit = bits[0]
        
        # 7. Record measurement data
        self._record_measurement(
            transmitted_pulse, measurement_basis, detector_number, 
            measured_bit, detection_prob, measurement_time
        )
        
        # 8. Update statistics
        self._update_statistics(measurement_basis, detector_number, measured_bit)

    def _select_measurement_basis(self) -> Basis:
        """Randomly select measurement basis."""
        if self._rng.random() < self.config.basis_z_probability:
            return Basis.Z
        else:
            return Basis.X

    def _record_measurement(self, pulse: Pulse, basis: Basis, detector_number: Optional[int],
                           bit: Optional[Bit], detection_prob: float, measurement_time: float) -> None:
        """Record measurement data."""
        self.data.received_pulses.append(pulse)
        self.data.measurement_bases.append(basis)
        self.data.detector_numbers.append(detector_number)
        self.data.measured_bits.append(bit)
        self.data.detection_probabilities.append(detection_prob)
        self.data.measurement_times.append(measurement_time)

    def _update_statistics(self, basis: Basis, detector_number: Optional[int], bit: Optional[Bit]) -> None:
        """Update measurement statistics."""
        self.stats.total_pulses_received += 1
        self.stats.basis_selections[basis.value] += 1
        
        if detector_number is not None:
            self.stats.total_detections += 1
            self.stats.detector_events[detector_number] += 1
            
            if bit is not None:
                self.stats.successful_measurements += 1
        
        # Update detection efficiency
        if self.stats.total_pulses_received > 0:
            self.stats.detection_efficiency = self.stats.total_detections / self.stats.total_pulses_received

    def process_pulse_batch(self, pulses: List[Pulse]) -> List[Optional[int]]:
        """
        Process a batch of pulses and return detector numbers.
        
        Args:
            pulses: List of pulses to process
            
        Returns:
            List of detector numbers (None for no detection)
        """
        detector_numbers = []
        
        for pulse in pulses:
            # Transmit through channel
            transmitted_pulse = self.quantum_channel.transmit_pulse(pulse)
            if transmitted_pulse is None:
                detector_numbers.append(None)
                continue
            
            # Select basis and measure
            measurement_basis = self._select_measurement_basis()
            self.optical_table.set_measurement_basis(measurement_basis)
            detector_id, detection_prob = self.optical_table.measure_polarization(transmitted_pulse)
            
            # Detect
            detector_number = self.detector_system.detect_pulse(
                transmitted_pulse, measurement_basis, (detector_id, detection_prob)
            )
            detector_numbers.append(detector_number)
            
            # Update statistics
            self._update_statistics(measurement_basis, detector_number, None)
        
        return detector_numbers

    def set_channel_attenuation(self, attenuation_db: float) -> None:
        """Set quantum channel attenuation."""
        self.quantum_channel.set_attenuation(attenuation_db)
        self.logger.info(f"Channel attenuation set to {attenuation_db} dB")

    def set_pass_through_mode(self, enable: bool = True) -> None:
        """Enable or disable channel pass-through mode."""
        self.quantum_channel.set_pass_through_mode(enable)
        self.logger.info(f"Pass-through mode: {'enabled' if enable else 'disabled'}")

    def set_optical_deviation(self, deviation_degrees: float) -> None:
        """Set optical table angular deviation."""
        self.optical_table.set_angular_deviation(deviation_degrees)
        self.logger.info(f"Optical deviation set to {deviation_degrees}°")

    def set_detector_efficiency(self, efficiency: float) -> None:
        """Set detector quantum efficiency."""
        self.detector_system.set_quantum_efficiency(efficiency)
        self.logger.info(f"Detector efficiency set to {efficiency}")

    def set_perfect_mode(self, enable: bool = True) -> None:
        """Enable or disable perfect measurement mode."""
        self.optical_table.set_perfect_measurement(enable)
        self.detector_system.set_perfect_detection(enable)
        if enable:
            self.quantum_channel.set_pass_through_mode(True)
        self.logger.info(f"Perfect mode: {'enabled' if enable else 'disabled'}")

    def get_measurement_summary(self) -> Dict:
        """Get summary of measurements."""
        return {
            "statistics": {
                "total_pulses": self.stats.total_pulses_received,
                "total_detections": self.stats.total_detections,
                "detection_efficiency": self.stats.detection_efficiency,
                "basis_selections": self.stats.basis_selections,
                "detector_events": self.stats.detector_events,
                "successful_measurements": self.stats.successful_measurements
            },
            "data_points": len(self.data.received_pulses),
            "channel_efficiency": self.quantum_channel.get_transmission_efficiency(),
            "detector_stats": self.detector_system.get_detection_statistics()
        }

    def reset_measurements(self) -> None:
        """Reset all measurement data and statistics."""
        self.stats = SimpleBobStatistics()
        self.data = SimpleBobData()
        self.detector_system.reset_statistics()
        self.logger.info("Measurement data and statistics reset")

    def is_measuring(self) -> bool:
        """Check if measurement is currently active."""
        return self._running

    def get_detector_name(self, detector_number: int) -> str:
        """Get detector name from number."""
        return self.detector_system.detector_number_to_name(detector_number)

    def __del__(self):
        """Cleanup when object is destroyed."""
        if self._running:
            self.stop_measurement()
