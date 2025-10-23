"""
Bob's main controller for QKD reception.
Similar structure to Alice's CPU but focused on detection and reception.
"""
import logging
import time
import threading
import socket
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple, Union
from queue import Queue, Empty
import numpy as np
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.data_structures import Pulse, Basis, Bit
from src.bob.timetagger.timetagger_controller import (
    TimeTaggerController, TimeTaggerControllerConfig, 
    DetectionBasis, DetectionResult, QKDDetection
)
from src.bob.timetagger.timetagger_base import TimeTaggerConfig, ChannelConfig
from src.bob.timetagger.timetagger_hardware import TimeTaggerHardware
from src.bob.timetagger.timetagger_simulator import TimeTaggerSimulator

# Protocol imports for classical communication and post-processing
try:
    from src.protocol.qkd_bob_implementation_class import QKDBobImplementation
except ImportError:
    QKDBobImplementation = None


class BobMode(Enum):
    """Detection modes for Bob.
        - CONTINUOUS: Continuously detect without basis selection
        - SYNCHRONIZED: Detect based on basis information from Alice
        - RANDOM_BASIS: Randomly select basis for each detection
        - FIXED_BASIS: Use a fixed basis for all detections
    """
    CONTINUOUS = "continuous"
    SYNCHRONIZED = "synchronized"
    RANDOM_BASIS = "random_basis"
    FIXED_BASIS = "fixed_basis"


# Default values for Bob parameters
KEY_LENGTH = 5
DETECTION_TIMEOUT = 10.0  # seconds to wait for each detection
ERROR_THRESHOLD = 0.11  # Max tolerable QBER
USE_HARDWARE = False
DETECTION_EFFICIENCY = 0.8
DARK_COUNT_RATE = 100.0  # Hz
USE_MOCK_TRANSMITTER = True
BOBMODE = BobMode.CONTINUOUS

# Network configuration defaults
IP_ADDRESS_ALICE = "localhost"
IP_ADDRESS_BOB = "localhost" 
PORT_NUMBER_ALICE = 65432
PORT_NUMBER_BOB = 65433
PORT_NUMBER_QUANTUM_CHANNEL = 12345
SHARED_SECRET_KEY = "IzetXlgAnY4oye56"  # 16 bytes for AES-128


@dataclass
class BobConfig:
    """Configuration for Bob CPU."""
    # Detection parameters
    num_expected_pulses: int = 10
    detection_timeout_s: float = 10.0
    test_fraction: float = 0.1
    error_threshold: float = 0.11  # Max tolerable QBER
    
    # Hardware parameters
    use_hardware: bool = False
    detection_efficiency: float = 0.8
    dark_count_rate_hz: float = 100.0
    detection_jitter_ps: int = 1000
    mode: BobMode = BobMode.CONTINUOUS
    
    # Basis selection (only used if mode=FIXED_BASIS)
    fixed_basis: DetectionBasis = DetectionBasis.Z
    # Predetermined basis sequence (only used if mode=SYNCHRONIZED)
    predetermined_bases: Optional[List[str]] = None
    
    # Network configuration for quantum channel
    use_mock_transmitter: bool = False  # For testing without actual Alice
    client_qch_host: str = "localhost"
    client_qch_port: int = 12345
    
    # Classical communication configuration
    alice_ip: str = "localhost"
    alice_port: int = 65432
    bob_ip: str = "localhost"
    bob_port: int = 65433
    shared_secret_key: str = "IzetXlgAnY4oye56"  # 16 bytes for AES-128
    
    # Multi-threading parameters
    num_threads: int = 1
    
    # Post-processing parameters
    enable_post_processing: bool = True
    pa_compression_rate: float = 0.5


@dataclass
class BobResults:
    """Comprehensive results from Bob's QKD operation."""
    # Detection data
    detected_bits: List[Optional[str]] = field(default_factory=list)
    measurement_bases: List[str] = field(default_factory=list)
    detection_results: List[DetectionResult] = field(default_factory=list)
    detection_confidences: List[float] = field(default_factory=list)
    pulse_ids: List[int] = field(default_factory=list)
    
    # Timing data
    detection_timestamps: List[float] = field(default_factory=list)
    processing_times: List[float] = field(default_factory=list)
    
    # Statistics
    detections_attempted: int = 0
    successful_detections: int = 0
    no_detections: int = 0
    ambiguous_detections: int = 0
    total_runtime_seconds: float = 0.0
    average_detection_rate_hz: float = 0.0
    
    # Error tracking
    errors: List[str] = field(default_factory=list)


class BobCPU:
    """
    The main controller for Bob's side of the QKD protocol.
    
    Controls detection system with TimeTagger hardware/simulator for 
    BB84 quantum key distribution reception. Supports various detection
    modes including synchronized detection with Alice.
    """

    def __init__(self, config: BobConfig):
        """
        Initialize Bob's CPU.

        Args:
            config: Configuration parameters for Bob's operation
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config = config

        # Configuration parameters
        # Detection parameters
        self.num_expected_pulses = self.config.num_expected_pulses
        self.detection_timeout_s = self.config.detection_timeout_s
        self.test_fraction = self.config.test_fraction
        self.error_threshold = self.config.error_threshold

        # Hardware parameters
        self.use_hardware = self.config.use_hardware
        self.detection_efficiency = self.config.detection_efficiency
        self.dark_count_rate_hz = self.config.dark_count_rate_hz
        self.detection_jitter_ps = self.config.detection_jitter_ps
        self.mode = self.config.mode

        # Basis selection parameters
        self.fixed_basis = self.config.fixed_basis
        self.predetermined_bases = self.config.predetermined_bases

        # Network configuration for quantum channel
        self.use_mock_transmitter = self.config.use_mock_transmitter
        self.mock_thread = None
        self.mock_thread_stop_event = None
        self.client_qch_host = self.config.client_qch_host
        self.client_qch_port = self.config.client_qch_port

        # Classical communication configuration
        self.alice_ip = self.config.alice_ip
        self.alice_port = self.config.alice_port
        self.bob_ip = self.config.bob_ip
        self.bob_port = self.config.bob_port
        self.shared_secret_key = bytes(self.config.shared_secret_key, 'utf-8')

        # Post-processing configuration
        self.enable_post_processing = self.config.enable_post_processing
        self.pa_compression_rate = self.config.pa_compression_rate
        self.num_threads = self.config.num_threads

        # Classical communication objects (initialized later)
        self.classical_channel_participant_for_pp: Optional[Any] = None
        self.quantum_client: Optional[socket.socket] = None

        # Detection system components  
        self.timetagger_controller: Optional[TimeTaggerController] = None

        # Operation state
        self.results = BobResults()
        self._detection_thread: Optional[threading.Thread] = None
        self._stop_detection = threading.Event()

        self.logger.info("Bob CPU initialized")

    def _initialize_components(self) -> None:
        """Initialize the detection components following Alice's pattern."""
        # Create TimeTagger driver based on hardware flag (like Alice chooses laser driver)
        if self.use_hardware:
            # Create base config for hardware
            base_config = TimeTaggerConfig(
                resolution_ps=1000,
                buffer_size=100000,
                channels={i: ChannelConfig(
                    channel_id=i,
                    enabled=True
                ) for i in [1, 2, 3, 4]}
            )
            driver = TimeTaggerHardware(base_config)
            self.logger.info("Using hardware TimeTagger driver")
        else:
            # Create base config for simulator
            base_config = TimeTaggerConfig(
                resolution_ps=1000,
                buffer_size=100000,
                channels={i: ChannelConfig(
                    channel_id=i,
                    enabled=True
                ) for i in [1, 2, 3, 4]}
            )
            driver = TimeTaggerSimulator(base_config)
            self.logger.info("Using simulated TimeTagger driver")
        
        # Create controller config for QKD detection
        controller_config = TimeTaggerControllerConfig(
            use_hardware=self.use_hardware,
            quantum_efficiency=self.detection_efficiency,
            dark_count_rate_hz=self.dark_count_rate_hz,
            detection_jitter_ps=self.detection_jitter_ps,
            min_detection_threshold=1,
            max_dark_count_threshold=10
        )
        
        # Create controller with driver (like Alice's LaserController)
        self.timetagger_controller = TimeTaggerController(driver, controller_config)

        self.logger.info("Bob components initialized")

    def initialize_system(self) -> bool:
        """Initialize all Bob subsystems."""
        try:
            self.logger.info("Initializing Bob system...")
            
            # Initialize hardware components
            self._initialize_components()
            
            # Initialize TimeTagger controller (like Alice initializes laser controller)
            if not self.timetagger_controller.initialize():
                self.logger.error("Failed to initialize TimeTagger controller")
                return False
            
            # Validate configuration
            if not self._validate_configuration():
                return False
            
            self.logger.info("Bob system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Bob system: {e}")
            return False

    def _validate_configuration(self) -> bool:
        """Validate Bob's configuration."""
        try:
            # Validate predetermined bases if needed
            if self.mode == BobMode.SYNCHRONIZED:
                if not self._validate_predetermined_bases():
                    return False
            
            # Validate detection parameters
            if self.detection_timeout_s <= 0:
                self.logger.error("Detection timeout must be positive")
                return False
            
            if self.num_expected_pulses <= 0:
                self.logger.error("Number of expected pulses must be positive")
                return False
            
            self.logger.info("Bob configuration validated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def _validate_predetermined_bases(self) -> bool:
        """Validate predetermined basis sequences."""
        if self.predetermined_bases is None:
            self.logger.error("Predetermined bases required for synchronized mode")
            return False
        
        if len(self.predetermined_bases) != self.num_expected_pulses:
            self.logger.error(f"Predetermined bases length ({len(self.predetermined_bases)}) "
                            f"must match expected pulses ({self.num_expected_pulses})")
            return False
        
        # Validate basis values
        valid_bases = {'Z', 'X'}
        for i, basis in enumerate(self.predetermined_bases):
            if basis not in valid_bases:
                self.logger.error(f"Invalid basis '{basis}' at position {i}. Must be 'Z' or 'X'")
                return False
        
        return True

    @staticmethod
    def setup_quantum_channel_client(host: str, port: int, timeout: float = 10.0) -> socket.socket:
        """Setup connection to Alice's quantum channel server."""
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(timeout)
        client.connect((host, port))
        return client

    def setup_classical_communication(self) -> bool:
        """Setup classical communication with Alice for post-processing."""
        try:
            if QKDBobImplementation is None:
                self.logger.warning("QKDBobImplementation not available, skipping classical communication")
                return True
                
            self.classical_channel_participant_for_pp = QKDBobImplementation(
                alice_ip=self.alice_ip,
                alice_port=self.alice_port,
                bob_ip=self.bob_ip,
                bob_port=self.bob_port,
                shared_secret_key=self.shared_secret_key
            )
            self.logger.info("Classical communication setup successful")
            return True
        except Exception as e:
            self.logger.error(f"Failed to setup classical communication: {e}")
            return False

    def setup_mock_transmitter(self) -> None:
        """Setup mock transmitter for testing without Alice."""
        if self.mock_thread is not None:
            self.logger.warning("Mock transmitter already running")
            return
        
        self.mock_thread_stop_event = threading.Event()
        self.mock_thread = threading.Thread(
            target=self._mock_transmitter_client,
            args=(self.mock_thread_stop_event,),
            daemon=True
        )
        self.mock_thread.start()
        self.logger.info("Mock transmitter started")

    def _mock_transmitter_client(self, stop_event: threading.Event) -> None:
        """Mock transmitter that sends simulated pulses."""
        try:
            time.sleep(2)  # Wait for Bob to be ready
            
            pulse_id = 0
            while not stop_event.is_set() and pulse_id < self.num_expected_pulses:
                # Generate random pulse
                import random
                basis = random.choice([DetectionBasis.Z, DetectionBasis.X])
                bit_value = random.choice(["0", "1"])
                
                # Calculate arrival time
                arrival_time_ps = int((time.time() + 0.1) * 1e12)  # 100ms in future
                
                # Add simulated pulse to controller
                polarization_degrees = 0.0 if bit_value == "0" else 90.0  # Simple mapping
                if basis == DetectionBasis.X:
                    polarization_degrees += 45.0  # Shift for X basis
                
                self.timetagger_controller.add_simulated_pulse(
                    arrival_time_ps, polarization_degrees, photon_count=1
                )
                
                pulse_id += 1
                time.sleep(1.0)  # 1 second between pulses
                
        except Exception as e:
            self.logger.error(f"Mock transmitter error: {e}")

    def stop_mock_transmitter(self) -> None:
        """Stop the mock transmitter."""
        if self.mock_thread_stop_event:
            self.mock_thread_stop_event.set()
        
        if self.mock_thread and self.mock_thread.is_alive():
            self.mock_thread.join(timeout=5.0)
            if self.mock_thread.is_alive():
                self.logger.warning("Mock transmitter thread did not stop gracefully")
        
        self.mock_thread = None
        self.mock_thread_stop_event = None
        self.logger.info("Mock transmitter stopped")

    def run_complete_qkd_protocol(self) -> bool:
        """Run the complete QKD protocol from Bob's perspective."""
        try:
            self.logger.info("Starting complete QKD protocol...")
            
            # Setup classical communication if enabled
            if self.enable_post_processing:
                if not self.setup_classical_communication():
                    return False
            
            # Setup mock transmitter if needed
            if self.use_mock_transmitter:
                self.setup_mock_transmitter()
            
            # Connect to quantum channel or wait for simulated pulses
            if not self.use_mock_transmitter:
                # Connect to Alice's quantum channel
                self.quantum_client = self.setup_quantum_channel_client(
                    self.client_qch_host, self.client_qch_port
                )
                self.logger.info("Connected to quantum channel")
            
            # Run detection process
            success = self.run_detection_process()
            
            # Run post-processing if enabled
            if success and self.enable_post_processing:
                success = self._run_post_processing()
            
            # Cleanup
            if self.use_mock_transmitter:
                self.stop_mock_transmitter()
            
            if self.quantum_client:
                self.quantum_client.close()
            
            return success
            
        except Exception as e:
            self.logger.error(f"QKD protocol failed: {e}")
            return False

    def run_detection_process(self) -> bool:
        """Run the quantum detection process."""
        try:
            self.logger.info(f"Starting detection process for {self.num_expected_pulses} pulses...")
            
            start_time = time.time()
            successful_detections = 0
            
            for pulse_id in range(self.num_expected_pulses):
                # Select measurement basis
                basis = self._select_measurement_basis(pulse_id)
                
                # Perform QKD detection using controller (like Alice calls laser.trigger_once())
                detection = self.timetagger_controller.perform_qkd_detection(basis)
                
                # Process and store results
                self._process_detection_result(detection, pulse_id)
                
                if detection.detection_result not in [DetectionResult.NO_DETECTION]:
                    successful_detections += 1
                
                self.logger.debug(f"Pulse {pulse_id}: basis={basis.value}, "
                                f"result={detection.detection_result.value}, "
                                f"bit={detection.bit_value}")
            
            # Calculate statistics
            end_time = time.time()
            self.results.total_runtime_seconds = end_time - start_time
            self.results.detections_attempted = self.num_expected_pulses
            self.results.successful_detections = successful_detections
            
            if self.results.total_runtime_seconds > 0:
                self.results.average_detection_rate_hz = successful_detections / self.results.total_runtime_seconds
            
            self.logger.info(f"Detection process completed. "
                           f"Successful detections: {successful_detections}/{self.num_expected_pulses}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Detection process failed: {e}")
            return False

    def _select_measurement_basis(self, pulse_id: int) -> DetectionBasis:
        """Select measurement basis based on mode and pulse ID."""
        if self.mode == BobMode.FIXED_BASIS:
            return self.fixed_basis
        elif self.mode == BobMode.SYNCHRONIZED:
            if self.predetermined_bases and pulse_id < len(self.predetermined_bases):
                basis_str = self.predetermined_bases[pulse_id]
                return DetectionBasis.Z if basis_str == 'Z' else DetectionBasis.X
            else:
                self.logger.warning(f"No predetermined basis for pulse {pulse_id}, using Z")
                return DetectionBasis.Z
        elif self.mode == BobMode.RANDOM_BASIS:
            import random
            return random.choice([DetectionBasis.Z, DetectionBasis.X])
        else:  # CONTINUOUS mode
            return DetectionBasis.Z  # Default to Z basis

    def _process_detection_result(self, detection: QKDDetection, pulse_id: int) -> None:
        """Process and store detection result."""
        self.results.detected_bits.append(detection.bit_value)
        self.results.measurement_bases.append(detection.measurement_basis.value)
        self.results.detection_results.append(detection.detection_result)
        self.results.detection_confidences.append(detection.detection_confidence)
        self.results.pulse_ids.append(pulse_id)
        self.results.detection_timestamps.append(detection.timestamp_ps / 1e12)  # Convert to seconds
        
        # Update counters
        if detection.detection_result == DetectionResult.NO_DETECTION:
            self.results.no_detections += 1
        elif detection.detection_result == DetectionResult.AMBIGUOUS:
            self.results.ambiguous_detections += 1

    def _run_post_processing(self) -> bool:
        """Run classical post-processing with Alice."""
        try:
            if not self.classical_channel_participant_for_pp:
                self.logger.error("Classical communication not initialized")
                return False
            
            self.logger.info("Starting post-processing...")
            
            # Convert results to format expected by post-processing
            bob_bits = [bit for bit in self.results.detected_bits if bit is not None]
            bob_bases = [basis for i, basis in enumerate(self.results.measurement_bases) 
                        if self.results.detected_bits[i] is not None]
            
            # Run post-processing protocol
            final_key = self.classical_channel_participant_for_pp.run_post_processing_protocol(
                bits=bob_bits,
                bases=bob_bases,
                test_fraction=self.test_fraction,
                pa_compression_rate=self.pa_compression_rate
            )
            
            if final_key:
                self.logger.info(f"Post-processing successful. Final key length: {len(final_key)}")
                return True
            else:
                self.logger.error("Post-processing failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Post-processing error: {e}")
            return False

    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics."""
        stats = {
            "bob_results": {
                "detections_attempted": self.results.detections_attempted,
                "successful_detections": self.results.successful_detections,
                "no_detections": self.results.no_detections,
                "ambiguous_detections": self.results.ambiguous_detections,
                "detection_rate": (self.results.successful_detections / max(1, self.results.detections_attempted)),
                "total_runtime_seconds": self.results.total_runtime_seconds,
                "average_detection_rate_hz": self.results.average_detection_rate_hz
            }
        }
        
        if self.timetagger_controller:
            detection_stats = self.timetagger_controller.get_detector_statistics()
            stats["timetagger_controller"] = detection_stats
        
        return stats

    def shutdown(self) -> None:
        """Shutdown Bob's system."""
        self.logger.info("Shutting down Bob CPU...")
        
        # Stop detection process
        self._stop_detection.set()
        
        # Stop mock transmitter
        if self.use_mock_transmitter:
            self.stop_mock_transmitter()
        
        # Close quantum channel connection
        if self.quantum_client:
            try:
                self.quantum_client.close()
            except Exception as e:
                self.logger.error(f"Error closing quantum client: {e}")
        
        # Shutdown TimeTagger controller
        if self.timetagger_controller:
            self.timetagger_controller.shutdown()
        
        self.logger.info("Bob CPU shutdown complete")

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.shutdown()