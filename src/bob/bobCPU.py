"""
Bob CPU
Includes all the features Alice has: network setup, classical communication, post-processing, etc.
"""
import socket
import time
import logging
import threading
import random
import os
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.data_structures import Pulse, Basis, Bit
from src.bob.timetagger.simple_timetagger_controller import SimpleTimeTaggerController
from src.bob.timetagger.simple_timetagger_base_hardware_simulator import SimpleTimeTaggerHardware, SimpleTimeTaggerSimulator

# Protocol imports for classical communication and post-processing
from src.protocol.qkd_bob_implementation_class import QKDBobImplementation


class BobMode(Enum):
    """Detection modes for Bob - simplified from original complex version."""
    CONTINUOUS = "continuous"      # Continuously measure during pulse periods
    RANDOM_BASIS = "random_basis"  # Randomly select measurement basis (future use)


# Default global values - matching Alice's structure
NUM_THREADS = 1
KEY_LENGTH = 5
LOSS_RATE = 0.0
PULSE_PERIOD = 1.0  # seconds
TEST_FRACTION = 0.1
ERROR_THRESHOLD = 0.11  # Max tolerable QBER
USE_HARDWARE = False
MEASUREMENT_FRACTION = 0.8
DARK_COUNT_RATE = 100.0
USE_MOCK_TRANSMITTER = True
BOBMODE = BobMode.CONTINUOUS

# Network configuration defaults - matching Alice's
IP_ADDRESS_ALICE = "localhost"
IP_ADDRESS_BOB = "localhost" 
PORT_NUMBER_ALICE = 65432
PORT_NUMBER_BOB = 65433
PORT_NUMBER_QUANTUM_CHANNEL = 12345
SHARED_SECRET_KEY = "IzetXlgAnY4oye56"  # 16 bytes for AES-128


@dataclass
class BobConfig:
    """Configuration for Bob CPU"""
    # Detection parameters
    num_expected_pulses: int = 10
    pulse_period_seconds: float = 1.0
    measurement_fraction: float = 0.8  # Fraction of pulse period to measure
    test_fraction: float = 0.1
    error_threshold: float = 0.11  # Max tolerable QBER
    loss_rate: float = 0.0
    
    # Hardware parameters
    use_hardware: bool = False
    detector_channels: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    dark_count_rate: float = 100.0  # For simulator
    mode: BobMode = BobMode.CONTINUOUS
    
    # Network configuration for quantum channel - matches Alice
    use_mock_transmitter: bool = False  # For testing without actual Alice
    listen_qch_host: str = "localhost"
    listen_qch_port: int = 12345
    
    # Classical communication configuration - matches Alice
    alice_ip: str = "localhost"
    alice_port: int = 65432
    bob_ip: str = "localhost"
    bob_port: int = 65433
    shared_secret_key: str = "IzetXlgAnY4oye56"  # 16 bytes for AES-128
    
    # Multi-threading parameters
    num_threads: int = 1
    
    # Post-processing parameters - matches Alice
    enable_post_processing: bool = True
    pa_compression_rate: float = 0.5


@dataclass
class BobResults:
    """Results from Bob's QKD operation."""
    # Detection data - simplified but comprehensive
    pulse_counts: List[Dict[int, int]] = field(default_factory=list)  # Per pulse: {channel: count}
    pulse_ids: List[int] = field(default_factory=list)
    
    # Timing data - matching Alice's structure
    pulse_timestamps: List[float] = field(default_factory=list)
    measurement_times: List[float] = field(default_factory=list)
    
    # Statistics
    pulses_received: int = 0
    pulses_with_counts: int = 0
    total_runtime_seconds: float = 0.0
    average_detection_rate_hz: float = 0.0
    
    # Error tracking - matching Alice
    errors: List[str] = field(default_factory=list)


class BobCPU:
    """
    The main controller for Bob's side of the QKD protocol.
    """
    
    def __init__(self, config: BobConfig):
        """
        Initialize Bob's CPU.

        Args:
            config: Configuration parameters for Bob's operation
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config = config

        # Configuration parameters - matching Alice's style
        self.num_expected_pulses = self.config.num_expected_pulses
        self.pulse_period_seconds = self.config.pulse_period_seconds
        self.measurement_fraction = self.config.measurement_fraction
        self.loss_rate = self.config.loss_rate

        # Hardware parameters
        self.use_hardware = self.config.use_hardware
        self.detector_channels = self.config.detector_channels
        self.dark_count_rate = self.config.dark_count_rate
        self.mode = self.config.mode

        # Network configuration - matching Alice's structure
        self.use_mock_transmitter = self.config.use_mock_transmitter
        self.mock_thread = None
        self.mock_thread_stop_event = None
        self.listen_qch_host = self.config.listen_qch_host
        self.listen_qch_port = self.config.listen_qch_port

        # Classical communication configuration - matching Alice
        self.alice_ip = self.config.alice_ip
        self.alice_port = self.config.alice_port
        self.bob_ip = self.config.bob_ip
        self.bob_port = self.config.bob_port
        self.shared_secret_key = bytes(self.config.shared_secret_key, 'utf-8')

        # Post-processing configuration - matching Alice
        self.enable_post_processing = self.config.enable_post_processing
        self.pa_compression_rate = self.config.pa_compression_rate
        self.test_fraction = self.config.test_fraction
        self.error_threshold = self.config.error_threshold
        self.num_threads = self.config.num_threads

        # Classical communication objects (initialized later) - matching Alice
        self.classical_channel_participant_for_pp: Optional[QKDBobImplementation] = None
        self.quantum_server: Optional[socket.socket] = None
        self.quantum_connection: Optional[socket.socket] = None

        # Protocol markers - EXACTLY matching Alice's values
        self._handshake_marker = 100
        self._handshake_end_marker = 50
        self._acknowledge_marker = 150
        self._not_ready_marker = 200
        self._end_qch_marker = 250
        
        # System state
        self.results = BobResults()
        self._running = False
        self._shutdown_event = threading.Event()
        
        # Components - following Alice's pattern
        self.timetagger_controller: Optional[SimpleTimeTaggerController] = None
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info(f"Bob CPU initialized in {self.config.mode.value} mode")

    def _initialize_components(self) -> None:
        """Initialize all Bob-side components."""
        self.logger.info("Initializing Bob detection components...")
        
        # Initialize TimeTagger Controller (like Alice's LaserController)
        if self.use_hardware:
            timetagger_driver = SimpleTimeTaggerHardware(self.detector_channels)
        else:
            timetagger_driver = SimpleTimeTaggerSimulator(self.detector_channels, self.dark_count_rate)

        self.timetagger_controller = SimpleTimeTaggerController(timetagger_driver)
        
        # Initialize controller
        if not self.timetagger_controller.initialize():
            raise RuntimeError("Failed to initialize TimeTagger controller")

        # Set measurement duration based on pulse period
        # TODO: CONFIRM WHAT IS THIS SUPPOSED TO BE
        # measurement_duration = self.pulse_period_seconds * self.measurement_fraction
        measurement_duration = self.pulse_period_seconds
        if not self.timetagger_controller.set_measurement_duration(measurement_duration):
            raise RuntimeError(f"Failed to set measurement duration to {measurement_duration}s")

        self.logger.info("All components initialized successfully")

    def initialize_system(self) -> bool:
        """
        Initialize the complete Bob system.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Validate configuration
            if not self._validate_configuration():
                return False
                
            self.logger.info("Bob system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            return False

    def _validate_configuration(self) -> bool:
        """Validate Bob's configuration."""
        if self.num_expected_pulses <= 0:
            self.logger.error("Number of expected pulses must be positive")
            return False
            
        if self.pulse_period_seconds <= 0:
            self.logger.error("Pulse period must be positive")
            return False
            
        if not (0.0 < self.measurement_fraction < 1.0):
            self.logger.error("Measurement fraction must be between 0 and 1")
            return False
            
        return True

    # ===============================
    # Network and Communication Setup
    # ===============================
    
    @staticmethod
    def setup_quantum_channel_listener(host: str, port: int, timeout: float = 10.0, max_retries: int = 10, use_mock_transmitter: bool = False) -> socket.socket:
        """Setup quantum channel listener/client with retry logic."""
        if use_mock_transmitter:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))
            listener.listen(1)  # Listen for one connection
            listener.settimeout(timeout)
            return listener
        
        for attempt in range(max_retries):
            try:
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.settimeout(timeout)
                listener.connect((host, port))
                return listener
            except ConnectionRefusedError:
                if attempt < max_retries - 1:
                    print(f"Trying to connect to Alice at {host}:{port}... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(1)  # Wait 1 second before retrying
                    continue
                else:
                    raise ConnectionRefusedError(f"Could not connect to Alice at {host}:{port} after {max_retries} attempts")
            except Exception as e:
                listener.close()
                raise e

    def setup_classical_communication(self) -> bool:
        """Setup classical communication channel."""
        try:
            if QKDBobImplementation is None:
                self.logger.warning("QKDBobImplementation not available, skipping classical communication setup")
                return True
            
            # Setup Bob role for classical communication
            self.classical_channel_participant_for_pp = QKDBobImplementation(
                self.alice_ip, self.alice_port,
                self.bob_ip, self.bob_port,
                self.shared_secret_key
            )
            self.classical_channel_participant_for_pp.setup_role_bob()
            self.classical_channel_participant_for_pp.start_read_communication_channel()
            self.logger.info("Classical communication channel setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup classical communication: {e}")
            return False

    def setup_mock_transmitter(self) -> None:
        """Setup mock transmitter for testing."""
        if not self.use_mock_transmitter:
            return
        
        self.mock_thread_stop_event = threading.Event()
        self.mock_thread = threading.Thread(
            target=self._mock_transmitter_client, 
            name="MockAliceTransmitter", 
            daemon=True
        )
        self.mock_thread.start()
        self.logger.info("Mock transmitter started")

    def _mock_transmitter_client(self) -> None:
        """Mock transmitter client that simulates Alice connecting to Bob."""
        try:
            time.sleep(2)  # Give Bob time to start listening
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.listen_qch_host, self.listen_qch_port))
                
                # Send handshake marker
                sock.sendall(self._handshake_marker.to_bytes(1, 'big'))
                self.logger.info("Mock transmitter: Sent handshake marker")
                
                # Wait for ACK from Bob
                ack = sock.recv(1)
                if ack and ack[0] == self._acknowledge_marker:
                    self.logger.info("Mock transmitter: Received ACK from Bob")
                    
                    # Simulate transmission duration - Bob will measure continuously
                    transmission_time = self.num_expected_pulses * self.pulse_period_seconds
                    self.logger.info(f"Mock transmitter: Simulating {transmission_time}s transmission")
                    time.sleep(transmission_time)
                    
                    # Send end marker
                    sock.sendall(self._handshake_end_marker.to_bytes(1, 'big'))
                    self.logger.info("Mock transmitter: Sent end marker")
                    
                    # Wait for final ACK from Bob
                    final_ack = sock.recv(1)
                    if final_ack and final_ack[0] == self._acknowledge_marker:
                        self.logger.info("Mock transmitter: Received final ACK from Bob")
                    else:
                        self.logger.warning("Mock transmitter: Did not receive expected final ACK")
                else:
                    self.logger.warning("Mock transmitter: Did not receive expected ACK from Bob")
                    
        except Exception as e:
            if not self.mock_thread_stop_event.is_set():
                self.logger.error(f"Mock transmitter error: {e}")

    def stop_mock_transmitter(self) -> None:
        """Stop mock transmitter."""
        if self.use_mock_transmitter and self.mock_thread_stop_event is not None:
            self.mock_thread_stop_event.set()
            if self.mock_thread and self.mock_thread.is_alive():
                self.mock_thread.join(timeout=2.0)
            self.logger.info("Mock transmitter stopped")

    # ===============================
    # Complete Protocol Methods
    # ===============================

    def run_complete_qkd_protocol(self) -> bool:
        """
        Run the complete QKD protocol including quantum reception and post-processing.
        
        Returns:
            bool: True if protocol completed successfully
        """
        self.logger.info("Starting complete QKD protocol...")
        
        try:
            # Step 1: Initialize system
            if not self.initialize_system():
                return False
            
            # Step 2: Setup classical communication
            if self.enable_post_processing:
                if not self.setup_classical_communication():
                    return False
            
            # Step 3: Setup mock transmitter if needed
            self.setup_mock_transmitter()

            # Step 4: Run quantum reception
            if not self.run_quantum_reception():
                return False
            # Step 5: Run post-processing if enabled
            if self.enable_post_processing:
                if not self.run_post_processing():
                    return False
            
            self.logger.info("Complete QKD protocol finished successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"QKD protocol failed: {e}")
            return False
        finally:
            self.shutdown_components()
            print("DEBUG: Completed shutdown components in run_complete_qkd_protocol. Waiting 2 seconds before closing server.")
            time.sleep(2) # Wait to ensure all threads close properly
            self.cleanup_network_resources()
                 
    def run_quantum_reception(self) -> bool:
        """
        Run quantum reception - waits for Alice and measures pulses.
        Similar to Alice's quantum transmission setup.
        """
        if self._running:
            self.logger.warning("Reception already running")
            return False
        
        try:
            if not self.initialize_system():
                return False

            self.quantum_connection = self.setup_quantum_channel_listener(self.listen_qch_host, self.listen_qch_port, use_mock_transmitter=self.use_mock_transmitter)
            self.logger.info(f"Connected to Alice at {self.listen_qch_host}:{self.listen_qch_port}")
            if self.use_mock_transmitter:
                self.logger.info("Using mock transmitter for testing")
                self.quantum_connection, client_address = self.quantum_connection.accept()
            with self.quantum_connection:
                return self.run_quantum_reception_with_connection(self.quantum_connection)
        except Exception as e:
            self.logger.error(f"Error in quantum reception: {e}")
            return False
        finally:
            self._running = False

    def run_quantum_reception_with_connection(self, connection: socket.socket) -> bool:
        """
        Run quantum reception using the provided connection.
        
        Args:
            connection: Established quantum channel connection
            
        Returns:
            bool: True if reception successful
        """
        if self._running:
            self.logger.warning("Reception already running")
            return False
                
        try:
            if not self.initialize_system():
                return False
            
            self._running = True

            # Wait for handshake from Alice
            handshake = connection.recv(1)
            if not handshake or handshake[0] != self._handshake_marker:
                raise RuntimeError("Invalid handshake from Alice")
            
            self.logger.debug("Received handshake from Alice")
            
            # Send acknowledgment (ready to receive)
            connection.sendall(self._acknowledge_marker.to_bytes(1, 'big'))
            self.logger.info("Sent ACK to Alice - ready to receive")
            
            # Run the reception loop
            self._quantum_reception_loop_with_connection(connection)
            
            # Wait for end marker
            end_marker = connection.recv(1)
            if end_marker and end_marker[0] == self._handshake_end_marker:
                self.logger.debug("Received end marker from Alice")
                # Send final ACK
                connection.sendall(self._acknowledge_marker.to_bytes(1, 'big'))
                self.logger.info("Sent final ACK to Alice")
            else:
                self.logger.warning("Did not receive expected end marker from Alice")
                
        except Exception as e:
            self.logger.error(f"Error receiving quantum bits: {e}")
            raise
        finally:
            self._running = False            
            self.logger.info("Quantum reception completed successfully")
            return True
            
    def _quantum_reception_loop_with_connection(self, connection: socket.socket) -> None:
        """
        Quantum reception loop - Continuous measurement with hardware-gated detectors.
        
        Unlike Alice's transmission timing, Bob measures continuously and relies on
        physical detector gating to handle pulse timing. The TimeTagger captures
        time-binned data showing when (if any) counts occurred during each period.
        """
        start_time = time.time()
        total_measurement_time = self.num_expected_pulses * self.pulse_period_seconds
        
        self.logger.info(f"Starting continuous measurement for {total_measurement_time}s ({self.num_expected_pulses} pulses)")
        
        # Measure continuously for the entire transmission period
        # Hardware detector gating handles the pulse timing
        measure_start = time.time()
        try:
            # Get detailed time-binned data from TimeTagger
            if hasattr(self.timetagger_controller.driver, 'get_timebin_data'):
                # Use advanced method if available
                result = self.timetagger_controller.driver.get_timebin_data(total_measurement_time)
                if 'error' in result:
                    raise Exception(f"TimeTagger error: {result['error']}")
                
                timebin_data = result['timebin_data']  # 2D array [channel][time_bin]
                binwidth_ps = result['binwidth_ps']
                total_counts = result['counts_per_channel']
                
                self.logger.info(f"Continuous measurement completed with {result['n_bins']} time bins")
            else:
                # Fallback to simple measurement
                total_counts = self.timetagger_controller.measure_counts()
                timebin_data = None
                binwidth_ps = int(100e9)  # 10ms default
            
            measure_time = time.time() - measure_start
            self.logger.info(f"Measurement completed in {measure_time:.3f}s, total counts: {sum(total_counts.values())}")
            
            # Extract pulse-by-pulse data from time bins
            for pulse_id in range(self.num_expected_pulses):
                timestamp = pulse_id * self.pulse_period_seconds + self.pulse_period_seconds/2  # Approximate center time
                
                # Extract counts for this specific pulse from time-binned data
                if timebin_data is not None:
                    pulse_counts = self._extract_pulse_counts_from_timebin_data(
                        timebin_data, pulse_id, self.pulse_period_seconds, binwidth_ps
                    )
                else:
                    # Simplified fallback: distribute total counts randomly
                    pulse_counts = {ch: random.randint(0, max(1, count//self.num_expected_pulses)) 
                                  for ch, count in total_counts.items()}
                
                # Record data
                self.results.pulse_counts.append(pulse_counts)
                self.results.pulse_ids.append(pulse_id)
                self.results.pulse_timestamps.append(timestamp)
                self.results.measurement_times.append(measure_time / self.num_expected_pulses)  # Average
                self.results.pulses_received += 1

                # Check if we detected anything for this pulse
                total_counts_this_pulse = sum(pulse_counts.values())
                if total_counts_this_pulse > 0:
                    self.results.pulses_with_counts += 1
                    self.logger.debug(f"Pulse {pulse_id}: {total_counts_this_pulse} total counts - {pulse_counts}")
                else:
                    self.logger.debug(f"Pulse {pulse_id}: No counts detected")
                    
        except Exception as e:
            self.logger.error(f"Error in continuous measurement: {e}")
            self.results.errors.append(f"Continuous measurement: {str(e)}")

        # Calculate final statistics
        self.results.total_runtime_seconds = time.time() - start_time
        if self.results.total_runtime_seconds > 0:
            self.results.average_detection_rate_hz = self.results.pulses_with_counts / self.results.total_runtime_seconds
    
    def _extract_pulse_counts_from_timebin_data(self, timebin_data: any, pulse_id: int, pulse_period_seconds: float, binwidth_ps: int) -> Dict[int, int]:
        """
        Extract counts for a specific pulse from time-binned data.
        
        Uses the actual TimeTagger time-bin data to determine which counts 
        occurred during this specific pulse's expected arrival window.
        
        Args:
            timebin_data: Raw 2D array from TimeTagger getData() [channel][time_bin]
            pulse_id: Which pulse period we're extracting (0, 1, 2, ...)
            pulse_period_seconds: Time between pulses
            binwidth_ps: Bin width in picoseconds
            
        Returns:
            Dict[int, int]: Counts for this specific pulse
        """
        pulse_counts = {}
        
        try:
            # Calculate which time bins correspond to this pulse
            pulse_start_time_ps = pulse_id * pulse_period_seconds * 1e12  # Convert to picoseconds
            pulse_end_time_ps = (pulse_id + 1) * pulse_period_seconds * 1e12
            
            # Convert to bin indices
            start_bin = int(pulse_start_time_ps / binwidth_ps)
            end_bin = int(pulse_end_time_ps / binwidth_ps)

            print(f"  DEBUG: Extracting pulse {pulse_id}: start_bin={start_bin} (time={start_bin * binwidth_ps / 1e12}s), end_bin={end_bin-1} (time={end_bin * binwidth_ps / 1e12}s), binwidth={binwidth_ps / 1e12}s")
            
            # Extract counts from the relevant time bins for each channel
            for i, channel in enumerate(self.detector_channels):
                if i < len(timebin_data) and timebin_data[i] is not None:
                    channel_bins = timebin_data[i]
                    
                    # Sum counts in bins corresponding to this pulse period
                    pulse_total = 0
                    # print(f"DEBUG: Channel {channel} has {len(channel_bins)} bins")
                    for bin_idx in range(max(0, start_bin), min(len(channel_bins), end_bin)):
                        pulse_total += channel_bins[bin_idx]
                        # print(f"DEBUG: Channel {channel}, bin {bin_idx}, count {channel_bins[bin_idx]}")
                    
                    pulse_counts[channel] = int(pulse_total)
                    # print(f"DEBUG: Channel {channel}, pulse {pulse_id}, total counts: {pulse_total}")
                else:
                    pulse_counts[channel] = 0
            
            # Debug logging
            total_pulse_counts = sum(pulse_counts.values())
            if total_pulse_counts > 0:
                print(f"    DEBUG: Pulse {pulse_id} counts: {pulse_counts} -> {total_pulse_counts} total from bins {start_bin}-{end_bin-1}")
                self.logger.debug(f"Pulse {pulse_id} counts: {pulse_counts} -> {total_pulse_counts} total from bins {start_bin}-{end_bin-1}")

        except Exception as e:
            self.logger.error(f"Error extracting pulse {pulse_id} from timebin data: {e}")
            # Fallback: zero counts
            pulse_counts = {ch: 0 for ch in self.detector_channels}
        
        return pulse_counts
               
    def run_post_processing(self) -> bool:
        """
        Run post-processing including sifting, error correction, and privacy amplification.
        
        Returns:
            bool: True if post-processing successful
        """
        if not self.classical_channel_participant_for_pp:
            self.logger.error("Classical channel not initialized")
            return False
        
        try:
            # Convert detection data to format expected by post-processing
            # We need to convert our count data to detected bits and bases
            detected_bits, detected_bases, detected_idxs = self._convert_counts_to_detection_data()
            
            if not detected_bits:
                self.logger.warning("No detection data available for post-processing")
                return False
            
            self.logger.info(f"Starting post-processing with {len(detected_bits)} detected bits")
            
            # Calculate average time bin from pulse period (in nanoseconds like original BB84)
            # Original protocol uses nanoseconds, our pulse_period is in seconds
            average_time_bin_ns = self.pulse_period_seconds * 1e9  # Convert to nanoseconds
            self.logger.debug(f"Calculated average_time_bin: {average_time_bin_ns} ns ({average_time_bin_ns/1e6:.3f} ms) from pulse period {self.pulse_period_seconds}s")
            
            # Set Bob's detected quantum variables (like in the original protocol)
            self.classical_channel_participant_for_pp.set_bob_detected_quantum_variables(
                detected_bits=detected_bits,
                detected_bases=detected_bases, 
                detected_qubits_bytes=detected_bits,  # Same as bits for now
                detected_idxs=detected_idxs,
                average_time_bin=average_time_bin_ns
            )
            
            # Run post-processing using Bob's classical process
            self.classical_channel_participant_for_pp.bob_run_qkd_classical_process_threading(
                detected_bits, detected_bases, detected_bits,  # qubits_bytes same as bits
                detected_idxs, average_time_bin_ns,  # Use calculated time bin
                True, self.test_fraction, self.pa_compression_rate
            )

            self.classical_channel_participant_for_pp.bob_join_threads()
            final_key = self.classical_channel_participant_for_pp.get_secured_key()
            qber = self.classical_channel_participant_for_pp.get_qber()

            if final_key is not None:
                self.logger.info(f"Post-processing completed. Final key generated successfully")
                self.logger.info(f" -----> FINAL KEY: {final_key} <----- ")
                self.logger.info(f" -----> QBER: {qber} <----- ")

                return True
            else:
                self.logger.warning("Post-processing failed to generate a key")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in post-processing: {e}")
            return False
    
    def _convert_counts_to_detection_data(self) -> tuple:
        """
        Convert TimeTagger count data to detection bits, bases, and indices.
        This simulates the detection process for post-processing.
        
        Returns:
            tuple: (detected_bits, detected_bases, detected_idxs)
        """
        detected_bits = []
        detected_bases = []
        detected_idxs = []
        
        try:
            # Use actual pulse IDs and timestamps from results instead of enumerate
            for (pulse_id, counts, timestamp) in zip(
                self.results.pulse_ids, 
                self.results.pulse_counts, 
                self.results.pulse_timestamps
            ):
                # Simple detection logic: if any counts detected, determine bit value
                total_counts = sum(counts.values())
                
                if total_counts > 0:
                    # Use the actual pulse_id from results
                    detected_idxs.append(pulse_id)

                    # For BB84: channels 3,4 = Z basis, channels 1,2 = X basis
                    z_counts = counts.get(3, 0) + counts.get(4, 0)
                    x_counts = counts.get(1, 0) + counts.get(2, 0)

                    if z_counts > x_counts:
                        # Z basis measurement
                        detected_bases.append(0)  # Z basis = 0
                        # Bit value: channel 1 = bit 1, channel 2 = bit 0
                        if counts.get(1, 0) > counts.get(2, 0):
                            detected_bits.append(1)
                        else:
                            detected_bits.append(0)
                    else:
                        # X basis measurement  
                        detected_bases.append(1)  # X basis = 1
                        # Bit value: channel 3 = bit 1, channel 4 = bit 0
                        if counts.get(3, 0) > counts.get(4, 0):
                            detected_bits.append(1)
                        else:
                            detected_bits.append(0)

            self.logger.debug(f"Converted {len(detected_bits)} detections from {len(self.results.pulse_counts)} pulses")
            self.logger.debug(f"Detection pulse IDs: {detected_idxs}")
            self.logger.debug(f"Detection bases: {detected_bases}")
            self.logger.debug(f"Detection bits: {detected_bits}")
            return detected_bits, detected_bases, detected_idxs
            
        except Exception as e:
            self.logger.error(f"Error converting counts to detection data: {e}")
            return [], [], []

    def shutdown_components(self) -> None:
        """Shutdown components."""
        try:
            if self.timetagger_controller:
                self.timetagger_controller.shutdown()
        except Exception as e:
            self.logger.error(f"Error shutting down components: {e}")

    def cleanup_network_resources(self) -> None:
        """Cleanup network resources."""
        try:
            # if self.classical_channel_participant_for_pp:
            #     try:
            #         self.logger.info("Stopping classical communication...")
            #         self.classical_channel_participant_for_pp.bob_join_threads()
            #         self.classical_channel_participant_for_pp._role_bob._stop_all_threads()
            #     except (ConnectionResetError, OSError) as e:
            #         # These are expected during shutdown when the other party closes first
            #         self.logger.debug(f"Expected connection error during shutdown: {e}")
            #     except Exception as e:
            #         self.logger.error(f"Unexpected error stopping classical channel: {e}")

            if self.use_mock_transmitter:
                self.stop_mock_transmitter()
            
            # if self.quantum_connection:
            #     self.quantum_connection.close()
            #     self.quantum_connection = None
                
            # if self.quantum_server:
            #     self.quantum_server.close()
            #     self.quantum_server = None
                
        except Exception as e:
            self.logger.error(f"Error cleaning up network resources: {e}")

    def get_results(self) -> BobResults:
        """Get results."""
        return self.results

    def is_running(self) -> bool:
        """Check if running."""
        return self._running

    def get_progress(self) -> float:
        """Get progress."""
        if self.num_expected_pulses == 0:
            return 0.0
        return (self.results.pulses_received / self.num_expected_pulses) * 100.0

    def get_component_info(self) -> Dict[str, Any]:
        """Get component info."""
        info = {
            "timetagger_controller": None,
            "detector_channels": self.detector_channels,
            "use_hardware": self.use_hardware,
        }
        
        if self.timetagger_controller:
            info["timetagger_controller"] = self.timetagger_controller.get_status()
        
        return info
        
    def __del__(self):
        """Cleanup on destruction."""
        self.shutdown_components()
        self.cleanup_network_resources()


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("BobCPU.Main")
    
    # Example configuration for complete QKD protocol
    config = BobConfig(
        # Detection parameters
        num_expected_pulses=100,
        pulse_period_seconds=0.1,
        measurement_fraction=0.8,
        loss_rate=0.0,
        # Hardware parameters
        use_hardware=False,
        detector_channels=[1, 2, 3, 4],
        dark_count_rate=50.0,
        mode=BobMode.CONTINUOUS,
        # Quantum channel parameters
        use_mock_transmitter=False,
        # listen_qch_host="localhost",
        listen_qch_host="127.0.0.1",
        listen_qch_port=12345,
        # Classical communication parameters
        # alice_ip="localhost",
        alice_ip="127.0.0.1",
        alice_port=54321,
        # bob_ip="localhost", 
        bob_ip="127.0.0.1",
        bob_port=54322,
        shared_secret_key="IzetXlgAnY4oye56",
        # Post-processing parameters
        enable_post_processing=True,
        test_fraction=0.25,
        error_threshold=0.61,
        pa_compression_rate=0.5,
        num_threads=1
    )
    
    # Create Bob CPU instance
    bob_cpu = BobCPU(config)
    
    try:
        logger.info("Starting Bob QKD protocol...")
        
        if bob_cpu.run_complete_qkd_protocol():
            results = bob_cpu.get_results()
            logger.info(f"QKD protocol completed successfully")
            logger.info(f"Received {results.pulses_received} pulses, {results.pulses_with_counts} with detections")
        else:
            logger.error("QKD protocol failed")
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        bob_cpu.shutdown_components()
        bob_cpu.cleanup_network_resources()