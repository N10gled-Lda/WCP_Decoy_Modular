"""
Alice's main controller for QKD transmission.
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

from src.utils.data_structures import Pulse, Basis, Bit, LaserInfo
from src.alice.qrng.qrng_simulator import QRNGSimulator, OperationMode
from src.alice.laser.laser_controller import LaserController
from src.alice.laser.laser_simulator import SimulatedLaserDriver
from src.alice.laser.laser_hardware_digital import DigitalHardwareLaserDriver
from src.alice.polarization.polarization_controller import PolarizationController, PolarizationOutput
from src.alice.polarization.polarization_simulator import PolarizationSimulator
from src.alice.polarization.polarization_hardware import PolarizationHardware

# Protocol imports for classical communication and post-processing
from src.protocol.qkd_alice_implementation_class import QKDAliceImplementation

class AliceMode(Enum):
    """Test modes for Alice hardware.
        - RANDOM_STREAM: Generate random bits using QRNG bit by bit
        - RANDOM_BATCH: Generate random bits in batch using QRNG all before
        - SEEDED: Generate bits using a fixed seed using QRNG
        - PREDETERMINED: Use pre-defined sequence
    """
    RANDOM_STREAM = "random_stream"     # Generate random bits using QRNG bit by bit
    RANDOM_BATCH = "random_batch"       # Generate random bits in batch using QRNG all before
    SEEDED = "seeded"                   # Generate bits using a fixed seed using QRNG
    PREDETERMINED = "predetermined"     # Use pre-defined sequence


# Default global values for parameters
NUM_THREADS = 1
KEY_LENGTH = 5
LOSS_RATE = 0.0
PULSE_PERIOD = 1.0  # seconds
TEST_FRACTION = 0.1
ERROR_THRESHOLD = 0.11  # Max tolerable QBER
USE_HARDWARE = True
COM_PORT = "COM4"
LASER_CHANNEL = 8
USE_MOCK_RECEIVER = True
ALICEMODE = AliceMode.RANDOM_STREAM
QRNG_SEED = 40
# Network configuration defaults
IP_ADDRESS_ALICE = "localhost"
IP_ADDRESS_BOB = "localhost" 
# IP_ADDRESS_ALICE = "127.0.0.1"
# IP_ADDRESS_BOB = "127.0.0.2"
PORT_NUMBER_ALICE = 65432
PORT_NUMBER_BOB = 65433
PORT_NUMBER_QUANTUM_CHANNEL = 12345
SHARED_SECRET_KEY = "IzetXlgAnY4oye56"  # 16 bytes for AES-128


@dataclass
class AliceConfig:
    """Configuration for Alice CPU."""
    # Quantum transmission parameters
    # Protocol parameters
    num_pulses: int = 10
    pulse_period_seconds: float = 1.0
    test_fraction: float = 0.1
    error_threshold: float = 0.11  # Max tolerable QBER
    loss_rate: float = 0.0

    # Hardware parameters
    use_hardware: bool = False
    com_port: Optional[str] = None  # For polarization hardware
    laser_channel: Optional[int] = None  # For hardware laser
    qrng_seed: Optional[int] = None
    mode: AliceMode = AliceMode.RANDOM_STREAM
    # Predetermined sequences (only used if mode=PREDETERMINED); 
    # must be of the size of num_pulses
    predetermined_bits: Optional[List[int]] = None
    predetermined_bases: Optional[List[int]] = None
    # Information about the laser (for simulators)
    laser_info: LaserInfo = field(default_factory=LaserInfo) # For simulated laser
    
    # Network configuration for quantum channel
    use_mock_receiver: bool = False  # For testing without actual Bob
    server_qch_host: str = "localhost"
    server_qch_port: int = 12345
    
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
class AliceResults:
    """Comprehensive results from Alice's QKD operation."""
    # Quantum data
    bits: List[Bit] = field(default_factory=list)
    bases: List[Basis] = field(default_factory=list)
    polarization_angles: List[float] = field(default_factory=list)
    pulse_ids: List[int] = field(default_factory=list)
    
    # Timing data
    pulse_timestamps: List[float] = field(default_factory=list)
    rotation_times: List[float] = field(default_factory=list)
    wait_times: List[float] = field(default_factory=list)  # Time waiting for polarization readiness
    laser_times: List[float] = field(default_factory=list)
    
    # Statistics
    pulses_sent: int = 0
    total_runtime_seconds: float = 0.0
    average_pulse_rate_hz: float = 0.0
    
    # Error tracking
    errors: List[str] = field(default_factory=list)


class AliceCPU:
    """
    The main controller for Alice's side of the QKD protocol.
    
    Controls laser and polarization hardware/simulators with QRNG for 
    BB84 quantum key distribution. Supports both batch and streaming modes
    for random bit generation.
    """

    def __init__(self, config: AliceConfig):
        """
        Initialize Alice's CPU.

        Args:
            config: Configuration parameters for Alice's operation
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config = config

        # Configuration parameters
        # Protocol parameters
        self.num_pulses = self.config.num_pulses
        self.pulse_period_seconds = self.config.pulse_period_seconds
        self.test_fraction = self.config.test_fraction
        self.error_threshold = self.config.error_threshold
        self.loss_rate = self.config.loss_rate

        # Hardware parameters
        self.use_hardware = self.config.use_hardware
        self.com_port = self.config.com_port
        self.laser_channel = self.config.laser_channel
        self.qrng_seed = self.config.qrng_seed
        self.mode = self.config.mode

        # Simulation parameters
        self.laser_info = self.config.laser_info

        # Predetermined sequences
        self.predetermined_bits = self.config.predetermined_bits
        self.predetermined_bases = self.config.predetermined_bases

        # Server configuration for quantum channel
        self.use_mock_receiver = self.config.use_mock_receiver
        self.mock_thread = None
        self.mock_thread_stop_event = None
        self.server_qch_host = self.config.server_qch_host
        self.server_qch_port = self.config.server_qch_port

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
        self.classical_channel_participant_for_pp: Optional[QKDAliceImplementation] = None
        self.quantum_server: Optional[socket.socket] = None
        self.quantum_connection: Optional[socket.socket] = None

        # Protocol markers
        self._handshake_marker = 100
        self._handshake_end_marker = 50
        self._acknowledge_marker = 150
        self._not_ready_marker = 200
        self._end_qch_marker = 250
        
        # System state
        self.results = AliceResults()

        self._running = False
        self._shutdown_event = threading.Event()
        
        # Pre-generated random bits for batch mode
        self.batch_bases: List[Basis] = []
        self.batch_bits: List[Bit] = []
        self.batch_index = 0
        
        # Threading
        self._transmission_thread: Optional[threading.Thread] = None # Needed???
        self._pulse_queue = Queue()
        self._polarized_pulses_queue = Queue()
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info(f"Alice CPU initialized in {self.config.mode.value} mode")

    def _initialize_components(self) -> None:
        """Initialize all Alice-side components."""
        self.logger.info("Initializing Alice hardware components...")
        
        # Initialize QRNG
        self.qrng = QRNGSimulator(
            seed=self.qrng_seed,
            mode=OperationMode.STREAMING if self.mode == AliceMode.RANDOM_STREAM else
                 OperationMode.BATCH if self.mode == AliceMode.RANDOM_BATCH else
                 OperationMode.DETERMINISTIC if self.mode == AliceMode.SEEDED else
                 None  # None for predetermined mode
        )
        
        # Initialize Laser Controller
        if self.use_hardware and self.laser_channel is not None:
            laser_driver = DigitalHardwareLaserDriver(
                digital_channel=self.laser_channel
            )
            self.logger.info(f"Using hardware laser driver on channel {self.laser_channel}")
        else:
            laser_driver = SimulatedLaserDriver(pulses_queue=self._pulse_queue, laser_info=self.laser_info)
            self.logger.info("Using simulated laser driver")

        self.laser_controller = LaserController(laser_driver)
        
        # Initialize Polarization Controller
        if self.use_hardware:
            if self.com_port is None:
                raise ValueError("COM port must be specified for polarization hardware")
            pol_driver = PolarizationHardware(com_port=self.com_port)
            self.logger.info(f"Using hardware polarization driver on COM port {self.com_port}")
        else:
            pol_driver = PolarizationSimulator(pulses_queue=self._pulse_queue, polarized_pulses_queue=self._polarized_pulses_queue, laser_info=self.laser_info)
            self.logger.info("Using simulated polarization driver")

        self.polarization_controller = PolarizationController(
            driver=pol_driver,
            qrng_driver=self.qrng
        )
        
        # Initialize controllers if not initialized
        if not self.laser_controller.initialize():
            raise RuntimeError("Failed to initialize laser controller")

        if not self.polarization_controller.initialize():
            raise RuntimeError("Failed to initialize polarization controller")


        # TODO: Check in the future if this still happens a period after last pulse
        # Set polarization controller period to 1ms to avoid delays when sending one by one
        try:
            if (self.use_hardware):
                self.polarization_controller.driver.set_operation_period(1)
                self.polarization_controller.driver.set_stepper_frequency(1000)
                self.polarization_controller.driver.set_polarization_device(2) # Set HWP                
                self.laser_controller.set_pulse_parameters(duty_cycle=0.1, frequency=1000)
        except Exception as e:
            self.logger.error(f"Error setting polarization hardware parameters (period/frequency): {e}")
            self.results.errors.append(f"Error setting polarization hardware parameters (period/frequency): {e}")
            raise e


        self.logger.info("All components initialized successfully")

    def initialize_system(self) -> bool:
        """
        Initialize the complete Alice system.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize laser
            if not self.laser_controller.initialize():
                self.logger.error("Failed to initialize laser controller")
                return False
            
            # Initialize polarization controller
            if not self.polarization_controller.initialize():
                self.logger.error("Failed to initialize polarization controller")
                return False

            # Validate predetermined sequences
            if not self._validate_seeded_mode():
                raise ValueError("Invalid seeded mode")
            if not self._validate_random_modes():
                raise ValueError("Invalid random modes")
            if not self._validate_predetermined_sequences():
                raise ValueError("Invalid predetermined sequences")
                        
            self.logger.info("Alice system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Alice system: {e}")
            self.results.errors.append(f"Initialization error: {e}")
            return False

    def _validate_predetermined_sequences(self) -> bool:
        """Validate predetermined sequences if provided."""
        if self.mode not in (AliceMode.PREDETERMINED, AliceMode.RANDOM_BATCH):
            return True

        # Validate predetermined sequences sizes
        if self.predetermined_bits is None or self.predetermined_bases is None:
            self.logger.error("Predetermined mode requires both bits and bases to be specified")
            return False
            
        if len(self.predetermined_bits) != self.num_pulses:
            self.logger.error(f"Predetermined bits length ({len(self.predetermined_bits)}) doesn't match num_pulses ({self.num_pulses})")
            return False
            
        if len(self.predetermined_bases) != self.num_pulses:
            self.logger.error(f"Predetermined bases length ({len(self.predetermined_bases)}) doesn't match num_pulses ({self.num_pulses})")
            return False
            
        # Validate values
        for i, bit in enumerate(self.predetermined_bits):
            if bit not in [0, 1]:
                self.logger.error(f"Invalid bit value at index {i}: {bit} (must be 0 or 1)")
                return False
                
        for i, basis in enumerate(self.predetermined_bases):
            if basis not in [0, 1]:
                self.logger.error(f"Invalid basis value at index {i}: {basis} (must be 0 or 1)")
                return False

        self.logger.debug("Predetermined sequences validated successfully: %s", self.predetermined_bits)

        return True

    def _validate_seeded_mode(self) -> bool:
        """Validate seeded mode if specified."""
        if self.mode != AliceMode.SEEDED:
            return True
        if self.qrng.get_mode() != OperationMode.DETERMINISTIC:
            self.logger.warning("Seeded mode requires QRNG to be in deterministic mode")
            self.qrng.set_mode(OperationMode.DETERMINISTIC)
            return True
        if self.qrng_seed is None:
            self.logger.error("Seeded mode requires a seed to be specified")
            return False
        return True
    
    def _validate_random_modes(self) -> bool:
        """Validate random modes if specified."""
        if self.mode == AliceMode.RANDOM_STREAM:
            if self.qrng.get_mode() != OperationMode.STREAMING:
                self.logger.warning("Random stream mode requires QRNG to be in streaming mode")
                self.qrng.set_mode(OperationMode.STREAMING)
                return True
        if self.mode == AliceMode.RANDOM_BATCH:
            if self.qrng.get_mode() != OperationMode.BATCH:
                self.logger.warning("Random batch mode requires QRNG to be in batch mode")
                self.qrng.set_mode(OperationMode.BATCH)
            # Get batch of bases and bits before sending
            self.predetermined_bases = self.qrng.get_random_bit(size=self.num_pulses)
            self.predetermined_bits = self.qrng.get_random_bit(size=self.num_pulses)
            self._validate_predetermined_sequences()
            return True
        return True
    
    # ===============================
    # Network and Communication Setup
    # ===============================
    
    @staticmethod
    def setup_quantum_channel_server(host: str, port: int, timeout: float = 10.0) -> socket.socket:
        """Setup server socket for quantum channel."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(timeout)
        server.bind((host, port))
        server.listen(1)
        return server

    def setup_classical_communication(self) -> bool:
        """Setup classical communication channel. Creates the QKDAliceImplementation instance, and starts the communication thread (alice role)."""
        try:
            print(f"DEBUG: Setting up classical communication channel to Bob at {self.bob_ip}:{self.bob_port} from Alice at {self.alice_ip}:{self.alice_port}")
            self.classical_channel_participant_for_pp = QKDAliceImplementation(
                self.alice_ip, self.alice_port, 
                self.bob_ip, self.bob_port, 
                self.shared_secret_key
            )
            self.classical_channel_participant_for_pp.setup_role_alice()
            self.classical_channel_participant_for_pp.start_read_communication_channel()
            self.logger.info("Classical communication channel setup completed")
            return True
        except Exception as e:
            self.logger.error(f"Failed to setup classical communication: {e}")
            return False

    def setup_mock_receiver(self) -> None:
        """Setup mock receiver for testing."""
        if not self.use_mock_receiver:
            raise RuntimeError("Mock receiver not enabled in configuration")
        
        self.mock_thread_stop_event = threading.Event()
        self.mock_thread = threading.Thread(
            target=self._mock_receiver_client, 
            name="MockBobEcho", 
            daemon=True
        )
        self.mock_thread.start()
        self.logger.info("Mock receiver started")

    def _mock_receiver_client(self) -> None:
        """Mock receiver client that echoes received data."""
        try:
            with socket.create_connection((self.server_qch_host, self.server_qch_port), timeout=5) as s:
                # Handle handshake protocol
                total_bytes = 0
                frames_seen = 0
                handshake_done = False
                while not self.mock_thread_stop_event.is_set():
                    try:
                        data = s.recv(1024)
                    except socket.timeout:
                        continue  # periodic check of stop_event
                    if not data:
                        break
                    total_bytes += len(data)
                    for b in data:
                        if b == self._handshake_marker:  # Listen for handshake marker (200)
                            if not handshake_done:
                                # Send acknowledge marker (150) to signal Bob ready
                                try:
                                    s.sendall(self._acknowledge_marker.to_bytes(1, 'big'))
                                    handshake_done = True
                                    self.logger.info("Mock receiver: Received handshake, sent ACK")
                                except Exception as e:
                                    self.logger.error(f"Mock receiver: Failed to send ACK: {e}")
                        elif b == self._handshake_end_marker:
                            if handshake_done:
                                handshake_done = False
                                self.logger.info("Mock receiver: Received handshake end")
                                # Here you might want to send a final ACK or process the end of the handshake
                                try:
                                    s.sendall(self._acknowledge_marker.to_bytes(1, 'big'))
                                    self.logger.info("Mock receiver: Sent final ACK after handshake end")
                                except Exception as e:
                                    self.logger.error(f"Mock receiver: Failed to send final ACK: {e}")
                        elif b == self._end_qch_marker:
                            self.logger.info("Mock receiver: Received end of quantum channel marker, exiting")
                            if self.mock_thread_stop_event is not None:
                                self.mock_thread_stop_event.set()
                            self.logger.info("Mock receiver: Stopped")
                            break
                        else:
                            # Send back data only if handshake done
                            if not handshake_done:
                                continue
                            # Echo strategy: just send bytes back exactly
                            try:
                                s.sendall(data)
                            except (socket.error, ConnectionResetError) as e:
                                self.logger.error(f"Mock receiver: Error echoing data: {e}")
                                break

        except ConnectionRefusedError:
            self.logger.error("Mock receiver could not connect to Alice server")
        except Exception as e:
            self.logger.error(f"Mock receiver unexpected error: {e}")

    def stop_mock_receiver(self) -> None:
        """Stop mock receiver."""
        if self.use_mock_receiver and self.mock_thread_stop_event is not None:
            self.mock_thread_stop_event.set()
            if self.mock_thread and self.mock_thread.is_alive():
                self.mock_thread.join(timeout=2)
            self.logger.info("Mock receiver stopped")

    # ===============================
    # Complete Protocol Methods
    # ===============================

    def run_complete_qkd_protocol(self) -> bool:
        """
        Run the complete QKD protocol including quantum transmission and post-processing.
        
        Returns:
            bool: True if protocol completed successfully
        """
        self.logger.info("Starting complete QKD protocol...")
        
        try:
            if not self.initialize_system():
                return False

            # Setup network components
            if self.enable_post_processing:
                if not self.setup_classical_communication():
                    return False
            
            # Setup quantum channel server
            server = self.setup_quantum_channel_server(self.server_qch_host, self.server_qch_port)
            self.logger.info(f"Quantum channel server listening on {self.server_qch_host}:{self.server_qch_port}")

            # Setup mock receiver if needed
            if self.use_mock_receiver:
                self.setup_mock_receiver()
            
            self.logger.info("Waiting for Bob to connect to quantum channel...")
            connection, address = server.accept()
            self.quantum_connection = connection
            self.logger.info(f"Quantum channel connected to {address}")
            
            # Run quantum transmission
            success = self.run_quantum_transmission_with_connection(connection)
            
            if success and self.enable_post_processing:
                # Run post-processing
                success = self.run_post_processing()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error in complete QKD protocol: {e}")
            return False
        finally:
            self.shutdown_components()
            print("DEBUG: Completed shutdown components in run_complete_qkd_protocol. Waiting 2 seconds before closing server.")
            time.sleep(2)  # Wait a bit to ensure all data is sent/received
            self.cleanup_network_resources()
            if self.use_mock_receiver:
                self.stop_mock_receiver()
                
    def run_quantum_transmission_with_connection(self, connection: socket.socket) -> bool:
        """
        Run quantum transmission using the provided connection.
        
        Args:
            connection: Established quantum channel connection
            
        Returns:
            bool: True if transmission successful
        """
        if self._running:
            self.logger.warning("Transmission already running")
            return False
                
        # Send quantum bits
        try:
            self._running = True

            # Send handshake to signal start of transmission
            connection.sendall(self._handshake_marker.to_bytes(1, 'big'))
            self.logger.debug("Sent handshake to Bob")
            
            # Wait for acknowledgment
            ack = connection.recv(1)
            if ack == self._not_ready_marker.to_bytes(1, 'big'):
                raise RuntimeError("Bob not ready for quantum transmission")
            if ack != self._acknowledge_marker.to_bytes(1, 'big'):
                raise RuntimeError("Invalid acknowledgment from Bob")
            
            self.logger.info("Received ACK from Bob, starting quantum transmission")
            
            # Run the transmission loop (modified to work with connection)
            self._quantum_transmission_loop_with_connection(connection)
            
            # Send end marker
            connection.sendall(self._handshake_end_marker.to_bytes(1, 'big'))
            self.logger.debug("Sent end marker to Bob")
            
            # Wait for final ACK
            final_ack = connection.recv(1)
            if final_ack != self._acknowledge_marker.to_bytes(1, 'big'):
                self.logger.warning("Did not receive final ACK from Bob")
            else:
                self.logger.info("Received final ACK from Bob")
                
        except Exception as e:
            self.logger.error(f"Error sending quantum bits: {e}")
            raise
        finally:
            self._running = False            
            self.logger.info("Quantum transmission completed successfully")
            return True
            
    def _quantum_transmission_loop_with_connection(self, connection: socket.socket) -> None:
        """Quantum transmission loop that sends data over connection."""
        start_time = time.time()
        laser_fire_fraction = 0.8  # Fraction of pulse period to fire laser after polarization set
        for pulse_id in range(self.num_pulses):
            if self._shutdown_event.is_set():
                break
            
            pulse_start_time = time.time()
            # Calculate when to fire laser for this pulse (consistent timing)
            target_laser_time = start_time + (pulse_id + laser_fire_fraction) * self.config.pulse_period_seconds 
 
            try:
                # Get basis and bit
                basis, bit = self._get_basis_and_bit(pulse_id)
                self.logger.debug(f"Sending qubit {pulse_id}: basis={basis}, bit={bit}")

                # Set polarization
                print(f"🔸 Pulse {pulse_id}: Setting polarization Basis={basis.name}, Bit={bit.value}")
                rotation_start = time.time()
                pol_output = self.polarization_controller.set_polarization_manually(basis, bit)
                
                # Wait for polarization readiness
                print(f"🔸 Pulse {pulse_id}: Waiting for polarization readiness...")
                wait_start = time.time()
                if not self.polarization_controller.wait_for_availability(timeout=10.0):
                    error_msg = f"Timeout waiting for polarization readiness for pulse {pulse_id}"
                    self.logger.error(error_msg)
                    self.results.errors.append(error_msg)
                    raise RuntimeError(f"Polarization not ready for pulse {pulse_id}")
                wait_time = time.time() - wait_start
                rotation_time = time.time() - rotation_start
                self.results.rotation_times.append(rotation_time)
                self.results.wait_times.append(wait_time)
                print(f"   ➡️  Polarization ready after {wait_time:.3f}s")
                print(f"       (Rotation time: {rotation_time:.3f}s)")
                print(f"   ➡️  Polarization set to {pol_output.angle_degrees}°")

                # Wait until the scheduled laser fire time for consistent timing
                current_time = time.time()
                time_until_laser = target_laser_time - current_time
                
                if time_until_laser > 0:
                    print(f"🔸 Pulse {pulse_id}: Waiting {time_until_laser:.3f}s to fire laser at scheduled time")
                    time.sleep(time_until_laser)
                elif time_until_laser < -0.001:  # More than 1ms late
                    self.logger.warning(f" ⚠️ Pulse {pulse_id} is {-time_until_laser:.3f}s late! Polarization took too long.")
                

                # Fire laser
                print(f"🔸 Pulse {pulse_id}: Firing laser at scheduled time")
                laser_start = time.time()
                if not self.laser_controller.trigger_once():
                    self.logger.error(f"Failed to trigger laser pulse {pulse_id}")
                    self.results.errors.append(f"Failed to trigger laser pulse {pulse_id}")
                    raise RuntimeError(f"Failed to trigger laser pulse {pulse_id}")
                self.logger.debug(f"Laser pulse {pulse_id} triggered successfully")
                laser_time = time.time() - laser_start
                timing_error = laser_start - target_laser_time
                print(f"   ➡️  Laser fired in {laser_time:.3f}s (timing error: {timing_error*1000:.1f}ms)")

                # Record data
                self.results.bases.append(basis)
                self.results.bits.append(bit)
                self.results.polarization_angles.append(pol_output.angle_degrees)
                self.results.pulse_ids.append(pulse_id)
                # Record timestamps
                self.results.pulse_timestamps.append(laser_start-start_time)
                self.results.rotation_times.append(rotation_time)
                self.results.wait_times.append(wait_time)
                self.results.laser_times.append(laser_time)

                self.results.pulses_sent += 1

                # Wait for next pulse period (until next pulse should start)
                next_pulse_start = start_time + (pulse_id + 1) * self.config.pulse_period_seconds
                current_time = time.time()
                remaining_time = next_pulse_start - current_time

                if remaining_time > 0:
                    print(f"--------------------------> DEBUG: Waiting {remaining_time:.3f}s for next pulse period, Fired at {laser_start - start_time:.3f}, Target was {target_laser_time- start_time:.3f}")
                    self.logger.debug(f"Waiting {remaining_time:.3f}s for next pulse period, Fired at {laser_start - start_time:.3f}, Target was {target_laser_time- start_time:.3f}")
                    time.sleep(remaining_time)
                elif remaining_time < -0.05:  # More than 50ms late
                    self.logger.warning(f" Pulse {pulse_id} is {-remaining_time:.3f}s late for just a little late.")
                else:
                    total_pulse_time = current_time - pulse_start_time
                    self.logger.warning(f" ⚠️ Pulse {pulse_id} exceeded period: {total_pulse_time:.3f}s > {self.config.pulse_period_seconds}s")
                
                print(f"   ✅ Pulse {pulse_id} completed\n")
          
                
            except Exception as e:
                self.logger.error(f"Error processing pulse {pulse_id}: {e}")
                self.results.errors.append(f"Pulse {pulse_id} error: {e}")
                continue
        
        # Update statistics
        self.results.total_runtime_seconds = time.time() - start_time
        if self.results.total_runtime_seconds > 0:
            self.results.average_pulse_rate_hz = self.results.pulses_sent / self.results.total_runtime_seconds
 
    def _get_basis_and_bit(self, pulse_id: int) -> Tuple[Basis, Bit]:
        """Get basis and bit for the given pulse (unified method)."""
        if self.mode == AliceMode.PREDETERMINED or self.mode == AliceMode.RANDOM_BATCH:
            # Use predetermined values
            basis_val = self.predetermined_bases[pulse_id]
            bit_val = self.predetermined_bits[pulse_id]
        elif self.mode == AliceMode.SEEDED or self.mode == AliceMode.RANDOM_STREAM:
            # Generate random values using QRNG
            basis_val = int(self.qrng.get_random_bit())
            bit_val = int(self.qrng.get_random_bit())
        else:
            raise ValueError("Invalid test mode")
        
        basis = Basis.Z if basis_val == 0 else Basis.X
        bit = Bit(bit_val)
        
        return basis, bit
    
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
            # Convert transmission data to format expected by post-processing
            alice_bits = [int(bit) for bit in self.results.bits]
            alice_bases = [int(basis) for basis in self.results.bases]
            
            # Run post-processing
            self.classical_channel_participant_for_pp.alice_run_qkd_classical_process_threading(
                alice_bits, alice_bases, True, self.test_fraction, self.error_threshold, self.pa_compression_rate
            )
            
            # Wait for all threads to finish
            self.classical_channel_participant_for_pp.alice_join_threads()
            final_key = self.classical_channel_participant_for_pp.get_secured_key()
            qber = self.classical_channel_participant_for_pp.get_qber()

            if final_key is not None:
                self.logger.info(f"Post-processing completed. Final key length: {len(final_key)}")
                self.logger.info(f" -----> FINAL KEY: {final_key} <----- ")
                self.logger.info(f" -----> QBER: {qber:.2f}% <----- ")
                return True
            else:
                self.logger.warning("Post-processing failed to generate a key")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in post-processing: {e}")
            return False

    def shutdown_components(self) -> None:
        """Shutdown all components."""
        self.logger.info("Shutting down hardware components...")        
        try:
            self.laser_controller.shutdown()
            self.polarization_controller.shutdown()
        except Exception as e:
            self.logger.error(f"Error shutting down hardware: {e}")

    def cleanup_network_resources(self) -> None:
        """Cleanup network resources."""
        # if self.classical_channel_participant_for_pp:
        #     try:
        #         self.logger.info("Stopping classical communication...")
        #         self.classical_channel_participant_for_pp.alice_join_threads()
        #         self.classical_channel_participant_for_pp._role_alice._stop_all_threads()
        #     except (ConnectionResetError, OSError) as e:
        #         # These are expected during shutdown when the other party closes first
        #         self.logger.debug(f"Expected connection error during shutdown: {e}")
        #     except Exception as e:
        #         self.logger.error(f"Unexpected error stopping classical channel: {e}")
        
        # if self.quantum_connection:
        #     try:
        #         self.quantum_connection.close()
        #     except Exception as e:
        #         self.logger.error(f"Error closing quantum connection: {e}")
        
        # if self.quantum_server:
        #     try:
        #         self.quantum_server.close()
        #     except Exception as e:
        #         self.logger.error(f"Error closing quantum server: {e}")
    
        if self.use_mock_receiver:
            self.stop_mock_receiver()
    
        self.logger.info("Network resources cleaned up")
   

    def get_results(self) -> AliceResults:
        """Get comprehensive results including statistics and transmission data."""
        return self.results

    def is_running(self) -> bool:
        """Check if transmission is currently running."""
        return self._running

    def get_progress(self) -> float:
        """Get transmission progress as percentage."""
        if self.config.num_pulses == 0:
            return 100.0
        return (self.results.pulses_sent / self.config.num_pulses) * 100.0

    def get_component_info(self) -> Dict[str, Any]:
        """Get information about all components."""
        return {
            "laser_controller": {
                "type": type(self.laser_controller._driver).__name__,
                "initialized": self.laser_controller.is_initialized(),
                "active": self.laser_controller.is_active(),
                "pulse_count": self.laser_controller.pulse_count
            },
            "polarization_controller": {
                "type": type(self.polarization_controller.driver).__name__,
                "initialized": self.polarization_controller.is_initialized()
            },
            "qrng": {
                "type": type(self.qrng).__name__,
                "mode": self.qrng._mode.value if self.qrng._mode else "unknown",
                "bits_generated": self.qrng.get_bits_generated()
            }
        }
        
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.shutdown_components()
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        self.cleanup_network_resources()


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("AliceCPU.Main")
    
    # Example configuration for complete QKD protocol
    config = AliceConfig(
        # Quantum transmission parameters
        num_pulses=100,
        pulse_period_seconds=0.1,  # 1 second between pulses
        use_hardware=False,  # Set to True for actual hardware
        com_port="COM4",  # Replace with actual COM port for polarization hardware
        laser_channel=8,  # Replace with actual digital channel for laser hardware
        mode=AliceMode.RANDOM_STREAM,
        qrng_seed=42,
        
        # Network configuration
        use_mock_receiver=False,  # For testing without actual Bob
        # server_qch_host="localhost",
        server_qch_host="127.0.0.1", 
        server_qch_port=12345,
        
        # Classical communication
        # alice_ip="localhost",
        alice_ip="127.0.0.1",
        alice_port=54321,
        # bob_ip="localhost", 
        bob_ip="127.0.0.1", 
        bob_port=54322,
        shared_secret_key="IzetXlgAnY4oye56",
        
        # Post-processing
        enable_post_processing=True,
        test_fraction=0.25,
        error_threshold=0.61,
        pa_compression_rate=0.5
    )
    
    # Create Alice CPU instance
    alice_cpu = AliceCPU(config)
    
    try:
        logger.info("Starting complete QKD protocol...")
        
        # Run the complete QKD protocol (quantum + post-processing)
        success = alice_cpu.run_complete_qkd_protocol()
        
        if success:
            # Get results
            results = alice_cpu.get_results()
            
            logger.info(f"QKD protocol completed successfully!")
            logger.info(f"Transmitted {results.pulses_sent} pulses in {results.total_runtime_seconds:.2f}s")
            logger.info(f"Average pulse rate: {results.average_pulse_rate_hz:.2f} Hz")
            logger.info(f"Collected data for {len(results.pulse_ids)} pulses")
            
            if results.errors:
                logger.warning(f"Encountered {len(results.errors)} errors during transmission")
        else:
            logger.error("QKD protocol failed")
            
    except KeyboardInterrupt:
        logger.info("QKD protocol interrupted by user")
    except Exception as e:
        logger.error(f"Error running QKD protocol: {e}")
    finally:
        # Cleanup is handled automatically by __del__
        logger.info("Alice CPU session ended")


#### MISSING BEFORE STARTING THE GENERATION AND SENDING, A READY TO RECEIVE SIGNAL WITH BOB THROUGH THE CLASSICAL CHANNEL