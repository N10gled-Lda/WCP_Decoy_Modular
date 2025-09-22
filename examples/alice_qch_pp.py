"""
Alice's BB84 implementation using actual hardware components.
Integrates laser, QRNG, and polarization control with the classical communication protocol.
"""

# Configure logging
import logging

import sys
import os
import socket
import threading
import time
import argparse
import pickle
import queue
from unittest import mock
from enum import Enum
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from examples.logging_setup import setup_logger
logger = setup_logger("Alice Hardware BB84", logging.INFO)

# Protocol imports
from src.protocol.classical_communication_channel.communication_channel.connection_info import ConnectionInfo
from src.protocol.classical_communication_channel.communication_channel.mac_config import MAC_Config, MAC_Algorithms 
from src.protocol.classical_communication_channel.communication_channel.role import Role
from src.protocol.ErrorReconciliation.cascade.key import Key
from src.protocol.qkd_alice_implementation_class import QKDAliceImplementation

# Alice hardware components
from src.alice.laser.laser_controller import LaserController
from src.alice.laser.laser_simulator import SimulatedLaserDriver
from src.alice.laser.laser_hardware_digital import DigitalHardwareLaserDriver
from src.alice.polarization.polarization_controller import PolarizationController, PolarizationOutput
from src.alice.polarization.polarization_simulator import PolarizationSimulator
from src.alice.polarization.polarization_hardware import PolarizationHardware
from src.alice.qrng.qrng_simulator import QRNGSimulator, OperationMode
from src.utils.data_structures import Basis, Bit, Pulse, LaserInfo

TIMEOUT_WAIT_FOR_ROTATION = 10 # 10s

class AliceTestMode(Enum):
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


@dataclass
class HardwareAliceConfig:
    """Configuration for hardware Alice."""
    num_qubits: int = 20
    pulse_period_seconds: float = 1.0  # 1 Hz for demo
    use_hardware: bool = False
    com_port: Optional[str] = None  # For polarization hardware
    laser_channel: Optional[int] = 8  # For hardware laser
    mode: AliceTestMode = AliceTestMode.RANDOM_STREAM
    qrng_seed: Optional[int] = None
    # Predetermined sequences (only used if mode=PREDETERMINED); 
    # must be of the size of num_pulses
    predetermined_bits: Optional[List[int]] = None
    predetermined_bases: Optional[List[int]] = None
    # Server configuration for quantum channel
    use_mock_receiver: bool = False  # For testing without actual Bob
    server_host: str = "localhost"
    server_port: int = 12345
    # Not sure about this ones if are still needed
    test_fraction: float = 0.11
    loss_rate: float = 0.0

@dataclass
class AliceTestResults:
    """Results from Alice hardware test."""
    bits: List[int] = field(default_factory=list)
    bases: List[int] = field(default_factory=list)
    polarization_angles: List[float] = field(default_factory=list)
    pulse_times: List[float] = field(default_factory=list)
    rotation_times: List[float] = field(default_factory=list)
    wait_times: List[float] = field(default_factory=list) # Not needed (included in rotation time) ???
    laser_elapsed: List[float] = field(default_factory=list)
    total_runtime: float = 0.0
    errors: List[str] = field(default_factory=list)


class AliceHardwareQubits:
    """
    Alice's qubit generation and transmission using actual hardware components.
    Replaces the old AliceQubits simulation with real hardware control.
    """

    def __init__(self, config: HardwareAliceConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Protocol parameters
        self.num_qubits = self.config.num_qubits
        self.pulse_period_seconds = self.config.pulse_period_seconds
        self.loss_rate = self.config.loss_rate
        
        # Hardware components
        self.use_hardware = self.config.use_hardware
        self.com_port = self.config.com_port
        self.laser_channel = self.config.laser_channel
        self.mode = self.config.mode
        self.qrng_seed = self.config.qrng_seed

        # Server configuration
        self.use_mock_receiver = self.config.use_mock_receiver
        self.server_host_qch = self.config.server_host
        self.server_port_qch = self.config.server_port
        self.mock_thread = None
        self.mock_thread_stop_event = None
        
        # Predetermined sequences
        self.predetermined_bits = self.config.predetermined_bits
        self.predetermined_bases = self.config.predetermined_bases

        # Protocol markers
        self._handshake_marker = 100
        self._handshake_end_marker = 50
        self._acknowledge_marker = 150
        self._not_ready_marker = 200
        self._end_qch_marker = 250

        # Generated data
        self.bits: List[int] = []
        self.bases: List[int] = []
        self.qubits_bytes: List[int] = []
        self.pulse_times: List[float] = []

        # Replace Generate data with AliceTestResults
        # TODO: Later add a list of these for multiple threads
        self.alice_results = AliceTestResults()
        
        # Hardware components
        self._initialize_hardware()
        
        # Threading
        self._running = False
        self._pulse_thread: Optional[threading.Thread] = None

    def _initialize_hardware(self) -> None:
        """Initialize all hardware components."""
        self.logger.info("Initializing Alice hardware components...")
        
        # Initialize QRNG
        self.qrng = QRNGSimulator(
            seed=self.qrng_seed,
            mode=OperationMode.STREAMING if self.mode == AliceTestMode.RANDOM_STREAM else
                 OperationMode.BATCH if self.mode == AliceTestMode.RANDOM_BATCH else
                 OperationMode.DETERMINISTIC if self.mode == AliceTestMode.SEEDED else
                 None  # None for predetermined mode
        )
        
        # Create shared queues for components
        pulses_queue = queue.Queue()
        polarized_pulses_queue = queue.Queue()
        laser_info = LaserInfo()
        
        # Initialize Laser Controller
        if self.config.use_hardware and self.config.laser_channel is not None:
            laser_driver = DigitalHardwareLaserDriver(
                digital_channel=self.config.laser_channel
            )
            self.logger.info(f"Using hardware laser on channel {self.config.laser_channel}")
        else:
            laser_driver = SimulatedLaserDriver(pulses_queue=pulses_queue, laser_info=laser_info)
            self.logger.info("Using simulated laser")
        
        self.laser_controller = LaserController(laser_driver)
        
        # Initialize Polarization Controller
        if self.config.use_hardware and self.config.com_port is not None:
            pol_driver = PolarizationHardware(com_port=self.config.com_port)
            self.logger.info(f"Using hardware polarization on {self.config.com_port}")
        else:
            pol_driver = PolarizationSimulator(
                pulses_queue=pulses_queue,
                polarized_pulses_queue=polarized_pulses_queue,
                laser_info=laser_info
            )
            self.logger.info("Using simulated polarization")
        
        self.polarization_controller = PolarizationController(
            driver=pol_driver,
            qrng_driver=self.qrng
        )
        
        # Initialize components
        if not self.laser_controller.initialize():
            raise RuntimeError("Failed to initialize laser controller")
        
        if not self.polarization_controller.initialize():
            raise RuntimeError("Failed to initialize polarization controller")
        
        self.logger.info("All hardware components initialized successfully")

    def _validate_predetermined_sequences(self) -> bool:
        """Validate predetermined sequences if provided."""
        if self.config.mode not in (AliceTestMode.PREDETERMINED, AliceTestMode.RANDOM_BATCH):
            return True
        
        # Validade predetertermined sequences sizes
        if self.config.predetermined_bits is None or self.config.predetermined_bases is None:
            self.logger.error("Predetermined mode requires both bits and bases to be specified")
            return False
            
        if len(self.config.predetermined_bits) != self.config.num_qubits:
            self.logger.error(f"Predetermined bits length ({len(self.config.predetermined_bits)}) doesn't match num_qubits ({self.config.num_qubits})")
            return False
            
        if len(self.config.predetermined_bases) != self.config.num_qubits:
            self.logger.error(f"Predetermined bases length ({len(self.config.predetermined_bases)}) doesn't match num_qubits ({self.config.num_qubits})")
            return False
            
        # Validate values
        for i, bit in enumerate(self.config.predetermined_bits):
            if bit not in [0, 1]:
                self.logger.error(f"Invalid bit value at index {i}: {bit} (must be 0 or 1)")
                return False
                
        for i, basis in enumerate(self.config.predetermined_bases):
            if basis not in [0, 1]:
                self.logger.error(f"Invalid basis value at index {i}: {basis} (must be 0 or 1)")
                return False

        self.logger.debug("Predetermined sequences validated successfully: %s", self.config.predetermined_bits)

        return True

    def _validate_seeded_mode(self) -> bool:
        """Validate seeded mode if specified."""
        if self.config.mode != AliceTestMode.SEEDED:
            return True
        if self.qrng.get_mode() != OperationMode.DETERMINISTIC:
            self.logger.warning("Seeded mode requires QRNG to be in deterministic mode")
            self.qrng.set_mode(OperationMode.DETERMINISTIC)
            return True
        if self.config.qrng_seed is None:
            self.logger.error("Seeded mode requires a seed to be specified")
            return False
        return True
    
    def _validate_random_modes(self) -> bool:
        """Validate random modes if specified."""
        if self.config.mode == AliceTestMode.RANDOM_STREAM:
            if self.qrng.get_mode() != OperationMode.STREAMING:
                self.logger.warning("Random stream mode requires QRNG to be in streaming mode")
                self.qrng.set_mode(OperationMode.STREAMING)
                return True
        if self.config.mode == AliceTestMode.RANDOM_BATCH:
            if self.qrng.get_mode() != OperationMode.BATCH:
                self.logger.warning("Random batch mode requires QRNG to be in batch mode")
                self.qrng.set_mode(OperationMode.BATCH)
            # Get batch of bases and bits before sending
            self.config.predetermined_bases = self.qrng.get_random_bit(size=self.config.num_qubits)
            self.config.predetermined_bits = self.qrng.get_random_bit(size=self.config.num_qubits)
            self._validate_predetermined_sequences()
            return True
        return True
    
    @staticmethod
    def setup_server(host: str, port: int, timeout: float = 10.0) -> socket.socket:
        """Setup server socket for quantum channel."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(timeout)
        server.bind((host, port))
        server.listen(1)
        return server

    # TODO: Check if this mock receiver is doing the send back what it recevies after implemening the send_qubits
    def _mock_receiver_client(self):
        try:
            with socket.create_connection((self.server_host_qch, self.server_port_qch), timeout=5) as s:
                s.settimeout(1.0)
                total_bytes = 0
                frames_seen = 0
                # Simple handshake: wait for handshake marker (200) then send acknowledge marker (150)
                handshake_done = False
                while not self.mock_thread_stop_event.is_set():
                    try:
                        data = s.recv(1024)
                    except socket.timeout:
                        continue  # periodic check of stop_event
                    if not data:
                        break
                    total_bytes += len(data)
                    # Basic parsing / logging of markers
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
                self.logger.info(f"Mock receiver exiting. Total echoed bytes={total_bytes}, frames_seen~={frames_seen}")
        except ConnectionRefusedError:
            self.logger.error("Mock receiver could not connect to Alice server (connection refused)")
        except Exception as e:
            self.logger.error(f"Mock receiver unexpected error: {e}")
    
    def setup_mock_receiver(self):
        # If requested, spin up a lightweight local mock "Bob" that connects as a client
        if self.config.use_mock_receiver:
            self.mock_thread_stop_event = threading.Event()

            # Start client thread BEFORE accept so that connect() happens and accept() unblocks
            self.mock_thread = threading.Thread(target=self._mock_receiver_client, name="MockBobEcho", daemon=True)
            self.mock_thread.start()
        else:
            raise RuntimeError("Mock receiver not enabled, cannot start mock thread (define in config use_mock_receiver=True)")

    def stop_mock_receiver(self):
        if self.config.use_mock_receiver and self.mock_thread_stop_event is not None:
            self.mock_thread_stop_event.set()
            if self.mock_thread and self.mock_thread.is_alive():
                self.mock_thread.join(timeout=3.0)
            self.logger.info("Mock receiver stopped")

    def run_trl4_protocol_quantum_part(self, connection: socket.socket) -> AliceTestResults:
        """Run the protocol TRL4 for simple test of the hardware components with the network."""
        self.logger.info("Starting TRL4 protocol...")
        
        start_time = time.time()
        try:
            with connection:
                # Run the qubit generation using the hardware/simulation
                try:
                    # Send hardware qubits directly (no need for thread since we wait for completion anyway)
                    self.alice_results = AliceTestResults()  # Reset results
                    self.send_alice_hardware_qubits(connection, self.alice_results)
                    self.logger.info(f"Sent {len(self.alice_results.bits)} qubits using hardware")
                except Exception as e:
                    self.logger.error(f"Error during TRL4 qubit run: {e}")
                    self.alice_results.errors.append(str(e))
        finally:
            # Send end of qubit transmission marker if connection still open
            try:
                if connection and not connection._closed:
                    connection.sendall(self._end_qch_marker.to_bytes(1, 'big'))
            except (socket.error, AttributeError, OSError) as e:
                self.logger.error(f"Error sending end of quantum channel marker: {e}")

            self._running = False
            # TODO: Shutdown hardware components only after all threads are done. Need to change this to a function 
            # that is called at the end of the main program or when the object is deleted
            self._shutdown_hardware()

        # Populate results structure
        self.bits = self.alice_results.bits.copy()
        self.bases = self.alice_results.bases.copy()
        self.qubits_bytes = [self._bits_bases_to_byte(bit,base) for bit, base in zip(self.bits, self.bases)]
        self.pulse_times = self.alice_results.pulse_times.copy()
        self.alice_results.total_runtime = time.time() - start_time

        self.logger.info(f"TRL4 protocol completed in {self.alice_results.total_runtime:.3f}s; sent {len(self.alice_results.bits)} qubits")
        return self.alice_results


    def send_alice_hardware_qubits(self, connection: socket.socket, results: AliceTestResults) -> None:
        """Send Alice's qubits using hardware components."""
        # TODO: When multiple threads of sending qubits, this can't wait here - check in the before starting
        if not self._validate_seeded_mode():
            raise ValueError("Invalid seeded mode")
        if not self._validate_random_modes():
            raise ValueError("Invalid random modes")
        if not self._validate_predetermined_sequences():
            raise ValueError("Invalid predetermined sequences")

        if not self.laser_controller.is_initialized():
            raise RuntimeError("Laser controller is not initialized")
    
        # TODO: When multiple threads of sending qubits, this can't wait here - set one time before starting all threads
        # !!! Set period of stepmottor for 1ms since the stepmottor wait for this period after sending a single pulse afecting the check availability if high !!!
        try:
            if (self.use_hardware):
                self.polarization_controller.driver.set_operation_period(1)
                self.polarization_controller.driver.set_stepper_frequency(500)
        except Exception as e:
            self.logger.error(f"Error setting polarization hardware parameters (period/frequency): {e}")
            results.errors.append(f"Error setting polarization hardware parameters (period/frequency): {e}")
            raise e
        
        # Start pulse generation thread
        self._running = True
        try:
            # Send handshake to bob to signal start of qubit transmission
            connection.sendall(self._handshake_marker.to_bytes(1, 'big'))
            self.logger.debug("Sent handshake to Bob")
            # Wait for acknowledgment from Bob to start sending qubits
            ack = connection.recv(1)
            if ack == self._not_ready_marker.to_bytes(1, 'big'):
                self.logger.error("Bob not ready to receive qubits, aborting")
                results.errors.append("Bob not ready to receive qubits, aborting")
                self._running = False
                return
            if ack != self._acknowledge_marker.to_bytes(1, 'big'):
                self.logger.error(f"Did not receive proper ACK from Bob, aborting. Received invalid ACK byte: {ack}")
                results.errors.append(f"Did not receive proper ACK from Bob, aborting. Received invalid ACK byte: {ack}")
                self._running = False
                return
            self.logger.info("Received ACK from Bob, starting qubit transmission")
            
            start_time = time.time()
            # Calculate laser fire time: 90% through each period for consistent timing
            laser_fire_fraction = 0.8  # Fire laser at 90% of period

            # TODO: Confirm laser fire timing is correct at 90% of the period intervalled
            for pulse_id in range(self.num_qubits):
                if not self._running:
                    break
                
                pulse_start_time = time.time()
                
                # Calculate when to fire laser for this pulse (consistent timing)
                target_laser_time = start_time + (pulse_id + laser_fire_fraction) * self.config.pulse_period_seconds

                # Get bases and bits
                basis, bit = self._get_basis_and_bit(pulse_id)

                # Send qubit to Bob
                self.logger.debug(f"Sending qubit {pulse_id}: basis={basis}, bit={bit}")
                
                # Set polarization
                print(f"🔸 Pulse {pulse_id}: Setting polarization Basis={basis.name}, Bit={bit.value}")
                rotation_start = time.time()
                pol_output = self.polarization_controller.set_polarization_manually(basis, bit)
                print(f"   ➡️  Polarization set to {pol_output.angle_degrees}°")
                
                # Wait for polarization readiness
                print(f"🔸 Pulse {pulse_id}: Waiting for polarization readiness...")
                wait_start = time.time()
                if not self.polarization_controller.wait_for_availability(timeout=TIMEOUT_WAIT_FOR_ROTATION):
                    error_msg = f"Timeout waiting for polarization readiness for pulse {pulse_id}"
                    self.logger.error(error_msg)
                    results.errors.append(error_msg)
                    continue
                wait_time = time.time() - wait_start
                rotation_time = time.time() - rotation_start
                print(f"   ➡️  Polarization ready after {wait_time:.3f}s")
                print(f"       (Rotation time: {rotation_time:.3f}s)")
                
                # Record results
                results.bits.append(int(bit))
                results.bases.append(basis.int)
                results.polarization_angles.append(pol_output.angle_degrees)
                results.rotation_times.append(rotation_time)
                results.wait_times.append(wait_time)

                # Wait until the scheduled laser fire time for consistent timing
                current_time = time.time()
                time_until_laser = target_laser_time - current_time
                
                if time_until_laser > 0:
                    print(f"🔸 Pulse {pulse_id}: Waiting {time_until_laser:.3f}s to fire laser at scheduled time")
                    time.sleep(time_until_laser)
                elif time_until_laser < -0.001:  # More than 1ms late
                    self.logger.warning(f" ⚠️ Pulse {pulse_id} is {-time_until_laser:.3f}s late! Polarization took too long.")
                
                # Fire laser at the scheduled time
                print(f"🔸 Pulse {pulse_id}: Firing laser at scheduled time")
                laser_send_time = time.time()
                if not self.laser_controller.trigger_once():
                    error_msg = f"Failed to fire laser for pulse {pulse_id}"
                    self.logger.error(error_msg)
                    results.errors.append(error_msg)
                    continue
                laser_elapsed_time = time.time() - laser_send_time
                
                # Calculate timing accuracy
                timing_error = laser_send_time - target_laser_time
                print(f"   ➡️  Laser fired in {laser_elapsed_time:.3f}s (timing error: {timing_error*1000:.1f}ms)")
                
                # Record more time results
                results.pulse_times.append(laser_send_time) # real time stamp is right before sending the laser command
                results.laser_elapsed.append(laser_elapsed_time)

                # Wait for next pulse period (until next pulse should start)
                next_pulse_start = start_time + (pulse_id + 1) * self.config.pulse_period_seconds
                current_time = time.time()
                remaining_time = next_pulse_start - current_time
                
                if remaining_time > 0:
                    print(f"--------------------------> DEBUG: Waiting {remaining_time:.3f}s for next pulse period, Fired at {laser_send_time - start_time:.3f}, Target was {target_laser_time- start_time:.3f}")
                    self.logger.debug(f"Waiting {remaining_time:.3f}s for next pulse period, Fired at {laser_send_time - start_time:.3f}, Target was {target_laser_time- start_time:.3f}")
                    time.sleep(remaining_time)
                elif remaining_time < -0.05:  # More than 50ms late
                    self.logger.warning(f" Pulse {pulse_id} is {-remaining_time:.3f}s late for just a little late.")
                else:
                    total_pulse_time = current_time - pulse_start_time
                    self.logger.warning(f" ⚠️ Pulse {pulse_id} exceeded period: {total_pulse_time:.3f}s > {self.config.pulse_period_seconds}s")
                
                print(f"   ✅ Pulse {pulse_id} completed\n")

            # Send end of qubit transmission marker
            connection.sendall(self._handshake_end_marker.to_bytes(1, 'big'))
            self.logger.debug("Sent end of qubit transmission marker to Bob")
            # Wait for final ACK from Bob
            final_ack = connection.recv(1)
            if final_ack != self._acknowledge_marker.to_bytes(1, 'big'):
                self.logger.error(f"Did not receive final ACK from Bob, received: {final_ack}")
                results.errors.append(f"Did not receive final ACK from Bob, received: {final_ack}")
            else:
                self.logger.info("Received final ACK from Bob, qubit transmission complete")



        except Exception as e:
            error_msg = f"Error in hardware test thread: {e}"
            self.logger.error(error_msg)
            results.errors.append(error_msg)
        except KeyboardInterrupt:
            self.logger.info("Hardware test interrupted by user")
            self._running = False
        finally:
            self._running = False


        # End of pulse generation thread
        end_time = time.time()
        results.total_runtime = end_time - start_time
        self.logger.info(f"All qubits generated and sent successfully in {results.total_runtime:.3f}s")
        return results

    def _get_basis_and_bit(self, pulse_id: int) -> Tuple[Basis, Bit]:
        """Get basis and bit for the given pulse."""
        if self.config.mode == AliceTestMode.PREDETERMINED or self.config.mode == AliceTestMode.RANDOM_BATCH:
            # Use predetermined values
            basis_val = self.config.predetermined_bases[pulse_id]
            bit_val = self.config.predetermined_bits[pulse_id]
        elif self.config.mode == AliceTestMode.SEEDED or self.config.mode == AliceTestMode.RANDOM_STREAM:
            # Generate random values using QRNG
            basis_val = int(self.qrng.get_random_bit())
            bit_val = int(self.qrng.get_random_bit())
        else:
            raise ValueError("Invalid test mode")
        
        basis = Basis.Z if basis_val == 0 else Basis.X
        bit = Bit(bit_val)
        
        return basis, bit
    
    @staticmethod
    def _bits_bases_to_byte(bit: int, base: int) -> int:
        """Convert bit and base to protocol byte format."""
        # Encode bit and base into a single byte
        # Bit 0: bit value, Bit 1: base value, rest unused
        return (base << 1) | bit

    def _shutdown_hardware(self) -> None:
        """Shutdown all hardware components."""
        self.logger.info("Shutting down hardware components...")
        self._running = False
        
        try:
            self.laser_controller.shutdown()
            self.polarization_controller.shutdown()
        except Exception as e:
            self.logger.error(f"Error shutting down hardware: {e}")

def join_keys(key_list: queue.Queue):
    """Join Keys from Threads given the QUEUE of Key Objects."""
    joined_key = []
    
    for i in range(key_list.qsize()):
        aux_key: Key = key_list.get()
        aux_key_list = aux_key.generate_array()
        joined_key.extend(aux_key_list)
    
    joined_key_obj = Key.create_key_from_list(joined_key)
    return joined_key_obj


def main():

    # Default global values for parameters
    NUM_THREADS = 1
    KEY_LENGTH = 5
    LOSS_RATE = 0.0
    PULSE_PERIOD = 1.0  # seconds
    TEST_FRACTION = 0.1
    USE_HARDWARE = True
    COM_PORT = "COM4"
    LASER_CHANNEL = 8
    USE_MOCK_RECEIVER = True

    # Network configuration
    IP_ADDRESS_ALICE = "localhost"
    IP_ADDRESS_BOB = "localhost" 
    # IP_ADDRESS_ALICE = "127.0.0.1"
    # IP_ADDRESS_BOB = "127.0.0.2"
    PORT_NUMBER_ALICE = 65432
    PORT_NUMBER_BOB = 65433
    PORT_NUMBER_QUANTUM_CHANNEL = 12345
    SHARED_SECRET_KEY = "IzetXlgAnY4oye56"  # 16 bytes for AES-128


    """Main function to run Alice's hardware BB84 protocol."""
    parser = argparse.ArgumentParser(description="Alice's BB84 protocol with hardware control")
    parser.add_argument("-nth", "--num_threads", type=int, default=NUM_THREADS, help="Number of threads")
    parser.add_argument("-k", "--key_length", type=int, default=KEY_LENGTH, help="Length of the key")
    parser.add_argument("-lr", "--loss_rate", type=float, default=LOSS_RATE, help="Loss rate for qubits")
    parser.add_argument("-pp", "--pulse_period", type=float, default=PULSE_PERIOD, help="Pulse period in seconds")
    parser.add_argument("-tf", "--test_fraction", type=float, default=TEST_FRACTION, help="Fraction for testing")
    parser.add_argument("--use_hardware", action="store_true", default=USE_HARDWARE, help="Use actual hardware")
    parser.add_argument("--com_port", type=str, default=COM_PORT, help="COM port for polarization")
    parser.add_argument("--laser_channel", type=int, default=LASER_CHANNEL, help="Laser channel")
    parser.add_argument("--use_mock_receiver", action="store_true", default=USE_MOCK_RECEIVER, help="Use mock Bob receiver for testing")
    parser.add_argument("--ip_address_alice", type=str, default=IP_ADDRESS_ALICE, help="IP address for Alice")
    parser.add_argument("--ip_address_bob", type=str, default=IP_ADDRESS_BOB, help="IP address for Bob")
    parser.add_argument("--port_number_alice", type=int, default=PORT_NUMBER_ALICE, help="Port number for Alice")
    parser.add_argument("--port_number_bob", type=int, default=PORT_NUMBER_BOB, help="Port number for Bob")
    parser.add_argument("--port_number_quantum_channel", type=int, default=PORT_NUMBER_QUANTUM_CHANNEL, help="Port number for quantum channel")
    parser.add_argument("--shared_secret_key", type=str, default=SHARED_SECRET_KEY, help="Shared secret key for MAC")



    
    args = parser.parse_args()

    # BB84 protocol parameters
    key_length = args.key_length
    nb_threads = args.num_threads
    loss_rate = args.loss_rate
    pulse_period = args.pulse_period
    test_fraction = args.test_fraction
    use_hardware = args.use_hardware
    com_port = args.com_port
    laser_channel = args.laser_channel
    use_mock_receiver = args.use_mock_receiver
    if use_hardware and (com_port is None or laser_channel is None):
        parser.error("When using hardware, both --com_port and --laser_channel must be specified")
        
    # Network configuration
    ip_address_alice = args.ip_address_alice
    ip_address_bob = args.ip_address_bob
    # ip_address_alice = "127.0.0.1"
    # ip_address_bob = "127.0.0.2"
    port_number_alice = args.port_number_alice
    port_number_bob = args.port_number_bob
    port_number_quantum_channel = args.port_number_quantum_channel
    shared_secret_key = bytes(args.shared_secret_key, 'utf-8')

    # Privacy Amplification parameters
    pa_compression_rate = 0.5
    key_queue = queue.Queue()
    
    # Classical communication setup
    mac_configuration = MAC_Config(MAC_Algorithms.CMAC, shared_secret_key)
    alice_info = ConnectionInfo(ip_address_alice, port_number_alice)
    role_alice = Role.get_instance(alice_info)
    bob_info = ConnectionInfo(ip_address_bob, port_number_bob)

    logger.info(f"Alice {alice_info.ip}:{alice_info.port} connecting to Bob {bob_info.ip}:{bob_info.port}")
    
    try:
        # Setup quantum channel server
        logger.info("Setting up quantum channel server...")
        server = AliceHardwareQubits.setup_server(ip_address_alice, port_number_quantum_channel)
        logger.info(f"Quantum channel server listening on {ip_address_alice}:{port_number_quantum_channel}")
        logger.info("Waiting for Bob to connect... (Use --help to see mock receiver option)")
 
        # Configure hardware Alice for this thread
        hardware_config = HardwareAliceConfig(
            num_qubits=key_length,
            pulse_period_seconds=pulse_period,
            use_hardware=use_hardware,
            com_port=com_port,
            laser_channel=laser_channel,
            test_fraction=test_fraction,
            loss_rate=loss_rate,
            use_mock_receiver=use_mock_receiver,
            server_host=ip_address_alice,
            server_port=port_number_quantum_channel,
        )

        # Create instance hardware/quantum part of Alice
        # TODO: I thinkg the mock receiver will not work since it will get stuck in the setup_server ???
        alice_hardware = AliceHardwareQubits(hardware_config)
        if hardware_config.use_mock_receiver:
            logger.info("Launching mock receiver (echo) client for local testing...")
            alice_hardware.setup_mock_receiver()
        
        connection, address = server.accept()
        logger.info(f"Quantum channel connected to {address}")
        
        with connection:

            # Create instance of the quantum channel participant Alice for post-processing
            classical_channel_participant_alice_obj = QKDAliceImplementation(ip_address_alice, port_number_alice, ip_address_bob,
                                                                         port_number_bob, shared_secret_key)
            # Setup role Alice
            classical_channel_participant_alice_obj.setup_role_alice()
            classical_channel_participant_alice_obj.start_read_communication_channel()

            start_execution_time_tick = time.perf_counter()

            # Function to run hardware-based BB84 protocol for each thread
            # NOTE: Multiple threads share the same quantum connection, so they run sequentially
            # The quantum part only advances when done and the classical part creates its own threads
            def run_all_hardware_pp_thread(start_event: threading.Event, thread_id: int):
                logger.info(f"Alice Thread {thread_id} - starting hardware BB84")
                
                try:
                    # Run hardware qubit generation and transmission
                    results_ith = alice_hardware.run_trl4_protocol_quantum_part(connection)
                    logger.info(f"Alice Thread {thread_id} - hardware qubits sent")
                    
                    start_event.set()  # Signal next thread can start

                    # Now run classical communication protocol for post processing
                    #TODO: Fonte doesn't save each key thread in a list to later access the individual keys if needed to debug
                    logger.info(f"Alice Thread {thread_id} - starting classical post-processing thread")
                    classical_channel_participant_alice_obj.alice_run_qkd_classical_process_threading(results_ith.bits,
                                                                                              results_ith.bases,
                                                                                              do_test=True, test_fraction=test_fraction,
                                                                                              error_threshold=0.1,
                                                                                              privacy_amplification_compression_rate=pa_compression_rate)
                    
                    # logger.info(f"Alice Thread {thread_id} - ER complete: key length {classical_channel_participant_alice_obj.correct_key.get_size()}")
                    # key_queue.put(correct_key)
                    
                except Exception as e:
                    logger.error(f"Alice Thread {thread_id} - Error: {e}")
                    raise
            
            # Start threads
            alice_threads = []
            start_event = threading.Event()
            start_event.set()
            
            for i in range(nb_threads):
                alice_thread = threading.Thread(
                    target=run_all_hardware_pp_thread, 
                    args=[start_event, i]
                )
                alice_threads.append(alice_thread)
                alice_thread.start()
                
                if i < nb_threads - 1:
                    start_event.clear()
                    start_event.wait()
            
            # Wait for all threads to complete
            for thread in alice_threads:
                thread.join()
            
            classical_channel_participant_alice_obj.alice_join_threads()
            classical_channel_participant_alice_obj._role_alice._stop_all_threads()
            end_execution_time_tick = time.perf_counter()
            print(f"Alice Total Execution Threading Time: {end_execution_time_tick - start_execution_time_tick}")

            classical_channel_participant_alice_obj.alice_produce_statistical_data()
            
            
            # Show results
            logger.info("All Alice threads completed")
            
            final_results = []
            final_failed_percentage = []
            logger.info(f"Final failed percentages: {final_failed_percentage}")

            
            # # Privacy Amplification - ALREADY DONE INSIDE EACH THREAD?
            # if PA_COMPRESSION_RATE != 0.0:
            #     logger.info("Performing Privacy Amplification")
                
            #     joined_key_final = join_keys(KEY_QUEUE)
            #     joined_key_final_list = joined_key_final.generate_array()
                
            #     logger.info(f"Final Joined Key Size: {joined_key_final.get_size()}")
                
            #     privacy_amplification = PrivacyAmplification(joined_key_final_list)
            #     initial_key_length = joined_key_final.get_size()
            #     final_key_length = int(initial_key_length * PA_COMPRESSION_RATE)
                
            #     if PA_COMPRESSION_RATE > 0.0:
            #         logger.info("Performing Toeplitz PA")
            #         _, _, secured_key = privacy_amplification.do_privacy_amplification(
            #             initial_key_length, final_key_length
            #         )
            #     else:
            #         logger.info("Performing XOR PA")
            #         secured_key = privacy_amplification.xor_privacy_amplification()
                
            #     secured_key_vis = Key.create_key_from_list(secured_key)
            #     logger.info(f"SECURED FINAL KEY LENGTH: {secured_key_vis.get_size()}")
    
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        logger.info("Closing quantum channel...")
        if 'connection' in locals():
            connection.close()
        if 'server' in locals():
            server.close()
        role_alice.clean()


if __name__ == "__main__":
    main()


