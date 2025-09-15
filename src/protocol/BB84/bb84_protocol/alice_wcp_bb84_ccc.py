import random
import socket
import time
import threading
import queue
import pickle
import numpy as np
from typing import List, Tuple, Dict, Any

from BB84.bb84_protocol.wcp_pulse import WCPPulse, PulseType, WCPIntensityManager
from BB84.bb84_protocol.wcp_parameter_estimation import WCPParameterEstimator
from classical_communication_channel.communication_channel.role import Role
from interface_stepperMotor.imports.stm32_interface import STM32Interface

# Configure logging
import logging
from examples.logging_setup import setup_logger
logger = setup_logger("WCP BB84 Alice Log", logging.INFO)

def read(role: Role, queues: list[queue.Queue]):
    """Read data from classical communication channel"""
    try:
        while True:
            payload = role.get_from_inbox()
            (thread_id, data) = pickle.loads(payload)
            queues[thread_id].put(data)
    except KeyboardInterrupt:
        role.clean()

class AliceWCPQubits:
    """
    Alice-specific methods for the WCP BB84 protocol with decoy states.
    """
    
    def __init__(self, num_qubits: int = 100, qubit_delay_us: float = 1000, 
                 num_frames: int = 10, bytes_per_frame: int = 10,
                 sync_frames: int = 100, sync_bytes_per_frame: int = 50,
                 mu_signal: float = 0.5, mu_decoy: float = 0.1, mu_vacuum: float = 0.0,
                 prob_signal: float = 0.7, prob_decoy: float = 0.25, prob_vacuum: float = 0.05,
                 transmission_efficiency: float = 0.1, 
                 detection_efficiency: float = 0.1, dark_count_rate: float = 1e-6, # ???                 
                 com_interface: STM32Interface = None):
        """
        Initialize Alice for WCP BB84 protocol.
        
        :param num_qubits: Number of qubits to generate
        :param qubit_delay_us: Delay between qubit transmissions (microseconds)
        :param num_frames: Number of frames to send
        :param bytes_per_frame: Number of bytes per frame
        :param sync_frames: Number of synchronization frames
        :param sync_bytes_per_frame: Bytes per synchronization frame
        :param mu_signal: Mean photon number for signal pulses
        :param mu_decoy: Mean photon number for decoy pulses
        :param mu_vacuum: Mean photon number for vacuum pulses (usually 0)
        :param prob_signal: Probability of sending signal pulse
        :param prob_decoy: Probability of sending decoy pulse
        :param prob_vacuum: Probability of sending vacuum pulse
        :param transmission_efficiency: Channel transmission efficiency # LOSS RATE???
        :param com_interface: Communication interface for sending data (optional)
        """
        self.num_qubits = num_qubits
        self.qubit_delay_us = qubit_delay_us
        self.num_frames = num_frames
        self.bytes_per_frame = bytes_per_frame
        self.sync_frames = sync_frames
        self.sync_bytes_per_frame = sync_bytes_per_frame

        # WCP specific parameters
        self.mu_signal = mu_signal
        self.mu_decoy = mu_decoy
        self.mu_vacuum = mu_vacuum
        self.prob_signal = prob_signal
        self.prob_decoy = prob_decoy
        self.prob_vacuum = prob_vacuum
        # If probabilities do not sum to 1, they will be normalized in the WCPIntensityManager
        if sum([prob_signal, prob_decoy, prob_vacuum]) != 1.0:
            logger.warning("Probabilities do not sum to 1. Will be normalized in WCPIntensityManager.")
        
        # Transmission efficiency (loss rate???)
        self.transmission_efficiency = transmission_efficiency

        # Detection efficiency WHY???
        self.detection_efficiency = detection_efficiency

        # Dark count rate WHY???
        self.dark_count_rate = dark_count_rate

        # Communication interface for sending data to stepper motor or other devices
        # Initialize STM32 interface
        if com_interface is not None:
            logger.info(f"Connecting to STM32 on port {com_interface.serial_port.portstr}...")
            self.stm = com_interface
            self.stm_connected = True
        elif com_interface is None:
            logger.info("No COM port provided. STM32 interface will not be initialized.")
            self.stm_connected = False
        elif not isinstance(com_interface, STM32Interface):
            logger.error("Provided communication interface is not an instance of STM32Interface.")
            raise TypeError("com_interface must be an instance of STM32Interface")
        self.stm_available = False
        self.polarization_ready = False


        # WCP intensity manager for pulse creation
        self.intensity_manager = WCPIntensityManager(
            signal_intensity=mu_signal,
            decoy_intensity=mu_decoy,
            vacuum_intensity=mu_vacuum,
            signal_prob=prob_signal,
            decoy_prob=prob_decoy,
            vacuum_prob=prob_vacuum
        )

        # Data storage
        self.bits = []
        self.bases = []
        self.wcp_pulses = []
        self.pulse_bytes = []
        self.pulse_types = []
        self.pulse_intensities = []
        
        # Timing and markers
        self._start_marker = 100
        self._end_marker = 50
        
    @staticmethod
    def generate_random_bases_bits(number: int) -> List[int]:
        """Generate random bits or bases"""
        return [random.randint(0, 1) for _ in range(number)]
    
    @staticmethod
    def time_sleep(time_sleep: int):
        """Sleep for specified microseconds"""
        time.sleep(time_sleep / 1_000_000)
    
    @staticmethod
    def setup_server(host: str, port: int, timeout: float = 10.0) -> socket.socket:
        """Set up server socket for quantum channel"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(timeout)
        server.bind((host, port))
        server.listen(1)
        logger.info(f"Server listening on {host}:{port}")
        return server
    
    def create_wcp_pulses(self):
        """Create WCP pulses with random bits, bases, and intensities"""
        logger.info("Creating WCP pulses with decoy states...")
        
        # Check if self.num_frames * self.bytes_per_frame is equal to self.num_qubits
        if self.num_frames * self.bytes_per_frame != self.num_qubits:
            logger.warning(f"Number of qubits ({self.num_qubits}) does not match total pulses ({self.num_frames * self.bytes_per_frame}). Adjusting num_qubits to match total pulses.")
            self.bytes_per_frame = self.num_qubits // self.num_frames

        total_pulses = self.num_qubits
        self.bits = self.generate_random_bases_bits(total_pulses)
        self.bases = self.generate_random_bases_bits(total_pulses)
        
        # Create WCP pulses with intensity selection
        self.wcp_pulses = []
        self.pulse_types = []
        self.pulse_types_bytes = []
        self.pulse_intensities = []
        
        for bit, base in zip(self.bits, self.bases):
            wcp_pulse = self.intensity_manager.create_pulse(bit, base)
            self.wcp_pulses.append(wcp_pulse)
            self.pulse_types.append(wcp_pulse.pulse_type)
            self.pulse_intensities.append(wcp_pulse.intensity)
        
        # Convert to bytes for transmission
        self.pulse_bytes = [pulse.get_byte() for pulse in self.wcp_pulses]
        self.pulse_types_bytes = [pulse.get_type_byte() for pulse in self.wcp_pulses]

        # Log pulse creation
        logger.debug(f"Created {len(self.wcp_pulses)} WCP pulses with intensities:\n\
                      {self.pulse_bytes} for bytes, {self.pulse_types_bytes} for types, ")
        
        # Log pulse distribution
        signal_count = self.pulse_types.count(PulseType.SIGNAL)
        decoy_count = self.pulse_types.count(PulseType.DECOY)
        vacuum_count = self.pulse_types.count(PulseType.VACUUM)

        logger.info(f"Created {total_pulses} pulses: {signal_count} signal, {decoy_count} decoy, {vacuum_count} vacuum")
    
    def perform_sync(self, connection: socket.socket):
        """Perform synchronization with Bob"""
        logger.info("Performing synchronization...")
        try:
            for frame in range(self.sync_frames):
                sync_data = [self._start_marker] * self.sync_bytes_per_frame + [self._end_marker]
                frame_bytes = bytes(sync_data)
                connection.send(frame_bytes)
                self.time_sleep(self.qubit_delay_us)
                
            logger.info("Synchronization completed")
        except Exception as e:
            logger.error(f"Synchronization failed: {e}")
            raise
    def perform_sync2(self, connection: socket.socket):
        """Perform clock synchronization with Bob."""
        logger.info("Sending synchronization...")
        try:
            for frame in range(self.sync_frames):
                start_time = time.time()
                
                # Send start marker
                connection.sendall(self._start_marker.to_bytes(1, 'big'))
                self.time_sleep(self.qubit_delay_us)

                # Send sync bytes
                for byte_idx in range(self.sync_bytes_per_frame):
                    sync_byte = random.randint(0, 255)
                    connection.sendall(sync_byte.to_bytes(1, 'big'))
                    self.time_sleep(self.qubit_delay_us)
                
                # Send end marker
                connection.sendall(self._end_marker.to_bytes(1, 'big'))
                self.time_sleep(self.qubit_delay_us)

                frame_time = time.time() - start_time
                logger.debug(f"Sync frame {frame} sent in {frame_time:.4f}s")
                
        except Exception as e:
            logger.error(f"Error during synchronization: {e}")


    def send_wcp_pulses(self, connection: socket.socket):
        """Send WCP pulses to Bob"""
        logger.info("Sending WCP pulses...")
        
        try:
            pulse_idx = 0
            for frame in range(self.num_frames):
                frame_data = [self._start_marker]
                
                # Add pulses for this frame
                for _ in range(self.bytes_per_frame):
                    if pulse_idx < len(self.pulse_bytes):
                        frame_data.append(self.pulse_bytes[pulse_idx])
                        pulse_idx += 1
                
                frame_data.append(self._end_marker)
                frame_bytes = bytes(frame_data)
                
                # Apply channel effects to pulses
                for i in range(1, len(frame_data) - 1):  # Skip markers
                    pulse_idx_in_frame = i - 1 + frame * self.bytes_per_frame
                    if pulse_idx_in_frame < len(self.wcp_pulses):
                        self.wcp_pulses[pulse_idx_in_frame].apply_channel_loss(self.transmission_efficiency)
                
                connection.send(frame_bytes)
                self.time_sleep(self.qubit_delay_us)
                
            logger.info(f"Sent {pulse_idx} WCP pulses")
            
        except Exception as e:
            logger.error(f"Failed to send WCP pulses: {e}")
            raise
    def send_wcp_pulses2(self, connection: socket.socket):
        """Send WCP pulses to Bob"""
        logger.info("Sending WCP pulses...")
        
        try:
            for frame in range(self.num_frames):
                start_time = time.time()
                
                # Send start marker
                connection.sendall(self._start_marker.to_bytes(1, 'big'))
                self.time_sleep(self.qubit_delay_us)
                
                # Send pulses for this frame
                for byte_idx in range(self.bytes_per_frame):
                    pulse_idx = frame * self.bytes_per_frame + byte_idx
                # for i in range(frame * self.bytes_per_frame, (frame + 1) * self.bytes_per_frame):
                #     if i >= len(self.pulse_bytes):
                #         break

                    # Send the pulse byte
                    pulse_byte = self.pulse_bytes[pulse_idx]
                    connection.sendall(pulse_byte.to_bytes(1, 'big'))
                    self.time_sleep(self.qubit_delay_us)
                
                # Send end marker
                connection.sendall(self._end_marker.to_bytes(1, 'big'))
                self.time_sleep(self.qubit_delay_us)
                
                frame_time = time.time() - start_time
                logger.debug(f"WCP frame {frame} sent in {frame_time:.4f}s")
                
        except Exception as e:
            logger.error(f"Error sending WCP pulses: {e}")

    def run_alice_wcp_protocol(self, connection: socket.socket):
        """Run Alice's complete WCP protocol"""
        logger.info("Starting Alice WCP BB84 protocol...")
        
        try:
            self.perform_sync(connection)
            self.create_wcp_pulses()
            self.send_wcp_pulses(connection)
            logger.info("Alice WCP protocol completed successfully")
            
        except Exception as e:
            logger.error(f"Alice WCP protocol failed: {e}")
            raise
    
    def send_qubits(self, connection: socket.socket):
        """Send qubits to Bob."""
        logger.info(f"Sending qubits...")
        self.time_sleep(time_sleep=100000)  # Initial delay
        
        try:
            for frame in range(self.num_frames):
                start_time = time.time_ns()
                connection.sendall(bytes([self._start_marker]))
                
                for byte_idx in range(self.bytes_per_frame):
                    idx = frame * self.bytes_per_frame + byte_idx
                    if idx < len(self.pulse_bytes):
                        # Send actual qubit
                        connection.sendall(bytes([self.pulse_bytes[idx]]))
                    else:
                        # Send padding if needed
                        connection.sendall(bytes([0]))
                    
                    self.time_sleep(self.qubit_delay_us)
                
                connection.sendall(bytes([self._end_marker]))
                end_time = time.time_ns()
                logger.debug(f"Frame {frame} sent in {(end_time - start_time) / 1000} us")
                
        except Exception as e:
            logger.error(f"Error sending qubits: {e}")
            
        logger.info("Alice: All qubits sent.")

    def run_mock_alice_wcp_protocol(self, connection: socket.socket, wait_time: bool = True):
        """Run a mock Alice WCP protocol for testing purposes."""
        logger.info("Starting mock Alice WCP BB84 protocol...")
        
        try:
            self.create_wcp_pulses()
            self.send_wcp_pulses_mock(connection, wait_time)
            
        except Exception as e:
            logger.error(f"Mock Alice WCP protocol failed: {e}")
            raise

    def send_wcp_pulses_mock(self, connection: socket.socket, wait_time: bool = True):
        """Send WCP pulses in a mock manner for testing."""
        logger.info("Sending mock WCP pulses...")
        
        try:
            for pulse in self.pulse_bytes:
                # Simulate sending pulse
                connection.sendall(pulse)
                if wait_time:
                    self.time_sleep(self.qubit_delay_us)

            logger.info(f"Sent {len(self.pulse_bytes)} mock WCP pulses")

        except Exception as e:
            logger.error(f"Failed to send mock WCP pulses: {e}")
            raise
    def send_wcp_pulses_mock2(self, connection: socket.socket, wait_time: bool = True):
        """Send WCP pulses in a mock manner for testing purposes."""
        logger.info("Sending mock WCP pulses (version 2)...")

        try:
            payload = pickle.dumps([self.pulse_bytes, self.qubit_delay_us])
            framed_payload = (self._start_marker.to_bytes(1, 'big') +
                                payload +
                                self._end_marker.to_bytes(1, 'big'))
            connection.sendall(framed_payload)

            # wait for acknowledgment
            if wait_time:
                self.time_sleep(self.qubit_delay_us * len(self.pulse_bytes))
            logger.info(f"Sent {len(self.pulse_bytes)} mock WCP pulses (version 2)")
        except Exception as e:
            logger.error(f"Failed to send mock WCP pulses (version 2): {e}")
            raise



    def run_mock_alice_qubits(self, connection: socket.socket, wait_time: bool = True, send_lock: threading.Lock = None):
        """Run Alice's qubit operations for testing. Sends all qubits at once and sleeps for the corresponded time."""
        logger.info("Running Mock Alice's qubit operations for testing...")
        self.send_lock = send_lock

        try:
            self.create_wcp_pulses()
            self.send_qubits_mock(connection, wait_time=wait_time)

            # data = connection.recv(1)  # Wait for end signal
            # if data == b'\x01':
            #     logger.info("Received end signal from Bob, mock qubit protocol completed successfully.")
            # else:
            #     logger.warning(f"Expected acknowledgment but received {data}")
        except Exception as e:
            logger.error(f"Mock Alice qubit protocol failed: {e}")
            raise

    def send_qubits_mock(self, connection: socket.socket, wait_time: bool = True):
        """Send qubits to Bob similar to sync and saves the idx of each."""
        logger.info(f"Sending qubits...")
        try:
            payload = pickle.dumps([self.pulse_bytes, self.qubit_delay_us])
            framed = (
                self._start_marker.to_bytes(1, 'big') +
                payload +
                self._end_marker.to_bytes(1, 'big')
            )
            connection.sendall(framed)

            # Wait for the corresponded time
            if wait_time:
                self.time_sleep(time_sleep=int(self.qubit_delay_us * self.num_qubits))
                pass
            
        except Exception as e:
            logger.error(f"Error in sending qubits: {e}")
            
        logger.info(f"Alice: All {len(self.pulse_bytes)} qubits sent.")



    ### USING QUEUE-BASED COMMUNICATION FOR SINGLE FILE PROGRAM

    def run_mock_alice_wcp_protocol_queue(self, connection: socket.socket, shared_queue: queue.Queue, wait_time: bool = True):
        """Run a mock Alice WCP protocol using a shared queue for testing."""
        logger.info("Starting mock Alice WCP BB84 protocol with queue...")
        
        try:
            self.create_wcp_pulses()
            self.send_wcp_pulses_mock_queue(shared_queue, wait_time=wait_time)

            # Wait message from socket of ending - expected b'\x01'
            data = connection.recv(1)  # Wait for end signal
            if data == b'\x01':
                logger.info("Received end signal from Bob, mock protocol completed successfully.")
            else:
                logger.warning(f"Expected acknowledgment but received {data}")

        except Exception as e:
            logger.error(f"Mock Alice WCP protocol with queue failed: {e}")
            raise

    def send_wcp_pulses_mock_queue(self, shared_queue: queue.Queue, wait_time: bool = True):
        """Send WCP pulses in a mock manner using a shared queue for testing."""
        logger.info("Sending mock WCP pulses using shared queue...")
        
        try:
            # for i, pulse in enumerate(self.pulse_bytes):
            #     # Simulate sending pulse by putting it in the queue
            #     shared_queue.put((i, pulse))
            #     if wait_time:
            #         self.time_sleep(self.qubit_delay_us)
            # Send all pulses at once
            shared_queue.put([self.pulse_bytes, self.qubit_delay_us])
            if wait_time:
                self.time_sleep(self.qubit_delay_us * len(self.pulse_bytes))

            logger.info(f"Sent {len(self.pulse_bytes)} mock WCP pulses to queue")
            logger.debug(f"Pulse types: {self.pulse_types}, Intensities: {self.pulse_intensities}")

        except Exception as e:
            logger.error(f"Failed to send mock WCP pulses to queue: {e}")

    def get_pulse_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated pulses"""
        total_pulses = len(self.pulse_types)
        
        if total_pulses == 0:
            return {}
        
        signal_count = sum(1 for p in self.pulse_types if p == PulseType.SIGNAL)
        decoy_count = sum(1 for p in self.pulse_types if p == PulseType.DECOY)
        vacuum_count = sum(1 for p in self.pulse_types if p == PulseType.VACUUM)
        
        avg_mu_signal = np.mean([p.intensity for p in self.wcp_pulses if p.pulse_type == PulseType.SIGNAL]) if signal_count > 0 else 0
        avg_mu_decoy = np.mean([p.intensity for p in self.wcp_pulses if p.pulse_type == PulseType.DECOY]) if decoy_count > 0 else 0
        
        return {
            'total_pulses': total_pulses,
            'signal_count': signal_count,
            'decoy_count': decoy_count,
            'vacuum_count': vacuum_count,
            'signal_fraction': signal_count / total_pulses,
            'decoy_fraction': decoy_count / total_pulses,
            'vacuum_fraction': vacuum_count / total_pulses,
            'avg_mu_signal': avg_mu_signal,
            'avg_mu_decoy': avg_mu_decoy
        }
    
    def reset(self):
        """Reset all pulse data"""
        self.bits = []
        self.bases = []
        self.wcp_pulses = []
        self.pulse_bytes = []
        self.pulse_types = []
        self.pulse_intensities = []

    def get_alice_reseted_wcp(self) -> 'AliceWCPQubits':
        """Return a reset Alice WCP instance"""
        self.reset()
        return AliceWCPQubits(
            num_qubits=self.num_qubits,
            qubit_delay_us=self.qubit_delay_us,
            num_frames=self.num_frames,
            bytes_per_frame=self.bytes_per_frame,
            sync_frames=self.sync_frames,
            sync_bytes_per_frame=self.sync_bytes_per_frame,
            mu_signal=self.mu_signal,
            mu_decoy=self.mu_decoy,
            mu_vacuum=self.mu_vacuum,
            prob_signal=self.prob_signal,
            prob_decoy=self.prob_decoy,
            prob_vacuum=self.prob_vacuum,
            transmission_efficiency=self.transmission_efficiency
        )
        # return self # why not return self?

    ####################
    # STM32 Functions - TODO: Implement these methods if needed
    ####################
        
    # def _handle_stm_connected(self):
    #     logger.info("STM32 connected")
    #     self.stm_available = True

    # def _handle_polarization_status(self, status):
    #     logger.info(f"Polarization status: {status}")
    #     self.polarization_ready = status

    # def _handle_stm_available(self):
    #     logger.info("STM32 available")
    #     self.stm_available = True

    # def cleanup(self):
    #     """Cleanup resources"""
    #     if self.stm:
    #         self.stm.cleanup()


class AliceWCPThread:
    """
    Alice thread for WCP BB84 classical communication processing.
    """
    
    def __init__(self, role: Role, receive_queue: queue.Queue, thread_id: int,
                 test_bool: bool = True, test_fraction: float = 0.1,
                 error_threshold: float = 0.1, start_idx: int = 0):
        """
        Initialize Alice WCP thread.
        
        :param role: Communication role
        :param receive_queue: Queue for receiving data
        :param thread_id: Thread identifier
        :param test_bool: Whether to perform eavesdropping test
        :param test_fraction: Fraction of bits to use for testing
        :param error_threshold: Error threshold for eavesdropping detection
        :param start_idx: Starting index for this thread's data
        """
        self.role = role
        self.receive_queue = receive_queue
        self.thread_id = thread_id
        self.test_bool = test_bool
        self.test_fraction = test_fraction
        self.error_threshold = error_threshold
        self.start_idx = start_idx
        
        # WCP-specific data
        self.bits = []
        self.bases = []
        self.pulse_types = []
        self.pulse_intensities = []
        self.bob_detected_idxs = []
        self.matched_base_indices = []
        self.bob_test_indices = []
        self.final_key = []

        # self.failed_percentage = 0.0
        self.alice_payload_size = []
        
        # Parameter estimator
        self.test_success_bool = None
        self.test_size = 0
        self.pe_qber_percentage = 0.0
        
        self.parameter_estimator = WCPParameterEstimator()
        
        logger.info(f"Alice WCP Thread {thread_id} initialized")
    
    def set_wcp_data(self, bits: List[int], bases: List[int], 
                     pulse_types: List[PulseType], pulse_intensities: List[float]):
        """Set WCP pulse data for this thread:
        :param bits: List of bits sent by Alice
        :param bases: List of bases used by Alice
        :param pulse_types: List of pulse types (signal, decoy, vacuum)
        :param pulse_intensities: List of pulse intensities for each pulse
        """
        self.bits = bits
        self.bases = bases
        self.pulse_types = pulse_types
        self.pulse_intensities = pulse_intensities

        # Log the number of WCP pulses set
        if not bits or not bases or not pulse_types or not pulse_intensities:
            logger.warning(f"Thread {self.thread_id}: No WCP data set. Bits, bases, pulse types, or intensities are empty.")
            return
        if len(bits) != len(bases) or len(bits) != len(pulse_types) or len(bits) != len(pulse_intensities):
            logger.error(f"Thread {self.thread_id}: Mismatched lengths in WCP data. Bits: {len(bits)}, Bases: {len(bases)}, Pulse Types: {len(pulse_types)}, Intensities: {len(pulse_intensities)}")
            return
        # Create WCP pulses from the provided data
        self.wcp_pulses = [WCPPulse(bit, base, pulse_type, intensity)
                           for bit, base, pulse_type, intensity in zip(bits, bases, pulse_types, pulse_intensities)]
        
        logger.info(f"Thread {self.thread_id}: Set {len(bits)} WCP pulses")

    def send_data(self, data: Any, list_size_to_append: List = None):
        """Send data through classical channel"""
        payload = pickle.dumps((self.thread_id, data))
        self.role.put_in_outbox(payload)
        
        if list_size_to_append is not None:
            list_size_to_append.append(len(data) if hasattr(data, '__len__') else 1)
        else:
            self.alice_payload_size.append(len(data) if hasattr(data, '__len__') else 1)
    
    def receive_data(self) -> Any:
        """Receive data from classical channel"""
        return self.receive_queue.get()
    
    def receive_detected_indices(self) -> List[int]:
        """Receive indices of detected pulses from Bob"""
        logger.info(f"Thread {self.thread_id}: Waiting for detected indices from Bob...")
        detected_indices = self.receive_data()
        self.bob_detected_idxs = detected_indices  # Store for later use

        # Validate indices before building the list
        for idx in detected_indices:
            if idx >= len(self.bits):
                raise IndexError(f"Index {idx} out of bounds for bits list of length {len(self.bits)}")
        self.alice_detected_bits = [self.bits[idx] for idx in detected_indices]

        logger.info(f"Thread {self.thread_id}: Received {len(detected_indices)} detected indices")

        # P.Fonte info
        # payload_received = pickle.dumps((self.thread_id, self.bob_detected_idxs))
        # self.messages_received.append(
        #     ("A", "R", self.number_messages_received, time.perf_counter(), len(payload_received), 1, self.thread_id))
        # self.number_messages_received += 1

        return self.bob_detected_idxs
    
    ### TODO: THIS AFTER RECEIVE COMMON INDICES SO THAT ONLY SEND PULSE TYPES FOR CORRECT INDICES ??? OR NOT SINCE ERRORS ALSO NEED TO BE KNOWN ###
    def send_pulse_type_information(self, detected_indices: List[int]):
        """Send pulse type information for detected pulses"""
        logger.info(f"Thread {self.thread_id}: Preparing to send pulse type information for {len(detected_indices)} detected pulses")
        pulse_type_info = []
        
        for idx in detected_indices:
            if idx < len(self.pulse_types):
                pulse_type = self.pulse_types[idx]

                # Send pulse type as integer (0=signal, 1=decoy, 2=vacuum)
                if pulse_type == PulseType.SIGNAL:
                    type_int = 0
                elif pulse_type == PulseType.DECOY:
                    type_int = 1
                else:  # VACUUM
                    type_int = 2
                
                pulse_type_info.append(type_int)

        # Create intensity map
        # Map pulse types to their corresponding intensities
        pulse_type_indices = {pt: i for i, pt in enumerate(self.pulse_types)}
        intensity_map = {
            0: self.pulse_intensities[pulse_type_indices.get(PulseType.SIGNAL, -1)] if PulseType.SIGNAL in pulse_type_indices else 0,
            1: self.pulse_intensities[pulse_type_indices.get(PulseType.DECOY, -1)] if PulseType.DECOY in pulse_type_indices else 0,
            2: self.pulse_intensities[pulse_type_indices.get(PulseType.VACUUM, -1)] if PulseType.VACUUM in pulse_type_indices else 0
        }
        ## Alternative intensity map creation - Worst Performance
        # intensity_map = {
        #     0: self.pulse_intensities[next(i for i, pt in enumerate(self.pulse_types) if pt == PulseType.SIGNAL)] if PulseType.SIGNAL in self.pulse_types else self.mu_signal,
        #     1: self.pulse_intensities[next(i for i, pt in enumerate(self.pulse_types) if pt == PulseType.DECOY)] if PulseType.DECOY in self.pulse_types else self.mu_decoy,
        #     2: self.pulse_intensities[next(i for i, pt in enumerate(self.pulse_types) if pt == PulseType.VACUUM)] if PulseType.VACUUM in self.pulse_types else self.mu_vacuum
        # }
        
        # Send pulse types and intensity map
        pulse_data = {
            'pulse_types': pulse_type_info,
            'intensity_map': intensity_map
        }
        self.detected_pulses_types_map = pulse_data  # Store for later use
        
        logger.info(f"Thread {self.thread_id}: Sending pulse type info for {len(pulse_type_info)} detected pulses with intensity map")
        self.send_data(pulse_data)
        logger.info(f"Thread {self.thread_id}: Pulse type info sent successfully")

    def send_detected_bases(self, detected_indices: List[int]):
        """Send bases for detected pulses"""
        detected_bases = [self.bases[idx] for idx in detected_indices if idx < len(self.bases)]
        logger.info(f"Thread {self.thread_id}: Sending {len(detected_bases)} detected bases")
        # detected_bases = [self.bases[idx] for idx in self.detected_indices if idx < len(self.bases)]
        self.detected_bases = detected_bases  # Store for later use
        self.detected_bits = [self.bits[idx] for idx in detected_indices if idx < len(self.bits)]
        self.send_data(detected_bases)
        logger.debug(f"Thread {self.thread_id}: Sent detected bases: {detected_bases}")
    
    def receive_matched_base_indices(self) -> List[int]:
        """Receive indices where bases match"""
        logger.info(f"Thread {self.thread_id}: Waiting for matched base indices...")
        self.matched_base_indices = self.receive_data()
        if not isinstance(self.matched_base_indices, list):
            logger.error(f"Thread {self.thread_id}: Expected list of matched base indices, got {type(self.matched_base_indices)}")
            raise TypeError("Expected list of matched base indices")
        
        self.alice_matched_base_bases = [self.detected_bases[idx] for idx in self.matched_base_indices if idx < len(self.detected_bases)]
        self.alice_matched_base_bits = [self.detected_bits[idx] for idx in self.matched_base_indices if idx < len(self.detected_bits)]
        logger.info(f"Thread {self.thread_id}: Received {len(self.matched_base_indices)} matched base indices")

        # P.Fonte info
        # payload_received = pickle.dumps((self.thread_id, self.matched_base_indices))
        # self.messages_received.append(
        #     ("A", "R", self.number_messages_received, time.perf_counter(), len(payload_received), 1, self.thread_id))
        # self.number_messages_received += 1

        return self.matched_base_indices
    
    def receive_parameter_estimation_data(self) -> Dict[str, Any]:
        """Receive parameter estimation data from Bob"""
        logger.info(f"Thread {self.thread_id}: Waiting for parameter estimation data...")
        param_data = self.receive_data()
        logger.info(f"Thread {self.thread_id}: Received parameter estimation data")
        return param_data

    # Not here but in bob
    def perform_parameter_estimation(self, param_data: Dict[str, Any], margin: float = 0.1) -> bool:
        """
        Perform parameter estimation and security analysis.
        
        :param param_data: Parameter estimation data from Bob
        :return: True if secure, False if eavesdropping detected
        """
        logger.info(f"Thread {self.thread_id}: Performing WCP parameter estimation...")
        
        # # Organize data by pulse type and basis
        # for local_idx in self.matched_base_indices:
        #     global_idx = self.bob_detected_idxs[local_idx]

        #     if global_idx < len(self.wcp_pulses):
        #         wcp_qubit = self.wcp_pulses[global_idx]
        #         alice_bit = wcp_qubit.bit
        #         bob_bit = bob_bases[local_idx] if local_idx < len(bob_bases) else None
                
        #         if bob_bit is not None:
        #             # Check for bit error
        #             bob_measured_bit = self.bits[global_idx]  # This should be Bob's actual measured bit
        #             error = (alice_bit != bob_measured_bit)
                    
        #             # Determine basis (0=Z, 1=X)
        #             basis = 'Z' if wcp_qubit.base == 0 else 'X'
                    
        #             # Add data to estimator
        #             self.parameter_estimator.add_measurement_data(
        #                 pulse_type=wcp_qubit.pulse_type,
        #                 basis=basis,
        #                 sent=1,
        #                 detected=1,
        #                 errors=1 if error else 0
        #             )
        
        # # Also add data for non-detected pulses (for accurate gain calculation)
        # all_indices = set(range(len(self.wcp_pulses)))
        # detected_indices = set(self.bob_detected_idxs)
        # non_detected_indices = all_indices - detected_indices
        
        # for idx in non_detected_indices:
        #     if idx < len(self.wcp_pulses):
        #         wcp_qubit = self.wcp_pulses[idx]
        #         basis = 'Z' if wcp_qubit.base == 0 else 'X'

        #         self.parameter_estimator.add_measurement_data(
        #             pulse_type=wcp_qubit.pulse_type,
        #             basis=basis,
        #             sent=1,
        #             detected=0,
        #             errors=0
        #         )


        # Update parameter estimator with received data
        for pulse_type in ['signal', 'decoy', 'vacuum']:
            if pulse_type in param_data:
                data = param_data[pulse_type]
                for basis in [0, 1]:  # Z and X bases
                    basis_key = f'basis_{basis}'
                    if basis_key in data:
                        basis_data = data[basis_key]
                        
                        # Add measurement data
                        self.parameter_estimator.add_measurement_data(
                            pulse_type=pulse_type,
                            basis=basis,
                            sent=True,  # We know we sent these
                            detected=basis_data.get('detected', 0) > 0,
                            error=basis_data.get('errors', 0) > 0
                        )
        
        # Perform security analysis
        Y_1, e_1 = self.parameter_estimator.estimate_single_photon_parameters()
        attack_detected, attack_message = self.parameter_estimator.detect_pns_attack(margin=margin)
        
        # Log results
        logger.info(f"Thread {self.thread_id}: Single photon yield: {Y_1:.6f}")
        logger.info(f"Thread {self.thread_id}: Single photon error rate: {e_1:.6f}")
        
        if attack_detected:
            logger.warning(f"Thread {self.thread_id}: Security threat detected: {attack_message}")
            return False
        
        if e_1 > self.error_threshold:
            logger.warning(f"Thread {self.thread_id}: Error rate {e_1:.3f} exceeds threshold {self.error_threshold}")
            return False
        
        logger.info(f"Thread {self.thread_id}: Parameter estimation passed security checks")
        return True

    # # ---- NEW AND DONT KNOW WHAT IT DOES ----
    # def receive_parameter_estimates(self):
    #     """Receive parameter estimates from Bob."""
    #     logger.debug(f"Thread id: {self._thread_id}: Receiving parameter estimates...")
    #     parameter_estimates = self.receive_data()
    #     logger.info(f"Thread id: {self._thread_id}: Received parameter estimates: {parameter_estimates}")
        
    #     payload_received = pickle.dumps((self._thread_id, parameter_estimates))
    #     self.messages_received.append(
    #         ("A", "R", self.number_messages_received, time.perf_counter(), len(payload_received), 1, self._thread_id))
    #     self.number_messages_received += 1
        
    #     return parameter_estimates
    
    def receive_test_bits(self) -> List[int]:
        """Receive and verify test bits for eavesdropping detection"""
        if not self.test_bool:
            return []

        self.test_size = int(len(self.matched_base_indices) * self.test_fraction)
        if self.test_size == 0:
            logger.warning(f"Thread {self.thread_id}: Test size is 0, skipping test bits reception")
            return []
        logger.info(f"Thread {self.thread_id}: Waiting for test bits...")
        # bob_test_bits = self.receive_data()
        [self.bob_test_indices, bob_test_bits] = self.receive_data()

        self.bob_test_bits = bob_test_bits  # Store for later use
        logger.info(f"Thread {self.thread_id}: Received {len(bob_test_bits)} test bits from Bob")

        # P.Fonte info
        # payload_received = pickle.dumps((self.thread_id, [self.bob_test_indices, bob_test_bits]))
        # self.messages_received.append(
        #     ("A", "R", self.number_messages_received, time.perf_counter(), len(payload_received), 4, self.thread_id))
        # self.number_messages_received += 1

        return self.bob_test_bits

    def test_eavesdropping(self) -> bool:
        """Perform eavesdropping test"""
        
        logger.info(f"Thread {self.thread_id}: Performing eavesdropping test...")
        if self.test_size == 0:
            logger.warning(f"Thread {self.thread_id}: Test size is 0, skipping eavesdropping test")
            return False
        # # Compare with our test bits
        # test_size = min(len(self.bob_test_bits), int(len(self.matched_base_indices) * self.test_fraction))
        # if test_size == 0:
        #     return True
        # alice_test_bits = [self.bits[self.matched_base_indices[i]] for i in range(test_size)]

        # Compare with Alice's bits
        if logger.isEnabledFor(logging.DEBUG):
            for i in self.bob_test_indices:
                if i >= len(self.alice_matched_base_bits):
                    logger.error(f"Thread {self.thread_id}: Bob's test index {i} out of bounds for Alice's bits of length {len(self.alice_matched_base_bits)}")
                    return False
        alice_test_bits = [self.alice_matched_base_bits[i] for i in self.bob_test_indices if i < len(self.alice_matched_base_bits)]
        errors = sum(1 for a, b in zip(alice_test_bits, self.bob_test_bits) if a != b)
        # error_rate = errors / len(bob_test_bits) if bob_test_bits else 0

        self.pe_qber_percentage = (errors / len(alice_test_bits)) * 100 if alice_test_bits else 0
        self.test_success_bool = self.pe_qber_percentage <= (self.error_threshold * 100)

        logger.info(f"Thread {self.thread_id}: Test error rate: {self.pe_qber_percentage:.3f}% ({errors} errors out of {len(alice_test_bits)})")
        logger.info(f"Thread {self.thread_id}: Eavesdropping test {'passed' if self.test_success_bool else 'failed'}")

        # Send test result
        self.send_data([self.test_success_bool, self.pe_qber_percentage])
        logger.info(f"Thread {self.thread_id}: Sent test result: {self.test_success_bool}, QBER: {self.pe_qber_percentage:.3f}%")
        
        return self.test_success_bool

    def final_remaining_key(self):
        """Generate final remaining key after testing"""
        if not self.matched_base_indices:
            logger.warning(f"Thread {self.thread_id}: No matched base indices for key generation")
            raise ValueError("No matched base indices for key generation")
        
        test_size = int(len(self.matched_base_indices) * self.test_fraction) if self.test_bool else 0
        if test_size != len(self.bob_test_indices):
            logger.warning(f"Thread {self.thread_id}: Test size {test_size} does not match Bob's test indices length {len(self.bob_test_indices)}")
            raise ValueError("Test size does not match Bob's test indices length")
        
        # remaining_matched_base_indices = [self.matched_base_indices[i] for i in range(len(self.matched_base_indices)) if i not in self.bob_test_indices]
        
        # before the loop
        test_idx = set(self.bob_test_indices)
        # build remaining_matched_base_indices
        remaining_matched_base_indices = [idx for i, idx in enumerate(self.matched_base_indices) if i not in test_idx]

        self.final_key = [self.alice_matched_base_bits[i] for i in remaining_matched_base_indices if i < len(self.alice_matched_base_bits)]
        logger.info(f"Thread {self.thread_id}: Generated final key of length {len(self.final_key)}")
    
    def run_alice_wcp_thread(self):
        """Run the complete Alice WCP thread protocol"""
        logger.info(f"Thread {self.thread_id}: Starting Alice WCP protocol...")
        
        try:
            # Receive detected indices from Bob
            detected_indices = self.receive_detected_indices()
            
            # Maybe this after test eavesdropping?
            # Send pulse type information
            self.send_pulse_type_information(detected_indices)
            
            # Send detected bases
            self.send_detected_bases(detected_indices)

            # Receive matched base indices
            self.receive_matched_base_indices()
            
            # Receive test bits if testing is enabled
            if self.test_bool:
                self.receive_test_bits()
                # Perform eavesdropping test
                self.test_eavesdropping()

            if not self.test_success_bool:
                logger.error(f"Thread {self.thread_id}: Eavesdropping test failed with QBER {self.pe_qber_percentage:.3f}%")
                return None, False

            # # Receive and perform parameter estimation
            # param_data = self.receive_parameter_estimation_data()
            # ### TODO: Add margin parameter to this method/init ###
            # security_passed = self.perform_parameter_estimation(param_data, margin=0.1)

            # if not security_passed:
            #     logger.error(f"Thread {self.thread_id}: Security check failed!")
            #     return None, False
            
            
            # Generate final key
            if self.test_success_bool:
                self.final_remaining_key()
            
            logger.info(f"Thread {self.thread_id}: WCP protocol completed successfully")
            return self.final_key, True
            
        except Exception as e:
            logger.error(f"Thread {self.thread_id}: WCP protocol failed: {e}")
            return None, False
    
    @staticmethod
    def get_void_alice_wcp_thread():
        """Return a void Alice WCP thread object"""
        return AliceWCPThread(None, queue.Queue(), -1)

