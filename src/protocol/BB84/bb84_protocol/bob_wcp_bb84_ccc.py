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

# Configure logging
import logging
from examples.logging_setup import setup_logger
logger = setup_logger("WCP BB84 Bob Log", logging.INFO)

def read(role: Role, queues: list[queue.Queue]):
    """Read data from classical communication channel"""
    try:
        while True:
            payload = role.get_from_inbox()
            (thread_id, data) = pickle.loads(payload)
            queues[thread_id].put(data)
    except KeyboardInterrupt:
        role.clean()

class BobWCPQubits:
    """
    Bob-specific methods for the WCP BB84 protocol with decoy state analysis.
    """
    
    def __init__(self, num_qubits: int = 100, num_frames: int = 10, bytes_per_frame: int = 10,
                 sync_frames: int = 100, sync_bytes_per_frame: int = 50,
                 detector_efficiency: float = 0.1, dark_count_rate: float = 1e-6,
                 loss_rate: float = 0.0,
                 signal_intensity: float = 0.5, decoy_intensity: float = 0.1, vacuum_intensity: float = 0.0):
        """
        Initialize Bob for WCP BB84 protocol.
        
        :param num_qubits: Expected number of qubits
        :param num_frames: Number of frames to receive
        :param bytes_per_frame: Number of bytes per frame
        :param sync_frames: Number of synchronization frames
        :param sync_bytes_per_frame: Bytes per synchronization frame
        :param detector_efficiency: Detector efficiency
        :param dark_count_rate: Dark count rate
        :param loss_rate: Channel loss rate
        :param signal_intensity: Expected signal intensity
        :param decoy_intensity: Expected decoy intensity
        """
        self.num_qubits = num_qubits
        self.num_frames = num_frames
        self.bytes_per_frame = bytes_per_frame
        self.sync_frames = sync_frames
        self.sync_bytes_per_frame = sync_bytes_per_frame
        self.detector_efficiency = detector_efficiency
        self.dark_count_rate = dark_count_rate
        self.loss_rate = loss_rate
        
        # WCP intensities (for reference)
        self.intensities = {
            'signal': signal_intensity,
            'decoy': decoy_intensity,
            'vacuum': vacuum_intensity
        }
        
        # Detection data
        self.detected_indices = []
        self.detected_bases = []
        self.detected_bits = []
        self.detected_pulse_bytes = []
        self.detected_timestamps = []
        self.measurement_bases = []

        self.detected_pulse_types = []
        self.detected_intensities = []
        
        # Timing
        self.average_time_bin = None
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
    def setup_client(host: str, port: int) -> socket.socket:
        """Set up client socket for quantum channel"""
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((host, port))
        logger.info(f"Connected to {host}:{port}")
        return client
    
    def receive_sync(self, connection: socket.socket):
        """Receive synchronization from Alice"""
        logger.info("Receiving synchronization from Alice...")
        
        sync_times = []
        try:
            for frame in range(self.sync_frames):
                start_time = time.time()
                data = connection.recv(self.sync_bytes_per_frame + 2)  # +2 for markers
                
                if data and len(data) > 0:
                    sync_times.append(time.time() - start_time)
                
            # Calculate average timing
            if sync_times:
                self.average_time_bin = np.mean(sync_times)
                logger.info(f"Synchronization completed. Average time bin: {self.average_time_bin:.6f}s")
            else:
                self.average_time_bin = 0.001  # Default fallback
                
        except Exception as e:
            logger.error(f"Synchronization failed: {e}")
            raise
        
        return self.average_time_bin

    def receive_sync2(self, connection: socket.socket):
        """Receive synchronization from Alice."""
        logger.info("Receiving synchronization from Alice...")
        
        times_between_qubits = []
        try:
            for frame in range(self.sync_frames):
                # Wait for start marker
                data = connection.recv(1)
                while data[0] != self._start_marker:
                    data = connection.recv(1)
                
                start_time = time.time_ns()
                prev_time = start_time
                
                # Receive sync bytes
                for _ in range(self.sync_bytes_per_frame):
                    byte = connection.recv(1)
                    current_time = time.time_ns()
                    time_diff = current_time - prev_time
                    times_between_qubits.append(time_diff)
                    prev_time = current_time
                
                # Wait for end marker
                data = connection.recv(1)
                while data[0] != self._end_marker:
                    data = connection.recv(1)
                
                logger.debug(f"Sync frame {frame} received")
                
            # Calculate average time bin
            if times_between_qubits:
                self.average_time_bin = np.mean(times_between_qubits)
            
            logger.info(f"Average time between qubits: {self.average_time_bin / 1000:.2f} us")
            
        except Exception as e:
            logger.error(f"Error during synchronization: {e}")
            
        logger.info("Synchronization received.")
        return self.average_time_bin

    def receive_wcp_pulses(self, connection: socket.socket):
        """Receive and detect WCP pulses from Alice"""
        logger.info("Receiving WCP pulses from Alice...")
        
        self.detected_indices = []
        self.detected_pulse_bytes = []
        self.detected_timestamps = []
        self.measurement_bases = []
        self.detected_bits = []
        
        try:
            pulse_idx = 0
            
            for frame in range(self.num_frames):
                frame_data = connection.recv(self.bytes_per_frame + 2)  # +2 for markers
                
                if not frame_data or len(frame_data) < 3:
                    continue
                
                # Process frame (skip start and end markers)
                pulse_data = frame_data[1:-1]
                
                for byte_val in pulse_data:
                    if pulse_idx >= self.num_qubits:
                        break
                    
                    # Generate random measurement basis
                    measurement_base = random.randint(0, 1)
                    
                    # Simulate WCP pulse detection
                    detected, measured_bit = WCPPulse.measure_byte(
                        byte_val, measurement_base, self.intensities,
                        self.detector_efficiency, self.dark_count_rate
                    )
                    
                    if detected:
                        self.detected_indices.append(pulse_idx)
                        self.detected_pulse_bytes.append(byte_val)
                        self.measurement_bases.append(measurement_base)
                        self.detected_bits.append(measured_bit)
                        self.detected_timestamps.append(time.time())
                    
                    pulse_idx += 1
                
        except Exception as e:
            logger.error(f"Failed to receive WCP pulses: {e}")
            raise
        
        logger.info(f"Detected {len(self.detected_indices)} pulses out of {pulse_idx} sent")
        return self.detected_pulse_bytes

    def receive_wcp_pulses2(self, connection: socket.socket):
        """Simulates receiving qubits with detection times with some loss."""
        logger.info("Receiving qubits from Alice...")
        placeholder = []
        
        try:
            # Set up randomly chosen measurement bases
            measurement_bases = self.generate_random_bases_bits(self.num_qubits)
            
            for frame in range(self.num_frames):
                # Wait for start marker
                data = connection.recv(1)
                while data[0] != self._start_marker:
                    data = connection.recv(1)
                
                start_time = time.time_ns()
                
                # Receive qubits in frame
                for byte_idx in range(self.bytes_per_frame):
                    idx = frame * self.bytes_per_frame + byte_idx
                    qubit_byte = connection.recv(1)[0]
                    
                    current_time = time.time_ns()
                    
                    # Skip if beyond expected number of qubits
                    if idx >= self.num_qubits:
                        continue
                    
                    ### TODO: Move this to after the loop to avoid unnecessary processing while waiting for the end marker
                    # Convert byte to WCP qubit
                    wcp_qubit = WCPPulse.from_byte_wcp(qubit_byte)
                    # wcp_qubit = WCPPulse.from_byte_and_intensity(qubit_byte)
                    
                    # Simulate measurement with detection efficiency and dark count
                    if idx < len(measurement_bases):
                        is_detected, result = wcp_qubit.measure_wcp(
                            measurement_bases[idx],
                            self.detector_efficiency,
                            self.dark_count_rate
                        )
                        
                        # If the qubit is detected, record it
                        if is_detected and result is not None:
                            self.detected_indicess.append(idx)
                            self.detected_bases.append(measurement_bases[idx])
                            self.detected_bits.append(result)
                            self.detected_pulse_bytes.append(qubit_byte)
                            self.detected_timestamps.append(current_time)
                            self.detected_intensities.append(wcp_qubit.get_intensity_type())
                
                # Wait for end marker
                data = connection.recv(1)
                while data[0] != self._end_marker:
                    data = connection.recv(1)
                
                logger.debug(f"Frame {frame} received in {(time.time_ns() - start_time) / 1000} us")
                
        except Exception as e:
            logger.error(f"Error receiving qubits: {e}")

        logger.info(f"Qubits received: {len(self.detected_pulse_bytes)}")
        return self.detected_pulse_bytes
    
    def run_bob_wcp_protocol(self, connection: socket.socket):
        """Run Bob's complete WCP protocol"""
        logger.info("Starting Bob WCP BB84 protocol...")
        
        try:
            # Receive synchronization
            self.receive_sync(connection)
            # self.average_time_bin = self.receive_sync(connection)
            
            # Receive WCP pulses
            self.receive_wcp_pulses(connection)
            
            logger.info("Bob WCP protocol completed successfully")
            
        except Exception as e:
            logger.error(f"Bob WCP protocol failed: {e}")
            raise
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt in run_bob_qubits.")
            raise



    # def run_mock_bob_qubits(self, connection: socket.socket, wait_time: bool = True, 
    #                        rand_fake_error_bool: bool = False, rand_fake_error_rate: float = 0.0, 
    #                        fixed_fake_error_bool: bool = False, fixed_fake_error_rate: float = 0.0, 
    #                        lock: threading.Lock = None):
    #     """Run Bob's part for a single thread to obtain part of the final key."""
    #     self.recv_lock = lock
    #     try:
    #         logger.info("Running Mock Bob's WCP qubits operations for testing...")
    #         self.receive_qubits_mock(connection, wait_time, rand_fake_error_bool, 
    #                                 rand_fake_error_rate, fixed_fake_error_bool, 
    #                                 fixed_fake_error_rate)
    #         logger.info(f"Bob received {len(self.detected_pulse_bytes)} qubits.")
    #         return self.detected_pulse_bytes
    #     except Exception as e:
    #         logger.error(f"Error in run_mock_bob_qubits: {e}")
    #         return []
    #     except KeyboardInterrupt:
    #         logger.warning("Keyboard interrupt in run_mock_bob_qubits.")
    #         return []

    # def receive_qubits_mock(self, connection: socket.socket, wait_time: bool = True, 
    #                        rand_fake_error_bool: bool = False, rand_fake_error_rate: float = 0.0,
    #                        fixed_fake_error_bool: bool = False, fixed_fake_error_rate: float = 0.0):
    #     """Simulates receiving qubits with socket connection."""
    #     logger.info("Receiving qubits from Alice via socket...")
        
    #     try:
    #         if self.recv_lock:
    #             self.recv_lock.acquire()
            
    #         # Get number of qubits
    #         size_bytes = connection.recv(2)
    #         num_qubits_expected = size_bytes[0] * 256 + size_bytes[1]
            
    #         # Set up randomly chosen measurement bases
    #         self.measurement_bases = self.generate_random_bases_bits(num_qubits_expected)
            
    #         # Receive qubits
    #         for i in range(num_qubits_expected):
    #             byte = connection.recv(1)[0]
                
    #             if i < len(self.measurement_bases):
    #                 measurement_base = self.measurement_bases[i]
                    
    #                 # Convert byte to WCP pulse
    #                 wcp_pulse = WCPPulse.from_byte_and_intensity(byte, self.intensities)
                    
    #                 # Simulate measurement
    #                 is_detected, result = wcp_pulse.measure(
    #                     measurement_base,
    #                     self.detection_efficiency,
    #                     self.dark_count_prob
    #                 )
                    
    #                 # If the pulse is detected
    #                 if is_detected and result is not None:
    #                     self.detected_indicess.append(i)
    #                     self.detected_bases.append(measurement_base)
    #                     self.detected_bits.append(result)
    #                     self.detected_pulse_bytes.append(byte)
    #                     self.detected_intensities.append(wcp_pulse.get_intensity_type())
                
    #             if wait_time:
    #                 self.time_sleep(100)  # Simulating processing time
            
    #         if self.recv_lock:
    #             self.recv_lock.release()
            
    #         # Simulate random errors if requested
    #         if rand_fake_error_bool and 0 < rand_fake_error_rate < 1:
    #             for i in range(len(self.detected_bits)):
    #                 if random.random() < rand_fake_error_rate:
    #                     self.detected_bits[i] = 1 - self.detected_bits[i]  # Flip the bit
            
    #         # Simulate fixed error rate if requested
    #         if fixed_fake_error_bool and 0 < fixed_fake_error_rate < 1:
    #             num_errors = int(fixed_fake_error_rate * len(self.detected_bits))
    #             error_indices = random.sample(range(len(self.detected_bits)), num_errors)
    #             for i in error_indices:
    #                 self.detected_bits[i] = 1 - self.detected_bits[i]  # Flip the bit
                    
    #     except Exception as e:
    #         logger.error(f"Error receiving qubits via socket: {e}")
    #         if self.recv_lock and self.recv_lock.locked():
    #             self.recv_lock.release()

    #     logger.info(f"Qubits received via socket: {len(self.detected_pulse_bytes)} with detection efficiency {self.detection_efficiency}")
    #     return self.detected_pulse_bytes
    
    
    def run_mock_bob_qubits(self, connection: socket.socket, wait_time: bool = True, 
                            rand_fake_error_bool: bool = False, rand_fake_error_rate: float = 0.0, 
                            fixed_fake_error_bool: bool = False, fixed_fake_error_rate: float = 0.0, 
                            lock: threading.Lock = None):
        """Run Bob's part for a single thread to obtain part of the final key."""
        self.recv_lock = lock
        try:
            #self.receive_sync(connection)
            self.receive_qubits_mock(connection, wait_time, rand_fake_error_bool, rand_fake_error_rate, fixed_fake_error_bool, fixed_fake_error_rate)
            # Send a 1 through indicating end
            # connection.sendall(b'\x01')
        except Exception as e:
            logger.error(f"Bob encountered an error in run_bob_qubits: {e}")
        except KeyboardInterrupt:
            logger.error("Stopped by user.")

    def receive_qubits_mock(self, connection: socket.socket, wait_time: bool = True,
                            rand_fake_error_bool: bool = False, rand_fake_error_rate: float = 0.0,
                            fixed_fake_error_bool: bool = False, fixed_fake_error_rate: float = 0.0):
        """Simulates receiving qubits with socket connection."""
        logger.info("Receiving qubits from Alice via socket...")
        try:
    
            #### ????
            buffer = b''
            while True:
                chunk = connection.recv(4096)
                if not chunk:
                    raise ConnectionError("Connection closed unexpectedly.")
                buffer += chunk

                # Find start and end markers in the buffer
                start_idx = buffer.find(self._start_marker.to_bytes(1, 'big'))
                end_idx = buffer.find(self._end_marker.to_bytes(1, 'big'))

                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    # Extract payload
                    payload = buffer[start_idx + 1:end_idx]
                    try:
                        self.pulse_bytes, self.average_time_bin = pickle.loads(payload)
                        logger.info(f"Received {len(self.pulse_bytes)} qubits from Alice without loss")
                        # Remove used data from buffer in case there are more messages
                        buffer = buffer[end_idx + 1:]
                        break
                    except (pickle.PickleError, EOFError):
                        logger.error("Failed to deserialize payload, skipping...")
                        buffer = buffer[end_idx + 1:]
                        continue

            logger.info("Transmission complete. Transform time of detections to indices.")

            if wait_time:
                # Simulate processing time based on average time bin
                self.time_sleep(self.average_time_bin * len(self.pulse_bytes))

            logger.info(f"Qubits received via socket: {len(self.pulse_bytes)} with detection efficiency {self.detector_efficiency}")
            self.process_data_mock_queue(rand_fake_error_bool, rand_fake_error_rate, fixed_fake_error_bool, fixed_fake_error_rate)
            logger.info(f"Bob processed {len(self.detected_pulse_types)} qubits from socket.")

        except Exception as e:
            logger.error(f"Error receiving qubits via socket: {e}")
            if self.recv_lock and self.recv_lock.locked():
                self.recv_lock.release()
            raise
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt in receive_qubits_mock.")
            if self.recv_lock and self.recv_lock.locked():
                self.recv_lock.release()
            raise
        return self.pulse_bytes

    ### USING QUEUE-BASED COMMUNICATION FOR SINGLE FILE PROGRAM

    def run_mock_bob_qubits_queue(self, shared_queue: queue.Queue, connection: socket.socket, 
                                 wait_time: bool = True, rand_fake_error_bool: bool = False, 
                                 rand_fake_error_rate: float = 0.0, fixed_fake_error_bool: bool = False, 
                                 fixed_fake_error_rate: float = 0.0):
        """Run Bob's part for a single thread with queue-based communication."""
        logger.info("Receiving WCP qubits from Alice via queue...")
        try:
            self.receive_qubits_mock_queue(shared_queue, wait_time)
            self.process_data_mock_queue(rand_fake_error_bool, rand_fake_error_rate, 
                                        fixed_fake_error_bool, fixed_fake_error_rate)
            logger.info(f"Bob processed {len(self.detected_pulse_bytes)} qubits from queue.")
            
            # Send acknowledgment to Alice
            connection.sendall(b'\x01')
            return self.detected_pulse_bytes
            
        except Exception as e:
            logger.error(f"Error in run_mock_bob_qubits_queue: {e}")
            return []
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt in run_mock_bob_qubits_queue.")
            return []

    def receive_qubits_mock_queue(self, shared_queue: queue.Queue, wait_time: bool = True):
        """Simulates receiving qubits from a shared queue."""
        logger.info("Receiving qubits from Alice via queue...")
        try:

            # Get qubits from the queue until it's empty
            # self.received_qubits = []
            # while not shared_queue.empty():
            #     idx, byte = shared_queue.get()
            #     self.received_qubits.append((idx, byte))
            #     if wait_time:
            #         self.time_sleep(100)  # Simulating processing time
            
            data = shared_queue.get()
            [self.pulse_bytes, self.qubit_delay_us] = data
            logger.info(f"Received {len(self.pulse_bytes)} qubits from queue with delay {self.qubit_delay_us} us but only part will be measured.")
            if wait_time:
                self.time_sleep(self.qubit_delay_us * len(self.pulse_bytes))
            

        except Exception as e:
            logger.error(f"Error receiving qubits from queue: {e}")

    def process_data_mock_queue(self, rand_fake_error_bool: bool = False, 
                               rand_fake_error_rate: float = 0.0,
                               fixed_fake_error_bool: bool = False, 
                               fixed_fake_error_rate: float = 0.0):
        """Process the received qubits data."""
        logger.info("Processing received qubits...")
        
        self.detected_indices = []
        self.detected_bases = []
        self.detected_bits = []
        self.detected_pulse_bytes = []
        self.detected_intensities = []
        self.detected_pulse_types = []
        self.measurement_bases = []

        # Set up randomly chosen measurement bases for all possible qubits
        self.measurement_bases_temp = self.generate_random_bases_bits(self.num_qubits)        

        # Process each received pulse
        if len(self.pulse_bytes) != len(self.measurement_bases_temp):
            logger.error(f"Mismatch in pulse bytes and measurement bases length: "
                         f"{len(self.pulse_bytes)} vs {len(self.measurement_bases_temp)}")
            raise ValueError("Mismatch in pulse bytes and measurement bases length")
        
        for idx, byte in enumerate(self.pulse_bytes):
            measurement_base = self.measurement_bases_temp[idx]

            # Convert byte to WCP pulse
            wcp_pulse = WCPPulse.from_byte_and_intensity(byte, self.intensities)

            # # Apply channel loss
            # if self.loss_rate > 0:
            #     # if random.random() < self.loss_rate:
            #     #     continue
            #     wcp_pulse.apply_channel_loss(transmission_efficiency= 1 - self.loss_rate)
            
            # Simulate measurement
            is_detected, result = wcp_pulse.measure(
                measurement_base,
                self.detector_efficiency,
                self.dark_count_rate
            )
            print(f"Pulse {idx}: Detected={is_detected}, Result={result}, Base={measurement_base}, Byte={byte}")

            # If the pulse is detected
            if is_detected and result is not None:
                self.detected_indices.append(idx)
                self.detected_bases.append(measurement_base)
                self.detected_bits.append(result)
                self.detected_pulse_bytes.append(byte) # In reality bob don't know the type of the pulse
                self.detected_intensities.append(wcp_pulse.intensity)
                self.detected_pulse_types.append(wcp_pulse.pulse_type)
        
        # Simulate random errors if requested
        if rand_fake_error_bool and 0 < rand_fake_error_rate < 1:
            for i in range(len(self.detected_bits)):
                if random.random() < rand_fake_error_rate:
                    self.detected_bits[i] = 1 - self.detected_bits[i]  # Flip the bit
        
        # Simulate fixed error rate if requested
        if fixed_fake_error_bool and 0 < fixed_fake_error_rate < 1:
            num_errors = int(fixed_fake_error_rate * len(self.detected_bits))
            error_indices = random.sample(range(len(self.detected_bits)), num_errors)
            for i in error_indices:
                self.detected_bits[i] = 1 - self.detected_bits[i]  # Flip the bit
                
        logger.info(f"Processed {len(self.detected_pulse_bytes)} qubits after detection simulation")
            

    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected pulses"""
        total_detected = len(self.detected_indices)
        
        if total_detected == 0:
            return {'total_detected': 0}
        
        # Analyze detected pulse types
        pulse_type_counts = {'signal': 0, 'decoy': 0, 'vacuum': 0}
        
        for byte_val in self.detected_pulse_bytes:
            bit, base, pulse_type_idx = WCPPulse.byte_to_pulse_info(byte_val)
            
            if pulse_type_idx == 0:
                pulse_type_counts['signal'] += 1
            elif pulse_type_idx == 1:
                pulse_type_counts['decoy'] += 1
            elif pulse_type_idx == 2:
                pulse_type_counts['vacuum'] += 1
        
        return {
            'total_detected': total_detected,
            'signal_detected': pulse_type_counts['signal'],
            'decoy_detected': pulse_type_counts['decoy'],
            'vacuum_detected': pulse_type_counts['vacuum'],
            'types_rates': {
                'signal_rate': pulse_type_counts['signal'] / total_detected,
                'decoy_rate': pulse_type_counts['decoy'] / total_detected,
                'vacuum_rate': pulse_type_counts['vacuum'] / total_detected
            }
        }
    
    def reset(self):
        """Reset all detection data"""
        self.detected_indices = []
        self.detected_bases = []
        self.detected_bits = []
        self.detected_pulse_bytes = []
        self.detected_timestamps = []
        self.measurement_bases = []

    def return_bob_reseted(self):
        """Return the reset Bob object."""
        self.reset()
        return self
    

class BobWCPThread:
    """
    Bob thread for WCP BB84 classical communication processing.
    """
    
    def __init__(self, role: Role, receive_queue: queue.Queue, thread_id: int,
                 test_bool: bool = True, test_fraction: float = 0.1,
                 error_threshold: float = 0.1, start_idx: int = 0,
                 detector_efficiency: float = 0.1, dark_count_rate: float = 1e-6):
        """
        Initialize Bob WCP thread.
        
        :param role: Communication role
        :param receive_queue: Queue for receiving data
        :param thread_id: Thread identifier
        :param test_bool: Whether to perform eavesdropping test
        :param test_fraction: Fraction of bits to use for testing
        :param error_threshold: Error threshold for eavesdropping detection
        :param start_idx: Starting index for this thread's data
        :param detector_efficiency: Detector efficiency
        :param dark_count_rate: Dark count rate
        """
        # Dealing with mandatory parameters
        self.role = role
        self.receive_queue = receive_queue
        self.thread_id = thread_id
        # Dealing with optional parameters
        self.test_bool = test_bool
        self.test_fraction = test_fraction
        self.error_threshold = error_threshold
        self.start_idx = start_idx
        self.detector_efficiency = detector_efficiency
        self.dark_count_rate = dark_count_rate
        
        # Detection data
        self.detected_bits = []
        self.detected_bases = []
        self.detected_pulse_bytes = []
        self.detected_indices = []
        self.measurement_bases = []
        self.pulse_types = []
        self.pulse_intensities = []
        
        # Processing results
        self.matched_base_indices = []
        self.matched_base_bits = []
        self.matched_base_bases = []
        self.final_key = []
        self.bob_payload_size = []
        
        # Parameter estimator
        self.test_success_bool = None
        self.test_size = 0
        self.pe_qber_percentage = 0.0

        self.parameter_estimator = WCPParameterEstimator(
            detector_efficiency=detector_efficiency,
            dark_count_rate=dark_count_rate
        )
        
        logger.info(f"Bob WCP Thread {thread_id} initialized")
    
    def set_wcp_detection_data(self, detected_bits: List[int], measurement_bases: List[int],
                               detected_pulse_bytes: List[int], detected_indices: List[int]):
        """Set WCP detection data for this thread"""
        self.detected_bits = detected_bits
        self.measurement_bases = measurement_bases
        self.detected_pulse_bytes = detected_pulse_bytes
        self.detected_indices = detected_indices
        # self.detected_intensities = detected_intensities
        # self.average_time_bin = average_time_bin

        logger.info(f"Thread {self.thread_id}: Set {len(detected_bits)} detected WCP pulses")
    
    def send_data(self, data: Any, list_size_to_append: List = None):
        """Send data through classical channel"""
        payload = pickle.dumps((self.thread_id, data))
        self.role.put_in_outbox(payload)
        
        if list_size_to_append is not None:
            list_size_to_append.append(len(data) if hasattr(data, '__len__') else 1)
        else:
            self.bob_payload_size.append(len(data) if hasattr(data, '__len__') else 1)
    
    def receive_data(self) -> Any:
        """Receive data from classical channel"""
        return self.receive_queue.get()
    
    def send_detected_indices(self):
        """Send indices of detected pulses to Alice"""
        logger.info(f"Thread {self.thread_id}: Sending {len(self.detected_indices)} detected indices")
        self.send_data(self.detected_indices)
        logger.info(f"Thread {self.thread_id}: Detected indices sent successfully")
    
    def receive_pulse_type_information(self) -> Dict[str, Any]:
        """Receive pulse type information from Alice"""
        logger.info(f"Thread {self.thread_id}: Waiting for pulse type information...")
        pulse_data = self.receive_data()
        
        # Extract pulse types and intensity map
        self.pulse_types = pulse_data['pulse_types']
        intensity_map = pulse_data['intensity_map']
        
        # Map pulse type integers back to intensities
        self.pulse_intensities = []
        for pulse_type_int in self.pulse_types:
            self.pulse_intensities.append(intensity_map[pulse_type_int])
        # self.pulse_intensities = [intensity_map.get(pt, 0.0) for pt in self.pulse_types]        

        logger.info(f"Thread {self.thread_id}: Received pulse type info for {len(self.pulse_types)} pulses with intensity map")
        return pulse_data
    
    def receive_detected_bases(self) -> List[int]:
        """Receive bases for detected pulses from Alice"""
        logger.info(f"Thread {self.thread_id}: Waiting for detected bases...")
        self.detected_bases = self.receive_data()
        logger.info(f"Thread {self.thread_id}: Received {len(self.detected_bases)} detected bases")
        return self.detected_bases
    
    def match_bases(self):
        """Find indices where measurement bases match Alice's bases"""
        self.matched_base_indices = []
        self.matched_base_bits = []
        self.matched_base_bases = []

        for i, (alice_base, bob_base) in enumerate(zip(self.detected_bases, self.measurement_bases)):
            if alice_base == bob_base:
                self.matched_base_indices.append(i)
                self.matched_base_bits.append(self.detected_bits[i])
                self.matched_base_bases.append(alice_base)
        
        logger.info(f"Thread {self.thread_id}: Found {len(self.matched_base_indices)} matching bases")
    
    def send_matched_base_indices(self) -> List[int]:
        """Send indices where bases match"""
        logger.info(f"Thread {self.thread_id}: Sending {len(self.matched_base_indices)} common indices")
        self.send_data(self.matched_base_indices)
        logger.info(f"Thread {self.thread_id}: Common indices sent successfully")
        return self.matched_base_indices
    
    def perform_parameter_estimation(self) -> Dict[str, Any]:
        """
        Perform parameter estimation with detected WCP pulses.
        
        :return: Parameter estimation data
        """
        logger.info(f"Thread {self.thread_id}: Performing parameter estimation...")
        
        # Reset parameter estimator
        self.parameter_estimator.reset_statistics()
        
        # Organize data by pulse type and basis
        param_data = {
            'signal': {'basis_0': {'sent': 0, 'detected': 0, 'errors': 0}, 
                      'basis_1': {'sent': 0, 'detected': 0, 'errors': 0}},
            'decoy': {'basis_0': {'sent': 0, 'detected': 0, 'errors': 0}, 
                     'basis_1': {'sent': 0, 'detected': 0, 'errors': 0}},
            'vacuum': {'basis_0': {'sent': 0, 'detected': 0, 'errors': 0}, 
                      'basis_1': {'sent': 0, 'detected': 0, 'errors': 0}}
        }
        
        # Process detected pulses for parameter estimation
        for i in range(len(self.detected_indices)):
            # check if all have the same length
            print(f"Thread {self.thread_id}: Checking lengths: "
                    f"detected_indices={len(self.detected_indices)}, "
                    f"pulse_types={len(self.pulse_types)}, "
                    f"detected_bases={len(self.detected_bases)}, "
                    f"measurement_bases={len(self.measurement_bases)}, ")

            if i >= len(self.pulse_types) or i >= len(self.detected_bases) or i >= len(self.measurement_bases):
                continue
            
            # Determine pulse type
            pulse_type_idx = self.pulse_types[i]
            if pulse_type_idx == 0:
                pulse_type_str = 'signal'
            elif pulse_type_idx == 1:
                pulse_type_str = 'decoy'
            elif pulse_type_idx == 2:
                pulse_type_str = 'vacuum'
            else:
                print(f"No match for the pulse type. Got {pulse_type_idx}, expected 0,1,2")
            # Determine if bases match (for error calculation)
            alice_base = self.detected_bases[i]
            bob_base = self.measurement_bases[i]
            bases_match = (alice_base == bob_base)
            
            # Count detections
            basis_key = f'basis_{alice_base}'
            param_data[pulse_type_str][basis_key]['sent'] += 1
            param_data[pulse_type_str][basis_key]['detected'] += 1
            
            ### TODO: THIS ONLY WORKS IF WE HAD THE DETECTED PULSE BYTES FROM ALICE BUT IN REALITY ONLY DETECTED BITS ARE OBTAINED FROM QUANTUM CHANNEL
            # Count errors (only meaningful when bases match)
            if bases_match and i < len(self.matched_base_bits):
                # Compare with Alice's bit (we need to simulate this)
                alice_bit, _, _ = WCPPulse.byte_to_pulse_info(self.detected_pulse_bytes[i])
                bob_bit = self.detected_bits[i]
                
                if alice_bit != bob_bit:
                    param_data[pulse_type_str][basis_key]['errors'] += 1
            
            # Add to parameter estimator
            self.parameter_estimator.add_measurement_data(
                pulse_type=pulse_type_str,
                basis=alice_base,
                sent=True,
                detected=True,
                error=(bases_match and i < len(self.matched_base_bits) and 
                      WCPPulse.byte_to_pulse_info(self.detected_pulse_bytes[i])[0] != self.detected_bits[i])
            )

            ### THE OTHER WAY AROUND, TO ADD MEASUREMENT DATA WITHOUT PULSE BYTES
            # self.parameter_estimator.add_measurement_data(
            #     pulse_type=pulse_type_str,
            #     basis=alice_base,
            #     sent=True,
            #     detected=True,
            #     error=False # This will be always false, although after receive test bits we should update this
            # )
        
        logger.info(f"Thread {self.thread_id}: Parameter estimation completed")
        return param_data
    
    def send_parameter_estimation_data(self):
        """Send parameter estimation data to Alice"""
        param_data = self.perform_parameter_estimation()
        logger.info(f"Thread {self.thread_id}: Sending parameter estimation data")
        self.send_data(param_data)
        return param_data
    
    def send_test_bits(self):
        """Send test bits and their indices for eavesdropping detection"""
        if not self.test_bool:
            logger.info(f"Thread {self.thread_id}: Skipping test bits as test_bool is False")
            return
        
        print(len(self.matched_base_bits))
        print(len(self.matched_base_bits) * self.test_fraction)
        self.test_size = int(len(self.matched_base_bits) * self.test_fraction)
        if self.test_size == 0:
            logger.info(f"Thread {self.thread_id}: Skipping test bits as test size is 0")
            return 
        
        # Randomly select test indices
        print("Here")
        self.test_indices = random.sample(range(len(self.matched_base_bits)), self.test_size)
        print("Here")
        self.test_bits = [self.matched_base_bits[i] for i in self.test_indices]
        
        logger.info(f"Thread {self.thread_id}: Sending {len(self.test_bits)} test bits and their indices")
        self.send_data([self.test_indices, self.test_bits])
        logger.info(f"Thread {self.thread_id}: Test bits sent successfully")


    def receive_test_results(self):
        """Receive test results from Alice"""
        if self.test_size == 0:
            logger.info(f"Thread {self.thread_id}: Skipping test bits as test size is 0")
            return 
        logger.info(f"Thread {self.thread_id}: Receive Test Results from Alice")
        # Receive test results from Alice
        self.test_results = self.receive_data()
        [self.test_success_bool, self.pe_qber_percentage] = self.test_results
        logger.info(f"Thread {self.thread_id}: Received test results: "
                    f"success={self.test_success_bool}, QBER={self.pe_qber_percentage:.2f}%")
        return self.test_success_bool, self.pe_qber_percentage

    def final_remaining_key(self):
        """Generate final remaining key after testing"""
        if not self.matched_base_bits:
            logger.warning(f"Thread {self.thread_id}: No matched bits for key generation")
            return

        if self.test_size != len(self.test_indices):
            logger.warning(f"Thread {self.thread_id}: Test size {self.test_size} does not match received test indices length {len(self.test_indices)}")
            return
        test_idx = set(self.test_indices)
        remaining_indices = [idx for i, idx in enumerate(self.matched_base_indices) if i not in test_idx]
        self.final_key = [self.matched_base_bits[i] for i in remaining_indices if i < len(self.matched_base_bits)]

        logger.info(f"Thread {self.thread_id}: Generated final key of length {len(self.final_key)}")
    
    def analyze_security_parameters(self) -> Tuple[bool, str]:
        """
        Analyze security parameters using WCP parameter estimation.
        
        :return: Tuple of (secure, message)
        """
        # Get parameter estimation results
        Y_1, e_1 = self.parameter_estimator.estimate_single_photon_parameters()
        attack_detected, attack_message = self.parameter_estimator.detect_pns_attack()
        
        logger.info(f"Thread {self.thread_id}: Single photon yield: {Y_1:.6f}")
        logger.info(f"Thread {self.thread_id}: Single photon error rate: {e_1:.6f}")
        
        if attack_detected:
            logger.warning(f"Thread {self.thread_id}: PNS attack detected: {attack_message}")
            return False, attack_message
        
        if e_1 > self.error_threshold:
            message = f"Single photon error rate {e_1:.3f} exceeds threshold {self.error_threshold}"
            logger.warning(f"Thread {self.thread_id}: {message}")
            return False, message
        
        logger.info(f"Thread {self.thread_id}: Security analysis passed")
        return True, "Security analysis passed"
    
    def run_bob_wcp_thread(self):
        """Run the complete Bob WCP thread protocol"""
        logger.info(f"Thread {self.thread_id}: Starting Bob WCP protocol...")
        
        try:
            # Send detected indices to Alice
            self.send_detected_indices()
            
            # Maybe this after receiving test results?
            # Receive pulse type information
            self.receive_pulse_type_information()
            
            # Receive detected bases from Alice
            self.receive_detected_bases()
            
            # Match bases and find common indices
            self.match_bases()
            
            # Send common indices to Alice
            self.send_matched_base_indices()
            
            if self.test_bool:
                logger.info(f"Thread {self.thread_id}: Sending test bits for eavesdropping detection")
                # Send test bits for eavesdropping detection
                self.send_test_bits()
                # Receive test results from Alice
                self.receive_test_results()
            else:
                logger.info(f"Thread {self.thread_id}: Skipping test bits as test_bool is False")
            
            # # Perform and send parameter estimation
            # self.send_parameter_estimation_data()
            
            # # Analyze security parameters
            # secure, security_message = self.analyze_security_parameters()
            
            # if not secure:
            #     logger.error(f"Thread {self.thread_id}: Security check failed: {security_message}")
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
    def get_void_bob_wcp_thread():
        """Return a void Bob WCP thread object"""
        return BobWCPThread(None, queue.Queue(), -1)
