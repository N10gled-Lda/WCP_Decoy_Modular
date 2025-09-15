import pickle
import socket
import time
import random
import numpy as np
import queue

from BB84.bb84_protocol.qubit2 import Qubit2
from classical_communication_channel.communication_channel.role import Role

# Configure logging
import logging
from logging_setup import setup_logger
# Setup logger
logger = setup_logger("BB84 Simulation Log", logging.INFO)

def read(role: Role, queues: list[queue.Queue]):
    try:
        while True:
            payload = role.get_from_inbox()
            (thread_id, data) = pickle.loads(payload)
            queues[thread_id].put(data)
    except KeyboardInterrupt:
        role.clean()

class BobQubits():
    """
    Bob-specific methods for the BB84 protocol.
    """
    def __init__(self, num_qubits: int = 100, num_frames: int = 10, bytes_per_frame: int = 10, sync_frames: int = 100, sync_bytes_per_frame: int = 50, loss_rate: float = 0.0):
        """Initialize Bob's qubits and bits."""
        self.num_qubits = num_qubits
        self.num_frames = num_frames
        self.bytes_per_frame = bytes_per_frame
        self.sync_frames = sync_frames
        self.sync_bytes_per_frame = sync_bytes_per_frame
        self.loss_rate = loss_rate
        # self.classical_channel = ClassicalChannel()

        # Bits Bases and Qubits for each role
        self.bases = [] # Not needed
        self.bits = [] # Not needed
        self.qubits = [] # Not needed
        self.qubits_bytes = [] # Not needed
        self.detected_idxs = []
        self.detected_bases = []
        self.detected_bits = []
        self.detected_qubits_bytes = []
        self.detected_timestamps = []

        self._start_marker = 100
        self._end_marker = 50

    @staticmethod
    def generate_random_bases_bits(number: int):
        """Generate random bits or bases."""
        return [random.choice([0, 1]) for _ in range(number)]
    
    @staticmethod
    def time_sleep(time_sleep: int):
        start_sleep_time = time.perf_counter_ns()
        while time.perf_counter_ns() - start_sleep_time < time_sleep * 1000:
            pass

    @staticmethod
    def setup_client(host: str, port: int) -> socket.socket:
        """Setup client socket for Bob."""
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((host, port))
        return client

    def run_bob_qubits(self, connection: socket.socket):
        """Run Bob's part for a single thread to obtian part of the final key."""
        try:
            self.receive_sync(connection)
            self.receive_qubits(connection)
        except Exception as e:
            logger.error(f"Bob encountered an error in run_bob_qubits: {e}")
        except KeyboardInterrupt:
            logger.error("Stopped by user.")


    def receive_sync(self, connection: socket.socket):
        """Receive synchronization from Alice."""
        logger.info("Receiving synchronization from Alice...")
        try:
            time_bins = []
            times_all = []
            time_bins_sigma = []
            bytes_placehoder_list = []
            times_elapsed = []
            self.average_time_bin = None
            for frame in range(self.sync_frames):
                
                while True:
                    data = connection.recv(1)
                    # thread_id_b, data = self.receive_with_thread_id(connection)
                    # if thread_id_b != self.thread_id_byte:
                    #     logger.error(f"Thread ID mismatch. Expected {self.thread_id_byte}, received {thread_id_b}.")
                    #     return None

                    # if not data:
                    #     break
                    
                    byte = int.from_bytes(data, 'big')

                    if byte == self._start_marker:
                        start_time = time.perf_counter_ns()
                        times_elapsed = []
                    elif byte == self._end_marker:
                        start_time = None
                        break
                    else:
                        if start_time is not None:
                            bytes_placehoder_list.append(byte)
                            times_elapsed.append(time.perf_counter_ns() - start_time)

                if len(times_elapsed) > 1:
                    time_from_last = [times_elapsed[i] - times_elapsed[i - 1] for i in range(1, len(times_elapsed))]
                    time_bin = np.mean(time_from_last)
                    times_all.append(times_elapsed)
                    time_bin_sigma = np.std(time_from_last)
                    time_bins.append(time_bin)
                    time_bins_sigma.append(time_bin_sigma)
                else:
                    logger.warning(f"Not enough data to calculate time bin for frame {frame + 1}.")
            
            # for frame in range(self.sync_frames):
            #     for i in range(len(times_elapsed)//self.sync_frames - 1):
            #         time_from_last = times_elapsed[i + 1 + frame * self.sync_bytes_per_frame] - times_elapsed[i + frame * self.sync_bytes_per_frame]
            #         time_bins.append(time_from_last)

            if time_bins:
                formatted_bins = [round(float(val), 2) for val in time_bins]
                logger.debug2(f"Bob: Time bins calculated: {formatted_bins}")
                logger.debug2(f"Bob: Time bins standard deviation: {time_bins_sigma}")
                self.average_time_bin = np.mean(time_bins)
                logger.info(f"Average time bin for all frames: {self.average_time_bin/1000:.6f} us")
                logger.info(f"Standard deviation of time bins: {np.mean(time_bins_sigma)/1000:.6f} us")
                return self.average_time_bin
            else:
                logger.error("No time bins calculated.")
                return None
        except Exception as e:
            logger.error(f"Error in receiving frames: {e}")
            
        logger.info("Synchronization received.")
        return self.average_time_bin

    def receive_qubits(self, connection: socket.socket):
        """Simulates receiving qubits with detection times with some loss."""
        logger.info("Receiving qubits from Alice...")
        placeholder = []
        try:
            for frame in range(self.num_frames):

                while True:
                    data = connection.recv(1)
                    # if not data:
                    #     break
                    
                    byte = int.from_bytes(data, 'big')

                    if byte == self._start_marker:
                        start_time = time.perf_counter_ns()
                        placeholder = []
                    elif byte == self._end_marker:
                        start_time = None
                        break
                    else:
                        if start_time is not None:
                            self.detected_qubits_bytes.append(byte)
                            self.detected_timestamps.append(time.perf_counter_ns() - start_time)
            
            logger.info("Transmission complete. Transform time of detections to indices.")
            for byte in self.detected_qubits_bytes:
                self.detected_bases.append(random.choice([0, 1]))
                qubit = Qubit2.from_byte(byte)
                self.detected_bits.append(qubit.measure(measurement_base=self.detected_bases[-1]))
            self.detected_idxs = []
            for frame in range(self.num_frames):
                for i in range(len(self.detected_timestamps)//self.num_frames):
                    qt = self.detected_timestamps[i + frame * self.bytes_per_frame]
                    idx = int((qt-self.average_time_bin/2) / self.average_time_bin) + frame * self.bytes_per_frame
                    logger.debug2("qt %d: %d -> %.2f -> %d", i + frame * self.bytes_per_frame, qt/1000, (qt-self.average_time_bin/2)/self.average_time_bin, int((qt-self.average_time_bin/2) / self.average_time_bin))
                    self.detected_idxs.append(idx)

        except Exception as e:
            print(f"Error in receiving qubits: {e}")

        logger.info(f"Qubits received: {len(self.detected_qubits_bytes)}")
        return self.detected_qubits_bytes
    
    def run_mock_bob_qubits(self, connection: socket.socket, wait_time: bool = True, fake_error: bool = False, fake_error_rate: float = 0.0):
        """Run Bob's part for a single thread to obtain part of the final key."""
        try:
            self.receive_qubits_mock(connection, wait_time, fake_error, fake_error_rate)
        except Exception as e:
            logger.error(f"Bob encountered an error in run_bob_qubits: {e}")
        except KeyboardInterrupt:
            logger.error("Stopped by user.")

    def receive_qubits_mock(self, connection: socket.socket, wait_time: bool = True, fake_error: bool = False, fake_error_rate: float = 0.0):
        """Simulates receiving qubits with detection times with some loss."""
        logger.info("Receiving qubits from Alice...")
        # self.time_sleep(time_sleep=100000)
        try:
            # Receive the list of the qubits bytes all at once
            # Receive start marker
            data = connection.recv(1)
            byte = int.from_bytes(data, 'big')
            if byte == self._start_marker:
                # Receive the serialized list of qubit bytes
                received_data = b''
                while True:
                    chunk = connection.recv(4096)  # Receive in chunks
                    received_data += chunk
                    try:
                        # Try to see if we've received complete data by deserializing
                        # Load the received data
                        [self.detected_qubits_bytes, self.average_time_bin] = pickle.loads(received_data)
                        logger.info(f"Received {len(self.detected_qubits_bytes)} qubits from Alice without loss")
                        break
                    except (pickle.PickleError, EOFError):
                        # If we haven't received complete data, continue
                        continue
                
                # Receive end marker
                data = connection.recv(1)
                byte = int.from_bytes(data, 'big')
                if byte != self._end_marker:
                    logger.warning("End marker not received properly")


                print(f"Average time bin: {self.average_time_bin}")
                # If wait_time is True, simulate processing time
                if wait_time:
                    self.time_sleep(self.num_qubits * self.average_time_bin)
            else:
                logger.warning("Start marker not received properly")

            logger.info("Transmission complete. Transform time of detections to indices.")

            # Simulate the fake error, if fake_error is True, change the detected qubits
            if fake_error and 0 < fake_error_rate < 1:
                num_error = int(fake_error_rate * len(self.detected_qubits_bytes))
                error_indices = random.sample(range(len(self.detected_qubits_bytes)), num_error)
                for i in error_indices:
                    self.detected_qubits_bytes[i] = random.choice([0, 1, 2, 3])

            for byte in self.detected_qubits_bytes:
                self.detected_bases.append(random.choice([0, 1]))
                qubit = Qubit2.from_byte(byte)
                self.detected_bits.append(qubit.measure(measurement_base=self.detected_bases[-1]))
            self.detected_idxs = []

            # Simulate the ideal detection of the time and indices
            for i in range(self.num_qubits):
                self.detected_timestamps.append((i+1) * self.average_time_bin)
            for frame in range(self.num_frames):
                for i in range(self.bytes_per_frame):
                    self.detected_idxs.append(i + frame * self.bytes_per_frame)

            # Simulate the loss of qubits
            if self.loss_rate > 0:
                num_loss = int(self.loss_rate * len(self.detected_qubits_bytes))
                loss_indices = random.sample(range(len(self.detected_qubits_bytes)), num_loss)
                self.detected_qubits_bytes = [qubit for i, qubit in enumerate(self.detected_qubits_bytes) if i not in loss_indices]
                self.detected_bits = [bit for i, bit in enumerate(self.detected_bits) if i not in loss_indices]
                self.detected_bases = [base for i, base in enumerate(self.detected_bases) if i not in loss_indices]
                self.detected_idxs = [idx for i, idx in enumerate(self.detected_idxs) if i not in loss_indices]
                self.detected_timestamps = [ts for i, ts in enumerate(self.detected_timestamps) if i not in loss_indices]


        except Exception as e:
            logger.error(f"Error in receiving qubits: {e}")

        logger.info(f"Qubits received: {len(self.detected_qubits_bytes)} with loss")
        return self.detected_qubits_bytes
    
    def reset(self):
        """Reset all the detected qubits and bits."""
        self.detected_idxs = []
        self.detected_bases = []
        self.detected_bits = []
        self.detected_qubits_bytes = []
        self.detected_timestamps = []
    
    def return_bob_reseted(self):
        """Return the reseted Bob object."""
        self.reset()
        return self
    


class BobThread():
    def __init__(self, role: Role, receive_queue: queue.Queue, thread_id: int, test_bool: bool = True, test_fraction: float = 0.1, error_threshold: float = 0.0, start_idx: int = 0):

        # Dealing with mandatory parameters
        self._role = role
        self._receive_queue = receive_queue
        self._thread_id = thread_id

        # Dealing with optional parameters
        self._test_bool = test_bool
        self._test_fraction = test_fraction
        self._error_threshold = error_threshold
        self._start_idx = start_idx

        # Bits Bases and Qubits for the role
        self._bases = [] # Not needed
        self._bits = [] # Not needed
        self._qubits = [] # Not needed
        self._qubits_bytes = [] # Not needed
        self._detected_timestamps = [] # Not needed
        self._detected_idxs = []
        self._detected_bases = []
        self._detected_bits = []
        self._detected_qubits_bytes = [] # Not needed
        self._length_detected_key = 0 # Not needed
        # self._common_indices = [] 
        # self._common_bits = []
        # self._common_test_indices = []
        # self._remaining_indices = []
        # self.final_key = []
        self.test_success_bool = True
        self.average_time_bin = None

        # Variables for bandwidth testing
        self.bob_payload_size = []
        self.bob_detected_idx_size = None
        self.bob_common_idx_size = None
        self.bob_test_bits_size = None

    def set_bits_bases_qubits_idxs(self, detected_bits: list[int], detected_bases: list[int], detected_qubits_bytes: list[int], detected_idxs: list[int], average_time_bin: float):
        """Set the detected bits, detected bases, detected qubits and detected indices for Bob."""
        if len(detected_bits) != len(detected_bases) or len(detected_bits) != len(detected_qubits_bytes) or len(detected_bits) != len(detected_idxs):
            logger.error("Bob: Detected bits, bases, qubits and indices have different lengths.")
            raise ValueError("Detected bits, bases, qubits and indices have different lengths.")
        if detected_bits:
            if not all(bit in [0, 1] for bit in detected_bits):
                logger.error("Bob: Detected bits are not binary.")
                raise ValueError("Detected bits are not binary.")
            self._detected_bits = detected_bits
        else:
            logger.error("Bob: Detected bits are empty.")
            raise ValueError("Detected bits are empty.")
        if detected_bases:
            if not all(base in [0, 1] for base in detected_bases):
                logger.error("Bob: Detected bases are not binary.")
                raise ValueError("Detected bases are not binary.")
            self._detected_bases = detected_bases
        else:
            logger.error("Bob: Detected bases are empty.")
            raise ValueError("Detected bases are empty.")
        if detected_qubits_bytes:
            if not all(qubit in [0, 1, 2, 3] for qubit in detected_qubits_bytes):
                logger.error("Bob: Detected qubits are not binary.")
                raise ValueError("Detected qubits are not binary.")
            self._detected_qubits_bytes = detected_qubits_bytes
        else:
            logger.error("Bob: Detected qubits are empty.")
            raise ValueError("Detected qubits are empty.")
        if detected_idxs:
            if not all(idx >= 0 for idx in detected_idxs):
                logger.error("Bob: Detected indices are negative.")
                raise ValueError("Detected indices are negative.")
            self._detected_idxs = detected_idxs
        else:
            logger.error("Bob: Detected indices are empty.")
            raise ValueError("Detected indices are empty.")
        if average_time_bin:
            self.average_time_bin = average_time_bin
        else:
            logger.error("Bob: Average time bin is empty.")
            raise ValueError("Average time bin is empty.")

    def run_bob_thread(self):
        """Run Bob's part for a single thread to obtain part of the final key."""
        try:
            self.send_detected_idx()
            self.receive_detected_bases()
            self.match_bases()
            self.send_common_indices()
            if self._test_bool:
                self.send_test_bits()
                self.receive_test_result()
            if self.test_success_bool:
                self.final_remaining_key()
            else:
                logger.warning(f"Thread id: {self._thread_id}: Test failed. Potential eavesdropping detected. Key exchange aborted.")

        except Exception as e:
            logger.error(f"Thread {self._thread_id}: Bob encountered an error in run_bob: {e}")
        except KeyboardInterrupt:
            logger.error(f"Thread {self._thread_id}: Stopped by user.")
        finally:
            logger.info(f"Thread {self._thread_id}: Finished.")

    def send_data(self, data: any, list_size_to_append: list = None):
        payload = pickle.dumps((self._thread_id, data))
        # Size of the payload
        print(f"Payload size: {len(payload)} bytes")
        if list_size_to_append:
            list_size_to_append.append(len(payload))
        else:
            self.bob_payload_size.append(len(payload))
        self._role.put_in_outbox(payload)

    def receive_data(self):
        """Receives data from the role's receive queue."""
        return self._receive_queue.get()

    
    def send_detected_idx(self):
        """Sends detected qubit indices to Alice."""
        logger.debug(f"Thread id: {self._thread_id}: Sending detected qubit indices...")
        # Putting on the outbox queue of the role the detected qubit indices
        self.send_data(self._detected_idxs)
        logger.info(f"Thread id: {self._thread_id}: Sent {len(self._detected_idxs)} detected qubit indices.")
        logger.debug2(f"Thread id: {self._thread_id}: Detec idx - {self._detected_idxs}")
        return self._detected_idxs
    
    def receive_detected_bases(self):
        """Receives bases corresponding to time bins from Alice."""
        logger.debug(f"Thread id: {self._thread_id}: Receiving bases from Alice...")
        # Getting out of the receive queue of the role thread Alice bases
        self._other_detected_bases = self.receive_data()
        logger.info(f"Thread id: {self._thread_id}: Received {len(self._other_detected_bases)} Alice bases.")
        logger.debug2(f"Thread id: {self._thread_id}: Alice bases - {self._other_detected_bases}")
        return self._other_detected_bases
    
    def match_bases(self):
        """Match alice basis in time_bins with bobs bases. Key shifting."""
        logger.info(f"Thread id: {self._thread_id}: Comparing bases and filtering bits...")
        logger.debug3(f"Thread id: {self._thread_id}: Alice bases in time bins: {self._other_detected_bases}")
        logger.debug3(f"Thread id: {self._thread_id}: Bob bases in time bins: {self._detected_bases}")
        self._common_indices = [i for i in range(len(self._other_detected_bases)) if self._other_detected_bases[i] == self._detected_bases[i]]
        self._common_bits = [self._detected_bits[i] for i in self._common_indices]
        logger.info(f"Thread id: {self._thread_id}: Key length of common bases: {len(self._common_bits)}")
        return self._common_bits

    def send_common_indices(self):
        """Sends common indices to Alice."""
        logger.debug(f"Thread id: {self._thread_id}: Sending common indices to Alice...")
        # Putting on the outbox queue of the role the common indices
        self.send_data(self._common_indices)
        logger.info(f"Thread id: {self._thread_id}: Sent {len(self._common_indices)} common indices.")
        logger.debug2(f"Thread id: {self._thread_id}: Matched common indices - {self._common_indices}")
        return self._common_indices

    def send_test_bits(self):
        """Sends part of the common bits to Alice to check for eavesdropping."""
        logger.debug(f"Thread id: {self._thread_id}: Sending fraction ({self._test_fraction}) of the common key bits and their indices for testing...")
        self._common_test_indices = random.sample(self._common_indices, int(self._test_fraction * len(self._common_indices)))
        self._common_test_bits = [self._detected_bits[i] for i in self._common_test_indices]
        # Putting on the outbox queue of the role the common test indices and bits
        self.send_data([self._common_test_indices, self._common_test_bits])
        logger.info(f"Thread id: {self._thread_id}: Sent {len(self._common_test_bits)} test bits.")
        logger.debug2(f"Thread id: {self._thread_id}: Common key fraction indices sent: {self._common_test_indices}")
        return self._common_test_bits
    
    def receive_test_result(self):
        """Receives test result from Alice."""
        logger.debug(f"Thread id: {self._thread_id}: Receiving test result from Alice...")
        # Getting out of the receive queue of the role thread Alice the test result
        [self.test_success_bool, self.failed_percentage] = self.receive_data()
        logger.info(f"Thread id: {self._thread_id}: Test success: {self.test_success_bool} with {self.failed_percentage:.2f}%")
        # Save the payload sizes
        self.bob_detected_idx_size = self.bob_payload_size[0]
        self.bob_common_idx_size = self.bob_payload_size[1]
        self.bob_test_bits_size = self.bob_payload_size[2]

        if not self.test_success_bool:
            logger.info("Thread id: %s: TEST FAILED: %.2f%% bits do not match. Potential eavesdropping detected. Ignore this key.", self._thread_id, self.failed_percentage)
            self.final_key = []
            self.final_key_length = 0
        return self.test_success_bool

    def final_remaining_key(self):
        """Forms the final bits after error checking removing the indices of testing."""
        self._remaining_indices = set(self._common_indices) - set(self._common_test_indices)
        logger.debug2("Remaining indices: %s", self._remaining_indices)
        logger.info("Thread id: %s: Forming final key with remaining %s bits (%s - %s)...", self._thread_id, len(self._remaining_indices), len(self._common_indices), len(self._common_test_indices))
        self.final_key = [self._detected_bits[i] for i in sorted(self._remaining_indices)]
        self.final_key_length = len(self.final_key)
        logger.debug(f"Thread id: {self._thread_id}: Final key: {self.final_key}")
        return self.final_key
    
    @staticmethod
    def return_bob_reseted(bob: 'BobThread'):
        """Return a reseted BobThread object."""
        return BobThread(role=bob._role, receive_queue=bob._receive_queue, thread_id=bob._thread_id, test_bool=bob._test_bool, test_fraction=bob._test_fraction, error_threshold=bob._error_threshold, start_idx=bob._start_idx)
    
    # Return a void BobThread object
    @staticmethod
    def return_bob_void():
        """Return a void BobThread object."""
        return BobThread(role=None, receive_queue=None, thread_id=-1)