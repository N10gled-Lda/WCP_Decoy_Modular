import pickle
import socket
import time
import random
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

class AliceQubits():
    """
    Alice-specific methods for the BB84 protocol.
    """
    def __init__(self, num_qubits: int = 100, qubit_delay_us: int = 1000, num_frames: int = 10, bytes_per_frame: int = 10, sync_frames: int = 100, sync_bytes_per_frame: int = 50, loss_rate: float = 0.9):
        self.sync_delay = qubit_delay_us
        self.sync_frames = sync_frames
        self.sync_bytes_per_frame = sync_bytes_per_frame
        self.num_qubits = num_qubits
        self.bytes_per_frame = bytes_per_frame
        self.num_frames = num_frames
        self.qubit_delay_us = qubit_delay_us
        self.loss_rate = loss_rate
        # self.classical_channel = ClassicalChannel()

        # Bits Bases and Qubits for each role
        self.bases = []
        self.bits = []
        self.qubits = []
        self.qubits_bytes = []

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
    def setup_server(host: str, port: int, time: float = 10.0) -> socket.socket:
        """
        Setup server socket for Alice for a certain time.
            - host: str: Hostname or IP address
            - port: int: Port number
            - time: float: Timeout in seconds
        """
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(time)
        server.bind((host, port))
        server.listen(1)
        return server

    def run_alice_qubits(self, connection: socket.socket):
        """Run Alice's qubit operations."""

        logger.info("Running Alice's qubit operations...")

        self.perform_sync(connection)
        self.create_bit_base_qubits()
        self.send_qubits(connection)
        

    def perform_sync(self, connection: socket.socket):
        """Perform clock synchronization with Bob."""
        logger.info(f"Sending synchronization...")
        try:
            random_bytes = [random.randint(0, 3) for _ in range(self.sync_frames * self.sync_bytes_per_frame)]
            for frame in range(self.sync_frames):
                # Send start marker
                connection.sendall(self._start_marker.to_bytes(1, 'big'))
                # self._send_with_thread_id(connection, self._start_marker.to_bytes(1, 'big'))
                self.time_sleep(time_sleep=self.sync_delay)

                # Simulate sending sync bytes
                for i in range(frame * self.sync_bytes_per_frame, (frame + 1) * self.sync_bytes_per_frame):
                    if i >= len(random_bytes):
                        break
                    connection.sendall(random_bytes[i].to_bytes(1, 'big'))
                    # self._send_with_thread_id(connection, random_bytes[i].to_bytes(1, 'big'))
                    self.time_sleep(time_sleep=self.sync_delay)

                # Send end marker
                connection.sendall(self._end_marker.to_bytes(1, 'big'))
                # self._send_with_thread_id(connection, self._end_marker.to_bytes(1, 'big'))
        except Exception as e:
            logger.error(f"Error in sending frames: {e}")
        
    def create_bit_base_qubits(self):
        """Create random bits and bases for qubits."""
        logger.info(f"Creating bits and bases...")

        self.bits = self.generate_random_bases_bits(number=self.num_frames * self.bytes_per_frame)
        self.bases = self.generate_random_bases_bits(number=self.num_frames * self.bytes_per_frame)
        # self.qubits = [Qubit2(bit, base) for bit, base in zip(self.bits, self.bases)]
        # self.qubits_bytes = [qubit.get_byte() for qubit in self.qubits]
        self.qubits_bytes = [Qubit2.bitbase_to_byte(bit, base) for bit, base in zip(self.bits, self.bases)]

    def send_qubits(self, connection: socket.socket):
        """Send qubits to Bob similar to sycn and saves the idx of each."""
        logger.info(f"Sending qubits...")
        self.time_sleep(time_sleep=100000)
        times = []
        times_from_last = []
        try:        
            for frame in range(self.num_frames):
                # Send start marker
                connection.sendall(self._start_marker.to_bytes(1, 'big'))
                # self._send_with_thread_id(connection, self._start_marker.to_bytes(1, 'big'))
                self.time_sleep(time_sleep=self.qubit_delay_us)

                # Send qubits bytes
                for i in range(frame * self.bytes_per_frame, (frame + 1) * self.bytes_per_frame):
                    if i >= self.num_qubits:
                        break
                    # times.append(time.perf_counter_ns())
                    connection.sendall(self.qubits_bytes[i].to_bytes(1, 'big'))
                    # self._send_with_thread_id(connection, self.qubits_bytes[i].to_bytes(1, 'big'))
                    self.time_sleep(time_sleep=self.qubit_delay_us)

                # Send end marker
                times.append(time.perf_counter_ns())
                connection.sendall(self._end_marker.to_bytes(1, 'big'))
                # self._send_with_thread_id(connection, self._end_marker.to_bytes(1, 'big'))

            #     for i in range(frame * self.bytes_per_frame, (frame + 1) * self.bytes_per_frame):
            #         time_from_last = times[i+1] - times[i] if i > 0 else 0
            #         times_from_last.append(time_from_last)
            
            # print("Times from last mean: ", np.mean(times_from_last))
            # print("Times from last sigma: ", np.std(times_from_last))
        except Exception as e:
            logger.error(f"Error in sending qubits: {e}")
            
        logger.info("Alice: All qubits sent.")

    def run_mock_alice_qubits(self, connection: socket.socket, wait_time: bool = True):
        """Run Alice's qubit operations for testing. Sends all qubits at once and sleeps for the corresponded time."""
        logger.info("Running Mock Alice's qubit operations for testing...")
        self.create_bit_base_qubits()
        self.send_qubits_mock(connection, wait_time=wait_time)

    def send_qubits_mock(self, connection: socket.socket, wait_time: bool = True):
        """Send qubits to Bob similar to sycn and saves the idx of each."""
        logger.info(f"Sending qubits...")
        self.time_sleep(time_sleep=100000)
        try:
            # Send start marker
            connection.sendall(self._start_marker.to_bytes(1, 'big'))
            # Send the list of qubits bytes at once
            connection.sendall(pickle.dumps([self.qubits_bytes, self.qubit_delay_us]))
            # Send end marker
            connection.sendall(self._end_marker.to_bytes(1, 'big'))
            # Wait for the corresponded time
            if wait_time:
                self.time_sleep(time_sleep=self.qubit_delay_us * self.num_qubits)
            
        except Exception as e:
            logger.error(f"Error in sending qubits: {e}")
            
        logger.info(f"Alice: All {len(self.qubits_bytes)} qubits sent.")

    def reset(self):
        """Reset all the qubits and bits."""
        self.bits = []
        self.bases = []
        self.qubits = []
        self.qubits_bytes = []
    
    def return_alice_reseted(self):
        """Return the reseted Alice object."""
        self.reset()
        return self
    
class AliceThread():
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
        self._bases = []
        self._bits = []
        self._qubits = []
        self._qubits_bytes = []
        self._detected_idxs = []
        self._detected_bases = []
        self._detected_bits = []
        self._detected_qubits_bytes = []
        self._length_detected_key = 0
        # self._common_indices = []
        # self._common_bits = []
        # self._common_test_indices = []
        # self._remaining_indices = []
        # self.final_key = []
        self.test_success_bool = True

        # Variables for bandwidth test
        self.alice_payload_size = []
        self.alice_detected_bases_size = None
        self.alice_test_result_size = None

    def set_bits_bases_qubits(self, bits: list[int], bases: list[int], qubits_bytes: list[bytes]):
        """Set the bits, bases and qubits for Alice."""
        if len(bits) != len(bases) or len(bases) != len(qubits_bytes):
            raise ValueError("Length of bits, bases and qubits should be the same.")
        if bits:
            if not all(bit in [0, 1] for bit in bits):
                raise ValueError("Bits should only contain 0s and 1s.")
            self._bits = bits
        else:
            raise ValueError("Bits should not be empty.")
        if bases:
            if not all(base in [0, 1] for base in bases):
                raise ValueError("Bases should only contain 0s and 1s.")
            self._bases = bases
        else:
            raise ValueError("Bases should not be empty.")
        if qubits_bytes:
            if not all(qubits_byte in [0, 1, 2, 3] for qubits_byte in qubits_bytes):
                raise ValueError("Qubits should only contain 0s, 1s, 2s and 3s.")
            self._qubits_bytes = qubits_bytes
        else:
            raise ValueError("Qubits should not be empty.")
    

    def run_alice_thread(self):
        """Run Alice's part for a single thread to obtain part of the final key."""
        try:
            self.receive_detected_idx()
            self.send_detected_bases()
            self.receive_common_indices()
            if self._test_bool:
                self.receive_test_bits()
                self.test_eavesdropping()
            if self.test_success_bool:
                self.final_remaining_key()
                # logger.debug(f"Thread id: {self._thread_id} - Final key (size: {self.final_key_length}): {self.final_key}")
            else:
                logger.warning(f"Thread id: {self._thread_id}: Test failed. Potential eavesdropping detected. Key exchange aborted.")
        except Exception as e:
            logger.error(f"Thread id: {self._thread_id}: Alice encountered an error in run_alice: {e}")
        except KeyboardInterrupt:
            logger.error(f"Thread id: {self._thread_id}: Stopped by user.")
        finally:
            logger.info(f"Thread id: {self._thread_id}: Finished.")

    def send_data(self, data: any, list_size_to_append: list = None):
        payload = pickle.dumps((self._thread_id, data))
        # Size of the payload
        print(f"Payload size: {len(payload)} bytes")
        if list_size_to_append:
            list_size_to_append.append(len(payload))
        else:
            self.alice_payload_size.append(len(payload))
        self._role.put_in_outbox(payload)

    def receive_data(self):
        """Receives data from the role's receive queue."""
        return self._receive_queue.get()

    def receive_detected_idx(self):
        """Receive time bins index of the qubit from Bob."""
        logger.debug(f"Thread id: {self._thread_id}: Receiving detected time bins indices...")
        # Getting out of the receive queue of the role thread the detected indices
        self._detected_idxs = self.receive_data()
        logger.info(f"Thread id: {self._thread_id}: Received {len(self._detected_idxs)} detected indices.")
        logger.debug2(f"Thread id: {self._thread_id}:Detec idx: {self._detected_idxs}")
        return self._detected_idxs
    
    def send_detected_bases(self):
        """Find the bases and bits for the detected times and sends those corresponding bases to Bob."""
        logger.debug2(f"Thread id: {self._thread_id}: Shifting indices given the start index of the thread: {self._start_idx}")
        # Shift the indices to get the bases and bits detected correctly given the start index of the thread
        for idx in self._detected_idxs:
            shifted_idx = idx - self._start_idx
            # print("Index: ", idx, " Start index: ", self._start_idx, " Shifted index: ", shifted_idx)
            if 0 <= shifted_idx < len(self._bases):
                self._detected_bases.append(self._bases[shifted_idx])
                self._detected_bits.append(self._bits[shifted_idx])
                # self.detected_qubits.append(self.qubits[idx])
                self._detected_qubits_bytes.append(self._qubits_bytes[shifted_idx])
            else:
                logger.warning("Index out of range: %s", idx)
                self._detected_bases.append(None)
                self._detected_bits.append(None)
                # self.detected_qubits.append(None)
                self._detected_qubits_bytes.append(None)

        logger.debug("Sending bases in Bob detected index...")
        # Puting on the outbox of the role the detected bases
        self.send_data(self._detected_bases)
        logger.info(f"Thread id: {self._thread_id}: Sent {len(self._detected_bases)} detected bases.")
        logger.debug2(f"Thread id: {self._thread_id}: Bases sent: {self._detected_bases}")
        return self._detected_bases
    
    def receive_common_indices(self):
        """Receives indices of the matching/common bases from Bob."""
        logger.debug(f"Thread id: {self._thread_id}: Receiving common indices...")
        # Getting out of the receive queue of the role thread the common indices
        self._common_indices = self.receive_data()
        logger.info(f"Thread id: {self._thread_id}: Received {len(self._common_indices)} common indices.")
        logger.debug2(f"Thread id: {self._thread_id}: Common indices: {self._common_indices}")

        # Getting the common bits from the common indices and the detected bits
        self._common_bits = [self._detected_bits[i] for i in self._common_indices]
        logger.debug2(f"Thread id: {self._thread_id}: Alice Bits before common key indices: {self._detected_bits}")
        logger.debug(f"Thread id: {self._thread_id}: Alice Bits for the common key indices: {self._common_bits}")
        return self._common_indices
    
    def receive_test_bits(self) -> bool:
        """Receives test bits and corresponding indices from Bob and checks for eavesdropping."""
        logger.debug("Receiving fraction (%s) of the common key bits and their indices for testing...", self._test_fraction)
        # Getting out of the receive queue of the role thread the test bits and indices
        [self._common_test_indices, self._bob_test_bits] = self.receive_data()
        logger.info(f"Thread id: {self._thread_id}: Received {len(self._common_test_indices)} bits and indices for testing.")
        logger.debug2(f"Thread id: {self._thread_id}: Bob test indices: {self._common_test_indices}")

        self._alice_test_bits = [self._detected_bits[i] for i in self._common_test_indices]
        logger.debug2(f"Thread id: {self._thread_id}: Bob test bits: {self._bob_test_bits}")
        logger.debug2(f"Thread id: {self._thread_id}: Alice test bits: {self._alice_test_bits}")

    def test_eavesdropping(self) -> bool:
        assert len(self._alice_test_bits) == len(self._bob_test_bits), "Test bits length mismatch."
        
        logger.debug(f"Thread id: {self._thread_id}:Confirming test bits, if Bob's bits match Alice's bits...")
        self._bits_check = [self._alice_test_bits[i] == self._bob_test_bits[i] for i in range(len(self._bob_test_bits))]
        logger.debug("Test bits check: %s", self._bits_check)
        self.failed_percentage = self._bits_check.count(False) / len(self._bits_check) * 100
        self.test_success_bool = self.failed_percentage/100 < self._error_threshold
        logger.info(f"Thread id: {self._thread_id}: Test success status: {self.test_success_bool} with failed percentage: {self.failed_percentage}")

        # Putting the test result on the outbox of the role
        self.send_data([self.test_success_bool, self.failed_percentage])

        # Save the payload sizes
        self.alice_detected_bases_size = self.alice_payload_size[0]
        self.alice_test_result_size = self.alice_payload_size[1]
        
        # self.classical_channel.send_data(self.failed_percentage, connection)
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
        logger.info(f"Thread id: {self._thread_id}: Final key length: {self.final_key_length}")
        logger.debug(f"Thread id: {self._thread_id}: Final key: {self.final_key}")
        return self.final_key
    
    @staticmethod
    def return_alice_reseted(alice: 'AliceThread'):
        """Return a reseted Alice object."""
        logger.debug(f"Thread id: {alice._thread_id}: Alice payload size recorded: {len(alice.alice_payload_size)}")
        return AliceThread(role=alice._role, receive_queue=alice._receive_queue, thread_id=alice._thread_id, test_bool=alice._test_bool, test_fraction=alice._test_fraction, error_threshold=alice._error_threshold, start_idx=alice._start_idx)
    
    # Return a void AliceThread object
    @staticmethod
    def return_void_alice():
        return AliceThread(role=None, receive_queue=None, thread_id=-1)
    