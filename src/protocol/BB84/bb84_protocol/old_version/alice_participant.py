
import socket
import time
import random
import pickle
import numpy as np

from bb84_protocol.qubit2 import Qubit2
# from bb84_bytes import BB84ProtocolSimulation
from bb84_protocol.old_version.bb84_alice_bob_class import BB84ProtocolSimulationParticipant

# Configure logging
import logging
from logging_setup import setup_logger
# Setup logger
logger = setup_logger("BB84 Protocol Simulation Logger", logging.INFO)


class Alice(BB84ProtocolSimulationParticipant):
    """
    Alice-specific methods for the BB84 protocol.
    """
    def __init__(self, num_qubits: int = 100, qubit_delay_us: int = 1000, num_frames: int = 10, bytes_per_frame: int = 10, sync_interval: float = 1, sync_frames: int = 100, sync_bytes_per_frame: int = 50, loss_rate: float = 0.9, test_bool: bool = True, test_fraction: float = 0.1, error_threshold: float = 0.0):
        super().__init__(num_qubits=num_qubits, qubit_delay_us=qubit_delay_us, num_frames=num_frames, bytes_per_frame=bytes_per_frame, sync_interval=sync_interval, sync_frames=sync_frames, sync_bytes_per_frame=sync_bytes_per_frame, loss_rate=loss_rate, test_bool=test_bool, test_fraction=test_fraction, error_threshold=error_threshold)
    
    def run_alice(self, host: str='localhost', port: int=12345):
        """Run Alice's part of the BB84 protocol."""
        
        try:
            server = self.setup_server(host, port)
            logger.info("Alice: Waiting for Bob to connect...")
            connection, addr = server.accept()
            with connection:
                logger.info(f"Alice: Connected by {addr}")

                self.perform_sync(connection)

                self.create_bit_base_qubits()

                self.send_qubits(connection)

                self.receive_detected_idx(connection)

                self.send_detected_bases(connection)
                
                self.receive_common_indices(connection)

                if self.test_bool:
                    self.receive_test_bits(connection)
                    self.test_eavesdropping(connection)

                if self.test_success_bool:
                    self.final_remaining_key()
                    logger.info("Alice's final key (size:%s): %s", self.final_key_length, self.final_key)
                else:
                    logger.warning("Test failed. Potential eavesdropping detected. Key exchange aborted.")
                
                ## TODO - Given an interval perform multiple synchronizations

        except Exception as e:
            print(f"Alice encountered an error: {e}")
        finally:
            print("Alice: Closing connection...")
            server.close()

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

    def perform_sync(self, connection: socket.socket):
        """Perform clock synchronization with Bob."""
        logger.info("Sending synchronization to Bob...")
        try:
            random_bytes = [random.randint(0, 3) for _ in range(self.sync_frames * self.sync_bytes_per_frame)]
            for frame in range(self.sync_frames):
                # Send start marker
                connection.sendall(self.start_marker.to_bytes(1, 'big'))
                self.time_sleep(time_sleep=self.sync_delay)

                # Simulate sending sync bytes
                for i in range(frame * self.sync_bytes_per_frame, (frame + 1) * self.sync_bytes_per_frame):
                    if i >= len(random_bytes):
                        break
                    connection.sendall(random_bytes[i].to_bytes(1, 'big'))
                    self.time_sleep(time_sleep=self.sync_delay)

                # Send end marker
                connection.sendall(self.end_marker.to_bytes(1, 'big'))
        except Exception as e:
            logger.error(f"Error in sending frames: {e}")
        
    def create_bit_base_qubits(self):
        """Create random bits and bases for qubits."""

        self.bits = self.generate_random_bases_bits(number=self.num_frames * self.bytes_per_frame)
        self.bases = self.generate_random_bases_bits(number=self.num_frames * self.bytes_per_frame)
        # self.qubits = [Qubit2(bit, base) for bit, base in zip(self.bits, self.bases)]
        # self.qubits_bytes = [qubit.get_byte() for qubit in self.qubits]
        self.qubits_bytes = [Qubit2.bitbase_to_byte(bit, base) for bit, base in zip(self.bits, self.bases)]

    def send_qubits(self, connection: socket.socket):
        """Send qubits to Bob similar to sycn and saves the idx of each."""
        logger.info("Sending qubits to Bob...")
        self.time_sleep(time_sleep=100000)
        times = []
        times_from_last = []
        try:        
            for frame in range(self.num_frames):
                # Send start marker
                connection.sendall(self.start_marker.to_bytes(1, 'big'))
                self.time_sleep(time_sleep=self.qubit_delay_us)

                # Send qubits bytes
                for i in range(frame * self.bytes_per_frame, (frame + 1) * self.bytes_per_frame):
                    if i >= self.num_qubits:
                        break
                    # times.append(time.perf_counter_ns())
                    connection.sendall(self.qubits_bytes[i].to_bytes(1, 'big'))
                    self.time_sleep(time_sleep=self.qubit_delay_us)

                # Send end marker
                times.append(time.perf_counter_ns())
                connection.sendall(self.end_marker.to_bytes(1, 'big'))

            #     for i in range(frame * self.bytes_per_frame, (frame + 1) * self.bytes_per_frame):
            #         time_from_last = times[i+1] - times[i] if i > 0 else 0
            #         times_from_last.append(time_from_last)
            
            # print("Times from last mean: ", np.mean(times_from_last))
            # print("Times from last sigma: ", np.std(times_from_last))
        except Exception as e:
            logger.error(f"Error in sending qubits: {e}")
            
        logger.info("Alice: All qubits sent.")

    def receive_detected_idx(self, connection: socket.socket):
        """Receive time bins index of the qubit from Bob."""
        logger.info("Receiving detected time bins indices...")
        self.detected_idxs = self.classical_channel.receive_data(connection)
        logger.info("%s Detected time bins indices received.", len(self.detected_idxs))

    def send_detected_bases(self, connection: socket.socket):
        """Find the bases and bits for the detected times and sends those corresponding bases to Bob."""

        logger.info("Sending bases in Bob detected index...")

        for idx in self.detected_idxs:
            if 0 <= idx < len(self.bases):
                self.detected_bases.append(self.bases[idx])
                self.detected_bits.append(self.bits[idx])
                # self.detected_qubits.append(self.qubits[idx])
            else:
                logger.warning("Index out of range: %s", idx)
                self.detected_bases.append(None)
                self.detected_bits.append(None)
                # self.detected_qubits.append(None)

        self.classical_channel.send_data(self.detected_bases, connection)
        logger.info("%s Bases sent.", len(self.detected_bases))
        logger.debug("Alice Bases in time bins: %s", self.detected_bases)
        return self.detected_bases

    def receive_common_indices(self, connection: socket.socket):
        """Receives indices of the matching/common bases from Bob."""
        logger.info("Receiving common indices...")
        self.common_indices = self.classical_channel.receive_data(connection)
        logger.info("%s Common indices received.", len(self.common_indices))
        logger.debug("Common indices: %s", self.common_indices)
        self.common_bits = [self.detected_bits[i] for i in self.common_indices]
        logger.debug2("Alice Bits before common key indices: %s", self.detected_bits)
        logger.debug("Alice Bits for the common key indices: %s", self.common_bits)
        return self.common_indices
    
    def receive_test_bits(self, connection: socket.socket) -> bool:
        """Receives test bits and corresponding indices from Bob and checks for eavesdropping."""
        logger.info("Receiving fraction (%s) of the common key bits and their indices for testing...", self.test_fraction)
        self.common_test_indices, self.bob_test_bits = self.classical_channel.receive_data(connection)
        logger.info("Test %s bits and indices received.", len(self.common_test_indices))
        logger.info("Confirming test bits, if Bob's bits match Alice's bits...")
        logger.debug2("Bob test indices: %s", self.common_test_indices)
        logger.debug("Bob test bits: %s", self.bob_test_bits)
        self.alice_test_bits = [self.detected_bits[i] for i in self.common_test_indices]
        logger.debug("Alice test bits: %s", self.alice_test_bits)

    def test_eavesdropping(self, connection: socket.socket) -> bool:
        assert len(self.alice_test_bits) == len(self.bob_test_bits), "Test bits length mismatch."
        self.bits_check = [self.alice_test_bits[i] == self.bob_test_bits[i] for i in range(len(self.bob_test_bits))]
        logger.debug("Test bits check: %s", self.bits_check)
        self.failed_percentage = self.bits_check.count(False) / len(self.bits_check) * 100
        self.test_success_bool = self.failed_percentage/100 < self.error_threshold
        logger.info("Test result: %s with %.2f%%", self.test_success_bool, self.failed_percentage)
        self.classical_channel.send_data([self.test_success_bool,self.failed_percentage], connection)
        # self.classical_channel.send_data(self.failed_percentage, connection)
        if not self.test_success_bool:
            logger.info("Test failed. %.2f%% bits do not match.", self.failed_percentage)
        return self.test_success_bool