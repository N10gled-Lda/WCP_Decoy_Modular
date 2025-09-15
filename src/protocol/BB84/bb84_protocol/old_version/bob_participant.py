
import socket
import time
import random
import numpy as np

from bb84_protocol.qubit2 import Qubit2
# from bb84_bytes import BB84ProtocolSimulation
from bb84_protocol.old_version.bb84_alice_bob_class import BB84ProtocolSimulationParticipant

# Configure logging
import logging
from logging_setup import setup_logger
# Setup logger
logger = setup_logger("BB84 Protocol Simulation Logger", logging.INFO)


class Bob(BB84ProtocolSimulationParticipant):
    """
    Bob-specific methods for the BB84 protocol.
    """
    def __init__(self, num_qubits: int = 100, qubit_delay_us: int = 1000, num_frames: int = 10, bytes_per_frame: int = 10, sync_interval: float = 1, sync_frames: int = 100, sync_bytes_per_frame: int = 50, loss_rate: float = 0.9, test_bool: bool = True, test_fraction: float = 0.1, error_threshold: float = 0.0):
        super().__init__(num_qubits=num_qubits, qubit_delay_us=qubit_delay_us, num_frames=num_frames, bytes_per_frame=bytes_per_frame, sync_interval=sync_interval, sync_frames=sync_frames, sync_bytes_per_frame=sync_bytes_per_frame, loss_rate=loss_rate, test_bool=test_bool, test_fraction=test_fraction, error_threshold=error_threshold)

    def run_bob(self, host: str='localhost', port: int=12345):
        """Run the Bob participant in the BB84 protocol."""

        max_retries = 4  # 4x5 seconds timeout
        retry_count = 0
        connection = None
        while retry_count <= max_retries:
            try:
                connection = self.setup_client(host, port)
                logger.info('Bob: Connected to Alice at %s:%s.', host, port)
                
                self.time_bin = self.receive_sync(connection)

                self.receive_qubits(connection)

                self.send_detected_idx(connection)

                self.receive_detected_bases(connection)

                self.match_bases()
                
                self.send_common_indices(connection)

                if self.test_bool:
                    self.send_test_bits(connection)
                    self.receive_test_result(connection)
                
                if self.test_success_bool:
                    self.final_remaining_key()
                    logger.info("Bob's final key (size:%s): %s", self.final_key_length, self.final_key)
                else:
                    logger.warning("Test failed. Potential eavesdropping. Aborting key establishment.")
                    return None
                return self.final_key
                    
                ## TODO - Given an interval perform multiple    
                    
            except ConnectionRefusedError:
                if retry_count >= max_retries:
                    logger.error("Timeout: Could not connect to Alice.")
                    return None
                logger.warning(f"Retry in 5s for Alice to start... ({retry_count + 1}/{max_retries})")
                time.sleep(5)
                retry_count += 1
            except KeyboardInterrupt:
                logger.error("Bob: Connection aborted.")
                return None
            except Exception as e:
                logger.error(f"Bob encountered an error: {e}")
                return None
            finally:
                if connection:
                    print("Closing connection...")
                    connection.close()

    @staticmethod
    def setup_client(host: str, port: int) -> socket.socket:
        """Setup client socket for Bob."""
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((host, port))
        return client

    def receive_sync(self, connection: socket.socket):
        """Receive synchronization from Alice."""
        logger.info("Receiving synchronization from Alice...")
        try:
            time_bins = []
            times_all = []
            time_bins_sigma = []
            bytes_placehoder_list = []
            times_elapsed = []
            for frame in range(self.sync_frames):
                
                while True:
                    data = connection.recv(1)
                    # if not data:
                    #     break

                    byte = int.from_bytes(data, 'big')

                    if byte == self.start_marker:
                        start_time = time.perf_counter_ns()
                        times_elapsed = []
                    elif byte == self.end_marker:
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
                average_time_bin = np.mean(time_bins)
                logger.info(f"Average time bin for all frames: {average_time_bin/1000:.6f} us")
                logger.info(f"Standard deviation of time bins: {np.mean(time_bins_sigma)/1000:.6f} us")
                return average_time_bin
            else:
                logger.error("No time bins calculated.")
                return None
        except Exception as e:
            logger.error(f"Error in receiving frames: {e}")
            
        logger.info("Synchronization received.")
        return average_time_bin

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

                    if byte == self.start_marker:
                        start_time = time.perf_counter_ns()
                        placeholder = []
                    elif byte == self.end_marker:
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
                    idx = int((qt-self.time_bin/2) / self.time_bin) + frame * self.bytes_per_frame
                    logger.debug2("qt %d: %d -> %.2f -> %d", i + frame * self.bytes_per_frame, qt/1000, (qt-self.time_bin/2)/self.time_bin, int((qt-self.time_bin/2) / self.time_bin))
                    self.detected_idxs.append(idx)

        except Exception as e:
            print(f"Error in receiving qubits: {e}")

        logger.info(f"Qubits received: {len(self.detected_qubits_bytes)}")
        return self.detected_qubits_bytes
    
    def send_detected_idx(self, connection: socket.socket):
        """Sends detected qubit indices to Alice."""
        logger.info("Sending detected qubit indices to Alice...")
        self.classical_channel.send_data(self.detected_idxs, connection)
        logger.info("Detected qubit indices sent.")
        logger.debug2("Detected qubit indices sent: %s", self.detected_idxs)
        return self.detected_idxs
    
    def receive_detected_bases(self, connection: socket.socket):
        """Receives bases corresponding to time bins from Alice."""
        logger.info("Receiving Alice bases in time bins...")
        self.other_detected_bases = self.classical_channel.receive_data(connection)
        logger.info("Alice Bases in time bins received.")
        # self.other_detected_bases = [self.bases_in_time_bins[time_stamp] for time_stamp in self.detected_timestamps]
        # logger.debug2("Alice Bases and Times in time bins: %s", self.bases_in_time_bins)
        return self.other_detected_bases
    
    def match_bases(self):
        """Match alice basis in time_bins with bobs bases. Key shifting."""
        logger.info("Comparing bases and filtering bits...")
        logger.debug("Alice bases in time bins: %s", self.other_detected_bases)
        logger.debug("Bob bases in time bins: %s", self.detected_bases)
        self.common_indices = [i for i in range(len(self.other_detected_bases)) if self.other_detected_bases[i] == self.detected_bases[i]]
        self.common_bits = [self.detected_bits[i] for i in self.common_indices]
        logger.info("Key length of common bases: %s", len(self.common_bits))
        return self.common_bits

    def send_common_indices(self, connection: socket.socket):
        """Sends common indices to Alice."""
        logger.info("Sending common indices...")
        self.classical_channel.send_data(self.common_indices, connection)
        logger.debug("Common indices sent: %s", self.common_indices)
        logger.debug2("Bob bits before common indices: %s", self.detected_bits)
        logger.debug("Bob bits for the common indices: %s", self.common_bits)
        return self.common_indices

    def send_test_bits(self, connection: socket.socket):
        """Sends part of the common bits to Alice to check for eavesdropping."""
        logger.info("Sending fraction (%s) of the common key bits and their indices for testing...", self.test_fraction)
        self.common_test_indices = random.sample(self.common_indices, int(self.test_fraction * len(self.common_indices)))
        self.common_test_bits = [self.detected_bits[i] for i in self.common_test_indices]
        self.classical_channel.send_data((self.common_test_indices, self.common_test_bits), connection)
        logger.info("Test bits and indices sent. Number of test bits: %s", len(self.common_test_bits))
        logger.debug("Common key fraction indices sent: %s", self.common_test_indices)
        logger.debug("Common key fraction bits sent: %s", self.common_test_bits)
        return self.common_test_bits
    
    def receive_test_result(self, connection: socket.socket):
        logger.info("Received test result from Alice...")
        [self.test_success_bool, self.failed_percentage] = self.classical_channel.receive_data(connection)
        # self.failed_percentage = self.classical_channel.receive_data(connection)
        logger.info("Test success: %s with %.2f%%", self.test_success_bool, self.failed_percentage)
        return self.test_success_bool
    