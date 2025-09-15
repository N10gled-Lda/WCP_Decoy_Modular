import random
import time

from bb84_protocol.old_version.ch_classical import ClassicalChannel
from bb84_protocol.old_version.ch_quantum import QuantumChannel2 # not used
from bb84_protocol.old_version.synchronization import SynchronizationChannel # not used

# Configure logging
import logging
from logging_setup import setup_logger, DEBUG_L1, DEBUG_L2, DEBUG_L3
# Setup logger
logger = setup_logger("BB84 Protocol Simulation Logger", logging.INFO)

class BB84ProtocolSimulationParticipant:
    def __init__(self, num_qubits: int = 100, qubit_delay_us: int = 1000, num_frames: int = 10, bytes_per_frame: int = 10, sync_interval: float = 1, sync_frames: int = 100, sync_bytes_per_frame: int = 50, loss_rate: float = 0.9, test_bool: bool = True, test_fraction: float = 0.1, error_threshold: float = 0.0):
        self.sync_delay = qubit_delay_us
        self.sync_interval = sync_interval
        self.sync_frames = sync_frames
        self.sync_bytes_per_frame = sync_bytes_per_frame
        self.num_qubits = num_qubits
        self.bytes_per_frame = bytes_per_frame
        self.num_frames = num_frames
        self.qubit_delay_us = qubit_delay_us
        self.loss_rate = loss_rate
        self.test_bool = test_bool
        self.test_fraction = test_fraction
        self.error_threshold = error_threshold

        self.quantum_channel = QuantumChannel2(loss_rate)
        self.synchronization_channel = SynchronizationChannel(qubit_delay_us,sync_bytes_per_frame)
        self.classical_channel = ClassicalChannel()

        # Bits Bases and Qubits for each participant
        self.bases = []
        self.bits = []
        self.qubits = []
        self.qubits_bytes = []
        self.detected_idxs = []
        self.detected_bases = []
        self.detected_bits = []
        self.detected_qubits_bytes = []
        self.detected_timestamps = []
        self.length_detected_key = 0
        self.common_indices = []
        self.common_bits = []
        self.common_test_indices = []
        self.remaining_indices = []
        self.final_key = []
        self.test_success_bool = True

        self.start_marker = 100
        self.end_marker = 50
    
    @staticmethod
    def generate_random_bases_bits(number: int):
        """Generate random bits or bases."""
        return [random.choice([0, 1]) for _ in range(number)]
    
    @staticmethod
    def time_sleep(time_sleep: int):
        start_sleep_time = time.perf_counter_ns()
        while time.perf_counter_ns() - start_sleep_time < time_sleep * 1000:
            pass

    def final_remaining_key(self):
        """Forms the final bits after error checking removing the indices of testing."""
        self.remaining_indices = set(self.common_indices) - set(self.common_test_indices)
        logger.debug2("Remaining indices: %s", self.remaining_indices)
        logger.info("Forming final key with remaining %s bits (%s - %s)...", len(self.remaining_indices), len(self.common_indices), len(self.common_test_indices))
        self.final_key = [self.detected_bits[i] for i in sorted(self.remaining_indices)]
        self.final_key_length = len(self.final_key)
        return self.final_key