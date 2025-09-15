import unittest
from unittest.mock import MagicMock, patch
import socket
import queue

from bb84_protocol.alice_side_thread_ccc import AliceQubits, AliceThread
from bb84_protocol.bob_side_thread_ccc import BobQubits, BobThread
from BB84.bb84_separte_bob import partition_data

class TestBB84Protocol(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.NUM_QUBITS = 100
        self.NUM_FRAMES = 10
        self.BYTES_PER_FRAME = 10
        self.SYNC_FRAMES = 2
        self.SYNC_BYTES = 10
        self.QUBIT_DELAY_US = 100
        self.LOSS_RATE = 0.0

        self.alice_TEST = AliceQubits(num_qubits=self.NUM_QUBITS, num_frames=self.NUM_FRAMES, bytes_per_frame=self.BYTES_PER_FRAME, sync_frames=self.SYNC_FRAMES, sync_bytes_per_frame=self.SYNC_BYTES, qubit_delay_us=self.QUBIT_DELAY_US, loss_rate=self.LOSS_RATE)
        self.bob_TEST = BobQubits(num_qubits=self.NUM_QUBITS, num_frames=self.NUM_FRAMES, bytes_per_frame=self.BYTES_PER_FRAME, sync_frames=self.SYNC_FRAMES, sync_bytes_per_frame=self.SYNC_BYTES)
        # COMLPLET WITH INPUTS
        # self.alice_thread_TEST = AliceThread()
        # self.bob_thread_TEST = BobThread()

    # ------- AliceQubits Tests -------
    def test_generate_random_bases_bits(self):
        """Test the generation of random bits and bases."""
        num_bits = 100
        random_bits = AliceQubits.generate_random_bases_bits(num_bits)
        self.assertEqual(len(random_bits), num_bits)
        self.assertTrue(all(bit in [0, 1] for bit in random_bits))

    def test_server_setup(self):
        """Test server setup for Alice."""
        server = AliceQubits.setup_server("localhost", 12345)
        self.assertIsNotNone(server)
        server.close()

    # @patch('socket.socket')
    # def test_perform_sync(self, mock_socket):
    #     """Test synchronization logic for Alice."""
    #     mock_connection = MagicMock()
    #     alice = AliceQubits(num_qubits=100, sync_frames=2)
    #     alice.perform_sync(mock_connection)
    #     self.assertTrue(mock_connection.sendall.called)
    # def test_sync_performance(self):
    #     """Test synchronization logic between Alice and Bob."""
    #     alice = AliceQubits(num_qubits=100, sync_frames=5)
    #     with self.assertLogs('BB84 Simulation Log', level='INFO') as cm:
    #         # Assuming you mock a connection or use a test socket
    #         mock_connection = MagicMock()
    #         alice.perform_sync(mock_connection)
    #     self.assertIn('Sending synchronization...', cm.output[0])
    @patch('socket.socket')
    def test_perform_sync_comprehensive(self, mock_socket):
        """Test synchronization logic for Alice comprehensively."""
        # Setup
        mock_connection = MagicMock()
        alice = self.alice_TEST.return_alice_reseted()
        # Test execution and logging
        with self.assertLogs('BB84 Simulation Log', level='INFO') as cm:
            alice.perform_sync(mock_connection)
        
        # Verify logging
        self.assertIn('Sending synchronization...', cm.output[0])
        # Verify socket communication
        self.assertTrue(mock_connection.sendall.called)
        # Verify number of calls
        expected_calls = (2 * alice.sync_bytes_per_frame) * alice.sync_frames  # Data bytes + start/end markers
        self.assertEqual(mock_connection.sendall.call_count, expected_calls)
        # Verify start and end markers
        first_call = mock_connection.sendall.call_args_list[0]
        self.assertEqual(first_call[0][0], alice._start_marker.to_bytes(1, 'big'))
        
    def test_create_bit_base_qubits(self):
        """Test creation of random bits and bases for qubits."""
        alice = AliceQubits(num_qubits=10, num_frames=1, bytes_per_frame=10)
        alice.create_bit_base_qubits()
        self.assertEqual(len(alice.bits), 10)
        self.assertEqual(len(alice.bases), 10)
        self.assertEqual(len(alice.qubits_bytes), 10)

    @patch('socket.socket')
    def test_send_qubits(self, mock_socket):
        """Test sending qubits for Alice."""
        mock_connection = MagicMock()
        alice = self.alice_TEST.return_alice_reseted()
        alice.create_bit_base_qubits()
        
        with self.assertLogs('BB84 Simulation Log', level='INFO') as cm:
            alice.send_qubits(mock_connection)

        self.assertIn('Sending qubits...', cm.output[0])
        self.assertTrue(mock_connection.sendall.called)
        expected_calls = (2 * alice.bytes_per_frame) * alice.num_frames  # Data bytes + start/end markers
        self.assertEqual(mock_connection.sendall.call_count, expected_calls)

    # def test_qubit_sending(self):
    #     """Test that qubits are sent without errors."""
    #     alice = AliceQubits(num_qubits=10)
    #     with self.assertLogs('BB84 Simulation Log', level='INFO') as cm:
    #         # Simulate sending qubits (mock socket connection)
    #         mock_connection = MagicMock()
    #         alice.perform_sync(mock_connection)
    #         alice.send_qubits(mock_connection)
    #     self.assertIn('Sending qubits...', cm.output[0])

        
    # ------- BobQubits Tests -------
    def test_client_connection(self):
        """Test client connection for Bob."""
        with self.assertRaises(ConnectionRefusedError):
            BobQubits.setup_client("localhost", 9999)  # Assuming port 9999 is not in use

    @patch('socket.socket')
    def test_receive_sync(self, mock_socket):
        """Test receiving synchronization for Bob."""
        mock_connection = MagicMock()
        bob = BobQubits(num_qubits=10, sync_frames=2)
        bob.receive_sync(mock_connection)
        self.assertIsNotNone(bob.average_time_bin)

    @patch('socket.socket')
    def test_receive_qubits(self, mock_socket):
        """Test receiving qubits for Bob."""
        mock_connection = MagicMock()
        mock_connection.recv = MagicMock(side_effect=[b'\x64', b'\x32', b'\x50'])  # Mocking start, data, end marker
        bob = BobQubits(num_qubits=3, num_frames=1)
        bob.receive_qubits(mock_connection)
        self.assertEqual(len(bob.detected_qubits_bytes), 1)


class TestBB84Threads(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.NUM_QUBITS = 100
        self.NUM_FRAMES = 10
        self.BYTES_PER_FRAME = 10
        self.TEST_BOOL = True
        self.TEST_FRACTION = 0.2
        self.TEST_THRESHOLD = 0.11
        self.THREAD_ID = 0
        self.mock_queue = queue.Queue()
        self.mock_role = MagicMock()

        self.known_bases = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        self.known_bits = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
        self.known_qubits_bytes = [0, 2, 1, 3, 0, 2, 1, 3, 0, 2] # 2*base + bit
        self.known_detected_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.known_detected_bases = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
        self.known_detected_bits = [0, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        self.known_detected_qubits_bytes = [0, 2, 1, 3, 0, 1, 2, 0, 3, 1] # 2*det_base + det_bit

        # COMLPLET WITH INPUTS
        self.alice_thread_TEST = AliceThread(test_bool=self.TEST_BOOL, test_fraction=self.TEST_FRACTION, error_threshold=self.TEST_THRESHOLD, thread_id=self.THREAD_ID, role=self.mock_role, receive_queue=self.mock_queue)
        self.bob_thread_TEST = BobThread(test_bool=self.TEST_BOOL, test_fraction=self.TEST_FRACTION, thread_id=self.THREAD_ID, role=self.mock_role, receive_queue=self.mock_queue)

    # ------- AliceThread Tests -------
    def test_alice_thread_run(self):
        """Test the AliceThread run method."""
        mock_role = MagicMock()
        alice_thread = AliceThread.return_alice_reseted(self.alice_thread_TEST)
        
        with patch.object(alice_thread, 'receive_detected_idx', return_value=[0, 1]):
            with patch.object(alice_thread, 'send_detected_bases') as mock_send_bases:
                alice_thread.run_alice_thread()
                self.assertTrue(mock_send_bases.called)

    # ------- BobThread Tests -------
    def test_bob_thread_run(self):
        """Test the BobThread run method."""
        mock_role = MagicMock()
        bob_thread = BobThread.return_bob_reseted(self.bob_thread_TEST)
        
        with patch.object(bob_thread, 'send_detected_idx') as mock_send_idx:
            with patch.object(bob_thread, 'receive_detected_bases', return_value=[0, 1]):
                bob_thread.run_bob_thread()
                self.assertTrue(mock_send_idx.called)

    def test_partition_data(self):
        """Test data partitioning for Bob."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sizes = [3, 4, 3]
        partitioned_data = partition_data(data, sizes)
        self.assertEqual(len(partitioned_data), 3)
        self.assertEqual(partitioned_data[0], [1, 2, 3])
        self.assertEqual(partitioned_data[1], [4, 5, 6, 7])
        self.assertEqual(partitioned_data[2], [8, 9, 10])


if __name__ == '__main__':
    unittest.main()

