import heapq
import logging
import math
import os
import threading
import unittest
from time import sleep
from unittest.mock import patch, MagicMock
import queue

from communication_channel.common import convert_int_to_bytes, convert_from_bytes_to_int, timestamp, MAC_SIZE_BYTES, \
    convert_float_to_bytes, TIMESTAMP_SIZE_BYTES
from communication_channel.connection_info import ConnectionInfo
from communication_channel.exception.role_has_no_peer import RoleHasNoPeerException
from communication_channel.mac_config import MAC_Config, MAC_Algorithms
from communication_channel.packet import FRAME_END, get_payloads_from_received_frames
from classical_communication_channel.communication_channel.role import MACDisabledRole, Role, DATA_PACKET_TYPE, TOTAL_SIZE_DATA_PACKET_OFFSET, \
    FRAME_SIZE, HEADER_LENGTH_BYTES, ACK_PACKET_TYPE, ACK_TIMEOUT_SECONDS, MACEnabledRole, SEQUENCE_NUMBER_LENGTH, \
    OFFSET_LENGTH_BYTES
from test_general import SHARED_SECRET_KEY


# Mocks for socket context managers:
# They return a mock socket-like object that does nothing (or logs calls if you prefer).
def fake_socket_context_manager_connect(*args, **kwargs):
    """
    Fake context manager that returns a mock 'socket-like' object
    for the sending (connect) side.
    """
    mock_socket = MagicMock()
    # .connect(...) does nothing
    # .sendall(...) does nothing
    mock_socket.connect = MagicMock()
    mock_socket.sendall = MagicMock()
    return mock_socket


def fake_socket_context_manager_receive(*args, **kwargs):
    """
    Fake context manager that returns a mock 'socket-like' object
    for the receiving (accept) side.
    """
    mock_socket = MagicMock()
    # .accept(...) returns (conn, addr) which is also a mock
    mock_conn = MagicMock()
    mock_conn.recv = MagicMock()
    mock_socket.accept = MagicMock(return_value=(mock_conn, ("mock_addr", 1234)))
    return mock_socket


class TestMACDisabledRole(unittest.TestCase):

    def setUp(self):
        """
        Common setup for each test.
        We'll create a Role with no real bandwidth limit or capacity constraints.
        We also patch the thread starters so no real I/O _threads run.
        """
        # Patch out the _threads so they do not run automatically upon setting peer_connection_info.
        # We inject a manual patch here for each test method in setUp.
        self._start_data_exchange_threads_patch = patch.object(
            MACDisabledRole,
            "_start_data_exchange_threads",
            side_effect=lambda: None  # do nothing
        )
        self.mock_start_threads = self._start_data_exchange_threads_patch.start()

        # Patch out the _threads so they do not run automatically upon setting peer_connection_info.
        # We inject a manual patch here for each test method in setUp.
        self._start_fill_inbox_thread_patch = patch.object(
            MACDisabledRole,
            "_start_fill_inbox_thread",
            side_effect=lambda: None  # do nothing
        )
        self.mock_start_fill_inbox_threads = self._start_fill_inbox_thread_patch.start()

        # Create a test role
        self.test_role = Role.get_instance(
            receive_data_socket_info=ConnectionInfo("127.0.0.1", 9999),
            bandwidth_limit_megabytes_per_second=None,
            inbox_capacity_megabytes=None,
            outbox_capacity_megabytes=None
        )

    def tearDown(self):
        """
        Cleanup after each test.
        """
        self._start_data_exchange_threads_patch.stop()
        self.mock_start_fill_inbox_threads.stop()
        self.test_role.clean()

    @patch("communication_channel.common.create_connect_socket",
           side_effect=lambda: fake_socket_context_manager_connect())
    @patch("communication_channel.common.create_receive_socket",
           side_effect=lambda _: fake_socket_context_manager_receive())
    def test_put_in_outbox_small_data(self, mock_receive_socket, mock_connect_socket):
        """
        Test that put_in_outbox enqueues a small data payload (shorter than FRAME_SIZE).
        We also ensure no exception is thrown and confirm the outbox gets the correct frames.
        """
        # We need a peer set before we can put data in outbox
        self.test_role.peer_connection_info = ConnectionInfo("127.0.0.1", 8888)

        test_data = b"Hello World"
        self.test_role.put_in_outbox(test_data)

        # The outbox queue should have 2 payloads:
        #   1) The "total size" control packet
        #   2) The actual data packet
        packets_enqueued = []
        try:
            while True:
                packets_enqueued.append(self.test_role._outbox.get_nowait())
        except queue.Empty:
            pass

        self.assertEqual(len(packets_enqueued), 2, "Should enqueue exactly 2 payloads: total-size + 1 data packet.")

        # Validate the first packet is the total-size control packet
        first_packet = packets_enqueued[0]
        self.assertTrue(first_packet.startswith(FRAME_END + DATA_PACKET_TYPE),
                        "First packet must have data packet type (control).")
        # The offset in the control packet is TOTAL_SIZE_DATA_PACKET_OFFSET
        # offset is at bytes [1+4 : 1+4+4]
        offset_bytes = first_packet[1 + 1 + 4: 1 + 1 + 4 + 4]
        offset_int = convert_from_bytes_to_int(offset_bytes)
        self.assertEqual(offset_int, TOTAL_SIZE_DATA_PACKET_OFFSET,
                         "Offset in the control packet must be the 'total size' offset.")

        # Validate the second packet is the actual data payload
        second_packet = packets_enqueued[1]
        self.assertTrue(second_packet.startswith(FRAME_END + DATA_PACKET_TYPE),
                        "Data payload should also start with data packet type.")
        # The offset here should be 0
        offset_bytes = second_packet[1 + 1 + 4: 1 + 1 + 4 + 4]
        offset_int = convert_from_bytes_to_int(offset_bytes)
        self.assertEqual(offset_int, 0, "Offset for the first data payload must be 0.")

        data_payload = second_packet[10:-1]
        self.assertEqual(data_payload, test_data,
                         "The data payload in the outbox packet must match what was put in outbox.")

    @patch("communication_channel.common.create_connect_socket",
           side_effect=lambda: fake_socket_context_manager_connect())
    @patch("communication_channel.common.create_receive_socket",
           side_effect=lambda _: fake_socket_context_manager_receive())
    def test_put_in_outbox_larger_than_frame_size(self, mock_receive_socket, mock_connect_socket):
        """
        Test that put_in_outbox splits a larger payload into multiple frames.
        """
        # We need a peer set before we can put data in outbox
        self.test_role.peer_connection_info = ConnectionInfo("127.0.0.1", 8888)

        # Construct a payload larger than FRAME_SIZE
        test_data = os.urandom(FRAME_SIZE + 10)  # e.g. frame_size + 10

        self.test_role.put_in_outbox(test_data)

        # Outbox queue: 1 control packet + multiple data payloads
        packets_enqueued = []
        try:
            while True:
                packets_enqueued.append(self.test_role._outbox.get_nowait())
        except queue.Empty:
            pass

        # 1 control packet + 2 data payloads
        #   - The first data packet should contain the first FRAME_SIZE bytes
        #   - The second data packet should contain the remaining 10 bytes
        self.assertEqual(len(packets_enqueued), 3, "Should be total-size packet + 2 frames for the data.")

        # Check the second packet (first data payload) has offset = 0
        first_data_frame = packets_enqueued[1]
        packet_type = first_data_frame[1:2]
        seq_num = first_data_frame[2:6]
        seq_num_int = convert_from_bytes_to_int(seq_num)
        offset_bytes = first_data_frame[6:10]
        offset_int = convert_from_bytes_to_int(offset_bytes)
        self.assertEqual(packet_type, DATA_PACKET_TYPE)
        self.assertEqual(seq_num_int, 1)
        self.assertEqual(offset_int, 0)

        # The data portion is the last `FRAME_SIZE` bytes
        packets = get_payloads_from_received_frames(first_data_frame)
        data_part = packets[0]
        self.assertEqual(data_part,
                         DATA_PACKET_TYPE + convert_int_to_bytes(1) + convert_int_to_bytes(0) + test_data[:FRAME_SIZE])

        # Check the third packet (second data payload) offset = FRAME_SIZE
        second_data_frame = packets_enqueued[2]
        packet_type = second_data_frame[1:2]
        seq_num = second_data_frame[2:6]
        seq_num_int = convert_from_bytes_to_int(seq_num)
        offset_bytes = second_data_frame[6:10]
        offset_int = convert_from_bytes_to_int(offset_bytes)
        self.assertEqual(packet_type, DATA_PACKET_TYPE)
        self.assertEqual(seq_num_int, 2)
        # The data portion is the last 10 bytes
        data_part = second_data_frame[1 + HEADER_LENGTH_BYTES:-1]
        self.assertEqual(len(data_part), 10, "Should have 10 leftover bytes in the second data payload.")
        self.assertEqual(offset_int, FRAME_SIZE)

    def test_put_in_outbox_no_peer(self):
        """
        If we haven't set the peer, calling put_in_outbox should raise RoleHasNoPeerException.
        """
        with self.assertRaises(RoleHasNoPeerException):
            self.test_role.put_in_outbox(b"Should raise exception")

    @patch("communication_channel.common.create_connect_socket",
           side_effect=lambda: fake_socket_context_manager_connect())
    @patch("communication_channel.common.create_receive_socket",
           side_effect=lambda _: fake_socket_context_manager_receive())
    def test_get_from_inbox_reassembles_data(self, mock_receive_socket, mock_connect_socket):
        """
        Test that get_from_inbox can reconstruct data from the queue, given a control packet
        plus subsequent data payloads that arrive out of order.
        """
        # We need a peer set before get_from_inbox can do reassembly
        self.test_role.peer_connection_info = ConnectionInfo("127.0.0.1", 8888)

        # Manually feed the "control" packet (with total size) plus two data frames
        # in out-of-order offsets into the _inbox. The code under test will run
        # inside get_from_inbox’s logic.
        total_size = 15
        seq_num_control = 0
        seq_num_data1 = 1
        seq_num_data2 = 2

        # 1) Control packet: offset == TOTAL_SIZE_DATA_PACKET_OFFSET
        control_packet_type = DATA_PACKET_TYPE
        control_packet = (
                control_packet_type +
                convert_int_to_bytes(seq_num_control) +
                convert_int_to_bytes(TOTAL_SIZE_DATA_PACKET_OFFSET) +
                convert_int_to_bytes(total_size)  # the "payload" that says total_size
        )
        # 2) Data packet (payload offset=10, length=5)
        frame_offset_2 = 10
        payload_2 = b"World"  # 5 bytes
        data_packet_2 = (
                control_packet_type +
                convert_int_to_bytes(seq_num_data2) +
                convert_int_to_bytes(frame_offset_2) +
                payload_2
        )
        # 3) Data packet (payload offset=0, length=10)
        frame_offset_1 = 0
        payload_1 = b"HelloHello"  # 10 bytes
        data_packet_1 = (
                control_packet_type +
                convert_int_to_bytes(seq_num_data1) +
                convert_int_to_bytes(frame_offset_1) +
                payload_1
        )

        # Put the control packet first
        self.test_role._inbox.put(control_packet)

        # Then put data packet #2 out of order
        self.test_role._inbox.put(data_packet_2)

        # Then put data packet #1
        self.test_role._inbox.put(data_packet_1)

        # Call get_from_inbox, which should block until reassembly is done
        reassembled = self.test_role.get_from_inbox()

        # Validate result
        self.assertEqual(len(reassembled), total_size,
                         "Reassembled size must match the total_size in the control packet.")
        # We expect 'HelloHelloWorld'
        self.assertEqual(reassembled, b"HelloHelloWorld",
                         "The final data should be the combined frames in correct offset order.")

        # Ensure the role acked all three sequence numbers
        # (We can check that the outbox has the ack payloads.)
        acks_in_outbox = []
        try:
            while True:
                acks_in_outbox.append(self.test_role._outbox.get_nowait())
        except queue.Empty:
            pass

        # We expect 3 ACK payloads: for seq_num_control, seq_num_data2, seq_num_data1
        self.assertEqual(len(acks_in_outbox), 3, "Should produce exactly 3 ack payloads.")
        for ack_packet in acks_in_outbox:
            self.assertTrue(ack_packet.startswith(FRAME_END + ACK_PACKET_TYPE), "Should be an ACK packet.")

    @patch("communication_channel.common.create_connect_socket",
           side_effect=lambda: fake_socket_context_manager_connect())
    @patch("communication_channel.common.create_receive_socket",
           side_effect=lambda _: fake_socket_context_manager_receive())
    def test_get_from_inbox_empty_data(self, mock_receive_socket, mock_connect_socket):
        """
        If we push a control packet that says total_data_size=0, get_from_inbox should
        return immediately with empty bytes. (Though the current code early-returns in put_in_outbox
        if len(data)==0, this test shows how you'd handle a 0-length scenario anyway.)
        """
        self.test_role.peer_connection_info = ConnectionInfo("127.0.0.1", 8888)

        seq_num_control = 0
        zero_size_packet = (
                DATA_PACKET_TYPE +
                convert_int_to_bytes(seq_num_control) +
                convert_int_to_bytes(TOTAL_SIZE_DATA_PACKET_OFFSET) +
                convert_int_to_bytes(0)  # total_size = 0
        )
        self.test_role._inbox.put(zero_size_packet)

        result = self.test_role.get_from_inbox()
        self.assertEqual(result, b"", "Should return empty bytes if total size is 0.")

    @patch("communication_channel.common.create_connect_socket",
           side_effect=lambda: fake_socket_context_manager_connect())
    @patch("communication_channel.common.create_receive_socket",
           side_effect=lambda _: fake_socket_context_manager_receive())
    def test_put_in_outbox_100MB_with_mocked_acks(self, mock_recv_socket, mock_conn_socket):
        """
        Sends a 100MB random payload via put_in_outbox, while manually injecting
        fake ACK packets so the sliding _sender_window never stalls.
        """
        # 1) Set the peer to enable outbox sending.
        self.test_role.peer_connection_info = ConnectionInfo("127.0.0.1", 8888)
        payload_size = 100 * 1_024 * 1_024

        # 2) Generate random data
        test_data = os.urandom(payload_size)

        # 3) We'll run put_in_outbox on a separate thread so it can block/wait for ACKs if needed.
        sender_thread = threading.Thread(
            target=self.test_role.put_in_outbox,
            args=(test_data,),
            daemon=True
        )

        # 4) Start the sending thread.
        t_init = timestamp()
        sender_thread.start()

        # 5) On the main thread, consume packets from _outbox.
        #    For each data packet, we generate a corresponding ACK and feed it back in.
        #    This simulates that the peer is acknowledging everything immediately,
        #    preventing the sending _sender_window from stalling.
        data_packets_count = 0
        control_packet_count = 0
        all_packets = []  # We'll store them for optional verification.

        # We'll keep reading until the sender_thread finishes and the queue is empty.
        while sender_thread.is_alive() or not self.test_role._outbox.empty():
            try:
                packet = self.test_role._outbox.get(timeout=0.1)
            except queue.Empty:
                # Possibly the thread is still sending or done. Just continue looping.
                continue

            all_packets.append(packet)

            # Check if it's data or control
            if packet.startswith(FRAME_END + DATA_PACKET_TYPE):
                # Parse sequence number
                seq_num = convert_from_bytes_to_int(packet[2:6])
                offset = convert_from_bytes_to_int(packet[6:10])

                # If offset == TOTAL_SIZE_DATA_PACKET_OFFSET, it's the "control" packet
                if offset == TOTAL_SIZE_DATA_PACKET_OFFSET:
                    control_packet_count += 1
                else:
                    data_packets_count += 1

                # 6) Build a mock ACK packet for this seq_num and feed it back in.
                ack_payload = ACK_PACKET_TYPE + convert_int_to_bytes(seq_num)
                # We call _handle_received_packets directly (since the real code
                # processes incoming data in _fill_inbox _threads).
                # This will remove the packet from _sender_window,
                # freeing the _sender_window for the sender to continue.
                self.test_role._handle_received_payloads([ack_payload])

            else:
                # You might want to handle unexpected packet types or just ignore.
                pass

        # 7) Join the sender thread to ensure it completed.
        sender_thread.join(timeout=5)
        self.assertFalse(sender_thread.is_alive(), "Sender thread should have finished sending 100MB data.")
        t = timestamp()
        logging.error(f"took {t - t_init} seconds")

        # Basic checks:
        #   - Exactly 1 control packet should have been sent.
        #   - The number of data packets is ceil(100MB/FRAME_SIZE).
        expected_data_packets = math.ceil(payload_size / FRAME_SIZE)
        self.assertEqual(control_packet_count, 1, "Should have exactly one control packet.")
        self.assertEqual(data_packets_count, expected_data_packets,
                         f"Should have {expected_data_packets} data packets for 100MB payload.")

        # OPTIONAL: Thoroughly verify each data packet's offset/payload if you wish.
        # But note that verifying 100 MB of data is memory/time intensive in a unit test.
        # For demonstration, here is how you *could* do it in a streaming manner:

        # Example partial validation: track the total data bytes accounted for
        total_data = 0
        for pkt in all_packets[1:]:
            payloads = get_payloads_from_received_frames(pkt)
            total_data += len(payloads[0][HEADER_LENGTH_BYTES:])

        self.assertEqual(
            total_data,
            payload_size,
            "All 100MB should be present across the data packets."
        )

        print(f"\nSuccessfully sent and 'acked' {payload_size} bytes in {data_packets_count} data packets.\n")

    @patch("communication_channel.common.create_connect_socket",
           side_effect=lambda: fake_socket_context_manager_connect())
    @patch("communication_channel.common.create_receive_socket",
           side_effect=lambda _: fake_socket_context_manager_receive())
    def test_get_from_inbox_100MB(self, mock_recv_socket, mock_conn_socket):
        """
        Test that get_from_inbox can reassemble 100 MB of data
        fed (in multiple frames) into the role's _inbox queue.
        """
        # Set the peer so the role starts any needed internal logic (though we've patched out the real _threads).
        self.test_role.peer_connection_info = ConnectionInfo("127.0.0.1", 8888)

        # Generate 100 MB of random data
        payload_size = 100 * 1024 * 1024
        original_data = os.urandom(payload_size)

        # We'll store the result of get_from_inbox in a thread-safe queue
        # so we can retrieve it after the thread completes.
        result_queue = queue.Queue()

        def run_get_from_inbox():
            """Thread target that calls get_from_inbox and puts the result in a queue."""
            reassembled = self.test_role.get_from_inbox()
            result_queue.put(reassembled)

        # Start the thread that will block on get_from_inbox
        receiver_thread = threading.Thread(target=run_get_from_inbox, daemon=True)
        receiver_thread.start()

        # ---------------------------------------------------------------------
        # Phase 1: Feed a "control" packet indicating the total payload size.
        # ---------------------------------------------------------------------
        seq_num_control = 0
        packet_type = DATA_PACKET_TYPE
        offset_control = TOTAL_SIZE_DATA_PACKET_OFFSET

        control_packet = (
                packet_type +
                convert_int_to_bytes(seq_num_control) +
                convert_int_to_bytes(offset_control) +
                convert_int_to_bytes(payload_size)
        )
        self.test_role._inbox.put(control_packet)

        # ---------------------------------------------------------------------
        # Phase 2: Split the 100 MB data into frames, feed them to _inbox.
        #         This simulates "receiving" many data packets from the peer.
        # ---------------------------------------------------------------------
        seq_num_data = 1
        offset = 0
        while offset < payload_size:
            chunk = original_data[offset: offset + FRAME_SIZE]
            data_packet = (
                    packet_type +
                    convert_int_to_bytes(seq_num_data) +
                    convert_int_to_bytes(offset) +
                    chunk
            )
            self.test_role._inbox.put(data_packet)
            offset += len(chunk)
            seq_num_data += 1

        # Wait for the thread to finish reassembling
        receiver_thread.join(timeout=5)
        self.assertFalse(
            receiver_thread.is_alive(),
            "get_from_inbox thread should have completed reassembly of 100MB."
        )

        # Retrieve the data from the queue and verify correctness
        reassembled_data = result_queue.get()
        self.assertEqual(
            len(reassembled_data),
            payload_size,
            "Reassembled data must match 100MB in length."
        )
        self.assertEqual(
            reassembled_data,
            original_data,
            "All bytes in the reassembled data must match the original 100MB payload."
        )

        print(f"\nReassembled 100MB successfully with {seq_num_data - 1} data frames.\n")

    def test_full_exchange_1_megabyte(self):
        alice = self.test_role
        bob = Role.get_instance(
            receive_data_socket_info=ConnectionInfo("127.0.0.1", 8888),
            bandwidth_limit_megabytes_per_second=None,
            inbox_capacity_megabytes=None,
            outbox_capacity_megabytes=None
        )
        # 1) Set the peer to enable outbox sending.
        alice.peer_connection_info = bob._receive_data_socket_info
        bob.peer_connection_info = bob._receive_data_socket_info
        payload_size = 1 * 1_024 * 1_024

        # We'll store the result of get_from_inbox in a thread-safe queue
        # so we can retrieve it after the thread completes.
        result_queue = queue.Queue()

        def run_get_from_inbox():
            """Thread target that calls get_from_inbox and puts the result in a queue."""
            reassembled = bob.get_from_inbox()
            result_queue.put(reassembled)

        # 2) Generate random data
        test_data = os.urandom(payload_size)

        # 3) We'll run put_in_outbox on a separate thread so it can block/wait for ACKs if needed.
        sender_thread = threading.Thread(
            target=alice.put_in_outbox,
            args=(test_data,),
            daemon=True
        )

        # Start the thread that will block on get_from_inbox
        receiver_thread = threading.Thread(target=run_get_from_inbox, daemon=True)
        receiver_thread.start()

        # 4) Start the sending thread.
        t_init = timestamp()
        sender_thread.start()

        # We'll keep reading until the sender_thread finishes and the queue is empty.
        while sender_thread.is_alive() or not alice._outbox.empty():
            try:
                packet = alice._outbox.get(timeout=0.1)
            except queue.Empty:
                # Possibly the thread is still sending or done. Just continue looping.
                continue

            bob._inbox.put(get_payloads_from_received_frames(packet)[0], timeout=0.1)
            packet = bob._outbox.get(timeout=0.1)
            # Check if it's data or control
            if packet.startswith(FRAME_END + ACK_PACKET_TYPE):
                self.test_role._handle_received_payloads([packet[1:-1]])
            else:
                # You might want to handle unexpected packet types or just ignore.
                pass

        # 7) Join the sender thread to ensure it completed.
        sender_thread.join(timeout=5)
        self.assertFalse(sender_thread.is_alive(), "Sender thread should have finished sending 1MB data.")
        t = timestamp()
        logging.error(f"took {t - t_init} seconds")

        # Wait for the thread to finish reassembling
        receiver_thread.join(timeout=5)
        self.assertFalse(
            receiver_thread.is_alive(),
            "get_from_inbox thread should have completed reassembly of 100MB."
        )

        # Retrieve the data from the queue and verify correctness
        reassembled_data = result_queue.get(timeout=5)
        self.assertEqual(
            len(reassembled_data),
            payload_size,
            "Reassembled data must match 100MB in length."
        )
        self.assertEqual(
            reassembled_data,
            test_data,
            "All bytes in the reassembled data must match the original 100MB payload."
        )



class TestCheckTimeouts(unittest.TestCase):
    def setUp(self):
        # Patch out the _threads that normally start on peer assignment
        self._start_data_exchange_threads_patch = patch.object(
            MACDisabledRole,
            "_start_data_exchange_threads",
            side_effect=lambda: None
        )
        self._start_data_exchange_threads_patch.start()

        # Create a role instance
        self.test_role = Role.get_instance(
            receive_data_socket_info=ConnectionInfo("127.0.0.1", 9999),
            bandwidth_limit_megabytes_per_second=None,
            inbox_capacity_megabytes=None,
            outbox_capacity_megabytes=None
        )

        # Manually set a peer so that _put_packet_in_outbox doesn't raise an exception
        self.test_role._peer_connection_info = ConnectionInfo("127.0.0.1", 8888)

    def tearDown(self):
        self._start_data_exchange_threads_patch.stop()
        self.test_role.clean()


    def test_check_timeouts_retransmits_expired_packet(self):

        call_count = {"n": 0}

        def fake_wait_synchronously(seconds):
            call_count["n"] += 1
            # After a couple of iterations, stop the thread
            if call_count["n"] > 2:
                self.test_role._thread_stop_flag = True
            sleep(0.01)

        @patch("communication_channel.role.wait_synchronously", side_effect= lambda x: fake_wait_synchronously(x))
        @patch("communication_channel.common.create_receive_socket",
               side_effect=lambda _: fake_socket_context_manager_receive())
        @patch("communication_channel.common.create_connect_socket",
               side_effect=lambda: fake_socket_context_manager_connect())
        @patch("communication_channel.role.timestamp", return_value=1000)
        def _test_check_timeouts_retransmits_expired_packet(
                mock_timestamp,
                mock_conn_socket,
                mock_recv_socket,
                mock_wait_synchronously,
        ):
            """
            Verify that _check_timeouts retransmits packets whose timestamps exceed ACK_TIMEOUT_SECONDS
            and does not retransmit those that haven't expired.
            """
            # We'll control what 'timestamp()' returns and how 'wait_synchronously' behaves.

            # We'll store two packets:
            #   1) One is "expired"
            #   2) One is "not expired"
            expired_seq_num = 42
            unexpired_seq_num = 43

            # By definition: 'now - timestamp' should be >= ACK_TIMEOUT_SECONDS to be expired
            expired_timestamp = 1000 - (ACK_TIMEOUT_SECONDS + 5.0)
            unexpired_timestamp = 1000 - (ACK_TIMEOUT_SECONDS / 2.0)  # hasn't timed out yet

            # The code snippet's structure for _sender_window is:
            #   self._sender_window[seq_num] = (packet_bytes, _timestamp, packet_for_retransmit)
            self.test_role._sender_window[expired_seq_num] = (
                b"expired packet", expired_timestamp)
            self.test_role._sender_window[unexpired_seq_num] = (
                b"unexpired packet", unexpired_timestamp )

            # The heap holds tuples (seq_num, _timestamp)
            heapq.heappush(self.test_role._sender_window_timestamp_heap, (expired_seq_num, expired_timestamp))
            heapq.heappush(self.test_role._sender_window_timestamp_heap, (unexpired_seq_num, unexpired_timestamp))

            # We'll let _check_timeouts run in a separate thread. We'll make wait_synchronously a no-op that
            # quickly sets _thread_stop_flag after 1-2 calls so the loop doesn't spin forever.

            # Run _check_timeouts in a thread
            timeout_thread = threading.Thread(target=self.test_role._check_timeouts, daemon=True)
            timeout_thread.start()

            # Wait for the thread to exit
            #timeout_thread.join(timeout=5)
            timeout_thread.join(timeout=5)
            self.assertFalse(timeout_thread.is_alive(), "_check_timeouts thread should have stopped by now.")

            # Check the outbox for retransmitted packets
            retransmitted_packets = []
            try:
                while True:
                    pkt = self.test_role._outbox.get_nowait()
                    retransmitted_packets.append(pkt)
            except queue.Empty:
                pass

            # We expect exactly 1 retransmitted packet (the expired one)
            self.assertEqual(
                len(retransmitted_packets), 1,
                "Should have retransmitted exactly 1 expired packet."
            )
            self.assertEqual(
                retransmitted_packets[0], b"expired packet",
                "The retransmitted packet data should match 'expired packet'."
            )

            # Expired packet should be removed from _sender_window
            self.assertNotIn(
                expired_seq_num,
                self.test_role._sender_window,
                "Expired packet must be removed from _sender_window."
            )

            # Unexpired packet should still be outstanding
            self.assertIn(
                unexpired_seq_num,
                self.test_role._sender_window,
                "Unexpired packet should remain in _sender_window."
            )

            # Also confirm the heap does not contain the expired packet
            remaining_heap_entries = list(self.test_role._sender_window_timestamp_heap)
            expired_in_heap = any(seq == expired_seq_num for seq, ts in remaining_heap_entries)
            self.assertFalse(
                expired_in_heap,
                "Expired packet must be popped from the heap."
            )
            unexpired_in_heap = any(seq == unexpired_seq_num for seq, ts in remaining_heap_entries)
            self.assertTrue(
                unexpired_in_heap,
                "Unexpired packet must remain in the heap."
            )

            print("\n_check_timeouts test passed: expired packet retransmitted, unexpired remained.\n")

        _test_check_timeouts_retransmits_expired_packet()

class TestMACEnabledRole(unittest.TestCase):

    def setUp(self):
        """
        Common setup for each test.
        We'll create a Role with no real bandwidth limit or capacity constraints.
        We also patch the thread starters so no real I/O _threads run.
        """
        # Patch out the _threads so they do not run automatically upon setting peer_connection_info.
        # We inject a manual patch here for each test method in setUp.
        self._start_data_exchange_threads_patch = patch.object(
            MACEnabledRole,
            "_start_data_exchange_threads",
            side_effect=lambda: None  # do nothing
        )
        self.mock_start_threads = self._start_data_exchange_threads_patch.start()

        # Patch out the _threads so they do not run automatically upon setting peer_connection_info.
        # We inject a manual patch here for each test method in setUp.
        self._start_fill_inbox_thread_patch = patch.object(
            MACEnabledRole,
            "_start_fill_inbox_thread",
            side_effect=lambda: None  # do nothing
        )
        self.mock_start_fill_inbox_threads = self._start_fill_inbox_thread_patch.start()

        # Create a test role
        self.test_role = Role.get_instance(
            receive_data_socket_info=ConnectionInfo("127.0.0.1", 9999),
            bandwidth_limit_megabytes_per_second=None,
            inbox_capacity_megabytes=None,
            outbox_capacity_megabytes=None,
            mac_config=MAC_Config(MAC_Algorithms.CMAC, SHARED_SECRET_KEY)
        )

    def tearDown(self):
        """
        Cleanup after each test.
        """
        self._start_data_exchange_threads_patch.stop()
        self.mock_start_fill_inbox_threads.stop()
        self.test_role.clean()

    @patch("communication_channel.common.create_connect_socket",
           side_effect=lambda: fake_socket_context_manager_connect())
    @patch("communication_channel.common.create_receive_socket",
           side_effect=lambda _: fake_socket_context_manager_receive())
    def test_put_in_outbox_small_data(self, mock_receive_socket, mock_connect_socket):
        """
        Test that put_in_outbox enqueues a small data payload (shorter than FRAME_SIZE).
        We also ensure no exception is thrown and confirm the outbox gets the correct frames.
        """
        # We need a peer set before we can put data in outbox
        self.test_role.peer_connection_info = ConnectionInfo("127.0.0.1", 8888)

        test_data = b"Hello World"
        self.test_role.put_in_outbox(test_data)

        # The outbox queue should have 2 payloads:
        #   1) The "total size" control packet
        #   2) The actual data packet
        packets_enqueued = []
        try:
            while True:
                packets_enqueued.append(self.test_role._outbox.get_nowait())
        except queue.Empty:
            pass

        self.assertEqual(len(packets_enqueued), 2, "Should enqueue exactly 2 payloads: total-size + 1 data packet.")

        # Validate the first packet is the total-size control packet
        first_packet = packets_enqueued[0]
        self.assertTrue(first_packet.startswith(FRAME_END + DATA_PACKET_TYPE),
                        "First packet must have data packet type (control).")
        # The offset in the control packet is TOTAL_SIZE_DATA_PACKET_OFFSET
        # offset is at bytes [1+4 : 1+4+4]
        offset_bytes = first_packet[1 + 1 + 4: 1 + 1 + 4 + 4]
        offset_int = convert_from_bytes_to_int(offset_bytes)
        self.assertEqual(offset_int, TOTAL_SIZE_DATA_PACKET_OFFSET,
                         "Offset in the control packet must be the 'total size' offset.")

        # Validate the second packet is the actual data payload
        second_packet = packets_enqueued[1]
        self.assertTrue(second_packet.startswith(FRAME_END + DATA_PACKET_TYPE),
                        "Data payload should also start with data packet type.")
        # The offset here should be 0
        offset_bytes = get_payloads_from_received_frames(second_packet)[0][1 + SEQUENCE_NUMBER_LENGTH:1 + SEQUENCE_NUMBER_LENGTH + OFFSET_LENGTH_BYTES]
        offset_int = convert_from_bytes_to_int(offset_bytes)
        self.assertEqual(offset_int, 0, "Offset for the first data payload must be 0.")

        data_payload = get_payloads_from_received_frames(second_packet)[0][1 + SEQUENCE_NUMBER_LENGTH + OFFSET_LENGTH_BYTES + TIMESTAMP_SIZE_BYTES:-MAC_SIZE_BYTES]
        self.assertEqual(data_payload, test_data,
                         "The data payload in the outbox packet must match what was put in outbox.")

    @patch("communication_channel.common.create_connect_socket",
           side_effect=lambda: fake_socket_context_manager_connect())
    @patch("communication_channel.common.create_receive_socket",
           side_effect=lambda _: fake_socket_context_manager_receive())
    def test_put_in_outbox_larger_than_frame_size(self, mock_receive_socket, mock_connect_socket):
        """
        Test that put_in_outbox splits a larger payload into multiple frames.
        """
        # We need a peer set before we can put data in outbox
        self.test_role.peer_connection_info = ConnectionInfo("127.0.0.1", 8888)

        # Construct a payload larger than FRAME_SIZE
        test_data = os.urandom(FRAME_SIZE + 10)  # e.g. frame_size + 10

        self.test_role.put_in_outbox(test_data)

        # Outbox queue: 1 control packet + multiple data payloads
        packets_enqueued = []
        try:
            while True:
                packets_enqueued.append(self.test_role._outbox.get_nowait())
        except queue.Empty:
            pass

        # 1 control packet + 2 data payloads
        #   - The first data packet should contain the first FRAME_SIZE bytes
        #   - The second data packet should contain the remaining 10 bytes
        self.assertEqual(len(packets_enqueued), 3, "Should be total-size packet + 2 frames for the data.")

        # Check the second packet (first data payload) has offset = 0
        first_data_frame = packets_enqueued[1]
        packet_type = first_data_frame[1:2]
        seq_num = first_data_frame[2:6]
        seq_num_int = convert_from_bytes_to_int(seq_num)
        offset_bytes = first_data_frame[6:10]
        offset_int = convert_from_bytes_to_int(offset_bytes)
        self.assertEqual(packet_type, DATA_PACKET_TYPE)
        self.assertEqual(seq_num_int, 1)
        self.assertEqual(offset_int, 0)

        # The data portion is the last `FRAME_SIZE` bytes
        packets = get_payloads_from_received_frames(first_data_frame)
        data_part = packets[0]
        self.assertEqual(data_part[13:-MAC_SIZE_BYTES],test_data[:FRAME_SIZE])

        # Check the third packet (second data payload) offset = FRAME_SIZE
        second_data_frame = packets_enqueued[2]
        packet_type = second_data_frame[1:2]
        seq_num = second_data_frame[2:6]
        seq_num_int = convert_from_bytes_to_int(seq_num)
        offset_bytes = second_data_frame[6:10]
        offset_int = convert_from_bytes_to_int(offset_bytes)
        self.assertEqual(packet_type, DATA_PACKET_TYPE)
        self.assertEqual(seq_num_int, 2)
        # The data portion is the last 10 bytes
        data_part = second_data_frame[14:-(1 + MAC_SIZE_BYTES)]
        self.assertEqual(len(data_part), 10, "Should have 10 leftover bytes in the second data payload.")
        self.assertEqual(offset_int, FRAME_SIZE)

    def test_put_in_outbox_no_peer(self):
        """
        If we haven't set the peer, calling put_in_outbox should raise RoleHasNoPeerException.
        """
        with self.assertRaises(RoleHasNoPeerException):
            self.test_role.put_in_outbox(b"Should raise exception")

    @patch("communication_channel.common.create_connect_socket",
           side_effect=lambda: fake_socket_context_manager_connect())
    @patch("communication_channel.common.create_receive_socket",
           side_effect=lambda _: fake_socket_context_manager_receive())
    def test_put_in_outbox_100MB_with_mocked_acks(self, mock_recv_socket, mock_conn_socket):
        """
        Sends a 100MB random payload via put_in_outbox, while manually injecting
        fake ACK packets so the sliding _sender_window never stalls.
        """
        # 1) Set the peer to enable outbox sending.
        self.test_role.peer_connection_info = ConnectionInfo("127.0.0.1", 8888)
        payload_size = 100 * 1_024 * 1_024

        # 2) Generate random data
        test_data = os.urandom(payload_size)

        # 3) We'll run put_in_outbox on a separate thread so it can block/wait for ACKs if needed.
        sender_thread = threading.Thread(
            target=self.test_role.put_in_outbox,
            args=(test_data,),
            daemon=True
        )

        # 4) Start the sending thread.
        t_init = timestamp()
        sender_thread.start()

        # 5) On the main thread, consume packets from _outbox.
        #    For each data packet, we generate a corresponding ACK and feed it back in.
        #    This simulates that the peer is acknowledging everything immediately,
        #    preventing the sending _sender_window from stalling.
        data_packets_count = 0
        control_packet_count = 0
        all_packets = []  # We'll store them for optional verification.

        # We'll keep reading until the sender_thread finishes and the queue is empty.
        while sender_thread.is_alive() or not self.test_role._outbox.empty():
            try:
                packet = self.test_role._outbox.get(timeout=0.1)
            except queue.Empty:
                # Possibly the thread is still sending or done. Just continue looping.
                continue

            all_packets.append(packet)

            # Check if it's data or control
            if packet.startswith(FRAME_END + DATA_PACKET_TYPE):
                # Parse sequence number
                seq_num = convert_from_bytes_to_int(packet[2:6])
                offset = convert_from_bytes_to_int(packet[6:10])

                # If offset == TOTAL_SIZE_DATA_PACKET_OFFSET, it's the "control" packet
                if offset == TOTAL_SIZE_DATA_PACKET_OFFSET:
                    control_packet_count += 1
                else:
                    data_packets_count += 1

                # 6) Build a mock ACK packet for this seq_num and feed it back in.
                ack_payload = ACK_PACKET_TYPE + convert_int_to_bytes(seq_num)
                # We call _handle_received_packets directly (since the real code
                # processes incoming data in _fill_inbox _threads).
                # This will remove the packet from _sender_window,
                # freeing the _sender_window for the sender to continue.
                self.test_role._handle_received_payloads([ack_payload])

            else:
                # You might want to handle unexpected packet types or just ignore.
                pass

        # 7) Join the sender thread to ensure it completed.
        sender_thread.join(timeout=5)
        self.assertFalse(sender_thread.is_alive(), "Sender thread should have finished sending 100MB data.")
        t = timestamp()
        logging.error(f"took {t - t_init} seconds")

        # Basic checks:
        #   - Exactly 1 control packet should have been sent.
        #   - The number of data packets is ceil(100MB/FRAME_SIZE).
        expected_data_packets = math.ceil(payload_size / FRAME_SIZE)
        self.assertEqual(control_packet_count, 1, "Should have exactly one control packet.")
        self.assertEqual(data_packets_count, expected_data_packets,
                         f"Should have {expected_data_packets} data packets for 100MB payload.")

        # OPTIONAL: Thoroughly verify each data packet's offset/payload if you wish.
        # But note that verifying 100 MB of data is memory/time intensive in a unit test.
        # For demonstration, here is how you *could* do it in a streaming manner:

        # Example partial validation: track the total data bytes accounted for
        total_data = 0
        t_avg = 0
        for pkt in all_packets[1:]:
            payloads = get_payloads_from_received_frames(pkt)
            t_init = timestamp()
            self.assertTrue(self.test_role._has_valid_mac(payloads[0]))
            self.assertTrue(self.test_role._has_valid_timestamp(payloads[0]))
            t = timestamp()
            total_data += len(payloads[0][HEADER_LENGTH_BYTES + 4:-MAC_SIZE_BYTES])
            t_avg += (t - t_init) / (total_data / FRAME_SIZE)

        logging.getLogger().error(f"average mac check: {t_avg}s")

        self.assertEqual(
            total_data,
            payload_size,
            "All 100MB should be present across the data packets."
        )

        print(f"\nSuccessfully sent and 'acked' {payload_size} bytes in {data_packets_count} data packets.\n")

    @patch("communication_channel.common.create_connect_socket",
           side_effect=lambda: fake_socket_context_manager_connect())
    @patch("communication_channel.common.create_receive_socket",
           side_effect=lambda _: fake_socket_context_manager_receive())
    def test_get_from_inbox_100MB(self, mock_recv_socket, mock_conn_socket):
        """
        Test that get_from_inbox can reassemble 100 MB of data
        fed (in multiple frames) into the role's _inbox queue.
        """
        # Set the peer so the role starts any needed internal logic (though we've patched out the real _threads).
        self.test_role.peer_connection_info = ConnectionInfo("127.0.0.1", 8888)

        # Generate 100 MB of random data
        payload_size = 100 * 1024 * 1024
        original_data = os.urandom(payload_size)

        # We'll store the result of get_from_inbox in a thread-safe queue
        # so we can retrieve it after the thread completes.
        result_queue = queue.Queue()

        def run_get_from_inbox():
            """Thread target that calls get_from_inbox and puts the result in a queue."""
            reassembled = self.test_role.get_from_inbox()
            result_queue.put(reassembled)

        # Start the thread that will block on get_from_inbox
        receiver_thread = threading.Thread(target=run_get_from_inbox, daemon=True)
        receiver_thread.start()

        # ---------------------------------------------------------------------
        # Phase 1: Feed a "control" packet indicating the total payload size.
        # ---------------------------------------------------------------------
        seq_num_control = 0
        offset_control = TOTAL_SIZE_DATA_PACKET_OFFSET

        header = DATA_PACKET_TYPE + convert_int_to_bytes(seq_num_control) + convert_int_to_bytes(
            offset_control) + convert_float_to_bytes(timestamp())
        control_packet = header + convert_int_to_bytes(payload_size)
        control_packet += self.test_role._calculate_mac(control_packet)
        self.test_role._inbox.put(control_packet)

        # ---------------------------------------------------------------------
        # Phase 2: Split the 100 MB data into frames, feed them to _inbox.
        #         This simulates "receiving" many data packets from the peer.
        # ---------------------------------------------------------------------
        seq_num_data = 1
        offset = 0
        t_avg = 0
        while offset < payload_size:
            chunk = original_data[offset: offset + FRAME_SIZE]
            header = DATA_PACKET_TYPE + convert_int_to_bytes(seq_num_data) + convert_int_to_bytes(offset) + convert_float_to_bytes(timestamp())
            data_packet = header + chunk
            t_init = timestamp()
            _mac = self.test_role._calculate_mac(data_packet)
            t = timestamp()
            t_avg += (t - t_init) / seq_num_data
            data_packet += _mac
            self.test_role._inbox.put(data_packet)
            offset += len(chunk)
            seq_num_data += 1

        logging.getLogger().error(f"Average time to calculate mac: {t_avg}s")

        # Wait for the thread to finish reassembling
        receiver_thread.join(timeout=5)
        self.assertFalse(
            receiver_thread.is_alive(),
            "get_from_inbox thread should have completed reassembly of 100MB."
        )

        # Retrieve the data from the queue and verify correctness
        reassembled_data = result_queue.get()
        self.assertEqual(
            len(reassembled_data),
            payload_size,
            "Reassembled data must match 100MB in length."
        )
        self.assertEqual(
            reassembled_data,
            original_data,
            "All bytes in the reassembled data must match the original 100MB payload."
        )

        print(f"\nReassembled 100MB successfully with {seq_num_data - 1} data frames.\n")


    def test_full_exchange_1_megabyte(self):
        alice = self.test_role
        bob = Role.get_instance(
            receive_data_socket_info=ConnectionInfo("127.0.0.1", 8888),
            bandwidth_limit_megabytes_per_second=None,
            inbox_capacity_megabytes=None,
            outbox_capacity_megabytes=None,
            mac_config=MAC_Config(MAC_Algorithms.CMAC, SHARED_SECRET_KEY)
        )
        # 1) Set the peer to enable outbox sending.
        alice.peer_connection_info = bob._receive_data_socket_info
        bob.peer_connection_info = bob._receive_data_socket_info
        payload_size = 1 * 1_024 * 1_024

        # We'll store the result of get_from_inbox in a thread-safe queue
        # so we can retrieve it after the thread completes.
        result_queue = queue.Queue()

        def run_get_from_inbox():
            """Thread target that calls get_from_inbox and puts the result in a queue."""
            reassembled = bob.get_from_inbox()
            result_queue.put(reassembled)

        # 2) Generate random data
        test_data = os.urandom(payload_size)

        # 3) We'll run put_in_outbox on a separate thread so it can block/wait for ACKs if needed.
        sender_thread = threading.Thread(
            target=alice.put_in_outbox,
            args=(test_data,),
            daemon=True
        )

        # Start the thread that will block on get_from_inbox
        receiver_thread = threading.Thread(target=run_get_from_inbox, daemon=True)
        receiver_thread.start()

        # 4) Start the sending thread.
        t_init = timestamp()
        sender_thread.start()

        # We'll keep reading until the sender_thread finishes and the queue is empty.
        while sender_thread.is_alive() or not alice._outbox.empty():
            try:
                packet = alice._outbox.get(timeout=0.1)
            except queue.Empty:
                # Possibly the thread is still sending or done. Just continue looping.
                continue

            bob._inbox.put(get_payloads_from_received_frames(packet)[0], timeout=0.1)
            packet = bob._outbox.get(timeout=0.1)
            # Check if it's data or control
            if packet.startswith(FRAME_END + ACK_PACKET_TYPE):
                self.test_role._handle_received_payloads([packet[1:-1]])
            else:
                # You might want to handle unexpected packet types or just ignore.
                pass

        # 7) Join the sender thread to ensure it completed.
        sender_thread.join(timeout=5)
        self.assertFalse(sender_thread.is_alive(), "Sender thread should have finished sending 1MB data.")
        t = timestamp()
        logging.error(f"took {t - t_init} seconds")

        # Wait for the thread to finish reassembling
        receiver_thread.join(timeout=5)
        self.assertFalse(
            receiver_thread.is_alive(),
            "get_from_inbox thread should have completed reassembly of 100MB."
        )

        # Retrieve the data from the queue and verify correctness
        reassembled_data = result_queue.get()
        self.assertEqual(
            len(reassembled_data),
            payload_size,
            "Reassembled data must match 100MB in length."
        )
        self.assertEqual(
            reassembled_data,
            test_data,
            "All bytes in the reassembled data must match the original 100MB payload."
        )

if __name__ == "__main__":
    unittest.main()
