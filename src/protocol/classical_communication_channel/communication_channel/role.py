import logging
import math
import pickle
import queue
import threading
import time
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from math import ceil
from time import sleep

from unittest.mock import Mock

from cryptography.exceptions import InvalidSignature

from .byte_queue import ByteQueue
from .common import (create_connect_socket, create_receive_socket, timestamp, MAC_SIZE_BYTES,
                     MAX_CREATION_TIMESTAMP_AGE_SECONDS, convert_int_to_bytes,
                     convert_float_to_bytes,
                     TIMESTAMP_SIZE_BYTES, convert_from_bytes_to_int,
                     MEGABYTES_IN_BYTES, convert_from_bytes_to_float)
from .connection_info import ConnectionInfo
from .exception.peer_has_role import RoleAlreadyHasPeerException
from .exception.role_has_no_peer import RoleHasNoPeerException
from .mac_config import MAC_Config

from datetime import datetime, timedelta

MAX_PUT_PACKET_TRIES = 5
PACKET_SEND_TIMEOUT = 0.1
FRAME_OUTBOX_PACKET_SEND_TIMEOUT = 0.1
#FILL_INBOX_TIMEOUT_SECONDS = 0.1
FILL_INBOX_TIMEOUT_SECONDS = 20.0
WAIT_FOR_ALL_THREADS_TO_STOP_SECONDS = 5.0
WAIT_FOR_INBOX_PACKET_SECONDS = 0.1
BACKOFF_MAX_SECONDS = 5.0

FRAME_SIZE = 4096
REAL_BANDWIDTH = 833_000_000


class InvalidPacketType(Exception):
    pass


class Role(ABC):
    """
    Represents a participant's role (e.g. "Alice from Quantum Algorithm", "Bob from Post Processing")
    """

    def __init__(self, receive_data_socket_info: ConnectionInfo, bandwidth_limit_megabytes_per_second: float = None,
                 inbox_capacity_megabytes: float = None, outbox_capacity_megabytes: float = None,
                 mock_receive_socket: Mock = None, mock_send_socket: Mock = None, frame_size_bytes: int = FRAME_SIZE,
                 latency_seconds: float = None, is_client: bool = False):
        """
        :param receive_data_socket_info: the (IP, Port) to which a peer may connect to if they want to exchange packet with this role instance.
        :param bandwidth_limit_megabytes_per_second: (Optional) the maximum amount of packet (in megabytes) that can be sent by this role in a second
        :param inbox_capacity_megabytes: (Optional) The maximum capacity of this role's inbox in megabytes
        :param outbox_capacity_megabytes: (Optional) The maximum capacity of this role's outbox in megabytes
        :param mock_receive_socket: (Optional | Testing purposes) useful for mocking the socket that receives packets.
        :param mock_send_socket: (Optional | Testing purposes) useful for mocking the socket that sends packets.
        :param frame_size_bytes: (Optional) defines the size (in bytes) of fractions of data that will be sent
        :param latency_seconds: (Optional | Testing purposes) delay between sending data and receiving data 
        """

        self.is_client = is_client
        self._authentication_size = 0
        self._frame_size_bytes = frame_size_bytes
        # Use 4 bytes for header to support payloads up to 4GB (increased from dynamic calculation)
        self._frame_size_header_size_in_bytes = ceil(ceil(math.log2(self._frame_size_bytes)) / 8) # old calculation: gave only 2 bytes for 4KB frames
        # self._frame_size_header_size_in_bytes = 4  # Supports up to 4,294,967,295 bytes
        self._round_trip_time = 2 * latency_seconds if latency_seconds else 0
        self.total_time_sleep_inbox = 0
        self.total_time_sleep_outbox = 0
        self.number_sleeps_outbox = 0
        self.number_sleeps_inbox = 0
        self.number_sleeps_inbox = 0
        self.number_messages_frame = 0
        self._set_up_logger()

        self._mock_receive_socket = mock_receive_socket
        self._mock_connect_socket = mock_send_socket

        if bandwidth_limit_megabytes_per_second:
            self._bandwidth_limit_bytes_per_second = bandwidth_limit_megabytes_per_second * MEGABYTES_IN_BYTES
            self._real_bandwidth = REAL_BANDWIDTH * MEGABYTES_IN_BYTES
        else:
            self._bandwidth_limit_bytes_per_second = None

        # Information for setting up sockets
        self._receive_data_socket_info = receive_data_socket_info

        # Queues to insert and receive data (bytes)
        if inbox_capacity_megabytes is None:
            self._inbox = queue.Queue()
        else:
            self._inbox = ByteQueue(inbox_capacity_megabytes * MEGABYTES_IN_BYTES)

        self.outbox_capacity = outbox_capacity_megabytes
        if self.outbox_capacity is None:
            self._outbox = queue.Queue()
        else:
            self._outbox = ByteQueue(self.outbox_capacity * MEGABYTES_IN_BYTES)

        # Queues that handle data frames
        self._frame_inbox = queue.Queue()
        self._frame_outbox = queue.Queue()

        # Peer information setup
        self._peer_connection_info = None

        # Threads to exchange payloads
        #self._thread_stop_flag = False
        self._thread_stop_flag = threading.Event()
        self._last_exception = None
        self._threads = []
        self._bytes_read = 0
        self._total_bytes_expected = 0
        self._reassembled_data = bytearray()
        self._start_fill_frame_inbox_thread()

        # Representation of the current time for each message sent by outbox - (id-message - counter, process_time, Propagation Delay (single trip), Transfer Time, total (process_time + PD + TT))
        #self.message_time_lst = []
        self.single_trip_delay = latency_seconds if latency_seconds else 0
        self.number_messages_sent = 0
        self.number_messages_received = 0
        self.number_message_frames_sent = 0
        self.number_message_frames_received = 0
        #This saves the time message progressively, meaning that each message sent, the initial time is register as the simulation of the previous send
        #self.message_time_lst_progressive = []


        self.message_frame_received = []
        self.message_frame_sent = []
        self.message_received = []
        self.message_sent = []

        # Variable to keep all sum all it took transferring the data
        self.total_time_transfer = 0

        # Represents the total Time it is waiting for getting message (identical to waiting information to send, or idle)
        self.total_waiting_get_outbox = 0

        #self.start_time = time.process_time()
        #self.artificial_time_tick = time.process_time()
        #self.previous_pivot_time_to_transfer = 0
        #self.previous_pivot_tick_time = 0

        #self.start_time_time = time.time()
        self.time_to_transfer = 0
        self.latency = latency_seconds
        self.total_time_latency_per_message = 0
        self.total_transfer_and_latency = 0

        self.time_to_transfer_frame = 0
        self.total_time_latency_per_frame = 0
        self.total_transfer_and_latency_frame = 0

        self.number_messages_put_in_frame_outbox = 0
        self.total_size_bytes_put_in_frame_outbox = 0
        self.total_size_bytes_put_in_frame_outbox_clean = 0
        self.total_size_w_mac_frames = 0
        self.total_size_w_mac_messages = 0
        self.total_size_wo_mac_frames = 0

        self.thread_counter_message = {}



    @staticmethod
    def get_instance(receive_data_socket_info: ConnectionInfo, bandwidth_limit_megabytes_per_second: float = None,
                     inbox_capacity_megabytes: float = None, outbox_capacity_megabytes: float = None,
                     mac_config: MAC_Config = None,
                     mock_receive_socket: Mock = None, mock_send_socket: Mock = None, frame_size_bytes: int = FRAME_SIZE,
                     latency_seconds: float = None, is_client: bool = False):
        """
        Gets a representation of a participant's role (e.g. "Alice from Quantum Algorithm", "Bob from Post Processing")

        :param receive_data_socket_info: the (IP, Port) to which a peer may connect to if they want to exchange packet with this role instance.
        :param bandwidth_limit_megabytes_per_second: (Optional) the maximum amount of packet (in megabytes) that can be sent by this role in a second
        :param inbox_capacity_megabytes: (Optional) The maximum capacity of this role's inbox in megabytes
        :param outbox_capacity_megabytes: (Optional) The maximum capacity of this role's outbox in megabytes
        :param mac_config: (Optional) the configuration to enable Message Authentication Codes (MAC) for this role
        :param mock_receive_socket: (Optional | Testing purposes) useful for mocking the socket that receives packets.
        :param mock_send_socket: (Optional | Testing purposes) useful for mocking the socket that sends packets.
        :param frame_size_bytes: (Optional) defines the size (in bytes) of fractions of data that will be sent
        :param latency_seconds: (Optional | Testing purposes) delay between sending data and receiving data 
        """
        if mac_config is not None:
            return MACEnabledRole(receive_data_socket_info, bandwidth_limit_megabytes_per_second,
                                  inbox_capacity_megabytes, mac_config, outbox_capacity_megabytes,
                                  mock_receive_socket, mock_send_socket, frame_size_bytes, latency_seconds, is_client)
        return MACDisabledRole(receive_data_socket_info, bandwidth_limit_megabytes_per_second,
                               inbox_capacity_megabytes, outbox_capacity_megabytes,
                               mock_receive_socket, mock_send_socket, frame_size_bytes, latency_seconds, is_client)

    @property
    def peer_connection_info(self) -> ConnectionInfo:
        """
        The (IP, Port) to the peer that this instance will send packet to. For example, if this is "Alice", the peer is "Bob". Only one peer is allowed per role.
        """
        return self._peer_connection_info

    @peer_connection_info.setter
    def peer_connection_info(self, peer_connection_info: ConnectionInfo) -> None:
        """
        :param peer_connection_info: The (IP, Port) to the peer that this instance will send packet to. For example, if this is "Alice", the peer is "Bob". Only one peer is allowed per role.
        """
        if self._peer_connection_info is None:
            self._peer_connection_info = peer_connection_info
            self._start_empty_frame_outbox_thread()
            # Start a thread to check for timeouts
            self._start_rebuild_data_thread()
            self._start_put_in_frame_outbox_thread()
        else:
            raise RoleAlreadyHasPeerException()

    def put_in_outbox(self, data: bytes, block: bool = True, timeout: float | None = None) -> None:
        """
        Inserts data into the outbox to send to the peer, respecting byte capacity constraints.

        :param data: The data (bytes) to be put in the outbox.
                     If the data size exceeds 'Role.outbox_capacity' (if it is not None), a ValueError is raised.
        :param block: If True (default), waits until enough space is available in the outbox.
                      If False, attempts immediate insertion and raises queue.Full if there's insufficient space.
                      If True and 'timeout' is specified, waits up to 'timeout' seconds for space,
                      then raises queue.Full if space isn't available.
        :param timeout: Maximum wait time (in seconds) if 'block' is True. Ignored if 'block' is False.
        :raises ValueError: If the data size exceeds the outbox's maximum capacity.
        :raises queue.Full: If the outbox lacks space for the data and 'block' is False or timeout expires.
        """
        #print(f"PUT IN OUTBOX: {self._frame_outbox.qsize()} | {self._frame_outbox.empty()}")
        self._outbox.put(data, block, timeout)
        if self._bandwidth_limit_bytes_per_second is not None:
            time_to_transfer = len(data) / self._bandwidth_limit_bytes_per_second
            self.time_to_transfer += time_to_transfer
            self.total_transfer_and_latency += time_to_transfer + (self.latency if self.latency else 0)

        #self.message_sent.append(("S", self.number_messages_sent, time.process_time(), len(data)))
        self.message_sent.append(("S", self.number_messages_sent, time.perf_counter(), len(data)))
        self.number_messages_sent += 1
        self.total_time_latency_per_message += self.latency if self.latency else 0
        # packet_size_w_mac = self._get_sendable_packet(data)
        # self.total_size_w_mac_messages += len(packet_size_w_mac)

            #thread_id, message_inside = pickle.loads(data)

            #number_messages_in_thread = threads_counter.set_default




    def get_from_inbox(self, block: bool = True, timeout: float | None = None) -> bytes | None:
        """
        Retrieves and removes an item from the inbox.

        :param block: If True (default), waits until data is available.
                      If False, tries to get data immediately and raises queue.Empty if the inbox is empty.
                      If True and 'timeout' is specified, waits up to 'timeout' seconds for data,
                      then raises queue.Empty if data isn't available.
        :param timeout: Maximum wait time (in seconds) if 'block' is True. Ignored if 'block' is False.
        :raises queue.Empty: If the inbox is empty and 'block' is False or if 'block' is True and 'timeout' expires.
        :return: The retrieved data (bytes) from the inbox.
        """

        #print(f"GET FROM INBOX: {self._frame_inbox.qsize()} | {self._frame_inbox.empty()}")

        return self._inbox.get(block, timeout)

    def clean(self):
        """Shuts down a role, stopping all of its threads."""
        self._stop_all_threads()

    def total_transfer_time_estimation(self, num_threads: int, message_size_bytes: int) -> float:
        """
        Returns a rough estimation (not considering processing time, TCP congestion control details, MAC performance impact)
        of the time it takes for n (num_threads) threads to send each b (message_size_bytes) bytes across the channel.
        :param num_threads: the number of sender threads
        :param message_size_bytes: the size of the payload of each message send by each thread
        :return: seconds taken to deliver the payloads
        """
        bandwidth_limit_Bps = self._bandwidth_limit_bytes_per_second
        total_bytes_sent = num_threads * message_size_bytes
        total_time_seconds = (total_bytes_sent / bandwidth_limit_Bps) + ceil(total_bytes_sent /
                                                                             self._frame_size_bytes) * self._round_trip_time
        return total_time_seconds

    def _put_in_frame_outbox_thread(self) -> None:
        while not self._thread_stop_flag.is_set():
            #print(f"OUTBOX: {self._outbox.qsize()} | {self._outbox.empty()}")
            data = self._outbox.get(block=True)

            if self._peer_connection_info is None:
                raise RoleHasNoPeerException()

            total_data_size = len(data)

            if total_data_size == 0:
                raise ValueError("Tried sending empty byte.")

            # Send total size packet
            self._try_putting_packet_in_frame_outbox(convert_int_to_bytes(total_data_size))

            self.number_messages_put_in_frame_outbox += 1
            self.total_size_bytes_put_in_frame_outbox += len(convert_int_to_bytes(total_data_size))
            self.total_size_bytes_put_in_frame_outbox_clean += total_data_size

            self._logger.debug("Sent total size packet")

            self._bytes_sent = 0
            # Then split into frames:
            for offset in range(0, total_data_size, self._frame_size_bytes):
                end_offset = min(offset + self._frame_size_bytes, total_data_size)
                self._try_putting_packet_in_frame_outbox(data[offset:end_offset])
                self._bytes_sent += end_offset - offset
            self._logger.debug(f"Put all packets in frame outbox. Total size:"
                               f" {round(self._bytes_sent / 1_000_000, 3)} MB")

    def _try_putting_packet_in_frame_outbox(self, payload: bytes) -> None:
        times_repeated = 0
        while times_repeated < MAX_PUT_PACKET_TRIES:
            try:
                self._frame_outbox.put(payload, block=True,
                                       timeout=self._exponential_backoff(times_repeated, PACKET_SEND_TIMEOUT))

                return
            except queue.Full:
                times_repeated += 1
        raise ValueError(f"Failed to send packet: Queue is Full")

    def _set_up_logger(self):
        self._logger = logging.getLogger(__name__)
        if not self._logger.hasHandlers():
            self._logger.setLevel(logging.WARNING)
            console_handler = logging.StreamHandler()
            self._logger.addHandler(console_handler)

    def _start_empty_frame_outbox_thread(self):
        outbox_thread = threading.Thread(target=self._empty_frame_outbox, daemon=True)
        self._threads.append(outbox_thread)
        outbox_thread.start()

    def _start_fill_frame_inbox_thread(self):
        fill_inbox_thread = threading.Thread(target=self._fill_frame_inbox, daemon=True)
        self._threads.append(fill_inbox_thread)
        fill_inbox_thread.start()

    def _start_rebuild_data_thread(self):
        rebuild_data_thread = threading.Thread(target=self._rebuild_data_thread, daemon=True)
        self._threads.append(rebuild_data_thread)
        rebuild_data_thread.start()

    def _start_put_in_frame_outbox_thread(self):
        put_in_frame_outbox_thread = threading.Thread(target=self._put_in_frame_outbox_thread, daemon=True)
        self._threads.append(put_in_frame_outbox_thread)
        put_in_frame_outbox_thread.start()

    def _empty_frame_outbox(self):
        try:
            if self._mock_connect_socket:
                self._process_empty_frame_outbox(self._mock_connect_socket)
            else:
                with create_connect_socket() as s:
                    self._process_empty_frame_outbox(s)
        except Exception as e:
            self._logger.error(f"Error sending frame_outbox payloads - {e}")
            if not self._thread_stop_flag.is_set():
                self._stop_all_threads()
                self._save_and_raise_exception(e)

    def _process_empty_frame_outbox(self, socket):
        self._peer_connection_socket = socket
        self._peer_connection_socket.connect(self._peer_connection_info.to_tuple())
        conn_info = pickle.dumps(self._receive_data_socket_info)
        socket.sendall(conn_info)

        previous_pivot_tick_time = 0#time.process_time() - self.start_time
        previous_pivot_time_to_transfer = 0#time.process_time() - self.start_time
        #time_tick_artificial = time.process_time() - self.start_time
        #time_tick = time.process_time() - self.start_time

        while not self._thread_stop_flag.is_set():
            times_tried = 0
            try:
                before_getting_pack = time.process_time()
                packet = self._frame_outbox.get(block=True, timeout=self._exponential_backoff(times_tried,
                                                                                              FRAME_OUTBOX_PACKET_SEND_TIMEOUT))
                after_getting_pack = time.process_time()
                self.total_waiting_get_outbox += after_getting_pack - before_getting_pack
                self.total_size_wo_mac_frames += len(packet)

                packet = self._get_sendable_packet(packet)
                times_tried = 0
                if self._bandwidth_limit_bytes_per_second and self._real_bandwidth > self._bandwidth_limit_bytes_per_second:
                    time_to_send = len(packet) / self._bandwidth_limit_bytes_per_second
                    real_time_to_send = len(packet) / self._real_bandwidth
                    time_to_sleep = time_to_send - real_time_to_send
                    time_to_sleep_aux = time_to_send
                    #sleep(time_to_sleep)

                    time_to_send_aux = len(packet) / self._bandwidth_limit_bytes_per_second
                    #time_to_sleep_aux = time_to_send - real_time_to_send
                    #time_tick = time.process_time()
                    #time_tick = datetime.now() - self.start_time
                    #time_tick = time_tick_artificial + time.process_time()
                    #time_tick = time.process_time() - self.start_time
                    time_tick = previous_pivot_tick_time
                    #self.total_time_sleep_outbox += time_to_sleep
                    self.total_time_sleep_outbox += time_to_send_aux
                    self.number_sleeps_outbox += 1
                    self.total_time_transfer += time_to_send_aux
                    #self.message_time_lst.append((self.number_messages, time_tick,  self.single_trip_delay, time_to_send_aux, time_tick + self.single_trip_delay + time_to_send_aux))
                    #self.message_time_lst.append((self.number_messages, time_tick, self.single_trip_delay, time_to_send_aux,
                                                #time_tick + timedelta(self.single_trip_delay) + timedelta(time_to_send_aux)))
                    self.time_to_transfer_frame += time_to_send_aux
                    self.total_time_latency_per_frame += self.latency if self.latency else 0
                    self.total_transfer_and_latency_frame += time_to_send_aux + (self.latency if self.latency else 0)

                self.total_size_w_mac_frames += len(packet)
                self.number_message_frames_sent += 1
                self.message_frame_sent.append(("S", self.number_message_frames_sent - 1, time.process_time(), len(packet)))

                self._peer_connection_socket.sendall(packet)

            except queue.Empty:
                times_tried += 1

    def _fill_frame_inbox(self):
        try:
            if self._mock_receive_socket:
                self._fill_frame_inbox_process(self._mock_receive_socket)
            else:
                with create_receive_socket(self._receive_data_socket_info) as s:
                    self._fill_frame_inbox_process(s)
        except Exception as e:
            self._logger.error(f"Error receiving _frame_inbox payloads - {e}")
            if not self._thread_stop_flag.is_set():
                self._stop_all_threads()
                self._save_and_raise_exception(e)

    def _fill_frame_inbox_process(self, socket):
        conn, addr = socket.accept()
        with conn:
            conn_info = pickle.loads(conn.recv(4096))
            if self.peer_connection_info is None:
                self.peer_connection_info = conn_info
            conn.settimeout(FILL_INBOX_TIMEOUT_SECONDS)
            receive_buffer = bytearray()



            while not self._thread_stop_flag.is_set():
                try:
                    start_sleep_inbox = time.process_time()
                    _bytes = conn.recv(self._frame_size_header_size_in_bytes)
                    end_sleep_inbox = time.process_time()
                    self.total_time_sleep_inbox += end_sleep_inbox - start_sleep_inbox
                    if self._round_trip_time:
                        #sleep(self._round_trip_time)
                        #self.total_time_sleep_inbox += self._round_trip_time
                        self.number_sleeps_inbox += 1
                    if not _bytes:
                        continue
                    receive_buffer.extend(_bytes)
                    while len(receive_buffer) >= self._frame_size_header_size_in_bytes:
                        complete_header = receive_buffer[:self._frame_size_header_size_in_bytes]
                        excess_bytes = receive_buffer[self._frame_size_header_size_in_bytes:]
                        payload_size = int.from_bytes(complete_header, byteorder='little')
                        if len(excess_bytes) >= payload_size + self._authentication_size:
                            payload = excess_bytes[:payload_size + self._authentication_size]
                            extra_bytes = excess_bytes[payload_size + self._authentication_size:]
                            #Commented for Testing purposes - Should be UNCOMMENTED.
                            self._verify_authentication(complete_header + payload)
                            self._frame_inbox.put(payload)
                            receive_buffer = extra_bytes

                            self.message_frame_received.append(("R", self.number_message_frames_received, time.process_time(), len(payload)))
                            self.number_message_frames_received += 1
                        else:
                            break
                except TimeoutError:
                    continue
                except Exception as e:
                    self._logger.error(f"client connection error - {e}")
                    raise e

    def _save_and_raise_exception(self, e):
        self._last_exception = e
        raise self._last_exception

    def _stop_all_threads(self):
        #self._thread_stop_flag = True
        self._thread_stop_flag.set()
        if len(self._threads) > 0:
            while not self._thread_stop_flag and any(th.is_alive() for th in self._threads):
                sleep(WAIT_FOR_ALL_THREADS_TO_STOP_SECONDS)

    @staticmethod
    def _exponential_backoff(times_repeated: int, timeout: float) -> float:
        if times_repeated > 0:
            return min(BACKOFF_MAX_SECONDS, 2 * times_repeated * timeout)
        return timeout

    def _rebuild_data_thread(self) -> None:

        while not self._thread_stop_flag.is_set():
            packet = None
            if self.is_client:
                print("Waiting for the control packet.")
            trials = 0
            while not self._thread_stop_flag.is_set() and packet is None:
                try:
                    packet = self._frame_inbox.get(block=True,
                                                   timeout=min(BACKOFF_MAX_SECONDS, self._exponential_backoff(trials,
                                                                                              WAIT_FOR_INBOX_PACKET_SECONDS)))
                    trials = 0
                except queue.Empty:
                    trials += 1
                    self._logger.debug("Control Packet - Frame inbox empty. Trying again!")
                    continue
            if self._thread_stop_flag.is_set():
                return
            # This is our control packet. Extract total data size.
            if self.is_client:
                print("Got control packet")
            self._total_bytes_expected = convert_from_bytes_to_int(self._get_payload_from_packet(packet))
            if self.is_client:
                print(f"Expecting {self._total_bytes_expected} bytes")
            # Phase 2: Reassemble the data payloads.
            self._reassembled_data = bytearray()
            self._bytes_read = 0

            # Wait until reassembly is complete.
            if self.is_client:
                print("Waiting until reassembly is complete")

            trials = 0
            while not self._thread_stop_flag.is_set() and self._bytes_read < self._total_bytes_expected:
                try:
                    packet = self._frame_inbox.get(block=True,
                                                   timeout=min(BACKOFF_MAX_SECONDS, self._exponential_backoff(trials,
                                                                                                              WAIT_FOR_INBOX_PACKET_SECONDS)))
                    trials = 0

                except queue.Empty:
                    trials += 1
                    if self.is_client:
                        print("Rebuilding - Frame inbox empty. Trying again!")
                    continue
                payload = self._get_payload_from_packet(packet)
                bytes_len = len(payload)
                self._reassembled_data.extend(payload)
                self._bytes_read += bytes_len
            if self._thread_stop_flag.is_set():
                return
            extra_bytes_read = self._bytes_read - self._total_bytes_expected
            if extra_bytes_read:
                self._reassembled_data = self._reassembled_data[:-extra_bytes_read]
            self._inbox.put(self._reassembled_data, block=False)

            #PF
            #self.message_received.append(("R", self.number_messages_received, time.process_time(), len(self._reassembled_data)))
            self.message_received.append(("R", self.number_messages_received, time.perf_counter(), len(self._reassembled_data)))
            self.number_messages_received += 1
            #print(f"INBOX: {self._inbox.qsize()}")

    def _get_sendable_packet(self, packet: bytes) -> bytes:
        packet = self._add_header(packet)
        packet = self._add_authentication(packet)
        return packet

    def _add_header(self, packet: bytes) -> bytes:
        packet_size = len(packet)
        payload_size_header = packet_size.to_bytes(self._frame_size_header_size_in_bytes, byteorder='little', signed=False)[:self._frame_size_header_size_in_bytes]
        return bytearray(payload_size_header + packet)

    @abstractmethod
    def _get_payload_from_packet(self, packet: bytes) -> bytes:
        pass

    @abstractmethod
    def _add_authentication(self, packet_bytes: bytes) -> bytes:
        pass

    @abstractmethod
    def _verify_authentication(self, packet_bytes: bytes) -> None:
        pass


class MACDisabledRole(Role):

    def __init__(self, receive_data_socket_info: ConnectionInfo, bandwidth_limit_megabytes_per_second: float,
                 inbox_capacity_megabytes: float, outbox_capacity_megabytes: float = None,
                 mock_receive_socket: Mock = None, mock_send_socket: Mock = None, frame_size_bytes: int = FRAME_SIZE,
                 latency_seconds: float = None, is_client: bool = False):
        super().__init__(receive_data_socket_info, bandwidth_limit_megabytes_per_second, inbox_capacity_megabytes,
                         outbox_capacity_megabytes, mock_receive_socket, mock_send_socket, frame_size_bytes,
                         latency_seconds, is_client)

    def _get_payload_from_packet(self, packet: bytes) -> bytes:
        return packet

    def _add_authentication(self, packet_bytes: bytes) -> bytes:
        return packet_bytes

    def _verify_authentication(self, packet_bytes: bytes) -> None:
        pass


class MACEnabledRole(Role):
    def __init__(self, receive_data_socket_info: ConnectionInfo, bandwidth_limit_megabytes_per_second: float,
                 inbox_capacity_megabytes: float, mac_config: MAC_Config, outbox_capacity_megabytes: float = None,
                 mock_receive_socket: Mock = None, mock_send_socket: Mock = None, frame_size_bytes: int = FRAME_SIZE,
                 latency_seconds: float = None, is_client: bool = False):
        super().__init__(receive_data_socket_info, bandwidth_limit_megabytes_per_second, inbox_capacity_megabytes,
                         outbox_capacity_megabytes, mock_receive_socket, mock_send_socket, frame_size_bytes,
                         latency_seconds, is_client)
        self._mac = mac_config.get_mac()
        self._authentication_size = MAC_SIZE_BYTES + TIMESTAMP_SIZE_BYTES

    def _get_payload_from_packet(self, packet: bytes) -> bytes:
        return packet[:-(TIMESTAMP_SIZE_BYTES + MAC_SIZE_BYTES)]

    def _add_authentication(self, packet_bytes: bytes) -> bytes:
        _timestamp = convert_float_to_bytes(timestamp())
        packet_bytes = packet_bytes + _timestamp
        _mac = self._calculate_mac(packet_bytes)
        return packet_bytes + _mac

    def _verify_authentication(self, packet_bytes: bytes) -> None:
        if not self._has_valid_mac(bytearray(packet_bytes)):
            raise ValueError("Received packet with invalid MAC")
        if not self._has_valid_timestamp(bytearray(packet_bytes)):
            raise ValueError("Received packet with invalid timestamp")

    def _calculate_mac(self, packet_bytes: bytes) -> bytes:
        mac = self._mac.copy()
        mac.update(packet_bytes)
        return mac.finalize()

    def _has_valid_mac(self, packet: bytearray) -> bool:
        try:
            _mac = self._mac.copy()
            _mac.update(bytes(packet[:-MAC_SIZE_BYTES]))
            _mac.verify(bytes(packet[-MAC_SIZE_BYTES:]))
            return True
        except InvalidSignature:
            return False

    @staticmethod
    def _has_valid_timestamp(frame: bytearray) -> bool:
        frame_timestamp = convert_from_bytes_to_float(
            frame[-(TIMESTAMP_SIZE_BYTES + MAC_SIZE_BYTES):-MAC_SIZE_BYTES])
        return ceil(timestamp() - frame_timestamp) < MAX_CREATION_TIMESTAMP_AGE_SECONDS
