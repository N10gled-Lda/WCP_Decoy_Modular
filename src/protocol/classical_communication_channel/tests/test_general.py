import logging
import math
import os
import pickle
import queue
import threading
import unittest
from abc import ABC, abstractmethod
from random import randint, randbytes
from time import time, sleep

import xxhash

from communication_channel.common import (
    timestamp,
    INTEGER_LENGTH_IN_BYTES,
    BYTES_IN_MEGABYTES,
    convert_int_to_bytes,
    convert_from_bytes_to_int, generate_connection_info
)
from communication_channel.mac_config import MAC_Config, MAC_Algorithms
from classical_communication_channel.communication_channel.role import Role
from dotenv import load_dotenv

READ_SLEEP_SECONDS = 0.01

MAX_READ_ATTEMPTS = 5

# -------------------------------------------------------------------
# Constants / Configuration
# -------------------------------------------------------------------
load_dotenv()

INPUT_FILE_NAME = os.getenv("INPUT_FILE_NAME")
OUTPUT_FILE_NAME = os.getenv("OUTPUT_FILE_NAME")
INPUT_FILE_2_NAME = os.getenv("INPUT_FILE_2_NAME")
OUTPUT_FILE_2_NAME = os.getenv("OUTPUT_FILE_2_NAME")

LOCAL_IP = os.getenv("LOCAL_IP")

SHARED_SECRET_KEY = os.getenv("SHARED_SECRET_KEY").encode('utf-8')  # Must be exactly 16 bytes for CMAC
INBOX_READ_TIMEOUT = 60  # 1 Minute
BANDWIDTH_LIMIT_MBPS = 5  # 40 Mbps
INBOX_CAPACITY_MB = 5
OUTBOX_CAPACITY_MB = 5


# -------------------------------------------------------------------
# Helper functions for building payloads, sending packets, etc.
# -------------------------------------------------------------------
def build_random_int_array(size_in_bytes: int) -> list[int]:
    """
    Helper that returns an array of integers roughly filling `size_in_bytes`.
    """
    num_ints = math.ceil(size_in_bytes / INTEGER_LENGTH_IN_BYTES)
    seed = randint(0, 100)
    return [integer + seed for integer in range(num_ints)]


def send_packet(payload: bytes, sender: Role, receiver: Role) -> bytes:
    t_init = timestamp()
    put_in_inbox_thread = threading.Thread(target=sender.put_in_outbox, args=[payload])
    put_in_inbox_thread.start()
    received = read_inbox(receiver)
    t = timestamp()
    dt = t - t_init
    if dt > 0:
        payload_size_MB = len(payload) * BYTES_IN_MEGABYTES
        rate = payload_size_MB / dt
        logging.debug(
            f"took {dt} seconds to deliver {payload_size_MB} MB. "
            f"Rate: {rate} MB/s"
        )
    return received


def read_inbox(receiver: Role) -> bytes:
    logging.getLogger().warning("reading from inbox...")
    data_received = receiver.get_from_inbox()
    while not data_received:
        data_received = receiver.get_from_inbox()
    return data_received


def send_file(sender: Role, receiver: Role, file_name_at_sender: str, file_name_at_receiver: str) -> None:
    with open(file_name_at_sender, 'rb') as in_file:
        payload = in_file.read()
        message_received = send_packet(payload, sender, receiver)
        with open(file_name_at_receiver, 'wb') as out_file:
            out_file.write(message_received)


def hash_file(filepath) -> bytes:
    hasher = xxhash.xxh3_64()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.digest()


def compare_files(file1, file2) -> bool:
    return hash_file(file1) == hash_file(file2)

def multithread_write(values: list[bytes], sender: Role, idx: int, total_variables: int, num_queues: int):
    corresponding_indices = [i for i in range(total_variables) if i % num_queues == idx]
    for j in corresponding_indices:
        sender.put_in_outbox(values[j])

def multithread_read(queues: list[queue.Queue], role: Role):
    while True:
        val = read_inbox(role)
        if not val:
            break
        val_int = convert_from_bytes_to_int(val, 0)
        queues[val_int].put(val_int)



class TestRole(ABC, unittest.TestCase):

    roles = []
    channel = None

    def test_send_string(self):
        """
        Tests basic sends (strings, pickled integers, pickled lists, etc.).
        """
        alice, bob = self.create_roles()
        self.roles.append(alice)
        self.roles.append(bob)

        # Send some messages between sender and receiver
        message_sent = "hello".encode("utf-8")
        message_received = send_packet(message_sent, alice, bob)
        self.assertEqual(message_sent, message_received)

    def test_send_list(self):
        """
        Tests basic sends (strings, pickled integers, pickled lists, etc.).
        """
        alice, bob = self.create_roles()
        self.roles.append(alice)
        self.roles.append(bob)

        message_sent = [1, 0, 'hahaha', time()]
        message_sent = pickle.dumps(message_sent)
        message_received = send_packet(message_sent, alice, bob)
        self.assertEqual(message_sent, message_received)


    def test_send_bytes_list(self):
        """
        Tests basic sends (strings, pickled integers, pickled lists, etc.).
        """
        alice, bob = self.create_roles()
        self.roles.append(alice)
        self.roles.append(bob)

        message_sent = randbytes(1_000_000)
        message_received = send_packet(message_sent, alice, bob)
        self.assertEqual(message_sent, message_received)

    def test_y(self):
        """
        Tests sending a file from sender to receiver.
        """
        alice, bob = self.create_roles()
        self.roles.append(alice)
        self.roles.append(bob)
        send_file(alice, bob, INPUT_FILE_NAME, OUTPUT_FILE_NAME)
        self.assertTrue(compare_files(INPUT_FILE_NAME, OUTPUT_FILE_NAME))
        os.remove(OUTPUT_FILE_NAME)

    def test_z(self):
        """
        Example test that sends files back and forth multiple times.
        """
        alice, bob = self.create_roles()
        self.roles.append(alice)
        self.roles.append(bob)
        for i in range(5):
            send_file(alice, bob, INPUT_FILE_NAME, OUTPUT_FILE_NAME)
            self.assertTrue(compare_files(INPUT_FILE_NAME, OUTPUT_FILE_NAME))
            os.remove(OUTPUT_FILE_NAME)
            send_file(bob, alice, INPUT_FILE_2_NAME, OUTPUT_FILE_2_NAME)
            self.assertTrue(compare_files(INPUT_FILE_2_NAME, OUTPUT_FILE_2_NAME))
            os.remove(OUTPUT_FILE_2_NAME)

    def test_z_2(self):
        num_queues = 6
        variables_per_queue = 10_000
        total_variables = num_queues * variables_per_queue
        alice, bob = self.create_roles()
        self.roles.append(alice)
        self.roles.append(bob)
        queues = [queue.SimpleQueue() for _ in range(num_queues)]
        values = [convert_int_to_bytes(i % num_queues) for i in range(total_variables)]
        writer_threads = []
        reader_threads = []
        for i in range(num_queues//2):
            writer_thread = threading.Thread(target=multithread_write, args=[values, alice, i, bob, total_variables, num_queues], daemon=True)
            writer_threads.append(writer_thread)
        reader_thread = threading.Thread(target=multithread_read, args=[queues, alice], daemon=True)
        reader_threads.append(reader_thread)
        for j in range(num_queues//2, num_queues):
            writer_thread = threading.Thread(target=multithread_write, args=[values, alice, j, bob, total_variables, num_queues], daemon=True)
            writer_threads.append(writer_thread)
        reader_thread = threading.Thread(target=multithread_read, args=[queues, bob], daemon=True)
        reader_threads.append(reader_thread)
        for t in reader_threads:
            t.start()
        for t in writer_threads:
            t.start()
        all_finished = False
        while not all_finished:
            all_finished = all([not t.is_alive() for t in writer_threads]) and all(
                [not t.is_alive() for t in reader_threads])
            sleep(5)
        for k in range(num_queues):
            values_in_queue = []
            while not queues[k].empty():
                values_in_queue.append(queues[k].get())

            self.assertTrue(all([i == k for i in values_in_queue]))
            self.assertTrue(len(values_in_queue) == variables_per_queue, f"{len(values_in_queue)} != {variables_per_queue}")

    @classmethod
    def tearDown(cls):
        for role in cls.roles:
            role.clean()

    @abstractmethod
    def create_roles(self) -> tuple[Role, Role]:
        pass


class TestDefaultRole(TestRole):

    def create_roles(self) -> tuple[Role, Role]:
        alice_info = generate_connection_info(LOCAL_IP)
        bob_info = generate_connection_info(LOCAL_IP)
        alice = Role.get_instance(alice_info)
        bob = Role.get_instance(bob_info)
        alice.peer_connection_info = bob_info
        return alice, bob


class TestBandwidthLimitRole(TestRole):
    def create_roles(self) -> tuple[Role, Role]:
        alice_info = generate_connection_info(LOCAL_IP)
        bob_info = generate_connection_info(LOCAL_IP)
        alice = Role.get_instance(alice_info, BANDWIDTH_LIMIT_MBPS)
        bob = Role.get_instance(bob_info, BANDWIDTH_LIMIT_MBPS)
        alice.peer_connection_info = bob_info
        bob.peer_connection_info = alice_info
        return alice, bob


class TestCMACRole(TestRole):
    def create_roles(self) -> tuple[Role, Role]:
        alice_info = generate_connection_info(LOCAL_IP)
        bob_info = generate_connection_info(LOCAL_IP)
        mac_config = MAC_Config(MAC_Algorithms.CMAC, SHARED_SECRET_KEY)
        alice = Role.get_instance(alice_info, mac_config=mac_config)
        bob = Role.get_instance(bob_info, mac_config=mac_config)
        alice.peer_connection_info = bob_info
        bob.peer_connection_info = alice_info
        return alice, bob


class TestHMACRole(TestRole):
    def create_roles(self) -> tuple[Role, Role]:
        alice_info = generate_connection_info(LOCAL_IP)
        bob_info = generate_connection_info(LOCAL_IP)
        mac_config = MAC_Config(MAC_Algorithms.HMAC, SHARED_SECRET_KEY)
        alice = Role.get_instance(alice_info, mac_config=mac_config)
        bob = Role.get_instance(bob_info, mac_config=mac_config)
        alice.peer_connection_info = bob_info
        bob.peer_connection_info = alice_info
        return alice, bob


class TestCMACBandwidthLimitRole(TestRole):
    def create_roles(self) -> tuple[Role, Role]:
        alice_info = generate_connection_info(LOCAL_IP)
        bob_info = generate_connection_info(LOCAL_IP)
        mac_config = MAC_Config(MAC_Algorithms.HMAC, SHARED_SECRET_KEY)
        alice = Role.get_instance(alice_info, mac_config=mac_config, bandwidth_limit_megabytes_per_second=BANDWIDTH_LIMIT_MBPS)
        bob = Role.get_instance(bob_info, mac_config=mac_config, bandwidth_limit_megabytes_per_second=BANDWIDTH_LIMIT_MBPS)
        alice.peer_connection_info = bob_info
        bob.peer_connection_info = alice_info
        return alice, bob

if __name__ == '__main__':
    unittest.main()