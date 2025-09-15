import csv
import datetime
import sys
from pathlib import Path

import threading
from random import randbytes

from classical_communication_channel.communication_channel.common import INTEGER_LENGTH_IN_BYTES, convert_int_to_bytes, timestamp, \
    convert_from_bytes_to_int, MAC_SIZE_BYTES
from classical_communication_channel.communication_channel.connection_info import ConnectionInfo
from classical_communication_channel.communication_channel.mac_config import MAC_Config, MAC_Algorithms
from classical_communication_channel.communication_channel.mock_socket import MockSockets
from classical_communication_channel.communication_channel.role import Role


def send_thread(_role: Role, _sent_queues: list[list], _thread_num: int, _seq_num: int, _message: bytes):
    _t_init = timestamp()
    _role.put_in_outbox(_message)
    _sent_queues[_thread_num].append((_seq_num, len(_message), _t_init))


def build_message(_thread_num: int, _seq_num: int, _size: int) -> bytes:
    _size = round(_size)
    return convert_int_to_bytes(_thread_num) + convert_int_to_bytes(_seq_num) + randbytes(
        _size - 2 * INTEGER_LENGTH_IN_BYTES)


def _run_server_side(server, send_times, client_receive_event, num_threads, message_size_bytes, sent_messages):
    size_str = str(message_size_bytes / 1000)
    # Create logfile
    data_filename = Path(
        'server_' + f'{num_threads}t_{size_str}_kb_' + datetime.datetime.now(datetime.UTC).strftime(
            '%Y-%m-%dT%H%M%S') + '.csv')
    data_file = "bandwidth_server_test_logs" / data_filename
    with open(data_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Server Thread", "Server seq num", "Bytes received", "T_init"])
        sent_queues = [[] for _ in range(num_threads)]
        try:
            messages = [build_message(thread, 0, message_size_bytes) for thread in range(num_threads)]
            threads = []
            for thread in range(num_threads):
                threads.append(threading.Thread(target=send_thread,
                                                args=[server, sent_queues, thread, 0,
                                                      messages[thread]]))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            min_t_init = sys.maxsize
            for thread in range(num_threads):
                for seq_num, size, t_init in sent_queues[thread]:
                    min_t_init = min(min_t_init, t_init)
                    writer.writerow([thread, seq_num, size, t_init])
            send_times.append(min_t_init)
        except KeyboardInterrupt:
            pass
    client_receive_event.wait()
    print(f"[server]: complete: {num_threads} threads | {size_str} kB | t: {min_t_init}s")
    sent_messages.extend(messages)


def _run_client_side(client: Role, receive_times, client_receive_event, num_threads, message_size_bytes,
                     received_messages):
    client_receive_event.clear()
    size_str = str(message_size_bytes / 1000)
    # Create logfile
    data_filename = Path(
        'client_' + f'{num_threads}t_{size_str}_kb_' + datetime.datetime.now(datetime.UTC).strftime(
            '%Y-%m-%dT%H%M%S') + '.csv')
    data_file = "bandwidth_client_test_logs" / data_filename
    with open(data_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Client Thread", "Client seq num", "Bytes received", "T_arr"])
        receiver_queues = [[] for _ in range(num_threads)]
        messages_received = 0
        messages = []
        try:
            while messages_received < num_threads:
                message_received = client.get_from_inbox()
                t_arrival = timestamp()
                size = len(message_received)
                messages.append(message_received)
                size_str = str(size / 1000)
                thread = convert_from_bytes_to_int(message_received)
                seq_num = convert_from_bytes_to_int(message_received, INTEGER_LENGTH_IN_BYTES)
                receiver_queues[thread].append((seq_num, size, t_arrival))
                messages_received += 1
                print(
                    f"[client]: received message | thread: {thread} | seq_num: {seq_num} | size: {size_str} kB | t: {t_arrival}s")
            max_t_arrival = -1
            for thread in range(num_threads):
                for seq_num, size, t_arrival in receiver_queues[thread]:
                    max_t_arrival = max(max_t_arrival, t_arrival)
                    writer.writerow([thread, seq_num, size, t_arrival])
            receive_times.append(max_t_arrival)
        except KeyboardInterrupt:
            pass
    client_receive_event.set()
    print(f"[client]: complete: {num_threads} threads | {size_str} kB | t: {max_t_arrival} s")
    received_messages.extend(messages)


def run_test():
    simulation_mode = bool(input("Simulation (s) or calculation (c) mode? ").lower().strip() == "s")
    bandwidth_limit_MBps = float(input("enter bandwidth limit in MBps: "))
    frame_size = int(input("enter frame size in kB: ")) * 1_000
    latency_seconds = float(input("enter the latency (seconds): "))
    num_threads = int(input("how many threads? "))
    message_size_bytes = float(input("how much data to send in each thread in MB? ")) * 1_000_000
    using_mac = input("using MAC? (y/n) ").lower().strip() == "y"

    server_ip = '127.0.0.1'
    server_port = 4001
    server_connection_info = ConnectionInfo(server_ip, server_port)
    client_ip = '127.0.0.1'
    client_port = 4000
    client_connection_info = ConnectionInfo(client_ip, client_port)
    inbox_capacity_MB = None
    outbox_capacity_MB = None
    shared_secret_key = "" if not using_mac else randbytes(MAC_SIZE_BYTES)
    client_receive_event = threading.Event()
    mock_socket = MockSockets()
    server_receive_socket = mock_socket.alice_receive_socket
    server_send_socket = mock_socket.alice_send_socket
    client_receive_socket = mock_socket.bob_receive_socket
    client_send_socket = mock_socket.bob_send_socket
    if shared_secret_key == "":
        server = Role.get_instance(server_connection_info, bandwidth_limit_MBps, inbox_capacity_MB,
                                   outbox_capacity_MB,
                                   mock_receive_socket=server_receive_socket, mock_send_socket=server_send_socket,
                                   frame_size_bytes=frame_size, latency_seconds=latency_seconds)
    else:
        mac_configuration = MAC_Config(MAC_Algorithms.CMAC, shared_secret_key)
        server = Role.get_instance(server_connection_info, bandwidth_limit_MBps, inbox_capacity_MB,
                                   outbox_capacity_MB,
                                   mac_configuration, server_receive_socket, server_send_socket,
                                   frame_size_bytes=frame_size, latency_seconds=latency_seconds)
    if shared_secret_key == "":
        client = Role.get_instance(client_connection_info, bandwidth_limit_MBps, inbox_capacity_MB,
                                   outbox_capacity_MB,
                                   mock_receive_socket=client_receive_socket, mock_send_socket=client_send_socket,
                                   frame_size_bytes=frame_size, latency_seconds=latency_seconds)
    else:
        mac_configuration = MAC_Config(MAC_Algorithms.CMAC, shared_secret_key)
        client = Role.get_instance(client_connection_info, bandwidth_limit_MBps, inbox_capacity_MB,
                                   outbox_capacity_MB,
                                   mac_configuration, client_receive_socket, client_send_socket,
                                   frame_size_bytes=frame_size, latency_seconds=latency_seconds)

    if simulation_mode:
        _simulation(client, server, client_connection_info, client_receive_event, num_threads, message_size_bytes)
    else:
        _calculation(server, num_threads, message_size_bytes)


def _simulation(client, server, client_connection_info, client_receive_event, num_threads, message_size_bytes):
    send_times = []
    receive_times = []
    sent_messages = []
    received_messages = []
    # Must be set before sending data to the client
    server.peer_connection_info = client_connection_info
    server_side_thread = threading.Thread(target=_run_server_side,
                                          args=[server, send_times, client_receive_event, num_threads,
                                                message_size_bytes, sent_messages])
    client_side_thread = threading.Thread(target=_run_client_side,
                                          args=[client, receive_times, client_receive_event, num_threads,
                                                message_size_bytes, received_messages])
    client_side_thread.start()
    server_side_thread.start()
    server_side_thread.join()
    client_side_thread.join()
    if sent_messages == received_messages:
        print("messages match!")
    else:
        raise ValueError("messages did not match")
    for i in range(len(receive_times)):
        print(f'Sent payloads took {round(receive_times[i] - send_times[i], 3)} seconds to receive.')
    server.clean()
    client.clean()


def _calculation(server: Role, num_threads, message_size_bytes):
    total_time_seconds = server.total_transfer_time_estimation(num_threads, message_size_bytes)
    print(f"calculated a total time of {round(total_time_seconds, 3)} seconds")


if __name__ == "__main__":
    run_test()
