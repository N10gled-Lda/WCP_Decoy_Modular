import logging
import socket
from time import perf_counter
import time

from classical_communication_channel.communication_channel.connection_info import ConnectionInfo

INTEGER_LENGTH_IN_BYTES = 4

_MILLISECOND_PRECISION_DECIMAL_PLACES = 3

_MICROSECOND_PRECISION_DECIMAL_PLACES = 6

PRECISION_DECIMAL_PLACES = _MILLISECOND_PRECISION_DECIMAL_PLACES

FLOAT_TO_INT_FACTOR = pow(10, PRECISION_DECIMAL_PLACES)

INT_TO_FLOAT_FACTOR = 1 / FLOAT_TO_INT_FACTOR


BLOCK_OFFSET_LENGTH_BYTES = INTEGER_LENGTH_IN_BYTES

TOTAL_SIZE_LENGTH_BYTES = INTEGER_LENGTH_IN_BYTES

FLOAT_SIZE_BYTES = INTEGER_LENGTH_IN_BYTES

TIMESTAMP_SIZE_BYTES = FLOAT_SIZE_BYTES

MEGABYTES_IN_BYTES = 1_000_000

MAC_SIZE_BYTES = 16

BYTES_IN_MEGABYTES = 1 / MEGABYTES_IN_BYTES

#MAX_CREATION_TIMESTAMP_AGE_SECONDS = 120  # 2 minutes
MAX_CREATION_TIMESTAMP_AGE_SECONDS = 300  # 5 minutes


def create_receive_socket(connection: ConnectionInfo) -> socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # enable address reuse
    ip, port = -1, -1
    try:
        ip, port = connection.to_tuple()
        sock.bind((ip, port))
        sock.listen()
        return sock
    except Exception as e:
        logging.error(f"Error binding socket  - {e}\nIP: \'{ip}\'\nPort: \'{port}\'")
        sock.close()
        raise e


def create_connect_socket() -> socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    return sock


def convert_int_to_bytes(val: int) -> bytes:
    return val.to_bytes(length=INTEGER_LENGTH_IN_BYTES, byteorder='little', signed=True)


def convert_from_bytes_to_int(data: bytes, start_index_inclusive: int = 0) -> int:
    return int.from_bytes(data[start_index_inclusive:start_index_inclusive + INTEGER_LENGTH_IN_BYTES],
                          byteorder='little', signed=True)


def convert_float_to_bytes(val: float) -> bytes:
    #return convert_int_to_bytes(int(val * FLOAT_TO_INT_FACTOR))
    return convert_int_to_bytes(int(val)) # Changed to fix overflow problem of int too big - using different computers.

def convert_from_bytes_to_float(data: bytes) -> float:
    #return convert_from_bytes_to_int(data, 0) * INT_TO_FLOAT_FACTOR
    return convert_from_bytes_to_int(data, 0) # Changed to fix overflow problem of int too big - using different computers.

def timestamp():
    #return perf_counter()
    return time.time() # Changed to fix different computers timestamp difference

def wait_synchronously(wait_seconds: float):
    deadline = timestamp() + wait_seconds
    while timestamp() < deadline:
        pass


def standardize_time(time_in_seconds: float) -> float:
    return max(round(time_in_seconds, PRECISION_DECIMAL_PLACES), 0)


def generate_connection_info(local_ip: str = '') -> ConnectionInfo:
    port = _get_free_port(local_ip)
    return ConnectionInfo(local_ip, port)


def _get_free_port(local_ip):
    with create_receive_socket(ConnectionInfo(local_ip, 0)) as s:
        host, port = s.getsockname()
    return port
