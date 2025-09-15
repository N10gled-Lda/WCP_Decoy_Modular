# Description: Classical channel class for sending and receiving data between participants.
# The class is used to send and receive data between participants in the BB84 protocol.

import socket
import time
import pickle

# Configure logging
import logging
from logging_setup import setup_logger
# Setup logger
logger = setup_logger("BB84 Protocol Simulation Logger", logging.INFO)

CHANNEL_DELAY = 0.001

class ClassicalChannel:
    """
    Classical channel that handles transmission (sending and receiving) of data between participants.
    
    Attributes:
    - CHUNK_SIZE: int - Size of the chunk of data for sending and receiving (default: 4096)
    - END_MARKER: bytes - Marker to indicate the end of the data
    """
    def __init__(self, chunk_size: int=4096):
        self.CHUNK_SIZE = chunk_size
        self.END_MARKER = b'<END>'

    def send_data(self, data: any, connection: socket.socket, max_retries: int=3) -> None:
        """Sends data through the connection to a participant with end marker. Retry a maximum number of times."""
        for attempt in range(max_retries):
            try:
                # Send data in chunks
                data = pickle.dumps(data)
                connection.sendall(data)
                # Send end marker
                connection.sendall(self.END_MARKER)
                time.sleep(CHANNEL_DELAY)  # Simulate transmission delay
                break
            except Exception as e:
                logger.error(f"Error sending data: {e}")
                if attempt == max_retries - 1:
                    raise

    def send_data_th(self, data: any, thread_id, connection: socket.socket, max_retries: int=3) -> None:
        """Sends data through the connection to a participant with end marker. Retry a maximum number of times."""
        for attempt in range(max_retries):
            try:
                # Send data in chunks
                data = pickle.dumps(data)
                data = thread_id + data
                print(f"Sending data: {data}")
                connection.sendall(data)
                # Send end marker
                connection.sendall(self.END_MARKER)
                time.sleep(CHANNEL_DELAY)  # Simulate transmission delay
                break
            except Exception as e:
                logger.error(f"Error sending data: {e}")
                if attempt == max_retries - 1:
                    raise

    def receive_data(self, connection: socket.socket) -> any:
        """Receives data using buffer until end marker is found."""
        try:
            buffer = bytearray()
            while True:
                # Receive chunk of data
                chunk = connection.recv(self.CHUNK_SIZE)
                if not chunk:
                    break
                
                buffer.extend(chunk)
                
                print(f"Received data: {buffer}")
                # Check for end marker
                if buffer.endswith(self.END_MARKER):
                    print(f"Received data: {buffer}")
                    buffer = buffer[:-len(self.END_MARKER)]
                    break
            print(f"Received data: {buffer}")
            return pickle.loads(buffer)
        
        except Exception as e:
            logger.error(f"Error receiving data: {e}")
            raise

    def receive_data_th(self, connection: socket.socket, size_thread_id) -> any:
        """Receives data using buffer until end marker is found."""
        try:
            buffer = bytearray()
            while True:
                # Receive chunk of data
                chunk = connection.recv(self.CHUNK_SIZE)
                if not chunk:
                    break
                
                buffer.extend(chunk)
                
                # Check for end marker
                if buffer.endswith(self.END_MARKER):
                    buffer = buffer[:-len(self.END_MARKER)]
                    break
            
            # Extract thread id and data
            thread_id_byte = bytes(buffer[:size_thread_id])
            data = buffer[size_thread_id:]
            data_load = pickle.loads(data)
            return thread_id_byte, data_load
        
        except Exception as e:
            logger.error(f"Error receiving data: {e}")
            raise