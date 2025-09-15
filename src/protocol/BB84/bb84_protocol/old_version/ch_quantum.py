# Description: Quantum channel class for simulating qubit transmission between participants.
# The class is used to simulate the transmission of qubits between participants in the BB84 protocol.

import socket
import time
import pickle
import random
from typing import Tuple

# Configure logging
import logging
from logging_setup import setup_logger
# Setup logger
logger = setup_logger("BB84 Protocol Simulation Logger", logging.INFO)

from bb84_protocol.old_version.qubit import Qubit

CHANNEL_DELAY = 0.00001

class QuantumChannel:
    """
    Handles data transmission of qubits between participants over a quantum channel
    This class is used to simulate the transmission with a given loss rate.

    Attributes:
    - loss_rate: float - The probability of losing a qubit during transmission
    """
    def __init__(self, loss_rate: int=0.9):
        """
        Initialize the quantum channel with a given loss rate.
        :param loss_rate: The probability of losing a qubit during transmission (default: 0.9)
        """
        self.loss_rate = loss_rate

    def send_qubit(self, qubit: Qubit, time_stamp: float, connection: socket.socket) -> None:
        """Sends a qubit to a participant with some loss."""
        try:
            if random.random() < self.loss_rate:
                logger.debug3("Qubit sent: %s", qubit.show())
                data = pickle.dumps((qubit, time_stamp))
                connection.sendall(data)
            else:
                logger.debug3("Qubit lost.")
            time.sleep(CHANNEL_DELAY)  # Simulate transmission delay
        except ConnectionAbortedError as error:
            logger.error(f"Connection aborted: {error}")
        except Exception as error:
            logger.error(f"Error occurred: {error}")

    def receive_qubit(self, connection: socket.socket, buffer_size: int=4096) -> Tuple[Qubit, float]:
        """
        Receives qubits through the connection from a participant.
        """
        logger.debug3("Receiving qubit...")        
        data = connection.recv(buffer_size)
        if data == pickle.dumps(b'<END>'):
            return None, None
        qubit_state, time_stamp = pickle.loads(data)
        logger.debug3("Qubit received: %s", qubit_state.show())
        logger.debug3("Time stamp: %s", time_stamp)
        
        return qubit_state, time_stamp
    

class QuantumChannel2:
    def __init__(self, loss_rate: int=0.9):
        self.loss_rate = loss_rate
        self._END_MARKER = b'<END>'

    def send_qubit(self, qubit: Qubit, connection: socket.socket) -> None:
        """Sends a qubit to a participant with some loss."""
        try:
            if qubit is not None:
                if random.random() < self.loss_rate:
                    data = pickle.dumps(qubit)
                    logger.debug3("Qubit sent: %s", qubit.show())
                    # Byte Size of the data to be sent
                    # print(f"Byte Size of the data to be sent: {len(data)}")
                    connection.sendall(data) # The size of the qubit is 252 bytes and is constant
                    # # Send the length of the data first
                    # data_length = len(data)
                    # # Convert the length to 4 bytes using big-endian byte order
                    # length_header = data_length.to_bytes(4, byteorder='big')
                    # # Send the length header followed by the actual data
                    # connection.sendall(length_header + data)
                else:
                    logger.debug3("Qubit lost.")
            elif qubit is None:
                data = pickle.dumps(self._END_MARKER)
                logger.debug3("End marker sent.")
                connection.sendall(data)
                
        except ConnectionAbortedError as error:
            logger.error(f"Connection aborted: {error}")
        except Exception as error:
            logger.error(f"Error occurred: {error}")

    def receive_qubit(self, connection: socket.socket, buffer_size: int=241) -> Qubit:
        """
        Receives qubits through the connection from a participant.
        """
        # logger.debug3("Receiving qubit...")       
        try:
            # # In case we don't know the length of the data
            # Read the 4-byte length header
            # length_header = connection.recv(4)
            # if not length_header:
            #     return None
            # buffer_size = int.from_bytes(length_header, byteorder='big')

            # Read the actual data based on the length
            data = b''
            while len(data) < buffer_size:
                packet = connection.recv(buffer_size - len(data))
                if not packet:
                    break
                data += packet
                if data == pickle.dumps(self._END_MARKER):
                    return None
            # time.sleep(CHANNEL_DELAY)  # Simulate transmission delay
            qubit_state = pickle.loads(data)
            logger.debug3("Qubit received: %s", qubit_state.show())
            return qubit_state
        except Exception as e:
            logger.error("Error receiving qubit: %s", e)
            return None