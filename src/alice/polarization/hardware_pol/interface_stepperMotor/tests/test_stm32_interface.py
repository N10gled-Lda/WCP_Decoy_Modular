
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../imports')))

import pytest
from unittest.mock import MagicMock, patch
from stm32_interface import *

@pytest.fixture
def mock_serial():
    with patch("stm32_interface.serial.Serial") as mock_serial_class:
        mock_instance = MagicMock()
        mock_serial_class.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def interface(mock_serial):
    return STM32Interface(port="COM_TEST")

def test_connect_sends_correct_bytes(interface, mock_serial):
    interface.connect()
    expected = bytearray([COMMAND_CONNECTION, 0, TERMINATOR])
    mock_serial.write.assert_called_once_with(expected)

def test_send_polarization_numbers_success(interface, mock_serial):
    interface.connected = True
    interface.available = True
    result = interface.send_polarization_numbers([0, 1, 2])
    expected = bytearray([COMMAND_POLARIZATION, SUB_COMMAND_POLARIZATION_NUMBERS, 3, TERMINATOR])
    assert result is True
    mock_serial.write.assert_called_once_with(expected)

def test_send_polarization_numbers_invalid(interface):
    interface.connected = True
    interface.available = True
    with pytest.raises(ValueError):
        interface.send_polarization_numbers([5])  # Invalid value

def test_send_polarization_numbers_not_connected(interface):
    interface.connected = False
    interface.available = True
    result = interface.send_polarization_numbers([0])
    assert result is False

def test_handler_loop_connected_event(interface):
    called = False

    def on_connected():
        nonlocal called
        called = True

    interface.on_connected = on_connected
    interface.running = True
    
    # We need to start the handler loop in a separate thread to simulate its behavior asynchronously
    handler_thread = threading.Thread(target=interface._handler_loop, daemon=True)
    handler_thread.start()

    # Simulate receiving the connection command
    interface.com_queue.put([RESPONSE | COMMAND_CONNECTION, COMMAND_CONNECTION_CONNECTED, TERMINATOR])

    # Wait a bit to give the handler thread time to process the queue
    sleep(0.1)

    assert called
    assert interface.connected is True

    # Make sure to stop the threads to prevent hanging
    interface.stop()
    handler_thread.join(timeout=1)

def test_handler_loop_available_event(interface):
    called = False

    def on_available():
        nonlocal called
        called = True

    interface.on_available = on_available
    interface.running = True

    # Create an event to signal when the handler loop has finished processing
    event = threading.Event()

    # Start the handler loop in a separate thread
    def handler_thread_func():
        interface._handler_loop()
        event.set()  # Signal that the loop is finished processing

    handler_thread = threading.Thread(target=handler_thread_func, daemon=True)
    handler_thread.start()

    # Simulate receiving the availability command
    interface.com_queue.put([COMMAND_CONNECTION, COMMAND_CONNECTION_AVAILABLE, TERMINATOR])

    # Wait for the event to be set (i.e., processing to complete)
    event.wait(timeout=1)

    # Check if the callback was triggered and state is updated
    assert called
    assert interface.available is True

    # Stop the threads after the test to prevent blocking
    interface.stop()
    handler_thread.join(timeout=1)



def test_handler_loop_polarization_status(interface):
    called_status = None

    def on_status(status):
        nonlocal called_status
        called_status = status

    interface.on_polarization_status = on_status
    interface.running = True
    
    # Start the handler loop in a separate thread to simulate asynchronous processing
    handler_thread = threading.Thread(target=interface._handler_loop, daemon=True)
    handler_thread.start()

    # Simulate receiving the polarization command response
    interface.com_queue.put([RESPONSE | COMMAND_POLARIZATION, SUB_COMMAND_POLARIZATION_NUMBERS_SET, 0, TERMINATOR])

    # Wait a bit to give the handler thread time to process the queue
    sleep(0.1)

    # Check the callback was triggered and status is set correctly
    assert called_status == 0
    assert interface.status == 0

    # Stop the threads after the test to prevent blocking
    interface.stop()
    handler_thread.join(timeout=1)
