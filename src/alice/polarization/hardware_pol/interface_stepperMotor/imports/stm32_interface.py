import serial
import threading
from queue import Queue
from time import sleep

TERMINATOR  = 0xFF # '\n gives problems. Switching for 0xFF for now, and later on for fixed lengths.
RESPONSE    = 0x80 # Response identifier

# Command types
COMMAND_CONNECTION      = 0
COMMAND_POLARIZATION    = 1

# CommandConnection_Status_t
COMMAND_CONNECTION_TRY_CONNECTION   = 0
COMMAND_CONNECTION_CONNECTED        = 1
COMMAND_CONNECTION_AVAILABLE        = 2
COMMAND_CONNECTION_UNAVAILABLE      = 3

# SubCommandPolarization_Status_t
SUB_COMMAND_POLARIZATION_NUMBERS        = 0
SUB_COMMAND_POLARIZATION_NUMBERS_SET    = 1

# CommandPolarization_Status_t
COMMAND_POLARIZATION_SUCCESS                = 0
COMMAND_POLARIZATION_INVALID_ID             = 1
COMMAND_POLARIZATION_WRONG_POLARIZATIONS    = 2
COMMAND_POLARIZATION_MISMATCH_QUANTITY      = 3
COMMAND_POLARIZATION_OVERFLOW               = 4
COMMAND_POLARIZATION_UNKNOWN_ERROR          = 5

class STM32Interface:
    def __init__(self, port: str, baudrate=115200):
        self.serial_port = serial.Serial(port=port, baudrate=baudrate, timeout=1)
        self.com_queue = Queue()
        self.values = []

        self.running            = False
        self._reception_thread  = None
        self._handler_thread    = None

        self.connected  = False
        self.available  = False
        self.status     = None

        # Callbacks
        self.on_connected           = None
        self.on_available           = None
        self.on_polarization_status = None

    def start(self):
        if not self.running:
            self.running = True
            self._reception_thread = threading.Thread(target=self._reception_loop, daemon=True)
            self._handler_thread = threading.Thread(target=self._handler_loop, daemon=True)
            self._reception_thread.start()
            self._handler_thread.start()

    def stop(self):
        self.running = False
        if self._reception_thread:
            self._reception_thread.join(timeout=1)
        if self._handler_thread:
            self._handler_thread.join(timeout=1)

    def _reception_loop(self):
        buffer = bytearray()
        while self.running:
            byte = self.serial_port.read(1)
            if byte:
                buffer.append(byte[0])
                if byte[0] == TERMINATOR:
                    self.com_queue.put(list(buffer))
                    print("Received:", list(buffer))
                    buffer.clear()

    def _handler_loop(self):
        while self.running:
            while not self.com_queue.empty():
                response = self.com_queue.get()

                if response == [RESPONSE | COMMAND_CONNECTION, COMMAND_CONNECTION_CONNECTED, TERMINATOR]:
                    self.connected = True
                    if self.on_connected:
                        self.on_connected()

                elif response == [COMMAND_CONNECTION, COMMAND_CONNECTION_AVAILABLE, TERMINATOR]:
                    self.send_bytes(RESPONSE | COMMAND_CONNECTION, COMMAND_CONNECTION_AVAILABLE)
                    self.available = True
                    sleep(0.1)  # Allow time for the STM32 to process the command
                    if self.on_available:
                        self.on_available()

                elif response == [RESPONSE | COMMAND_POLARIZATION, SUB_COMMAND_POLARIZATION_NUMBERS, len(self.values), TERMINATOR]:
                    self.send_bytes(COMMAND_POLARIZATION, SUB_COMMAND_POLARIZATION_NUMBERS_SET, self.values)

                elif response[0] == (RESPONSE | COMMAND_POLARIZATION) and response[1] == SUB_COMMAND_POLARIZATION_NUMBERS_SET and response[3] == TERMINATOR:
                    self.status = response[2]
                    if self.on_polarization_status:
                        self.on_polarization_status(self.status)

    def send_bytes(self, command_type, status, values=None):
        message = bytearray([command_type, status])
        if values:
            message.extend(values)
        message.append(TERMINATOR)
        print("Sending:", list(message))
        self.serial_port.write(message)

    def connect(self):
        self.send_bytes(COMMAND_CONNECTION, COMMAND_CONNECTION_TRY_CONNECTION)

    def send_polarization_numbers(self, values):
        if self.available and self.connected:
            if not all(v in [0, 1, 2, 3] for v in values):
                raise ValueError("Invalid values. Must be 0, 1, 2, or 3.")
            self.values = values
            self.available = False
            self.send_bytes(COMMAND_POLARIZATION, SUB_COMMAND_POLARIZATION_NUMBERS, [len(values)])
            return True
        print(self.available, self.connected)
        return False