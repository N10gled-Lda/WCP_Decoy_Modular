import serial
import threading
from queue import Queue, Empty
from time import sleep

from enum import IntEnum

START_BYTE  = 0xAA
RESPONSE    = 0x80

class CommandStatus(IntEnum):
    COMMAND_CRC_ERROR = 0
    COMMAND_VALID = 1
    MISSING_CONNECTION = 2
    COMMAND_INVALID = 3
    COMMAND_MEMORY_ERROR = 4

class Command(IntEnum):
    COMMAND_CONNECTION = 0
    COMMAND_POLARIZATION = 1
    COMMAND_ANGLE = 2
    COMMAND_FREQUENCY = 3

class SubCommandConnection(IntEnum):
    SUB_COMMAND_CONNECTION_CONNECT = 0

class SubCommandPolarization(IntEnum):
    SUB_COMMAND_POLARIZATION_NUMBERS = 0
    SUB_COMMAND_POLARIZATION_DEVICE = 1

class SubCommandAngle(IntEnum):
    SUB_COMMAND_SET_OFFSET = 0
    SUB_COMMAND_SET_ANGLE = 1

class SubCommandFrequency(IntEnum):
    SUB_COMMAND_SET_OPERATION_FREQUENCY = 0
    SUB_COMMAND_SET_STEPPER_MOTOR_FREQUENCY = 1

class STM32Interface:
    def __init__(self, port: str, baudrate=115200):
        self.serial_port    = serial.Serial(port=port, baudrate=baudrate, timeout=1)
        self.receive_queue  = Queue()
        self.send_queue     = Queue()

        self.running            = False
        self._reception_thread  = None
        self._send_thread       = None
        self._handler_thread    = None

        self.connected  = False
        self.available  = False

        self.on_connected   = None
        self.on_available   = None

    def start(self):
        if not self.running:
            self.running = True
            
            self._reception_thread = threading.Thread(target=self._reception_loop, daemon=True)
            self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self._handler_thread = threading.Thread(target=self._handler_loop, daemon=True)
            
            self._reception_thread.start()
            self._send_thread.start()
            self._handler_thread.start()

    def stop(self):
        self.running = False
        if self._reception_thread:
            self._reception_thread.join(timeout=1)
        if self._send_thread:
            self._send_thread.join(timeout=1)
        if self._handler_thread:
            self._handler_thread.join(timeout=1)

    def _send_loop(self):
        while self.running:
            try:
                msg = self.send_queue.get(timeout=0.1)
                self.serial_port.write(msg)
                print(f"Sent message: {msg}")
            except Empty:
                continue

    def _reception_loop(self):
        while self.running:
            # Wait for start byte
            start = self.serial_port.read(1)
            while not start or start[0] != START_BYTE:
                start = self.serial_port.read(1)

            # Read command
            cmd = self.serial_port.read(1)
            if not cmd:
                continue

            is_response = bool(cmd[0] & RESPONSE)  # RESPONSE = 0x80
            true_cmd = cmd[0] & 0x7F  # Remove response bit

            if true_cmd not in [c.value for c in Command]:
                continue

            # Read sub-command
            sub_cmd = self.serial_port.read(1)
            if true_cmd == Command.COMMAND_CONNECTION:
                valid_subs = [sc.value for sc in SubCommandConnection]
            elif true_cmd == Command.COMMAND_POLARIZATION:
                valid_subs = [sc.value for sc in SubCommandPolarization]
            elif true_cmd == Command.COMMAND_ANGLE:
                valid_subs = [sc.value for sc in SubCommandAngle]
            elif true_cmd == Command.COMMAND_FREQUENCY:
                valid_subs = [sc.value for sc in SubCommandFrequency]
            else:
                continue  # Unknown command

            if sub_cmd[0] not in valid_subs:
                continue

            # Read length
            length = self.serial_port.read(1)
            if not length:
                continue
            # Read payload
            payload = self.serial_port.read(length[0])
            if len(payload) != length[0]:
                continue
            # Read CRC
            crc = self.serial_port.read(1)
            if not crc:
                continue
            # CRC check
            crc_data = bytearray([start[0], cmd[0], sub_cmd[0], length[0]]) + payload
            print(f"crc_data: {list(crc_data)}")
            if crc[0] == self.crc_calculate(crc_data):
                print(f"Received message: cmd={true_cmd}, sub_cmd={sub_cmd[0]}, length={length[0]}, payload={list(payload)}, is_response={is_response}")
                self.receive_queue.put({
                    'cmd': true_cmd,
                    'is_response': is_response,
                    'sub_cmd': sub_cmd[0],
                    'length': length[0],
                    'payload': payload
                })
            else:
                # CRC error, queue error response
                print(f"CRC error: cmd={true_cmd}, sub_cmd={sub_cmd[0]}, length={length[0]}, payload={list(payload)}, is_response={is_response}, crc={crc[0]}, expected={self.crc_calculate(crc_data)}")
                response = [START_BYTE, RESPONSE | true_cmd, sub_cmd[0], 1, CommandStatus.COMMAND_CRC_ERROR]
                response.append(self.crc_calculate(bytearray(response)))
                self.serial_port.write(bytearray(response))

    def _handler_loop(self):
        while self.running:
            while not self.receive_queue.empty():
                msg = self.receive_queue.get()
                print(f"Handler received: {msg}")  # <-- Print the received message                
                # Check if msg is a dict (valid message) or a list (error response)
                if isinstance(msg, dict):
                    cmd = msg['cmd']
                    is_response = msg['is_response']
                    sub_cmd = msg['sub_cmd']
                    length = msg['length']
                    payload = msg['payload']

                    if cmd == Command.COMMAND_CONNECTION:
                        if sub_cmd == SubCommandConnection.SUB_COMMAND_CONNECTION_CONNECT and is_response:
                            if payload[0] == CommandStatus.COMMAND_VALID:
                                self.connected = True
                                if self.on_connected:
                                    self.on_connected()
                                    print("Connected to STM32")
                            elif payload[0] == CommandStatus.MISSING_CONNECTION:
                                self.connect()
                                print("Missing connection, retrying...")
                            elif payload[0] == CommandStatus.COMMAND_INVALID:
                                # TODO
                                pass
                            else:
                                # TODO
                                pass
                    elif cmd == Command.COMMAND_POLARIZATION:
                        if sub_cmd == SubCommandPolarization.SUB_COMMAND_POLARIZATION_NUMBERS:
                            if payload[0] == CommandStatus.COMMAND_VALID:
                                self.available = True
                                if self.on_available:
                                    self.on_available()
                            elif payload[0] == CommandStatus.COMMAND_INVALID:
                                self
                                # TODO
                                pass
                            else:
                                # TODO
                                pass
                        elif sub_cmd == SubCommandPolarization.SUB_COMMAND_POLARIZATION_DEVICE:
                            if payload[0] == CommandStatus.COMMAND_VALID:
                                if self.on_available:
                                    self.on_available()
                                pass
                            elif payload[0] == CommandStatus.COMMAND_INVALID:
                                # TODO
                                pass
                            else:
                                # TODO
                                pass
                    elif cmd == Command.COMMAND_ANGLE:
                        if sub_cmd == SubCommandAngle.SUB_COMMAND_SET_OFFSET:
                            if payload[0] == CommandStatus.COMMAND_VALID:
                                self.available = True
                                if self.on_available:
                                    self.on_available()
                            elif payload[0] == CommandStatus.COMMAND_INVALID:
                                # TODO
                                pass
                            else:
                                # TODO
                                pass
                        elif sub_cmd == SubCommandAngle.SUB_COMMAND_SET_ANGLE:
                            # Handle angle setting
                            if payload[0] == CommandStatus.COMMAND_VALID:
                                self.available = True
                                if self.on_available:
                                    self.on_available()
                            elif payload[0] == CommandStatus.COMMAND_INVALID:
                                # TODO
                                pass
                            else:
                                # TODO
                                pass
                elif cmd == Command.COMMAND_FREQUENCY:
                    if sub_cmd == SubCommandFrequency.SUB_COMMAND_SET_OPERATION_FREQUENCY:
                        if payload[0] == CommandStatus.COMMAND_VALID:
                                self.available = True
                                if self.on_available:
                                    self.on_available()
                        elif payload[0] == CommandStatus.COMMAND_INVALID:
                            # TODO
                            pass
                        else:
                            # TODO
                            pass
                    elif sub_cmd == SubCommandFrequency.SUB_COMMAND_SET_STEPPER_MOTOR_FREQUENCY:
                        if payload[0] == CommandStatus.COMMAND_VALID:
                            self.available = True
                            if self.on_available:
                                self.on_available()
                        elif payload[0] == CommandStatus.COMMAND_INVALID:
                            # TODO
                            pass
                        else:
                            # TODO
                            pass

    def connect(self):
        if not self.connected:
            msg = [START_BYTE, Command.COMMAND_CONNECTION.value, SubCommandConnection.SUB_COMMAND_CONNECTION_CONNECT.value, 0]
            msg.append(self.crc_calculate(bytearray(msg)))
            self.send_queue.put(bytearray(msg))
            print(f"Connection command queued: {msg}")

    def send_cmd_polarization_numbers(self, numbers):
        if not self.connected:
            return False

        # Validate: must be a list of int, each int 0-3, and quantity < 100
        if (
            not isinstance(numbers, list)
            or not all(isinstance(n, int) and 0 <= n <= 3 for n in numbers)
            or len(numbers) >= 100
        ):
            print("Invalid polarization numbers. Must be a list of integers (0-3) with less than 100 elements.")
            return False

        self.available = False
        self.on_available()

        payload = bytearray([len(numbers)]) + bytearray(numbers)
        msg = [START_BYTE, Command.COMMAND_POLARIZATION.value, SubCommandPolarization.SUB_COMMAND_POLARIZATION_NUMBERS.value]
        msg.extend(payload)
        msg.append(self.crc_calculate(bytearray(msg)))

        self.send_queue.put(bytearray(msg))

        print("Polarization numbers on message queue:", numbers)
        return True
    
    def send_cmd_polarization_device(self, device):
        if not self.connected:
            print("Not connected.")
            return False

        # Validate: must be an integer, and either 1 or 2
        if not isinstance(device, int) or device not in (1, 2):
            print("Invalid device. Must be 1 or 2.")
            return False

        payload = bytearray([device])
        msg = [START_BYTE, Command.COMMAND_POLARIZATION.value, SubCommandPolarization.SUB_COMMAND_POLARIZATION_DEVICE.value, len(payload)]
        msg.extend(payload)
        msg.append(self.crc_calculate(bytearray(msg)))

        self.send_queue.put(bytearray(msg))
        print(f"Polarization device command {device} sent.")
        return True

    def send_cmd_set_angle(self, value, is_offset=False):
        if not self.connected:
            return False

        # Validate: must be an integer between 0 and 360
        if not isinstance(value, int) or not (0 <= value <= 360):
            print("Invalid value. Must be an integer between 0 and 360.")
            return False

        # Choose sub-command based on is_offset
        sub_command = (
            SubCommandAngle.SUB_COMMAND_SET_OFFSET.value
            if is_offset
            else SubCommandAngle.SUB_COMMAND_SET_ANGLE.value
        )

        # Prepare payload (MSB first)
        payload = value.to_bytes(2, byteorder='big')
        msg = [
            START_BYTE,
            Command.COMMAND_ANGLE.value,
            sub_command,
            len(payload)
        ]
        msg.extend(payload)
        msg.append(self.crc_calculate(bytearray(msg)))

        self.send_queue.put(bytearray(msg))
        print(f"Angle command {value} sent as {'offset' if is_offset else 'angle'}.")
        return True
    
    def send_cmd_set_frequency(self, frequency, is_stepper=False):
        if not self.connected:
            return False

        if is_stepper:
            # Stepper motor frequency: 1 Hz to 1000 Hz
            if not isinstance(frequency, int) or not (1 <= frequency <= 1000):
                print("Invalid stepper motor frequency. Must be an integer between 1 and 500 Hz.")
                return False
        else:
            # Operation period: 100 ms to 60000 ms
            if not isinstance(frequency, int) or not (1 <= frequency <= 60000):
                print("Invalid operation period. Must be an integer between 1 and 60000 ms.")
                return False

        # Choose sub-command based on is_stepper
        sub_command = (
            SubCommandFrequency.SUB_COMMAND_SET_STEPPER_MOTOR_FREQUENCY.value
            if is_stepper
            else SubCommandFrequency.SUB_COMMAND_SET_OPERATION_FREQUENCY.value
        )

        # Prepare payload (MSB first)
        payload = frequency.to_bytes(2, byteorder='big')
        msg = [
            START_BYTE,
            Command.COMMAND_FREQUENCY.value,
            sub_command,
            len(payload)
        ]
        msg.extend(payload)
        msg.append(self.crc_calculate(bytearray(msg)))

        self.send_queue.put(bytearray(msg))
        print(f"Frequency command {frequency} sent as {'stepper motor' if is_stepper else 'operation period'}.")
        return True

    def crc_calculate(self, data: bytes) -> int:
        crc = 0x00
        length = len(data)
        for i in range(1, length):
            crc ^= data[i]
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ 0x07) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
        return crc