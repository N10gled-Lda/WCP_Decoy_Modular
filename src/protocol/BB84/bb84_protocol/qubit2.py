import random

class Qubit2:
    """
    A class to represent a qubit, managing states in bases 0 and 1,
    and providing utilities for communication and measurement.
    """
    def __init__(self, bit=None, base=0):
        """
        Initialize the qubit with a given bit and base.
        :param bit: Optional, the initial bit of the qubit (0 or 1).
        :param base: The base of the qubit (0 or 1).
        """
        self.bit = bit  # Qubit bit (0 or 1)
        self.base = base    # Qubit base (0 or 1)
        self._measured = False

    @staticmethod
    def byte_to_bitbase(byte) -> tuple[int, int]:
        """
        Translate a single byte to a qubit bit and base.
        :param byte: The byte value (0-255).
        :return: A tuple (bit, base) representing the qubit.
        """
        bit_map = {
            0: (0, 0),
            1: (1, 0),
            2: (0, 1),
            3: (1, 1),
        }
        return bit_map.get(byte, (None, None))

    @staticmethod
    def bitbase_to_byte(bit, base) -> int:
        """
        Translate a qubit bit and base to a single byte for communication.
        :param bit: An integer representing the qubit bit (0 or 1).
        :param base: The base of the qubit (0 or 1).
        :return: A byte value (0-255).
        """
        byte_map = {
            (0, 0): 0,
            (1, 0): 1,
            (0, 1): 2,
            (1, 1): 3,
        }
        return byte_map.get((bit, base), 255)  # 255 represents an undefined bit

    @staticmethod
    def from_byte(byte):
        """
        Create a new Qubit instance from a single byte.
        :param byte: The byte value (0-255).
        :return: A new Qubit instance with the bit and base set from the byte.
        """
        bit, base = Qubit2.byte_to_bitbase(byte)
        return Qubit2(bit, base)

    def set_from_byte(self, byte):
        """
        Set the qubit's bit and base from a single byte.
        :param byte: The byte value (0-255).
        """
        self.bit, self.base = self.byte_to_bitbase(byte)

    def get_byte(self) -> int:
        """
        Get the single byte representation of the qubit's current bit and base.
        :return: A byte value (0-255).
        """
        return self.bitbase_to_byte(self.bit, self.base)
    
    def get_bit(self) -> int:
        return self.bit

    def get_base(self) -> int:
        return self.base

    def measure(self, measurement_base):
        """
        Simulate a measurement of the qubit in a given base.
        :param measurement_base: The base to measure in (0 or 1).
        :return: The measured outcome (0 or 1) based on the measurement base. Random if bases are different.
        """
        if self._measured:
            raise Exception("Qubit already measured!")
        self._measured = True
        if self.base == measurement_base:
            return self.bit  # Measurement matches the preparation base
        else:
            return random.choice([0, 1])
        
    @staticmethod
    def measure_byte(byte, measurement_base):
        """
        Simulate a measurement of the qubit in a given base and the byte received.
        :param byte: The byte value received (0, 1, 2, 3).
        :param measurement_base: The base to measure in (0 or 1).
        :return: The measured outcome (0 or 1) based on the measurement base. Random if bases are different.
        """
        bit, base = Qubit2.byte_to_bitbase(byte)
        if base == measurement_base:
            return bit
        else:
            return random.choice([0, 1])

    def hadamard(self):
        """
        Apply the Hadamard gate to the qubit, changing its bit.
        """
        if self.base == 0:
            self.bit = 0 if self.bit == 0 else 1
        elif self.base == 1:
            self.bit = 0 if self.bit == 0 else 1
    
    def bit_flip(self, QBER):
        """
        Apply a bit flip error to the qubit with a given QBER.
        :param QBER: The bit error rate for flipping the qubit bit.
        """
        if random.random() < QBER:
            self.bit = 1 if self.bit == 0 else 0

    def __repr__(self):
        """
        Return a string representation of the qubit's current bit and base.
        """
        return f"Qubit(bit={self.bit}, base={self.base})"