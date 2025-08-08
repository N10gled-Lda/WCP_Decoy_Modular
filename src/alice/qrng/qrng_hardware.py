"""QRNG Hardware Interface."""
import logging

class QRNGHardware:
    """Interface to the physical QRNG hardware."""
    def __init__(self):
        logging.info("QRNG hardware interface initialized.")
        # TODO: Initialize hardware connection
        raise NotImplementedError

    def get_random_bit(self) -> int:
        """Returns a random bit from the hardware."""
        logging.info("Getting random bit from QRNG hardware.")
        # TODO: Implement hardware control
        raise NotImplementedError

    def get_random_bits(self, size: int) -> list[int]:
        """Returns a list of random bits from the hardware."""
        logging.info(f"Getting {size} random bits from QRNG hardware.")
        # TODO: Implement hardware control
        raise NotImplementedError
