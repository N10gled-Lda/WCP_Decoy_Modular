"""Time Tagger Hardware Interface."""
import logging

class TimeTaggerHardware:
    """Interface to the physical time tagger hardware."""
    def __init__(self):
        logging.info("Time tagger hardware interface initialized.")
        # TODO: Initialize hardware connection
        raise NotImplementedError

    def get_timestamps(self) -> list[float]:
        """Gets timestamps from the hardware."""
        logging.info("Getting timestamps from time tagger hardware.")
        # TODO: Implement hardware control
        raise NotImplementedError
