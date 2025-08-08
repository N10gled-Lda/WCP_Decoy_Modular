"""
Bob's main controller.
"""
import logging
from typing import Optional

class BobCPU:
    """
    The main class for Bob's side of the QKD simulation.
    It controls all of Bob's hardware and software components.
    """

    def __init__(self, physical: bool = False, seed: Optional[int] = None):
        """
        Initializes Bob's CPU.

        Args:
            physical: Whether to use physical hardware or simulators.
            seed: The seed for the random number generator.
        """
        self.physical = physical
        self.seed = seed
        logging.info("BobCPU initialized.")
        # TODO: Initialize all of Bob's components
        raise NotImplementedError

    def run(self):
        """
        Runs the QKD protocol from Bob's side.
        """
        logging.info("BobCPU running.")
        raise NotImplementedError
