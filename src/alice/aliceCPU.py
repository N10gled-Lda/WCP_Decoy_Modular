"""
Alice's main controller.
"""
import logging
from typing import Optional

class AliceCPU:
    """
    The main class for Alice's side of the QKD simulation.
    It controls all of Alice's hardware and software components.
    """

    def __init__(self, physical: bool = False, seed: Optional[int] = None):
        """
        Initializes Alice's CPU.

        Args:
            physical: Whether to use physical hardware or simulators.
            seed: The seed for the random number generator.
        """
        self.physical = physical
        self.seed = seed
        logging.info("AliceCPU initialized.")
        # TODO: Initialize all of Alice's components
        raise NotImplementedError

    def run(self):
        """
        Runs the QKD protocol from Alice's side.
        """
        logging.info("AliceCPU running.")
        raise NotImplementedError
