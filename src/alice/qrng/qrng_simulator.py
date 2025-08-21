"""QRNG Simulator."""
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np
from typing import Optional, Union



class OperationMode(Enum):
    """QRNG operation modes."""
    BATCH = "batch"        # Pre-generate all bits
    STREAMING = "streaming"  # Generate on-demand
    DETERMINISTIC = "deterministic"  # Use PRNG for testing


# TODO: Add later like entropy or noise or bias possibility in the simulator
class QRNGSimulator:
    """
    Simulated Quantum Random Number Generator.

    For simulation purposes, uses a PRNG with optional seeding
    for reproducible results. Provides different operation modes for
    flexibility in testing and production use.
    """
    def __init__(self, seed: Optional[int] = None, mode: OperationMode = None):
        """
        Initializes the QRNG simulator with an optional seed and operation mode.
        """
        self._seed = seed
        self._mode = mode
        self._bits_generated = 0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        if self._seed is not None:
            self._rng = np.random.default_rng(self._seed)
            self.logger.info(f"QRNG simulator initialized with seed {self._seed}.")
        else:
            self._rng = np.random.default_rng()
            self.logger.info("QRNG simulator initialized with random seed.")

    def set_seed(self, seed: int):
        """Set the seed for the random number generator."""
        if not isinstance(seed, int):
            raise ValueError("Seed must be an integer.")
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self.logger.info(f"QRNG seed set to {self._seed}.")
    def set_random_seed(self):
        """Set a new random seed for the QRNG."""
        self._seed = None
        self._rng = np.random.default_rng()
        self.logger.info("QRNG seed reset to random value.")
        
    def get_seed(self) -> Optional[int]:
        """Get the current seed of the QRNG."""
        return self._seed
    
    def set_mode(self, mode: OperationMode):
        """Set the operation mode of the QRNG."""
        if not isinstance(mode, OperationMode):
            raise ValueError("Mode must be an instance of OperationMode Enum.")
        self._mode = mode
        self.logger.info(f"QRNG mode set to {self._mode.value}.")
    def get_mode(self) -> Optional[OperationMode]:
        """Get the current operation mode of the QRNG."""
        return self._mode

    def get_rng(self) -> np.random.Generator:
        """Get the underlying random number generator. Helps to confirm the state (and if is initialized)."""
        return self._rng

    def get_random_bit(self, mode: OperationMode=None, size: int=1) -> Union[int, list[int]]:
        """Returns a random bit."""
        if mode is None:
            mode = self._mode
            if mode is None:
                raise ValueError("Operation mode must be specified or set in the QRNGSimulator instance.")
        if size != 1 and mode != OperationMode.BATCH:
            raise ValueError("Size must be 1 for streaming or deterministic modes.")
        
        if mode == OperationMode.DETERMINISTIC:
            # For deterministic mode, use a fixed seed
            if self._seed is None:
                raise ValueError("Seed must be set for deterministic mode.")
            self._rng = np.random.default_rng(self._seed)
            return self._get_random_bit()
        elif mode == OperationMode.STREAMING:
            # For streaming mode, generate bits on demand
            return self._get_random_bit()
        elif mode == OperationMode.BATCH:
            # For batch mode, generate a batch of bits
            return [self._get_random_bit() for _ in range(size)]

    def _get_random_bit(self) -> int:
        """Generates a single random bit."""
        bit = self._rng.integers(0, 2)
        self._bits_generated += 1
        self.logger.debug(f"QRNG generated bit: {bit}")
        return bit

        
    def get_random_bits_biased(self, mode: OperationMode, size: int=1, bias: float=0.5) -> Union[int, list[int]]:
        """
        Returns a biased random bit. Bias is the probability of returning 0.
        
        If mode is DETERMINISTIC, the seed must be set.
        If mode is STREAMING, returns a single bit.
        If mode is BATCH, returns a list of bits of the specified size.       
        """
        if not (0 <= bias <= 1):
            raise ValueError("Bias must be between 0 and 1.")
        if mode == OperationMode.DETERMINISTIC:
            # For deterministic mode, use a fixed seed
            if self._seed is None:
                raise ValueError("Seed must be set for deterministic mode.")
            self._rng = np.random.default_rng(self._seed)
            return self._get_biased_random_bit(bias)
        elif mode == OperationMode.STREAMING:
            # For streaming mode, generate bits on demand
            return self._get_biased_random_bit(bias)
        elif mode == OperationMode.BATCH:
            # For batch mode, generate a batch of bits
            return [self._get_biased_random_bit(bias) for _ in range(size)]
        
    def _get_biased_random_bit(self, bias: float) -> int:
        """Generates a single biased random bit. Bias is the probability of returning 0."""
        if not (0 <= bias <= 1):
            raise ValueError("Bias must be between 0 and 1.")
        bit = self._rng.choice([0, 1], p=[bias, 1 - bias])
        self._bits_generated += 1
        self.logger.debug(f"QRNG generated biased bit: {bit} with bias {bias}")
        return bit

    def get_bits_generated(self) -> int:
        """Returns the total number of bits generated."""
        return self._bits_generated
    
    def reset(self):
        """Resets the QRNG state."""
        self._bits_generated = 0
        self.logger.info("QRNG state reset.")
        if self._seed is not None:
            self._rng = np.random.default_rng(self._seed)
            self.logger.info(f"QRNG reinitialized with seed {self._seed}.")
        else:
            self._rng = np.random.default_rng()
            self.logger.info("QRNG reinitialized with random seed.")