
from abc import ABC, abstractmethod
from typing import List


class BaseLaserDriver(ABC):
    """Hardware-independent laser driver interface."""

    @abstractmethod
    def turn_on(self) -> None:
        """Turn on the laser hardware."""

    @abstractmethod
    def turn_off(self) -> None:
        """Turn off the laser hardware."""

    @abstractmethod
    def stop(self) -> None:
        """Safely disable emission (e.g. close shutter)."""
        
    # Optional: subclasses can override if they support continuous emission
    def arm(self, repetition_rate_hz: float) -> None:
        """Prepare the source for a sequence of pulses. No-op by default."""
        pass

    # Optional: subclasses can override if they support firing a frame directly
    def fire(self, pattern: List) -> None:
        """Emit one frame's worth of pulses. No-op by default."""
        pass

    # Optional: subclasses can override if they support single-shot emission
    def fire_single_pulse(self) -> None:
        """Emit a single pulse. No-op by default."""
        pass

    # Optional: subclasses can override for initialization
    def initialize(self) -> bool:
        """Initialize the laser hardware. Returns True if successful."""
        return True

    # Optional: subclasses can override for shutdown
    def shutdown(self) -> None:
        """Shutdown the laser hardware."""
        pass

