
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseLaserDriver(ABC):
    """
    Hardware-independent laser driver interface.
    Only requires initialize/shutdown for lifecycle management.
    """

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the laser hardware. Returns True if successful."""
        pass

    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown the laser hardware."""
        pass

    @abstractmethod
    def trigger_once(self) -> bool:
        """Send a single trigger pulse. Returns True if successful."""
        pass

    @abstractmethod
    def send_frame(self, n_triggers: int, rep_rate_hz: float = None) -> bool:
        """
        Send a frame of multiple trigger pulses.
        Args:
            n_triggers: Number of trigger pulses
            rep_rate_hz: Repetition rate in Hz
        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def start_continuous(self, rep_rate_hz: float = None) -> bool:
        """
        Start continuous trigger pulse generation.
        
        Args:
            rep_rate_hz: Repetition rate in Hz
            
        Returns:
            True if successful
        """
        pass
    # @abstractmethod
    # def start_continuous(self, *args: Any, **kwargs: Any) -> bool:
    #     """
    #     Start continuous trigger pulse generation.
    #     Concrete drivers may accept different parameters (positional or keyword).
    #     """
    #     raise NotImplementedError

    @abstractmethod
    def stop_continuous(self) -> bool:
        """Stop continuous trigger pulse generation. Returns True if successful."""
        pass

    def get_status(self) -> Dict[str, Any]:
        """Get status information. Override in subclasses for specific status."""
        return {
            "driver_type": self.__class__.__name__,
            "initialized": True,
            "active": True
        }

