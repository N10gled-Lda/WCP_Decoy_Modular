"""Base class for time tagger drivers."""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time


class ChannelState(Enum):
    """Channel states for time tagger."""
    DISABLED = "disabled"
    ENABLED = "enabled"
    TRIGGERED = "triggered"
    ERROR = "error"


@dataclass
class TimeStamp:
    """Represents a time stamp event."""
    channel: int
    time_ps: int  # Time in picoseconds
    rising_edge: bool = True  # True for rising edge, False for falling edge
    
    def __post_init__(self):
        if self.time_ps < 0:
            raise ValueError("Time cannot be negative")
        if self.channel < 0:
            raise ValueError("Channel number cannot be negative")


@dataclass 
class ChannelConfig:
    """Simplified channel configuration - only essential parameters."""
    channel_id: int
    enabled: bool = True


@dataclass
class TimeTaggerConfig:
    """Unified configuration for time tagger system - works for both simulator and hardware."""
    channels: Dict[int, ChannelConfig]
    resolution_ps: int = 1  # Time resolution in picoseconds
    buffer_size: int = 1000000  # Event buffer size
    measurement_duration_s: float = 1.0  # Default measurement duration
    sync_channel: Optional[int] = None  # Sync/reference channel
    
    # Hardware-specific parameters (used by TimeTaggerHardware, ignored by simulator)
    detector_channels: List[int] = None  # Detection channels for hardware
    gate_mode: Optional[Any] = None  # GateMode enum (set by hardware class)
    gate_begin_channel: int = 21  # Trigger channel for gate start
    gate_end_channel: int = -21   # Trigger channel for gate end (negative for inverted)
    gate_length_ps: int = 1000000  # Gate length in picoseconds (1ms default)
    binwidth_ps: int = int(1e9)  # 1ms bin width in picoseconds
    n_values: int = 1000  # Number of bins
    use_test_signal: bool = False  # Enable built-in test signals
    test_signal_channels: List[int] = None  # Test signal channels
    
    def __post_init__(self):
        # Set default values for lists
        if self.detector_channels is None:
            self.detector_channels = [1, 2, 3, 4]
        if self.test_signal_channels is None:
            self.test_signal_channels = [1, 2]


# Removed complex TimeTaggerStatistics dataclass - using simple dictionaries instead


class BaseTimeTaggerDriver(ABC):
    """
    Abstract base class for time tagger drivers.
    
    Simplified interface similar to Alice's laser drivers.
    Defines only the essential methods that both hardware and simulator must implement.
    """
    
    def __init__(self, config: TimeTaggerConfig):
        """
        Initialize the time tagger driver.
        
        Args:
            config: Time tagger configuration
        """
        self.config = config
        self._stats = {
            'total_events': 0,
            'events_per_channel': {},
            'buffer_overflows': 0,
            'measurement_time_s': 0.0,
            'count_rates_hz': {}
        }
        
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the time tagger hardware/simulator.
        
        Returns:
            bool: True if initialization successful
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """
        Shutdown the time tagger hardware/simulator.
        
        Returns:
            bool: True if shutdown successful
        """
        pass
    
    @abstractmethod
    def get_single_gate_counts(self) -> Dict[int, int]:
        """
        Get detection counts for a single gate window.
        This is the main measurement method for QKD.
        
        Returns:
            Dict[int, int]: Detector channel ID -> count mapping
        """
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        Default implementation returns basic stats - override for specific details.
        
        Returns:
            Dict[str, Any]: Current statistics as simple dictionary
        """
        return self._stats.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information. Override in subclasses for specific status."""
        return {
            "driver_type": self.__class__.__name__,
            "initialized": True,
            "measuring": False
        }

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self._stats = {
            'total_events': 0,
            'events_per_channel': {},
            'buffer_overflows': 0,
            'measurement_time_s': 0.0,
            'count_rates_hz': {}
        }
