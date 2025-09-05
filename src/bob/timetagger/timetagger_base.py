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
    """Configuration for a time tagger channel."""
    channel_id: int
    enabled: bool = True
    trigger_level_v: float = 0.5  # Trigger level in volts
    trigger_edge: str = "rising"  # "rising", "falling", "both"
    input_impedance_ohm: int = 50  # Input impedance
    dead_time_ps: int = 1000  # Dead time in picoseconds
    delay_ps: int = 0  # Channel delay in picoseconds


@dataclass
class TimeTaggerConfig:
    """Configuration for time tagger system."""
    channels: Dict[int, ChannelConfig]
    resolution_ps: int = 1  # Time resolution in picoseconds
    buffer_size: int = 1000000  # Event buffer size
    measurement_duration_s: float = 1.0  # Default measurement duration
    sync_channel: Optional[int] = None  # Sync/reference channel


@dataclass
class TimeTaggerStatistics:
    """Statistics for time tagger performance."""
    total_events: int = 0
    events_per_channel: Dict[int, int] = None
    lost_events: int = 0
    buffer_overflows: int = 0
    measurement_time_s: float = 0.0
    count_rates_hz: Dict[int, float] = None
    
    def __post_init__(self):
        if self.events_per_channel is None:
            self.events_per_channel = {}
        if self.count_rates_hz is None:
            self.count_rates_hz = {}


class BaseTimeTaggerDriver(ABC):
    """
    Abstract base class for time tagger drivers.
    
    Defines the interface that both hardware and simulator drivers must implement.
    """
    
    def __init__(self, config: TimeTaggerConfig):
        """
        Initialize the time tagger driver.
        
        Args:
            config: Time tagger configuration
        """
        self.config = config
        self.stats = TimeTaggerStatistics()
        self._is_measuring = False
        self._start_time = 0.0
        
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the time tagger.
        
        Returns:
            bool: True if initialization successful
        """
        pass
    
    @abstractmethod
    def start_measurement(self) -> bool:
        """
        Start time stamp data acquisition.
        
        Returns:
            bool: True if measurement started successfully
        """
        pass
    
    @abstractmethod
    def stop_measurement(self) -> bool:
        """
        Stop time stamp data acquisition.
        
        Returns:
            bool: True if measurement stopped successfully
        """
        pass
    
    @abstractmethod
    def get_timestamps(self, max_events: Optional[int] = None) -> List[TimeStamp]:
        """
        Get collected time stamps.
        
        Args:
            max_events: Maximum number of events to retrieve
            
        Returns:
            List of time stamp events
        """
        pass
    
    @abstractmethod
    def configure_channel(self, channel_id: int, config: ChannelConfig) -> bool:
        """
        Configure a specific channel.
        
        Args:
            channel_id: Channel to configure
            config: Channel configuration
            
        Returns:
            bool: True if configuration successful
        """
        pass
    
    @abstractmethod
    def get_channel_state(self, channel_id: int) -> ChannelState:
        """
        Get the current state of a channel.
        
        Args:
            channel_id: Channel to query
            
        Returns:
            Current channel state
        """
        pass
    
    @abstractmethod
    def get_count_rates(self) -> Dict[int, float]:
        """
        Get count rates for all enabled channels.
        
        Returns:
            Dictionary mapping channel ID to count rate in Hz
        """
        pass
    
    @abstractmethod
    def clear_buffer(self) -> bool:
        """
        Clear the internal event buffer.
        
        Returns:
            bool: True if buffer cleared successfully
        """
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get device information and status.
        
        Returns:
            Dictionary with device information
        """
        pass
    
    def is_measuring(self) -> bool:
        """Check if measurement is currently active."""
        return self._is_measuring
    
    def get_statistics(self) -> TimeTaggerStatistics:
        """Get time tagger statistics."""
        return self.stats
    
    def get_enabled_channels(self) -> List[int]:
        """Get list of enabled channel IDs."""
        return [ch_id for ch_id, config in self.config.channels.items() if config.enabled]
    
    def enable_channel(self, channel_id: int) -> bool:
        """Enable a specific channel."""
        if channel_id in self.config.channels:
            self.config.channels[channel_id].enabled = True
            return True
        return False
    
    def disable_channel(self, channel_id: int) -> bool:
        """Disable a specific channel."""
        if channel_id in self.config.channels:
            self.config.channels[channel_id].enabled = False
            return True
        return False
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.stats = TimeTaggerStatistics()
    
    def _update_statistics(self, timestamps: List[TimeStamp]) -> None:
        """Update statistics with new timestamp data."""
        self.stats.total_events += len(timestamps)
        
        # Update per-channel statistics
        for ts in timestamps:
            if ts.channel not in self.stats.events_per_channel:
                self.stats.events_per_channel[ts.channel] = 0
            self.stats.events_per_channel[ts.channel] += 1
        
        # Update measurement time
        if self._is_measuring:
            self.stats.measurement_time_s = time.time() - self._start_time
            
            # Calculate count rates
            if self.stats.measurement_time_s > 0:
                for ch_id, count in self.stats.events_per_channel.items():
                    self.stats.count_rates_hz[ch_id] = count / self.stats.measurement_time_s
