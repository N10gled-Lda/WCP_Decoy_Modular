"""Time Tagger Controller - Hardware/Simulator Selection and Management."""
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .timetagger_base import (
    BaseTimeTaggerDriver, TimeStamp, ChannelConfig, TimeTaggerConfig, 
    ChannelState, TimeTaggerStatistics
)
from .timetagger_hardware import TimeTaggerHardware
from .timetagger_simulator import TimeTaggerSimulator, SimulatorConfig


@dataclass
class TimeTaggerControllerConfig:
    """Configuration for the time tagger controller."""
    use_hardware: bool = False  # Use hardware (True) or simulator (False)
    timetagger_config: TimeTaggerConfig = None
    simulator_config: Optional[SimulatorConfig] = None
    auto_fallback: bool = True  # Fallback to simulator if hardware fails
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.timetagger_config is None:
            # Default configuration with 4 channels
            default_channels = {
                0: ChannelConfig(
                    enabled=True,
                    trigger_level_v=0.5,
                    dead_time_ps=50000,  # 50 ns
                    input_delay_ps=0
                ),
                1: ChannelConfig(
                    enabled=True,
                    trigger_level_v=0.5,
                    dead_time_ps=50000,
                    input_delay_ps=0
                ),
                2: ChannelConfig(
                    enabled=False,
                    trigger_level_v=0.5,
                    dead_time_ps=50000,
                    input_delay_ps=0
                ),
                3: ChannelConfig(
                    enabled=False,
                    trigger_level_v=0.5,
                    dead_time_ps=50000,
                    input_delay_ps=0
                )
            }
            
            self.timetagger_config = TimeTaggerConfig(
                resolution_ps=1000,  # 1 ps resolution
                buffer_size=100000,
                max_count_rate_hz=10000000,  # 10 MHz
                channels=default_channels
            )
        
        if self.simulator_config is None:
            self.simulator_config = SimulatorConfig()


class TimeTaggerController:
    """
    Controller for time tagger operations with hardware/simulator selection.
    
    Features:
    - Automatic hardware/simulator selection
    - Fallback to simulator if hardware fails
    - Unified interface for both modes
    - Configuration management
    - Error handling and logging
    """
    
    def __init__(self, config: TimeTaggerControllerConfig):
        """
        Initialize the time tagger controller.
        
        Args:
            config: Controller configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self._driver: Optional[BaseTimeTaggerDriver] = None
        self._using_hardware = False
        
        # Initialize the appropriate driver
        self._initialize_driver()

    def _initialize_driver(self) -> None:
        """Initialize the time tagger driver (hardware or simulator)."""
        if self.config.use_hardware:
            try:
                self.logger.info("Attempting to initialize hardware time tagger...")
                self._driver = TimeTaggerHardware(self.config.timetagger_config)
                
                if self._driver.initialize():
                    self._using_hardware = True
                    self.logger.info("Hardware time tagger initialized successfully")
                    return
                else:
                    self.logger.warning("Hardware time tagger initialization failed")
                    
            except Exception as e:
                self.logger.error(f"Hardware time tagger initialization error: {e}")
            
            # Fallback to simulator if hardware fails and auto_fallback is enabled
            if self.config.auto_fallback:
                self.logger.info("Falling back to simulator...")
                self._initialize_simulator()
            else:
                raise RuntimeError("Hardware time tagger initialization failed and auto_fallback is disabled")
        else:
            self.logger.info("Using time tagger simulator (hardware disabled)")
            self._initialize_simulator()

    def _initialize_simulator(self) -> None:
        """Initialize the time tagger simulator."""
        try:
            self._driver = TimeTaggerSimulator(
                self.config.timetagger_config,
                self.config.simulator_config
            )
            
            if self._driver.initialize():
                self._using_hardware = False
                self.logger.info("Time tagger simulator initialized successfully")
            else:
                raise RuntimeError("Failed to initialize time tagger simulator")
                
        except Exception as e:
            self.logger.error(f"Simulator initialization error: {e}")
            raise

    @property
    def is_using_hardware(self) -> bool:
        """Check if currently using hardware (True) or simulator (False)."""
        return self._using_hardware

    @property
    def driver(self) -> BaseTimeTaggerDriver:
        """Get the current driver instance."""
        if self._driver is None:
            raise RuntimeError("Time tagger driver not initialized")
        return self._driver

    def start_measurement(self) -> bool:
        """Start timestamp data acquisition."""
        if self._driver is None:
            self.logger.error("Driver not initialized")
            return False
        
        try:
            result = self._driver.start_measurement()
            if result:
                self.logger.info(f"Measurement started ({'hardware' if self._using_hardware else 'simulator'})")
            return result
        except Exception as e:
            self.logger.error(f"Failed to start measurement: {e}")
            return False

    def stop_measurement(self) -> bool:
        """Stop timestamp data acquisition."""
        if self._driver is None:
            self.logger.error("Driver not initialized")
            return False
        
        try:
            result = self._driver.stop_measurement()
            if result:
                self.logger.info("Measurement stopped")
            return result
        except Exception as e:
            self.logger.error(f"Failed to stop measurement: {e}")
            return False

    def get_timestamps(self, max_events: Optional[int] = None) -> List[TimeStamp]:
        """Get collected timestamps from buffer."""
        if self._driver is None:
            self.logger.error("Driver not initialized")
            return []
        
        try:
            timestamps = self._driver.get_timestamps(max_events)
            self.logger.debug(f"Retrieved {len(timestamps)} timestamps")
            return timestamps
        except Exception as e:
            self.logger.error(f"Failed to get timestamps: {e}")
            return []

    def configure_channel(self, channel_id: int, config: ChannelConfig) -> bool:
        """Configure a specific channel."""
        if self._driver is None:
            self.logger.error("Driver not initialized")
            return False
        
        try:
            result = self._driver.configure_channel(channel_id, config)
            if result:
                self.logger.info(f"Channel {channel_id} configured successfully")
            return result
        except Exception as e:
            self.logger.error(f"Failed to configure channel {channel_id}: {e}")
            return False

    def get_channel_state(self, channel_id: int) -> ChannelState:
        """Get current state of a channel."""
        if self._driver is None:
            self.logger.error("Driver not initialized")
            return ChannelState.ERROR
        
        try:
            return self._driver.get_channel_state(channel_id)
        except Exception as e:
            self.logger.error(f"Failed to get channel state for {channel_id}: {e}")
            return ChannelState.ERROR

    def get_count_rates(self) -> Dict[int, float]:
        """Get count rates for all enabled channels."""
        if self._driver is None:
            self.logger.error("Driver not initialized")
            return {}
        
        try:
            return self._driver.get_count_rates()
        except Exception as e:
            self.logger.error(f"Failed to get count rates: {e}")
            return {}

    def clear_buffer(self) -> bool:
        """Clear the internal event buffer."""
        if self._driver is None:
            self.logger.error("Driver not initialized")
            return False
        
        try:
            result = self._driver.clear_buffer()
            if result:
                self.logger.debug("Buffer cleared")
            return result
        except Exception as e:
            self.logger.error(f"Failed to clear buffer: {e}")
            return False

    def get_statistics(self) -> TimeTaggerStatistics:
        """Get measurement statistics."""
        if self._driver is None:
            self.logger.error("Driver not initialized")
            return TimeTaggerStatistics()
        
        try:
            return self._driver.stats
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return TimeTaggerStatistics()

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information and status."""
        if self._driver is None:
            return {
                "status": "Driver not initialized",
                "using_hardware": False,
                "error": True
            }
        
        try:
            info = self._driver.get_device_info()
            info.update({
                "using_hardware": self._using_hardware,
                "controller_status": "OK"
            })
            return info
        except Exception as e:
            self.logger.error(f"Failed to get device info: {e}")
            return {
                "status": f"Error: {e}",
                "using_hardware": self._using_hardware,
                "error": True
            }

    def is_measuring(self) -> bool:
        """Check if measurement is currently active."""
        if self._driver is None:
            return False
        return self._driver.is_measuring

    def enable_channel(self, channel_id: int, enable: bool = True) -> bool:
        """Enable or disable a specific channel."""
        if self._driver is None:
            self.logger.error("Driver not initialized")
            return False
        
        if channel_id not in self.config.timetagger_config.channels:
            self.logger.error(f"Channel {channel_id} not found")
            return False
        
        try:
            # Get current configuration
            current_config = self.config.timetagger_config.channels[channel_id]
            
            # Update enabled state
            new_config = ChannelConfig(
                enabled=enable,
                trigger_level_v=current_config.trigger_level_v,
                dead_time_ps=current_config.dead_time_ps,
                input_delay_ps=current_config.input_delay_ps
            )
            
            result = self.configure_channel(channel_id, new_config)
            if result:
                # Update local configuration
                self.config.timetagger_config.channels[channel_id] = new_config
                action = "enabled" if enable else "disabled"
                self.logger.info(f"Channel {channel_id} {action}")
            
            return result
        except Exception as e:
            self.logger.error(f"Failed to enable/disable channel {channel_id}: {e}")
            return False

    def set_trigger_level(self, channel_id: int, level_v: float) -> bool:
        """Set trigger level for a specific channel."""
        if self._driver is None:
            self.logger.error("Driver not initialized")
            return False
        
        if channel_id not in self.config.timetagger_config.channels:
            self.logger.error(f"Channel {channel_id} not found")
            return False
        
        try:
            # Get current configuration
            current_config = self.config.timetagger_config.channels[channel_id]
            
            # Update trigger level
            new_config = ChannelConfig(
                enabled=current_config.enabled,
                trigger_level_v=level_v,
                dead_time_ps=current_config.dead_time_ps,
                input_delay_ps=current_config.input_delay_ps
            )
            
            result = self.configure_channel(channel_id, new_config)
            if result:
                # Update local configuration
                self.config.timetagger_config.channels[channel_id] = new_config
                self.logger.info(f"Channel {channel_id} trigger level set to {level_v} V")
            
            return result
        except Exception as e:
            self.logger.error(f"Failed to set trigger level for channel {channel_id}: {e}")
            return False

    def reset(self) -> bool:
        """Reset the time tagger (reinitialize)."""
        try:
            self.logger.info("Resetting time tagger...")
            
            # Stop measurement if active
            if self._driver and self._driver.is_measuring:
                self.stop_measurement()
            
            # Reinitialize driver
            self._initialize_driver()
            
            self.logger.info("Time tagger reset successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reset time tagger: {e}")
            return False

    def switch_to_simulator(self) -> bool:
        """Switch to simulator mode (useful for testing)."""
        try:
            self.logger.info("Switching to simulator mode...")
            
            # Stop current measurement
            if self._driver and self._driver.is_measuring:
                self.stop_measurement()
            
            # Initialize simulator
            self._initialize_simulator()
            
            self.logger.info("Switched to simulator mode")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to switch to simulator: {e}")
            return False

    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self._driver:
                if self._driver.is_measuring:
                    self.stop_measurement()
                
                # Additional cleanup if needed
                if hasattr(self._driver, 'cleanup'):
                    self._driver.cleanup()
            
            self.logger.info("Time tagger controller cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
