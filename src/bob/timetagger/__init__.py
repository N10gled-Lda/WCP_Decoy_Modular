"""Time Tagger module for Bob CPU."""

# Base classes and data structures
from .timetagger_base import (
    BaseTimeTaggerDriver,
    TimeStamp,
    ChannelConfig,
    TimeTaggerConfig,
    ChannelState,
    TimeTaggerStatistics
)

# Hardware implementation
from .timetagger_hardware import TimeTaggerHardware

# Simulator implementation
from .timetagger_simulator import TimeTaggerSimulator, SimulatorConfig

# Controller for hardware/simulator selection
from .timetagger_controller import TimeTaggerController, TimeTaggerControllerConfig

__all__ = [
    # Base classes
    'BaseTimeTaggerDriver',
    'TimeStamp',
    'ChannelConfig', 
    'TimeTaggerConfig',
    'ChannelState',
    'TimeTaggerStatistics',
    
    # Hardware implementation
    'TimeTaggerHardware',
    
    # Simulator implementation
    'TimeTaggerSimulator',
    'SimulatorConfig',
    
    # Controller
    'TimeTaggerController',
    'TimeTaggerControllerConfig'
]
