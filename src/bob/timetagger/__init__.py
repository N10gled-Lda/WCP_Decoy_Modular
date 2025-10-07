"""Time Tagger module for Bob CPU."""

# Base classes and data structures
from .timetagger_base import (
    BaseTimeTaggerDriver,
    TimeStamp,
    ChannelConfig,
    TimeTaggerConfig,
    ChannelState
)

# Hardware implementation
from .timetagger_hardware import TimeTaggerHardware

# Simulator implementation
from .timetagger_simulator import TimeTaggerSimulator

# Controller for hardware/simulator selection
from .timetagger_controller import TimeTaggerController, TimeTaggerControllerConfig

from .simple_timetagger_base_hardware_simulator import (
    SimpleTimeTagger,
    SimpleTimeTaggerHardware,
    SimpleTimeTaggerSimulator
)
from .simple_timetagger_controller import SimpleTimeTaggerController

__all__ = [
    # Base classes
    'BaseTimeTaggerDriver',
    'TimeStamp',
    'ChannelConfig', 
    'TimeTaggerConfig',
    'ChannelState',
    
    # Hardware implementation
    'TimeTaggerHardware',
    
    # Simulator implementation
    'TimeTaggerSimulator',
    
    # Controller
    'TimeTaggerController',
    'TimeTaggerControllerConfig'

    # Simple versions
    'SimpleTimeTagger',
    'SimpleTimeTaggerHardware',
    'SimpleTimeTaggerSimulator',
    'SimpleTimeTaggerController'
]
