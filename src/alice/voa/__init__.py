"""VOA (Variable Optical Attenuator) module for Alice."""

from .voa_base import BaseVOADriver
from .voa_controller import VOAController, DecoyInfoExtended, VOAOutput
from .voa_simulator import VOASimulator
from .voa_hardware import VOAHardwareDriver

__all__ = [
    'BaseVOADriver',
    'VOAController', 
    'DecoyInfoExtended',
    'VOAOutput',
    'VOASimulator',
    'VOAHardwareDriver'
]