"""Alice's components."""
# Core CPU
from .aliceCPU import AliceCPU, AliceConfig, AliceMode
from .alice_cpu_general import AliceCPUGeneral, SimulationConfig

# Laser components
from .laser.laser_controller import LaserController
from .laser.laser_base import BaseLaserDriver
from .laser.laser_simulator import SimulatedLaserDriver
from .laser.laser_hardware import HardwareLaserDriver
from .laser.laser_hardware_digital import (
    DigitalHardwareLaserDriver,
    LaserState,
    LaserTriggerMode
)

# VOA components  
from .voa.voa_controller import VOAController, DecoyInfoExtended, VOAOutput
from .voa.voa_base import BaseVOADriver
from .voa.voa_simulator import VOASimulator
from .voa.voa_hardware import VOAHardwareDriver

# Polarization components
from .polarization.polarization_controller import PolarizationController, PolarizationOutput
from .polarization.polarization_base import (
    BasePolarizationDriver,
    PolarizationState,
    JonesVector
)
from .polarization.polarization_simulator import PolarizationSimulator
from .polarization.polarization_hardware import PolarizationHardware

# QRNG components
from .qrng.qrng_simulator import QRNGSimulator, OperationMode
from .qrng.qrng_hardware import QRNGHardware

__all__ = [
    # Core
    "AliceCPU",
    "AliceCPUGeneral",
    "SimulationConfig",
    "AliceConfig",
    "AliceMode",
    
    # Laser
    "LaserController",
    "BaseLaserDriver",
    "SimulatedLaserDriver", 
    "HardwareLaserDriver",
    "DigitalHardwareLaserDriver",
    "LaserState",
    "LaserTriggerMode",
    
    # VOA
    "VOAController",
    "DecoyInfoExtended",
    "VOAOutput",
    "BaseVOADriver",
    "VOASimulator",
    "VOAHardwareDriver",
    
    # Polarization
    "PolarizationController",
    "PolarizationOutput",
    "BasePolarizationDriver",
    "PolarizationState", 
    "JonesVector",
    "PolarizationSimulator",
    "PolarizationHardware",
    
    # QRNG
    "QRNGSimulator",
    "OperationMode",
    "QRNGHardware",
]
