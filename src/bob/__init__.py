"""Bob's components."""
# Core CPU
from .bobCPU import BobCPU, BobConfig, BobMode, BobStatistics, BobData

# Optical table components
from .ot_components import (
    OpticalTableSimulator, OpticalTableConfig, MeasurementOutcome,
    PhotonDetectorSimulator, DetectorConfig, DetectorType,
    DetectorsSimulator  # Legacy
)

# Simple components
from .simple_bob_cpu import SimpleBobCPU, SimpleBobConfig, SimpleBobMode, SimpleBobStatistics, SimpleBobData
from .ot_components.simple_optical_table import SimpleOpticalTable, SimpleOpticalConfig, SimpleDetectorId
from .ot_components.simple_detectors import SimpleDetectorSystem, SimpleDetectorConfig

# Time tagger components  
from .timetagger import (
    TimeTaggerController, TimeTaggerControllerConfig,
    TimeTaggerSimulator, SimulatorConfig,
    TimeTaggerHardware,
    BaseTimeTaggerDriver, TimeStamp, ChannelConfig, 
    TimeTaggerConfig, ChannelState, TimeTaggerStatistics
)

__all__ = [
    # Core
    "BobCPU",
    "BobConfig", 
    "BobMode",
    "BobStatistics",
    "BobData",
    
    # Optical Table
    "OpticalTableSimulator",
    "OpticalTableConfig",
    "MeasurementOutcome",
    
    # Detectors
    "PhotonDetectorSimulator",
    "DetectorConfig",
    "DetectorType",
    "DetectorsSimulator",  # Legacy
    
    # Simple Components
    "SimpleBobCPU",
    "SimpleBobConfig",
    "SimpleBobMode", 
    "SimpleBobStatistics",
    "SimpleBobData",
    "SimpleOpticalTable",
    "SimpleOpticalConfig",
    "SimpleDetectorId",
    "SimpleDetectorSystem",
    "SimpleDetectorConfig",
    
    # Time Tagger
    "TimeTaggerController",
    "TimeTaggerControllerConfig", 
    "TimeTaggerSimulator",
    "SimulatorConfig",
    "TimeTaggerHardware",
    "BaseTimeTaggerDriver",
    "TimeStamp",
    "ChannelConfig",
    "TimeTaggerConfig",
    "ChannelState", 
    "TimeTaggerStatistics",
]
