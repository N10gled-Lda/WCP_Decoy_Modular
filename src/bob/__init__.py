"""Bob's components."""
# Core CPU
from .bobCPU_general import BobCPU, BobConfig, BobMode, BobStatistics, BobData

# Optical table components
from .ot_components import (
    OpticalTableSimulator, OpticalTableConfig, MeasurementOutcome,
    PhotonDetectorSimulator, DetectorConfig, DetectorType,
    DetectorsSimulator  # Legacy
)

# Simple components
from .bob_cpu import SimpleBobCPU, SimpleBobConfig, SimpleBobMode, SimpleBobStatistics, SimpleBobData
from .ot_components.simple_optical_table import SimpleOpticalTable, SimpleOpticalConfig, SimpleDetectorId
from .ot_components.simple_detectors import SimpleDetectorSystem, SimpleDetectorConfig

# Time tagger components  
from .timetagger import (
    TimeTaggerController, TimeTaggerControllerConfig,
    TimeTaggerSimulator,
    TimeTaggerHardware,
    BaseTimeTaggerDriver, TimeStamp, ChannelConfig, 
    TimeTaggerConfig, ChannelState
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
    "TimeTaggerHardware",
    "BaseTimeTaggerDriver",
    "TimeStamp",
    "ChannelConfig",
    "TimeTaggerConfig",
    "ChannelState",
]
