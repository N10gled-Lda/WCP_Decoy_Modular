"""Bob's optical table components."""

# Enhanced simulators
from .optical_table_simulator import (
    OpticalTableSimulator, 
    OpticalTableConfig, 
    MeasurementOutcome,
    OpticalTableStatistics
)
from .detectors_simulator import (
    PhotonDetectorSimulator,
    DetectorConfig,
    DetectorType,
    DetectorStatistics,
    DetectorsSimulator  # Legacy class
)

__all__ = [
    # Optical Table
    "OpticalTableSimulator",
    "OpticalTableConfig", 
    "MeasurementOutcome",
    "OpticalTableStatistics",
    
    # Detectors
    "PhotonDetectorSimulator",
    "DetectorConfig",
    "DetectorType", 
    "DetectorStatistics",
    "DetectorsSimulator",  # Legacy
]
