"""
Simple TimeTagger Controller - Follows Alice's controller pattern.
Clean interface to TimeTagger drivers with proper initialization and management.
"""
import logging
from typing import Dict, Optional, Union
from .simple_timetagger_base_hardware_simulator import SimpleTimeTagger, SimpleTimeTaggerHardware, SimpleTimeTaggerSimulator


class SimpleTimeTaggerController:
    """
    Simple TimeTagger Controller - Mirrors Alice's LaserController and PolarizationController pattern.
    
    Provides a clean interface to TimeTagger drivers (hardware or simulator)
    with proper initialization, configuration, and measurement management.
    """
    
    def __init__(self, driver: Union[SimpleTimeTaggerHardware, SimpleTimeTaggerSimulator]):
        """
        Initialize the TimeTagger controller with a specific driver.
        
        Args:
            driver: The TimeTagger driver (hardware or simulator)
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.driver = driver
        self._initialized = False
        self._measurement_duration = None  # Must be set before measuring
        
        self.logger.info(f"TimeTagger controller initialized with {type(driver).__name__}")

    def initialize(self) -> bool:
        """Initialize the TimeTagger controller."""
        try:
            if not self.driver.initialize():
                self.logger.error("Failed to initialize TimeTagger driver")
                return False
                
            self._initialized = True
            self.logger.info("TimeTagger controller initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TimeTagger controller: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the TimeTagger controller."""
        try:
            if self.driver:
                self.driver.shutdown()
            self._initialized = False
            self.logger.info("TimeTagger controller shutdown")
            
        except Exception as e:
            self.logger.error(f"Error shutting down controller: {e}")
    
    def measure_counts(self) -> Dict[int, int]:
        """
        Measure counts using the pre-configured duration.
        
        Returns:
            Dict[int, int]: Channel ID -> count mapping
        """
        if not self._initialized:
            self.logger.error("Controller not initialized")
            return {}
        
        if self._measurement_duration is None:
            self.logger.error("Measurement duration not set. Call set_measurement_duration() first")
            return {}
        
        try:
            counts = self.driver.measure_for_duration(self._measurement_duration)
            self.logger.debug(f"Measured counts: {counts}")
            return counts
            
        except Exception as e:
            self.logger.error(f"Measurement failed: {e}")
            return {}
  
    def set_measurement_duration(self, duration_seconds: float) -> bool:
        """
        Set the measurement duration for subsequent measurements.
        This must be called before measure_counts().
        
        Args:
            duration_seconds: Duration for measurements
            
        Returns:
            bool: True if duration set successfully
        """
        if duration_seconds <= 0:
            self.logger.error(f"Invalid measurement duration: {duration_seconds}")
            return False
            
        self._measurement_duration = duration_seconds
        
        # If hardware driver, update its configuration
        if hasattr(self.driver, 'set_measurement_duration'):
            return self.driver.set_measurement_duration(duration_seconds)
        
        self.logger.debug(f"Measurement duration set to {duration_seconds}s")
        return True
    

    def get_measurement_duration(self) -> Optional[float]:
        """Get current measurement duration."""
        return self._measurement_duration
        
    def get_detector_channels(self) -> list:
        """Get list of detector channels."""
        return self.driver.detector_channels
    
    def is_initialized(self) -> bool:
        """Check if controller is initialized."""
        return self._initialized
  
    def get_status(self) -> Dict:
        """Get controller status."""
        return {
            "initialized": self._initialized,
            "measurement_duration": self._measurement_duration,
            "driver_type": type(self.driver).__name__,
            "detector_channels": self.driver.detector_channels
        }
    