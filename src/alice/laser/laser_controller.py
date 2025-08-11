"""Laser Controller."""
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Union
import time

from src.alice.laser.laser_base import BaseLaserDriver
from src.utils.data_structures import Pulse


# MIGHT HAVE ISSUES WITH IMPORTS
from src.alice.laser.laser_simulator import SimulatedLaserDriver
from src.alice.laser.laser_hardware import HardwareLaserDriver
from src.alice.laser.laser_hardware_digital import DigitalHardwareLaserDriver


class LaserController:
    """
    Laser controller that holds protocol-level state and delegates physical work to a pluggable driver.
    """
    
    # def __init__(self, driver: BaseLaserDriver):
    def __init__(self, driver: Union[BaseLaserDriver, SimulatedLaserDriver, HardwareLaserDriver, DigitalHardwareLaserDriver]):
        """
        Initialize the laser controller with a specific driver.
        
        Args:
            driver: The laser driver (simulator or hardware)
        """
        self._d = driver
        # TODO: Decide if we want as input the driver or the hardware flag to initialize the driver in the controller.
        # def __init__(self, physical_hardware: bool = False):
        #     self.physical_hardware = physical_hardware
        #     if self.physical_hardware:
        #         self.laser = LaserHardware()
        #     else:
        #         self.laser = LaserSimulator()

        # TODO: Decide if we want as input the driver or the hardware flag to initialize the driver in the controller.
        # def __init__(self, physical_hardware: bool = False):
        #     self.physical_hardware = physical_hardware
        #     if self.physical_hardware:
        #         self.laser = LaserHardware()
        #     else:
        #         self.laser = LaserSimulator()

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # State variables
        self._active = False
        self._continuous = False
        self._initialized = False

        self.pulse_count = 0
        self.frame_count = 0
        self.total_pulses_fired = 0

        self.logger.info(f"Laser controller initialized with {type(driver).__name__}")

        
    # TODO: Decide if we want to use context manager for the laser controller
    # def __enter__(self):
    #     self.turn_on()
    #     return self
        
    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     if self._continuous:
    #         self.stop_continuous()
    #     self.turn_off()

    def initialize(self) -> bool:
        """
        Initialize the laser controller and underlying hardware.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if hasattr(self._d, 'initialize'):
                success = self._d.initialize()
                if not success:
                    self.logger.error("Failed to initialize laser driver")
                    return False
            
            self._initialized = True
            self.logger.info("Laser controller initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing laser controller: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown the laser controller and underlying hardware."""
        try:
            if self._continuous:
                self.stop_continuous()
            
            if self._active:
                self.turn_off()
            
            if hasattr(self._d, 'shutdown'):
                self._d.shutdown()
            
            self._initialized = False
            self.logger.info("Laser controller shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during laser controller shutdown: {e}")

    def _assert(self, condition: bool, message: str) -> None:
        """Assert a condition and raise an error with a message if it fails."""
        if not condition:
            raise RuntimeError(message)

    def _ensure_initialized(self) -> None:
        """Ensure the controller is initialized."""
        self._assert(self._initialized, "Laser controller must be initialized first. Call initialize().")

    # ---- Turn on/off -----------------------------------
    def turn_on(self) -> None:
        """Turn on the laser."""
        self._ensure_initialized()
        
        try:
            self._d.turn_on()
            self._active = True
            self.logger.info("Laser turned on")
        except Exception as e:
            self.logger.error(f"Error turning on laser: {e}")
            raise

    def turn_off(self) -> None:
        """Turn off the laser."""
        self._ensure_initialized()
        
        try:
            if self._continuous:
                self.stop_continuous()
            
            self._d.turn_off()
            self._active = False
            self.logger.info("Laser turned off")
        except Exception as e:
            self.logger.error(f"Error turning off laser: {e}")
            raise

    def is_active(self) -> bool:
        """Check if the laser is currently active."""
        self.logger.debug(f"Laser active status: {self._active}")
        return self._active

    def is_initialized(self) -> bool:
        """Check if the laser controller is initialized."""
        return self._initialized

    # ---- 'continuous' mode ----------------------------------
    def start_continuous(self, rep_rate_hz: float) -> None:
        """
        Start continuous laser emission.
        
        Args:
            rep_rate_hz: Pulse repetition rate in Hz
        """
        self._assert(self._active, "Cannot start continuous mode when laser is off.")

        try:
            self._d.arm(rep_rate_hz)
            
            # For hardware that supports continuous mode directly
            if hasattr(self._d, 'start_continuous'):
                self._d.start_continuous(rep_rate_hz)
            
            self._continuous = True
            self.logger.info(f"Started continuous mode at {rep_rate_hz} Hz")
        except Exception as e:
            self.logger.error(f"Error starting continuous mode: {e}")
            raise

    def stop_continuous(self) -> None:
        """Stop continuous laser emission."""
        self._assert(self._active, "Cannot stop continuous mode when laser is off.")
        self._assert(self._continuous, "Continuous mode is not active.")
        
        try:
            # For hardware that supports continuous mode directly
            if hasattr(self._d, 'stop_continuous'):
                self._d.stop_continuous()
            else:
                self._d.stop()
            
            self._continuous = False
            self.logger.info("Stopped continuous mode")
        except Exception as e:
            self.logger.error(f"Error stopping continuous mode: {e}")
            raise

    # ---- 'triggered' mode -----------------------------------
    def send_frame(self, n_triggers: int, rep_rate_hz: float) -> None:
        """
        Emit n_triggers pulses at the given rep-rate, one trigger at a time.
        
        Args:
            n_triggers: Number of pulses to emit
            rep_rate_hz: Pulse repetition rate in Hz
        """
        self._assert(self._active, "Cannot send frame when laser is off.")

        try:
            self._d.arm(rep_rate_hz)
            
            for i in range(n_triggers):
                self._d.fire_single_pulse()
                self.pulse_count += 1
                self.total_pulses_fired += 1
                
                if i < n_triggers - 1:  # Don't wait after the last pulse
                    time.sleep(1 / rep_rate_hz)

            self._d.stop()
            self.frame_count += 1
            
            self.logger.info(f"Sent frame with {n_triggers} pulses at {rep_rate_hz} Hz")
        except Exception as e:
            self.logger.error(f"Error sending frame: {e}")
            raise

    def send_pattern(self, pattern: List, rep_rate_hz: float) -> None:
        """
        Emit pulses according to a specific pattern.
        
        Args:
            pattern: List defining the pulse pattern
            rep_rate_hz: Base repetition rate in Hz
        """
        self._assert(self._active, "Cannot send pattern when laser is off.")

        try:
            self._d.arm(rep_rate_hz)
            self._d.fire(pattern)
            self._d.stop()
            
            self.pulse_count += len(pattern)
            self.total_pulses_fired += len(pattern)
            self.frame_count += 1
            
            self.logger.info(f"Sent pattern with {len(pattern)} pulses")
        except Exception as e:
            self.logger.error(f"Error sending pattern: {e}")
            raise

    # ---- 'single-shot' mode -----------------------------
    def trigger_once(self) -> None:
        """Emit a single pulse."""
        self._assert(self._active, "Cannot trigger laser when it is off.")
        
        try:
            self._d.fire_single_pulse()
            self.pulse_count += 1
            self.total_pulses_fired += 1
            self.logger.info("Single pulse triggered")
        except Exception as e:
            self.logger.error(f"Error triggering single pulse: {e}")
            raise

    # ---- Status and monitoring -------------------------
    def get_status(self) -> dict:
        """
        Get current status of the laser controller.
        
        Returns:
            Dictionary with status information
        """
        status = {
            'initialized': self._initialized,
            'active': self._active,
            'continuous': self._continuous,
            'pulse_count': self.pulse_count,
            'frame_count': self.frame_count,
            'total_pulses_fired': self.total_pulses_fired
        }
        
        # Add driver-specific status if available
        if hasattr(self._d, 'get_status'):
            status['driver_status'] = self._d.get_status()
        
        return status

    def reset_counters(self) -> None:
        """Reset pulse and frame counters."""
        self.pulse_count = 0
        self.frame_count = 0
        self.total_pulses_fired = 0
        self.logger.info("Counters reset")

    # ---- Context manager support -----------------------
    def __enter__(self):
        """Context manager entry."""
        if not self._initialized:
            self.initialize()
        self.turn_on()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._continuous:
            self.stop_continuous()
        if self._active:
            self.turn_off()
        self.shutdown()

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        try:
            self.shutdown()
        except:
            pass  # Ignore errors during cleanup













    # ---- Turn on/off -----------------------------------
    def turn_on(self):
        """
        Turn on the laser.
        """
        self._d.turn_on()
        self._active = True
        self.logger.info("Laser turned on.")

    def turn_off(self):
        """
        Turn off the laser.
        """
        self._d.turn_off()
        self._active = False
        self.logger.info("Laser turned off.")

    def is_active(self) -> bool:
        """
        Check if the laser is currently active.
        """
        self.logger.debug(f"Laser active status: {self._active}")
        return self._active
    
    ### ---- Modes of operation -----------------------------
    ### TODO: NOT SURE IF THIS GOES IN THE CONTROLLER OR ONLY IN THE DRIVER
    ### AND LET IT BE ONLY TURNING ON/OFF THE LASER

    def _assert(self, condition: bool, message: str) -> None:
        """
        Assert a condition and raise an error with a message if it fails.
        """
        if not condition:
            raise RuntimeError(message)
        
    # ---- ‘continuous’ mode ----------------------------------
    def start_continuous(self, rep_rate_hz: float):
        """
        Start continuous laser emission.
        """
        self._assert(self._active, "Cannot start continuous mode when laser is off.")

        self._d.arm(rep_rate_hz)
        self._continuous = True

    def stop_continuous(self):
        """
        Stop continuous laser emission.
        """
        self._assert(self._active, "Cannot stop continuous mode when laser is off.")
        self._assert(self._continuous, "Continuous mode is not active.")
        
        self._d.stop()
        self._continuous = False

    # ---- ‘triggered’ mode -----------------------------------
    def send_frame(self, n_triggers: int, rep_rate_hz: float):
        """
        Emit *n_triggers* pulses at the given rep-rate,
        one trigger at a time.
        """
        self._assert(self._active, "Cannot send frame when laser is off.")

        for _ in range(n_triggers):
            self._d.fire_single_pulse()
            time.sleep(1 / rep_rate_hz)

        self._d.stop()

    # ---- ‘single-shot’ mode -----------------------------
    def trigger_once(self):
        """
        Emit a single pulse.
        """
        self._assert(self._active, "Cannot trigger laser when it is off.")
        
        self._d.fire_single_pulse()
        self.logger.info("Single pulse triggered.")




# # Choose driver at runtime:
# driver = HardwareLaserDriver("USB0::...") if use_hw else SimulatedLaserDriver()
# controller = LaserController(driver)
# controller.send_frame()