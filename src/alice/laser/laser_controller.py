"""Laser Controller."""
import logging
from typing import Optional, List, Union
import time

from src.alice.laser.laser_base import BaseLaserDriver
from src.utils.data_structures import Pulse


# Import specific driver classes for type hints
from src.alice.laser.laser_simulator import SimulatedLaserDriver
from src.alice.laser.laser_hardware_digital import (
    DigitalHardwareLaserDriver,
    create_digital_laser_driver,
)

class LaserController:
    """
    Laser controller that holds protocol-level state and delegates physical work to a pluggable driver.
    """
    
    # def __init__(self, driver: BaseLaserDriver):
    def __init__(self, driver: Union[BaseLaserDriver, SimulatedLaserDriver, DigitalHardwareLaserDriver]): # type: ignore
        """
        Initialize the laser controller with a specific driver.
        
        Args:
            driver: The laser driver (simulator or hardware)
        """
        self._driver = driver
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # State variables
        self._active = False
        self._continuous = False
        self._initialized = False

        # Statistics
        self.pulse_count = 0
        self.frame_count = 0
        self.total_pulses_fired = 0

        self.logger.info(f"Laser controller initialized with {type(driver).__name__}")

    def initialize(self) -> bool:
        """
        Initialize the laser controller and underlying hardware.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            success = self._driver.initialize()
            if not success:
                self.logger.error("Failed to initialize laser driver")
                return False
            
            self._initialized = True
            self._active = False  # Only send if not already active
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
            
            self._driver.shutdown()
            
            self._initialized = False
            self._active = False
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

    def is_active(self) -> bool:
        """Check if the laser is currently active."""
        return self._active and self._initialized

    def is_initialized(self) -> bool:
        """Check if the laser controller is initialized."""
        return self._initialized

    # ---- 'continuous' mode ----------------------------------
    def start_continuous(self, rep_rate_hz: float) -> bool:
        """
        Start continuous laser emission.
        
        Args:
            rep_rate_hz: Repetition rate in Hz
            
        Returns:
            True if successful
        """
        self._assert(self._initialized, "Cannot start continuous mode when laser is not initialized.")
        self._assert(not self._continuous, "Continuous mode is already active.")
        self._assert(not self._active, "Cannot start continuous mode when laser is already active.")

        try:
            success = self._driver.start_continuous(rep_rate_hz=rep_rate_hz)
            if success:
                self._continuous = True
                self._active = True
                self.logger.info(f"Started continuous mode at {rep_rate_hz} Hz")
            else:
                self.logger.error("Failed to start continuous mode")
            return success
        except Exception as e:
            self.logger.error(f"Error starting continuous mode: {e}")
            raise

    def stop_continuous(self) -> bool:
        """Stop continuous laser emission."""
        self._assert(self._initialized, "Cannot stop continuous mode when laser is not initialized.")
        self._assert(self._active, "Cannot stop continuous mode when laser is not active.")
        self._assert(self._continuous, "Continuous mode is not active.")
        
        try:
            success = self._driver.stop_continuous()
            if success:
                self._continuous = False
                self.logger.info("Stopped continuous mode")
                self._active = False
            else:
                self.logger.error("Failed to stop continuous mode")
            return success
        except Exception as e:
            self.logger.error(f"Error stopping continuous mode: {e}")
            raise

    # ---- 'triggered' mode -----------------------------------
    def send_frame(self, n_triggers: int, rep_rate_hz: float) -> bool:
        """
        Send a single frame with specific parameters.
        
        Args:
            n_triggers: Number of trigger pulses
            rep_rate_hz: Repetition rate in Hz
            
        Returns:
            True if successful
        """
        self._assert(self._initialized, "Cannot send frame when laser is not initialized.")
        self._assert(not self._active, "Cannot send frame when laser is already active.")

        try:
            self._active = True  # Set active state before sending frame
            success = self._driver.send_frame(n_triggers, rep_rate_hz)
            if success:
                self.pulse_count += 1
                self.total_pulses_fired += 1
                self.frame_count += 1
                self.logger.info(f"Sent frame at {rep_rate_hz} Hz")
            else:
                self.logger.error("Failed to send frame")
            self._active = False  # Reset active state after sending
            return success
        except Exception as e:
            self.logger.error(f"Error sending frame: {e}")
            raise

    # ---- 'single-shot' mode -----------------------------
    def trigger_once(self) -> bool:
        """
        Emit a single pulse.
        
        Returns:
            True if successful
        """
        self._assert(self._initialized, "Cannot trigger laser when it is not initialized.")
        self._assert(not self._active, "Cannot send frame when laser is already active.")
        
        try:
            success = self._driver.trigger_once()
            if success:
                self.pulse_count += 1
                self.total_pulses_fired += 1
                self.logger.info("Single pulse triggered")
            else:
                self.logger.error("Failed to trigger single pulse")
            return success
        except Exception as e:
            self.logger.error(f"Error triggering single pulse: {e}")
            raise

    def set_pulse_parameters(self, duty_cycle: Optional[float] = None, frequency: Optional[float] = None, idle_state: Optional[bool] = None) -> bool:
        """Configure the underlying driver's pulse-generation parameters.

        Args:
            duty_cycle: Optional desired duty cycle. Accepts 0.0-1.0; values outside the range are clamped by the driver.
            frequency: Optional pulse repetition frequency in Hz. When omitted the driver's current frequency is preserved.
            idle_state: Optional idle state (True for logic high) for hardware drivers that expose this attribute.

        Returns:
            True if the driver successfully applied the update, False otherwise.
        """

        self._ensure_initialized()

        driver_setter = getattr(self._driver, "set_pulse_parameters", None)
        if not callable(driver_setter):
            raise NotImplementedError("The configured laser driver does not support pulse parameter updates.")

        if idle_state is not None and hasattr(self._driver, "idle_state"):
            try:
                setattr(self._driver, "idle_state", bool(idle_state))
            except Exception as exc:
                self.logger.warning(f"Failed to update driver idle state: {exc}")

        try:
            driver_setter(duty_cycle, frequency)
            self.logger.info(
                "Laser pulse parameters updated via controller: duty_cycle=%s, frequency=%s", duty_cycle, frequency
            )
            return True
        except Exception as exc:
            self.logger.error(f"Error applying pulse parameters: {exc}")
            return False
        
    # ---- Status and monitoring -------------------------
    def get_status(self) -> dict:
        """
        Get current status of the laser controller.
        
        Returns:
            Dictionary with status information
        """
        status = {
            'controller': {
                'initialized': self._initialized,
                'active': self._active,
                'continuous': self._continuous,
                'pulse_count': self.pulse_count,
                'frame_count': self.frame_count,
                'total_pulses_fired': self.total_pulses_fired
            }
        }
        
        # Add driver-specific status
        try:
            status['driver'] = self._driver.get_status()
        except Exception as e:
            status['driver'] = {"error": str(e)}
        
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
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._continuous:
            self.stop_continuous()
        self.shutdown()

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        try:
            self.shutdown()
        except:
            pass  # Ignore errors during cleanup


def create_laser_controller_with_hardware(
    device_index: int = -1, digital_channel: int = 8
) -> "LaserController":
    """Convenience factory returning a hardware-backed laser controller."""

    driver = create_digital_laser_driver(device_index=device_index, digital_channel=digital_channel)
    return LaserController(driver)
