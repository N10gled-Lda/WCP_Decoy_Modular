"""Laser Hardware Interface using Digilent Device."""
import logging
import time
from typing import Optional, List, Dict, Any
from functools import wraps

from src.alice.laser.laser_base import BaseLaserDriver
from .hardware_laser.digilent_interface import DigilentInterface, TriggerMode


def ensure_connected(fn):
    """Decorator to ensure the laser hardware is connected before operation."""
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self.digilent.connected:
            self.logger.error("Digilent device is not connected.")
            raise RuntimeError("Laser hardware must be connected first. Call initialize().")
        return fn(self, *args, **kwargs)
    return wrapper


class HardwareLaserDriver(BaseLaserDriver):
    """Hardware laser driver using Digilent device for triggering."""
    
    def __init__(self, device_index: int = -1, trigger_channel: int = 0):
        """
        Initialize the hardware laser driver.
        
        Args:
            device_index: Index of the Digilent device to use (-1 for first available)
            trigger_channel: Analog output channel to use for laser triggering
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize Digilent interface
        self.digilent = DigilentInterface(device_index=device_index, trigger_channel=trigger_channel)
        
        # Laser state
        self._is_on = False
        self._is_armed = False
        self._current_frequency = 0.0
        
        # Default pulse parameters
        self.default_amplitude = 5.0  # 5V trigger pulse
        self.default_width = 1e-6     # 1μs pulse width
        self.default_frequency = 1000.0  # 1kHz
        
        # Set up callbacks
        self.digilent.on_connected = self._on_connected
        self.digilent.on_disconnected = self._on_disconnected
        self.digilent.on_error = self._on_error
        
        self.logger.info(f"Hardware laser driver initialized for device {device_index}, channel {trigger_channel}")

    def _on_connected(self):
        """Callback when Digilent device connects."""
        self.logger.info("Digilent device connected successfully")
        # Set default pulse parameters
        self.digilent.set_pulse_parameters(
            self.default_amplitude,
            self.default_width,
            self.default_frequency
        )

    def _on_disconnected(self):
        """Callback when Digilent device disconnects."""
        self.logger.info("Digilent device disconnected")
        self._is_on = False
        self._is_armed = False

    def _on_error(self, error_msg: str):
        """Callback when Digilent device encounters an error."""
        self.logger.error(f"Digilent device error: {error_msg}")

    def initialize(self) -> bool:
        """
        Initialize and connect to the Digilent device and make it ready to emit.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.digilent.connect()
            if success:
                # Set default pulse parameters
                self.digilent.set_pulse_parameters(
                    self.default_amplitude,
                    self.default_width,
                    self.default_frequency
                )
                
                # Enable the trigger system
                self._is_on = True
                self.logger.info("Laser hardware initialized and ready")
            else:
                self.logger.error("Failed to initialize laser hardware")
            return success
        except Exception as e:
            self.logger.error(f"Error initializing laser hardware: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown the laser hardware."""
        try:
            if self.digilent.running:
                self.digilent.stop_continuous()
            self.digilent.disconnect()
            self._is_on = False
            self._is_armed = False
            self.logger.info("Laser hardware shut down")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    @ensure_connected
    def trigger_once(self) -> bool:
        """Send a single trigger pulse."""
        if not self._is_on:
            self.logger.warning("Laser not ready - call initialize() first")
            return False
        
        try:
            success = self.digilent.fire_single_pulse()
            if success:
                self.logger.debug("Single trigger pulse sent")
            else:
                self.logger.error("Failed to send trigger pulse")
            return success
        except Exception as e:
            self.logger.error(f"Error sending trigger pulse: {e}")
            return False

    @ensure_connected
    def send_frame(self, n_triggers: int, rep_rate_hz: float) -> bool:
        """
        Send a frame of multiple trigger pulses.
        
        Args:
            n_triggers: Number of trigger pulses
            rep_rate_hz: Repetition rate in Hz
        """
        if not self._is_on:
            self.logger.error("Laser must be turned on before sending frame")
            return False
        
        try:
            # Update pulse parameters
            self.digilent.set_pulse_parameters(
                self.default_amplitude,
                self.default_width,
                rep_rate_hz
            )
            
            # Send burst of pulses
            success = self.digilent.fire_burst(n_triggers, rep_rate_hz)
            if success:
                self.logger.info(f"Sent frame with {n_triggers} pulses at {rep_rate_hz} Hz")
            else:
                self.logger.error("Failed to send frame")
            return success
        except Exception as e:
            self.logger.error(f"Error sending frame: {e}")
            return False

    @ensure_connected
    def start_continuous(self, rep_rate_hz: float) -> bool:
        """
        Start continuous laser emission.
        
        Args:
            rep_rate_hz: Pulse repetition rate in Hz
        """
        if not self._is_on:
            self.logger.error("Laser must be turned on before starting continuous mode")
            return False
        
        try:
            success = self.digilent.start_continuous(rep_rate_hz)
            if success:
                self._current_frequency = rep_rate_hz
                self.logger.info(f"Started continuous mode at {rep_rate_hz} Hz")
            else:
                self.logger.error("Failed to start continuous mode")
            return success
        except Exception as e:
            self.logger.error(f"Error starting continuous mode: {e}")
            return False

    @ensure_connected
    def stop_continuous(self) -> bool:
        """Stop continuous laser emission."""
        try:
            success = self.digilent.stop_continuous()
            if success:
                self.logger.info("Stopped continuous mode")
            else:
                self.logger.error("Failed to stop continuous mode")
            return success
        except Exception as e:
            self.logger.error(f"Error stopping continuous mode: {e}")
            return False

    def set_pulse_parameters(self, amplitude: float, width: float, frequency: float) -> None:
        """
        Set pulse parameters for laser triggering.
        
        Args:
            amplitude: Trigger pulse amplitude in volts
            width: Trigger pulse width in seconds
            frequency: Pulse repetition frequency in Hz
        """
        self.default_amplitude = amplitude
        self.default_width = width
        self.default_frequency = frequency
        
        if self.digilent.connected:
            self.digilent.set_pulse_parameters(amplitude, width, frequency)
        
        self.logger.info(f"Set pulse parameters: {amplitude}V, {width*1e6:.1f}μs, {frequency}Hz")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the laser hardware.
        
        Returns:
            Dictionary with status information
        """
        try:
            digilent_status = self.digilent.get_status()
        except Exception as e:
            digilent_status = {"error": str(e)}
        
        return {
            "driver_type": "hardware_digilent",
            "initialized": True,
            "active": self._is_on,
            "armed": self._is_armed,
            "current_frequency": self._current_frequency,
            "default_amplitude": self.default_amplitude,
            "default_width": self.default_width,
            "default_frequency": self.default_frequency,
            "digilent_status": digilent_status
        }

    def is_connected(self) -> bool:
        """Check if the hardware is connected."""
        return self.digilent.connected

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        try:
            self.shutdown()
        except:
            pass  # Ignore errors during cleanup






# class HardwareLaserDriver(BaseLaserDriver):
#     def __init__(self, resource: str, trigger_mode: str = "internal"):
#         # TODO: Adapt the instant of the hardware driver to your specific hardware
#         self._inst = pylist.ResourceManager().open_resource(resource)
#         self._trigger_mode = trigger_mode
#         self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
#         self.logger.info("Hardware laser driver initialized.")

#     def turn_on(self):
#         # TODO: Implement hardware laser
#         """Turn on the laser hardware."""
#         self._inst.write("OUTP ON")
#         self.logger.info("Laser hardware turned on.")

#     def turn_off(self):
#         # TODO: Implement hardware laser
#         """Turn off the laser hardware."""
#         self._inst.write("OUTP OFF")
#         self.logger.info("Laser hardware turned off.")

#     def stop(self):
#         # TODO: Implement hardware laser
#         self._inst.write("OUTP OFF")
#         self.logger.info("Laser hardware stopped.")

#     def arm(self, repetition_rate_hz: float) -> None:
#         # TODO: Implement hardware laser
#         self._inst.write(f"SOUR:OSC:FREQ {repetition_rate_hz}")
#         self._inst.write(f"TRIG:MODE {self._trigger_mode}")

#     def fire(self, pattern):
#         # TODO: Implement hardware laser
#         for p in pattern:
#             self._inst.write(
#                 f"PULS:WIDT {p.width_ps}PS; POW {p.energy_pJ}PJ; SHAP {p.shape}"
#             )
#             self._inst.write("PULS:IMM")          # single-shot
#             self._inst.query("*OPC?")             # wait for completion

#     def fire_single_pulse(self, power: float = None, linewidth: float = None, wavelength: float = None) -> None:
#         """Emit a single pulse with optional parameters."""
#         # TODO: Implement hardware-specific single pulse emission. Not supported???
#         pass
    
#     # IDEAS FOR FUTURE HARDWARE FUNCTIONS
#     # def set_power(self, power_mW: float):
