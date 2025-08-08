"""Laser Hardware Interface using Digilent Device."""
import logging
from typing import Optional, List
from functools import wraps

from src.alice.laser.laser_controller import BaseLaserDriver
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
        self.digilent = DigilentInterface(device_index=device_index)
        self.digilent.trigger_channel = trigger_channel
        
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
        Initialize and connect to the Digilent device.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.digilent.connect()
            if success:
                self.logger.info("Laser hardware initialized successfully")
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
    def turn_on(self) -> None:
        """Turn on the laser (enable trigger output)."""
        try:
            # The laser itself is controlled by external hardware
            # Here we just enable the trigger system
            self._is_on = True
            self.logger.info("Laser trigger system turned on")
        except Exception as e:
            self.logger.error(f"Error turning on laser: {e}")
            raise

    @ensure_connected
    def turn_off(self) -> None:
        """Turn off the laser (disable trigger output)."""
        try:
            # Stop any continuous operation
            if self.digilent.running:
                self.digilent.stop_continuous()
            
            self._is_on = False
            self._is_armed = False
            self.logger.info("Laser trigger system turned off")
        except Exception as e:
            self.logger.error(f"Error turning off laser: {e}")
            raise

    @ensure_connected
    def stop(self) -> None:
        """Safely stop laser emission."""
        try:
            if self.digilent.running:
                self.digilent.stop_continuous()
            self._is_armed = False
            self.logger.info("Laser emission stopped")
        except Exception as e:
            self.logger.error(f"Error stopping laser: {e}")
            raise

    @ensure_connected
    def arm(self, repetition_rate_hz: float) -> None:
        """
        Prepare the laser for a sequence of pulses.
        
        Args:
            repetition_rate_hz: Pulse repetition rate in Hz
        """
        if not self._is_on:
            raise RuntimeError("Laser must be turned on before arming")
        
        try:
            self._current_frequency = repetition_rate_hz
            
            # Update pulse parameters on Digilent device
            self.digilent.set_pulse_parameters(
                self.default_amplitude,
                self.default_width,
                repetition_rate_hz
            )
            
            self._is_armed = True
            self.logger.info(f"Laser armed at {repetition_rate_hz} Hz")
        except Exception as e:
            self.logger.error(f"Error arming laser: {e}")
            raise

    @ensure_connected
    def fire(self, pattern: List) -> None:
        """
        Emit pulses according to a pattern.
        
        Args:
            pattern: List of pulse parameters (not fully implemented yet)
        """
        if not self._is_on:
            raise RuntimeError("Laser must be turned on before firing")
        
        try:
            # For now, fire a burst equal to the pattern length
            pulse_count = len(pattern) if pattern else 1
            frequency = self._current_frequency if self._current_frequency > 0 else self.default_frequency
            
            success = self.digilent.fire_burst(pulse_count, frequency)
            if not success:
                raise RuntimeError("Failed to fire pulse pattern")
            
            self.logger.info(f"Fired pattern of {pulse_count} pulses at {frequency} Hz")
        except Exception as e:
            self.logger.error(f"Error firing pulse pattern: {e}")
            raise

    @ensure_connected
    def fire_single_pulse(self, power: Optional[float] = None, 
                         linewidth: Optional[float] = None, 
                         wavelength: Optional[float] = None) -> None:
        """
        Emit a single pulse.
        
        Args:
            power: Pulse power (not used for trigger-based system)
            linewidth: Pulse linewidth (not used for trigger-based system)
            wavelength: Pulse wavelength (not used for trigger-based system)
        """
        if not self._is_on:
            raise RuntimeError("Laser must be turned on before firing")
        
        try:
            success = self.digilent.fire_single_pulse()
            if not success:
                raise RuntimeError("Failed to fire single pulse")
            
            self.logger.info("Fired single pulse")
        except Exception as e:
            self.logger.error(f"Error firing single pulse: {e}")
            raise

    def start_continuous(self, repetition_rate_hz: float) -> None:
        """
        Start continuous laser emission.
        
        Args:
            repetition_rate_hz: Pulse repetition rate in Hz
        """
        if not self._is_on:
            raise RuntimeError("Laser must be turned on before starting continuous mode")
        
        try:
            success = self.digilent.start_continuous(repetition_rate_hz)
            if not success:
                raise RuntimeError("Failed to start continuous mode")
            
            self._current_frequency = repetition_rate_hz
            self.logger.info(f"Started continuous mode at {repetition_rate_hz} Hz")
        except Exception as e:
            self.logger.error(f"Error starting continuous mode: {e}")
            raise

    def stop_continuous(self) -> None:
        """Stop continuous laser emission."""
        try:
            self.digilent.stop_continuous()
            self.logger.info("Stopped continuous mode")
        except Exception as e:
            self.logger.error(f"Error stopping continuous mode: {e}")
            raise

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

    def get_status(self) -> dict:
        """
        Get current status of the laser hardware.
        
        Returns:
            Dictionary with status information
        """
        digilent_status = self.digilent.get_status()
        
        return {
            'laser_on': self._is_on,
            'armed': self._is_armed,
            'current_frequency': self._current_frequency,
            'digilent_status': digilent_status
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
