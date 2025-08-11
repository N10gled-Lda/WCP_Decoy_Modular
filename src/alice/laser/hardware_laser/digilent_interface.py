"""Digilent Interface for Laser Control using WaveForms SDK."""
import logging
import threading
import time
from ctypes import *
from typing import Optional, Callable, List
from enum import Enum
import platform

# Try to import the WaveForms SDK
try:
    if platform.system() == "Windows":
        dwf = cdll.dwf  # Windows
    else:
        dwf = cdll.LoadLibrary("libdwf.so")  # Linux/Mac
except OSError:
    dwf = None
    print("Warning: WaveForms SDK not found. Hardware functionality will be limited.")

# Import constants (fallback if not available)
from .dwfconstants import *
# try:
#     from dwfconstants import *
# except ImportError:
#     # Fallback constants if dwfconstants is not available
#     hdwfNone = c_int(0)
#     DwfStateReady = c_ubyte(0)
#     DwfStateDone = c_ubyte(2)
#     funcDC = c_ubyte(0)
#     funcPulse = c_ubyte(7)


class TriggerMode(Enum):
    """Trigger modes for laser control."""
    SINGLE = "single"
    CONTINUOUS = "continuous"
    TRAIN = "train"


class DigilentInterface:
    """Interface for controlling laser through Digilent device using WaveForms SDK."""

    def __init__(self, device_index: int = -1, trigger_channel: int = 0, auto_configure: c_int = DWFUser.AutoConfigure.DISABLED.value):
        """
        Initialize the Digilent interface.
        
        Args:
            device_index: Index of the Digilent device to use (-1 for first available)
            trigger_channel: Analog output channel for laser triggering
            auto_configure: Auto Configure setting (default is disabled)
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Device management
        self.dwf = dwf
        self.hdwf = c_int()
        self.autoConfigure = auto_configure  # Auto Configure setting
        self.device_index = device_index
        self.connected = False
        self.running = False
        
        # Laser control parameters
        self.trigger_channel = trigger_channel  # Analog output channel for trigger
        self.pulse_amplitude = 5.0  # Volts
        self.pulse_width = 1e-6  # 1 microsecond
        self.repetition_rate = 1000.0  # Hz
        
        # Threading for continuous operation
        self._control_thread = None
        self._stop_event = threading.Event()
        
        # Callbacks
        self.on_connected: Optional[Callable] = None
        self.on_disconnected: Optional[Callable] = None
        self.on_pulse_complete: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        self.logger.info("Digilent interface initialized")

    def connect(self) -> bool:
        """
        Connect to the Digilent device.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.dwf:
            self.logger.error("WaveForms SDK not available")
            return False
        
        try:
            # Open device
            if self.device_index == -1:
                # Open first available device
                self.dwf.FDwfDeviceOpen(DWFUser.Devices.FIRST.value, byref(self.hdwf))
            else:
                self.dwf.FDwfDeviceOpen(c_int(self.device_index), byref(self.hdwf))
            
            if self.hdwf.value == hdwfNone.value:
                error_msg = create_string_buffer(512)
                self.dwf.FDwfGetLastErrorMsg(error_msg)
                self.logger.error(f"Failed to open device: {error_msg.value.decode()}")
                return False
            
            # Configure device for laser control
            self._configure_device()
            
            self.connected = True
            self.logger.info("Successfully connected to Digilent device")
            
            if self.on_connected:
                self.on_connected()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to device: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from the Digilent device."""
        if self.running:
            self.stop_continuous()
        
        if self.connected and self.hdwf.value != hdwfNone.value:
            try:
                # Reset all analog outputs
                self.dwf.FDwfAnalogOutReset(self.hdwf, DWFUser.ALL_CHANNELS)
                self.dwf.FDwfAnalogOutConfigure(self.hdwf, DWFUser.ALL_CHANNELS, DWFUser.OutputBehaviour.APPLY.value)
                # self.dwf.FDwfAnalogOutConfigure(self.hdwf, DWFUser.ALL_CHANNELS, c_bool(False))
                
                # Close device
                self.dwf.FDwfDeviceClose(self.hdwf)
                
                self.connected = False
                self.logger.info("Disconnected from Digilent device")
                
                if self.on_disconnected:
                    self.on_disconnected()
                    
            except Exception as e:
                self.logger.error(f"Error during disconnect: {e}")

    def _configure_device(self) -> None:
        """Configure the device for laser triggering."""
        try:
            # Enable auto-configuration for immediate updates
            self.dwf.FDwfDeviceAutoConfigureSet(self.hdwf, self.autoConfigure)
            
            # Configure trigger channel
            channel = c_int(self.trigger_channel)
            
            # Enable the channel
            self.dwf.FDwfAnalogOutEnableSet(self.hdwf, channel, c_bool(True))
            
            # Set to pulse function initially
            self.dwf.FDwfAnalogOutFunctionSet(self.hdwf, channel, funcPulse)
            
            # Set initial amplitude
            self.dwf.FDwfAnalogOutAmplitudeSet(self.hdwf, channel, c_double(self.pulse_amplitude))
            
            # Set initial frequency (for repetition rate)
            self.dwf.FDwfAnalogOutFrequencySet(self.hdwf, channel, c_double(self.repetition_rate))
            
            self.logger.info(f"Device configured for laser triggering on channel {self.trigger_channel}")
            
        except Exception as e:
            self.logger.error(f"Error configuring device: {e}")
            raise

    def set_pulse_parameters(self, amplitude: float, width: float, frequency: float) -> None:
        """
        Set pulse parameters for laser triggering.
        
        Args:
            amplitude: Pulse amplitude in volts
            width: Pulse width in seconds
            frequency: Repetition frequency in Hz
        
        Using the enum DWFUser.AnalogOutFunctions for function selection
        """
        self.pulse_amplitude = amplitude
        self.pulse_width = width
        self.repetition_rate = frequency
        
        if self.connected:
            try:
                channel = c_int(self.trigger_channel)
                
                # Update amplitude
                self.dwf.FDwfAnalogOutAmplitudeSet(self.hdwf, channel, c_double(amplitude))
                
                # Update frequency
                self.dwf.FDwfAnalogOutFrequencySet(self.hdwf, channel, c_double(frequency))
                
                # Calculate duty cycle from pulse width and frequency
                duty_cycle = width * frequency * 100  # Convert to percentage
                if duty_cycle > 90.0:
                    self.logger.warning(f"Duty cycle {duty_cycle}% exceeds 90%, clamping to 90%")
                duty_cycle = min(90.0, max(1.0, duty_cycle))  # Clamp between 1% and 90%
                self.logger.debug(f"Setting duty cycle: {duty_cycle}%")

                # Set symmetry (duty cycle)
                self.dwf.FDwfAnalogOutSymmetrySet(self.hdwf, channel, c_double(duty_cycle))
                
                self.logger.info(f"Updated pulse parameters: {amplitude}V, {width*1e6:.1f}μs, {frequency}Hz")
                
            except Exception as e:
                self.logger.error(f"Error setting pulse parameters: {e}")

    def fire_single_pulse(self) -> bool:
        """
        Fire a single laser pulse.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            self.logger.error("Device not connected")
            return False
        
        try:
            channel = c_int(self.trigger_channel)
            
            # Set to pulse mode
            self.dwf.FDwfAnalogOutFunctionSet(self.hdwf, channel, funcPulse)
            
            # Configure for single shot
            self.dwf.FDwfAnalogOutRunSet(self.hdwf, channel, c_double(self.pulse_width))
            
            # Start the pulse
            self.dwf.FDwfAnalogOutConfigure(self.hdwf, channel, c_bool(True))
            
            # Wait for completion
            time.sleep(self.pulse_width * 2)  # Wait twice the pulse width
            
            # Stop output
            self.dwf.FDwfAnalogOutConfigure(self.hdwf, channel, c_bool(False))
            # self.dwf.FDwfAnalogOutConfigure(self.hdwf, channel, DWFUser.OutputBehaviour.STOP.value)
            
            self.logger.debug("Single pulse fired")
            
            if self.on_pulse_complete:
                self.on_pulse_complete()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error firing single pulse: {e}")
            if self.on_error:
                self.on_error(str(e))
            return False

    def start_continuous(self, frequency: float) -> bool:
        """
        Start continuous laser pulsing.
        
        Args:
            frequency: Pulse repetition frequency in Hz
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            self.logger.error("Device not connected")
            return False
        
        if self.running:
            self.logger.warning("Continuous mode already running")
            return True
        
        try:
            self.repetition_rate = frequency
            channel = c_int(self.trigger_channel)
            
            # Set to pulse mode
            self.dwf.FDwfAnalogOutFunctionSet(self.hdwf, channel, funcPulse)
            
            # Set frequency
            self.dwf.FDwfAnalogOutFrequencySet(self.hdwf, channel, c_double(frequency))
            
            # Set for continuous operation (no run limit)
            self.dwf.FDwfAnalogOutRunSet(self.hdwf, channel, c_double(0))  # 0 = infinite
            
            # Start continuous output
            self.dwf.FDwfAnalogOutConfigure(self.hdwf, channel, c_bool(True))
            
            self.running = True
            self._stop_event.clear()
            
            # Start monitoring thread
            self._control_thread = threading.Thread(target=self._continuous_monitor, daemon=True)
            self._control_thread.start()
            
            self.logger.info(f"Started continuous mode at {frequency} Hz")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting continuous mode: {e}")
            if self.on_error:
                self.on_error(str(e))
            return False

    def stop_continuous(self) -> None:
        """Stop continuous laser pulsing."""
        if not self.running:
            return
        
        try:
            self._stop_event.set()
            
            # Stop output
            channel = c_int(self.trigger_channel)
            self.dwf.FDwfAnalogOutConfigure(self.hdwf, channel, c_bool(False))
            
            # Wait for thread to finish
            if self._control_thread and self._control_thread.is_alive():
                self._control_thread.join(timeout=1.0)
            
            self.running = False
            self.logger.info("Stopped continuous mode")
            
        except Exception as e:
            self.logger.error(f"Error stopping continuous mode: {e}")

    def fire_burst(self, pulse_count: int, frequency: float) -> bool:
        """
        Fire a burst of pulses.
        
        Args:
            pulse_count: Number of pulses to fire
            frequency: Pulse repetition frequency in Hz
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            self.logger.error("Device not connected")
            return False
        
        try:
            channel = c_int(self.trigger_channel)
            
            # Set to pulse mode
            self.dwf.FDwfAnalogOutFunctionSet(self.hdwf, channel, funcPulse)
            
            # Set frequency
            self.dwf.FDwfAnalogOutFrequencySet(self.hdwf, channel, c_double(frequency))
            
            # Calculate run time for the burst
            burst_duration = pulse_count / frequency
            self.dwf.FDwfAnalogOutRunSet(self.hdwf, channel, c_double(burst_duration))
            
            # Start burst
            self.dwf.FDwfAnalogOutConfigure(self.hdwf, channel, c_bool(True))
            
            # Wait for completion
            time.sleep(burst_duration + 0.1)  # Add small buffer
            
            # Stop output
            self.dwf.FDwfAnalogOutConfigure(self.hdwf, channel, c_bool(False))
            
            self.logger.info(f"Fired burst of {pulse_count} pulses at {frequency} Hz")
            return True
            
        except Exception as e:
            self.logger.error(f"Error firing burst: {e}")
            if self.on_error:
                self.on_error(str(e))
            return False

    def _continuous_monitor(self) -> None:
        """Monitor continuous operation in a separate thread."""
        while not self._stop_event.is_set():
            try:
                # Check device status
                if self.connected:
                    time.sleep(0.1)  # Check every 100ms
                else:
                    break
            except Exception as e:
                self.logger.error(f"Error in continuous monitor: {e}")
                break

    def get_status(self) -> dict:
        """
        Get current status of the Digilent interface.
        
        Returns:
            Dictionary with status information
        """
        return {
            'connected': self.connected,
            'running': self.running,
            'trigger_channel': self.trigger_channel,
            'pulse_amplitude': self.pulse_amplitude,
            'pulse_width': self.pulse_width,
            'repetition_rate': self.repetition_rate
        }

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        try:
            self.disconnect()
        except:
            pass  # Ignore errors during cleanup
