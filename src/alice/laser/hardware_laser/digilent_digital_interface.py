"""
Digilent Digital Interface for Laser Control using WaveForms SDK.

This module provides a simplified interface for controlling lasers via digital
trigger signals using Digilent devices. The interface supports three core
trigger modes that cover all practical laser control scenarios:

- SINGLE: Send one trigger pulse (rising edge) for single laser pulse
- TRAIN: Send N trigger pulses at specified frequency for pulse trains  
- CONTINUOUS: Send continuous trigger pulses for continuous laser operation

Since laser triggering is typically edge-triggered, these three modes provide
complete coverage of laser control requirements.
"""
import logging
import math
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
# try:
from .dwfconstants import *
# except ImportError:
#     # Fallback constants for digital I/O
#     class DwfDigitalOut:
#         IDLE_INIT = 0
#         IDLE_LOW = 1
#         IDLE_HIGH = 2
#         IDLE_ZET = 3
    
#     class DwfState:
#         READY = 0
#         ARMED = 1
#         WAIT = 2
#         TRIGGERED = 3
#         RUNNING = 4
#         DONE = 5


class DigitalTriggerMode(Enum):
    """Digital trigger modes for laser control."""
    SINGLE = "single"
    TRAIN = "train"
    CONTINUOUS = "continuous"


class DigilentDigitalInterface:
    """Interface for controlling laser through Digilent device using digital channels.
    Hardware emits rising edges; higher-level 'modes' are just scheduling strategies.
    This class provides methods to:
    - Connect / disconnect from the device
    - Set pulse parameters (width, frequency, idle state)
    - Trigger laser in SINGLE, TRAIN, or CONTINUOUS modes:
      - send_single_pulse() for one pulse
      - start_pulse_train(n_pulses, frequency) for multiple pulses at specified frequency
      - start_continuous(frequency) for continuous pulses at specified frequency
    - Stop pulse generation
    - Get current status (connected, running, pulse parameters)
    - Monitor status and handle callbacks for events
    """
        
    def __init__(self, device_index: int = -1, digital_channel: int = 8):
        """
        Initialize the Digilent digital interface.
        
        Args:
            device_index: Index of the Digilent device to use (-1 for first available)
            digital_channel: Digital channel to use for triggering (default: 8)
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Device management
        self.dwf = dwf
        self.hdwf = c_int()
        self.device_index = device_index
        self.digital_channel = digital_channel
        self.connected = False
        self.running = False
        
        # Digital pulse parameters
        self.duty_cycle = 0.5  # 50% duty cycle default
        self.frequency = 1000.0  # 1 kHz default frequency
        self.idle_state = False  # Idle low
        
        # Threading and callbacks
        self._callback = None
        self._thread = None
        self._stop_event = threading.Event()
        
        # Status tracking
        self.pulse_count = 0
        self.last_pulse_time = 0.0
        self.error_count = 0
        
        # Callbacks
        self.on_connected: Optional[Callable] = None
        self.on_disconnected: Optional[Callable] = None
        self.on_pulse_complete: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

        self.logger.info(f"DigilentDigitalInterface initialized for channel {digital_channel}")
    
    def connect(self) -> bool:
        """
        Connect to the Digilent device.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """

        if not self.dwf:
            self.logger.error("WaveForms SDK not available")
            return False
        
        if self.connected:
            self.logger.warning("Already connected to a Digilent device")
            return True
        
        try:
            # Enumerate devices
            cdevices = c_int()
            self.dwf.FDwfEnum(c_int(0), byref(cdevices))
            
            if cdevices.value == 0:
                self.logger.error("No Digilent devices found")
                return False
            
            self.logger.info(f"Found {cdevices.value} Digilent device(s)")
            
            # Open device
            if self.device_index == -1:
                # Open first available device
                self.dwf.FDwfDeviceOpen(DWFUser.Devices.FIRST.value, byref(self.hdwf))
            elif self.device_index >= 0:
                self.dwf.FDwfDeviceOpen(c_int(self.device_index), byref(self.hdwf))
            else:
                self.logger.error(f"Invalid device index: {self.device_index}")
                return False
            if self.hdwf.value == hdwfNone.value:
                error_msg = create_string_buffer(512)
                self.dwf.FDwfGetLastErrorMsg(error_msg)
                self.logger.error(f"Failed to open device: {error_msg.value.decode()}")
                return False
            
            # Configure digital I/O
            if not self._configure_digital_io():
                self.disconnect()
                return False
            
            self.connected = True
            self.logger.info(f"Connected to Digilent device, using digital channel {self.digital_channel}")
            
            # Trigger on connected callback
            if self.on_connected:
                self.on_connected()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to device: {e}")
            return False
    
    def _configure_digital_io(self) -> bool:
        """Configure the digital I/O settings."""
        try:
            self.logger.debug("Configuring digital I/O...")
            # Enable digital out
            if self.dwf.FDwfDigitalOutEnableSet(self.hdwf, c_int(self.digital_channel), c_int(1)) != 1:
                self.logger.error("Failed to enable digital output")
                return False
            
            # # Set Type to Pulse
            # if self.dwf.FDwfDigitalOutTypeSet(self.hdwf, c_int(self.digital_channel), DwfDigitalOutTypePulse) != 1:
            #     self.logger.error("Failed to set digital output type to Pulse")
            #     return False
            
            # Set idle state (low by default)
            # FDwfDigitalOutIdleSet(HDWF hdwf, int idxChannel, DwfDigitalOutIdle v)
            # idle_value = DwfDigitalOutIdleHigh if self.idle_state else DwfDigitalOutIdleLow
            # if self.dwf.FDwfDigitalOutIdleSet(self.hdwf, c_int(self.digital_channel), idle_value) != 1:
            #     self.logger.error("Failed to set idle state")
            #     return False
            
            # Set initial pulse parameters
            self._update_pulse_parameters()
            
            self.logger.info("Digital I/O configured successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error configuring digital I/O: {e}")
            return False
    
    def _update_pulse_parameters(self):
        try:
            # internal sample-clock (100 MHz) ÷ divider → sample_rate

            internal_clock = c_double()
            max_count = c_uint()
            if self.dwf.FDwfDigitalOutInternalClockInfo(self.hdwf, byref(internal_clock)) != 1:
                self.logger.error("Failed to get internal clock info - defaulting to 100 MHz")
                internal_clock.value = 100e6
            else:
                self.logger.debug(f"Internal clock frequency: {internal_clock.value} Hz")
            if self.dwf.FDwfDigitalOutCounterInfo(self.hdwf, c_int(self.digital_channel), 0, byref(max_count)) != 1:
                self.logger.error("Failed to get counter info - defaulting to 32768")
                max_count.value = 32768
            else:
                self.logger.debug(f"Max counter value: {max_count.value}")

            divider = int(math.ceil(internal_clock.value / self.frequency / max_count.value))  # 2 for pulse width
            self.dwf.FDwfDigitalOutDividerSet(self.hdwf, c_int(self.digital_channel), c_int(divider))

            print(f"Divider set to: {divider} (internal clock {internal_clock.value} Hz / frequency {self.frequency} Hz / max count {max_count.value})")
            
            # Calculate low/high ticks based on duty cycle
            pulse_ticks = int(math.ceil(internal_clock.value / self.frequency / divider))
            duty_cycle = max(0.0001, min(self.duty_cycle, 0.99))  # Clamp to 0.01%-99%
            print(f"Duty cycle: {duty_cycle*100:.1f}% of pulse are high")
            
            if duty_cycle > 0.9:
                self.logger.warning(f"Duty cycle {duty_cycle*100:.1f}% is quite high, consider reducing for better edge triggering")
            if duty_cycle < 0.0001:
                self.logger.warning(f"Duty cycle {duty_cycle*100:.1f}% is quite low, consider increasing for better edge triggering")   

            high_ticks = int(round(pulse_ticks * duty_cycle))
            low_ticks = pulse_ticks - high_ticks
            t_ticks = 1 / (internal_clock.value / divider)  # Time per tick

            print(f"Counter Set Pulse parameters: duty_cycle={duty_cycle*100:.1f}%, "
                  f"freq={self.frequency:.1f}Hz, "
                  f"pulse={pulse_ticks} clk, "
                  f"low_ticks={low_ticks}, high_ticks={high_ticks}"
                  f" (t_tick={t_ticks:.1e}s)")
            
            # API expects low then high counts typically (depending on version). We'll assume (low, high)
            # low then high
            fpol = 0  # 0 = low, 1 = high
            self.dwf.FDwfDigitalOutCounterInitSet(self.hdwf, c_int(self.digital_channel), c_int(fpol), c_int(0)) 
            if self.dwf.FDwfDigitalOutCounterSet(
                    self.hdwf,
                    c_int(self.digital_channel),
                    c_int(low_ticks),
                    c_int(high_ticks)) != 1:
                self.logger.warning("Failed to set counter values")
            else:
                self.logger.info(f"Counter set: low={low_ticks}, high={high_ticks} (pulse {pulse_ticks} clk)")

            self.logger.debug(
                f"Updated pulse parameters: duty_cycle={self.duty_cycle*100:.1f}% "
                f"({high_ticks} clk), freq={self.frequency:.1f}Hz "
                f"(pulse {pulse_ticks} clk)")
        except Exception as e:
            self.logger.error(f"Error updating pulse parameters: {e}")
    
    def set_pulse_parameters(self, duty_cycle: float, frequency: float, idle_state: bool = False):
        """
        Set pulse parameters.
        
        Args:
            duty_cycle: Duty cycle as fraction (0.0-1.0, e.g., 0.5 = 50%)
            frequency: Pulse frequency in Hz
            idle_state: Idle state (True=high, False=low)
            [idle state is the resting logic-level of the digital line when you're not outputting a pulse]
        """
        self.duty_cycle = max(0.01, min(duty_cycle, 0.99))  # Clamp between 1% and 99%
        self.frequency = max(0.1, min(frequency, 50e6))  # Clamp between 0.1Hz and 50MHz
        self.idle_state = idle_state
        
        if self.connected:
            self._update_pulse_parameters()
        
        self.logger.info(f"Pulse parameters set: duty_cycle={self.duty_cycle*100:.1f}%, "
                        f"freq={self.frequency:.1f}Hz, idle={'HIGH' if idle_state else 'LOW'}")
    
    def trigger_laser(self, mode: str = "single", count: int = 1, frequency: float = None) -> bool:
        """
        Trigger the laser with specified parameters.
        
        Args:
            mode: "single", "train", or "continuous"
            count: Number of pulses (ignored for continuous mode)
            frequency: Pulse frequency in Hz (optional, uses current setting if None)
            
        Returns:
            bool: True if trigger was successful, False otherwise
        """
        if not self.connected:
            self.logger.error("Device not connected")
            return False
        
        # Validate mode
        valid_modes = [m.value for m in DigitalTriggerMode]
        if mode not in valid_modes:
            self.logger.error(f"Invalid mode '{mode}'. Valid modes: {valid_modes}")
            return False
        
        # Execute the appropriate trigger method
        if mode == DigitalTriggerMode.SINGLE.value:
            return self.send_single_pulse()
        elif mode == DigitalTriggerMode.TRAIN.value:
            return self.start_pulse_train(count, frequency)
        elif mode == DigitalTriggerMode.CONTINUOUS.value:
            return self.start_continuous(frequency)
        
        return False
    
    def send_single_pulse(self) -> bool:
        """Send a single pulse."""
        if not self.connected:
            self.logger.error("Device not connected")
            return False
        
        try:
            print(f"frequency: {self.frequency:.1f} Hz, duty_cycle: {self.duty_cycle*100:.1f}%")
            # Configure for single pulse
            if self.dwf.FDwfDigitalOutRepeatSet(self.hdwf, c_int(1)) != 1:
                self.logger.error("Failed to set single pulse mode")
                return False
            nb_pulses = 1  # Single pulse
            print(f"Running for {nb_pulses/self.frequency:.5f} s")
            if self.dwf.FDwfDigitalOutRunSet(self.hdwf, c_double(nb_pulses / self.frequency)) != 1:
                self.logger.error("Failed to set pulse run time")
                return False

            # Start the pulse
            if self.dwf.FDwfDigitalOutConfigure(self.hdwf, DWFUser.OutputBehaviour.START.value) != 1:
                self.logger.error("Failed to start pulse")
                return False
            
            print("Here1")
            # Wait for completion
            self._wait_for_completion()
            
            # Stop configuration
            if self.dwf.FDwfDigitalOutConfigure(self.hdwf, c_int(0)) != 1:
                self.logger.error("Failed to stop digital output")
                return False
            
            print("Here2")
            self.pulse_count += 1
            self.last_pulse_time = time.time()
            
            self.logger.debug("Single pulse sent")

            # Trigger on pulse complete callback
            if self.on_pulse_complete:
                self.on_pulse_complete()

            return True
            
        except Exception as e:
            self.logger.error(f"Error sending single pulse: {e}")
            self.error_count += 1

            if self.on_error:
                self.on_error(str(e))

            return False
    
    def start_pulse_train(self, n_pulses: int, frequency: float = None) -> bool:
        """
        Start a pulse train with specified number of pulses.
        
        Args:
            n_pulses: Number of pulses to send
            frequency: Pulse frequency (if different from current setting)
        """
        if not self.connected:
            self.logger.error("Device not connected")
            return False
        
        try:
            # Update frequency if specified
            if frequency is not None:
                self.set_pulse_parameters(self.duty_cycle, frequency, self.idle_state)
            
            print(f"Starting pulse train: {n_pulses} pulses at {self.frequency:.1f} Hz")
            # Configure for pulse train
            if self.dwf.FDwfDigitalOutRepeatSet(self.hdwf, c_int(1)) != 1:
                self.logger.error("Failed to set pulse train mode")
                return False
            
            nb_pulses = n_pulses # Number of pulses to send
            print(f"Running for {nb_pulses/self.frequency:.5f} s")
            if self.dwf.FDwfDigitalOutRunSet(self.hdwf, c_double(nb_pulses / self.frequency)) != 1:
                self.logger.error("Failed to set pulse run time")
                return False

            # Start the pulse
            if self.dwf.FDwfDigitalOutConfigure(self.hdwf, DWFUser.OutputBehaviour.START.value) != 1:
                self.logger.error("Failed to start pulse")
                return False
            
            print("Here1")
            # Wait for completion
            self._wait_for_completion()
            
            # Stop configuration
            if self.dwf.FDwfDigitalOutConfigure(self.hdwf, c_int(0)) != 1:
                self.logger.error("Failed to stop digital output")
                return False
            
            print("Here2")
            
            self.running = True
            self.pulse_count += n_pulses
            self.last_pulse_time = time.time()
            
            self.logger.info(f"Started pulse train: {n_pulses} pulses at {self.frequency:.1f} Hz")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting pulse train: {e}")
            self.error_count += 1

            if self.on_error:
                self.on_error(str(e))

            return False
    
    def start_continuous(self, frequency: float = None) -> bool:
        """
        Start continuous pulse generation.
        
        Args:
            frequency: Pulse frequency (if different from current setting)
        """
        if not self.connected:
            self.logger.error("Device not connected")
            return False
        
        try:
            # Update frequency if specified
            if frequency is not None:
                self.set_pulse_parameters(self.duty_cycle, frequency, self.idle_state)
            
            # Configure for continuous mode
            if self.dwf.FDwfDigitalOutRepeatSet(self.hdwf, c_int(0)) != 1:
                self.logger.error("Failed to set continuous mode") # 0 means infinite repeats
                return False
             
            # Start continuous pulses
            if self.dwf.FDwfDigitalOutConfigure(self.hdwf, c_int(1)) != 1:
                self.logger.error("Failed to start continuous pulses")
                return False
            
            self.running = True
            self.last_pulse_time = time.time()
            
            self.logger.info(f"Started continuous pulses at {self.frequency:.1f} Hz")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting continuous pulses: {e}")
            self.error_count += 1

            if self.on_error:
                self.on_error(str(e))

            return False
    
    def stop(self) -> bool:
        """Stop pulse generation."""
        if not self.connected:
            return True
        
        try:
            # Stop digital output
            if self.dwf.FDwfDigitalOutConfigure(self.hdwf, c_int(0)) != 1:
                self.logger.error("Failed to stop digital output")
                return False
            
            self.running = False
            self.logger.info("Pulse generation stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping pulse generation: {e}")
            return False
    
    def _wait_for_completion(self, timeout: float = 5.0):
        """Wait for pulse generation to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            sts = c_int()
            try:
                if self.dwf.FDwfDigitalOutStatus(self.hdwf, byref(sts)) == 1:
                    print(sts.value)
                    if sts.value == DwfStateDone.value:
                        break
                time.sleep(0.0001)  # 100us polling
            except:
                break
    
    def get_status(self) -> dict:
        """Get current status of the digital interface."""
        status = {
            "connected": self.connected,
            "running": self.running,
            "channel": self.digital_channel,
            "duty_cycle_percent": self.duty_cycle * 100,
            "frequency_hz": self.frequency,
            "idle_state": "HIGH" if self.idle_state else "LOW",
            "pulse_count": self.pulse_count,
            "error_count": self.error_count,
            "last_pulse_time": self.last_pulse_time
        }
        
        if self.connected:
            try:
                # Get device status
                sts = c_int()
                if self.dwf.FDwfDigitalOutStatus(self.hdwf, byref(sts)) == 1:
                    status["device_state"] = sts.value
            except:
                status["device_state"] = "unknown"
        
        return status
    
    def set_callback(self, callback: Callable[[dict], None]):
        """Set callback function for status updates."""
        self._callback = callback
        self.logger.info("Callback function set")
    
    def disconnect(self):
        """Close the connection to the device."""
        if self.running:
            self.stop()
        
        if self.connected and self.hdwf.value != 0:
            try:
                # Reset the device
                self.dwf.FDwfDigitalOutReset(self.hdwf)
                self.logger.info("Device reset")
                self.dwf.FDwfDigitalOutConfigure(self.hdwf, DWFUser.OutputBehaviour.STOP.value)
                self.logger.info("Device configuration reset")
                # Close the device
                self.dwf.FDwfDeviceClose(self.hdwf)
                self.logger.info("Device connection closed")
                
                self.connected = False
                # Trigger on disconnected callback
                if self.on_disconnected:
                    self.on_disconnected()
            except:
                pass
        
        self.hdwf = c_int()
    
    def __enter__(self):
        """Context manager entry."""
        if not self.connected:
            self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def __del__(self):
        """Destructor."""
        self.disconnect()


# Utility functions for digital interface
def list_digital_devices() -> List[dict]:
    """List available Digilent devices with digital capabilities."""
    if not dwf:
        return []
    
    devices = []
    try:
        cdevices = c_int()
        dwf.FDwfEnum(c_int(0), byref(cdevices))
        
        for i in range(cdevices.value):
            device_info = {
                "index": i,
                "name": "Unknown",
                "serial": "Unknown",
                "digital_channels": 16  # Most Digilent devices have 16 digital I/O
            }
            
            # Try to get device name
            try:
                szDeviceName = create_string_buffer(64)
                szSN = create_string_buffer(16)
                if dwf.FDwfEnumDeviceName(c_int(i), szDeviceName) == 1:
                    device_info["name"] = szDeviceName.value.decode('ascii')
                if dwf.FDwfEnumSN(c_int(i), szSN) == 1:
                    device_info["serial"] = szSN.value.decode('ascii')
            except:
                pass
            
            devices.append(device_info)
    
    except Exception as e:
        print(f"Error listing devices: {e}")
    
    return devices


def test_digital_interface(channel: int = 8, n_pulses: int = 5):
    """Test the digital interface with a simple pulse sequence."""
    print(f"Testing DigilentDigitalInterface on channel {channel}")
    
    with DigilentDigitalInterface(digital_channel=channel) as interface:
        if not interface.connected:
            print("❌ Failed to connect to device")
            return False
        
        print("✅ Connected to device")
        
        # Set pulse parameters
        interface.set_pulse_parameters(duty_cycle=0.2, frequency=1000.0)  # 20% duty cycle at 1kHz
        
        # Test single pulse
        print("🔸 Testing single pulse...")
        if interface.send_single_pulse():
            print("✅ Single pulse sent")
        else:
            print("❌ Single pulse failed")
        
        time.sleep(0.1)
        
        # Test pulse train
        print(f"🔸 Testing pulse train ({n_pulses} pulses)...")
        if interface.start_pulse_train(n_pulses, 2000.0):  # 2kHz
            time.sleep(n_pulses / 2000.0 + 0.1)  # Wait for completion
            print("✅ Pulse train completed")
        else:
            print("❌ Pulse train failed")
        
        # Show status
        status = interface.get_status()
        print(f"📊 Final status: {status}")
        
        return True


if __name__ == "__main__":
    # List available devices
    devices = list_digital_devices()
    print("Available Digilent devices:")
    for dev in devices:
        print(f"  {dev['index']}: {dev['name']} (SN: {dev['serial']})")
    
    if devices:
        # Test the interface
        test_digital_interface()
    else:
        print("No Digilent devices found for testing")
