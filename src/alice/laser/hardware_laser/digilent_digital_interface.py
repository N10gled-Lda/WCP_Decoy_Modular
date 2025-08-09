"""Digilent Digital Interface for Laser Control using WaveForms SDK."""
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
try:
    from .dwfconstants import *
except ImportError:
    # Fallback constants for digital I/O
    class DwfDigitalOut:
        IDLE_INIT = 0
        IDLE_LOW = 1
        IDLE_HIGH = 2
        IDLE_ZET = 3
    
    class DwfState:
        READY = 0
        ARMED = 1
        WAIT = 2
        TRIGGERED = 3
        RUNNING = 4
        DONE = 5


class DigitalTriggerMode(Enum):
    """Digital trigger modes for laser control."""
    SINGLE_PULSE = "single_pulse"
    PULSE_TRAIN = "pulse_train"
    CONTINUOUS = "continuous"
    BURST = "burst"


class DigilentDigitalInterface:
    """Interface for controlling laser through Digilent device using digital channels."""
    
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
        self.pulse_width = 1e-6  # 1 microsecond default pulse width
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
        
        self.logger.info(f"DigilentDigitalInterface initialized for channel {digital_channel}")
    
    def connect(self) -> bool:
        """Connect to the Digilent device."""
        if not self.dwf:
            self.logger.error("WaveForms SDK not available")
            return False
        
        try:
            # Enumerate devices
            cdevices = c_int()
            self.dwf.FDwfEnum(c_int(0), byref(cdevices))
            
            if cdevices.value == 0:
                self.logger.error("No Digilent devices found")
                return False
            
            self.logger.info(f"Found {cdevices.value} Digilent device(s)")
            
            # Open device
            device_idx = self.device_index if self.device_index >= 0 else 0
            if self.dwf.FDwfDeviceOpen(c_int(device_idx), byref(self.hdwf)) != 1:
                self.logger.error(f"Failed to open device {device_idx}")
                return False
            
            # Configure digital I/O
            if not self._configure_digital_io():
                self.close()
                return False
            
            self.connected = True
            self.logger.info(f"Connected to Digilent device, using digital channel {self.digital_channel}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to device: {e}")
            return False
    
    def _configure_digital_io(self) -> bool:
        """Configure the digital I/O settings."""
        try:
            # Enable digital out
            if self.dwf.FDwfDigitalOutEnableSet(self.hdwf, c_int(self.digital_channel), c_int(1)) != 1:
                self.logger.error("Failed to enable digital output")
                return False
            
            # Set idle state (low by default)
            idle_value = DwfDigitalOut.IDLE_HIGH if self.idle_state else DwfDigitalOut.IDLE_LOW
            if self.dwf.FDwfDigitalOutIdleSet(self.hdwf, c_int(self.digital_channel), c_int(idle_value)) != 1:
                self.logger.error("Failed to set idle state")
                return False
            
            # Set initial pulse parameters
            self._update_pulse_parameters()
            
            self.logger.info("Digital I/O configured successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error configuring digital I/O: {e}")
            return False
    
    def _update_pulse_parameters(self):
        """Update pulse parameters on the device."""
        try:
            # Set frequency (period)
            period_seconds = 1.0 / self.frequency
            if self.dwf.FDwfDigitalOutDividerSet(self.hdwf, c_int(self.digital_channel), c_int(1)) != 1:
                self.logger.warning("Failed to set divider")
            
            # Calculate counter values for the pulse
            # Assuming 100 MHz internal clock
            internal_freq = 100e6
            counter_max = int(period_seconds * internal_freq)
            pulse_counter = int(self.pulse_width * internal_freq)
            
            # Set counter values
            if self.dwf.FDwfDigitalOutCounterSet(self.hdwf, c_int(self.digital_channel), 
                                               c_int(pulse_counter), c_int(counter_max)) != 1:
                self.logger.warning("Failed to set counter values")
            
            self.logger.debug(f"Updated pulse parameters: width={self.pulse_width*1e6:.1f}μs, freq={self.frequency:.1f}Hz")
            
        except Exception as e:
            self.logger.error(f"Error updating pulse parameters: {e}")
    
    def set_pulse_parameters(self, width: float, frequency: float, idle_state: bool = False):
        """
        Set pulse parameters.
        
        Args:
            width: Pulse width in seconds
            frequency: Pulse frequency in Hz
            idle_state: Idle state (True=high, False=low)
        """
        self.pulse_width = max(1e-9, min(width, 1.0))  # Clamp between 1ns and 1s
        self.frequency = max(0.1, min(frequency, 50e6))  # Clamp between 0.1Hz and 50MHz
        self.idle_state = idle_state
        
        if self.connected:
            self._update_pulse_parameters()
        
        self.logger.info(f"Pulse parameters set: width={self.pulse_width*1e6:.1f}μs, "
                        f"freq={self.frequency:.1f}Hz, idle={'HIGH' if idle_state else 'LOW'}")
    
    def send_single_pulse(self) -> bool:
        """Send a single pulse."""
        if not self.connected:
            self.logger.error("Device not connected")
            return False
        
        try:
            # Configure for single pulse
            if self.dwf.FDwfDigitalOutRepeatSet(self.hdwf, c_int(self.digital_channel), c_int(1)) != 1:
                self.logger.error("Failed to set single pulse mode")
                return False
            
            # Start the pulse
            if self.dwf.FDwfDigitalOutConfigure(self.hdwf, c_int(1)) != 1:
                self.logger.error("Failed to start pulse")
                return False
            
            # Wait for completion
            self._wait_for_completion()
            
            self.pulse_count += 1
            self.last_pulse_time = time.time()
            
            self.logger.debug("Single pulse sent")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending single pulse: {e}")
            self.error_count += 1
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
                old_freq = self.frequency
                self.set_pulse_parameters(self.pulse_width, frequency, self.idle_state)
            
            # Configure for pulse train
            if self.dwf.FDwfDigitalOutRepeatSet(self.hdwf, c_int(self.digital_channel), c_int(n_pulses)) != 1:
                self.logger.error("Failed to set pulse train mode")
                return False
            
            # Start the pulse train
            if self.dwf.FDwfDigitalOutConfigure(self.hdwf, c_int(1)) != 1:
                self.logger.error("Failed to start pulse train")
                return False
            
            self.running = True
            self.pulse_count += n_pulses
            self.last_pulse_time = time.time()
            
            self.logger.info(f"Started pulse train: {n_pulses} pulses at {self.frequency:.1f} Hz")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting pulse train: {e}")
            self.error_count += 1
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
                self.set_pulse_parameters(self.pulse_width, frequency, self.idle_state)
            
            # Configure for continuous mode
            if self.dwf.FDwfDigitalOutRepeatSet(self.hdwf, c_int(self.digital_channel), c_int(0)) != 1:
                self.logger.error("Failed to set continuous mode")
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
            try:
                sts = c_int()
                if self.dwf.FDwfDigitalOutStatus(self.hdwf, byref(sts)) == 1:
                    if sts.value == DwfState.DONE:
                        break
                time.sleep(0.001)  # 1ms polling
            except:
                break
    
    def get_status(self) -> dict:
        """Get current status of the digital interface."""
        status = {
            "connected": self.connected,
            "running": self.running,
            "channel": self.digital_channel,
            "pulse_width_us": self.pulse_width * 1e6,
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
    
    def close(self):
        """Close the connection to the device."""
        if self.running:
            self.stop()
        
        if self.connected and self.hdwf.value != 0:
            try:
                self.dwf.FDwfDeviceClose(self.hdwf)
                self.logger.info("Device connection closed")
            except:
                pass
        
        self.connected = False
        self.hdwf = c_int()
    
    def __enter__(self):
        """Context manager entry."""
        if not self.connected:
            self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor."""
        self.close()


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
        interface.set_pulse_parameters(width=2e-6, frequency=1000.0)  # 2μs pulses at 1kHz
        
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
