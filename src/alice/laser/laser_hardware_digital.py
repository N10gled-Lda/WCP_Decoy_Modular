"""Digital laser hardware driver using Digilent digital channels."""
from functools import wraps
import logging
import time
import threading
from typing import Optional, Dict, Any
from enum import Enum
from abc import ABC, abstractmethod

from .hardware_laser.digilent_digital_interface import DigilentDigitalInterface, DigitalTriggerMode



def ensure_connected(fn):
    """Decorator to ensure the laser hardware is connected before operation."""
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self.interface.connected:
            self.logger.error("Digilent device is not connected.")
            raise RuntimeError("Laser hardware must be connected first. Call initialize().")
        return fn(self, *args, **kwargs)
    return wrapper


class LaserState(Enum):
    """Laser operation states."""
    OFF = "off"
    READY = "ready"
    ON = "on"
    ERROR = "error"
    FIRING = "firing"


class BaseLaserDriver(ABC):
    """Hardware-independent laser driver interface."""

    def __init__(self):
        """Initialize base laser driver."""
        self.state = LaserState.OFF

    @abstractmethod
    def turn_on(self) -> bool:
        """Turn on the laser hardware."""
        pass

    @abstractmethod
    def turn_off(self) -> bool:
        """Turn off the laser hardware."""
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the laser hardware."""
        pass

    @abstractmethod
    def shutdown(self):
        """Shutdown the laser hardware."""
        pass

    def get_status(self) -> Dict[str, Any]:
        """Get basic status information."""
        return {
            "state": self.state.value,
            "timestamp": time.time()
        }


class LaserTriggerMode(Enum):
    """Laser trigger modes."""
    SINGLE = "single"
    TRAIN = "train"
    CONTINUOUS = "continuous"


class DigitalHardwareLaserDriver(BaseLaserDriver):
    """Hardware laser driver using Digilent digital triggering."""
    
    def __init__(self, device_index: int = -1, digital_channel: int = 8):
        """
        Initialize the digital hardware laser driver.
        
        Args:
            device_index: Index of the Digilent device (-1 for first available)
            digital_channel: Digital channel for triggering (default: 8)
        """
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Hardware interface
        self.interface = DigilentDigitalInterface(device_index, digital_channel)
        self.device_index = device_index
        self.digital_channel = digital_channel
        
        # Laser parameters
        self.pulse_width = 1e-6      # 1 microsecond default
        self.frequency = 1000.0      # 1 kHz default
        self.max_frequency = 50e6    # 50 MHz maximum
        self.min_pulse_width = 10e-9 # 10 ns minimum
        
        # State tracking
        self.trigger_count = 0
        self.last_trigger_time = 0.0
        self.continuous_mode = False
        self.idle_state = False  # Idle low 
        
        # Threading for background operations
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
        
        # # Set up callbacks
        # self.interface.on_connected = self._on_connected
        # self.interface.on_disconnected = self._on_disconnected
        # self.interface.on_error = self._on_error
        
        self.logger.info(f"DigitalHardwareLaserDriver initialized for channel {digital_channel}")
    
    # def _on_connected(self):
    #     """Callback for when the device is connected."""
    #     self.logger.info("Digilent device connected")
    #     self.state = LaserState.READY
    #     self.interface.set_pulse_parameters(
    #         width=self.pulse_width,
    #         frequency=self.frequency,
    #         idle_state=self.idle_state 
    #     )
    
    # def _on_disconnected(self):
    #     """Callback for when the device is disconnected."""
    #     self.logger.warning("Digilent device disconnected")
    #     self.state = LaserState.OFF

    # def _on_error(self, error: str):
    #     """Callback for when an error occurs."""
    #     self.logger.error(f"Digilent device error: {error}")
    #     self.state = LaserState.ERROR

    def initialize(self) -> bool:
        """Initialize the laser hardware."""
        try:
            self.logger.info("Initializing digital laser hardware...")
            
            # Connect to device
            if not self.interface.connect():
                self.logger.error("Failed to connect to Digilent device")
                return False
            
            # Set initial pulse parameters
            self.interface.set_pulse_parameters(
                width=self.pulse_width,
                frequency=self.frequency,
                idle_state=False  # Idle low for laser triggering
            )
            
            # Start monitoring thread
            self._start_monitoring()
            
            self.state = LaserState.READY
            self.logger.info("Digital laser hardware initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize laser hardware: {e}")
            self.state = LaserState.ERROR
            return False
    
    def shutdown(self):
        """Shutdown the laser hardware."""
        self.logger.info("Shutting down digital laser hardware...")
        
        try:
            # Stop any ongoing operations
            self.turn_off()
            self.stop_continuous()
            
            # Stop monitoring
            self._stop_monitoring.set()
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=2.0)
            
            # Close hardware interface
            self.interface.disconnect()
            
            self.state = LaserState.OFF
            self.logger.info("Digital laser hardware shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def turn_on(self) -> bool:
        """Turn on the laser (prepare for triggering)."""
        if self.state == LaserState.ERROR:
            self.logger.error("Cannot turn on laser in error state")
            return False
        
        try:
            # For digital triggering, "on" means ready to receive triggers
            # No actual output until trigger is received
            self.state = LaserState.ON
            self.logger.info("Laser turned on (ready for digital triggers)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to turn on laser: {e}")
            self.state = LaserState.ERROR
            return False
    
    def turn_off(self) -> bool:
        """Turn off the laser."""
        try:
            # Stop any continuous operation
            if self.continuous_mode:
                self.stop_continuous()
            
            # Stop any ongoing pulses
            self.interface.stop()
            
            self.state = LaserState.READY
            self.logger.info("Laser turned off")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to turn off laser: {e}")
            return False
    
    @ensure_connected
    def trigger_once(self) -> bool:
        """Send a single trigger pulse."""
        if self.state != LaserState.ON:
            self.logger.error(f"Cannot trigger laser in state {self.state}")
            return False
        
        try:
            success = self.interface.send_single_pulse()
            if success:
                self.trigger_count += 1
                self.last_trigger_time = time.time()
                self.logger.debug("Single trigger pulse sent")
            else:
                self.logger.error("Failed to send trigger pulse")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending trigger pulse: {e}")
            return False
    
    def send_frame(self, n_triggers: int, rep_rate_hz: float) -> bool:
        """
        Send a frame of multiple trigger pulses.
        
        Args:
            n_triggers: Number of trigger pulses
            rep_rate_hz: Repetition rate in Hz
        """
        if self.state != LaserState.ON:
            self.logger.error(f"Cannot send frame in state {self.state}")
            return False
        
        if rep_rate_hz > self.max_frequency:
            self.logger.error(f"Frequency {rep_rate_hz} Hz exceeds maximum {self.max_frequency} Hz")
            return False
        
        try:
            # Send pulse train
            success = self.interface.start_pulse_train(n_triggers, rep_rate_hz)
            if success:
                self.trigger_count += n_triggers
                self.last_trigger_time = time.time()
                
                # Wait for completion
                duration = n_triggers / rep_rate_hz
                time.sleep(duration + 0.01)  # Small buffer
                
                self.logger.info(f"Frame sent: {n_triggers} pulses at {rep_rate_hz} Hz")
            else:
                self.logger.error("Failed to send frame")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending frame: {e}")
            return False
    
    def start_continuous(self, rep_rate_hz: float) -> bool:
        """
        Start continuous trigger pulse generation.
        
        Args:
            rep_rate_hz: Repetition rate in Hz
        """
        if self.state != LaserState.ON:
            self.logger.error(f"Cannot start continuous in state {self.state}")
            return False
        
        if rep_rate_hz > self.max_frequency:
            self.logger.error(f"Frequency {rep_rate_hz} Hz exceeds maximum {self.max_frequency} Hz")
            return False
        
        try:
            success = self.interface.start_continuous(rep_rate_hz)
            if success:
                self.continuous_mode = True
                self.frequency = rep_rate_hz
                self.last_trigger_time = time.time()
                self.logger.info(f"Continuous mode started at {rep_rate_hz} Hz")
            else:
                self.logger.error("Failed to start continuous mode")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error starting continuous mode: {e}")
            return False
    
    def stop_continuous(self) -> bool:
        """Stop continuous trigger pulse generation."""
        try:
            success = self.interface.stop()
            if success:
                self.continuous_mode = False
                self.logger.info("Continuous mode stopped")
            else:
                self.logger.error("Failed to stop continuous mode")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error stopping continuous mode: {e}")
            return False
    
    def set_pulse_parameters(self, width: float, frequency: float = None):
        """
        Set pulse parameters.
        
        Args:
            width: Pulse width in seconds
            frequency: Optional frequency in Hz (updates default)
        """
        # Validate parameters
        width = max(self.min_pulse_width, min(width, 1e-3))  # 10ns to 1ms
        
        if frequency is not None:
            frequency = max(0.1, min(frequency, self.max_frequency))
            self.frequency = frequency
        
        self.pulse_width = width
        
        try:
            self.interface.set_pulse_parameters(
                width=self.pulse_width,
                frequency=self.frequency,
                idle_state=self.idle_state
            )

            self.logger.info(f"Pulse parameters updated: width={self.pulse_width*1e6:.1f}μs, freq={self.frequency:.1f}Hz, idle_state={'high' if self.idle_state else 'low'}")

        except Exception as e:
            self.logger.error(f"Failed to set pulse parameters: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status information."""
        base_status = super().get_status()
        
        # Get interface status
        interface_status = self.interface.get_status()
        
        # Combine status information
        status = {
            **base_status,
            "hardware_type": "digital_digilent",
            "device_index": self.device_index,
            "digital_channel": self.digital_channel,
            "pulse_width_us": self.pulse_width * 1e6,
            "frequency_hz": self.frequency,
            "max_frequency_hz": self.max_frequency,
            "trigger_count": self.trigger_count,
            "last_trigger_time": self.last_trigger_time,
            "continuous_mode": self.continuous_mode,
            "interface_status": interface_status
        }
        
        return status
    
    
    def _start_monitoring(self):
        """Start background monitoring thread."""
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.debug("Monitoring thread started")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self._stop_monitoring.wait(1.0):  # Check every second
            try:
                # Check interface status
                status = self.interface.get_status()
                
                if not status.get("connected", False):
                    self.state = LaserState.ERROR
                    self.logger.error("Device connection lost")
                    break
                
                # Update counters from interface
                if "pulse_count" in status:
                    self.trigger_count = status["pulse_count"]
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                break
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


def create_digital_laser_driver(device_index: int = -1, digital_channel: int = 8) -> DigitalHardwareLaserDriver:
    """
    Factory function to create a digital laser driver.
    
    Args:
        device_index: Digilent device index (-1 for first available)
        digital_channel: Digital channel for triggering (default: 8)

    Returns:
        Configured DigitalHardwareLaserDriver instance
    """
    driver = DigitalHardwareLaserDriver(device_index, digital_channel)
    return driver


if __name__ == "__main__":
    """Test the digital hardware driver."""
    logging.basicConfig(level=logging.INFO)
    
    print("Testing DigitalHardwareLaserDriver...")
    
    # Test with context manager
    with create_digital_laser_driver(digital_channel=8) as laser:
        if laser.state == LaserState.ERROR or not laser.interface.connected:
            print("❌ Failed to initialize laser driver")
        else:
            print("✅ Laser driver initialized")
            
            # Turn on laser
            laser.turn_on()
            
            # Test single trigger
            print("🔸 Testing single trigger...")
            laser.trigger_once()
            time.sleep(0.1)
            
            # Test frame
            print("🔸 Testing frame (5 triggers at 1kHz)...")
            laser.send_frame(5, 1000.0)
            time.sleep(0.1)
            
            # Test continuous mode briefly
            print("🔸 Testing continuous mode...")
            laser.start_continuous(500.0)
            time.sleep(1.0)
            laser.stop_continuous()
            
            # Show final status
            status = laser.get_status()
            print(f"📊 Final status: trigger_count={status['trigger_count']}")
            
            print("✅ Digital laser driver test completed")
