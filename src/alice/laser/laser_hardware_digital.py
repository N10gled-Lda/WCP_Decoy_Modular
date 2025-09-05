"""Digital laser hardware driver using Digilent digital channels."""
from functools import wraps
import logging
import time
import threading
from typing import Optional, Dict, Any
from enum import Enum
from abc import ABC, abstractmethod

from src.alice.laser.hardware_laser.digilent_digital_interface import DigilentDigitalInterface, DigitalTriggerMode
from src.alice.laser.laser_base import BaseLaserDriver


def ensure_connected(fn):
    """Decorator to ensure the laser hardware is connected before operation."""
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self.interface.connected:
            self.logger.error("Digilent device is not connected.")
            self.state = LaserState.ERROR
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
        self.state = LaserState.OFF
        
        # Laser parameters
        self.duty_cycle = 0.1        # 10% duty cycle default (good for triggering)
        self.frequency = 1000.0      # 1 kHz default
        self.max_frequency = 50e6    # 50 MHz maximum
        self.min_duty_cycle = 0.01   # 1% minimum duty cycle
        
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
        """Initialize the laser hardware and make it ready to emit."""
        try:
            self.logger.info("Initializing digital laser hardware...")
            
            # Connect to device
            if not self.interface.connect():
                self.logger.error("Failed to connect to Digilent device")
                return False
            
            # Set initial pulse parameters
            self.interface.set_pulse_parameters(
                duty_cycle=self.duty_cycle,
                frequency=self.frequency,
                idle_state=False  # Idle low for laser triggering
            )
            
            # Start monitoring thread
            self._start_monitoring()
            
            self.state = LaserState.ON  # Ready to emit after initialization
            self.logger.info("Digital laser hardware initialized and ready")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize laser hardware: {e}")
            self.state = LaserState.ERROR
            return False
    
    def shutdown(self) -> bool:
        """Shutdown the laser hardware."""
        self.logger.info("Shutting down digital laser hardware...")
        
        try:
            # Stop any ongoing operations
            self.stop_continuous()
            
            # Stop monitoring
            self._stop_monitoring.set()
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=2.0)
            
            # Close hardware interface
            self.interface.disconnect()
            
            self.state = LaserState.OFF
            self.logger.info("Digital laser hardware shutdown complete")
            return True
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
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
    
    @ensure_connected
    def send_frame(self, n_triggers: int, rep_rate_hz: float = None) -> bool:
        """
        Send a frame of multiple trigger pulses.
        
        Args:
            n_triggers: Number of trigger pulses
            rep_rate_hz: Repetition rate in Hz
        """
        if self.state != LaserState.ON:
            self.logger.error(f"Cannot send frame in state {self.state}")
            return False
        
        if rep_rate_hz is None: rep_rate_hz = self.frequency
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
    
    @ensure_connected
    def start_continuous(self, rep_rate_hz: float = None) -> bool:
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
    
    def set_pulse_parameters(self, duty_cycle: float, frequency: float = None):
        """
        Set pulse parameters.
        
        Args:
            duty_cycle: Duty cycle as fraction (0.0-1.0, e.g., 0.1 = 10%)
            frequency: Optional frequency in Hz (updates default)
        """
        # Validate parameters
        duty_cycle = max(self.min_duty_cycle, min(duty_cycle, 0.99))  # 1% to 99%
        
        if frequency is not None:
            frequency = max(0.1, min(frequency, self.max_frequency))
            self.frequency = frequency
        
        self.duty_cycle = duty_cycle
        
        try:
            self.interface.set_pulse_parameters(
                duty_cycle=self.duty_cycle,
                frequency=self.frequency,
                idle_state=self.idle_state
            )

            self.logger.info(f"Pulse parameters updated: duty_cycle={self.duty_cycle*100:.1f}%, freq={self.frequency:.1f}Hz, idle_state={'high' if self.idle_state else 'low'}")

        except Exception as e:
            self.logger.error(f"Failed to set pulse parameters: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status information."""
        
        # Get interface status
        interface_status = self.interface.get_status()
        
        # Combine status information
        status = {
            "hardware_type": "digital_digilent",
            "device_index": self.device_index,
            "digital_channel": self.digital_channel,
            "duty_cycle_percent": self.duty_cycle * 100,
            "frequency_hz": self.frequency,
            "max_frequency_hz": self.max_frequency,
            "trigger_count": self.trigger_count,
            "last_trigger_time": self.last_trigger_time,
            "continuous_mode": self.continuous_mode,
            "interface_status": interface_status
        }
        
        return status
    
    def is_initialized(self) -> bool:
        """Check if the driver is initialized and ready."""
        return self.state == LaserState.ON and self.interface.connected

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