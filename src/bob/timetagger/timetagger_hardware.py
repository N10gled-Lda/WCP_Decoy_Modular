"""
Time Tagger Hardware Interface using Swabian Instruments API.
Provides gated detection capabilities for QKD measurements.
"""
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

from .timetagger_base import BaseTimeTaggerDriver, TimeTaggerConfig

try:
    import TimeTagger
    TIMETAGGER_AVAILABLE = True
except ImportError:
    TIMETAGGER_AVAILABLE = False
    logging.warning("TimeTagger module not available. Hardware interface will not work.")


class GateMode(Enum):
    """Gating modes for detection."""
    CONTINUOUS = "continuous"  # No gating, continuous detection
    GATED = "gated"           # Gated detection between markers
    SYNCHRONIZED = "synchronized"  # Synchronized with external trigger


# Removed duplicate TimeTaggerConfig - now using unified version from timetagger_base.py


class TimeTaggerHardware(BaseTimeTaggerDriver):
    """
    Interface to the physical time tagger hardware (Swabian Instruments).
    
    Provides gated detection capabilities for QKD measurements where:
    - Alice sends a trigger signal when transmitting a pulse
    - Bob gates the detectors only during expected arrival times
    - Counts are accumulated per gate window and returned as detection events
    """
    
    def __init__(self, config: TimeTaggerConfig = None):
        # Initialize parent class
        self.config = config or TimeTaggerConfig(channels={})
        
        # Set hardware defaults if not specified
        if self.config.gate_mode is None:
            self.config.gate_mode = GateMode.GATED
            
        super().__init__(self.config)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        if not TIMETAGGER_AVAILABLE:
            raise RuntimeError("TimeTagger module not available. Cannot use hardware interface.")
        
        # Hardware state
        self.tagger: Optional[TimeTagger.TimeTagger] = None
        self.counter: Optional[TimeTagger.Counter] = None
        self.gated_counters: Dict[int, TimeTagger.CountBetweenMarkers] = {}
        self.delay_channel: Optional[TimeTagger.DelayedChannel] = None
        
        # Measurement state
        self._measuring = False
        self._measurement_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Results
        self.detection_counts: Dict[int, List[int]] = {}
        self.measurement_times: List[float] = []
        
        self.logger.info("TimeTagger hardware interface initialized")

    def initialize(self) -> bool:
        """Initialize the TimeTagger hardware."""
        try:
            # Scan for available devices
            self.logger.info("Scanning for TimeTagger devices...")
            start_time = time.time()
            available_taggers = []
            
            while time.time() < start_time + 2:
                available_taggers = TimeTagger.scanTimeTagger()
                if available_taggers:
                    break
                time.sleep(0.1)
            
            if not available_taggers:
                self.logger.error("No TimeTagger devices found")
                return False
            
            self.logger.info(f"Found TimeTagger devices: {available_taggers}")
            
            # Create TimeTagger instance
            self.tagger = TimeTagger.createTimeTagger()
            
            # Get available channels
            channels = self.tagger.getChannelList()
            self.logger.info(f"Available channels: {channels}")
            
            # Setup test signals if requested
            if self.config.use_test_signal:
                for channel in self.config.test_signal_channels:
                    if channel <= channels:
                        self.tagger.setTestSignal(channel, True)
                        self.logger.info(f"Test signal enabled on channel {channel}")
            
            # Setup gating if needed
            if self.config.gate_mode == GateMode.GATED:
                self._setup_gated_detection()
            else:
                self._setup_continuous_detection()
            
            self.logger.info("TimeTagger hardware initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TimeTagger hardware: {e}")
            return False

    def _setup_gated_detection(self) -> None:
        """Setup gated detection between markers."""
        # Create delay channel for gate end if using same channel
        if self.config.gate_end_channel == -self.config.gate_begin_channel:
            self.delay_channel = TimeTagger.DelayedChannel(
                self.tagger, 
                self.config.gate_begin_channel, 
                delay=self.config.gate_length_ps
            )
            end_channel = self.delay_channel.getChannel()
        else:
            end_channel = self.config.gate_end_channel
        
        # Create gated counters for each detector channel
        for channel in self.config.detector_channels:
            try:
                gated_counter = TimeTagger.CountBetweenMarkers(
                    tagger=self.tagger,
                    click_channel=channel,
                    begin_channel=self.config.gate_begin_channel,
                    end_channel=end_channel,
                    n_values=self.config.n_values
                )
                self.gated_counters[channel] = gated_counter
                self.detection_counts[channel] = []
                self.logger.info(f"Gated counter setup for channel {channel}")
            except Exception as e:
                self.logger.error(f"Failed to setup gated counter for channel {channel}: {e}")

    def _setup_continuous_detection(self) -> None:
        """Setup continuous detection."""
        self.counter = TimeTagger.Counter(
            tagger=self.tagger,
            channels=self.config.detector_channels,
            binwidth=self.config.binwidth_ps,
            n_values=self.config.n_values
        )
        
        for channel in self.config.detector_channels:
            self.detection_counts[channel] = []
        
        self.logger.info("Continuous counter setup")

    def start_measurement(self, duration_s: Optional[float] = None) -> bool:
        """Start measurement."""
        if self._measuring:
            self.logger.warning("Measurement already running")
            return False
        
        duration = duration_s or self.config.measurement_duration_s
        duration_ps = int(duration * 1e12)
        
        try:
            if self.config.gate_mode == GateMode.GATED:
                # Start all gated counters
                for counter in self.gated_counters.values():
                    counter.startFor(duration_ps)
            else:
                # Start continuous counter
                self.counter.startFor(duration_ps)
            
            self._measuring = True
            self.logger.info(f"Measurement started for {duration}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start measurement: {e}")
            return False

    def wait_for_completion(self) -> None:
        """Wait for measurement to complete."""
        if not self._measuring:
            return
        
        try:
            if self.config.gate_mode == GateMode.GATED:
                # Wait for all gated counters
                for counter in self.gated_counters.values():
                    counter.waitUntilFinished()
            else:
                # Wait for continuous counter
                self.counter.waitUntilFinished()
            
            self._measuring = False
            self.logger.info("Measurement completed")
            
        except Exception as e:
            self.logger.error(f"Error waiting for measurement completion: {e}")

    def get_detection_data(self) -> Dict[int, List[int]]:
        """Get detection counts from completed measurement."""
        if self._measuring:
            self.logger.warning("Measurement still running")
            return {}
        
        try:
            results = {}
            
            if self.config.gate_mode == GateMode.GATED:
                # Get data from gated counters
                for channel, counter in self.gated_counters.items():
                    data = counter.getData()
                    results[channel] = data.tolist() if hasattr(data, 'tolist') else list(data)
            else:
                # Get data from continuous counter
                data = self.counter.getData()
                for i, channel in enumerate(self.config.detector_channels):
                    channel_data = data[i] if len(data) > i else []
                    results[channel] = channel_data.tolist() if hasattr(channel_data, 'tolist') else list(channel_data)
            
            self.logger.info(f"Retrieved detection data for {len(results)} channels")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get detection data: {e}")
            return {}

    def get_single_gate_counts(self) -> Dict[int, int]:
        """Get counts for a single gate window (immediate measurement)."""
        if self.config.gate_mode != GateMode.GATED:
            self.logger.error("Single gate measurement only available in gated mode")
            return {}
        
        try:
            # Start measurement for one gate
            gate_duration_ps = self.config.gate_length_ps * 2  # Allow some margin
            
            for counter in self.gated_counters.values():
                counter.startFor(gate_duration_ps)
            
            # Wait for completion
            for counter in self.gated_counters.values():
                counter.waitUntilFinished()
            
            # Get results
            results = {}
            for channel, counter in self.gated_counters.items():
                data = counter.getData()
                # Get the latest count (should be just one value)
                count = int(data[-1]) if len(data) > 0 else 0
                results[channel] = count
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get single gate counts: {e}")
            return {}

    def is_measuring(self) -> bool:
        """Check if measurement is currently running."""
        return self._measuring

    def stop_measurement(self) -> None:
        """Stop current measurement."""
        if not self._measuring:
            return
        
        try:
            if self.config.gate_mode == GateMode.GATED:
                for counter in self.gated_counters.values():
                    counter.stop()
            else:
                self.counter.stop()
            
            self._measuring = False
            self.logger.info("Measurement stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop measurement: {e}")

    def get_channel_info(self) -> Dict[str, Any]:
        """Get information about configured channels."""
        if not self.tagger:
            return {}
        
        info = {
            "detector_channels": self.config.detector_channels,
            "gate_begin_channel": self.config.gate_begin_channel,
            "gate_end_channel": self.config.gate_end_channel,
            "total_channels": self.tagger.getChannelList(),
            "gate_mode": self.config.gate_mode.value
        }
        
        return info

    def shutdown(self) -> bool:
        """Shutdown the TimeTagger hardware."""
        self.logger.info("Shutting down TimeTagger hardware...")
        
        try:
            # Stop any running measurements
            self.stop_measurement()
            
            # Clean up counters
            if self.gated_counters:
                for counter in self.gated_counters.values():
                    del counter
                self.gated_counters.clear()
            
            if self.counter:
                del self.counter
                self.counter = None
            
            if self.delay_channel:
                del self.delay_channel
                self.delay_channel = None
            
            # Free the TimeTagger instance
            if self.tagger:
                TimeTagger.freeTimeTagger(self.tagger)
                self.tagger = None
            
            self.logger.info("TimeTagger hardware shutdown complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during TimeTagger shutdown: {e}")
            return False

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.shutdown()
