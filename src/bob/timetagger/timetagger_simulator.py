"""Time Tagger Simulator - Realistic Physics-Based Model."""
import logging
import numpy as np
import time
import threading
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from queue import Queue, Empty

from .timetagger_base import (
    BaseTimeTaggerDriver, TimeStamp, ChannelConfig, TimeTaggerConfig, 
    ChannelState, TimeTaggerStatistics
)


@dataclass
class SimulatorConfig:
    """Configuration specific to the simulator."""
    # Noise and jitter
    timing_jitter_ps: float = 10.0  # RMS timing jitter
    dark_count_rate_hz: float = 100.0  # Dark count rate per channel
    
    # System effects
    dead_time_variation_percent: float = 5.0  # Dead time variation
    trigger_jitter_ps: float = 5.0  # Trigger level jitter
    temperature_drift_ppm_per_k: float = 1.0  # Temperature drift
    
    # Realism settings
    enable_crosstalk: bool = True
    crosstalk_probability: float = 0.001  # Probability of crosstalk event
    enable_afterpulsing: bool = True
    afterpulsing_probability: float = 0.01
    afterpulsing_time_constant_ns: float = 100.0


class TimeTaggerSimulator(BaseTimeTaggerDriver):
    """
    Comprehensive time tagger simulator with realistic physics.
    
    Simulates:
    - Multi-channel timestamp acquisition
    - Timing jitter and dead time effects
    - Dark counts and afterpulsing
    - Channel crosstalk
    - Buffer management and overflow
    - Count rate measurements
    """
    
    def __init__(self, config: TimeTaggerConfig, sim_config: Optional[SimulatorConfig] = None):
        """
        Initialize the time tagger simulator.
        
        Args:
            config: Time tagger configuration
            sim_config: Simulator-specific configuration
        """
        super().__init__(config)
        self.sim_config = sim_config if sim_config is not None else SimulatorConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Simulation state
        self._event_buffer = Queue(maxsize=config.buffer_size)
        self._acquisition_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Channel states
        self._channel_states = {ch_id: ChannelState.DISABLED for ch_id in config.channels.keys()}
        
        # Per-channel statistics
        self._last_event_times = {}  # For dead time simulation
        self._afterpulse_queues = {ch_id: [] for ch_id in config.channels.keys()}
        
        # Timing reference
        self._time_offset_ps = 0
        self._resolution_ps = config.resolution_ps
        
        self.logger.info("Time tagger simulator initialized")
        self.logger.info(f"Channels: {list(config.channels.keys())}")
        self.logger.info(f"Resolution: {config.resolution_ps} ps")

    def initialize(self) -> bool:
        """Initialize the time tagger simulator."""
        try:
            # Reset all states
            self._stop_event.clear()
            self._time_offset_ps = int(time.time() * 1e12)  # Current time in ps
            
            # Initialize channel states
            for ch_id, ch_config in self.config.channels.items():
                if ch_config.enabled:
                    self._channel_states[ch_id] = ChannelState.ENABLED
                else:
                    self._channel_states[ch_id] = ChannelState.DISABLED
            
            self.logger.info("Time tagger simulator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize time tagger simulator: {e}")
            return False

    def start_measurement(self) -> bool:
        """Start timestamp data acquisition."""
        if self._is_measuring:
            self.logger.warning("Measurement already active")
            return False
        
        try:
            self._is_measuring = True
            self._start_time = time.time()
            self._stop_event.clear()
            
            # Start acquisition thread
            self._acquisition_thread = threading.Thread(
                target=self._acquisition_loop,
                name="TimeTaggerAcquisition"
            )
            self._acquisition_thread.start()
            
            self.logger.info("Time tagger measurement started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start measurement: {e}")
            self._is_measuring = False
            return False

    def stop_measurement(self) -> bool:
        """Stop timestamp data acquisition."""
        if not self._is_measuring:
            self.logger.warning("Measurement not active")
            return False
        
        try:
            self._is_measuring = False
            self._stop_event.set()
            
            # Wait for acquisition thread to finish
            if self._acquisition_thread and self._acquisition_thread.is_alive():
                self._acquisition_thread.join(timeout=2.0)
            
            self.logger.info("Time tagger measurement stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop measurement: {e}")
            return False

    def get_timestamps(self, max_events: Optional[int] = None) -> List[TimeStamp]:
        """Get collected timestamps from buffer."""
        timestamps = []
        count = 0
        
        while not self._event_buffer.empty() and (max_events is None or count < max_events):
            try:
                timestamp = self._event_buffer.get_nowait()
                timestamps.append(timestamp)
                count += 1
            except Empty:
                break
        
        # Update statistics
        self._update_statistics(timestamps)
        
        self.logger.debug(f"Retrieved {len(timestamps)} timestamps")
        return timestamps

    def configure_channel(self, channel_id: int, config: ChannelConfig) -> bool:
        """Configure a specific channel."""
        if channel_id not in self.config.channels:
            self.logger.error(f"Channel {channel_id} not found")
            return False
        
        try:
            self.config.channels[channel_id] = config
            
            # Update channel state
            if config.enabled:
                self._channel_states[channel_id] = ChannelState.ENABLED
            else:
                self._channel_states[channel_id] = ChannelState.DISABLED
            
            self.logger.info(f"Channel {channel_id} configured successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to configure channel {channel_id}: {e}")
            return False

    def get_channel_state(self, channel_id: int) -> ChannelState:
        """Get current state of a channel."""
        return self._channel_states.get(channel_id, ChannelState.ERROR)

    def get_count_rates(self) -> Dict[int, float]:
        """Get count rates for all enabled channels."""
        return self.stats.count_rates_hz.copy()

    def clear_buffer(self) -> bool:
        """Clear the internal event buffer."""
        try:
            while not self._event_buffer.empty():
                try:
                    self._event_buffer.get_nowait()
                except Empty:
                    break
            
            self.logger.debug("Event buffer cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear buffer: {e}")
            return False

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information and status."""
        return {
            "device_type": "Time Tagger Simulator",
            "resolution_ps": self.config.resolution_ps,
            "buffer_size": self.config.buffer_size,
            "channels": {
                ch_id: {
                    "enabled": ch_config.enabled,
                    "state": self._channel_states[ch_id].value,
                    "trigger_level_v": ch_config.trigger_level_v,
                    "dead_time_ps": ch_config.dead_time_ps
                }
                for ch_id, ch_config in self.config.channels.items()
            },
            "measurement_active": self._is_measuring,
            "buffer_usage": self._event_buffer.qsize(),
            "statistics": {
                "total_events": self.stats.total_events,
                "measurement_time_s": self.stats.measurement_time_s,
                "buffer_overflows": self.stats.buffer_overflows
            }
        }

    def _acquisition_loop(self) -> None:
        """Main acquisition loop running in separate thread."""
        self.logger.debug("Acquisition loop started")
        
        try:
            while not self._stop_event.is_set() and self._is_measuring:
                # Generate events for all enabled channels
                current_time_ps = int(time.time() * 1e12) - self._time_offset_ps
                
                for ch_id, ch_config in self.config.channels.items():
                    if not ch_config.enabled:
                        continue
                    
                    # Generate events for this channel
                    events = self._generate_channel_events(ch_id, ch_config, current_time_ps)
                    
                    # Add events to buffer
                    for event in events:
                        try:
                            self._event_buffer.put_nowait(event)
                        except:
                            # Buffer overflow
                            self.stats.buffer_overflows += 1
                            self.logger.warning(f"Buffer overflow, dropping event on channel {ch_id}")
                
                # Sleep for a short time to control event generation rate
                time.sleep(0.001)  # 1 ms
                
        except Exception as e:
            self.logger.error(f"Error in acquisition loop: {e}")
        finally:
            self.logger.debug("Acquisition loop ended")

    def _generate_channel_events(self, channel_id: int, config: ChannelConfig, 
                                current_time_ps: int) -> List[TimeStamp]:
        """Generate timestamp events for a specific channel."""
        events = []
        
        # Check dead time
        if self._is_in_dead_time(channel_id, current_time_ps):
            return events
        
        # Generate dark count events
        if self._should_generate_dark_count():
            event_time = self._apply_timing_jitter(current_time_ps)
            events.append(TimeStamp(
                channel=channel_id,
                time_ps=event_time,
                rising_edge=True
            ))
            self._record_event_time(channel_id, event_time)
        
        # Generate afterpulse events
        afterpulse_events = self._generate_afterpulse_events(channel_id, current_time_ps)
        events.extend(afterpulse_events)
        
        # Generate crosstalk events
        if self.sim_config.enable_crosstalk:
            crosstalk_events = self._generate_crosstalk_events(channel_id, current_time_ps)
            events.extend(crosstalk_events)
        
        return events

    def _is_in_dead_time(self, channel_id: int, current_time_ps: int) -> bool:
        """Check if channel is in dead time."""
        if channel_id not in self._last_event_times:
            return False
        
        last_event_time = self._last_event_times[channel_id]
        dead_time_ps = self.config.channels[channel_id].dead_time_ps
        
        # Add dead time variation
        variation = np.random.normal(0, dead_time_ps * self.sim_config.dead_time_variation_percent / 100)
        actual_dead_time = dead_time_ps + variation
        
        return (current_time_ps - last_event_time) < actual_dead_time

    def _should_generate_dark_count(self) -> bool:
        """Determine if a dark count should be generated."""
        # Calculate probability of dark count in 1 ms window
        time_window_s = 0.001
        dark_prob = self.sim_config.dark_count_rate_hz * time_window_s
        return np.random.random() < dark_prob

    def _generate_afterpulse_events(self, channel_id: int, current_time_ps: int) -> List[TimeStamp]:
        """Generate afterpulse events based on previous detections."""
        events = []
        
        if not self.sim_config.enable_afterpulsing:
            return events
        
        # Check afterpulse queue for this channel
        if channel_id in self._afterpulse_queues:
            # Remove expired afterpulse events
            cutoff_time = current_time_ps - 10 * self.sim_config.afterpulsing_time_constant_ns * 1000
            self._afterpulse_queues[channel_id] = [
                t for t in self._afterpulse_queues[channel_id] if t > cutoff_time
            ]
            
            # Check for afterpulse generation
            for prev_time in self._afterpulse_queues[channel_id]:
                time_diff_ns = (current_time_ps - prev_time) / 1000
                decay_factor = np.exp(-time_diff_ns / self.sim_config.afterpulsing_time_constant_ns)
                afterpulse_prob = self.sim_config.afterpulsing_probability * decay_factor
                
                if np.random.random() < afterpulse_prob:
                    event_time = self._apply_timing_jitter(current_time_ps)
                    events.append(TimeStamp(
                        channel=channel_id,
                        time_ps=event_time,
                        rising_edge=True
                    ))
                    self._record_event_time(channel_id, event_time)
        
        return events

    def _generate_crosstalk_events(self, channel_id: int, current_time_ps: int) -> List[TimeStamp]:
        """Generate crosstalk events from other channels."""
        events = []
        
        # Check recent events on other channels
        for other_ch_id in self.config.channels.keys():
            if other_ch_id == channel_id or not self.config.channels[other_ch_id].enabled:
                continue
            
            if other_ch_id in self._last_event_times:
                time_diff = current_time_ps - self._last_event_times[other_ch_id]
                # Crosstalk is most likely shortly after events on other channels
                if time_diff < 10000:  # Within 10 ns
                    if np.random.random() < self.sim_config.crosstalk_probability:
                        event_time = self._apply_timing_jitter(current_time_ps)
                        events.append(TimeStamp(
                            channel=channel_id,
                            time_ps=event_time,
                            rising_edge=True
                        ))
                        self._record_event_time(channel_id, event_time)
        
        return events

    def _apply_timing_jitter(self, ideal_time_ps: int) -> int:
        """Apply timing jitter to event time."""
        jitter_ps = np.random.normal(0, self.sim_config.timing_jitter_ps)
        return int(ideal_time_ps + jitter_ps)

    def _record_event_time(self, channel_id: int, event_time_ps: int) -> None:
        """Record event time for dead time and afterpulse calculations."""
        self._last_event_times[channel_id] = event_time_ps
        
        # Add to afterpulse queue
        if channel_id in self._afterpulse_queues:
            self._afterpulse_queues[channel_id].append(event_time_ps)

    def inject_test_event(self, channel_id: int, time_ps: Optional[int] = None) -> bool:
        """
        Inject a test event for calibration/testing purposes.
        
        Args:
            channel_id: Channel to inject event on
            time_ps: Event time (if None, use current time)
            
        Returns:
            bool: True if event injected successfully
        """
        if channel_id not in self.config.channels:
            return False
        
        if time_ps is None:
            time_ps = int(time.time() * 1e12) - self._time_offset_ps
        
        try:
            test_event = TimeStamp(
                channel=channel_id,
                time_ps=time_ps,
                rising_edge=True
            )
            self._event_buffer.put_nowait(test_event)
            self._record_event_time(channel_id, time_ps)
            
            self.logger.debug(f"Test event injected on channel {channel_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to inject test event: {e}")
            return False

    def set_dark_count_rate(self, rate_hz: float) -> None:
        """Set dark count rate for simulation."""
        self.sim_config.dark_count_rate_hz = rate_hz
        self.logger.info(f"Dark count rate set to {rate_hz} Hz")

    def enable_realistic_effects(self, enable: bool = True) -> None:
        """Enable or disable realistic physics effects."""
        self.sim_config.enable_crosstalk = enable
        self.sim_config.enable_afterpulsing = enable
        self.logger.info(f"Realistic effects {'enabled' if enable else 'disabled'}")

    def get_simulator_config(self) -> SimulatorConfig:
        """Get current simulator configuration."""
        return self.sim_config
