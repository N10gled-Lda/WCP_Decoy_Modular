"""
Simple TimeTagger Interface - Just measure counts for a duration.
No complex modes, gating logic, or configurations needed.
"""
import time
import logging
from typing import Dict, Optional
from abc import ABC, abstractmethod
import random
import numpy as np


class SimpleTimeTagger(ABC):
    """
    Minimal TimeTagger interface - just measure counts for a given duration.
    Hardware gating is handled by the hardware itself.
    """
    
    def __init__(self, detector_channels: list = None):
        """
        Initialize with detector channels.
        
        Args:
            detector_channels: List of detector channel IDs (default: [1, 2, 3, 4])
        """
        self.detector_channels = detector_channels or [1, 2, 3, 4]
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the timetagger."""
        pass
    
    @abstractmethod
    def measure_for_duration(self, duration_seconds: float) -> Dict[int, int]:
        """
        Measure counts for the specified duration.
        
        Args:
            duration_seconds: How long to measure
            
        Returns:
            Dict[int, int]: Channel ID -> count mapping
        """
        pass
    
    @abstractmethod
    def set_measurement_duration(self, duration_seconds: float) -> bool:
        """Set measurement duration and configure counter."""
        pass
    
    @abstractmethod
    def get_timebin_data(self, duration_seconds: float) -> Dict[str, any]:
        """Get raw time-binned data for advanced analysis."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the timetagger."""
        pass


class SimpleTimeTaggerHardware(SimpleTimeTagger):
    """Hardware implementation using Swabian TimeTagger."""
    
    def __init__(self, detector_channels: list = None):
        super().__init__(detector_channels)
        self.tagger = None
        self.counter = None
        self._measurement_duration = None  # Must be set before measuring
        
        # For gated detectors: use SMALL binwidth to get time resolution
        # Binwidth should be much smaller than measurement duration to capture 
        # when during the measurement window the counts actually occurred        
        self.binwidth_ps = int(100e9)  # 100ms = 100e9 picoseconds (fine time resolution) 
        # self.binwidth_ps = int(duration_seconds )  # duration for single bin (no time structure) get me just the total counts of the measurement

        
        try:
            import TimeTagger as TimeTagger
            self.TimeTagger = TimeTagger
            self.hardware_available = True
        except ImportError:
            self.logger.warning("TimeTagger module not available")
            print(" WARNING: TimeTagger module not available - hardware mode disabled")
            self.hardware_available = False
    
    def initialize(self) -> bool:
        """Initialize hardware timetagger (without counter - that's created when duration is set)."""
        if not self.hardware_available:
            return False
            
        try:
            start_time = time.time()
            print("Scanning for TimeTaggers for 10s...")
            while time.time() < start_time + 10:
                available_taggers = self.TimeTagger.scanTimeTagger()
                if available_taggers:
                    print("\nTime Taggers available via TimeTagger.scanTimeTagger():")
                    print(available_taggers)
                    break
            if not available_taggers:
                print("There are no Time Taggers available. Connect one and retry.")
                return False

            self.tagger = self.TimeTagger.createTimeTagger()
            if self.tagger is None:
                self.logger.error("No TimeTagger device found")
                return False
            
            self.logger.info("TimeTagger hardware initialized (counter will be created when measurement duration is set)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hardware: {e}")
            return False

    def scan_for_devices(self, timeout: float = 10.0) -> list:
        """Scan for connected TimeTagger devices."""
        if not self.hardware_available:
            self.logger.warning("Hardware not available")
            return []

        try:
            start_time = time.time()
            self.logger.info(f"Scanning for TimeTagger devices for {timeout}s...")
            while time.time() < start_time + timeout:
                available_taggers = self.TimeTagger.scanTimeTagger()
                if available_taggers:
                    self.logger.info("Time Taggers available via TimeTagger.scanTimeTagger():")
                    self.logger.info(available_taggers)
                    return available_taggers
            self.logger.warning("No Time Taggers found.")
            return []

        except Exception as e:
            self.logger.error(f"Failed to scan for devices: {e}")
            return []
        
    def set_measurement_duration(self, duration_seconds: float) -> bool:
        """Set measurement duration and create appropriate counter with fine time resolution."""
        if not self.tagger:
            self.logger.error("Tagger not initialized")
            return False
            
        try:
            # Clean up old counter if exists
            if self.counter:
                self.counter = None            
            
            # Number of bins to cover the full measurement duration
            n_bins = int(duration_seconds * 1e12 / self.binwidth_ps)

            self.counter = self.TimeTagger.Counter(
                tagger=self.tagger,
                channels=self.detector_channels,
                binwidth=self.binwidth_ps,
                n_values=n_bins  # Multiple bins to capture timing structure
            )
            
            self._measurement_duration = duration_seconds
            self.logger.info(f"Counter configured for {duration_seconds}s measurements with {n_bins} bins of {self.binwidth_ps/1e9:.1f}ms each")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set measurement duration: {e}")
            return False
    
    def measure_for_duration(self, duration_seconds: float) -> Dict[int, int]:
        """
        Measure counts using hardware counter with time-binned data.
        
        Returns total counts per channel, but internally captures time-binned structure
        that can be used for pulse-by-pulse analysis with gated detectors.
        """
        if not self.counter:
            self.logger.error("Counter not configured. Call set_measurement_duration() first")
            return {ch: 0 for ch in self.detector_channels}
        
        # Check if duration matches configured duration
        if self._measurement_duration and abs(duration_seconds - self._measurement_duration) > 0.001:
            self.logger.warning(f"Requested duration {duration_seconds}s differs from configured {self._measurement_duration}s. Setting new duration.")
            self.set_measurement_duration(duration_seconds)
            if not self.counter:
                return {ch: 0 for ch in self.detector_channels}

        try:
            # Clear any previous data
            self.counter.clear()
            # Start measurement for the requested duration
            self.counter.startFor(int(self._measurement_duration * 1e12))  # Convert to picoseconds
            
            # Wait for completion
            # while self.counter.isRunning():
            #     time.sleep(0.001)  # 1ms polling
            self.counter.waitUntilFinished()
            
            # Get time-binned results according to Swabian documentation
            # getData() returns int[channels, n_values] - 2D array
            data = self.counter.getData(rolling=False)  # Use non-rolling for easier parsing
            counts = {}
            
            # Process the 2D array: data[channel_index][time_bin_index]
            for i, channel in enumerate(self.detector_channels):
                # print(f" DEBUG: Processing channel {channel} out of {len(data)}:")
                if i < len(data):
                    # data[i] is the array of time bins for this channel
                    time_bins = data[i]  # Array of counts per time bin
                    # print(f" DEBUG: Time bins data length {len(time_bins)}")
                    # print(f" DEBUG: Time bins data sample {time_bins} ...")
                    total_counts = sum(time_bins)  # Sum all bins for total counts
                    counts[channel] = int(total_counts)
                    
                    # Debug: show time structure (where/when counts occurred)
                    # TODO: If time of bin is needed for pulse-by-pulse analysis get it (and return) from here bin_idx * binwidth
                    if total_counts > 0:
                        non_zero_bins = [(bin_idx, int(count)) for bin_idx, count in enumerate(time_bins) if count > 0]
                        bin_time_s = [(bin_idx * self.binwidth_ps / 1e12) for bin_idx, _ in non_zero_bins]
                        self.logger.debug(f"Channel {channel}: {total_counts} total counts in {len(non_zero_bins)} bins: {non_zero_bins}")
                        self.logger.debug(f"  Bin times: {[f'{t:.1f}s' for t in bin_time_s]}")
                        # print(f" DEBUG: Channel {channel}: {total_counts} counts at times {[f'{t:.1f}s' for t in bin_time_s]}")
                    else:
                        self.logger.debug(f"Channel {channel}: No counts detected")
                        # print(f" DEBUG: Channel {channel}: No counts")
                else:
                    counts[channel] = 0
                    self.logger.debug(f"Channel {channel}: No data available")
                    # print(f" DEBUG: Channel {channel}: No data available")

            return counts
            
        except Exception as e:
            self.logger.error(f"Measurement failed: {e}")
            return {ch: 0 for ch in self.detector_channels}
    
    def get_timebin_data(self, duration_seconds: float) -> Dict[str, any]:
        """
        Get raw time-binned data for advanced pulse-by-pulse analysis.
        
        Returns:
            Dict containing:
            - 'counts_per_channel': Dict[int, int] (same as measure_for_duration)  
            - 'timebin_data': 2D array from TimeTagger getData()
            - 'binwidth_ps': Bin width in picoseconds
            - 'n_bins': Number of time bins
        """
        if not self.counter:
            return {'error': 'Counter not configured'}
        
        try:
            if self._measurement_duration and abs(duration_seconds - self._measurement_duration) > 0.001:
                self.logger.warning(f"Requested duration {duration_seconds}s differs from configured {self._measurement_duration}s. Setting new duration.")
                self.set_measurement_duration(duration_seconds)
                
            if not self.counter:
                return {'error': 'Failed to set measurement duration'}
            
            # Clear and measure (same as measure_for_duration)
            self.counter.clear()
            self.counter.startFor(int(duration_seconds * 1e12))
            
            # while self.counter.isRunning():
            #     time.sleep(0.001)
            self.logger.debug(f" Waiting for measurement to finish (~{duration_seconds}s)...")
            self.counter.waitUntilFinished()
            self.logger.debug(" Measurement finished.")

            # Get raw 2D time-binned data 
            timebin_data = self.counter.getData(rolling=False)
            self.logger.debug(f" Retrieved timebin data with shape {np.array(timebin_data).shape}")
            # Also compute total counts per channel
            counts_per_channel = {}
            for i, channel in enumerate(self.detector_channels):
                if i < len(timebin_data):
                    counts_per_channel[channel] = sum(timebin_data[i])
                else:
                    counts_per_channel[channel] = 0
            
            nbins = len(timebin_data[0]) if len(timebin_data) > 0 else 0
            return {
                'counts_per_channel': counts_per_channel,
                'timebin_data': timebin_data,  # Raw 2D array[channel][time_bin]
                'binwidth_ps': self.binwidth_ps,
                'n_bins': nbins,
                'channels': self.detector_channels
            }
            
        except Exception as e:
            print(f" DEBUG: Error getting timebin data: {e}")
            return {'error': str(e)}
    
    def shutdown(self) -> None:
        """Shutdown hardware."""
        try:
            if self.counter:
                self.counter = None
            if self.tagger:
                self.TimeTagger.freeTimeTagger(self.tagger)
                self.tagger = None
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")


class SimpleTimeTaggerSimulator(SimpleTimeTagger):
    """
    Simulator implementation that closely replicates hardware behavior.
    Includes time-binned data structure, proper counter management, and realistic timing.
    """
    
    def __init__(self, detector_channels: list = None, dark_count_rate: float = 100.0, 
                 signal_count_rate: float = 50.0, signal_probability: float = 0.1):
        super().__init__(detector_channels)
        
        # Simulation parameters
        self.dark_count_rate = dark_count_rate  # counts per second per channel (background)
        self.signal_count_rate = signal_count_rate  # additional counts per second when signal present
        self.signal_probability = signal_probability  # probability of signal being present in a time bin
        
        # Mirror hardware attributes
        self.tagger = None  # Simulated tagger object
        self.counter = None  # Simulated counter object
        self._measurement_duration = None
        self.initialized = False
        
        # Match hardware binning parameters
        self.binwidth_ps = int(100e9)  # 100ms = 100e9 picoseconds (same as hardware)
        
        # Simulation state
        self._last_timebin_data = None  # Store last measurement's time-binned data
    
    def initialize(self) -> bool:
        """Initialize simulator (mimic hardware initialization flow)."""
        try:
            # Simulate hardware scanning/connection
            print("Simulating TimeTagger scan...")
            time.sleep(0.01)  # Very brief delay to mimic hardware scan
            
            # Create simulated tagger
            self.tagger = "SimulatedTagger"  # Placeholder object
            self.initialized = True
            
            self.logger.info("TimeTagger simulator initialized (counter will be created when measurement duration is set)")
            print("Simulated TimeTagger connected successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize simulator: {e}")
            return False
    
    def set_measurement_duration(self, duration_seconds: float) -> bool:
        """
        Set measurement duration and create simulated counter (mirrors hardware behavior).
        """
        if not self.tagger:
            self.logger.error("Tagger not initialized")
            return False
            
        try:
            # Clean up old counter if exists (mimic hardware)
            if self.counter:
                self.counter = None
            
            # Calculate bins (same logic as hardware)
            n_bins = int(duration_seconds * 1e12 / self.binwidth_ps)
            
            # Create simulated counter configuration
            self.counter = {
                'channels': self.detector_channels.copy(),
                'binwidth_ps': self.binwidth_ps,
                'n_bins': n_bins,
                'duration_seconds': duration_seconds,
                'configured': True
            }
            
            self._measurement_duration = duration_seconds
            self.logger.info(f"Counter configured for {duration_seconds}s measurements with {n_bins} bins of {self.binwidth_ps/1e9:.1f}ms each")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set measurement duration: {e}")
            return False
    
    def measure_for_duration(self, duration_seconds: float) -> Dict[int, int]:
        """
        Simulate measurement with time-binned structure (mirrors hardware exactly).
        """
        if not self.counter:
            self.logger.error("Counter not configured. Call set_measurement_duration() first")
            return {ch: 0 for ch in self.detector_channels}
        
        # Check if duration matches configured duration (same as hardware)
        if self._measurement_duration and abs(duration_seconds - self._measurement_duration) > 0.001:
            self.logger.warning(f"Requested duration {duration_seconds}s differs from configured {self._measurement_duration}s. Setting new duration.")
            self.set_measurement_duration(duration_seconds)
            if not self.counter:
                return {ch: 0 for ch in self.detector_channels}

        try:
            # Simulate hardware measurement process
            print(f" DEBUG: Starting simulated measurement for {self._measurement_duration}s...")
            
            # Generate time-binned data (same structure as hardware returns)
            timebin_data = self._generate_timebin_data()
            self._last_timebin_data = timebin_data  # Store for get_timebin_data()
            
            # Process the 2D array exactly like hardware does
            counts = {}
            print(f" DEBUG: Length of data {len(timebin_data)}")
            
            for i, channel in enumerate(self.detector_channels):
                print(f" DEBUG: Processing channel {channel}")
                if i < len(timebin_data):
                    # timebin_data[i] is the array of time bins for this channel
                    time_bins = timebin_data[i]  # Array of counts per time bin
                    total_counts = sum(time_bins)  # Sum all bins for total counts
                    counts[channel] = int(total_counts)
                    
                    # Debug: show time structure (same as hardware)
                    if total_counts > 0:
                        non_zero_bins = [(bin_idx, int(count)) for bin_idx, count in enumerate(time_bins) if count > 0]
                        bin_time_s = [(bin_idx * self.binwidth_ps / 1e12) for bin_idx, _ in non_zero_bins]
                        self.logger.debug(f"Channel {channel}: {total_counts} total counts in {len(non_zero_bins)} bins: {non_zero_bins}")
                        self.logger.debug(f"  Bin times: {[f'{t:.1f}s' for t in bin_time_s]}")
                        print(f" DEBUG: Channel {channel}: {total_counts} counts at times {[f'{t:.1f}s' for t in bin_time_s]}")
                    else:
                        self.logger.debug(f"Channel {channel}: No counts detected")
                        print(f" DEBUG: Channel {channel}: No counts")
                else:
                    counts[channel] = 0
                    print(f" DEBUG: Channel {channel}: No data available")
            
            # Simulate realistic measurement time
            measurement_time = min(self._measurement_duration * 0.1, 0.2)  # Much faster for simulation
            measurement_time = self._measurement_duration
            time.sleep(measurement_time)
            
            return counts
            
        except Exception as e:
            self.logger.error(f"Measurement failed: {e}")
            return {ch: 0 for ch in self.detector_channels}
    
    def get_timebin_data(self, duration_seconds: float) -> Dict[str, any]:
        """
        Get raw time-binned data for advanced analysis (mirrors hardware method exactly).
        """
        if not self.counter:
            return {'error': 'Counter not configured'}
        
        try:
            self.set_measurement_duration(duration_seconds)
            if not self.counter:
                return {'error': 'Failed to set measurement duration'}
            
            # Simulate measurement process (same as measure_for_duration)
            print(f" DEBUG: Waiting for measurement to finish (~{duration_seconds}s)...")
            
            # Generate time-binned data
            timebin_data = self._generate_timebin_data()
            print(f" DEBUG: Retrieved timebin data with shape {np.array(timebin_data).shape}")
            
            # Compute total counts per channel (same logic as hardware)
            counts_per_channel = {}
            for i, channel in enumerate(self.detector_channels):
                if i < len(timebin_data):
                    counts_per_channel[channel] = sum(timebin_data[i])
                else:
                    counts_per_channel[channel] = 0
            
            # Simulate measurement time
            measurement_time = min(duration_seconds * 0.1, 0.2)  # Much faster for simulation
            measurement_time = duration_seconds
            time.sleep(measurement_time)
            print(" DEBUG: Measurement finished.")
            
            nbins = len(timebin_data[0]) if len(timebin_data) > 0 else 0
            return {
                'counts_per_channel': counts_per_channel,
                'timebin_data': timebin_data,  # Raw 2D array[channel][time_bin] 
                'binwidth_ps': self.binwidth_ps,
                'n_bins': nbins,
                'channels': self.detector_channels
            }
            
        except Exception as e:
            print(f" DEBUG: Error getting timebin data: {e}")
            return {'error': str(e)}
    
    def _generate_timebin_data(self) -> list:
        """
        Generate realistic time-binned data that mimics hardware behavior.
        
        Returns:
            List[List[int]]: 2D array where data[channel_idx][time_bin_idx] = count
        """
        n_bins = self.counter['n_bins']
        bin_duration_s = self.binwidth_ps / 1e12  # Convert to seconds
        
        timebin_data = []
        
        for channel in self.detector_channels:
            channel_bins = []
            
            for bin_idx in range(n_bins):
                # Simulate dark counts (always present, Poisson distributed)
                expected_dark = self.dark_count_rate * bin_duration_s
                dark_counts = max(0, int(np.random.poisson(expected_dark)))
                
                # Simulate signal counts (occasionally present)
                signal_counts = 0
                if random.random() < self.signal_probability:
                    expected_signal = self.signal_count_rate * bin_duration_s
                    signal_counts = max(0, int(np.random.poisson(expected_signal)))
                
                total_counts = dark_counts + signal_counts
                channel_bins.append(total_counts)
            
            timebin_data.append(channel_bins)
        
        return timebin_data
    
    def get_measurement_duration(self) -> Optional[float]:
        """Get current measurement duration."""
        return self._measurement_duration

    def get_measurement_duration(self) -> Optional[float]:
        """Get current measurement duration."""
        return self._measurement_duration

    def shutdown(self) -> None:
        """Shutdown simulator (mirrors hardware cleanup)."""
        try:
            if self.counter:
                self.counter = None
            if self.tagger:
                self.tagger = None
            self.initialized = False
            self.logger.info("TimeTagger simulator shutdown complete")
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")