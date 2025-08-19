"""Laser Simulator."""
import logging
from queue import Queue
import threading
import time
import numpy as np
from typing import Dict, Any

from src.alice.laser.laser_base import BaseLaserDriver
from src.utils.data_structures import LaserInfo, Pulse


class SimulatedLaserDriver(BaseLaserDriver):
    def __init__(self, pulses_queue: Queue[Pulse], laser_info: LaserInfo):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.pulses_queue = pulses_queue
        self.laser_info = laser_info
        self.default_current_power_mW = laser_info.max_power_mW
        self.default_pulse_width_fwhm_ns = laser_info.pulse_width_fwhm_ns
        self.default_central_wavelength_nm = laser_info.central_wavelength_nm

        # State tracking
        self._initialized = False
        self.is_on = False
        self.is_continuous = False
        self.continuous_thread = None
        
        # Statistics
        self.pulse_count = 0
        self.last_trigger_time = 0.0

        self.logger.info("Simulated laser driver initialized.")

    def initialize(self) -> bool:
        """Initialize the simulated laser (ready to emit)."""
        try:
            self._initialized = True
            self.is_on = True  # Ready to emit after initialization
            self.logger.info("Simulated laser initialized and ready")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize simulated laser: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown the simulated laser."""
        try:
            if self.is_continuous:
                self.stop_continuous()
            
            self.is_on = False
            self._initialized = False
            self.logger.info("Simulated laser shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    def trigger_once(self) -> bool:
        """Send a single trigger pulse."""
        if not self._initialized or not self.is_on:
            self.logger.error("Cannot trigger pulse when laser is not initialized")
            return False
        
        try:
            pulse = self._generate_single_pulse()
            self.pulses_queue.put(pulse)
            self.pulse_count += 1
            self.last_trigger_time = time.time()
            
            self.logger.debug(f"Triggered single pulse: {pulse}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to trigger single pulse: {e}")
            return False

    def send_frame(self, n_triggers: int, rep_rate_hz: float) -> bool:
        """
        Send a frame of multiple trigger pulses.
        
        Args:
            n_triggers: Number of trigger pulses
            rep_rate_hz: Repetition rate in Hz
        """
        if not self._initialized or not self.is_on:
            self.logger.error("Cannot send frame when laser is not initialized")
            return False
        
        try:
            period = 1.0 / rep_rate_hz
            
            for i in range(n_triggers):
                pulse = self._generate_single_pulse()
                self.pulses_queue.put(pulse)
                self.pulse_count += 1
                
                # Wait between pulses (except for the last one)
                if i < n_triggers - 1:
                    time.sleep(period)
            
            self.last_trigger_time = time.time()
            self.logger.info(f"Sent frame with {n_triggers} pulses at {rep_rate_hz} Hz")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send frame: {e}")
            return False

    def start_continuous(self, rep_rate_hz: float) -> bool:
        """
        Start continuous trigger pulse generation.
        
        Args:
            rep_rate_hz: Repetition rate in Hz
        """
        if not self._initialized or not self.is_on:
            self.logger.error("Cannot start continuous mode when laser is not initialized")
            return False
        
        if self.is_continuous:
            self.logger.warning("Continuous mode is already active.")
            return True
        
        try:
            self.is_continuous = True
            self.continuous_thread = threading.Thread(
                target=self._continuous_loop, 
                args=(rep_rate_hz,), 
                daemon=True
            )
            self.continuous_thread.start()
            
            self.logger.info(f"Started continuous mode at {rep_rate_hz} Hz")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start continuous mode: {e}")
            self.is_continuous = False
            return False

    def stop_continuous(self) -> bool:
        """Stop continuous trigger pulse generation."""
        if not self.is_continuous:
            self.logger.warning("Continuous mode is not active.")
            return True
        
        try:
            self.is_continuous = False
            
            if self.continuous_thread and self.continuous_thread.is_alive():
                self.continuous_thread.join(timeout=2.0)
            
            self.logger.info("Stopped continuous mode")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop continuous mode: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status information."""
        return {
            "driver_type": "simulated",
            "initialized": self._initialized,
            "active": self.is_on,
            "continuous_mode": self.is_continuous,
            "pulse_count": self.pulse_count,
            "last_trigger_time": self.last_trigger_time,
            "laser_info": {
                "max_power_mW": self.laser_info.max_power_mW,
                "pulse_width_fwhm_ns": self.laser_info.pulse_width_fwhm_ns,
                "central_wavelength_nm": self.laser_info.central_wavelength_nm
            }
        }

    def _continuous_loop(self, rep_rate_hz: float):
        """Background continuous pulse generation loop."""
        period = 1.0 / rep_rate_hz
        
        while self.is_continuous and self.is_on:
            try:
                pulse = self._generate_single_pulse()
                self.pulses_queue.put(pulse)
                self.pulse_count += 1
                self.last_trigger_time = time.time()
                
                time.sleep(period)
                
            except Exception as e:
                self.logger.error(f"Error in continuous loop: {e}")
                break

    def _generate_single_pulse(self) -> Pulse:
        """
        Generates a pulse with a photon number drawn from a Poisson distribution and a certain polarization.

        Returns:
            Pulse: A Pulse object with the generated photon number and polarization.
        """
        if not self.is_on:
            raise RuntimeError("Cannot generate pulse when the laser is off.")
        
        # Simulate photon number based on laser info
        mean_photon_number = self._calculate_mean_photon_number()
        photon_number = np.random.poisson(mean_photon_number)

        # Simulate polarization
        # For simplicity, we can randomly choose a polarization angle
        polarization = np.random.choice([0, 45, 90, 135], p=[0.25, 0.25, 0.25, 0.25])
        # TODO - Implement a more sophisticated polarization model from laser_info
        
        return Pulse(photon_number=photon_number, polarization=polarization)

    def _calculate_mean_photon_number(self) -> float:
        """
        Calculate mean photon number based on current power and pulse parameters.
        
        Returns:
            Mean photon number per pulse
        """
        # Energy per pulse (J)
        pulse_energy_J = (self.default_current_power_mW * 1e-3 * 
                         self.default_pulse_width_fwhm_ns * 1e-9)
        
        # Photon energy (J)
        h = 6.62607015e-34  # Planck constant
        c = 2.99792458e8    # Speed of light
        wavelength_m = self.default_central_wavelength_nm * 1e-9
        photon_energy_J = h * c / wavelength_m
        
        # Mean photon number
        mean_photons = pulse_energy_J / photon_energy_J
        
        return max(0.0, mean_photons)