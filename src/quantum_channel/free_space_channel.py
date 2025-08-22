"""Free Space Channel Simulator - Enhanced Physics-Based Model."""
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import time
from threading import Lock

from ..utils.data_structures import Pulse, Basis


@dataclass
class ChannelConfig:
    """Configuration for the quantum channel."""
    # Basic parameters
    distance_km: float = 20.0
    base_attenuation_db_km: float = 0.2
    
    # Atmospheric effects
    enable_atmospheric_turbulence: bool = True
    turbulence_strength: float = 0.1  # Relative strength
    weather_condition: str = "clear"  # "clear", "hazy", "light_rain", "heavy_rain"
    
    # Polarization effects
    enable_polarization_drift: bool = True
    pol_drift_rate_deg_per_km: float = 5.0
    pol_mode_dispersion_ps_per_km: float = 0.1
    
    # Timing effects
    enable_timing_jitter: bool = True
    timing_jitter_ps: float = 50.0
    
    # Background noise
    background_photon_rate_hz: float = 1000.0
    enable_background_noise: bool = True
    
    # Eavesdropping simulation
    enable_eavesdropper: bool = False
    eavesdropper_strength: float = 0.1  # Fraction of photons intercepted


@dataclass 
class ChannelStatistics:
    """Statistics for channel performance."""
    total_pulses_received: int = 0
    total_photons_transmitted: int = 0
    total_photons_received: int = 0
    average_transmission_efficiency: float = 0.0
    polarization_errors: int = 0
    timing_errors: int = 0
    background_counts: int = 0
    eavesdropper_intercepts: int = 0
    channel_losses: List[float] = field(default_factory=list)


class FreeSpaceChannelSimulator:
    """
    Enhanced quantum channel simulator with realistic physics effects.
    
    Simulates:
    - Distance-based attenuation
    - Atmospheric turbulence and weather effects  
    - Polarization drift and mode dispersion
    - Timing jitter
    - Background noise
    - Optional eavesdropping
    """
    
    def __init__(self, config: ChannelConfig):
        """
        Initialize the free space channel simulator.

        Args:
            config: Channel configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.stats = ChannelStatistics()
        self._lock = Lock()
        
        # Calculate base transmission probability
        self.base_transmission_prob = 10**(-self.config.base_attenuation_db_km * self.config.distance_km / 10)
        
        # Weather-dependent attenuation factors
        self.weather_factors = {
            "clear": 1.0,
            "hazy": 1.5,
            "light_rain": 3.0, 
            "heavy_rain": 10.0,
            "fog": 20.0
        }
        
        # Initialize atmospheric turbulence state
        self._turbulence_state = 0.0
        self._last_update_time = time.time()
        
        self.logger.info(f"Free space channel initialized: {self.config.distance_km} km, "
                        f"base transmission: {self.base_transmission_prob:.4f}")
        
        if self.config.enable_atmospheric_turbulence:
            self.logger.info("Atmospheric turbulence enabled")
        if self.config.enable_eavesdropper:
            self.logger.warning(f"Eavesdropper enabled with strength {self.config.eavesdropper_strength}")

    def transmit_pulse(self, pulse: Pulse) -> Pulse:
        """
        Transmit a pulse through the quantum channel.

        Args:
            pulse: Input pulse from Alice

        Returns:
            Transmitted pulse with channel effects applied
        """
        if not isinstance(pulse, Pulse):
            raise TypeError("transmit_pulse expects a Pulse object")

        with self._lock:
            # Create output pulse (copy input)
            output_pulse = Pulse(
                polarization=pulse.polarization,
                photons=pulse.photons
            )
            
            # Apply channel effects in sequence
            output_pulse = self._apply_attenuation(output_pulse)
            output_pulse = self._apply_polarization_effects(output_pulse)
            output_pulse = self._apply_timing_effects(output_pulse)
            output_pulse = self._apply_eavesdropping(output_pulse)
            output_pulse = self._add_background_noise(output_pulse)
            
            # Update statistics
            self._update_statistics(pulse, output_pulse)
            
            self.logger.debug(f"Channel transmission: {pulse.photons} → {output_pulse.photons} photons")
            
            return output_pulse

    def _apply_attenuation(self, pulse: Pulse) -> Pulse:
        """Apply distance and weather-based attenuation."""
        # Base attenuation
        transmission_prob = self.base_transmission_prob
        
        # Weather effects
        weather_factor = self.weather_factors.get(self.config.weather_condition, 1.0)
        transmission_prob /= weather_factor
        
        # Atmospheric turbulence (time-varying)
        if self.config.enable_atmospheric_turbulence:
            turbulence_factor = self._get_turbulence_factor()
            transmission_prob *= turbulence_factor
        
        # Apply photon loss using binomial distribution
        if pulse.photons > 0:
            surviving_photons = np.random.binomial(pulse.photons, transmission_prob)
            pulse.photons = surviving_photons
        
        # Record loss for statistics
        actual_loss_db = -10 * np.log10(max(transmission_prob, 1e-10))
        self.stats.channel_losses.append(actual_loss_db)
        
        return pulse

    def _apply_polarization_effects(self, pulse: Pulse) -> Pulse:
        """Apply polarization drift and mode dispersion."""
        if not self.config.enable_polarization_drift:
            return pulse
        
        # Polarization drift over distance
        drift_angle = self.config.pol_drift_rate_deg_per_km * self.config.distance_km
        
        # Add random fluctuations
        random_drift = np.random.normal(0, drift_angle * 0.1)
        total_drift = drift_angle + random_drift
        
        # Apply drift to polarization angle
        pulse.polarization = (pulse.polarization + total_drift) % 360.0
        
        # Mode dispersion can cause small timing variations (handled in timing effects)
        
        return pulse

    def _apply_timing_effects(self, pulse: Pulse) -> Pulse:
        """Apply timing jitter and dispersion."""
        if not self.config.enable_timing_jitter:
            return pulse
        
        # Add timing jitter (this would affect detection timing in Bob)
        jitter_ps = np.random.normal(0, self.config.timing_jitter_ps)
        
        # For now, just record the jitter (actual timing would be handled by Bob's detectors)
        if abs(jitter_ps) > self.config.timing_jitter_ps:
            self.stats.timing_errors += 1
        
        return pulse

    def _apply_eavesdropping(self, pulse: Pulse) -> Pulse:
        """Apply eavesdropping effects if enabled."""
        if not self.config.enable_eavesdropper or pulse.photons == 0:
            return pulse
        
        # Eavesdropper intercepts some photons
        intercepted_photons = np.random.binomial(
            pulse.photons, 
            self.config.eavesdropper_strength
        )
        
        pulse.photons -= intercepted_photons
        self.stats.eavesdropper_intercepts += intercepted_photons
        
        if intercepted_photons > 0:
            self.logger.debug(f"Eavesdropper intercepted {intercepted_photons} photons")
        
        return pulse

    def _add_background_noise(self, pulse: Pulse) -> Pulse:
        """Add background noise photons."""
        if not self.config.enable_background_noise:
            return pulse
        
        # Calculate background counts based on detection window
        # Assuming a detection window of ~1 ns
        detection_window_s = 1e-9
        expected_background = self.config.background_photon_rate_hz * detection_window_s
        
        # Generate background counts
        background_counts = np.random.poisson(expected_background)
        
        if background_counts > 0:
            # Background photons have random polarization
            # For now, just add to photon count
            pulse.photons += background_counts
            self.stats.background_counts += background_counts
        
        return pulse

    def _get_turbulence_factor(self) -> float:
        """Get time-varying atmospheric turbulence factor."""
        current_time = time.time()
        dt = current_time - self._last_update_time
        
        # Update turbulence state (simple random walk)
        if dt > 0.001:  # Update every millisecond
            self._turbulence_state += np.random.normal(0, 0.01 * dt)
            self._turbulence_state = np.clip(self._turbulence_state, -1, 1)
            self._last_update_time = current_time
        
        # Convert to transmission factor
        turbulence_effect = 1.0 + self.config.turbulence_strength * self._turbulence_state
        return max(turbulence_effect, 0.1)  # Prevent negative transmission

    def _update_statistics(self, input_pulse: Pulse, output_pulse: Pulse) -> None:
        """Update channel statistics."""
        self.stats.total_pulses_received += 1
        self.stats.total_photons_transmitted += input_pulse.photons
        self.stats.total_photons_received += output_pulse.photons
        
        if self.stats.total_photons_transmitted > 0:
            self.stats.average_transmission_efficiency = (
                self.stats.total_photons_received / self.stats.total_photons_transmitted
            )

    def set_weather_condition(self, weather: str) -> None:
        """Change weather conditions dynamically."""
        if weather in self.weather_factors:
            self.config.weather_condition = weather
            self.logger.info(f"Weather condition changed to: {weather}")
        else:
            available = list(self.weather_factors.keys())
            raise ValueError(f"Unknown weather condition '{weather}'. Available: {available}")

    def enable_eavesdropper(self, strength: float = 0.1) -> None:
        """Enable or modify eavesdropper."""
        self.config.enable_eavesdropper = True
        self.config.eavesdropper_strength = strength
        self.logger.warning(f"Eavesdropper enabled with strength {strength}")

    def disable_eavesdropper(self) -> None:
        """Disable eavesdropper."""
        self.config.enable_eavesdropper = False
        self.logger.info("Eavesdropper disabled")

    def get_statistics(self) -> ChannelStatistics:
        """Get channel performance statistics."""
        return self.stats

    def get_channel_info(self) -> Dict[str, Any]:
        """Get comprehensive channel information."""
        return {
            "config": {
                "distance_km": self.config.distance_km,
                "base_attenuation_db_km": self.config.base_attenuation_db_km,
                "weather_condition": self.config.weather_condition,
                "turbulence_enabled": self.config.enable_atmospheric_turbulence,
                "eavesdropper_enabled": self.config.enable_eavesdropper
            },
            "performance": {
                "base_transmission_probability": self.base_transmission_prob,
                "current_weather_factor": self.weather_factors.get(self.config.weather_condition, 1.0),
                "average_transmission_efficiency": self.stats.average_transmission_efficiency
            },
            "statistics": {
                "total_pulses": self.stats.total_pulses_received,
                "total_photons_in": self.stats.total_photons_transmitted,
                "total_photons_out": self.stats.total_photons_received,
                "background_counts": self.stats.background_counts,
                "eavesdropper_intercepts": self.stats.eavesdropper_intercepts
            }
        }

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.stats = ChannelStatistics()
        self.logger.info("Channel statistics reset")

    def calibrate_channel(self) -> Dict[str, float]:
        """Perform channel calibration measurements."""
        self.logger.info("Performing channel calibration...")
        
        # Simulate calibration pulses
        calibration_pulses = 1000
        total_input = 0
        total_output = 0
        
        for _ in range(calibration_pulses):
            test_pulse = Pulse(polarization=0.0, photons=1)
            total_input += test_pulse.photons
            
            # Temporarily disable noise for calibration
            original_background = self.config.enable_background_noise
            original_eavesdropper = self.config.enable_eavesdropper
            self.config.enable_background_noise = False
            self.config.enable_eavesdropper = False
            
            result_pulse = self.transmit_pulse(test_pulse)
            total_output += result_pulse.photons
            
            # Restore settings
            self.config.enable_background_noise = original_background
            self.config.enable_eavesdropper = original_eavesdropper
        
        measured_efficiency = total_output / total_input if total_input > 0 else 0
        theoretical_efficiency = self.base_transmission_prob
        
        calibration_results = {
            "theoretical_transmission": theoretical_efficiency,
            "measured_transmission": measured_efficiency,
            "calibration_accuracy": abs(measured_efficiency - theoretical_efficiency) / theoretical_efficiency,
            "calibration_pulses": calibration_pulses
        }
        
        self.logger.info(f"Calibration complete: {measured_efficiency:.4f} measured vs {theoretical_efficiency:.4f} theoretical")
        
        return calibration_results


# Legacy class for backward compatibility
class FreeSpaceChannel(FreeSpaceChannelSimulator):
    """Legacy class for backward compatibility."""
    def __init__(self, distance: float, attenuation_coefficient: float):
        config = ChannelConfig(
            distance_km=distance,
            base_attenuation_db_km=attenuation_coefficient,
            enable_atmospheric_turbulence=False,
            enable_polarization_drift=False,
            enable_timing_jitter=False,
            enable_background_noise=False,
            enable_eavesdropper=False
        )
        super().__init__(config)
        
    def transmit(self, pulse: Pulse) -> Pulse:
        """Legacy method for backward compatibility."""
        return self.transmit_pulse(pulse)
