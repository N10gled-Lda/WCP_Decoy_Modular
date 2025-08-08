"""Laser Simulator."""
import logging
from queue import Queue
import threading
import time
import numpy as np

from src.alice.laser.laser_controller import BaseLaserDriver
from src.utils.data_structures import LaserInfo, Pulse


class SimulatedLaserDriver(BaseLaserDriver):
    def __init__(self, pulses_queue: Queue[Pulse], laser_info: LaserInfo):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.pulses_queue = pulses_queue
        self.laser_info = laser_info
        self.default_current_power_mW = laser_info.max_power_mW
        self.default_pulse_width_fwhm_ns = laser_info.pulse_width_fwhm_ns
        self.default_central_wavelength_nm = laser_info.central_wavelength_nm

        self.is_on = False
        self.is_arm = False
        self.is_fire = False

        self.logger.info("Simulated laser driver initialized.")
        # TODO: Implement noise induced by the laser simulator

    def turn_on(self):
        """Turn on the simulated laser."""
        if self.is_on:
            self.logger.warning("Simulated laser is already on.")
            return
        
        self.is_on = True
        self.logger.info("Simulated laser turned on.")

    def turn_off(self):
        """Turn off the simulated laser."""
        if not self.is_on:
            self.logger.warning("Simulated laser is already off.")
            return
        
        self.is_on = False
        self.logger.info("Simulated laser turned off.")

    def stop(self):
        if self.is_arm:
            self.is_arm = False
            self.arm_thread.join()
            self.logger.info("Simulated laser arm stopped.")
        if self.is_fire:
            self.is_fire = False
            self.fire_thread.join()
            self.logger.info("Simulated laser fire stopped.")
        

    # ----------------------------------------- arm
    def arm(self, repetition_rate_hz: float):
        """Prepare the source for a sequence of pulses."""
        if not self.is_on:
            self.logger.error("Cannot arm the laser when it is off.")
            return
        self.logger.info(f"Arming simulated laser with repetition rate: {repetition_rate_hz} Hz until stopped.")

        self.rep_period = 1 / repetition_rate_hz

        # Put pulses into the queue based on the repetition rate in a separate thread until stopped
        self.is_arm = True
        self.arm_thread = threading.Thread(target=self._arm_laser, args=(self.rep_period,))
        self.arm_thread.start()

    def _arm_laser(self, rep_period: float):
        """
        Arms the laser by putting pulses into the queue at the specified repetition period.
        """
        while self.is_arm:
            self.pulses_queue.put(self._generate_single_pulse())
            time.sleep(rep_period)


    # ----------------------------------------- fire
    def fire(self, pattern: list[float]) -> None:
        """
        Emit (or pretend to emit) one frame's worth of pulses.
        pattern: list of times between each pulse in seconds.
        """
        if not self.is_on:
            self.logger.error("Cannot fire pulses when the laser is off.")
            return
        
        # Fire the pattern in a separate thread so the program can continue
        self.is_fire = True
        self.fire_thread = threading.Thread(target=self._fire_pattern, args=(pattern,))
        self.fire_thread.start()
        
        self.logger.info(f"Started firing {len(pattern)} pulses with pattern: {pattern} in separate thread.")

    def _fire_pattern(self, pattern: list[float]):
        """
        Fires the pulse pattern in a separate thread.
        """
        for time_delay in pattern:
            if not self.is_on:
                self.logger.error("Cannot fire pulses when the laser is off.")
                return
            if not self.is_fire:
                self.logger.error("Cannot fire pulses when the laser is not armed.")
                return
            pulse = self._generate_single_pulse()
            self.pulses_queue.put(pulse)
            time.sleep(time_delay)


    # ----------------------------------------- fire_single_pulse
    def fire_single_pulse(self, power: float = None, linewidth: float = None, wavelength: float = None):
        """
        Emit a single pulse.
        """
        if not self.is_on:
            self.logger.error("Cannot fire a single pulse when the laser is off.")
            return
        
        if power is not None: self.default_current_power_mW = power
        if linewidth is not None: self.default_pulse_width_fwhm_ns = linewidth
        if wavelength is not None: self.default_central_wavelength_nm = wavelength

        pulse = self._generate_single_pulse()
        self.pulses_queue.put(pulse)
        
        self.logger.info(f"Fired single pulse: {pulse}.")


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
        polarization = np.random.choice([0,45,90,135], p=[0.25, 0.25, 0.25, 0.25])  # Randomly choose polarization angle
        # TODO - Implement a more sophisticated polarization model from laser_info
        
        logging.debug(f"Generated pulse with {photon_number} photons and polarization {polarization} degrees.")
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