"""Free Space Channel Simulator."""
import logging
import numpy as np
from ..utils.data_structures import Pulse

class FreeSpaceChannel:
    """
    Simulates the quantum channel, including transmission loss.
    """
    def __init__(self, distance: float, attenuation_coefficient: float):
        """
        Initializes the free space channel.

        Args:
            distance: The distance of the channel in km.
            attenuation_coefficient: The attenuation coefficient of the channel in dB/km.
        """
        self.distance = distance
        self.attenuation_coefficient = attenuation_coefficient
        self.transmission_probability = 10**(-self.attenuation_coefficient * self.distance / 10)
        logging.info(f"Free space channel initialized with distance {distance} km and attenuation {attenuation_coefficient} dB/km.")
        logging.info(f"Transmission probability: {self.transmission_probability:.4f}")
        logging.warning("CONFIRM MODEL: The channel is modeled with a constant attenuation coefficient. This does not account for atmospheric turbulence or other environmental effects. Please confirm this is appropriate.")

    def transmit(self, pulse: Pulse) -> Pulse:
        """
        Transmits a pulse through the channel, simulating photon loss.

        Args:
            pulse: The pulse to transmit.

        Returns:
            The pulse after transmission, with potentially fewer photons.
        """
        if not isinstance(pulse, Pulse) or not hasattr(pulse, 'photons'):
            raise TypeError("transmit expects a Pulse object with a 'photons' attribute.")

        photons_out = np.random.binomial(pulse.photons, self.transmission_probability)
        
        logging.debug(f"Transmitted pulse with {pulse.photons} photons, {photons_out} photons survived.")
        
        transmitted_pulse = Pulse(
            intensity=pulse.intensity,
            polarization=pulse.polarization,
            basis=pulse.basis,
            timestamp=pulse.timestamp,
            photons=photons_out
        )
        return transmitted_pulse
