"""VOA Simulator."""
import logging
from queue import Queue
import queue
from functools import wraps
from typing import Optional, Dict, Any

import numpy as np

from src.alice.voa.voa_base import BaseVOADriver
from src.utils.data_structures import Pulse, LaserInfo

def ensure_on(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self._is_on:
            self.logger.error("VOA is not turned on.")
            raise RuntimeError("VOA must be initialized first.")
        return fn(self, *args, **kwargs)
    return wrapper

class VOASimulator(BaseVOADriver):
    """Simulates the behavior of a VOA."""
    
    def __init__(self, pulses_queue: Queue[Pulse], attenuated_pulses_queue: Queue[Pulse], 
                 laser_info: Optional[LaserInfo] = None):
        """Initialize the VOA simulator."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.pulses_queue = pulses_queue
        self.attenuated_pulses_queue = attenuated_pulses_queue
        self.laser_info = laser_info if laser_info else LaserInfo(
            max_power_mW=100.0,
            pulse_width_fwhm_ns=10.0,
            central_wavelength_nm=1550.0
        )

        self.attenuation_db = 0.0
        self._is_on = False

        self.logger.info("VOA simulator initialized.")

    @ensure_on
    def set_attenuation(self, attenuation: float) -> None:
        """Sets the VOA attenuation."""
        self.logger.debug(f"Setting VOA attenuation to {attenuation:f} dB")
        self.attenuation_db = attenuation

    def get_attenuation(self) -> float:
        """Returns the current VOA attenuation."""
        self.logger.debug(f"Getting VOA attenuation: {self.attenuation_db:.2f} dB")
        return self.attenuation_db
    
    def get_output_from_attenuation(self) -> float:
        """Returns the output power factor for the current attenuation."""
        factor = 10 ** (-self.attenuation_db / 10)
        out = 1.0 * factor
        self.logger.debug(f"VOA attenuation={self.attenuation_db:.2f} dB → output ={out:.6f}")
        return out

    @ensure_on
    def apply_attenuation_queue(self):
        """Attenuate all pulses in the queue based on the current attenuation."""
        # Drain all available pulses
        drained = []
        while True:
            try:
                drained.append(self.pulses_queue.get_nowait())
            except queue.Empty:
                break

        if not drained:
            self.logger.warning("Pulses queue is empty. No pulses to attenuate.")
            return

        if len(drained) > 1:
            self.logger.warning(f"Multiple pulses in the queue ({len(drained)}). Attenuating all pulses.")

        for pulse in drained:
            attenuated_pulse = self._apply_attenuation_pulse(pulse)
            self.attenuated_pulses_queue.put(attenuated_pulse)

    
        # while True:
        #     try:
        #         # Get pulse and attenuation from input queue
        #         item = input_queue.get(timeout=1.0)
        #         if item is None:  # Shutdown signal
        #             break
                
        #         pulse, attenuation_dB = item
                
        #         # Apply attenuation
        #         attenuated_pulse = self.apply_attenuation(pulse, attenuation_dB)
                
        #         # Put processed pulse in output queue
        #         output_queue.put(attenuated_pulse)
                
        #         input_queue.task_done()
                
        #     except:
        #         # Queue timeout or other error
        #         continue

    @ensure_on
    def _apply_attenuation_pulse(self, pulse: Pulse) -> Pulse:
        """Attenuate a pulse based on the current attenuation."""
        # Simulate attenuation by applying the attenuation into the number of photons of the pulse
        attenuation_linear = 10 ** (-self.attenuation_db / 10)

        # Apply the attenuation factor to the pulse's photon count
        self.logger.debug(f"Pulse before attenuation: {pulse.photons / attenuation_linear} photons")

        attenuated_pulse = pulse.copy()
        # attenuated_pulse.photons *= attenuation_linear
        attenuated_pulse.photons = round(np.random.binomial(n=pulse.photons, p=attenuation_linear))

        self.logger.debug(f"Attenuated pulse: {attenuated_pulse} (given attenuation {self.attenuation_db} dB)")

        return attenuated_pulse

    def initialize(self) -> None:
        """Initialize the VOA simulator."""
        self._is_on = True
        self.attenuation_db = 0.0
        self.logger.info("VOA simulator initialized.")

    def shutdown(self) -> None:
        """Shutdown the VOA simulator."""
        self._is_on = False
        self.logger.info("VOA simulator shut down.")
        
    def reset(self) -> None:
        """Reset attenuation, clear both queues and power setting."""
        self.attenuation_db = 0.0
        
        # Clear queues if they exist
        if self.pulses_queue:
            with self.pulses_queue.mutex:
                self.pulses_queue.queue.clear()
        if self.attenuated_pulses_queue:
            with self.attenuated_pulses_queue.mutex:
                self.attenuated_pulses_queue.queue.clear()
                
        self._is_on = False
        self.logger.info("VOA simulator reset.")

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the VOA simulator."""
        return {
            "driver_type": self.__class__.__name__,
            "attenuation_db": self.attenuation_db,
            "initialized": self._is_on,
            "active": self._is_on,
            "queue_sizes": {
                "input": self.pulses_queue.qsize() if self.pulses_queue else 0,
                "output": self.attenuated_pulses_queue.qsize() if self.attenuated_pulses_queue else 0
            }
        }