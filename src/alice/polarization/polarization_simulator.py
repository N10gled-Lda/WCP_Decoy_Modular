"""Polarization Simulator."""
import logging
import math
from queue import Queue
import queue
from functools import wraps
from typing import List, Optional, Tuple
from dataclasses import dataclass

from src.utils.data_structures import Pulse, LaserInfo
from .polarization_base import BasePolarizationDriver, PolarizationState


def ensure_on(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self._is_on:
            self.logger.error("Polarization controller is not turned on.")
            raise RuntimeError("Polarization controller must be initialized first.")
        return fn(self, *args, **kwargs)
    return wrapper


class PolarizationSimulator(BasePolarizationDriver):
    """Simulates the polarization controller behavior."""
    
    def __init__(self, pulses_queue: Queue[Pulse], polarized_pulses_queue: Queue[Pulse], laser_info: LaserInfo):
        """Initialize the polarization simulator."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.pulses_queue = pulses_queue
        self.polarized_pulses_queue = polarized_pulses_queue
        self.laser_info = laser_info
        
        # Current state
        self.current_angle = 0.0
        self.current_state = PolarizationState.H
        self._is_on = False
        
        # Performance parameters
        self.settling_time = 0.001  # 1ms settling time
        self.angle_tolerance = 0.1  # 0.1 degree tolerance
        
        self.logger.info("Polarization simulator initialized")

    def set_polarization_angle(self, angle_degrees: float) -> None:
        """
        Set the polarization angle in degrees.
        
        Args:
            angle_degrees: Target polarization angle (0-180°)
        """
        # Normalize angle to [0, 180) range
        normalized_angle = angle_degrees % 180.0
        
        # Determine corresponding polarization state
        state = self._angle_to_state(normalized_angle)
        
        # Update current state
        self.current_angle = normalized_angle
        self.current_state = state
        
        self.logger.debug(f"Set polarization angle: {normalized_angle}° → {state}")

    def set_polarization_state(self, state: PolarizationState) -> None:
        """
        Set the polarization to a specific BB84 state.
        
        Args:
            state: Target polarization state
        """
        angle = state.angle_degrees
        
        self.current_angle = angle
        self.current_state = state
        
        self.logger.debug(f"Set polarization state: {state} ({angle}°)")

    @ensure_on
    def _apply_polarization_pulse(self, pulse: Pulse) -> Pulse:
        """Apply polarization to a pulse based on the current state."""
        # Apply current polarization state to pulse
        processed_pulse = Pulse(
            # timestamp=pulse.timestamp,
            # intensity=pulse.intensity,
            # phase=pulse.phase,
            # wavelength=pulse.wavelength,
            photons=pulse.photons,
            polarization_angle=self.current_angle,
            # basis=self.current_state.basis,
            # bit_value=self.current_state.bit_value
        )
        
        self.logger.info(f"Applied polarization: {processed_pulse}")
        self.logger.debug(f"Pulse polarized with {self.current_state} ({self.current_angle}°)")
        return processed_pulse

    @ensure_on
    def apply_polarization_queue(self):
        """Apply polarization to all pulses in the queue based on the current state."""
        # Drain all available pulses
        drained = []
        while True:
            try:
                drained.append(self.pulses_queue.get_nowait())
            except queue.Empty:
                break

        if not drained:
            self.logger.warning("Pulses queue is empty. No pulses to polarize.")
            return

        if len(drained) > 1:
            self.logger.warning(f"Multiple pulses in the queue ({len(drained)}). Polarizing all pulses.")

        for pulse in drained:
            polarized_pulse = self._apply_polarization_pulse(pulse)
            self.polarized_pulses_queue.put(polarized_pulse)

        self.logger.info("All pulses in the queue have been polarized.")

    def _angle_to_state(self, angle_degrees: float) -> PolarizationState:
        """
        Convert angle to closest BB84 polarization state.
        
        Args:
            angle_degrees: Polarization angle
            
        Returns:
            Closest BB84 state
        """
        # Find closest standard angle
        standard_angles = {
            0.0: PolarizationState.H,
            45.0: PolarizationState.D,
            90.0: PolarizationState.V,
            135.0: PolarizationState.A
        }
        
        min_diff = float('inf')
        closest_state = PolarizationState.H
        
        for std_angle, state in standard_angles.items():
            diff = min(abs(angle_degrees - std_angle), 
                      abs(angle_degrees - std_angle - 180),
                      abs(angle_degrees - std_angle + 180))
            if diff < min_diff:
                min_diff = diff
                closest_state = state
        
        return closest_state

    def get_current_polarization(self) -> float:
        """
        Get the current polarization angle.
        
        Returns:
            Current polarization angle in degrees
        """
        return self.current_angle

    def get_current_state(self) -> PolarizationState:
        """
        Get the current polarization state.
        
        Returns:
            Current BB84 polarization state
        """
        return self.current_state

    def get_jones_vector(self) -> List[complex]:
        """
        Get the Jones vector for current polarization.
        
        Returns:
            Jones vector [E_H, E_V]
        """
        phi = math.radians(self.current_angle)
        return [
            complex(math.cos(phi), 0),  # Horizontal component
            complex(math.sin(phi), 0)   # Vertical component
        ]

    def process_pulse(self, pulse: Pulse) -> Pulse:
        """
        Process a single pulse through the polarization controller.
        Applies polarization transformation to the pulse.
        
        Args:
            pulse: Input pulse
            
        Returns:
            Pulse with polarization applied
        """
        return self._apply_polarization_pulse(pulse)

    def get_output_from_polarization(self):
        """Returns the output state for the current polarization setting."""
        self.logger.info(
            f"Polarization state={self.current_state} angle={self.current_angle}° → "
            f"basis={self.current_state.basis} bit={self.current_state.bit_value}"
        )
        return {
            'state': self.current_state,
            'angle_degrees': self.current_angle,
            'basis': self.current_state.basis,
            'bit_value': self.current_state.bit_value,
            'jones_vector': self.get_jones_vector()
        }

    def simulate_rotation_time(self, target_angle: float) -> float:
        """
        Simulate the time required to rotate to target angle.
        
        Args:
            target_angle: Target polarization angle
            
        Returns:
            Estimated rotation time in seconds
        """
        angle_change = abs(target_angle - self.current_angle)
        # Assume 100°/s rotation speed
        rotation_speed = 100.0  # degrees per second
        rotation_time = angle_change / rotation_speed + self.settling_time
        self.logger.debug(f"Simulated rotation time to {target_angle}°: {rotation_time:.3f}s -> rotation_speed {rotation_speed}°/s and settling_time {self.settling_time:.3f}s")
        
        return rotation_time

    def initialize(self) -> None:
        """Initialize the polarization simulator."""
        self._is_on = True
        self.current_angle = 0.0
        self.current_state = PolarizationState.H
        self.logger.info("Polarization simulator initialized to horizontal (0°)")

    def shutdown(self) -> None:
        """Shutdown the polarization simulator."""
        self._is_on = False
        self.logger.info("Polarization simulator shutdown")

    def reset(self) -> None:
        """Reset polarization to default state and clear queues."""
        self.current_angle = 0.0
        self.current_state = PolarizationState.H
        with self.pulses_queue.mutex:
            self.pulses_queue.queue.clear()
        with self.polarized_pulses_queue.mutex:
            self.polarized_pulses_queue.queue.clear()
        self._is_on = False
        self.logger.info("Polarization simulator reset.")

    def __str__(self) -> str:
        """String representation of simulator state."""
        return (f"PolarizationSimulator(angle={self.current_angle}°, "
                f"state={self.current_state}, is_on={self._is_on})")

