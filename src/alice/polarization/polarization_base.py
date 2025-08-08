"""Base classes and enums for polarization control."""
import math
import cmath
from abc import ABC, abstractmethod
from turtle import st
from typing import List
from enum import IntEnum

from src.utils.data_structures import Basis, Bit


class PolarizationState(IntEnum):
    """BB84 polarization states."""
    H = 0      # Horizontal (0°)
    V = 1      # Vertical (90°) 
    D = 2      # Diagonal (45°)
    A = 3      # Anti-diagonal (135°)

    def __str__(self):
        return self.name.lower()

    @property
    def angle_degrees(self) -> float:
        """Get the polarization angle in degrees."""
        angle_map = {
            PolarizationState.H: 0.0,    # Horizontal
            PolarizationState.V: 90.0,   # Vertical
            PolarizationState.D: 45.0,   # Diagonal
            PolarizationState.A: 135.0   # Anti-diagonal
        }
        return angle_map[self]

    @property
    def basis(self) -> Basis:
        """Get the measurement basis for this polarization."""
        if self in [PolarizationState.H, PolarizationState.V]:
            return Basis.Z  # Rectilinear basis
        else:
            return Basis.X  # Diagonal basis

    @property
    def bit_value(self) -> Bit:
        """Get the bit value encoded in this polarization."""
        if self in [PolarizationState.H, PolarizationState.D]:
            return Bit.ZERO
        else:
            return Bit.ONE


class JonesVector:
    """Jones vector representation of polarization state."""
    
    def __init__(self, e_h: complex, e_v: complex):
        """
        Initialize Jones vector.
        
        Args:
            e_h: Horizontal component
            e_v: Vertical component
        """
        self.e_h = e_h
        self.e_v = e_v
    
    @staticmethod
    def from_angle(angle_degrees: float, phase: float = 0.0) -> 'JonesVector':
        """Create Jones vector from polarization angle."""
        phi = math.radians(angle_degrees)
        return JonesVector(
            e_h=math.cos(phi) * cmath.exp(1j * phase),
            e_v=math.sin(phi) * cmath.exp(1j * phase)
        )
    
    def to_list(self) -> List[complex]:
        """Convert to list format [E_H, E_V]."""
        return [self.e_h, self.e_v]
    
    @property
    def intensity(self) -> float:
        """Calculate intensity (|E|²)."""
        return abs(self.e_h)**2 + abs(self.e_v)**2


class BasePolarizationDriver(ABC):
    """Hardware-independent polarization driver interface."""

    @abstractmethod
    def set_polarization_angle(self, angle_degrees: float) -> None:
        """Set the polarization angle in degrees."""
        pass

    @abstractmethod
    def set_polarization_state(self, state: PolarizationState) -> None:
        """Set the polarization to a specific BB84 state."""
        pass

    @abstractmethod
    def get_current_polarization(self) -> float:
        """Get the current polarization angle."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the polarization hardware/simulator."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the polarization hardware/simulator."""
        pass
