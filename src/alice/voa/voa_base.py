"""Base classes for VOA (Variable Optical Attenuator) control."""
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseVOADriver(ABC):
    """Hardware-independent VOA driver interface."""

    @abstractmethod
    def set_attenuation(self, attenuation_db: float) -> None:
        """Set the VOA attenuation in dB."""
        pass
    
    @abstractmethod
    def get_attenuation(self) -> float:
        """Get the current VOA attenuation in dB."""
        pass

    @abstractmethod
    def get_output_from_attenuation(self) -> float:
        """Get the output power factor for the current attenuation."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the VOA hardware or simulator."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the VOA hardware or simulator."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the VOA to its default state."""
        pass

    def get_status(self) -> Dict[str, Any]:
        """Get status information. Override in subclasses for specific status."""
        return {
            "driver_type": self.__class__.__name__,
            "attenuation_db": self.get_attenuation(),
            "initialized": True,
            "active": True
        }