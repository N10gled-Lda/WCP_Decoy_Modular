"""VOA Hardware Interface."""
import logging

from src.alice.voa.voa_controller import BaseVOADriver

class VOAHardwareDriver(BaseVOADriver):
    """Interface to the physical VOA hardware."""
    def __init__(self):
        logging.info("VOA hardware interface initialized.")
        # TODO: Initialize hardware connection
        raise NotImplementedError

    def set_attenuation(self, attenuation: float):
        """Sets the VOA attenuation."""
        logging.info(f"Setting VOA hardware attenuation to {attenuation}.")
        # TODO: Implement hardware control
        raise NotImplementedError








# class BaseVOADriver(ABC):
#     """Hardware-independent VOA controller interface."""
#     # The abstract methods need to be implemented by all subclasses.

#     @abstractmethod
#     def set_attenuation(self, attenuation_db: float) -> None:
#         """Set the VOA attenuation."""
    
#     @abstractmethod
#     def get_attenuation(self) -> float:
#         """Get the current VOA attenuation."""

#     @abstractmethod
#     def get_output_from_attenuation(self):
#         """Get the output needed for a given attenuation."""

#     @abstractmethod
#     def initialize(self) -> None:
#         """Initialize the VOA hardware or simulator."""
    
#     @abstractmethod
#     def shutdown(self) -> None:
#         """Shutdown the VOA hardware or simulator."""
    
#     @abstractmethod
#     def reset(self) -> None:
#         """Reset the VOA to its default state."""