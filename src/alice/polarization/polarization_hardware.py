"""Polarization Hardware Interface."""
import logging
import serial.tools.list_ports
from typing import Any, List, Optional
from .hardware_pol.stm32_interface import STM32Interface
from .hardware_pol.stm32_interface import (
    COMMAND_POLARIZATION_SUCCESS, COMMAND_POLARIZATION_INVALID_ID,
    COMMAND_POLARIZATION_WRONG_POLARIZATIONS, COMMAND_POLARIZATION_MISMATCH_QUANTITY,
    COMMAND_POLARIZATION_OVERFLOW, COMMAND_POLARIZATION_UNKNOWN_ERROR,
)
from .polarization_base import BasePolarizationDriver, PolarizationState


class PolarizationHardware(BasePolarizationDriver):
    """
    Interface to the physical polarization hardware using an STM32 controller.
    """
    def __init__(self, com_port: str = None):
        """
        Initialize the polarization hardware.

        Args:
            com_port: The COM port for the STM32 controller.
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.com_port = com_port
        self.stm = None
        self.current_angle = 0.0
        self.current_state = PolarizationState.H # current_state not usefull - REMOVE LATER
        self.connected = False
        
        self.logger.info(f"Polarization hardware interface initialized for port {com_port}")

    # def _handle_connection(self):
    #     """Handle the STM32 connection event."""
    #     self.logger.info(f"STM32 connected on port {self.com_port}")
    #     self.connected = True

    def initialize(self) -> None:
        """Initialize the STM32 hardware connection."""
        try:
            if self.connected:
                self.logger.warning("Already connected to STM32. Reinitializing...")
                self.shutdown()
            self.stm = STM32Interface(port=self.com_port)
            # self.stm.on_connected = self._handle_connection
            # self.stm.on_connected = self._handle_stm32_connected
            # self.stm.on_available = self._handle_stm32_available
            self.stm.on_polarization_status = self._handle_polarization_status
            self.stm.start()
            self.stm.connect()
            
            if not self.stm.connected:
                raise ConnectionError(f"Failed to connect to STM32 on port {self.com_port}")
            
            self.connected = True
            self.logger.info(f"Successfully connected to STM32 on port {self.com_port}")
            
            # Set to initial state
            self.set_polarization_state(PolarizationState.H)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize polarization hardware: {e}")
            raise

    def set_polarization_angle(self, angle_degrees: float) -> None:
        """
        Set the polarization angle in degrees.
        
        Args:
            angle_degrees: Target polarization angle (0-180°)
        """
        if not self.connected:
            raise RuntimeError("Hardware not connected. Call initialize() first.")
        
        # Normalize angle to [0, 180) range
        normalized_angle = angle_degrees % 180.0
        
        # Map angle to polarization number for STM32
        polarization_number = self._map_angle_to_polarization_number(normalized_angle)
        
        try:
            if self.stm is None:
                raise RuntimeError("STM32 interface not initialized.")
            # Send the polarization number to the STM32
            self.logger.debug(f"Sending polarization number {polarization_number} for normalized angle {normalized_angle}° and angle {angle_degrees}°")
            self.success_send_pol = self.stm.send_polarization_numbers([polarization_number])
            if not self.success_send_pol:
                self.logger.error(f"Failed to send polarization number {polarization_number} to STM32")
                raise RuntimeError(f"Failed to send polarization number {polarization_number} to STM32")
            else:
                self.logger.debug(f"Successfully set polarization angle to {angle_degrees}° (normalized: {normalized_angle}°) (number: {polarization_number})")
            self.current_angle = normalized_angle
            self.current_state = self._angle_to_state(normalized_angle)
            
        except Exception as e:
            self.logger.error(f"Failed to set polarization angle {normalized_angle}°: {e}")
            raise

    def set_polarization_state(self, state: PolarizationState) -> None:
        """
        Set the polarization to a specific BB84 state.
        
        Args:
            state: Target polarization state
        """
        angle = state.angle_degrees
        self.set_polarization_angle(angle)
        self.current_state = state
        
        self.logger.debug(f"Set polarization state: {state} ({angle}°)")

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

    def _map_angle_to_polarization_number(self, angle_degrees: float) -> int:
        """
        Map polarization angle to STM32 polarization number.
        
        Args:
            angle_degrees: Polarization angle in degrees
            
        Returns:
            Polarization number for STM32 (0-3)
        """
        # Round to nearest standard angle
        if abs(angle_degrees - 0.0) < 22.5:
            return 0  # Horizontal (0°)
        elif abs(angle_degrees - 45.0) < 22.5:
            return 2  # Diagonal (45°)
        elif abs(angle_degrees - 90.0) < 22.5:
            return 1  # Vertical (90°)
        elif abs(angle_degrees - 135.0) < 22.5:
            return 3  # Anti-diagonal (135°)
        else:
            # Default to closest standard angle
            angles = [0.0, 45.0, 90.0, 135.0]
            numbers = [0, 2, 1, 3]
            
            min_diff = float('inf')
            closest_number = 0
            
            for angle, number in zip(angles, numbers):
                diff = min(abs(angle_degrees - angle), 
                          abs(angle_degrees - angle - 180),
                          abs(angle_degrees - angle + 180))
                if diff < min_diff:
                    min_diff = diff
                    closest_number = number
            
            return closest_number

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

    def _map_to_polarization_number(self, basis: str, bit: int) -> int:
        """
        Maps basis and bit to a polarization number for the STM32.
        Legacy method for compatibility.
        
        Args:
            basis: Basis ('Z' or 'X')
            bit: Bit value (0 or 1)
            
        Returns:
            Polarization number (0-3)
        """
        if basis == 'Z':
            return 0 if bit == 0 else 1  # 0 -> 0°, 1 -> 90°
        elif basis == 'X':
            return 2 if bit == 0 else 3  # 0 -> 45°, 1 -> 135°
        else:
            raise ValueError(f"Unknown basis {basis}")

    def encode_bit(self, bit: int, basis: str | int = 'Z') -> None:
        """
        Legacy method: Encode a single bit onto the polarization.
        
        Args:
            bit: The bit to encode (0 or 1)
            basis: The basis to use ('Z', 'X', 0, or 1)
        """
        if bit not in [0, 1]:
            raise ValueError("Bit must be 0 or 1")
        
        # Map integer basis to string representation
        if basis == 0:
            basis = 'Z'
        elif basis == 1:
            basis = 'X'
        
        if basis not in ['Z', 'X']:
            raise ValueError("Basis must be 'Z', 'X', 0, or 1")
        
        polarization_number = self._map_to_polarization_number(basis, bit)
        
        try:
            self.stm.send_polarization_numbers([polarization_number])
            
            # Update internal state
            if basis == 'Z':
                self.current_angle = 0.0 if bit == 0 else 90.0
                self.current_state = PolarizationState.H if bit == 0 else PolarizationState.V
            else:  # basis == 'X'
                self.current_angle = 45.0 if bit == 0 else 135.0
                self.current_state = PolarizationState.D if bit == 0 else PolarizationState.A
            
            self.logger.info(f"Encoded bit {bit} in basis {basis} to polarization {polarization_number}")
            
        except Exception as e:
            self.logger.error(f"Failed to encode bit {bit} in basis {basis}: {e}")
            raise

    def encode_bits(self, bits: List[int], bases: List[str | int]) -> None:
        """
        Legacy method: Encode a sequence of bits onto the polarization.
        
        Args:
            bits: The sequence of bits to encode
            bases: The sequence of bases to use
        """
            
        if len(bits) != len(bases):
            raise ValueError("Length of bits and bases must be the same.")
        
        # Map integer basis to string representation
        if basis == 0:
            basis = 'Z'
        elif basis == 1:
            basis = 'X'
        
        polarization_numbers = [self._map_to_polarization_number(b, bit) 
                               for b, bit in zip(bases, bits)]
        
        try:
            self.stm.send_polarization_numbers(polarization_numbers)
            self.logger.info(f"Encoded sequence of {len(bits)} bits")
            
        except Exception as e:
            self.logger.error(f"Failed to encode bit sequence: {e}")
            raise

    def shutdown(self) -> None:
        """Shutdown the hardware connection."""
        if self.stm:
            try:
                self.stm.stop()
                self.connected = False
                self.logger.info("STM32 connection closed")
            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}")

    def __del__(self):
        """Destructor to ensure the serial connection is closed."""
        self.shutdown()

    @staticmethod
    def get_available_com_ports() -> List[str]:
        """Returns a list of available COM ports."""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]

    def is_connected(self) -> bool:
        """Check if hardware is connected."""
        return self.connected and self.stm is not None and self.stm.connected

    def set_com_port(self, com_port: str) -> None:
        """
        Set the COM port for the STM32 controller.
        
        Args:
            com_port: The new COM port to set
        """
        self.com_port = com_port
        self.logger.info(f"COM port set to {com_port}")
        
        # Reinitialize if already connected
        if self.connected and self.stm is not None:
            self.shutdown()
            self.initialize()



    def _handle_stm32_connected(self):
        """Callback for STM32 connection established."""
        self.is_connected = True
        self.logger.info("✓ STM32 interface connected")
    
    def _handle_stm32_available(self):
        """Callback for STM32 becoming available."""
        self.logger.info("STM32 interface available")

    def _handle_polarization_status(self, status_code: int, data: Any = None):
        """
        Callback for polarization status updates from STM32.
        
        Args:
            status_code: Status code from STM32
            data: Additional status data
        """
        if status_code == COMMAND_POLARIZATION_SUCCESS:
            self.logger.debug("Polarization command successful")
        elif status_code == COMMAND_POLARIZATION_INVALID_ID:
            self.logger.error("⚠️  Invalid polarization ID")
            self.last_error = "Invalid polarization ID"
        elif status_code == COMMAND_POLARIZATION_WRONG_POLARIZATIONS:
            self.logger.error("⚠️  Wrong polarization configuration")
            self.last_error = "Wrong polarization configuration"
        elif status_code == COMMAND_POLARIZATION_MISMATCH_QUANTITY:
            self.logger.error("⚠️  Polarization quantity mismatch")
            self.last_error = "Polarization quantity mismatch"
        elif status_code == COMMAND_POLARIZATION_OVERFLOW:
            self.logger.error("⚠️  Polarization buffer overflow")
            self.last_error = "Polarization buffer overflow"
        elif status_code == COMMAND_POLARIZATION_UNKNOWN_ERROR:
            self.logger.error("⚠️  Unknown polarization error")
            self.last_error = "Unknown polarization error"
        else:
            self.logger.warning(f"⚠️  Unknown status code: {status_code}")
            self.last_error = f"Unknown status code: {status_code}"
    