"""Polarization Hardware Interface."""
import logging
import serial.tools.list_ports
import time
from typing import Any, List, Optional
from .hardware_pol.stm32_interface import STM32Interface
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
        self.available = False
        
        self.logger.info(f"Polarization hardware interface initialized for port {com_port}")

    # def _handle_connection(self):
    #     """Handle the STM32 connection event."""
    #     self.logger.info(f"STM32 connected on port {self.com_port}")
    #     self.connected = True

    def initialize(self) -> bool:
        """Initialize the STM32 hardware connection."""
        try:
            if self.connected:
                self.logger.warning("Already connected to STM32. Reinitializing...")
                self.shutdown()
                
            if not self.com_port:
                raise ValueError("COM port not specified")
                
            self.stm = STM32Interface(self.com_port)
            
            # Set up callbacks for the new interface
            self.stm.on_connected = self._handle_stm32_connected
            self.stm.on_available = self._handle_stm32_available
            
            # Start the interface
            self.stm.start()
            self.stm.connect()
            
            # Wait for connection with timeout
            connection_timeout = 10  # seconds
            for i in range(connection_timeout):
                if self.stm.connected:
                    break
                time.sleep(1)
                self.logger.debug(f"Waiting for connection... ({i+1}/{connection_timeout})")
            
            if not self.stm.connected:
                raise ConnectionError(f"Failed to connect to STM32 on port {self.com_port}")
            
            self.connected = True
            self.available = True
            self.logger.info(f"Successfully connected to STM32 on port {self.com_port}")
            
            # Set to initial state
            # self.set_polarization_state(PolarizationState.H)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize polarization hardware: {e}")
            self.connected = False
            self.available = False
            return False

    def shutdown(self) -> None:
        """Shutdown the hardware connection."""
        if self.stm:
            try:
                self.stm.stop()
                self.connected = False
                self.logger.info("STM32 connection closed")
            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}")

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
            # Check and wait if available
            if not self.available:
                self.logger.warning("STM32 interface not available. Waiting for availability...")
                start_time = time.time()
                while not self.stm.available:
                    if time.time() - start_time > 5:
                        self.logger.error("STM32 interface did not become available within 5 seconds")
                        raise TimeoutError("STM32 interface not available within the timeout period")
                    time.sleep(0.01)
            # Send the polarization number to the STM32
            self.logger.debug(f"Sending polarization number {polarization_number} for normalized angle {normalized_angle}° and angle {angle_degrees}°")
            self.success_send_pol = self.stm.send_cmd_polarization_numbers([polarization_number])
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

    def set_polarization_device(self, device: int) -> bool:
        """
        Set the polarization device type.
        
        Args:
            device: Device type (1 = Linear Polarizer, 2 = Half Wave Plate)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            self.logger.error("Hardware not connected. Call initialize() first.")
            return False
            
        if device not in [1, 2]:
            self.logger.error("Invalid device type. Must be 1 (Linear Polarizer) or 2 (Half Wave Plate)")
            return False
            
        try:
            success = self.stm.send_cmd_polarization_device(device)
            if success:
                device_name = "Linear Polarizer" if device == 1 else "Half Wave Plate"
                self.logger.info(f"Set polarization device to {device_name}")
                return True
            else:
                self.logger.error(f"Failed to set polarization device to {device}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to set polarization device: {e}")
            return False

    def set_angle_direct(self, angle: int, is_offset: bool = False) -> bool:
        """
        Set angle directly using STM32 angle commands.
        
        Args:
            angle: Angle in degrees (0-360)
            is_offset: If True, set as offset; if False, set as absolute angle
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            self.logger.error("Hardware not connected. Call initialize() first.")
            return False
            
        if not (0 <= angle <= 360):
            self.logger.error("Angle must be between 0 and 360 degrees")
            return False
            
        try:
            success = self.stm.send_cmd_set_angle(angle, is_offset=is_offset)
            if success:
                angle_type = "offset" if is_offset else "absolute"
                self.logger.info(f"Set {angle_type} angle to {angle}°")
                if not is_offset:
                    # Update internal state for absolute angles
                    self.current_angle = angle % 180.0  # Normalize to polarization range
                    self.current_state = self._angle_to_state(self.current_angle)
                return True
            else:
                self.logger.error(f"Failed to set angle to {angle}°")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to set angle: {e}")
            return False

    def set_stepper_frequency(self, frequency: int) -> bool:
        """
        Set the stepper motor frequency. This frequency controls the speed of the stepper motor to rotate the target angle.
        
        Args:
            frequency: Frequency in Hz (1-1000)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            self.logger.error("Hardware not connected. Call initialize() first.")
            return False
            
        if not (1 <= frequency <= 1000):
            self.logger.error("Stepper frequency must be between 1 and 1000 Hz")
            return False
            
        try:
            success = self.stm.send_cmd_set_frequency(frequency, is_stepper=True)
            if success:
                self.logger.info(f"Set stepper motor frequency to {frequency} Hz")
                return True
            else:
                self.logger.error(f"Failed to set stepper frequency to {frequency} Hz")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to set stepper frequency: {e}")
            return False

    def set_operation_period(self, period: int) -> bool:
        """
        Set the operation period. This period controls the time between each polarization operation (not including the rotation time).
        
        Args:
            period: Period in milliseconds (1-60000)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            self.logger.error("Hardware not connected. Call initialize() first.")
            return False
            
        if not (1 <= period <= 60000):
            self.logger.error("Operation period must be between 1 and 60000 ms")
            return False
            
        try:
            success = self.stm.send_cmd_set_frequency(period, is_stepper=False)
            if success:
                self.logger.info(f"Set operation period to {period} ms")
                return True
            else:
                self.logger.error(f"Failed to set operation period to {period} ms")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to set operation period: {e}")
            return False

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
            return 1  # Diagonal (45°)
        elif abs(angle_degrees - 90.0) < 22.5:
            return 2  # Vertical (90°)
        elif abs(angle_degrees - 135.0) < 22.5:
            return 3  # Anti-diagonal (135°)
        else:
            # Default to closest standard angle
            angles = [0.0, 45.0, 90.0, 135.0]
            numbers = [0, 1, 2, 3]
            
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
            return 0 if bit == 0 else 2  # 0 -> 0°, 2 -> 90°
        elif basis == 'X':
            return 1 if bit == 0 else 3  # 1 -> 45°, 3 -> 135°
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
            # Check and wait if available
            if not self.available:
                self.logger.warning("STM32 interface not available. Waiting for availability...")
                start_time = time.time()
                while not self.stm.available:
                    if time.time() - start_time > 5:
                        self.logger.error("STM32 interface did not become available within 5 seconds")
                        raise TimeoutError("STM32 interface not available within the timeout period")
                    time.sleep(0.1)
            success = self.stm.send_cmd_polarization_numbers([polarization_number])
            if not success:
                raise RuntimeError(f"Failed to send polarization number {polarization_number}")
            
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
        
        # Map integer basis to string representation for each basis
        processed_bases = []
        for basis in bases:
            if basis == 0:
                processed_bases.append('Z')
            elif basis == 1:
                processed_bases.append('X')
            else:
                processed_bases.append(basis)
        
        polarization_numbers = [self._map_to_polarization_number(b, bit) 
                               for b, bit in zip(processed_bases, bits)]
        
        try:
            # Check and wait if available
            if not self.available:
                self.logger.warning("STM32 interface not available. Waiting for availability...")
                start_time = time.time()
                while not self.stm.available:
                    if time.time() - start_time > 5:
                        self.logger.error("STM32 interface did not become available within 5 seconds")
                        raise TimeoutError("STM32 interface not available within the timeout period")
                    time.sleep(0.1)            
            success = self.stm.send_cmd_polarization_numbers(polarization_numbers)
            if not success:
                raise RuntimeError(f"Failed to send polarization numbers {polarization_numbers}")
            self.logger.info(f"Encoded sequence of {len(bits)} bits")
            
        except Exception as e:
            self.logger.error(f"Failed to encode bit sequence: {e}")
            raise


    # Aditional commands
    #...

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

    def get_com_port(self) -> Optional[str]:
        """
        Get the current COM port.
        
        Returns:
            The current COM port or None if not set
        """
        return self.com_port if self.com_port else None

    def _handle_stm32_connected(self):
        """Callback for STM32 connection established."""
        self.connected = True
        self.logger.info("✓ STM32 interface connected to port %s", self.com_port)
    
    def _handle_stm32_available(self):
        """Callback for STM32 becoming available."""
        if self.stm.available:
            self.available = True
            self.logger.info("✓ STM32 interface is now available for commands")
        else:
            self.available = False
            self.logger.warning("STM32 interface is not available for commands yet")
        self.logger.info("STM32 interface available for commands")
    
    def __str__(self) -> str:
        """String representation of the polarization hardware."""
        return f"PolarizationHardware(com_port={self.com_port}, connected={self.connected}, current_angle={self.current_angle}, current_state={self.current_state})"
    
    def __enter__(self):
        """Context manager enter method."""
        if not self.initialize():
            raise RuntimeError("Failed to initialize polarization hardware")
        return self

    def __del__(self):
        """Destructor to ensure the serial connection is closed."""
        self.shutdown()