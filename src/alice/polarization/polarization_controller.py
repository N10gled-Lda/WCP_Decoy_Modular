"""Polarization Controller."""
import logging
from queue import Queue
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass

from src.utils.data_structures import Basis, Bit, Pulse, LaserInfo
from src.alice.qrng.qrng_simulator import QRNGSimulator, OperationMode
from src.alice.qrng.qrng_hardware import QRNGHardware

from .polarization_base import PolarizationState, JonesVector
from .polarization_simulator import PolarizationSimulator
from .polarization_hardware import PolarizationHardware


@dataclass
class PolarizationOutput:
    """Output of the polarization controller."""
    basis: Basis
    bit: Bit
    polarization_state: PolarizationState
    angle_degrees: float
    jones_vector: List[complex]


class PolarizationController:
    """Controls polarization with QRNG-based basis and bit selection for BB84 protocol."""
    # def __init__(self, pol_driver: Union[PolarizationSimulator, PolarizationHardware], com_port: str = None, qrng_driver: Union[QRNGSimulator, QRNGHardware] = None):
        # """
        # Initialize the polarization controller with a specific driver.
        # Args:
        #     pol_driver: The polarization driver (simulator or hardware).
        #     com_port: The COM port for the STM32 controller (if using hardware).
        #     qrng_driver: The QRNG driver (simulator or hardware).
        # """
        # self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # self.polarization_driver = pol_driver
        # if isinstance(self.polarization_driver, PolarizationHardware):
        #     if com_port is None:
        #         raise ValueError("COM port must be specified for physical polarization hardware.")
        #     if com_port not in [port.device for port in comports()]:
        #         raise ValueError(f"COM port {com_port} not found. Available ports: {[port.device for port in comports()]}")
        #     self.logger.info(f"Using PolarizationHardware with COM port: {com_port}")

        #     # This should not be in here, the com port should be set in the hardware interface since this controller is agnostic to the hardware implementation.
        #     # Initialize the hardware interface STM32 with the specified COM port
        #     self.polarization_driver.init_STM_com_port(com_port)

        # elif isinstance(self.polarization_driver, PolarizationSimulator):
        #     self.logger.warning("Using PolarizationSimulator, no COM port required.")
        # else:
        #     raise ValueError("Invalid polarization driver provided. Must be either PolarizationSimulator or PolarizationHardware.")

        # self.qrng_driver = qrng_driver if qrng_driver else QRNGSimulator()
        # self.logger.info(f"Polarization controller initialized with {type(self.polarization_driver).__name__} and {type(self.qrng_driver).__name__}.")

    def __init__(self, 
                 driver: Union[PolarizationSimulator, PolarizationHardware],
                 qrng_driver: Optional[Union[QRNGSimulator, QRNGHardware]] = None):
        """
        Initialize polarization controller with a specific driver.
        
        Args:
            driver: The polarization driver (simulator or hardware)
            qrng_driver: QRNG driver instance (optional, defaults to simulator)
        """
        self.driver = driver
        self.qrng = qrng_driver if qrng_driver else QRNGSimulator()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # State tracking
        self.current_basis = Basis.Z
        self.current_bit = Bit.ZERO
        self.current_state = PolarizationState.H
        self.current_angle = self.current_state.angle_degrees
        self._initialized = False
        
        self.logger.info(f"Polarization controller initialized with {type(driver).__name__}")

    def map_basis_bit_to_polarization(self, basis: Basis, bit: Bit) -> PolarizationState:
        """
        Map BB84 basis/bit pair to polarization state.
        
        BB84 mapping:
        - Z basis (rectilinear): bit 0 → H (0°), bit 1 → V (90°)
        - X basis (diagonal): bit 0 → D (45°), bit 1 → A (135°)
        
        Args:
            basis: Measurement basis (Z or X)
            bit: Bit value (0 or 1)
            
        Returns:
            Corresponding polarization state
        """
        if basis == Basis.Z:  # Rectilinear basis
            return PolarizationState.H if bit == Bit.ZERO else PolarizationState.V
        elif basis == Basis.X:  # Diagonal basis
            return PolarizationState.D if bit == Bit.ZERO else PolarizationState.A
        else:
            raise ValueError(f"Unknown basis: {basis}")

    def calculate_polarization_angle(self, basis: Basis, bit: Bit) -> float:
        """
        Calculate polarization angle using BB84 formula: angle = basis * 45° + bit * 90°.
        
        Args:
            basis: Measurement basis (0 for Z, 1 for X) 
            bit: Bit value (0 or 1)
            
        Returns:
            Polarization angle in degrees
        """
        basis_value = 0 if basis == Basis.Z else 1
        bit_value = int(bit)
        angle = basis_value * 45 + bit_value * 90
        return float(angle % 180)  # Keep in range [0, 180)

    def create_jones_vector(self, angle_degrees: float) -> JonesVector:
        """
        Create Jones vector for linear polarization at given angle.
        
        Args:
            angle_degrees: Polarization angle in degrees
            
        Returns:
            Jones vector representation
        """
        return JonesVector.from_angle(angle_degrees)

    def generate_random_basis_bit(self) -> Tuple[Basis, Bit]:
        """
        Generate random basis and bit using QRNG.
        
        Returns:
            Tuple of (basis, bit)
        """
        # Generate random basis (Z or X)
        basis_bit = self.qrng.get_random_bit(mode=OperationMode.STREAMING)
        basis = Basis.Z if basis_bit == 0 else Basis.X
        
        # Generate random bit (0 or 1)
        bit_value = self.qrng.get_random_bit(mode=OperationMode.STREAMING)
        bit = Bit.ZERO if bit_value == 0 else Bit.ONE
        
        self.logger.debug(f"QRNG generated: basis={basis}, bit={bit}")
        return basis, bit

    def set_polarization_from_qrng(self) -> PolarizationOutput:
        """
        Generate random basis/bit pair and set polarization accordingly.
        
        Returns:
            PolarizationOutput containing complete polarization information
        """
        # Generate random basis and bit
        basis, bit = self.generate_random_basis_bit()
        
        # Map to polarization state and angle
        state = self.map_basis_bit_to_polarization(basis, bit)
        angle = self.calculate_polarization_angle(basis, bit)
        # angle = state.angle_degrees
        
        # Create Jones vector
        jones_vector = self.create_jones_vector(angle)
        
        # Set polarization on hardware/simulator
        self.set_polarization_state(state)
        
        # Update internal state
        self.current_basis = basis
        self.current_bit = bit
        self.current_state = state
        self.current_angle = angle
        
        # Create output
        output = PolarizationOutput(
            basis=basis,
            bit=bit,
            polarization_state=state,
            angle_degrees=angle,
            jones_vector=jones_vector.to_list()
        )
        
        self.logger.info(f"Set polarization: {basis} basis, bit {bit} → {state} ({angle}°)")
        return output

    def set_polarization_manually(self, basis: Basis, bit: Bit) -> PolarizationOutput:
        """
        Manually set polarization for specific basis/bit combination.
        
        Args:
            basis: Measurement basis
            bit: Bit value
            
        Returns:
            PolarizationOutput containing complete polarization information
        """
        state = self.map_basis_bit_to_polarization(basis, bit)
        angle = self.calculate_polarization_angle(basis, bit)
        # angle = state.angle_degrees
        jones_vector = self.create_jones_vector(angle)
        
        self.set_polarization_state(state)
        
        self.current_basis = basis
        self.current_bit = bit
        self.current_state = state
        self.current_angle = angle
        
        output = PolarizationOutput(
            basis=basis,
            bit=bit,
            polarization_state=state,
            angle_degrees=angle,
            jones_vector=jones_vector.to_list()
        )
        
        self.logger.info(f"Manually set polarization: {basis} basis, bit {bit} → {state} ({angle}°)")
        return output

    def set_polarization_state(self, state: PolarizationState) -> None:
        """Set polarization to a specific BB84 state."""
        angle = state.angle_degrees
        
        if hasattr(self.driver, 'set_polarization_angle'):
            self.driver.set_polarization_angle(angle)
        elif hasattr(self.driver, 'set_polarization'):
            self.driver.set_polarization(angle)
        else:
            self.logger.warning("Polarization driver has no recognized set method")
        
        self.logger.debug(f"Set polarization state {state} at {angle}°")

    def get_current_state(self) -> dict:
        """Get current polarization state information."""
        return {
            'basis': self.current_basis,
            'bit': self.current_bit,
            'state': self.current_state,
            'angle_degrees': self.current_angle,
            'jones_vector': self.create_jones_vector(self.current_angle).to_list()
        }

    def apply_polarization_to_queue(self) -> None:
        """
        Apply current polarization to all pulses in the queue.
        This method processes pulses from the input queue and puts polarized pulses in the output queue.
        """
        if hasattr(self.driver, 'apply_polarization_queue'):
            self.driver.apply_polarization_queue()
            self.logger.info("Applied polarization to all queued pulses")
        else:
            self.logger.warning("Polarization driver doesn't support queue processing")

    def get_queue_info(self) -> dict:
        """Get information about the pulse queues (simulator only)."""
        if hasattr(self.driver, 'pulses_queue') and hasattr(self.driver, 'polarized_pulses_queue'):
            input_size = self.driver.pulses_queue.qsize()
            output_size = self.driver.polarized_pulses_queue.qsize()
            return {
                'pulses_queue_size': input_size,
                'polarized_pulses_queue_size': output_size,
                'current_state': self.current_state.name,
                'current_angle': self.current_angle
            }
        else:
            return {'message': 'Queue info only available for simulator driver with queues'}

    def initialize(self) -> bool:
        """Initialize the polarization controller and underlying driver."""
        try:
            if hasattr(self.driver, 'initialize'):
                success = self.driver.initialize()
                if not success:
                    self.logger.error("Failed to initialize polarization driver")
                    return False
            
            self._initialized = True
            self.logger.info("Polarization controller initialized successfully")

            # Initialize QRNG if not already done
            if self.qrng.get_rng() is None:
                self.qrng.set_random_seed()
                self.logger.warning("QRNG was not initialized: setting random seed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing polarization controller: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown the polarization controller and underlying driver."""
        try:
            if hasattr(self.driver, 'shutdown'):
                self.driver.shutdown()
            
            self._initialized = False
            self.logger.info("Polarization controller shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during polarization controller shutdown: {e}")

    def is_initialized(self) -> bool:
        """Check if the polarization controller is initialized."""
        return self._initialized

    # Convenience methods for setting driver-specific parameters
    def set_com_port(self, com_port: str) -> None:
        """Set COM port for hardware driver (if applicable)."""
        if hasattr(self.driver, 'set_com_port'):
            self.driver.set_com_port(com_port)
            self.logger.info(f"COM port set to {com_port}")
        else:
            self.logger.warning("Driver doesn't support COM port setting")

    def get_pulses_queue(self) -> Optional[Queue]:
        """Get input queue from simulator driver (if applicable)."""
        if hasattr(self.driver, 'pulses_queue'):
            return self.driver.pulses_queue
        return None

    def get_polarized_pulses_queue(self) -> Optional[Queue]:
        """Get output queue from simulator driver (if applicable)."""
        if hasattr(self.driver, 'polarized_pulses_queue'):
            return self.driver.polarized_pulses_queue
        return None

    def __enter__(self):
        """Context manager entry."""
        if not self._initialized:
            self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Factory functions for easy controller creation
def create_polarization_controller_with_simulator(pulses_queue: Optional[Queue] = None,
                                                 polarized_pulses_queue: Optional[Queue] = None,
                                                 laser_info: Optional[LaserInfo] = None,
                                                 qrng_driver: Optional[Union[QRNGSimulator, QRNGHardware]] = None) -> PolarizationController:
    """
    Create a polarization controller with simulator driver.
    
    Args:
        pulses_queue: Input queue for pulses (optional)
        polarized_pulses_queue: Output queue for polarized pulses (optional)
        laser_info: Laser information (optional)
        qrng_driver: QRNG driver (optional)
    
    Returns:
        PolarizationController with simulator driver
    """
    # Create default queues if not provided
    if pulses_queue is None:
        pulses_queue = Queue()
    if polarized_pulses_queue is None:
        polarized_pulses_queue = Queue()
    if laser_info is None:
        laser_info = LaserInfo(wavelength=1550.0, power=1.0, pulse_width=1e-9)
    
    # Create simulator driver
    simulator = PolarizationSimulator(
        pulses_queue=pulses_queue,
        polarized_pulses_queue=polarized_pulses_queue,
        laser_info=laser_info
    )
    
    # Create controller
    return PolarizationController(driver=simulator, qrng_driver=qrng_driver)


def create_polarization_controller_with_hardware(com_port: str,
                                                qrng_driver: Optional[Union[QRNGSimulator, QRNGHardware]] = None) -> PolarizationController:
    """
    Create a polarization controller with hardware driver.
    
    Args:
        com_port: COM port for STM32 interface
        qrng_driver: QRNG driver (optional)
    
    Returns:
        PolarizationController with hardware driver
    """
    # Create hardware driver
    hardware = PolarizationHardware(com_port=com_port)
    
    # Create controller
    return PolarizationController(driver=hardware, qrng_driver=qrng_driver)


