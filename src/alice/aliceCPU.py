"""
Alice's main controller for QKD transmission.
"""
import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from queue import Queue, Empty
import numpy as np
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.data_structures import Pulse, Basis, Bit, LaserInfo
from src.alice.qrng.qrng_simulator import QRNGSimulator, OperationMode
from src.alice.laser.laser_controller import LaserController
from src.alice.laser.laser_simulator import SimulatedLaserDriver
from src.alice.laser.laser_hardware_digital import DigitalHardwareLaserDriver
from src.alice.polarization.polarization_controller import PolarizationController, PolarizationOutput
from src.alice.polarization.polarization_simulator import PolarizationSimulator
from src.alice.polarization.polarization_hardware import PolarizationHardware


class AliceMode(Enum):
    """Operation modes for Alice CPU."""
    BATCH = "batch"          # Pre-generate all random bits before starting
    STREAMING = "streaming"  # Generate bits on-demand during transmission


@dataclass
class AliceConfig:
    """Configuration for Alice CPU."""
    num_pulses: int = 1000
    pulse_period_seconds: float = 1.0
    use_hardware: bool = False
    com_port: Optional[str] = None  # For polarization hardware
    laser_channel: Optional[int] = None  # For hardware laser
    qrng_seed: Optional[int] = None
    mode: AliceMode = AliceMode.STREAMING
    laser_repetition_rate_hz: float = 1000000  # 1 MHz for hardware laser


@dataclass
class AliceStats:
    """Statistics for Alice's operation."""
    pulses_sent: int = 0
    total_runtime_seconds: float = 0.0
    average_pulse_rate_hz: float = 0.0
    rotation_times: List[float] = field(default_factory=list)
    laser_times: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class TransmissionData:
    """Data collected during transmission."""
    pulse_times: List[float] = field(default_factory=list)
    pulses: List[Pulse] = field(default_factory=list)
    bases: List[Basis] = field(default_factory=list)
    bits: List[Bit] = field(default_factory=list)
    polarization_angles: List[float] = field(default_factory=list)
    pulse_ids: List[int] = field(default_factory=list)


class AliceCPU:
    """
    The main controller for Alice's side of the QKD protocol.
    
    Controls laser and polarization hardware/simulators with QRNG for 
    BB84 quantum key distribution. Supports both batch and streaming modes
    for random bit generation.
    """

    def __init__(self, config: AliceConfig):
        """
        Initialize Alice's CPU.

        Args:
            config: Configuration parameters for Alice's operation
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # System state
        self.stats = AliceStats()
        self.transmission_data = TransmissionData()
        self._running = False
        self._paused = False
        self._shutdown_event = threading.Event()
        self._pause_event = threading.Event()
        
        # Pre-generated random bits for batch mode
        self._batch_bases: List[Basis] = []
        self._batch_bits: List[Bit] = []
        self._batch_index = 0
        
        # Threading
        self._transmission_thread: Optional[threading.Thread] = None
        self._pulse_queue = Queue(maxsize=100)
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info(f"Alice CPU initialized in {self.config.mode.value} mode")

    def _initialize_components(self) -> None:
        """Initialize all Alice-side components."""
        
        # Initialize QRNG
        self.qrng = QRNGSimulator(
            seed=self.config.qrng_seed,
            mode=OperationMode.STREAMING if self.config.mode == AliceMode.STREAMING else OperationMode.BATCH
        )
        
        # Initialize Laser Controller
        if self.config.use_hardware:
            laser_driver = DigitalHardwareLaserDriver(
                digital_channel=self.config.laser_channel
            )
        else:
            laser_info = LaserInfo()
            laser_driver = SimulatedLaserDriver(pulses_queue=self._pulse_queue, laser_info=laser_info)
        
        self.laser_controller = LaserController(laser_driver)
        
        # Initialize Polarization Controller
        if self.config.use_hardware:
            if self.config.com_port is None:
                raise ValueError("COM port must be specified for hardware polarization control")
            pol_driver = PolarizationHardware(com_port=self.config.com_port,)
            # Note: COM port initialization should be handled within the hardware driver
        else:
            pol_driver = PolarizationSimulator()
        
        self.polarization_controller = PolarizationController(
            driver=pol_driver,
            qrng_driver=self.qrng
        )
        
        self.logger.info("All components initialized successfully")

    def initialize_system(self) -> bool:
        """
        Initialize the complete Alice system.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize laser
            if not self.laser_controller.initialize():
                self.logger.error("Failed to initialize laser controller")
                return False
            
            # Initialize polarization controller
            if not self.polarization_controller.initialize():
                self.logger.error("Failed to initialize polarization controller")
                return False
            
            # Pre-generate random bits for batch mode
            if self.config.mode == AliceMode.BATCH:
                self._generate_batch_random_bits()
            
            self.logger.info("Alice system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Alice system: {e}")
            self.stats.errors.append(f"Initialization error: {e}")
            return False

    def _generate_batch_random_bits(self) -> None:
        """Pre-generate all random bits for batch mode."""
        self.logger.info(f"Generating {self.config.num_pulses} random bits for batch mode")
        
        # Generate basis choices (Z or X)
        basis_bits = self.qrng.get_random_bit(size=self.config.num_pulses)
        self._batch_bases = [Basis.Z if bit == 0 else Basis.X for bit in basis_bits]
        
        # Generate bit values (0 or 1)
        bit_values = self.qrng.get_random_bit(size=self.config.num_pulses)
        self._batch_bits = [Bit(bit) for bit in bit_values]
        
        self.logger.info("Batch random bits generated successfully")

    def start_transmission(self) -> bool:
        """
        Start the QKD transmission.
        
        Returns:
            bool: True if transmission started successfully
        """
        if self._running:
            self.logger.warning("Transmission already running")
            return False
        
        if not self.initialize_system():
            return False
        
        self._running = True
        self._shutdown_event.clear()
        self._pause_event.clear()
        
        # Start transmission thread
        self._transmission_thread = threading.Thread(
            target=self._transmission_loop,
            name="AliceTransmission"
        )
        self._transmission_thread.start()
        
        self.logger.info("QKD transmission started")
        return True

    def stop_transmission(self) -> None:
        """Stop the QKD transmission."""
        if not self._running:
            self.logger.warning("Transmission not running")
            return
        
        self.logger.info("Stopping QKD transmission...")
        self._running = False
        self._shutdown_event.set()
        
        # Wait for transmission thread to finish
        if self._transmission_thread and self._transmission_thread.is_alive():
            self._transmission_thread.join(timeout=5.0)
        
        # Shutdown components
        self.laser_controller.shutdown()
        self.polarization_controller.shutdown()
        
        self.logger.info("QKD transmission stopped")

    def pause_transmission(self) -> None:
        """Pause the QKD transmission."""
        if not self._running:
            self.logger.warning("Transmission not running")
            return
        
        self._paused = True
        self._pause_event.set()
        self.logger.info("QKD transmission paused")

    def resume_transmission(self) -> None:
        """Resume the QKD transmission."""
        if not self._paused:
            self.logger.warning("Transmission not paused")
            return
        
        self._paused = False
        self._pause_event.clear()
        self.logger.info("QKD transmission resumed")

    def _transmission_loop(self) -> None:
        """Main transmission loop running in separate thread."""
        start_time = time.time()
        pulse_id = 0
        
        try:
            while self._running and pulse_id < self.config.num_pulses:
                # Check for pause
                if self._paused:
                    self._pause_event.wait()
                    continue
                
                # Check for shutdown
                if self._shutdown_event.is_set():
                    break
                
                pulse_start_time = time.time()
                
                try:
                    # Get random bits based on mode
                    if self.config.mode == AliceMode.BATCH:
                        basis, bit = self._get_batch_random_bits(pulse_id)
                    else:
                        basis, bit = self._get_streaming_random_bits()
                    
                    # Rotate polarization
                    rotation_start = time.time()
                    pol_output = self.polarization_controller.set_polarization_manually(basis, bit)
                    rotation_time = time.time() - rotation_start
                    self.stats.rotation_times.append(rotation_time)
                    
                    # Fire laser pulse
                    laser_start = time.time()
                    pulse = self._create_and_send_pulse(pol_output, pulse_id, pulse_start_time)
                    laser_time = time.time() - laser_start
                    self.stats.laser_times.append(laser_time)
                    
                    # Record data
                    self._record_transmission_data(pulse, pol_output, basis, bit, pulse_id)
                    
                    pulse_id += 1
                    self.stats.pulses_sent += 1
                    
                    # Calculate processing time and wait for next pulse
                    processing_time = time.time() - pulse_start_time
                    remaining_time = self.config.pulse_period_seconds - processing_time
                    
                    if remaining_time > 0:
                        time.sleep(remaining_time)
                        self.logger.debug(f"Pulse {pulse_id} processed in {processing_time:.3f}s, sleeping for {remaining_time:.3f}s")
                    else:
                        self.logger.warning(f"Pulse {pulse_id} took longer than period: {processing_time:.3f}s > {self.config.pulse_period_seconds}s")
                
                except Exception as e:
                    self.logger.error(f"Error processing pulse {pulse_id}: {e}")
                    self.stats.errors.append(f"Pulse {pulse_id} error: {e}")
                    continue
            
            # Update final statistics
            self.stats.total_runtime_seconds = time.time() - start_time
            if self.stats.total_runtime_seconds > 0:
                self.stats.average_pulse_rate_hz = self.stats.pulses_sent / self.stats.total_runtime_seconds
            
            self.logger.info(f"Transmission completed: {self.stats.pulses_sent} pulses in {self.stats.total_runtime_seconds:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Critical error in transmission loop: {e}")
            self.stats.errors.append(f"Critical transmission error: {e}")
        finally:
            self._running = False

    def _get_batch_random_bits(self, pulse_id: int) -> tuple[Basis, Bit]:
        """Get pre-generated random bits for batch mode."""
        if pulse_id >= len(self._batch_bases) or pulse_id >= len(self._batch_bits):
            raise IndexError(f"Pulse ID {pulse_id} exceeds batch size")
        
        return self._batch_bases[pulse_id], self._batch_bits[pulse_id]

    def _get_streaming_random_bits(self) -> tuple[Basis, Bit]:
        """Generate random bits on-demand for streaming mode."""
        # Generate basis choice (0 = Z, 1 = X)
        basis_bit = self.qrng.get_random_bit()
        basis = Basis.Z if basis_bit == 0 else Basis.X
        
        # Generate bit value
        bit_value = self.qrng.get_random_bit()
        bit = Bit(bit_value)
        
        return basis, bit

    def _create_and_send_pulse(self, pol_output: PolarizationOutput, pulse_id: int, timestamp: float) -> Pulse:
        """Create and send a laser pulse with the specified polarization."""
        
        # Fire the laser
        success = self.laser_controller.trigger_once()
        if not success:
            self.logger.error("Failed to trigger laser pulse")
            raise RuntimeError("Failed to trigger laser pulse")
        self.logger.debug(f"Laser pulse {pulse_id} triggered successfully")

        # Create pulse object for record-keeping
        pulse = Pulse(
            polarization=pol_output.angle_degrees,
            photons=1  # Assuming single photon per pulse
        )
        
        return pulse

    def _record_transmission_data(self, pulse: Pulse, pol_output: PolarizationOutput, 
                                basis: Basis, bit: Bit, pulse_id: int) -> None:
        """Record transmission data for analysis."""
        self.transmission_data.pulse_times.append(time.time())
        self.transmission_data.pulses.append(pulse)
        self.transmission_data.bases.append(basis)
        self.transmission_data.bits.append(bit)
        self.transmission_data.polarization_angles.append(pol_output.angle_degrees)
        self.transmission_data.pulse_ids.append(pulse_id)

    def get_statistics(self) -> AliceStats:
        """Get current system statistics."""
        return self.stats

    def get_transmission_data(self) -> TransmissionData:
        """Get collected transmission data."""
        return self.transmission_data

    def is_running(self) -> bool:
        """Check if transmission is currently running."""
        return self._running

    def is_paused(self) -> bool:
        """Check if transmission is currently paused."""
        return self._paused

    def get_progress(self) -> float:
        """Get transmission progress as percentage."""
        if self.config.num_pulses == 0:
            return 100.0
        return (self.stats.pulses_sent / self.config.num_pulses) * 100.0

    def get_component_info(self) -> Dict[str, Any]:
        """Get information about all components."""
        return {
            "laser_controller": {
                "type": type(self.laser_controller._driver).__name__,
                "initialized": self.laser_controller.is_initialized(),
                "active": self.laser_controller.is_active(),
                "pulse_count": self.laser_controller.pulse_count
            },
            "polarization_controller": {
                "type": type(self.polarization_controller.driver).__name__,
                "initialized": self.polarization_controller.is_initialized()
            },
            "qrng": {
                "type": type(self.qrng).__name__,
                "mode": self.qrng._mode.value if self.qrng._mode else "unknown",
                "bits_generated": self.qrng.get_bits_generated()
            }
        }

    def __del__(self):
        """Cleanup when object is destroyed."""
        if self._running:
            self.stop_transmission()


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("AliceCPU.Main")
    
    # Example configuration
    config = AliceConfig(
        num_pulses=10,
        pulse_period_seconds=1,  # 1 second between pulses
        use_hardware=True,
        com_port="COM4",  # Replace with actual COM port used for polarization hardware
        laser_channel=8,  # Replace with actual digital channel used for laser hardware
        mode=AliceMode.BATCH,
        qrng_seed=42
    )
    
    alice_cpu = AliceCPU(config)
    
    if alice_cpu.start_transmission():
        # Wait for transmission to complete
        while alice_cpu.is_running():
            time.sleep(1)
            progress = alice_cpu.get_progress()
            logger.info(f"Transmission progress: {progress:.2f}%")

        # Check if really stopped
        alice_cpu.stop_transmission()

        stats = alice_cpu.get_statistics()
        data = alice_cpu.get_transmission_data()
        
        logger.info(f"Transmission finished. Stats: {stats}")
        logger.info(f"Collected data for {len(data.pulse_ids)} pulses.")
    else:
        logger.error("Failed to start transmission")


#### MISSING BEFORE STARTING THE GENERATION AND SENDING, A READY TO RECEIVE SIGNAL WITH BOB THROUGH THE CLASSICAL CHANNEL