"""
Alice CPU - General Controller for QKD Transmitter.

Orchestrates all Alice-side components including QRNG, laser, VOA, and 
polarization control to implement the BB84 decoy-state protocol.
Supports both simulation and hardware modes for all components.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Iterator, Union
from queue import Queue as ThreadQueue
from threading import Thread, Event
from dataclasses import dataclass, field
import numpy as np

from ..utils.data_structures import (
    Pulse, DecoyState, Basis, Bit, LaserInfo, DecoyInfo, ChannelInfo, DetectorInfo
)

# QRNG imports
from .qrng.qrng_simulator import QRNGSimulator, OperationMode
from .qrng.qrng_hardware import QRNGHardware

# Laser imports
from .laser.laser_controller import LaserController
from .laser.laser_simulator import SimulatedLaserDriver
from .laser.laser_hardware_digital import DigitalHardwareLaserDriver
from .laser.laser_hardware import HardwareLaserDriver

# VOA imports
from .voa.voa_controller import VOAController
from .voa.voa_simulator import VOASimulator
from .voa.voa_hardware import VOAHardwareDriver

# Polarization imports
from .polarization.polarization_controller import PolarizationController
from .polarization.polarization_simulator import PolarizationSimulator
from .polarization.polarization_hardware import PolarizationHardware


logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    # General settings
    pulses_total: int = 10000
    random_seed: Optional[int] = None
    use_hardware: bool = False
    
    # Component-specific hardware flags
    use_hardware_qrng: bool = False
    use_hardware_laser: bool = False
    use_hardware_voa: bool = False
    use_hardware_polarization: bool = False
    
    # Hardware-specific settings
    com_port_polarization: Optional[str] = None
    laser_repetition_rate_hz: float = 1000000
    
    # Component configurations
    laser: LaserInfo = field(default_factory=LaserInfo)
    decoy_scheme: DecoyInfo = field(default_factory=DecoyInfo)
    channel: ChannelInfo = field(default_factory=ChannelInfo)
    detector: DetectorInfo = field(default_factory=DetectorInfo)
    
    # Timing configuration
    pulse_period_seconds: float = 1e-6  # 1 MHz default
    
    # QRNG operation mode
    qrng_mode: OperationMode = OperationMode.STREAMING


@dataclass
class ClockConfig:
    """Clock timing configuration."""
    repetition_rate_Hz: float = 1000000  # 1 MHz
    duty_cycle: float = 0.5


@dataclass 
class LaserControlConfig:
    """Laser controller configuration."""
    repetition_rate_Hz: float = 1000000
    duty_cycle: float = 0.5


@dataclass
class QRNGConfig:
    """QRNG configuration."""
    mode: OperationMode = OperationMode.STREAMING
    seed: Optional[int] = None
    bias_probability: float = 0.5


@dataclass
class VOAConfig:
    """VOA configuration."""
    signal_intensity: float = 0.5
    weak_intensity: float = 0.1
    vacuum_intensity: float = 0.0


@dataclass
class PolarizationConfig:
    """Polarization configuration."""
    calibration_required: bool = True


@dataclass
class AliceState:
    """Current state of Alice's system."""
    is_active: bool = False
    pulses_sent: int = 0
    current_pulse_rate_Hz: float = 0.0
    qrng_entropy: float = 0.0
    laser_power_mW: float = 0.0
    total_runtime_s: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass
class AliceData:
    """Alice's transmission data for post-processing."""
    pulse_times: List[float] = field(default_factory=list)
    decoy_states: List[DecoyState] = field(default_factory=list)
    bases: List[Basis] = field(default_factory=list)
    bits: List[int] = field(default_factory=list)
    intensities: List[float] = field(default_factory=list)
    polarization_angles: List[float] = field(default_factory=list)
    pulse_ids: List[int] = field(default_factory=list)


@dataclass
class SimulationResults:
    """Results from a complete simulation."""
    alice_data: AliceData
    alice_state: AliceState
    statistics: Dict[str, Any]
    component_info: Dict[str, Any]


class AliceCPUGeneral:
    """
    Alice's general controller for QKD transmission.
    
    Coordinates all transmitter components to implement the complete
    BB84 decoy-state protocol with weak coherent pulses.
    Supports both simulation and hardware modes for each component.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize Alice's CPU.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # System state
        self.state = AliceState()
        self.alice_data = AliceData()
        
        # Control signals
        self._shutdown_event = Event()
        self._pause_event = Event()
        
        # Pulse processing queues
        self.laser_queue: ThreadQueue = ThreadQueue(maxsize=1000)
        self.voa_queue: ThreadQueue = ThreadQueue(maxsize=1000)
        self.polarization_queue: ThreadQueue = ThreadQueue(maxsize=1000)
        self.output_queue: ThreadQueue = ThreadQueue(maxsize=1000)
        
        # Initialize components
        self._initialize_components()
        
        # Processing threads
        self.threads: List[Thread] = []
        
        self.logger.info("Alice CPU General initialized")
    
    def _initialize_components(self) -> None:
        """Initialize all Alice-side components."""
        
        # Initialize QRNG (hardware or simulator based on config)
        if self.config.use_hardware_qrng:
            self.qrng = QRNGHardware()
            self.logger.info("Using QRNG Hardware")
        else:
            qrng_config = QRNGConfig(
                mode=self.config.qrng_mode,
                seed=self.config.random_seed,
                bias_probability=0.5
            )
            self.qrng = QRNGSimulator(
                seed=qrng_config.seed,
                mode=qrng_config.mode
            )
            self.logger.info("Using QRNG Simulator")
        
        # Initialize Laser Controller with appropriate driver
        if self.config.use_hardware_laser:
            if hasattr(self.config, 'laser_type') and self.config.laser_type == 'digital':
                laser_driver = DigitalHardwareLaserDriver(
                    repetition_rate_hz=self.config.laser_repetition_rate_hz
                )
                self.logger.info("Using Digital Hardware Laser Driver")
            else:
                laser_driver = HardwareLaserDriver()
                self.logger.info("Using Hardware Laser Driver")
        else:
            laser_driver = SimulatedLaserDriver()
            self.logger.info("Using Simulated Laser Driver")
        
        self.laser_controller = LaserController(laser_driver)
        
        # Initialize VOA Controller with appropriate driver
        if self.config.use_hardware_voa:
            voa_driver = VOAHardwareDriver()
            self.logger.info("Using VOA Hardware Driver")
        else:
            voa_config = {
                "extinction_ratio_dB": 60.0,
                "insertion_loss_dB": 0.1
            }
            voa_driver = VOASimulator(voa_config)
            self.logger.info("Using VOA Simulator")
        
        voa_ctrl_config = VOAConfig(
            signal_intensity=self.config.decoy_scheme.intensities["signal"],
            weak_intensity=self.config.decoy_scheme.intensities["weak"],
            vacuum_intensity=self.config.decoy_scheme.intensities["vacuum"]
        )
        self.voa_controller = VOAController(voa_ctrl_config)
        self.voa_simulator = voa_driver
        
        # Initialize Polarization Controller with appropriate driver
        if self.config.use_hardware_polarization:
            if self.config.com_port_polarization is None:
                raise ValueError("COM port must be specified for hardware polarization control")
            pol_driver = PolarizationHardware()
            self.logger.info(f"Using Polarization Hardware on {self.config.com_port_polarization}")
        else:
            pol_sim_config = {
                "extinction_ratio_dB": self.config.laser.polarization_extinction_ratio_dB,
                "drift_rate_deg_per_hour": 0.1
            }
            pol_driver = PolarizationSimulator(pol_sim_config)
            self.logger.info("Using Polarization Simulator")
        
        self.polarization_controller = PolarizationController(
            pol_driver=pol_driver,
            com_port=self.config.com_port_polarization,
            qrng_driver=self.qrng
        )
        self.polarization_simulator = pol_driver
        
        self.logger.info("All Alice components initialized")
    
    def start_transmission(self) -> None:
        """Start the transmission process."""
        if self.state.is_active:
            self.logger.warning("Transmission already active")
            return
        
        self.state.is_active = True
        self._shutdown_event.clear()
        self._pause_event.clear()
        
        # Initialize components
        if not self.laser_controller.initialize():
            self.logger.error("Failed to initialize laser controller")
            return
        
        if not self.polarization_controller.initialize():
            self.logger.error("Failed to initialize polarization controller")
            return
        
        # Initialize simulator queues if using simulators
        if not self.config.use_hardware_laser:
            if hasattr(self.laser_controller._driver, 'initialize'):
                self.laser_controller._driver.initialize(self.laser_queue)
        
        # Start laser controller
        self.laser_controller.start()
        
        # Start processing threads
        self._start_processing_threads()
        
        # Start main transmission loop
        self.main_thread = Thread(target=self._transmission_loop, daemon=True)
        self.main_thread.start()
        self.threads.append(self.main_thread)
        
        self.logger.info("Alice transmission started")
    
    def stop_transmission(self) -> None:
        """Stop the transmission process."""
        if not self.state.is_active:
            self.logger.warning("Transmission not active")
            return
        
        self.logger.info("Stopping Alice transmission...")
        
        # Signal shutdown
        self._shutdown_event.set()
        self.state.is_active = False
        
        # Stop laser controller
        self.laser_controller.stop()
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        # Shutdown components
        if hasattr(self.laser_controller._driver, 'shutdown'):
            self.laser_controller._driver.shutdown()
        if hasattr(self.voa_simulator, 'shutdown'):
            self.voa_simulator.shutdown()
        if hasattr(self.polarization_simulator, 'shutdown'):
            self.polarization_simulator.shutdown()
        
        # Cleanup polarization controller
        self.polarization_controller.cleanup()
        
        self.logger.info("Alice transmission stopped")
    
    def _start_processing_threads(self) -> None:
        """Start background processing threads."""
        
        # Only start simulator threads if using simulators
        if not self.config.use_hardware_voa and hasattr(self.voa_simulator, 'process_pulse_queue'):
            # VOA processing thread
            voa_thread = Thread(
                target=self.voa_simulator.process_pulse_queue,
                args=(self.voa_queue, self.polarization_queue),
                daemon=True
            )
            voa_thread.start()
            self.threads.append(voa_thread)
        
        if not self.config.use_hardware_polarization and hasattr(self.polarization_simulator, 'process_pulse_queue'):
            # Polarization processing thread
            pol_thread = Thread(
                target=self.polarization_simulator.process_pulse_queue,
                args=(self.polarization_queue, self.output_queue),
                daemon=True
            )
            pol_thread.start()
            self.threads.append(pol_thread)
        
        self.logger.debug("Processing threads started")
    
    def _transmission_loop(self) -> None:
        """Main transmission loop."""
        start_time = time.time()
        pulse_id = 0
        
        try:
            while (not self._shutdown_event.is_set() and 
                   pulse_id < self.config.pulses_total):
                
                # Check for pause
                if self._pause_event.is_set():
                    time.sleep(0.001)
                    continue
                
                # Generate random choices for this pulse
                decoy_bits = self._get_random_bits(2)  # For decoy state selection
                basis_bit = self._get_random_bit()     # For basis choice
                data_bit = self._get_random_bit()      # For data bit
                
                # Determine decoy state
                decoy_state = self._map_bits_to_decoy_state(decoy_bits[0], decoy_bits[1])
                
                # Determine basis and bit
                basis = Basis.Z if basis_bit == 0 else Basis.X
                bit_value = data_bit
                
                # Generate trigger signal
                trigger = self.laser_controller.generate_trigger()
                
                if trigger:
                    # Generate laser pulse
                    pulse = self._generate_pulse(trigger, pulse_id)
                    
                    if pulse is not None:
                        # Store Alice's choices
                        self._record_alice_data(pulse, decoy_state, basis, bit_value)
                        
                        # Process through VOA
                        pulse = self._process_voa(pulse, decoy_state)
                        
                        # Process through polarization
                        pulse = self._process_polarization(pulse, basis, bit_value)
                        
                        # Add to output queue
                        try:
                            self.output_queue.put_nowait(pulse)
                        except:
                            self.logger.warning(f"Output queue full, dropping pulse {pulse_id}")
                        
                        pulse_id += 1
                        self.state.pulses_sent += 1
                
                # Update pulse rate
                if pulse_id % 10000 == 0:
                    self._update_pulse_rate(start_time, pulse_id)
                
                # Control timing
                time.sleep(self.config.pulse_period_seconds)
                
        except Exception as e:
            self.logger.error(f"Transmission loop error: {e}")
            self.state.errors.append(str(e))
        
        # Record total runtime
        self.state.total_runtime_s = time.time() - start_time
        
        self.logger.info(f"Transmission completed: {pulse_id} pulses in {self.state.total_runtime_s:.2f}s")
    
    def _get_random_bit(self) -> int:
        """Get a single random bit from QRNG."""
        if self.config.use_hardware_qrng:
            return self.qrng.get_random_bit()
        else:
            return self.qrng.get_random_bit()
    
    def _get_random_bits(self, count: int) -> List[int]:
        """Get multiple random bits from QRNG."""
        if self.config.use_hardware_qrng:
            return [self.qrng.get_random_bit() for _ in range(count)]
        else:
            if self.config.qrng_mode == OperationMode.BATCH:
                return self.qrng.get_random_bit(size=count)
            else:
                return [self.qrng.get_random_bit() for _ in range(count)]
    
    def _generate_pulse(self, trigger: bool, pulse_id: int) -> Optional[Pulse]:
        """Generate a laser pulse."""
        if self.config.use_hardware_laser:
            # Hardware laser pulse generation
            self.laser_controller.fire_pulse()
            return Pulse(polarization=0.0, photons=1)  # Basic pulse structure
        else:
            # Simulator pulse generation
            if hasattr(self.laser_controller._driver, 'generate_pulse'):
                return self.laser_controller._driver.generate_pulse(trigger, pulse_id)
            else:
                return Pulse(polarization=0.0, photons=1)
    
    def _process_voa(self, pulse: Pulse, decoy_state: DecoyState) -> Pulse:
        """Process pulse through VOA."""
        if self.config.use_hardware_voa:
            # Hardware VOA processing
            attenuation_dB = self.voa_controller.select_decoy_state(decoy_state)
            # Apply attenuation to hardware
            return pulse
        else:
            # Simulator VOA processing
            attenuation_dB = self.voa_controller.select_decoy_state(decoy_state)
            if hasattr(self.voa_simulator, 'apply_attenuation'):
                return self.voa_simulator.apply_attenuation(pulse, attenuation_dB)
            return pulse
    
    def _process_polarization(self, pulse: Pulse, basis: Basis, bit: int) -> Pulse:
        """Process pulse through polarization control."""
        if self.config.use_hardware_polarization:
            # Hardware polarization control
            pol_output = self.polarization_controller.set_polarization(basis, bit)
            pulse.polarization = pol_output.angle_degrees
            return pulse
        else:
            # Simulator polarization control
            pol_output = self.polarization_controller.set_polarization(basis, bit)
            pulse.polarization = pol_output.angle_degrees
            return pulse
    
    def _map_bits_to_decoy_state(self, bit1: int, bit2: int) -> DecoyState:
        """Map two random bits to decoy state according to probabilities."""
        # Use probabilistic mapping based on config
        probs = self.config.decoy_scheme.probabilities
        
        # Get random value for decoy selection
        rand_val = self._get_random_bit() / 2.0 + self._get_random_bit() / 4.0
        
        if rand_val < probs["signal"]:
            return DecoyState.SIGNAL
        elif rand_val < probs["signal"] + probs["weak"]:
            return DecoyState.WEAK
        else:
            return DecoyState.VACUUM
    
    def _record_alice_data(self, pulse: Pulse, decoy_state: DecoyState, basis: Basis, bit: int) -> None:
        """Record Alice's transmission data."""
        self.alice_data.pulse_times.append(time.time())
        self.alice_data.decoy_states.append(decoy_state)
        self.alice_data.bases.append(basis)
        self.alice_data.bits.append(bit)
        self.alice_data.intensities.append(getattr(pulse, 'mean_photon_number', 1.0))
        self.alice_data.polarization_angles.append(pulse.polarization)
        self.alice_data.pulse_ids.append(len(self.alice_data.pulse_ids))
    
    def _update_pulse_rate(self, start_time: float, pulse_count: int) -> None:
        """Update current pulse rate."""
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            self.state.current_pulse_rate_Hz = pulse_count / elapsed_time
    
    def get_state(self) -> AliceState:
        """Get current Alice system state."""
        return AliceState(
            is_active=self.state.is_active,
            pulses_sent=self.state.pulses_sent,
            current_pulse_rate_Hz=self.state.current_pulse_rate_Hz,
            qrng_entropy=0.0,  # TODO: Get from QRNG
            laser_power_mW=getattr(self.laser_controller._driver, 'current_power_mW', 0.0),
            total_runtime_s=self.state.total_runtime_s,
            errors=self.state.errors.copy()
        )
    
    def get_alice_data(self) -> AliceData:
        """Get Alice's transmission data."""
        return self.alice_data
    
    def get_output_pulses(self, max_pulses: Optional[int] = None) -> List[Pulse]:
        """
        Get processed pulses from output queue.
        
        Args:
            max_pulses: Maximum number of pulses to retrieve
            
        Returns:
            List of processed pulses
        """
        pulses = []
        count = 0
        
        while not self.output_queue.empty() and (max_pulses is None or count < max_pulses):
            try:
                pulse = self.output_queue.get_nowait()
                pulses.append(pulse)
                count += 1
            except:
                break
        
        return pulses
    
    def pause_transmission(self) -> None:
        """Pause the transmission."""
        self._pause_event.set()
        self.logger.info("Alice transmission paused")
    
    def resume_transmission(self) -> None:
        """Resume the transmission."""
        self._pause_event.clear()
        self.logger.info("Alice transmission resumed")
    
    def get_component_statistics(self) -> Dict[str, Any]:
        """Get statistics from all components."""
        stats = {
            "alice_state": self.get_state(),
            "component_types": {
                "qrng": "hardware" if self.config.use_hardware_qrng else "simulator",
                "laser": "hardware" if self.config.use_hardware_laser else "simulator",
                "voa": "hardware" if self.config.use_hardware_voa else "simulator",
                "polarization": "hardware" if self.config.use_hardware_polarization else "simulator"
            },
            "queue_sizes": {
                "laser": self.laser_queue.qsize(),
                "voa": self.voa_queue.qsize(),
                "polarization": self.polarization_queue.qsize(),
                "output": self.output_queue.qsize()
            }
        }
        
        # Add component-specific statistics if available
        if hasattr(self.qrng, 'get_statistics'):
            stats["qrng"] = self.qrng.get_statistics()
        if hasattr(self.laser_controller, 'get_timing_info'):
            stats["laser_controller"] = self.laser_controller.get_timing_info()
        if hasattr(self.voa_controller, 'get_statistics'):
            stats["voa_controller"] = self.voa_controller.get_statistics()
        if hasattr(self.polarization_controller, 'get_encoding_statistics'):
            stats["polarization_controller"] = self.polarization_controller.get_encoding_statistics()
        
        return stats
    
    def calibrate_system(self) -> bool:
        """Perform system calibration."""
        self.logger.info("Starting Alice system calibration...")
        
        calibration_success = True
        
        try:
            # Calibrate polarization controller
            if hasattr(self.polarization_controller, 'calibrate'):
                if not self.polarization_controller.calibrate():
                    calibration_success = False
                    self.logger.error("Polarization calibration failed")
            
            # Test laser functionality
            if hasattr(self.laser_controller, 'test_pulse'):
                if not self.laser_controller.test_pulse():
                    calibration_success = False
                    self.logger.error("Laser test failed")
            
            # Run QRNG quality tests (if available)
            if hasattr(self.qrng, 'run_quality_tests'):
                qrng_tests = self.qrng.run_quality_tests(10000)
                if qrng_tests.get("overall") != "PASS":
                    calibration_success = False
                    self.logger.error("QRNG quality tests failed")
            
        except Exception as e:
            self.logger.error(f"Calibration failed with exception: {e}")
            calibration_success = False
        
        if calibration_success:
            self.logger.info("Alice system calibration successful")
        else:
            self.logger.error("Alice system calibration failed")
        
        return calibration_success
    
    def get_simulation_results(self) -> SimulationResults:
        """Get complete simulation results."""
        return SimulationResults(
            alice_data=self.get_alice_data(),
            alice_state=self.get_state(),
            statistics=self.get_component_statistics(),
            component_info={
                "hardware_components": {
                    "qrng": self.config.use_hardware_qrng,
                    "laser": self.config.use_hardware_laser,
                    "voa": self.config.use_hardware_voa,
                    "polarization": self.config.use_hardware_polarization
                },
                "configuration": {
                    "pulses_total": self.config.pulses_total,
                    "pulse_period_seconds": self.config.pulse_period_seconds,
                    "qrng_mode": self.config.qrng_mode.value
                }
            }
        )
