#!/usr/bin/env python3
"""
Simple Alice Hardware Test - No Bob Communication
Tests Alice's hardware components (laser, QRNG, polarization) without classical communication.
Supports both predetermined sequences and random generation.
"""

import logging
import time
import threading
import sys
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from queue import Queue

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.data_structures import Pulse, Basis, Bit, LaserInfo
from src.alice.qrng.qrng_simulator import QRNGSimulator, OperationMode
from src.alice.laser.laser_controller import LaserController
from src.alice.laser.laser_simulator import SimulatedLaserDriver
from src.alice.laser.laser_hardware_digital import DigitalHardwareLaserDriver
from src.alice.polarization.polarization_controller import PolarizationController, PolarizationOutput
from src.alice.polarization.polarization_simulator import PolarizationSimulator
from src.alice.polarization.polarization_hardware import PolarizationHardware

TIMEOUT_WAIT_FOR_ROTATION = 10 # 10s

class AliceTestMode(Enum):
    """Test modes for Alice hardware.
        - RANDOM_STREAM: Generate random bits using QRNG bit by bit
        - RANDOM_BATCH: Generate random bits in batch using QRNG all before
        - SEEDED: Generate bits using a fixed seed using QRNG
        - PREDETERMINED: Use pre-defined sequence
    """
    RANDOM_STREAM = "random_stream"     # Generate random bits using QRNG bit by bit
    RANDOM_BATCH = "random_batch"       # Generate random bits in batch using QRNG all before
    SEEDED = "seeded"                   # Generate bits using a fixed seed using QRNG
    PREDETERMINED = "predetermined"     # Use pre-defined sequence


@dataclass
class AliceTestConfig:
    """Configuration for Alice hardware test."""
    num_pulses: int = 10
    pulse_period_seconds: float = 1.0
    use_hardware: bool = False
    com_port: Optional[str] = None
    laser_channel: Optional[int] = 8
    mode: AliceTestMode = AliceTestMode.RANDOM_STREAM
    qrng_seed: Optional[int] = None
    # Predetermined sequences (only used if mode=PREDETERMINED); 
    # must be of the size of num_pulses
    predetermined_bits: Optional[List[int]] = None
    predetermined_bases: Optional[List[int]] = None


@dataclass
class AliceTestResults:
    """Results from Alice hardware test."""
    bits: List[int] = field(default_factory=list)
    bases: List[int] = field(default_factory=list)
    polarization_angles: List[float] = field(default_factory=list)
    pulse_times: List[float] = field(default_factory=list)
    rotation_times: List[float] = field(default_factory=list)
    wait_times: List[float] = field(default_factory=list)
    laser_elapsed: List[float] = field(default_factory=list)
    total_runtime: float = 0.0
    errors: List[str] = field(default_factory=list)


class SimpleAliceHardwareTest:
    """
    Simple Alice hardware test without Bob communication.
    Tests the hardware control sequence with packets/threading but no network.
    """

    def __init__(self, config: AliceTestConfig):
        """Initialize Alice hardware test."""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Results
        self.results = AliceTestResults()
        
        # Threading
        self._running = False
        self._test_thread: Optional[threading.Thread] = None
        
        # Hardware components
        self._initialize_hardware()
        
        self.logger.info(f"Alice hardware test initialized in {config.mode.value} mode")

    def _initialize_hardware(self) -> None:
        """Initialize all hardware components."""
        self.logger.info("Initializing Alice hardware components...")
        
        # Initialize QRNG
        self.qrng = QRNGSimulator(
            seed=self.config.qrng_seed,
            mode=OperationMode.STREAMING if self.config.mode == AliceTestMode.RANDOM_STREAM else
                 OperationMode.BATCH if self.config.mode == AliceTestMode.RANDOM_BATCH else
                 OperationMode.DETERMINISTIC if self.config.mode == AliceTestMode.SEEDED else
                 None  # None for predetermined mode
        )
        
        # Create shared queues for components
        pulses_queue = Queue()
        polarized_pulses_queue = Queue()
        laser_info = LaserInfo()
        
        # Initialize Laser Controller
        if self.config.use_hardware and self.config.laser_channel is not None:
            laser_driver = DigitalHardwareLaserDriver(
                digital_channel=self.config.laser_channel
            )
            self.logger.info(f"Using hardware laser on channel {self.config.laser_channel}")
        else:
            laser_driver = SimulatedLaserDriver(pulses_queue=pulses_queue, laser_info=laser_info)
            self.logger.info("Using simulated laser")
        
        self.laser_controller = LaserController(laser_driver)
        
        # Initialize Polarization Controller
        if self.config.use_hardware and self.config.com_port is not None:
            pol_driver = PolarizationHardware(com_port=self.config.com_port)
            self.logger.info(f"Using hardware polarization on {self.config.com_port}")
        else:
            pol_driver = PolarizationSimulator(
                pulses_queue=pulses_queue,
                polarized_pulses_queue=polarized_pulses_queue,
                laser_info=laser_info
            )
            self.logger.info("Using simulated polarization")
        
        self.polarization_controller = PolarizationController(
            driver=pol_driver,
            qrng_driver=self.qrng
        )
        
        # Initialize components
        if not self.laser_controller.initialize():
            raise RuntimeError("Failed to initialize laser controller")
        
        if not self.polarization_controller.initialize():
            raise RuntimeError("Failed to initialize polarization controller")
        
        self.logger.info("All hardware components initialized successfully")

    def _validate_predetermined_sequences(self) -> bool:
        """Validate predetermined sequences if provided."""
        if self.config.mode != (AliceTestMode.PREDETERMINED or AliceTestMode.RANDOM_BATCH):
            return True
            
        if self.config.predetermined_bits is None or self.config.predetermined_bases is None:
            self.logger.error("Predetermined mode requires both bits and bases to be specified")
            return False
            
        if len(self.config.predetermined_bits) != self.config.num_pulses:
            self.logger.error(f"Predetermined bits length ({len(self.config.predetermined_bits)}) doesn't match num_pulses ({self.config.num_pulses})")
            return False
            
        if len(self.config.predetermined_bases) != self.config.num_pulses:
            self.logger.error(f"Predetermined bases length ({len(self.config.predetermined_bases)}) doesn't match num_pulses ({self.config.num_pulses})")
            return False
            
        # Validate values
        for i, bit in enumerate(self.config.predetermined_bits):
            if bit not in [0, 1]:
                self.logger.error(f"Invalid bit value at index {i}: {bit} (must be 0 or 1)")
                return False
                
        for i, basis in enumerate(self.config.predetermined_bases):
            if basis not in [0, 1]:
                self.logger.error(f"Invalid basis value at index {i}: {basis} (must be 0 or 1)")
                return False

        self.logger.debug("Predetermined sequences validated successfully: %s", self.config.predetermined_bits)

        return True

    def _validate_seeded_mode(self) -> bool:
        """Validate seeded mode if specified."""
        if self.config.mode != AliceTestMode.SEEDED:
            return True
        if self.qrng.get_mode() != OperationMode.DETERMINISTIC:
            self.logger.warning("Seeded mode requires QRNG to be in deterministic mode")
            self.qrng.set_mode(OperationMode.DETERMINISTIC)
            return True
        if self.config.qrng_seed is None:
            self.logger.error("Seeded mode requires a seed to be specified")
            return False
        return True
    
    def _validate_random_modes(self) -> bool:
        """Validate random modes if specified."""
        if self.config.mode == AliceTestMode.RANDOM_STREAM:
            if self.qrng.get_mode() != OperationMode.STREAMING:
                self.logger.warning("Random stream mode requires QRNG to be in streaming mode")
                self.qrng.set_mode(OperationMode.STREAMING)
                return True
        if self.config.mode == AliceTestMode.RANDOM_BATCH:
            if self.qrng.get_mode() != OperationMode.BATCH:
                self.logger.warning("Random batch mode requires QRNG to be in batch mode")
                self.qrng.set_mode(OperationMode.BATCH)
            # Get batch of bases and bits before sending
            self.config.predetermined_bases = self.qrng.get_random_bit(size=self.config.num_pulses)
            self.config.predetermined_bits = self.qrng.get_random_bit(size=self.config.num_pulses)
            return True
        return True

    def run_test(self) -> AliceTestResults:
        """Run the Alice hardware test."""
        if not self._validate_seeded_mode():
            raise ValueError("Invalid seeded mode")
        if not self._validate_random_modes():
            raise ValueError("Invalid random modes")
        if not self._validate_predetermined_sequences():
            raise ValueError("Invalid predetermined sequences")

        if not self.laser_controller.is_initialized():
            raise RuntimeError("Laser controller is not initialized")

        self.logger.info("Starting Alice hardware test...")

        # !!! Set period of stepmottor for 1ms since the stepmottor wait for this period after sending a single pulse afecting the check availability if high !!!
        self.polarization_controller.driver.set_operation_period(1)
        self.polarization_controller.driver.set_stepper_frequency(500)
        self._running = True
        start_time = time.time()
        
        # Run test in thread
        self._test_thread = threading.Thread(
            target=self._hardware_test_thread,
            name="AliceHardwareTest"
        )
        self._test_thread.start()
        
        # Wait for completion
        if self._test_thread:
            self._test_thread.join()
        
        end_time = time.time()
        self.results.total_runtime = end_time - start_time
        
        self.logger.info(f"Alice hardware test completed in {self.results.total_runtime:.2f}s")
        return self.results

    def _hardware_test_thread(self) -> None:
        """Main hardware test thread."""
        try:
            for pulse_id in range(self.config.num_pulses):
                if not self._running:
                    break
                    
                pulse_start_time = time.time()
                
                # Get basis and bit
                basis, bit = self._get_basis_and_bit(pulse_id)
                
                # Set polarization
                print(f"🔸 Pulse {pulse_id}: Setting polarization Basis={basis.name}, Bit={bit.value}")
                rotation_start = time.time()
                pol_output = self.polarization_controller.set_polarization_manually(basis, bit)
                rotation_time = time.time() - rotation_start
                print(f"   ➡️  Polarization set to {pol_output.angle_degrees}°")
                print(f"       (Rotation time: {rotation_time:.3f}s)")
                
                # Wait for polarization readiness
                print(f"🔸 Pulse {pulse_id}: Waiting for polarization readiness...")
                wait_start = time.time()
                if not self.polarization_controller.wait_for_availability(timeout=TIMEOUT_WAIT_FOR_ROTATION):
                    error_msg = f"Timeout waiting for polarization readiness for pulse {pulse_id}"
                    self.logger.error(error_msg)
                    self.results.errors.append(error_msg)
                    continue
                wait_time = time.time() - wait_start
                print(f"   ➡️  Polarization ready after {wait_time:.3f}s / {time.time() - pulse_start_time:.3f}s")
                
                # Fire laser
                print(f"🔸 Pulse {pulse_id}: Firing laser")
                laser_send_time = time.time()
                if not self.laser_controller.trigger_once():
                    error_msg = f"Failed to fire laser for pulse {pulse_id}"
                    self.logger.error(error_msg)
                    self.results.errors.append(error_msg)
                    continue
                laser_elapsed_time = time.time() - laser_send_time
                print(f"   ➡️  Laser fired in {laser_elapsed_time:.3f}s")
                
                # Record results
                self.results.bits.append(int(bit))
                self.results.bases.append(basis.int)
                self.results.polarization_angles.append(pol_output.angle_degrees)
                self.results.pulse_times.append(laser_send_time) # real time stamp is right before sending the laser command
                self.results.rotation_times.append(rotation_time)
                self.results.wait_times.append(wait_time)
                self.results.laser_elapsed.append(laser_elapsed_time)
                
                # Wait for next pulse
                total_pulse_time = time.time() - pulse_start_time
                remaining_time = self.config.pulse_period_seconds - total_pulse_time
                
                if remaining_time > 0:
                    print(f"--------------------------> DEBUG: Waiting {remaining_time:.3f}s for next pulse")
                    self.logger.debug(f"Waiting {remaining_time:.3f}s for next pulse")
                    time.sleep(remaining_time)
                else:
                    self.logger.warning(f" ⚠️ Pulse {pulse_id} exceeded period: {total_pulse_time:.3f}s > {self.config.pulse_period_seconds}s")
                
                print(f"   ✅ Pulse {pulse_id} completed\n")
                
        except Exception as e:
            error_msg = f"Error in hardware test thread: {e}"
            self.logger.error(error_msg)
            self.results.errors.append(error_msg)
        except KeyboardInterrupt:
            self.logger.info("Hardware test interrupted by user")
            self._running = False
            self._shutdown()
        finally:
            self._running = False
            self._shutdown()

    def _get_basis_and_bit(self, pulse_id: int) -> Tuple[Basis, Bit]:
        """Get basis and bit for the given pulse."""
        if self.config.mode == AliceTestMode.PREDETERMINED or self.config.mode == AliceTestMode.RANDOM_BATCH:
            # Use predetermined values
            basis_val = self.config.predetermined_bases[pulse_id]
            bit_val = self.config.predetermined_bits[pulse_id]
        elif self.config.mode == AliceTestMode.SEEDED or self.config.mode == AliceTestMode.RANDOM_STREAM:
            # Generate random values using QRNG
            basis_val = int(self.qrng.get_random_bit())
            bit_val = int(self.qrng.get_random_bit())
        else:
            raise ValueError("Invalid test mode")
        
        basis = Basis.Z if basis_val == 0 else Basis.X
        bit = Bit(bit_val)
        
        return basis, bit

    def _shutdown(self) -> None:
        """Shutdown hardware components."""
        self.logger.info("Shutting down hardware components...")
        try:
            self.laser_controller.shutdown()
            self.polarization_controller.shutdown()
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    def stop_test(self) -> None:
        """Stop the test."""
        self._running = False
        if self._test_thread and self._test_thread.is_alive():
            self._test_thread.join(timeout=5.0)

    def is_running(self) -> bool:
        """Check if test is running."""
        return self._running

    def get_results(self) -> AliceTestResults:
        """Get current test results."""
        return self.results


def demo_random_test():
    """Demonstrate random test mode."""
    print("=" * 60)
    print("Alice Hardware Test - Random Mode")
    print("=" * 60)
    
    config = AliceTestConfig(
        num_pulses=5,
        pulse_period_seconds=1.0,
        use_hardware=False,  # Use simulators for demo
        mode=AliceTestMode.RANDOM_STREAM,
        qrng_seed=42
    )
    
    test = SimpleAliceHardwareTest(config)
    results = test.run_test()
    
    print("\n" + "=" * 60)
    print("Random Test Results")
    print("=" * 60)
    print(f"✓ Pulses generated: {len(results.bits)}")
    print(f"✓ Total runtime: {results.total_runtime:.2f}s")
    print(f"✓ Average pulse rate: {len(results.bits) / results.total_runtime:.2f} Hz")
    print(f"✓ Bits: {results.bits}")
    print(f"✓ Bases: {results.bases}")
    print(f"✓ Angles: {[f'{angle:.1f}°' for angle in results.polarization_angles]}")
    if results.rotation_times:
        print(f"✓ Avg rotation time: {sum(results.rotation_times)/len(results.rotation_times)*1000:.1f} ms")
    if results.wait_times:
        print(f"✓ Avg wait time: {sum(results.wait_times)/len(results.wait_times)*1000:.1f} ms")
    if results.laser_elapsed:
        print(f"✓ Avg laser time: {sum(results.laser_elapsed)/len(results.laser_elapsed)*1000:.1f} ms")
    if results.errors:
        print(f"⚠️  Errors: {results.errors}")


def demo_predetermined_test():
    """Demonstrate predetermined test mode."""
    print("\n" + "=" * 60)
    print("Alice Hardware Test - Predetermined Mode")
    print("=" * 60)
    
    # Define specific sequence for testing
    predetermined_bits = [0, 1, 1, 0, 1]    # Bit sequence
    predetermined_bases = [0, 0, 1, 1, 0]   # Basis sequence (0=Z, 1=X)
    
    print(f"Predetermined sequence:")
    for i in range(len(predetermined_bits)):
        basis_name = "Z" if predetermined_bases[i] == 0 else "X"
        print(f"  Pulse {i}: bit={predetermined_bits[i]}, basis={basis_name}")
    print()
    
    config = AliceTestConfig(
        num_pulses=5,
        pulse_period_seconds=1,
        use_hardware=False,  # Use simulators for demo
        mode=AliceTestMode.PREDETERMINED,
        predetermined_bits=predetermined_bits,
        predetermined_bases=predetermined_bases
    )
    
    test = SimpleAliceHardwareTest(config)
    results = test.run_test()
    
    print("\n" + "=" * 60)
    print("Predetermined Test Results")
    print("=" * 60)
    print(f"✓ Pulses generated: {len(results.bits)}")
    print(f"✓ Total runtime: {results.total_runtime:.2f}s")
    
    # Verify sequence matches
    sequence_correct = (results.bits == predetermined_bits and 
                       results.bases == predetermined_bases)
    print(f"✓ Sequence verification: {'PASSED' if sequence_correct else 'FAILED'}")
    
    print(f"✓ Expected bits: {predetermined_bits}")
    print(f"✓ Actual bits:   {results.bits}")
    print(f"✓ Expected bases: {predetermined_bases}")
    print(f"✓ Actual bases:   {results.bases}")
    print(f"✓ Angles: {[f'{angle:.1f}°' for angle in results.polarization_angles]}")
    
    if results.errors:
        print(f"⚠️  Errors: {results.errors}")


def demo_hardware_test():
    """Demonstrate hardware test (if available)."""
    print("\n" + "=" * 60)
    print("Alice Hardware Test - Hardware Mode")
    print("=" * 60)
    
    # Hardware configuration (adjust COM port and laser channel as needed)
    config = AliceTestConfig(
        num_pulses=5,
        pulse_period_seconds=2.0,  # Slower for hardware
        use_hardware=True,
        com_port="COM4",  # Adjust for your setup
        laser_channel=8,   # Adjust for your setup
        mode=AliceTestMode.RANDOM_STREAM,
        qrng_seed=123
    )
    
    try:
        test = SimpleAliceHardwareTest(config)
        results = test.run_test()
        
        print("\n" + "=" * 60)
        print("Hardware Test Results")
        print("=" * 60)
        print(f"✓ Pulses generated: {len(results.bits)}")
        print(f"✓ Total runtime: {results.total_runtime:.2f}s")
        print(f"✓ Bits: {results.bits}")
        print(f"✓ Bases: {results.bases}")
        print(f"✓ Angles: {[f'{angle:.1f}°' for angle in results.polarization_angles]}")
        
        # Hardware-specific metrics
        if results.wait_times:
            avg_wait = sum(results.wait_times) / len(results.wait_times)
            max_wait = max(results.wait_times)
            print(f"✓ Avg wait time: {avg_wait:.3f}s")
            print(f"✓ Max wait time: {max_wait:.3f}s")
        
        if results.errors:
            print(f"⚠️  Errors: {results.errors}")
        else:
            print("✅ Hardware test completed successfully!")
            
    except Exception as e:
        print(f"❌ Hardware test failed: {e}")
        print("   (This is expected if no hardware is connected)")

def demo_hardware_test_predetermined():
    """Demonstrate hardware test with predetermined sequence (if available)."""
    print("\n" + "=" * 60)
    print("Alice Hardware Test - Predetermined Hardware Mode")
    print("=" * 60)
    
    # Define specific sequence for testing
    predetermined_bits = [0, 1, 1, 0, 1]    # Bit sequence
    predetermined_bases = [0, 0, 1, 1, 0]   # Basis sequence (0=Z, 1=X)
    predetermined_bits = [0, 1, 0, 1, 0, 0, 0, 1]    # Bit sequence
    predetermined_bases = [0, 0, 1, 1, 0, 1, 0, 1]   # Basis sequence (0=Z, 1=X)
    
    print(f"Predetermined sequence:")
    for i in range(len(predetermined_bits)):
        basis_name = "Z" if predetermined_bases[i] == 0 else "X"
        print(f"  Pulse {i}: bit={predetermined_bits[i]}, basis={basis_name}")
    print()
    
    # Hardware configuration (adjust COM port and laser channel as needed)
    config = AliceTestConfig(
        num_pulses=len(predetermined_bits),
        pulse_period_seconds=1,
        use_hardware=True,
        com_port="COM4",  # Adjust for your setup
        laser_channel=8,   # Adjust for your setup
        mode=AliceTestMode.PREDETERMINED,
        predetermined_bits=predetermined_bits,
        predetermined_bases=predetermined_bases
    )
    
    try:
        test = SimpleAliceHardwareTest(config)
        results = test.run_test()
        
        print("\n" + "=" * 60)
        print("Predetermined Hardware Test Results")
        print("=" * 60)
        print(f"✓ Pulses generated: {len(results.bits)}")
        print(f"✓ Total runtime: {results.total_runtime:.2f}s")
        
        # Verify sequence matches
        sequence_correct = (results.bits == predetermined_bits and 
                           results.bases == predetermined_bases)
        print(f"✓ Sequence verification: {'PASSED' if sequence_correct else 'FAILED'}")
        
        print(f"✓ Expected bits: {predetermined_bits}")
        print(f"✓ Actual bits:   {results.bits}")
        print(f"✓ Expected bases: {predetermined_bases}")
        print(f"✓ Actual bases:   {results.bases}")
        print(f"✓ Angles: {[f'{angle:.1f}°' for angle in results.polarization_angles]}")
        
        # Hardware-specific metrics
        if results.wait_times:
            avg_wait = sum(results.wait_times) / len(results.wait_times)
            max_wait = max(results.wait_times)
            print(f"✓ Avg wait time: {avg_wait:.3f}s")
            print(f"✓ Max wait time: {max_wait:.3f}s")
        if results.errors:
            print(f"⚠️  Errors: {results.errors}")
        else:
            print("✅ Predetermined hardware test completed successfully!")

    except Exception as e:
        print(f"❌ Predetermined hardware test failed: {e}")
        print("   (This is expected if no hardware is connected)")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Alice Hardware Test Suite")
    print("Testing Alice's hardware components without Bob communication")
    print()
    
    # Run all demo tests
    # demo_random_test()
    # demo_predetermined_test()
    # demo_hardware_test()
    demo_hardware_test_predetermined()
    
    print("\n" + "=" * 60)
    print("Test Suite Completed")
    print("=" * 60)
    print("To run with custom parameters:")
    print("  python simple_alice_hardware_test.py")
    print()
    print("For hardware testing, modify COM port and laser channel in demo_hardware_test()")
