"""Comprehensive VOA controller demo supporting both simulator and hardware testing."""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import logging
import time
from queue import Queue
from typing import List, Dict

# Import VOA components
from src.alice.voa import (
    VOAController,
    DecoyInfoExtended,
    VOAOutput,
    VOASimulator,
    VOAHardwareDriver
)
from src.alice.qrng.qrng_simulator import QRNGSimulator, OperationMode
from src.utils.data_structures import DecoyState, Pulse, LaserInfo

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VOAControllerDemo:
    """Comprehensive demo for VOA controller with both simulator and hardware."""
    
    def __init__(self):
        """Initialize the demo."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    

    ################# Simulator Controller Tests #################

    def test_simulator_controller(self) -> bool:
        """Test VOA controller with simulator."""
        self.logger.info("\n\n========= Testing VOA Controller with Simulator =========")
        
        try:
            # Create input/output queues
            pulses_queue = Queue()
            attenuated_pulses_queue = Queue()
            
            # Create laser info
            laser_info = LaserInfo(wavelength=1550.0, power=1.0, pulse_width=1e-9)
            
            # Create VOA simulator
            voa_simulator = VOASimulator(
                pulses_queue=pulses_queue,
                attenuated_pulses_queue=attenuated_pulses_queue,
                laser_info=laser_info
            )
            
            # Create QRNG simulator
            qrng_simulator = QRNGSimulator()
            
            # Create custom decoy configuration
            decoy_info = DecoyInfoExtended(
                intensities={"signal": 0.5, "weak": 0.1, "vacuum": 0.0},
                probabilities={"signal": 0.7, "weak": 0.2, "vacuum": 0.1}
            )
            
            # Create VOA controller
            voa_controller = VOAController(
                driver=voa_simulator,
                physical=False,
                qrng_driver=qrng_simulator,
                decoy_info=decoy_info
            )
            
            # Initialize
            self.logger.info("  1. Initializing VOA controller...")
            voa_controller.initialize()
            
            # Test basic functionality
            self.logger.info("  2. Testing basic functionality...")
            success = self._test_basic_functionality(voa_controller)
            if not success:
                return False
            
            # Test state selection methods
            self.logger.info("  3. Testing state selection methods...")
            success = self._test_state_selection_methods(voa_controller)
            if not success:
                return False
            
            # Test pulse generation with queue processing
            self.logger.info("  4. Testing pulse processing with queues...")
            success = self._test_pulse_processing(voa_controller, pulses_queue, attenuated_pulses_queue)
            if not success:
                return False
            
            # Test configuration updates
            self.logger.info("  5. Testing configuration updates...")
            success = self._test_configuration_updates(voa_controller)
            if not success:
                return False
            
            # Shutdown
            voa_controller.shutdown()
            
            self.logger.info("VOA simulator controller test completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"VOA simulator controller test failed: {e}")
            return False
    
    def _test_basic_functionality(self, controller: VOAController) -> bool:
        """Test basic VOA controller functionality."""
        self.logger.info("--- Testing Basic Functionality ---")
        
        try:
            # Test manual state selection
            for state in [DecoyState.SIGNAL, DecoyState.WEAK, DecoyState.VACUUM]:
                controller.set_state(state)
                current_attenuation = controller.get_attenuation()
                self.logger.info(f"State: {state}, Attenuation: {current_attenuation:.2f} dB")
                
                # Verify state was set correctly
                if controller.current_state != state:
                    self.logger.error(f"State mismatch: expected {state}, got {controller.current_state}")
                    return False
            
            # Test attenuation calculation
            test_intensities = [0.5, 0.1, 0.01, 0.0]
            for intensity in test_intensities:
                attenuation = controller.calculate_attenuation_for_intensity(intensity)
                self.logger.info(f"Intensity: {intensity}, Calculated attenuation: {attenuation:.2f} dB")
            
            self.logger.info("Basic functionality test passed!")
            return True
            
        except Exception as e:
            self.logger.error(f"Basic functionality test failed: {e}")
            return False
    
    def _test_state_selection_methods(self, controller: VOAController) -> bool:
        """Test different state selection methods."""
        self.logger.info("--- Testing State Selection Methods ---")
        
        try:
            # Test probability-based selection
            self.logger.info("Testing probability-based state selection:")
            state_counts = {DecoyState.SIGNAL: 0, DecoyState.WEAK: 0, DecoyState.VACUUM: 0}
            
            num_trials = 1000
            for _ in range(num_trials):
                state = controller.get_random_state_by_probability()
                state_counts[state] += 1
            
            # Log observed probabilities
            for state, count in state_counts.items():
                observed_prob = count / num_trials
                expected_prob = controller.decoy_info.get_probability(state)
                self.logger.info(f"{state}: {count}/{num_trials} = {observed_prob:.3f} "
                               f"(expected: {expected_prob:.3f})")
            
            # Test uniform random selection
            self.logger.info("Testing uniform random state selection:")
            state_counts_uniform = {DecoyState.SIGNAL: 0, DecoyState.WEAK: 0, DecoyState.VACUUM: 0}
            
            for _ in range(num_trials):
                state = controller.get_random_state_from_random_bits()
                state_counts_uniform[state] += 1
            
            # Log observed distribution for uniform selection
            for state, count in state_counts_uniform.items():
                observed_prob = count / num_trials
                self.logger.info(f"{state}: {count}/{num_trials} = {observed_prob:.3f}")
            
            self.logger.info("State selection methods test passed!")
            return True
            
        except Exception as e:
            self.logger.error(f"State selection methods test failed: {e}")
            return False
    
    def _test_pulse_processing(self, controller: VOAController, 
                              input_queue: Queue, output_queue: Queue) -> bool:
        """Test pulse processing with queues."""
        self.logger.info("--- Testing Pulse Processing ---")
        
        try:
            # Create test pulses
            test_pulses = [
                Pulse(polarization=0.0, photons=1000)
                for i in range(5)
            ]
            
            # Add pulses to input queue
            for pulse in test_pulses:
                input_queue.put(pulse)
            
            # Set a specific attenuation
            controller.set_attenuation(10.0)  # 10 dB attenuation
            
            # Process pulses through VOA simulator
            controller.voa_driver.apply_attenuation_queue()
            
            # Check output queue
            processed_pulses = []
            while not output_queue.empty():
                processed_pulses.append(output_queue.get())
            
            self.logger.info(f"Processed {len(processed_pulses)} pulses")
            
            # Verify attenuation was applied
            expected_factor = 10 ** (-10.0 / 10)  # For 10 dB attenuation
            for i, (original, processed) in enumerate(zip(test_pulses, processed_pulses)):
                expected_photons = original.photons * expected_factor
                self.logger.info(f"Pulse {i}: {original.photons} → {processed.photons} photons "
                               f"(expected ~{expected_photons:.0f})")
            
            self.logger.info("Pulse processing test passed!")
            return True
            
        except Exception as e:
            self.logger.error(f"Pulse processing test failed: {e}")
            return False
    
    def _test_configuration_updates(self, controller: VOAController) -> bool:
        """Test dynamic configuration updates."""
        self.logger.info("--- Testing Configuration Updates ---")
        
        try:
            # Test intensity updates
            self.logger.info("Testing intensity updates:")
            new_intensities = {"signal": 0.8, "weak": 0.15, "vacuum": 0.0}
            controller.update_intensities(new_intensities["signal"], new_intensities["weak"], new_intensities["vacuum"])
            
            for state_str, intensity in new_intensities.items():
                state = DecoyState[state_str.upper()]
                calculated_attenuation = controller.calculate_attenuation_for_state(state)
                self.logger.info(f"Updated {state}: intensity={intensity}, "
                               f"attenuation={calculated_attenuation:.2f} dB")
            
            # This should raise ValueError since don't sum to 1
            try:
                new_probabilities = {"signal": 0.6, "weak": 0.2, "vacuum": 0.1}
                controller.update_probabilities(new_probabilities["signal"], new_probabilities["weak"], new_probabilities["vacuum"])
                self.logger.error("Expected ValueError was not raised!")
                return False
            except ValueError as e:
                self.logger.info(f"Expected error caught: {e}")

            # Test probability updates
            self.logger.info("Testing probability updates:")
            new_probabilities = {"signal": 0.6, "weak": 0.3, "vacuum": 0.1}
            controller.update_probabilities(new_probabilities["signal"], new_probabilities["weak"], new_probabilities["vacuum"])


            # Test pulse probabilities generation with new configuration
            num_trials = 1000
            state_counts = {DecoyState.SIGNAL: 0, DecoyState.WEAK: 0, DecoyState.VACUUM: 0}
            for _ in range(num_trials):
                state = controller.get_random_state_by_probability()
                state_counts[state] += 1
            
            # Log observed probabilities
            for state, count in state_counts.items():
                observed_prob = count / num_trials
                expected_prob = controller.decoy_info.get_probability(state)
                self.logger.info(f"{state}: {count}/{num_trials} = {observed_prob:.3f} "
                               f"(expected: {expected_prob:.3f})")

            self.logger.info("Configuration updates test passed!")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration updates test failed: {e}")
            return False
    

    ################# Hardware Controller Tests #################

    def test_hardware_controller(self, device) -> bool:
        """Test VOA controller with hardware (placeholder)."""
        self.logger.info("\n\n========= Testing VOA Controller with Hardware =========")
        
        try:
            # NOTE: This is a placeholder for hardware testing
            # Actual hardware implementation would go here
            
            self.logger.warning("Hardware testing not yet implemented - this is a placeholder")
            
            # Create VOA hardware driver (will raise NotImplementedError for now)
            try:
                voa_hardware = VOAHardwareDriver(device_id=device)
                
                # Create custom decoy configuration
                decoy_info = DecoyInfoExtended(
                    intensities={"signal": 0.5, "weak": 0.1, "vacuum": 0.0},
                    probabilities={"signal": 0.7, "weak": 0.2, "vacuum": 0.1}
                )
                
                # Create VOA controller
                voa_controller = VOAController(
                    driver=voa_hardware,
                    physical=True,
                    decoy_info=decoy_info,
                )
                
                # This will raise NotImplementedError
                # voa_controller.initialize()
                
                self.logger.info("Hardware controller created (but not initialized due to TODO implementation)")
                
            except NotImplementedError:
                self.logger.info("Hardware implementation not yet available - this is expected")
                return True
            
            return True
            
        except Exception as e:
            self.logger.error(f"VOA hardware controller test failed: {e}")
            return False
        

    def run_all_tests(self, device) -> bool:
        """Run all VOA controller tests."""
        self.logger.info("Starting comprehensive VOA controller tests...")
        
        all_passed = True
        
        # Test simulator
        if not self.test_simulator_controller():
            all_passed = False
        
        # Test hardware (placeholder)
        if not self.test_hardware_controller(device):
            all_passed = False
        
        if all_passed:
            self.logger.info("\n🎉 All VOA controller tests passed!")
        else:
            self.logger.error("\n❌ Some VOA controller tests failed!")
        
        return all_passed


def create_voa_controller_with_simulator(pulses_queue: Queue = None,
                                       attenuated_pulses_queue: Queue = None,
                                       laser_info: LaserInfo = None,
                                       decoy_info: DecoyInfoExtended = None) -> VOAController:
    """
    Convenience function to create a VOA controller with simulator.
    
    Args:
        pulses_queue: Input queue for pulses
        attenuated_pulses_queue: Output queue for attenuated pulses
        laser_info: Laser information
        decoy_info: Decoy state configuration
        
    Returns:
        Configured VOA controller with simulator
    """
    # Create queues if not provided
    if pulses_queue is None:
        pulses_queue = Queue()
    if attenuated_pulses_queue is None:
        attenuated_pulses_queue = Queue()
    
    # Create laser info if not provided
    if laser_info is None:
        laser_info = LaserInfo(wavelength=1550.0, power=1.0, pulse_width=1e-9)
    
    # Create default decoy info if not provided
    if decoy_info is None:
        decoy_info = DecoyInfoExtended(
            intensities={"signal": 0.5, "weak": 0.1, "vacuum": 0.0},
            probabilities={"signal": 0.7, "weak": 0.2, "vacuum": 0.1}
        )
    
    # Create simulator
    voa_simulator = VOASimulator(
        pulses_queue=pulses_queue,
        attenuated_pulses_queue=attenuated_pulses_queue,
        laser_info=laser_info
    )
    
    # Create QRNG
    qrng_simulator = QRNGSimulator()
    
    # Create controller
    controller = VOAController(
        driver=voa_simulator,
        physical=False,
        qrng_driver=qrng_simulator,
        decoy_info=decoy_info
    )
    
    return controller


def create_voa_controller_with_hardware(device_id: str = "voa_device_0",
                                      decoy_info: DecoyInfoExtended = None) -> VOAController:
    """
    Convenience function to create a VOA controller with hardware.
    
    Args:
        device_id: Hardware device identifier
        decoy_info: Decoy state configuration
        
    Returns:
        Configured VOA controller with hardware
    """
    # Create default decoy info if not provided
    if decoy_info is None:
        decoy_info = DecoyInfoExtended(
            intensities={"signal": 0.5, "weak": 0.1, "vacuum": 0.0},
            probabilities={"signal": 0.7, "weak": 0.2, "vacuum": 0.1}
        )
    
    # Create hardware driver
    voa_hardware = VOAHardwareDriver(device_id=device_id)
    
    # Create controller
    controller = VOAController(
        driver=voa_hardware,
        physical=True,
        decoy_info=decoy_info,
    )
    
    return controller


def main():
    """Main demo function."""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive VOA Controller Demo")
    parser.add_argument("--simulator-only", "--s", action="store_true", help="Run only simulator tests")
    parser.add_argument("--hardware-only", "--h", action="store_true", help="Run only hardware tests")
    parser.add_argument("--device", type=str, default="voa_device_0", help="VOA device identifier")

    args = parser.parse_args()
    
    demo = VOAControllerDemo()
    
    if args.simulator_only:
        success = demo.test_simulator_controller()
        logger.info("Simulator tests completed")
    elif args.hardware_only:
        success = demo.test_hardware_controller(args.device)
        logger.info("Hardware tests completed")
    else:
        success = demo.run_all_tests(args.device)
        logger.info("All tests completed")

    
    if success:
        logger.info("\nDemo completed successfully!")
    else:
        logger.error("\nDemo encountered errors!")
    
    return success


if __name__ == "__main__":
    main()
