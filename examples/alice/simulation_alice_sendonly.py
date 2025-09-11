"""Integration test demonstrating the complete QKD system components."""
import sys
import os


# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging
import time
import threading
from queue import Queue
from typing import Dict, Any

# Import all components
from src.alice.laser.laser_controller import LaserController
from src.alice.laser.laser_simulator import SimulatedLaserDriver
from src.utils.data_structures import LaserInfo, Pulse
from src.alice.polarization.polarization_controller import PolarizationController
from src.alice.polarization.polarization_simulator import PolarizationSimulator, PolarizationState
from src.alice.qrng.qrng_simulator import OperationMode, QRNGSimulator
from src.quantum_channel.free_space_channel import FreeSpaceChannel
from src.utils.data_structures import Pulse
from configs.hardware_config import SYSTEM_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AliceSimulator:
    """Simulated Alice station for QKD."""
    
    def __init__(self):
        """Initialize Alice's components."""
        self.laser = None
        self.polarization = None
        self.qrng = None
        self.is_running = False
        
    def initialize(self):
        """Initialize all Alice components."""
        try:
            logger.info("Initializing Alice station...")
            
            # Initialize laser
            from queue import Queue
            pulses_queue = Queue()
            laser_info = LaserInfo(
                max_power_mW=1000.0,
                pulse_width_fwhm_ns=1000.0,
                central_wavelength_nm=1550.0
            )
            laser_driver = SimulatedLaserDriver(pulses_queue, laser_info)
            self.laser = LaserController(laser_driver)
            self.laser.initialize()
            
            # Initialize polarization controller with QRNG
            polarization_queue = Queue()
            pol_driver = PolarizationSimulator(pulses_queue, polarization_queue, laser_info)
            qrng_driver = QRNGSimulator(seed=42)
            self.polarization = PolarizationController(pol_driver, qrng_driver)
            self.polarization.initialize()
            
            logger.info("Alice station initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Alice station: {e}")
            return False
    
    def send_qkd_sequence(self, n_pulses: int = 100) -> Dict[str, Any]:
        """Send a sequence of QKD pulses."""
        logger.info(f"Alice sending {n_pulses} QKD pulses...")
        
        results = {
            "pulses_sent": 0,
            "basis_choices": [],
            "bit_values": [],
            "intensities": [],
            "timing": []
        }
        
        try:
            # Turn on laser and polarization
            if (self.laser.is_initialized is False):
                self.laser.initialize()
            if (self.polarization.is_initialized is False):
                self.polarization.initialize()

            # Start pulse sequence
            start_time = time.time()
            
            for i in range(n_pulses):
                # Generate random basis and bit
                basis = int(self.polarization.qrng.get_random_bit(mode=OperationMode.STREAMING))
                bit_value = int(self.polarization.qrng.get_random_bit(mode=OperationMode.STREAMING))
                logger.debug(f"DEBUG: Pulse {i}: basis={basis}, bit_value={bit_value}")
                
                # Trigger laser
                logger.debug(f"DEBUG: Pre trigger Laser pulses queue: {self.laser._driver.pulses_queue.qsize()}")
                self.laser.trigger_once()
                logger.debug(f"DEBUG: Post trigger Laser pulses queue: {self.laser._driver.pulses_queue.qsize()}")

                # Send pulse to polarization controller
                self.polarization.set_polarization_manually(basis, bit_value)
                logger.debug(f"DEBUG: Pre polarized pulses queue: {self.polarization.driver.pulses_queue.qsize()}, {self.polarization.driver.polarized_pulses_queue.qsize()}")
                self.polarization.apply_polarization_to_queue()
                logger.debug(f"DEBUG: Post polarized pulses queue: {self.polarization.driver.pulses_queue.qsize()}, {self.polarization.driver.polarized_pulses_queue.qsize()}")
                logger.debug(f"DEBUG: Pulse {i} sent: {self.polarization.driver.current_state}, {self.polarization.driver.current_angle}, {self.polarization.driver.current_photons}")

                send_time = time.time()
                # Record this pulse
                results["pulses_sent"] += 1
                results["basis_choices"].append(basis)  # H/V or D/A
                results["bit_values"].append(bit_value)
                results["intensities"].append(1)  # Simplified
                results["timing"].append(send_time)
                # results["timing"].append(send_time - start_time)
                
                # Small delay between pulses
                time.sleep(0.001)  # 1kHz rate
            
            # Turn off laser and polarization
            self.laser.shutdown()
            self.polarization.shutdown()
            
            logger.info(f"Alice completed {results['pulses_sent']} pulses")
            return results
            
        except Exception as e:
            logger.error(f"Error in Alice pulse sequence: {e}")
            return results
    
    def shutdown(self):
        """Shutdown Alice components."""
        logger.info("Shutting down Alice station...")
        
        if self.laser:
            self.laser.shutdown()
        
        if self.polarization:
            self.polarization.shutdown()
        
        logger.info("Alice station shutdown complete")

class QKDSystemIntegrationTest:
    """Integration test for the complete QKD system."""
    
    def __init__(self):
        """Initialize the test system."""
        self.alice = AliceSimulator()
        
    def run_test(self, n_pulses: int = 100, measurement_time: float = 10.0):
        """Run the complete integration test."""
        logger.info("=== QKD System Integration Test ===")
        
        try:
            # Initialize both stations
            if not self.alice.initialize():
                logger.error("Failed to initialize Alice")
                return False
            
            # Alice sends pulse sequence
            alice_results = self.alice.send_qkd_sequence(n_pulses)
            
            # Analyze results
            self.analyze_results(alice_results)
            
            return True
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return False
        
        finally:
            # Cleanup
            self.alice.shutdown()
            # self.bob.shutdown()
    
    def analyze_results(self, alice_results: Dict):
        """Analyze the QKD test results."""
        logger.info("=== Test Results Analysis ===")
        
        # Alice statistics
        logger.info(f"Alice sent: {alice_results['pulses_sent']} pulses")
        
        if alice_results['basis_choices']:
            h_count = alice_results['basis_choices'].count(0)
            d_count = alice_results['basis_choices'].count(1)
            logger.info(f"Alice basis distribution: H/V={h_count}, D/A={d_count}")
        
        # Timing analysis
        if alice_results['timing'] and len(alice_results['timing']) > 1:
            avg_interval = sum(alice_results['timing'][i+1] - alice_results['timing'][i] 
                             for i in range(len(alice_results['timing'])-1)) / (len(alice_results['timing'])-1)
            logger.info(f"Average pulse interval: {avg_interval*1000:.1f} ms")
        
        logger.info("Analysis complete")


def main():
    """Main test function."""
    logger.info("Starting QKD System Integration Test...")
    
    # Create and run test
    test_system = QKDSystemIntegrationTest()
    success = test_system.run_test(n_pulses=50, measurement_time=5.0)
    
    if success:
        logger.info("🎉 Integration test completed successfully!")
    else:
        logger.error("💥 Integration test failed!")
    
    return success


if __name__ == "__main__":
    main()
