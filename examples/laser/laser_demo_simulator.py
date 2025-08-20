"""Example usage of the laser controller with Digilent hardware interface."""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging
import time
from src.alice.laser.laser_controller import LaserController
from src.alice.laser.laser_hardware import HardwareLaserDriver
from src.alice.laser.laser_simulator import SimulatedLaserDriver

# Set up logging
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_simulator_laser():
    """Test the laser controller with simulator for comparison."""
    logger.info("\n=== Laser Controller Simulator Demo ===")
    
    try:
        # Initialize simulator driver - requires queue and laser_info
        from queue import Queue
        from src.utils.data_structures import LaserInfo, Pulse
        
        # Create required objects for simulator
        pulses_queue = Queue()
        laser_info = LaserInfo(
            max_power_mW=1000.0,
            pulse_width_fwhm_ns=1000.0,
            central_wavelength_nm=1550.0
        )
        
        simulator_driver = SimulatedLaserDriver(pulses_queue, laser_info)
        
        # Create laser controller with context manager
        with LaserController(simulator_driver) as laser:
            # Test basic operations
            logger.info("1. Testing simulator operations...")
            
            # Single pulse
            laser.trigger_once()
            
            # Frame
            laser.send_frame(n_triggers=3, rep_rate_hz=1000.0)
            
            # Continuous mode briefly
            laser.start_continuous(1000.0)
            time.sleep(0.5)
            laser.stop_continuous()
            
            # Status
            status = laser.get_status()
            logger.info(f"Simulator status: {status}")
            
            # Show some pulses that were generated
            pulse_count = 0
            while not pulses_queue.empty() and pulse_count < 5:
                pulse = pulses_queue.get()
                logger.info(f"Generated pulse: {pulse}")
                pulse_count += 1
        
        logger.info("Simulator laser test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during simulator laser test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demo function."""
    logger.info("Starting laser simulator controller demo...")
    
    # Test simulator
    success = test_simulator_laser()
    if success:
        logger.info("\n🎉 Simulated laser controller demo completed successfully!")
    else:
        logger.error("\n💥 Simulated laser controller demo failed!")

    # Return success status
    return success

if __name__ == "__main__":
    main()
