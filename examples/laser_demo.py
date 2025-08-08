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


def test_hardware_laser():
    """Test the laser controller with Digilent hardware."""
    logger.info("=== Laser Controller Hardware Demo ===")
    
    try:
        # Initialize hardware driver
        # device_index=-1 means use first available Digilent device
        # trigger_channel=0 means use analog output channel 0
        hardware_driver = HardwareLaserDriver(device_index=-1, trigger_channel=0)
        
        # Create laser controller
        laser_controller = LaserController(hardware_driver)
        
        # Initialize the system
        logger.info("1. Initializing laser controller...")
        if not laser_controller.initialize():
            logger.error("Failed to initialize laser controller")
            return False
        
        # Check status
        status = laser_controller.get_status()
        logger.info(f"Initial status: {status}")
        
        # Turn on laser
        logger.info("2. Turning on laser...")
        laser_controller.turn_on()
        
        # Test single pulse
        logger.info("3. Firing single pulse...")
        laser_controller.trigger_once()
        time.sleep(0.1)
        
        # Test frame (multiple pulses)
        logger.info("4. Sending frame of 5 pulses at 1kHz...")
        laser_controller.send_frame(n_triggers=5, rep_rate_hz=1000.0)
        time.sleep(0.1)
        
        # Test continuous mode
        logger.info("5. Starting continuous mode at 500Hz for 2 seconds...")
        laser_controller.start_continuous(500.0)
        time.sleep(2.0)
        laser_controller.stop_continuous()
        
        # Test pulse parameter adjustment
        logger.info("6. Testing pulse parameter adjustment...")
        if hasattr(hardware_driver, 'set_pulse_parameters'):
            hardware_driver.set_pulse_parameters(
                amplitude=3.3,  # 3.3V trigger
                width=2e-6,     # 2μs pulse width
                frequency=2000.0  # 2kHz
            )
            
            # Fire a burst with new parameters
            laser_controller.send_frame(n_triggers=3, rep_rate_hz=2000.0)
        
        # Final status
        final_status = laser_controller.get_status()
        logger.info(f"Final status: {final_status}")
        
        # Turn off laser
        logger.info("7. Turning off laser...")
        laser_controller.turn_off()
        
        # Shutdown
        laser_controller.shutdown()
        
        logger.info("Hardware laser test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during hardware laser test: {e}")
        return False


def test_simulator_laser():
    """Test the laser controller with simulator for comparison."""
    logger.info("\n=== Laser Controller Simulator Demo ===")
    
    try:
        # Initialize simulator driver
        simulator_driver = SimulatedLaserDriver()
        
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
        
        logger.info("Simulator laser test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during simulator laser test: {e}")
        return False


def test_both_modes():
    """Test both hardware and simulator modes."""
    logger.info("=== Testing Both Laser Modes ===")
    
    # Test simulator first (always available)
    sim_success = test_simulator_laser()
    
    # Test hardware (may fail if no Digilent device available)
    hw_success = test_hardware_laser()
    
    if sim_success and hw_success:
        logger.info("✅ Both hardware and simulator tests passed!")
    elif sim_success:
        logger.info("✅ Simulator test passed, ❌ hardware test failed (device may not be available)")
    else:
        logger.error("❌ Tests failed")
    
    return sim_success


def main():
    """Main demo function."""
    logger.info("Starting laser controller demo...")
    
    # # Test both modes
    # success = test_both_modes()
    
    # if success:
    #     logger.info("\n🎉 Laser controller demo completed successfully!")
    # else:
    #     logger.error("\n💥 Laser controller demo failed!")
    
    # return success

    # Test hardware
    success = test_hardware_laser()
    if success:
        logger.info("\n🎉 Hardware laser controller demo completed successfully!")
    else:
        logger.error("\n💥 Hardware laser controller demo failed!")

    # Return success status
    return success

if __name__ == "__main__":
    main()
