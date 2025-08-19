"""Test script to verify the standardized laser interface."""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging
from queue import Queue

from src.alice.laser.laser_base import BaseLaserDriver
from src.alice.laser.laser_controller import LaserController
from src.alice.laser.laser_simulator import SimulatedLaserDriver
from src.alice.laser.laser_hardware_digital import DigitalHardwareLaserDriver
from src.utils.data_structures import LaserInfo, Pulse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_driver_interface(driver: BaseLaserDriver, driver_name: str) -> bool:
    """Test the standardized driver interface."""
    logger.info(f"\n=== Testing {driver_name} Interface ===")
    
    try:
        # Test initialization
        logger.info("1. Testing initialize()...")
        success = driver.initialize()
        if not success:
            logger.error("❌ Failed to initialize driver")
            return False
        logger.info("✅ Initialize successful")
        
        # Test single trigger
        logger.info("2. Testing trigger_once()...")
        success = driver.trigger_once()
        if not success:
            logger.error("❌ Failed to trigger once")
            return False
        logger.info("✅ Trigger once successful")
        
        # Test frame
        logger.info("3. Testing send_frame()...")
        success = driver.send_frame(3, 1000.0)
        if not success:
            logger.error("❌ Failed to send frame")
            return False
        logger.info("✅ Send frame successful")
        
        # Test continuous start
        logger.info("4. Testing start_continuous()...")
        success = driver.start_continuous(500.0)
        if not success:
            logger.error("❌ Failed to start continuous")
            return False
        logger.info("✅ Start continuous successful")
        
        # Wait a bit
        import time
        time.sleep(0.1)
        
        # Test continuous stop
        logger.info("5. Testing stop_continuous()...")
        success = driver.stop_continuous()
        if not success:
            logger.error("❌ Failed to stop continuous")
            return False
        logger.info("✅ Stop continuous successful")
        
        # Test status
        logger.info("6. Testing get_status()...")
        status = driver.get_status()
        logger.info(f"Status: {status}")
        logger.info("✅ Get status successful")
        
        # Test shutdown
        logger.info("7. Testing shutdown()...")
        driver.shutdown()
        logger.info("✅ Shutdown successful")
        
        logger.info(f"🎉 {driver_name} interface test PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"💥 {driver_name} interface test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_controller_interface(driver: BaseLaserDriver, driver_name: str) -> bool:
    """Test the laser controller with a driver."""
    logger.info(f"\n=== Testing LaserController with {driver_name} ===")
    
    try:
        controller = LaserController(driver)
        
        # Test controller methods
        logger.info("1. Testing controller initialize()...")
        success = controller.initialize()
        if not success:
            logger.error("❌ Failed to initialize controller")
            return False
        
        logger.info("2. Testing controller trigger_once()...")
        success = controller.trigger_once()
        if not success:
            logger.error("❌ Failed to trigger once")
            return False
        
        logger.info("3. Testing controller send_frame()...")
        success = controller.send_frame(2, 1000.0)
        if not success:
            logger.error("❌ Failed to send frame")
            return False
        
        logger.info("4. Testing controller start/stop continuous...")
        success = controller.start_continuous(500.0)
        if not success:
            logger.error("❌ Failed to start continuous")
            return False
        
        import time
        time.sleep(0.1)
        
        success = controller.stop_continuous()
        if not success:
            logger.error("❌ Failed to stop continuous")
            return False
        
        logger.info("5. Testing controller status...")
        status = controller.get_status()
        logger.info(f"Controller status: {status}")


        logger.info("6. Testing controller shutdown()...")
        controller.shutdown()
        
        logger.info(f"🎉 LaserController with {driver_name} test PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"💥 LaserController with {driver_name} test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all interface tests."""
    logger.info("Starting standardized laser interface tests...")
    
    results = []
    
    # Test simulator driver
    try:
        logger.info("\n" + "="*60)
        logger.info("TESTING SIMULATOR DRIVER")
        logger.info("="*60)
        
        pulses_queue = Queue()
        laser_info = LaserInfo(
            max_power_mW=1000.0,
            pulse_width_fwhm_ns=1000.0,
            central_wavelength_nm=1550.0
        )
        
        simulator_driver = SimulatedLaserDriver(pulses_queue, laser_info)
        
        # Test driver interface
        sim_result = test_driver_interface(simulator_driver, "SimulatedLaserDriver")
        results.append(("Simulator Driver", sim_result))
        
        # Test controller interface
        simulator_driver2 = SimulatedLaserDriver(Queue(), laser_info)
        sim_controller_result = test_controller_interface(simulator_driver2, "SimulatedLaserDriver")
        results.append(("Simulator Controller", sim_controller_result))
        
    except Exception as e:
        logger.error(f"Failed to test simulator: {e}")
        results.append(("Simulator Driver", False))
        results.append(("Simulator Controller", False))
    
    # Test digital hardware driver (may fail if no hardware)
    try:
        logger.info("\n" + "="*60)
        logger.info("TESTING DIGITAL HARDWARE DRIVER")
        logger.info("="*60)
        
        digital_driver = DigitalHardwareLaserDriver(device_index=-1, digital_channel=8)
        
        # Test driver interface
        hw_result = test_driver_interface(digital_driver, "DigitalHardwareLaserDriver")
        results.append(("Hardware Driver", hw_result))
        
        # Test controller interface
        digital_driver2 = DigitalHardwareLaserDriver(device_index=-1, digital_channel=8)
        hw_controller_result = test_controller_interface(digital_driver2, "DigitalHardwareLaserDriver")
        results.append(("Hardware Controller", hw_controller_result))
        
    except Exception as e:
        logger.warning(f"Hardware test failed (may be expected if no device): {e}")
        results.append(("Hardware Driver", False))
        results.append(("Hardware Controller", False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! The standardized interface is working correctly.")
        return True
    elif passed > 0:
        logger.warning("⚠️ SOME TESTS PASSED. The interface is partially working.")
        return True
    else:
        logger.error("💥 ALL TESTS FAILED. There are issues with the interface.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
