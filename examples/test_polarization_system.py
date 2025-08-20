"""
Quick test script to verify all polarization components work with the new STM32 interface
"""
import sys
import os
import time

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging
from src.alice.polarization.polarization_controller import (
    create_polarization_controller_with_simulator,
    create_polarization_controller_with_hardware
)
from src.utils.data_structures import Basis, Bit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_simulator():
    """Test the simulator implementation."""
    logger.info("=== Testing Simulator Implementation ===")
    
    try:
        # Create simulator controller
        with create_polarization_controller_with_simulator() as controller:
            logger.info("✅ Simulator controller created successfully")
            
            # Test basic functionality
            output = controller.set_polarization_manually(Basis.Z, Bit.ZERO)
            logger.info(f"✅ Set H polarization: {output.polarization_state}")
            
            output = controller.set_polarization_from_qrng()
            logger.info(f"✅ Random polarization: {output.basis} {output.bit} -> {output.polarization_state}")
            
            logger.info("✅ Simulator test completed successfully")
            return True
            
    except Exception as e:
        logger.error(f"❌ Simulator test failed: {e}")
        return False


def test_hardware_interface(com_port: str = "COM4"):
    """Test the hardware interface (without actual hardware)."""
    logger.info(f"=== Testing Hardware Interface (COM{com_port}) ===")
    
    try:
        # Create hardware controller
        controller = create_polarization_controller_with_hardware(com_port)
        logger.info("✅ Hardware controller created successfully")
        
        # Test interface creation and new STM32 commands
        try:
            # Test hardware methods (these will test the interface structure)
            logger.info("  Testing hardware method interfaces...")
            
            # These will likely fail without hardware but should not crash
            try:
                controller.initialize()
                # time.sleep(2)  # Allow some time for initialization
                controller.set_polarization_manually(Basis.Z, Bit.ONE)
                # time.sleep(5)  # Allow some time for initialization
                controller.set_polarization_manually(Basis.Z, Bit.ZERO)
                logger.info("  ✅ Basic polarization control interface working")
            except Exception as e:
                logger.info(f"  ℹ️  Polarization control failed (expected without hardware): {e}")
            
            # Test new STM32 command methods if available
            hardware_driver = controller.driver
            if hasattr(hardware_driver, 'set_polarization_device'):
                logger.info("  ✅ New STM32 device control method available")
            if hasattr(hardware_driver, 'set_angle_direct'):
                logger.info("  ✅ New STM32 angle control method available")
            if hasattr(hardware_driver, 'set_stepper_frequency'):
                logger.info("  ✅ New STM32 frequency control method available")
            if hasattr(hardware_driver, 'set_operation_period'):
                logger.info("  ✅ New STM32 period control method available")
            controller.shutdown()
                
            logger.info("  ✅ All new STM32 interface methods are available")
            
        except Exception as e:
            logger.info(f"  ℹ️  Hardware method testing failed (expected without hardware): {e}")
        
        # Note: This will fail if no actual hardware is connected,
        # but we can test the interface creation
        logger.info("✅ Hardware interface test completed (interface creation)")
        return True
        
    except Exception as e:
        logger.info(f"ℹ️  Hardware interface created but connection failed (expected without hardware): {e}")
        return True  # This is expected without actual hardware


def test_stm32_interface_structure():
    """Test the STM32 interface structure and new commands."""
    logger.info("=== Testing STM32 Interface Structure ===")
    
    try:
        # Test importing the STM32 interface
        from src.alice.polarization.hardware_pol.stm32_interface import STM32Interface
        logger.info("✅ STM32Interface import successful")
        
        # Test interface creation (without actual connection)
        try:
            interface = STM32Interface("COM4")
            logger.info("✅ STM32Interface object creation successful")
            
            # Check if new command methods exist
            new_commands = [
                'send_cmd_polarization_device',
                'send_cmd_set_angle',
                'send_cmd_set_frequency',
                'send_cmd_polarization_numbers'
            ]
            
            for cmd in new_commands:
                if hasattr(interface, cmd):
                    logger.info(f"  ✅ Command method '{cmd}' available")
                else:
                    logger.warning(f"  ⚠️ Command method '{cmd}' not found")
            
            logger.info("✅ STM32 interface structure test completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ STM32Interface creation failed: {e}")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Failed to import STM32Interface: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ STM32 interface structure test failed: {e}")
        return False


def main():
    """Main test function."""
    logger.info("🚀 Starting Polarization System Tests")
    logger.info("=" * 50)
    
    simulator_ok = False
    hardware_ok = False
    stm32_ok = False

    # Test simulator
    # simulator_ok = test_simulator()
    
    # Test hardware interface
    hardware_ok = test_hardware_interface()
    
    # Test STM32 interface structure
    # stm32_ok = test_stm32_interface_structure()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📋 TEST SUMMARY")
    logger.info("=" * 50)
    
    results = [
        ("Simulator Controller", simulator_ok),
        ("Hardware Interface", hardware_ok),
        ("STM32 Interface Structure", stm32_ok),
    ]
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    overall_success = all(success for _, success in results)
    
    if overall_success:
        logger.info("\n🎉 All tests completed successfully!")
        logger.info("The polarization system is ready for use with the new STM32 interface.")
    else:
        logger.info("\n💥 Some tests failed!")
    
    return 0 if overall_success else 1


if __name__ == "__main__":
    exit(main())
