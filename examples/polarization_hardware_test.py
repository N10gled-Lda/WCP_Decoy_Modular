"""Simple hardware test for polarization controller - similar to the provided STM32 examples."""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging
import time
from src.alice.polarization.polarization_controller import create_polarization_controller_with_hardware
from src.alice.polarization.polarization_base import PolarizationState
from src.utils.data_structures import Basis, Bit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_polarization_hardware(com_port: str = "COM3"):
    """
    Simple test of polarization hardware - similar to main1.py example.
    
    Args:
        com_port: COM port for STM32 (default COM3)
    """
    logger.info(f"Testing Polarization Hardware on {com_port}")
    logger.info("=" * 50)
    
    try:
        # Create polarization controller with hardware
        controller = create_polarization_controller_with_hardware(com_port)
        
        # Initialize the controller
        logger.info("1. Initializing polarization controller...")
        if not controller.initialize():
            logger.error("Failed to initialize polarization controller")
            return False
        
        logger.info("✅ Polarization controller initialized successfully")
        
        # Test each BB84 polarization state
        logger.info("\n2. Testing BB84 polarization states...")
        
        test_sequence = [
            (Basis.Z, Bit.ZERO, "Horizontal (0°)"),
            (Basis.Z, Bit.ONE, "Vertical (90°)"),
            (Basis.X, Bit.ZERO, "Diagonal (45°)"),
            (Basis.X, Bit.ONE, "Anti-diagonal (135°)")
        ]
        
        for i, (basis, bit, description) in enumerate(test_sequence, 1):
            logger.info(f"  {i}. Setting {description}...")
            
            try:
                # Set polarization
                output = controller.set_polarization_manually(basis, bit)
                logger.info(f"     ✅ Successfully set to {output.polarization_state}")
                
                # Wait for hardware to respond
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"     ❌ Failed to set {description}: {e}")
        
        # Test random polarization generation
        logger.info("\n3. Testing random polarization generation...")
        for i in range(3):
            try:
                output = controller.set_polarization_from_qrng()
                logger.info(f"  Random {i+1}: {output.basis} basis, bit {output.bit} → "
                           f"{output.polarization_state} ({output.angle_degrees}°)")
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"  ❌ Random generation {i+1} failed: {e}")
        
        # Get final state
        logger.info("\n4. Final state information:")
        final_state = controller.get_current_state()
        for key, value in final_state.items():
            if key == 'jones_vector':
                logger.info(f"  {key}: [{value[0]:.3f}, {value[1]:.3f}]")
            else:
                logger.info(f"  {key}: {value}")
        
        # Shutdown
        controller.shutdown()
        logger.info("\n✅ Polarization hardware test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Polarization hardware test failed: {e}")
        return False


def test_with_gui_like_interface():
    """Test with a simple command-line interface similar to the GUI example."""
    logger.info("Polarization Controller - Interactive Test")
    logger.info("=" * 50)
    
    # Get COM port from user
    try:
        import serial.tools.list_ports
        available_ports = [port.device for port in serial.tools.list_ports.comports()]
        
        if available_ports:
            logger.info(f"Available COM ports: {available_ports}")
            com_port = input(f"Enter COM port (default: {available_ports[0]}): ").strip()
            if not com_port:
                com_port = available_ports[0]
        else:
            com_port = input("Enter COM port (e.g., COM3): ").strip()
            if not com_port:
                com_port = "COM3"
                
    except ImportError:
        com_port = input("Enter COM port (e.g., COM3): ").strip()
        if not com_port:
            com_port = "COM3"
    
    logger.info(f"Using COM port: {com_port}")
    
    try:
        # Create and initialize controller
        controller = create_polarization_controller_with_hardware(com_port)
        
        if not controller.initialize():
            logger.error("Failed to initialize controller")
            return False
        
        logger.info("✅ Controller initialized")
        
        # Interactive loop
        while True:
            print("\nPolarization Controller Menu:")
            print("1. Set Horizontal (H)")
            print("2. Set Vertical (V)")
            print("3. Set Diagonal (D)")
            print("4. Set Anti-diagonal (A)")
            print("5. Random polarization")
            print("6. Show current state")
            print("7. Exit")
            
            choice = input("Enter choice (1-7): ").strip()
            
            try:
                if choice == "1":
                    output = controller.set_polarization_manually(Basis.Z, Bit.ZERO)
                    logger.info(f"Set to Horizontal: {output.polarization_state}")
                    
                elif choice == "2":
                    output = controller.set_polarization_manually(Basis.Z, Bit.ONE)
                    logger.info(f"Set to Vertical: {output.polarization_state}")
                    
                elif choice == "3":
                    output = controller.set_polarization_manually(Basis.X, Bit.ZERO)
                    logger.info(f"Set to Diagonal: {output.polarization_state}")
                    
                elif choice == "4":
                    output = controller.set_polarization_manually(Basis.X, Bit.ONE)
                    logger.info(f"Set to Anti-diagonal: {output.polarization_state}")
                    
                elif choice == "5":
                    output = controller.set_polarization_from_qrng()
                    logger.info(f"Random: {output.basis} basis, bit {output.bit} → {output.polarization_state}")
                    
                elif choice == "6":
                    state = controller.get_current_state()
                    logger.info(f"Current state: {state}")
                    
                elif choice == "7":
                    break
                    
                else:
                    print("Invalid choice")
                    
                time.sleep(0.1)  # Small delay for hardware
                
            except Exception as e:
                logger.error(f"Error: {e}")
        
        controller.shutdown()
        logger.info("Controller shutdown")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Polarization Hardware Test")
    parser.add_argument("--com-port", type=str, default="COM3", help="COM port (default: COM3)")
    parser.add_argument("--interactive", action="store_true", help="Run interactive test")
    
    args = parser.parse_args()
    
    if args.interactive:
        success = test_with_gui_like_interface()
    else:
        success = test_polarization_hardware(args.com_port)
    
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")


if __name__ == "__main__":
    main()
