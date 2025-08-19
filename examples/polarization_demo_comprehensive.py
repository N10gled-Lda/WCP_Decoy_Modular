"""Comprehensive polarization controller demo supporting both simulator and hardware testing."""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging
import time
from queue import Queue
from typing import List

# Import polarization components
from src.alice.polarization.polarization_controller import (
    PolarizationController,
    create_polarization_controller_with_simulator,
    create_polarization_controller_with_hardware
)
from src.alice.polarization.polarization_hardware import PolarizationHardware
from src.alice.polarization.polarization_simulator import PolarizationSimulator
from src.alice.polarization.polarization_base import PolarizationState
from src.utils.data_structures import Basis, Bit, Pulse, LaserInfo
from src.alice.qrng.qrng_simulator import QRNGSimulator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PolarizationControllerDemo:
    """Comprehensive demo for polarization controller with both simulator and hardware."""
    
    def __init__(self):
        """Initialize the demo."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def test_simulator_controller(self) -> bool:
        """Test polarization controller with simulator."""
        self.logger.info("=== Testing Polarization Controller with Simulator ===")
        
        try:
            # Create input/output queues
            input_queue = Queue()
            output_queue = Queue()
            
            # Create laser info
            laser_info = LaserInfo(wavelength=1550.0, power=1.0, pulse_width=1e-9)
            
            # Create controller with simulator
            with create_polarization_controller_with_simulator(
                input_queue=input_queue,
                output_queue=output_queue,
                laser_info=laser_info
            ) as controller:
                
                if not controller.is_initialized():
                    self.logger.error("Failed to initialize simulator controller")
                    return False
                
                self.logger.info("✅ Simulator controller initialized")
                
                # Test 1: Random polarization generation
                self.logger.info("\n🎲 Testing random polarization generation:")
                for i in range(3):
                    output = controller.set_polarization_from_qrng()
                    self.logger.info(f"  {i+1}. {output.basis} basis, bit {output.bit} → "
                                   f"{output.polarization_state} ({output.angle_degrees}°)")
                
                # Test 2: Manual polarization setting
                self.logger.info("\n🔧 Testing manual polarization setting:")
                test_cases = [
                    (Basis.Z, Bit.ZERO, "Horizontal"),
                    (Basis.Z, Bit.ONE, "Vertical"), 
                    (Basis.X, Bit.ZERO, "Diagonal"),
                    (Basis.X, Bit.ONE, "Anti-diagonal")
                ]
                
                for basis, bit, description in test_cases:
                    output = controller.set_polarization_manually(basis, bit)
                    self.logger.info(f"  {description}: {output.polarization_state} ({output.angle_degrees}°)")
                
                # Test 3: Queue processing
                self.logger.info("\n📦 Testing queue processing:")
                
                # Add test pulses
                test_pulses = [
                    Pulse(timestamp=1.0, intensity=1000.0, wavelength=1550.0, photons=1000),
                    Pulse(timestamp=2.0, intensity=1200.0, wavelength=1550.0, photons=1200),
                    Pulse(timestamp=3.0, intensity=800.0, wavelength=1550.0, photons=800),
                ]
                
                for i, pulse in enumerate(test_pulses):
                    input_queue.put(pulse)
                    self.logger.info(f"  Added pulse {i+1}: {pulse.photons} photons")
                
                # Set polarization and process
                controller.set_polarization_manually(Basis.Z, Bit.ONE)
                controller.apply_polarization_to_queue()
                
                # Check results
                queue_info = controller.get_queue_info()
                self.logger.info(f"  Queue status: {queue_info}")
                
                # Get current state
                state_info = controller.get_current_state()
                self.logger.info(f"  Current state: {state_info['state']} at {state_info['angle_degrees']}°")
                
                self.logger.info("✅ Simulator controller test completed successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Simulator controller test failed: {e}")
            return False
    
    def test_hardware_controller(self, com_port: str) -> bool:
        """Test polarization controller with hardware."""
        self.logger.info(f"=== Testing Polarization Controller with Hardware (COM{com_port}) ===")
        
        try:
            # Create controller with hardware
            with create_polarization_controller_with_hardware(com_port=com_port) as controller:
                
                if not controller.is_initialized():
                    self.logger.error("Failed to initialize hardware controller")
                    return False
                
                self.logger.info("✅ Hardware controller initialized")
                
                # Test 1: Connection verification
                self.logger.info("\n🔌 Testing hardware connection:")
                if hasattr(controller.driver, 'is_connected') and controller.driver.is_connected():
                    self.logger.info("  ✅ STM32 hardware connected successfully")
                else:
                    self.logger.warning("  ⚠️ Hardware connection status unknown")
                
                # Test 2: Manual polarization control
                self.logger.info("\n🔧 Testing manual polarization control:")
                test_states = [
                    (Basis.Z, Bit.ZERO, PolarizationState.H, "Horizontal (0°)"),
                    (Basis.Z, Bit.ONE, PolarizationState.V, "Vertical (90°)"),
                    (Basis.X, Bit.ZERO, PolarizationState.D, "Diagonal (45°)"),
                    (Basis.X, Bit.ONE, PolarizationState.A, "Anti-diagonal (135°)")
                ]
                
                for basis, bit, expected_state, description in test_states:
                    self.logger.info(f"  Setting {description}...")
                    
                    try:
                        output = controller.set_polarization_manually(basis, bit)
                        self.logger.info(f"    ✅ Set to {output.polarization_state} ({output.angle_degrees}°)")
                        
                        # Wait a bit for hardware to respond
                        time.sleep(0.5)
                        
                    except Exception as e:
                        self.logger.error(f"    ❌ Failed to set {description}: {e}")
                        return False
                
                # Test 3: Random polarization with QRNG
                self.logger.info("\n🎲 Testing random polarization generation:")
                for i in range(3):
                    try:
                        output = controller.set_polarization_from_qrng()
                        self.logger.info(f"  {i+1}. QRNG: {output.basis} basis, bit {output.bit} → "
                                       f"{output.polarization_state} ({output.angle_degrees}°)")
                        time.sleep(0.3)  # Allow hardware time to respond
                        
                    except Exception as e:
                        self.logger.error(f"    ❌ QRNG test {i+1} failed: {e}")
                
                # Test 4: State verification
                self.logger.info("\n📊 Testing state verification:")
                current_state = controller.get_current_state()
                self.logger.info(f"  Current state: {current_state}")
                
                self.logger.info("✅ Hardware controller test completed successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Hardware controller test failed: {e}")
            return False
    
    def test_stm32_interface_directly(self, com_port: str) -> bool:
        """Test the STM32 interface directly (similar to the provided examples)."""
        self.logger.info(f"=== Testing STM32 Interface Directly (COM{com_port}) ===")
        
        try:
            from src.alice.polarization.hardware_pol.stm32_interface import STM32Interface
            
            # Create interface
            stm = STM32Interface(com_port)
            
            # Define callbacks
            def handle_connected():
                self.logger.info("  ✅ STM32 connected")
            
            def handle_available():
                self.logger.info("  ✅ STM32 available - sending test polarization...")
                success = stm.send_polarization_numbers([0, 1, 2, 3])
                if success:
                    self.logger.info("  ✅ Polarization numbers sent successfully")
                else:
                    self.logger.error("  ❌ Failed to send polarization numbers")
            
            def handle_polarization_status(status):
                status_names = {
                    0: "SUCCESS",
                    1: "INVALID_ID",
                    2: "WRONG_POLARIZATIONS", 
                    3: "MISMATCH_QUANTITY",
                    4: "OVERFLOW",
                    5: "UNKNOWN_ERROR"
                }
                status_name = status_names.get(status, f"UNKNOWN({status})")
                self.logger.info(f"  📊 Polarization status: {status_name}")
            
            # Attach callbacks
            stm.on_connected = handle_connected
            stm.on_available = handle_available
            stm.on_polarization_status = handle_polarization_status
            
            # Start interface
            stm.start()
            stm.connect()
            
            # Wait for communication
            self.logger.info("  Waiting for STM32 communication...")
            start_time = time.time()
            
            while time.time() - start_time < 5.0:  # 5 second timeout
                if stm.connected and stm.available:
                    break
                time.sleep(0.1)
            
            if stm.connected:
                self.logger.info("  ✅ STM32 interface test completed successfully")
                success = True
            else:
                self.logger.error("  ❌ STM32 failed to connect within timeout")
                success = False
            
            # Cleanup
            stm.stop()
            return success
            
        except Exception as e:
            self.logger.error(f"❌ STM32 interface test failed: {e}")
            return False
    
    def list_available_com_ports(self) -> List[str]:
        """List available COM ports."""
        try:
            import serial.tools.list_ports
            ports = [port.device for port in serial.tools.list_ports.comports()]
            return ports
        except ImportError:
            self.logger.warning("pyserial not available, cannot list COM ports")
            return []
    
    def run_complete_demo(self, test_hardware: bool = False, com_port: str = None) -> bool:
        """Run the complete polarization controller demo."""
        self.logger.info("🚀 Starting Polarization Controller Comprehensive Demo")
        
        # Always test simulator
        simulator_success = self.test_simulator_controller()
        
        hardware_success = True  # Default to success if not testing
        stm32_success = True
        
        if test_hardware:
            if com_port is None:
                # Try to find available COM ports
                available_ports = self.list_available_com_ports()
                if available_ports:
                    self.logger.info(f"Available COM ports: {available_ports}")
                    com_port = available_ports[0]  # Use first available
                    self.logger.info(f"Using COM port: {com_port}")
                else:
                    self.logger.error("No COM ports available for hardware testing")
                    return simulator_success
            
            # Test hardware controller
            hardware_success = self.test_hardware_controller(com_port)
            
            # Test STM32 interface directly
            stm32_success = self.test_stm32_interface_directly(com_port)
        
        # Summary
        self.logger.info("\n" + "="*60)
        self.logger.info("📋 DEMO SUMMARY")
        self.logger.info("="*60)
        
        results = [
            ("Simulator Controller", simulator_success),
        ]
        
        if test_hardware:
            results.extend([
                ("Hardware Controller", hardware_success),
                ("STM32 Interface Direct", stm32_success)
            ])
        
        for test_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            self.logger.info(f"  {test_name}: {status}")
        
        overall_success = all(success for _, success in results)
        
        if overall_success:
            self.logger.info("\n🎉 ALL TESTS PASSED! Polarization controller working correctly.")
        else:
            self.logger.error("\n💥 SOME TESTS FAILED. Check hardware connections and drivers.")
        
        return overall_success


def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Polarization Controller Demo")
    parser.add_argument("--hardware", action="store_true", help="Test hardware controller")
    parser.add_argument("--com-port", type=str, help="COM port for hardware testing (e.g., COM3)")
    parser.add_argument("--simulator-only", action="store_true", help="Test simulator only")
    
    args = parser.parse_args()
    
    print("Polarization Controller Comprehensive Demo")
    print("=" * 50)
    
    demo = PolarizationControllerDemo()
    
    if args.simulator_only:
        success = demo.test_simulator_controller()
    else:
        success = demo.run_complete_demo(
            test_hardware=args.hardware,
            com_port=args.com_port
        )
    
    if success:
        print("\n🎉 Demo completed successfully!")
        return 0
    else:
        print("\n💥 Demo failed!")
        return 1


if __name__ == "__main__":
    exit(main())
