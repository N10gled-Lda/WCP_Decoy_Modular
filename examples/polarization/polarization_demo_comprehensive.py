"""Comprehensive polarization controller demo supporting both simulator and hardware testing."""
import sys
import os


# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        self.logger.info("\n\n========= Testing Polarization Controller with Simulator =========")
        
        try:
            # Create input/output queues
            pulses_queue = Queue()
            polarized_pulses_queue = Queue()
            
            # Create laser info
            laser_info = LaserInfo(wavelength=1550.0, power=1.0, pulse_width=1e-9)
            
            # Create controller with simulator
            with create_polarization_controller_with_simulator(
                pulses_queue=pulses_queue,
                polarized_pulses_queue=polarized_pulses_queue,
                laser_info=laser_info
            ) as controller:
                
                if not controller.is_initialized():
                    self.logger.warning("Trying again to initialize simulator controller...")
                    controller.initialize()
                    if not controller.is_initialized():
                        self.logger.error("Failed to initialize simulator controller")
                        return False
                
                self.logger.info("✅ Simulator controller initialized")
                
                # Test 1: Random polarization generation
                self.logger.info("\n🎲 Testing random polarization generation:")
                for i in range(3):
                    pulses_queue.put(Pulse(polarization=10, photons=1000))
                    output = controller.set_polarization_from_qrng()
                    controller.apply_polarization_to_queue()
                    self.logger.info(f"  {i+1}. {output.basis} basis, bit {output.bit} → "
                                   f"{output.polarization_state} ({output.angle_degrees}°)")

                status = controller.get_queue_info()
                self.logger.info(f"  Queue status: {status}")

                self.logger.info("\n📦 Polarized pulses generated:")
                while not polarized_pulses_queue.empty():
                    pulse = polarized_pulses_queue.get()
                    self.logger.info(f"  {pulse.polarization}° ({pulse.photons} photons)")

                # Test 2: Manual polarization setting
                self.logger.info("\n🔧 Testing manual polarization setting:")
                test_cases = [
                    (Basis.Z, Bit.ZERO, "Horizontal"),
                    (Basis.Z, Bit.ONE, "Vertical"), 
                    (Basis.X, Bit.ZERO, "Diagonal"),
                    (Basis.X, Bit.ONE, "Anti-diagonal")
                ]
                
                for basis, bit, description in test_cases:
                    pulses_queue.put(Pulse(polarization=10, photons=1000))
                    output = controller.set_polarization_manually(basis, bit)
                    controller.apply_polarization_to_queue()
                    self.logger.info(f"  {description}: {output.polarization_state} ({output.angle_degrees}°)")
                
                logger.info("\n📦 Polarized pulses after manual setting:")
                while not polarized_pulses_queue.empty():
                    pulse = polarized_pulses_queue.get()
                    self.logger.info(f"  {pulse.polarization}° ({pulse.photons} photons)")

                # Test 3: Queue processing
                self.logger.info("\n📦 Testing queue processing:")
                
                # Add test pulses
                test_pulses = [
                    Pulse(polarization=PolarizationState.H, photons=1000),
                    Pulse(polarization=PolarizationState.V, photons=1100),
                    Pulse(polarization=PolarizationState.D, photons=800),
                    Pulse(polarization=PolarizationState.A, photons=900),
                ]
                
                for i, pulse in enumerate(test_pulses):
                    pulses_queue.put(pulse)
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
                self.logger.info(f"  Bit value: {state_info['bit']}, Basis: {state_info['basis']}")
                self.logger.info(f"  Jones vector: {state_info['jones_vector']}")
                
                self.logger.info("✅ Simulator controller test completed successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Simulator controller test failed: {e}")
            return False
    
    def test_hardware_controller(self, com_port: str) -> bool:
        """Test polarization controller with hardware."""
        self.logger.info(f"\n\n========= Testing Polarization Controller with Hardware ({com_port}) =========")
        
        try:
            # Create controller with hardware
            with create_polarization_controller_with_hardware(com_port=com_port) as controller:
                
                if not controller.is_initialized():
                    self.logger.warning("Trying again to initialize hardware controller...")
                    controller.initialize()
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
                
                # Test 4: STM32 commands through controller
                self.logger.info("\n🔧 Testing other STM32 commands through controller:")
                try:
                    hardware_driver = controller.driver
                    if hasattr(hardware_driver, 'set_polarization_device'):
                        self.logger.info("  Testing polarization device control...")
                        success = hardware_driver.set_polarization_device(device=1)
                        self.logger.info(f"    Set to Linear Polarizer: {'✅ Success' if success else '❌ Failed'}")
                        time.sleep(0.5)
                        
                        success = hardware_driver.set_polarization_device(device=2) 
                        self.logger.info(f"    Set to Half Wave Plate: {'✅ Success' if success else '❌ Failed'}")
                        time.sleep(0.5)
                    else:
                        self.logger.warning("  No set_polarization_device method available in hardware driver")
                    
                    if hasattr(hardware_driver, 'set_angle_direct'):
                        self.logger.info("  Testing direct angle control...")
                        success = hardware_driver.set_angle_direct(30)
                        self.logger.info(f"    Set angle to 30°: {'✅ Success' if success else '❌ Failed'}")
                        time.sleep(0.5)
                        
                        success = hardware_driver.set_angle_direct(60, is_offset=True)
                        self.logger.info(f"    Set angle with offset to 60°: {'✅ Success' if success else '❌ Failed'}")
                        time.sleep(0.5)
                    else:
                        self.logger.warning("  No set_angle_direct method available in hardware driver")
                    
                    if hasattr(hardware_driver, 'set_stepper_frequency'):
                        self.logger.info("  Testing stepper frequency control...")
                        success = hardware_driver.set_stepper_frequency(500)
                        self.logger.info(f"    Set stepper frequency to 500 Hz: {'✅ Success' if success else '❌ Failed'}")
                        time.sleep(0.5)
                    else:
                        self.logger.warning("  No set_stepper_frequency method available in hardware driver")
                    
                    if hasattr(hardware_driver, 'set_operation_period'):
                        self.logger.info("  Testing operation period control...")
                        success = hardware_driver.set_operation_period(10)
                        self.logger.info(f"    Set operation period to 10 ms: {'✅ Success' if success else '❌ Failed'}")
                        time.sleep(0.5)
                    else:
                        self.logger.warning("  No set_operation_period method available in hardware driver")
                        
                    self.logger.info("  ✅ New STM32 command testing completed")
                except Exception as e:
                    self.logger.error(f"  ❌ New STM32 command testing failed: {e}")
                
                
                # Test 5: Set multiples polarizations with a certain period and freq:
                logger.info("\n🔧 Testing setting multiple polarization states:")
                try:
                    states = [0,1,2,3,1]
                    period = 1000 # Period between states of 1s
                    freq = 500 # Stepper frequency of 500Hz
                    success = controller.set_polarization_multiple_states(states, period=period, stepper_freq=freq)
                    self.logger.info(f"  Set multiple states {states} with period {period} ms and freq {freq} Hz: {'✅ Success' if success else '❌ Failed'}")
                    # Wait until the device available meaning that its ended the sequence
                    start = time.time()
                    while not controller.is_available():
                        time.sleep(0.01)
                        # self.logger.info(f"  Waiting for device to finish the sequence... time pass: {time.time() - start}")
                    self.logger.info(f"  Device is now available after {time.time() - start} seconds")

                    self.logger.info("✅ Multiple polarization states test completed")
                    
                except Exception as e:
                    self.logger.error(f"  ❌ Setting multiple polarization states failed: {e}")

                # Test 6: State verification
                self.logger.info("\n📊 Testing state verification:")
                current_state = controller.get_current_state()
                self.logger.info(f"  Current state: {current_state}")
                
                self.logger.info("✅ Hardware controller test completed successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Hardware controller test failed: {e}")
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
    
    def run_complete_demo(self, com_port: str = None) -> bool:
        """Run the complete polarization controller demo."""
        self.logger.info("🚀 Starting Polarization Controller Comprehensive Demo")
        
        # Always test simulator
        simulator_success = self.test_simulator_controller()
        
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
        
        # Summary
        self.logger.info("\n" + "="*60)
        self.logger.info("📋 DEMO SUMMARY")
        self.logger.info("="*60)
        
        results = [
            ("Simulator Controller", simulator_success),
            ("Hardware Controller", hardware_success)
        ]
        
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
    parser.add_argument("--simulator-only", "--s", action="store_true", help="Test simulator only")
    parser.add_argument("--hardware-only", "--h", action="store_true", help="Test hardware only")
    parser.add_argument("--com-port", "--cp", type=str, help="COM port for hardware testing (e.g., COM4)")
    parser.add_argument("--list-com-ports", "--lcp", action="store_true", help="List available COM ports")

    args = parser.parse_args()
    
    print("Polarization Controller Comprehensive Demo")
    print("=" * 50)
    
    if args.list_com_ports:
        available_ports = PolarizationControllerDemo().list_available_com_ports()
        if available_ports:
            print("Available COM ports:")
            for port in available_ports:
                print(f"  - {port}")
        else:
            print("No COM ports available.")
        return 0

    demo = PolarizationControllerDemo()
    
    if args.simulator_only:
        success = demo.test_simulator_controller()
    elif args.hardware_only:
        if not args.com_port:
            print("❌ COM port must be specified for hardware testing.")
            return 1
        success = demo.test_hardware_controller(args.com_port)
    else:
        success = demo.run_complete_demo(com_port=args.com_port)
    
    if success:
        print("\n🎉 Demo completed successfully!")
        return 0
    else:
        print("\n💥 Demo failed!")
        return 1


if __name__ == "__main__":
    exit(main())
