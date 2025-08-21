"""Demo for Digital Laser Control using Digilent Device."""
import queue
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import logging
import time
import argparse
from typing import List

# Import digital components
from src.utils.data_structures import LaserInfo
from src.alice.laser.hardware_laser.digilent_digital_interface import (
    DigilentDigitalInterface, 
    list_digital_devices
)
from src.alice.laser.laser_hardware_digital import (
    DigitalHardwareLaserDriver,
    LaserState,
    create_digital_laser_driver
)
from src.alice.laser.laser_simulator import SimulatedLaserDriver
from src.alice.laser.laser_controller import LaserController

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DigitalLaserDemo:
    """Comprehensive demo for digital laser control."""
    
    def __init__(self, device_index: int = -1, digital_channel: int = 8):
        """
        Initialize the demo.
        
        Args:
            device_index: Digilent device index (-1 for first available)
            digital_channel: Digital channel for triggering
        """
        self.device_index = device_index
        self.digital_channel = digital_channel
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def test_digilent_digital_interface(self) -> bool:
        """Test the low-level digital interface."""
        self.logger.info("\n\n\n=============== Digital Interface Test ===============")

        try:
            interface = DigilentDigitalInterface(device_index=self.device_index, digital_channel=self.digital_channel)
            interface.connect()
            if not interface.connected:
                self.logger.error("Failed to connect to device")
                return False
            
            self.logger.info(f"✅ Connected to device on channel {self.digital_channel}")
            
            # Test pulse parameter configuration
            self.logger.info("🔧 Testing pulse parameter configuration...")
            interface.set_pulse_parameters(duty_cycle=0.25, frequency=1000.0, idle_state=False)
            print(f"📊 Pulse configuration:"
                  f"\n   • Duty cycle: {interface.duty_cycle*100:.1f}%"
                  f"\n   • Frequency: {interface.frequency:.1f} Hz"
                  f"\n   • Idle state: {'HIGH' if interface.idle_state else 'LOW'}")

            # Test single pulse
            self.logger.info("🔸 Testing single pulse...")
            if interface.send_single_pulse():
                self.logger.info("✅ Single pulse sent successfully")
            else:
                self.logger.error("❌ Single pulse failed")
                return False
            
            time.sleep(0.1)
            
            # Test pulse train
            self.logger.info("🔸 Testing pulse train (5 pulses at 1 kHz)...")
            if interface.start_pulse_train(5, 1000.0):
                # time.sleep(5 / 1000.0 + 0.1)  # Wait for completion
                self.logger.info("✅ Pulse train completed")
            else:
                self.logger.error("❌ Pulse train failed")
                return False
            
            # Test continuous mode briefly
            self.logger.info("🔸 Testing continuous mode (1 second at 1000 Hz)...")
            if interface.start_continuous(1000.0):
                time.sleep(1.0)
                if interface.stop():
                    self.logger.info("✅ Continuous mode test completed")
                else:
                    self.logger.error("❌ Failed to stop continuous mode")
                    return False
            else:
                self.logger.error("❌ Failed to start continuous mode")
                return False
            
            # Show final status
            status = interface.get_status()
            self.logger.info(f"📊 Interface status: {status}")
            shutdown_success = interface.stop()
            if shutdown_success:
                self.logger.info("✅ Interface shutdown successful")
            else:
                self.logger.error("❌ Failed to shutdown interface")
                return False
            
            return True
                
        except Exception as e:
            self.logger.error(f"Digital interface test failed: {e}")
            return False
    
    def test_hardware_driver(self) -> bool:
        """Test the high-level hardware driver."""
        self.logger.info("\n\n\n=============== Hardware Driver Test ===============")

        try:
            with create_digital_laser_driver(self.device_index, self.digital_channel) as laser:
                if laser.state == LaserState.ERROR:
                    self.logger.error("Failed to initialize laser driver")
                    return False
                
                self.logger.info(f"✅ Laser driver initialized (state: {laser.state.value})")
                
                # Test initialization
                self.logger.info("🔸 Testing initialize()...")
                if not laser.is_initialized():
                    self.logger.info("Laser is not initialized, calling initialize()")
                    if laser.initialize():
                        self.logger.info("✅ Initialization successful")
                    else:
                        self.logger.error("❌ Initialization failed")
                    return False

                # Test single trigger
                self.logger.info("🔸 Testing single trigger...")
                if laser.trigger_once():
                    self.logger.info("✅ Single trigger successful")
                else:
                    self.logger.error("❌ Single trigger failed")
                    return False
                
                time.sleep(0.1)
                
                # Test frame with different parameters
                self.logger.info("🔸 Testing frame (3 triggers at 1 kHz)...")
                if laser.send_frame(3, 1000.0):
                    self.logger.info("✅ Frame sent successfully")
                else:
                    self.logger.error("❌ Frame sending failed")
                    return False
                
                time.sleep(0.1)
                
                # Test pulse parameter adjustment
                self.logger.info("🔧 Testing pulse parameter adjustment...")
                laser.set_pulse_parameters(duty_cycle=0.25, frequency=800.0)  # 25% duty cycle at 800Hz

                # Test with new parameters
                self.logger.info("🔸 Testing with new parameters (2 triggers)...")
                if laser.send_frame(2):
                    self.logger.info("✅ Test with new parameters successful")
                else:
                    self.logger.error("❌ Test with new parameters failed")
                    return False
                
                # Test continuous mode
                self.logger.info("🔸 Testing continuous mode (2 seconds at 500 Hz)...")
                if laser.start_continuous(500.0):
                    time.sleep(2.0)
                    if laser.stop_continuous():
                        self.logger.info("✅ Continuous mode test completed")
                    else:
                        self.logger.error("❌ Failed to stop continuous mode")
                        return False
                else:
                    self.logger.error("❌ Failed to start continuous mode")
                    return False
                
                # Show comprehensive status
                status = laser.get_status()
                self.logger.info("📊 Final driver status:")
                for key, value in status.items():
                    if isinstance(value, dict):
                        self.logger.info(f"  {key}:")
                        for sub_key, sub_value in value.items():
                            self.logger.info(f"    {sub_key}: {sub_value}")
                    else:
                        self.logger.info(f"  {key}: {value}")
                
                return True
            
        except Exception as e:
            self.logger.error(f"Hardware driver test failed: {e}")
            return False
    
    def test_controller_interface(self) -> bool:
        """Test the high-level controller interface."""
        self.logger.info("\n\n\n=============== Controller Interface Test ===============")
        
        try:
            # Create a hardware driver for the controller
            hardware_driver = DigitalHardwareLaserDriver(self.device_index, self.digital_channel)
            
            # Create controller with the hardware driver
            controller = LaserController(hardware_driver)
            
            # Test initialization
            self.logger.info("🔸 Testing controller initialization...")
            if not controller.is_initialized():
                self.logger.info("Controller is not initialized, calling initialize()")
                if not controller.initialize():
                    self.logger.error("❌ Controller initialization failed")
                    return False
                else:
                    self.logger.info("✅ Controller initialized successfully")
                        
            # Test single pulse trigger
            self.logger.info("🔸 Testing single pulse trigger...")
            if controller.trigger_once():
                self.logger.info("✅ Single pulse trigger successful")
            else:
                self.logger.error("❌ Single pulse trigger failed")
                return False
            
            time.sleep(0.1)
            
            # Test frame sending
            self.logger.info("🔸 Testing frame sending (4 pulses at 1000 Hz)...")
            if controller.send_frame(4, 1000.0):
                self.logger.info("✅ Frame sending successful")
            else:
                self.logger.error("❌ Frame sending failed")
                return False
            
            time.sleep(0.1)
            
            # Test continuous mode
            self.logger.info("🔸 Testing continuous mode (2 seconds at 500 Hz)...")
            if controller.start_continuous(500.0):
                self.logger.info(f"Controller active: {controller.is_active()}")
                time.sleep(2)
                if controller.stop_continuous():
                    self.logger.info("✅ Continuous mode test completed")
                else:
                    self.logger.error("❌ Failed to stop continuous mode")
                    return False
            else:
                self.logger.error("❌ Failed to start continuous mode")
                return False
            
            # Test status reporting
            self.logger.info("🔸 Testing status reporting...")
            status = controller.get_status()
            self.logger.info("📊 Controller status:")
            for section, data in status.items():
                if isinstance(data, dict):
                    self.logger.info(f"  {section}:")
                    for key, value in data.items():
                        self.logger.info(f"    {key}: {value}")
                else:
                    self.logger.info(f"  {section}: {data}")
            
            # Test counter reset
            self.logger.info("🔸 Testing counter reset...")
            old_pulse_count = controller.pulse_count
            controller.reset_counters()
            if controller.pulse_count == 0:
                self.logger.info(f"✅ Counters reset (was {old_pulse_count}, now {controller.pulse_count})")
            else:
                self.logger.error("❌ Counter reset failed")
                return False
            
            # Test error handling - try to trigger when not active (should work)
            self.logger.info("🔸 Testing normal operation after reset...")
            if controller.trigger_once():
                self.logger.info("✅ Normal operation after reset successful")
            else:
                self.logger.error("❌ Normal operation after reset failed")
                return False
            
            # Shutdown controller
            self.logger.info("🔸 Testing controller shutdown...")
            controller.shutdown()
            if not controller.is_initialized():
                self.logger.info("✅ Controller shutdown successful")
            else:
                self.logger.error("❌ Controller shutdown failed")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Controller interface test failed: {e}")
            return False
    
    def test_simulator_interface(self) -> bool:
        """Test the laser simulator interface."""
        self.logger.info("\n\n\n=============== Simulator Interface Test ===============")

        try:
            
            # Create required objects for simulator
            pulses_queue = queue.Queue()
            laser_info = LaserInfo(
                name="Test Simulator",
                max_power_mW=1000.0,
                pulse_width_fwhm_ns=1000.0,
                central_wavelength_nm=1550.0
            )
            
            # Create simulator driver
            simulator = SimulatedLaserDriver(pulses_queue, laser_info)
            
            # Test initialization
            self.logger.info("🔸 Testing simulator initialization...")
            if not simulator.initialize():
                self.logger.error("Failed to initialize simulator")
                return False
            
            self.logger.info("✅ Simulator initialized successfully")
            
            # Test single pulse
            self.logger.info("🔸 Testing single pulse...")
            if simulator.trigger_once():
                self.logger.info("✅ Single pulse successful")
            else:
                self.logger.error("❌ Single pulse failed")
                return False
            
            time.sleep(0.1)
            
            # Test frame
            self.logger.info("🔸 Testing frame (3 pulses at 1500 Hz)...")
            if simulator.send_frame(3, 1500.0):
                self.logger.info("✅ Frame sent successfully")
            else:
                self.logger.error("❌ Frame sending failed")
                return False
            
            time.sleep(0.1)
            
            # Test continuous mode
            self.logger.info("🔸 Testing continuous mode (1 second at 2000 Hz)...")
            if simulator.start_continuous(2000.0):
                time.sleep(1.0)
                if simulator.stop_continuous():
                    self.logger.info("✅ Continuous mode test completed")
                else:
                    self.logger.error("❌ Failed to stop continuous mode")
                    return False
            else:
                self.logger.error("❌ Failed to start continuous mode")
                return False
            
            # Check generated pulses
            self.logger.info("🔸 Checking generated pulses...")
            pulse_count = 0
            sample_pulses = []
            while not pulses_queue.empty() and pulse_count < 5:
                pulse = pulses_queue.get()
                sample_pulses.append(pulse)
                pulse_count += 1
            
            if sample_pulses:
                self.logger.info(f"✅ Generated {pulse_count} sample pulses:")
                for i, pulse in enumerate(sample_pulses):
                    self.logger.info(f"  Pulse {i+1}: {pulse.photons:.2f} photons, polarization: {pulse.polarization}°")
            else:
                self.logger.warning("No pulses found in queue")
            
            # Test status
            status = simulator.get_status()
            self.logger.info("📊 Simulator status:")
            for key, value in status.items():
                if isinstance(value, dict):
                    self.logger.info(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        self.logger.info(f"    {sub_key}: {sub_value}")
                else:
                    self.logger.info(f"  {key}: {value}")
            
            # Test shutdown
            self.logger.info("🔸 Testing simulator shutdown...")
            if simulator.shutdown():
                self.logger.info("✅ Simulator shutdown successful")
            else:
                self.logger.error("❌ Simulator shutdown failed")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Simulator interface test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_simulator_controller(self) -> bool:
        """Test the controller interface with simulator."""
        self.logger.info("\n\n\n=============== Simulator Controller Test ===============")

        try:
            # Import simulator
            from src.alice.laser.laser_simulator import SimulatedLaserDriver
            
            # Create required objects for simulator
            pulses_queue = queue.Queue()
            laser_info = LaserInfo(
                name="Controller Test Simulator",
                max_power_mW=800.0,
                pulse_width_fwhm_ns=500.0,
                central_wavelength_nm=1550.0
            )
            
            # Create simulator driver and controller
            simulator_driver = SimulatedLaserDriver(pulses_queue, laser_info)
            controller = LaserController(simulator_driver)
            
            # Test initialization
            self.logger.info("🔸 Testing controller with simulator initialization...")
            if not controller.initialize():
                self.logger.error("Failed to initialize controller with simulator")
                return False
            
            self.logger.info(f"✅ Controller with simulator initialized (initialized: {controller.is_initialized()})")
            
            # Test single pulse
            self.logger.info("🔸 Testing single pulse via controller...")
            if controller.trigger_once():
                self.logger.info("✅ Single pulse via controller successful")
            else:
                self.logger.error("❌ Single pulse via controller failed")
                return False
            
            time.sleep(0.1)
            
            # Test frame
            self.logger.info("🔸 Testing frame via controller (5 pulses at 1000 Hz)...")
            if controller.send_frame(5, 1000.0):
                self.logger.info("✅ Frame via controller successful")
            else:
                self.logger.error("❌ Frame via controller failed")
                return False
            
            time.sleep(0.1)
            
            # Test continuous mode
            self.logger.info("🔸 Testing continuous mode via controller (1.5 seconds at 1500 Hz)...")
            if controller.start_continuous(1500.0):
                self.logger.info(f"Controller active: {controller.is_active()}")
                time.sleep(1.5)
                if controller.stop_continuous():
                    self.logger.info("✅ Continuous mode via controller completed")
                else:
                    self.logger.error("❌ Failed to stop continuous mode via controller")
                    return False
            else:
                self.logger.error("❌ Failed to start continuous mode via controller")
                return False
            
            # Check some generated pulses
            self.logger.info("🔸 Checking generated pulses...")
            pulse_count = 0
            total_photons = 0
            while not pulses_queue.empty() and pulse_count < 10:
                pulse = pulses_queue.get()
                total_photons += pulse.photons
                pulse_count += 1
            
            if pulse_count > 0:
                avg_photons = total_photons / pulse_count
                self.logger.info(f"✅ Processed {pulse_count} pulses, average photons: {avg_photons:.2f}")
            
            # Test status
            status = controller.get_status()
            self.logger.info("📊 Controller with simulator status:")
            for section, data in status.items():
                if isinstance(data, dict):
                    self.logger.info(f"  {section}:")
                    for key, value in data.items():
                        self.logger.info(f"    {key}: {value}")
                else:
                    self.logger.info(f"  {section}: {data}")
            
            # Shutdown
            controller.shutdown()
            self.logger.info("✅ Controller with simulator shutdown successful")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Simulator controller test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    
    def run_complete_demo(self, test_mode: str = "all") -> bool:
        """
        Run the complete demonstration.
        
        Args:
            test_mode: Test mode - "all", "hardware", "simulator"
        """
        self.logger.info("🚀 Starting Digital Laser Control Demo")
        self.logger.info(f"Test mode: {test_mode}")
        self.logger.info(f"Using device index: {self.device_index}, channel: {self.digital_channel}")
        
        # Define all available tests
        all_tests = [
            ("Digital Interface", self.test_digilent_digital_interface),
            ("Hardware Driver", self.test_hardware_driver),
            ("Controller Interface", self.test_controller_interface),
            ("Simulator Interface", self.test_simulator_interface),
            ("Simulator Controller", self.test_simulator_controller),
        ]
        
        # Select tests based on mode
        if test_mode == "hardware":
            tests = [
                ("Digital Interface", self.test_digilent_digital_interface),
                ("Hardware Driver", self.test_hardware_driver),
                ("Controller Interface", self.test_controller_interface),
            ]
        elif test_mode == "simulator":
            tests = [
                ("Simulator Interface", self.test_simulator_interface),
                ("Simulator Controller", self.test_simulator_controller),
            ]
        else:  # "all"
            tests = all_tests
        
        results = []
        for test_name, test_func in tests:
            self.logger.info(f"{'='*50}")
            try:
                result = test_func()
                results.append((test_name, result))
                
                if result:
                    self.logger.info(f"✅ {test_name} PASSED")
                else:
                    self.logger.error(f"❌ {test_name} FAILED")
                    
            except Exception as e:
                self.logger.error(f"💥 {test_name} CRASHED: {e}")
                results.append((test_name, False))
        
        # Summary
        self.logger.info(f"\n")
        self.logger.info(f"{'='*50}")
        self.logger.info("📋 DEMO SUMMARY")
        self.logger.info(f"{'='*50}")
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            self.logger.info(f"  {test_name}: {status}")
        
        self.logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            self.logger.info("🎉 ALL TESTS PASSED! Digital laser control is working correctly.")
            return True
        else:
            self.logger.error(f"💥 {total-passed} TEST(S) FAILED. Check hardware connections and drivers.")
            return False


def main():
    """Main demo function."""
    print("=" * 50, "Digital Laser Control Demo", "=" * 50)

    parser = argparse.ArgumentParser(description="Digital Laser Control Demo")
    parser.add_argument("--channel", "-c", type=int, default=8, 
                        help="Digital channel for triggering (default: 8)")
    parser.add_argument("--device", "-d", type=int, default=-1, 
                        help="Digilent device index (-1 for first available, default: -1)")
    parser.add_argument("--list-devices", "-l", action="store_true",
                        help="List available devices and exit")
    parser.add_argument("--test-mode", "-tm", choices=["all", "hardware", "simulator"], default="all",
                        help="Test mode: all (default), hardware only, or simulator only")
    parser.add_argument("--hardware-only", "-ho", action="store_true",
                        help="Run only hardware tests")
    parser.add_argument("--simulator-only", "-so", action="store_true",
                        help="Run only simulator tests")
    args = parser.parse_args()
    
    
    # Handle the list devices option
    if args.list_devices:
        try:
            devices = list_digital_devices()
            if not devices:
                print("No Digilent devices found")
            else:
                print(f"Found {len(devices)} Digilent device(s):")
                for i, device in enumerate(devices):
                    print(f"  [{i}] {device['name']} (SN: {device['serial']}) - "
                          f"{device['digital_channels']} digital channels")
            return 0
        except Exception as e:
            print(f"Error listing devices: {e}")
            return 1
    
    print(f"Using digital channel: {args.channel}")
    print(f"Using device index: {args.device}")
    print(f"Test mode: {args.test_mode}")
    
    # Create and run demo
    demo = DigitalLaserDemo(args.device, args.channel)
    
    # Handle legacy simulator mode
    if args.test_mode == "simulator":
        print("Running simulator tests only")
    elif args.test_mode == "hardware":
        print("Running hardware tests only")
    else:
        print("Running all tests")
    
    success = demo.run_complete_demo(test_mode=args.test_mode)
    
    if success:
        print("\n🎉 Demo completed successfully!")
        return 0
    else:
        print("\n💥 Demo failed!")
        return 1


if __name__ == "__main__":
    exit(main())

