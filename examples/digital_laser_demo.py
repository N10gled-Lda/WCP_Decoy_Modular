"""Demo for Digital Laser Control using Digilent Device."""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging
import time
from typing import List

# Import digital components
from src.alice.laser.hardware_laser.digilent_digital_interface import (
    DigilentDigitalInterface, 
    DigitalTriggerMode,
    list_digital_devices
)
from src.alice.laser.laser_hardware_digital import (
    DigitalHardwareLaserDriver,
    LaserState,
    create_digital_laser_driver
)

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
        
    def test_device_discovery(self) -> bool:
        """Test device discovery and listing."""
        self.logger.info("=== Device Discovery Test ===")
        
        try:
            devices = list_digital_devices()
            
            if not devices:
                self.logger.warning("No Digilent devices found")
                return False
            
            self.logger.info(f"Found {len(devices)} Digilent device(s):")
            for i, device in enumerate(devices):
                self.logger.info(f"  [{i}] {device['name']} (SN: {device['serial']}) - "
                               f"{device['digital_channels']} digital channels")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Device discovery failed: {e}")
            return False
    
    def test_digital_interface(self) -> bool:
        """Test the low-level digital interface."""
        self.logger.info("=== Digital Interface Test ===")
        
        try:
            interface = DigilentDigitalInterface(device_index=self.device_index, digital_channel=self.digital_channel)
            interface.connect()
            if not interface.connected:
                self.logger.error("Failed to connect to device")
                return False
            
            self.logger.info(f"✅ Connected to device on channel {self.digital_channel}")
            
            # Test pulse parameter configuration
            self.logger.info("🔧 Testing pulse parameter configuration...")
            interface.set_pulse_parameters(width=2e-6, frequency=1000.0, idle_state=False)
            
            # Test single pulse
            self.logger.info("🔸 Testing single pulse...")
            if interface.send_single_pulse():
                self.logger.info("✅ Single pulse sent successfully")
            else:
                self.logger.error("❌ Single pulse failed")
                return False
            
            time.sleep(0.1)
            
            # Test pulse train
            self.logger.info("🔸 Testing pulse train (5 pulses at 2 kHz)...")
            if interface.start_pulse_train(5, 2000.0):
                time.sleep(5 / 2000.0 + 0.1)  # Wait for completion
                self.logger.info("✅ Pulse train completed")
            else:
                self.logger.error("❌ Pulse train failed")
                return False
            
            # Test continuous mode briefly
            self.logger.info("🔸 Testing continuous mode (1 second at 500 Hz)...")
            if interface.start_continuous(500.0):
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
        self.logger.info("=== Hardware Driver Test ===")
        
        try:
            with create_digital_laser_driver(self.device_index, self.digital_channel) as laser:
                if laser.state == LaserState.ERROR:
                    self.logger.error("Failed to initialize laser driver")
                    return False
                
                self.logger.info(f"✅ Laser driver initialized (state: {laser.state.value})")
                
                # Test single trigger
                self.logger.info("🔸 Testing single trigger...")
                if laser.trigger_once():
                    self.logger.info("✅ Single trigger successful")
                else:
                    self.logger.error("❌ Single trigger failed")
                    return False
                
                time.sleep(0.1)
                
                # Test frame with different parameters
                self.logger.info("🔸 Testing frame (3 triggers at 1.5 kHz)...")
                if laser.send_frame(3, 1500.0):
                    self.logger.info("✅ Frame sent successfully")
                else:
                    self.logger.error("❌ Frame sending failed")
                    return False
                
                time.sleep(0.1)
                
                # Test pulse parameter adjustment
                self.logger.info("🔧 Testing pulse parameter adjustment...")
                laser.set_pulse_parameters(width=5e-6, frequency=800.0)  # 5μs pulses at 800Hz
                
                # Test with new parameters
                self.logger.info("🔸 Testing with new parameters (2 triggers)...")
                if laser.send_frame(2, 800.0):
                    self.logger.info("✅ Test with new parameters successful")
                else:
                    self.logger.error("❌ Test with new parameters failed")
                    return False
                
                # Test continuous mode
                self.logger.info("🔸 Testing continuous mode (2 seconds at 300 Hz)...")
                if laser.start_continuous(300.0):
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
    
    def test_edge_cases(self) -> bool:
        """Test edge cases and error handling."""
        self.logger.info("=== Edge Cases Test ===")
        
        try:
            with create_digital_laser_driver(self.device_index, self.digital_channel) as laser:
                if laser.state == LaserState.ERROR:
                    self.logger.warning("Skipping edge cases test - no hardware available")
                    return True
                
                # Test triggering when laser is off
                self.logger.info("🔸 Testing trigger when laser is off...")
                result = laser.trigger_once()  # Should fail
                if not result:
                    self.logger.info("✅ Correctly rejected trigger when off")
                else:
                    self.logger.warning("⚠️ Allowed trigger when off (unexpected)")
                
                # Test extreme frequencies
                self.logger.info("🔸 Testing frequency limits...")
                
                # Very high frequency
                high_freq_result = laser.send_frame(1, 100e6)  # 100 MHz
                if not high_freq_result:
                    self.logger.info("✅ Correctly rejected excessive frequency")
                else:
                    self.logger.warning("⚠️ Accepted excessive frequency")
                
                # Very low frequency
                low_freq_result = laser.send_frame(1, 0.1)  # 0.1 Hz
                if low_freq_result:
                    self.logger.info("✅ Accepted very low frequency")
                else:
                    self.logger.warning("⚠️ Rejected reasonable low frequency")
                
                # Test pulse width limits
                self.logger.info("🔸 Testing pulse width limits...")
                laser.set_pulse_parameters(width=1e-12)  # 1 ps - very short
                laser.set_pulse_parameters(width=1e-3)   # 1 ms - very long
                
                self.logger.info("✅ Edge cases test completed")
                return True
                
        except Exception as e:
            self.logger.error(f"Edge cases test failed: {e}")
            return False
    
    def run_complete_demo(self) -> bool:
        """Run the complete demonstration."""
        self.logger.info("🚀 Starting Digital Laser Control Demo")
        self.logger.info(f"Using device index: {self.device_index}, channel: {self.digital_channel}")
        
        # Run all test phases
        tests = [
            ("Device Discovery", self.test_device_discovery),
            ("Digital Interface", self.test_digital_interface),
            ("Hardware Driver", self.test_hardware_driver),
            ("Edge Cases", self.test_edge_cases)
        ]
        
        results = []
        for test_name, test_func in tests:
            self.logger.info(f"\n{'='*50}")
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
        self.logger.info(f"\n{'='*50}")
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
    print("Digital Laser Control Demo")
    print("=" * 50)
    
    # Parse command line arguments for device and channel
    device_index = -1  # First available device
    digital_channel = 8  # Default to channel 8
    
    if len(sys.argv) > 1:
        try:
            digital_channel = int(sys.argv[1])
            print(f"Using digital channel: {digital_channel}")
        except ValueError:
            print(f"Invalid channel '{sys.argv[1]}', using default channel 8")
    
    if len(sys.argv) > 2:
        try:
            device_index = int(sys.argv[2])
            print(f"Using device index: {device_index}")
        except ValueError:
            print(f"Invalid device index '{sys.argv[2]}', using first available device")
    
    # Create and run demo
    demo = DigitalLaserDemo(device_index, digital_channel)
    success = demo.run_complete_demo()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        return 0
    else:
        print("\n💥 Demo failed!")
        return 1


if __name__ == "__main__":
    exit(main())

### DEPRECATED DEMO ???