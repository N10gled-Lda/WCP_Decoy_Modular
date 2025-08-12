#!/usr/bin/env python3
"""
Test script for simplified trigger modes in DigilentDigitalInterface.

This script demonstrates the three core trigger modes:
- SINGLE: One trigger pulse
- TRAIN: N trigger pulses at specified frequency  
- CONTINUOUS: Continuous trigger pulses

The BURST mode has been removed as it was redundant with TRAIN mode.
"""

import os
import sys
import time
from pathlib import Path

# Add the src directory to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.alice.laser.hardware_laser.digilent_digital_interface import (
    DigilentDigitalInterface, 
    DigitalTriggerMode,
    list_digital_devices
)


def test_trigger_modes():
    """Test all three simplified trigger modes."""
    print("=" * 60)
    print("Testing Simplified Digital Trigger Modes")
    print("=" * 60)
    
    # List available devices
    devices = list_digital_devices()
    print(f"📱 Available devices: {len(devices)}")
    for dev in devices:
        print(f"  • {dev['index']}: {dev['name']} (SN: {dev['serial']})")
    
    if not devices:
        print("❌ No Digilent devices found - running simulation mode")
        # You could add simulation logic here
        return False
    
    print(f"\n🔧 Testing with device: ...")
    print(f"   • Device index: -1")
    print(f"   • Digital channel: 8")

    # Test the three modes
    try:
        interface = DigilentDigitalInterface(device_index=-1, digital_channel=8)
        sucess = interface.connect()
        print("\n\n🔌 Initializing DigilentDigitalInterface...")
        # with interface:
        if not sucess:
            print("❌ Failed to connect to device")
            return False
            
        print("✅ Connected successfully")
        
        # Configure pulse parameters
        interface.set_pulse_parameters(
            width=0.5e-6,      # 1 microsecond
            frequency=1000000.0, # 1 kHz
            idle_state=False  # Idle low
        )
        # Print new configuration
        print(f"📊 Pulse configuration:"
                f"\n   • Pulse width: {interface.pulse_width:.7f} μs"
                f"\n   • Frequency: {interface.frequency:.1f} Hz"
                f"\n   • Idle state: {'HIGH' if interface.idle_state else 'LOW'}")
        
        # # Test 1: SINGLE mode
        # print(f"\n🔸 Testing {DigitalTriggerMode.SINGLE.value} mode...")
        # success = interface.trigger_laser(mode="single")
        # print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
        # # time.sleep(0.05)
        # # success = interface.trigger_laser(mode="single")
        
        # Test 2: TRAIN mode
        print(f"\n🔸 Testing {DigitalTriggerMode.TRAIN.value} mode (5 pulses at 2 kHz)...")
        freq= 1000000.0
        nb_pulses = 10  # Number of pulses in the train
        success = interface.trigger_laser(mode="train", count=nb_pulses, frequency=freq)
        # if success:
        #     time.sleep(5/freq + 0.1)  # Wait for completion
        print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
        # time.sleep(1)
        
        # Test 3: CONTINUOUS mode (run for 1 second)
        print(f"\n🔸 Testing {DigitalTriggerMode.CONTINUOUS.value} mode (1 second at 1000 Hz)...")
        success = interface.trigger_laser(mode="continuous", frequency=1000)
        if success:
            print("   ⏳ Running continuous mode for 1 second...")
            time.sleep(1.0)
            interface.stop()
            print("   ⏹️ Stopped continuous mode")
        print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
        
        # Show final status
        status = interface.get_status()
        print(f"\n📊 Final status:")
        print(f"   • Pulse count: {status['pulse_count']}")
        print(f"   • Error count: {status['error_count']}")
        print(f"   • Channel: {status['channel']}")
        print(f"   • Pulse width: {status['pulse_width_us']:.1f} μs")
        print(f"   • Frequency: {status['frequency_hz']:.1f} Hz")
        
        return True
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False


def test_invalid_mode():
    """Test that invalid modes are properly rejected."""
    print(f"\n🔸 Testing invalid mode rejection...")
    
    try:
        interface = DigilentDigitalInterface(device_index=-1, digital_channel=8)
        if interface.connected:
            # Try to use the old BURST mode (should fail)
            success = interface.trigger_laser(mode="burst", count=5)
            if not success:
                print("   ✅ Invalid mode 'burst' properly rejected")
            else:
                print("   ❌ Invalid mode 'burst' was accepted (unexpected)")
            
            # Try another invalid mode
            success = interface.trigger_laser(mode="invalid_mode")
            if not success:
                print("   ✅ Invalid mode 'invalid_mode' properly rejected")
            else:
                print("   ❌ Invalid mode 'invalid_mode' was accepted (unexpected)")
        else:
            print("   ⚠️ Could not test (device not connected)")
                
    except Exception as e:
        print(f"   ❌ Error testing invalid modes: {e}")


def main():
    """Main test function."""
    print("🚀 Starting digital trigger mode tests...\n")
    
    # Show available modes
    print("📋 Available trigger modes:")
    for mode in DigitalTriggerMode:
        print(f"   • {mode.value}")
    
    # Run tests
    success = test_trigger_modes()
    test_invalid_mode()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests completed successfully!")
    else:
        print("⚠️ Some tests failed or could not run")
    print("="*60)


if __name__ == "__main__":
    main()
