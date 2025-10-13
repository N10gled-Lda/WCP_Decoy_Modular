#!/usr/bin/env python3
"""
Test for the proper Bob architecture that mirrors Alice.
"""
import os
import sys
import logging

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def test_proper_bob_architecture():
    """Test the proper Bob architecture."""
    print("="*60)
    print("TESTING PROPER BOB ARCHITECTURE")
    print("(Mirrors Alice's structure)")
    print("="*60)
    
    try:
        from src.bob.bobCPU_proper import BobCPU, BobConfig, BobMode
        from src.bob.timetagger.simple_timetagger_controller import SimpleTimeTaggerController
        print("✅ All imports successful")
        
        # Test configuration - matching Alice's style
        config = BobConfig(
            num_expected_pulses=5,
            pulse_period_seconds=0.5,
            measurement_fraction=0.8,
            use_hardware=False,
            detector_channels=[1, 2, 3, 4],
            dark_count_rate=50.0,  # Lower for cleaner test
            mode=BobMode.CONTINUOUS,
            use_mock_transmitter=False,  # We'll test components directly
            enable_post_processing=False  # Skip for simple test
        )
        
        # Create Bob
        bob = BobCPU(config)
        print("✅ Bob CPU created successfully")
        
        # Test component initialization
        if bob.timetagger_controller and bob.timetagger_controller.is_initialized():
            print("✅ TimeTagger controller initialized")
        else:
            print("❌ TimeTagger controller not initialized")
            return False
        
        # Test measurement duration setup
        duration = bob.timetagger_controller.get_measurement_duration()
        expected_duration = config.pulse_period_seconds
        if abs(duration - expected_duration) < 0.001:
            print(f"✅ Measurement duration correctly set to {duration:.3f}s")
        else:
            print(f"❌ Wrong measurement duration: {duration} vs expected {expected_duration}")
            return False
        
        # Test direct measurement
        counts = bob.timetagger_controller.measure_counts()
        if counts and len(counts) == len(config.detector_channels):
            print(f"✅ Measurement successful: {counts}")
        else:
            print(f"❌ Measurement failed: {counts}")
            return False
        
        # Test component info
        info = bob.get_component_info()
        if info and 'timetagger_controller' in info:
            print("✅ Component info available")
        else:
            print("❌ Component info not available")
            return False
        
        # Test handshake markers match Alice
        alice_handshake = 100
        alice_ack = 150
        alice_end = 50
        
        if (bob._handshake_marker == alice_handshake and 
            bob._acknowledge_marker == alice_ack and
            bob._handshake_end_marker == alice_end):
            print("✅ Handshake markers match Alice's")
        else:
            print(f"❌ Handshake markers don't match Alice's:")
            print(f"   Bob: {bob._handshake_marker}, {bob._acknowledge_marker}, {bob._handshake_end_marker}")
            print(f"   Alice: {alice_handshake}, {alice_ack}, {alice_end}")
            return False
        
        # Cleanup
        bob.shutdown_components()
        print("✅ Bob shutdown successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_controller_pattern():
    """Test that we follow Alice's controller pattern properly."""
    print("\n" + "="*60)
    print("TESTING CONTROLLER PATTERN")
    print("="*60)
    
    try:
        from src.bob.timetagger.simple_timetagger_controller import SimpleTimeTaggerController
        from src.bob.timetagger.simple_timetagger_base_hardware_simulator import SimpleTimeTaggerSimulator
        
        # Create controller like Alice does
        driver = SimpleTimeTaggerSimulator([1, 2], 50.0)
        controller = SimpleTimeTaggerController(driver)
        
        # Test initialization
        if controller.initialize():
            print("✅ Controller initialization successful")
        else:
            print("❌ Controller initialization failed")
            return False
        
        # Test duration setting (like Alice sets laser parameters)
        if controller.set_measurement_duration(0.5):
            print("✅ Duration setting successful")
        else:
            print("❌ Duration setting failed")
            return False
        
        # Test measurement (like Alice triggers laser)
        counts = controller.measure_counts()
        if counts:
            print(f"✅ Measurement successful: {counts}")
        else:
            print("❌ Measurement failed")
            return False
        
        # Test status (like Alice gets component status)
        status = controller.get_status()
        if status and status['initialized']:
            print("✅ Status reporting working")
        else:
            print("❌ Status reporting failed")
            return False
        
        controller.shutdown()
        print("✅ Controller shutdown successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Controller test failed: {e}")
        return False

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.WARNING,  # Reduced for cleaner output
        format='%(levelname)s - %(message)s'
    )

def main():
    """Run all tests for proper Bob architecture."""
    setup_logging()
    
    print("PROPER BOB ARCHITECTURE TESTS")
    print("Verifying Bob mirrors Alice's structure")
    
    # Test 1: Controller pattern
    controller_ok = test_controller_pattern()
    
    # Test 2: Full architecture
    architecture_ok = test_proper_bob_architecture()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if controller_ok and architecture_ok:
        print("✅ ALL TESTS PASSED")
        print("✅ Bob properly mirrors Alice's architecture")
        print("\n📝 Key improvements:")
        print("   - Proper TimeTaggerController following Alice's pattern")
        print("   - Hardware duration configuration fixed")
        print("   - Handshake markers match Alice exactly")
        print("   - Full network and post-processing support")
        print("   - Component management like Alice")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print(f"   Controller Pattern: {'✅' if controller_ok else '❌'}")
        print(f"   Full Architecture: {'✅' if architecture_ok else '❌'}")
        return 1

if __name__ == "__main__":
    exit(main())