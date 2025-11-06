#!/usr/bin/env python3
"""
Test script for the refactored AliceCPU with complete QKD protocol functionality.
"""
import logging
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.alice.aliceCPU import AliceCPU, AliceConfig, AliceMode

def test_basic_quantum_transmission():
    """Test basic quantum transmission without post-processing."""
    print("=" * 60)
    print("Testing Basic Quantum Transmission (No Post-Processing)")
    print("=" * 60)
    
    config = AliceConfig(
        num_pulses=5,
        pulse_period_seconds=0.5,
        use_hardware=False,
        mode=AliceMode.PREDETERMINED,
        predetermined_bits=[0, 1, 0, 1, 1],
        predetermined_bases=[0, 1, 0, 1, 0],
        enable_post_processing=False,
        use_mock_receiver=False
    )
    
    alice = AliceCPU(config)
    
    try:
        # Test just the quantum hardware initialization
        success = alice.initialize_system()
        if success:
            print("✓ Quantum hardware initialized successfully")
            
            # Test getting statistics and data
            stats = alice.get_results()
            
            print(f"✓ Initial stats: {stats.pulses_sent} pulses sent")
            print(f"✓ Initial data: {len(stats.pulse_ids)} recorded pulses")

            # Test component info
            component_info = alice.get_component_info()
            print(f"✓ Component info retrieved: {list(component_info.keys())}")
            
        else:
            print("✗ Failed to initialize quantum hardware")
            return False
            
    except Exception as e:
        print(f"✗ Error in basic test: {e}")
        return False
    finally:
        alice.shutdown_components()
    
    print("✓ Basic quantum transmission test completed\n")
    return True

def test_complete_qkd_with_mock():
    """Test complete QKD protocol with mock receiver."""
    print("=" * 60)
    print("Testing Complete QKD Protocol (With Mock Receiver)")
    print("=" * 60)
    
    config = AliceConfig(
        num_pulses=3,
        pulse_period_seconds=0.1,  # Faster for testing
        use_hardware=False,
        mode=AliceMode.PREDETERMINED,
        predetermined_bits=[0, 1, 1],
        predetermined_bases=[0, 1, 0],
        use_mock_receiver=True,
        server_qch_host="localhost",
        server_qch_port=12346,  # Different port to avoid conflicts
        enable_post_processing=False,  # Disable for now to test quantum part only
        alice_ip="localhost",
        alice_port=65434,
        bob_ip="localhost", 
        bob_port=65435
    )
    
    alice = AliceCPU(config)
    
    try:
        print("Starting complete QKD protocol...")
        
        # Add a small delay to ensure proper timing
        import time
        time.sleep(0.1)
        
        success = alice.run_complete_qkd_protocol()
        
        if success:
            stats = alice.get_results()
            
            print(f"✓ QKD protocol completed successfully!")
            print(f"✓ Transmitted {stats.pulses_sent} pulses")
            print(f"✓ Runtime: {stats.total_runtime_seconds:.2f}s")
            print(f"✓ Collected data for {len(stats.pulse_ids)} pulses")
            print(f"✓ Bits: {[int(bit) for bit in stats.bits]}")
            print(f"✓ Bases: {[int(basis) for basis in stats.bases]}")

            if stats.errors:
                print(f"⚠ Errors encountered: {len(stats.errors)}")
                for error in stats.errors:
                    print(f"  - {error}")
        else:
            print("✗ QKD protocol failed")
            return False
            
    except Exception as e:
        print(f"✗ Error in complete QKD test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ Complete QKD protocol test completed\n")
    return True

def test_configuration_validation():
    """Test configuration validation."""
    print("=" * 60)
    print("Testing Configuration Validation")
    print("=" * 60)
    
    # Test invalid predetermined sequences
    try:
        config = AliceConfig(
            num_pulses=3,
            mode=AliceMode.PREDETERMINED,
            predetermined_bits=[0, 1],  # Wrong length
            predetermined_bases=[0, 1, 0]
        )
        alice = AliceCPU(config)
        success = alice.initialize_system()
        if success:
            print("✗ Should have failed with invalid predetermined sequences")
            alice.shutdown_components()
            return False
        else:
            print("✓ Correctly rejected invalid predetermined sequences")
    except Exception as e:
        print(f"✓ Correctly caught invalid predetermined sequences: {type(e).__name__}")
    
    # Test valid configuration
    try:
        config = AliceConfig(
            num_pulses=3,
            mode=AliceMode.PREDETERMINED,
            predetermined_bits=[0, 1, 1],  # Correct length
            predetermined_bases=[0, 1, 0]
        )
        alice = AliceCPU(config)
        success = alice.initialize_system()
        if success:
            print("✓ Valid configuration accepted")
        else:
            print("✗ Valid configuration rejected")
            return False
        alice.shutdown_components()
    except Exception as e:
        print(f"✗ Error with valid configuration: {e}")
        return False
    
    print("✓ Configuration validation test completed\n")
    return True

def main():
    """Run all tests."""
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise for testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing Refactored AliceCPU Implementation")
    print("=" * 60)
    
    tests = [
        test_configuration_validation,
        test_basic_quantum_transmission,
        test_complete_qkd_with_mock
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All tests passed! AliceCPU refactoring successful.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)