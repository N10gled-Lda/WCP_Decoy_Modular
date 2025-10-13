#!/usr/bin/env python3
"""
Example demonstrating how to use the improved TimeTagger simulator in QKD context.
This shows how the simulator can be used as a drop-in replacement for hardware.
"""

import sys
import os
import logging
import time
from typing import Dict, List

# Add src to path for imports
import sys
import os

# Get the parent directory (project root)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.bob.timetagger.simple_timetagger_base_hardware_simulator import (
    SimpleTimeTaggerSimulator, 
    SimpleTimeTaggerHardware
)

def demonstrate_hardware_simulator_compatibility():
    """Show that simulator can be used as drop-in replacement for hardware."""
    
    print("=== Hardware/Simulator Compatibility Demo ===\n")
    
    # Configuration for QKD setup
    detector_channels = [1, 2, 3, 4]  # H, V, +, - polarization detectors
    measurement_duration = 1.0  # 1 second measurement
    
    # Try to create hardware timetagger first, fallback to simulator
    try:
        print("1. Attempting to initialize hardware TimeTagger...")
        timetagger = SimpleTimeTaggerHardware(detector_channels)
        if timetagger.initialize():
            print("✅ Hardware TimeTagger initialized successfully")
            device_type = "Hardware"
        else:
            raise Exception("Hardware initialization failed")
    except Exception as e:
        print(f"❌ Hardware not available: {e}")
        print("🔄 Falling back to simulator...")
        
        # Create simulator with QKD-appropriate parameters
        timetagger = SimpleTimeTaggerSimulator(
            detector_channels=detector_channels,
            dark_count_rate=10.0,     # Low dark counts (good detectors)
            signal_count_rate=200.0,  # Signal strength when photons arrive
            signal_probability=0.05   # 5% chance of signal in each time bin
        )
        
        if not timetagger.initialize():
            print("❌ Failed to initialize simulator")
            return False
        
        print("✅ Simulator initialized successfully")
        device_type = "Simulator"
    
    print(f"\n2. Running QKD measurement simulation with {device_type}...")
    
    # Configure measurement duration (same for both hardware and simulator)
    if not timetagger.set_measurement_duration(measurement_duration):
        print("❌ Failed to set measurement duration")
        return False
    
    # Simulate multiple QKD measurement rounds
    results = []
    for round_num in range(5):
        print(f"\n   Round {round_num + 1}:")
        
        # Measure detector counts (same call for hardware and simulator)
        counts = timetagger.measure_for_duration(measurement_duration)
        
        # Interpret results for QKD
        h_counts = counts.get(1, 0)  # Horizontal polarization
        v_counts = counts.get(2, 0)  # Vertical polarization  
        plus_counts = counts.get(3, 0)  # +45° polarization
        minus_counts = counts.get(4, 0)  # -45° polarization
        
        total_counts = sum(counts.values())
        
        print(f"     H: {h_counts:3d}, V: {v_counts:3d}, +: {plus_counts:3d}, -: {minus_counts:3d} | Total: {total_counts}")
        
        results.append({
            'round': round_num + 1,
            'H': h_counts,
            'V': v_counts,
            '+': plus_counts,
            '-': minus_counts,
            'total': total_counts
        })
    
    # Analyze time-binned data (advanced feature)
    print(f"\n3. Analyzing time-binned data with {device_type}...")
    timebin_result = timetagger.get_timebin_data(measurement_duration)
    
    if 'error' not in timebin_result:
        print(f"   ✅ Time-binned analysis successful:")
        print(f"      - Bin width: {timebin_result['binwidth_ps']/1e9:.1f}ms") 
        print(f"      - Number of bins: {timebin_result['n_bins']}")
        print(f"      - Channels analyzed: {timebin_result['channels']}")
        
        # Show timing structure for first channel
        channel_1_bins = timebin_result['timebin_data'][0]
        active_bins = [(i, count) for i, count in enumerate(channel_1_bins) if count > 0]
        if active_bins:
            print(f"      - Channel 1 active time bins: {len(active_bins)} of {len(channel_1_bins)}")
    
    # Cleanup
    print(f"\n4. Shutting down {device_type}...")
    timetagger.shutdown()
    
    # Summarize results
    print(f"\n=== QKD Measurement Summary ({device_type}) ===")
    avg_total = sum(r['total'] for r in results) / len(results)
    print(f"Average total counts per second: {avg_total:.1f}")
    print(f"Count rate stability: {min(r['total'] for r in results)}-{max(r['total'] for r in results)} counts/s")
    
    return True

def demonstrate_realistic_qkd_scenario():
    """Demonstrate realistic QKD scenario with alice sending and bob detecting."""
    
    print("\n=== Realistic QKD Scenario Demo ===\n")
    
    # Bob's detection setup
    bob_timetagger = SimpleTimeTaggerSimulator(
        detector_channels=[1, 2, 3, 4],
        dark_count_rate=5.0,       # Very low dark counts (good SPDs)  
        signal_count_rate=500.0,   # Strong signal when photon arrives
        signal_probability=0.02    # 2% detection efficiency * transmission
    )
    
    if not bob_timetagger.initialize():
        print("❌ Failed to initialize Bob's detector system")
        return False
    
    print("✅ Bob's detector system initialized")
    
    # Simulate BB84 protocol rounds
    print("\n1. Simulating BB84 Protocol Rounds...")
    
    bb84_rounds = [
        {"alice_basis": "rectilinear", "alice_bit": "0", "bob_basis": "rectilinear"},  # Should detect H
        {"alice_basis": "rectilinear", "alice_bit": "1", "bob_basis": "rectilinear"},  # Should detect V
        {"alice_basis": "diagonal", "alice_bit": "0", "bob_basis": "diagonal"},       # Should detect +
        {"alice_basis": "diagonal", "alice_bit": "1", "bob_basis": "diagonal"},       # Should detect -
        {"alice_basis": "rectilinear", "alice_bit": "0", "bob_basis": "diagonal"},    # Random (basis mismatch)
        {"alice_basis": "diagonal", "alice_bit": "1", "bob_basis": "rectilinear"},    # Random (basis mismatch)
    ]
    
    successful_detections = []
    
    for i, round_info in enumerate(bb84_rounds):
        print(f"\n   Round {i+1}: Alice sends {round_info['alice_bit']} in {round_info['alice_basis']} basis")
        print(f"            Bob measures in {round_info['bob_basis']} basis")
        
        # Set measurement time based on photon transmission timing
        measurement_time = 0.1  # 100ms measurement window
        bob_timetagger.set_measurement_duration(measurement_time)
        
        # Simulate higher signal probability for matching bases
        if round_info['alice_basis'] == round_info['bob_basis']:
            # Matching bases - higher detection probability
            bob_timetagger.signal_probability = 0.08  # 8% detection
        else:
            # Mismatched bases - random detection  
            bob_timetagger.signal_probability = 0.04  # 4% random detection
        
        # Measure
        counts = bob_timetagger.measure_for_duration(measurement_time)
        
        # Determine Bob's measurement result
        h_count = counts.get(1, 0)
        v_count = counts.get(2, 0) 
        plus_count = counts.get(3, 0)
        minus_count = counts.get(4, 0)
        
        # Find dominant detection
        detections = {'H': h_count, 'V': v_count, '+': plus_count, '-': minus_count}
        max_detector = max(detections, key=detections.get)
        max_count = detections[max_detector]
        
        if max_count > 0:
            print(f"            Bob detects: {max_detector} ({max_count} counts)")
            
            # Check if bases match for successful key bit
            if round_info['alice_basis'] == round_info['bob_basis']:
                expected_detectors = {
                    ("rectilinear", "0"): "H",
                    ("rectilinear", "1"): "V", 
                    ("diagonal", "0"): "+",
                    ("diagonal", "1"): "-"
                }
                expected = expected_detectors[(round_info['alice_basis'], round_info['alice_bit'])]
                
                if max_detector == expected:
                    print(f"            ✅ Correct detection! Key bit: {round_info['alice_bit']}")
                    successful_detections.append({
                        'round': i+1,
                        'bit': round_info['alice_bit'],
                        'basis': round_info['alice_basis'],
                        'detection': max_detector,
                        'counts': max_count
                    })
                else:
                    print(f"            ❌ Incorrect detection (expected {expected})")
            else:
                print(f"            ℹ️  Basis mismatch - bit discarded")
        else:
            print(f"            ❌ No detection (all counts: {detections})")
    
    # Summary
    print(f"\n2. BB84 Protocol Results:")
    print(f"   Total rounds: {len(bb84_rounds)}")
    print(f"   Successful detections: {len(successful_detections)}")
    print(f"   Key extraction rate: {len(successful_detections)/len(bb84_rounds)*100:.1f}%")
    
    if successful_detections:
        print(f"   Generated key bits: {''.join(d['bit'] for d in successful_detections)}")
    
    bob_timetagger.shutdown()
    print("\n✅ QKD simulation completed successfully")
    
    return True

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise for demo
    
    print("TimeTagger Simulator QKD Integration Demo")
    print("=" * 50)
    
    # Run demonstrations
    success1 = demonstrate_hardware_simulator_compatibility()
    success2 = demonstrate_realistic_qkd_scenario()
    
    if success1 and success2:
        print("\n🎉 All QKD integration demos completed successfully!")
        print("\nThe improved simulator enables:")
        print("• Drop-in replacement for hardware during development")
        print("• Realistic QKD protocol testing and validation")
        print("• Time-binned analysis for advanced algorithms")
        print("• CI/CD integration without hardware dependencies")
    else:
        print("\n❌ Some demos failed")
        sys.exit(1)