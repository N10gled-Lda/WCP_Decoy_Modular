"""
Example: Complete QKD System Integration
Demonstrates Alice and Bob working together with enhanced simulators.
"""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging
import time
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Alice components
from src.alice import AliceCPU, AliceConfig, AliceMode
from src.alice.alice_cpu_general import AliceCPUGeneral, SimulationConfig

# Bob components  
from src.bob import BobCPU, BobConfig, BobMode
from src.quantum_channel import ChannelConfig
from src.bob.ot_components import OpticalTableConfig, DetectorConfig, DetectorType

# Data structures
from src.utils.data_structures import Basis, Bit, DecoyInfo, LaserInfo


def run_simple_qkd_demo():
    """Run a simple QKD demonstration with Alice and Bob."""
    
    print("=" * 60)
    print("QKD System Integration Demo")
    print("=" * 60)
    
    # Configure Alice (simple hardware test version)
    alice_config = AliceConfig(
        num_pulses=10,
        pulse_period_seconds=1,  # 10 Hz for demo
        use_hardware=True,  # Use simulators
        com_port="COM4",
        laser_channel=8,
        mode=AliceMode.STREAMING,
        qrng_seed=40
    )
    
    # Configure Bob
    channel_config = ChannelConfig(
        distance_km=10.0,
        base_attenuation_db_km=0.2,
        enable_atmospheric_turbulence=True,
        enable_polarization_drift=True,
        enable_background_noise=True
    )
    
    optical_config = OpticalTableConfig(
        beam_splitter_ratio=0.5,
        polarizer_extinction_ratio_db=30.0,
        coupling_efficiency=0.9
    )
    
    detector_configs = {
        "H": DetectorConfig(detector_type=DetectorType.SPAD, quantum_efficiency=0.8, dark_count_rate_hz=50),
        "V": DetectorConfig(detector_type=DetectorType.SPAD, quantum_efficiency=0.8, dark_count_rate_hz=50),
        "D": DetectorConfig(detector_type=DetectorType.SPAD, quantum_efficiency=0.8, dark_count_rate_hz=50),
        "A": DetectorConfig(detector_type=DetectorType.SPAD, quantum_efficiency=0.8, dark_count_rate_hz=50)
    }
    
    bob_config = BobConfig(
        mode=BobMode.PASSIVE,
        measurement_duration_s=30.0,  # 30 seconds for demo
        channel_config=channel_config,
        optical_table_config=optical_config,
        detector_configs=detector_configs,
        basis_selection_seed=123
    )
    
    # Initialize Alice and Bob
    print("\nInitializing Alice and Bob...\n")
    alice = AliceCPU(alice_config)
    bob = BobCPU(bob_config)
    
    # Start Bob's measurement first
    print("\nStarting Bob's measurement system...\n")
    if not bob.start_measurement():
        print("Failed to start Bob's measurement")
        return
    
    # Start Alice's transmission
    print("\nStarting Alice's transmission...\n")
    if not alice.start_transmission():
        print("\nFailed to start Alice's transmission\n")
        bob.stop_measurement() if bob.is_running() else None
        return
    
    # Simulate QKD protocol
    print("\nRunning QKD protocol...\n")
    pulse_count = 0
    
    try:
        while alice.is_running() and pulse_count < alice_config.num_pulses:
            # # Get pulses from Alice
            # transmission_data = alice.get_transmission_data()
            # alice_pulses = transmission_data.pulses
            # print(f"DEBUG: Transmission data pulses: {alice_pulses}")
            # for pulse in alice_pulses:
            #     # Send pulse to Bob through the system
            #     bob.receive_pulse(pulse)
            #     pulse_count += 1
            # print(f"DEBUG: Total pulses sent to Bob: {pulse_count} / {alice_config.num_pulses} (alice runing: {alice.is_running()})")
            # # Show progress
            # # if pulse_count % 20 == 0:
            # #     alice_progress = alice.get_progress()
            # #     bob_efficiency = bob.get_detection_efficiency()
            # #     print(f"Progress: {alice_progress:.1f}%, Detection efficiency: {bob_efficiency:.3f}")
            
            # time.sleep(10)

            # Show progress:
            print(f"---------------> Alice Running: {alice.is_running()}, Bob Running: {bob.is_running()}")
            time.sleep(0.1)
            
    
    except KeyboardInterrupt:
        print("\n Demo interrupted by user")
    
    # Stop systems
    print("\n Alice Running:", alice.is_running())
    # Get transmission data
    alice_data = alice.get_transmission_data()
    print(f"DEBUG: Final Transmission data pulses: {alice_data.pulses}")

    pulse_count = len(alice_data.pulses)
    print(f"Total pulses sent to Bob: {pulse_count} / {alice_config.num_pulses}")
    print(f"Basis: {alice_data.bases}")
    print(f"Bits: {alice_data.bits}")
    print(f"Times: {alice_data.pulse_times}")
    print(f"Angles: {alice_data.polarization_angles}")


    print("Stopping Alice and Bob...")
    alice.stop_transmission()
    bob.stop_measurement()
    
    # Show results
    print("\n" + "=" * 60)
    print("QKD Demo Results")
    print("=" * 60)
    
    # Alice statistics
    alice_stats = alice.get_statistics()
    alice_data = alice.get_transmission_data()
    
    print(f"\n Alice Results:")
    print(f"  Pulses sent: {alice_stats.pulses_sent}")
    print(f"  Runtime: {alice_stats.total_runtime_seconds:.2f}s")
    print(f"  Average rate: {alice_stats.average_pulse_rate_hz:.1f} Hz")
    print(f"  Errors: {len(alice_stats.errors)}")
    
    if alice_stats.rotation_times:
        avg_rotation = sum(alice_stats.rotation_times) / len(alice_stats.rotation_times) * 1000
        print(f"  Avg rotation time: {avg_rotation:.1f} ms")
    
    # Bob statistics
    bob_stats = bob.get_statistics()
    bob_data = bob.get_measurement_data()
    
    print(f"\n Bob Results:")
    print(f"  Total measurements: {bob_stats.total_measurements}")
    print(f"  Successful detections: {bob_stats.successful_detections}")
    print(f"  Detection efficiency: {bob.get_detection_efficiency():.3f}")
    print(f"  Basis selections: Z={bob_stats.basis_selections['Z']}, X={bob_stats.basis_selections['X']}")
    print(f"  Detector counts: {bob_stats.detector_counts}")
    print(f"  Errors: {len(bob_stats.errors)}")
    
    # Component information
    print(f"\n Component Information:")
    bob_info = bob.get_component_info()
    channel_info = bob_info['quantum_channel']
    print(f"  Channel transmission: {channel_info['performance']['average_transmission_efficiency']:.4f}")
    print(f"  Channel weather: {channel_info['config']['weather_condition']}")
    
    # Key matching simulation (simplified)
    if len(alice_data.bases) > 0 and len(bob_data.measurement_bases) > 0:
        print(f"\n Basis Matching Analysis:")
        matching_bases = 0
        matching_bits = 0
        total_comparisons = min(len(alice_data.bases), len(bob_data.measurement_bases))
        
        for i in range(total_comparisons):
            if alice_data.bases[i] == bob_data.measurement_bases[i]:
                matching_bases += 1
                alice_bit = alice_data.bits[i]
                bob_bit = bob_data.measured_bits[i]
                if bob_bit is not None and alice_bit == bob_bit.value:
                    matching_bits += 1
        
        if total_comparisons > 0:
            basis_match_rate = matching_bases / total_comparisons
            if matching_bases > 0:
                bit_error_rate = 1 - (matching_bits / matching_bases)
            else:
                bit_error_rate = 1.0
            
            print(f"  Basis match rate: {basis_match_rate:.3f}")
            print(f"  Bit error rate: {bit_error_rate:.3f}")
            print(f"  Potential key bits: {matching_bits}")


if __name__ == "__main__":
    # Run the simple demo
    run_simple_qkd_demo()
        
    print("\n" + "=" * 60)
    print("Demo completed")
    print("=" * 60)
