"""
Example: Complete QKD System Integration
Demonstrates Alice and Bob working together with enhanced simulators.
"""

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
        num_pulses=100,
        pulse_period_seconds=0.1,  # 10 Hz for demo
        use_hardware=False,  # Use simulators
        mode=AliceMode.STREAMING,
        qrng_seed=42
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
        measurement_duration_s=15.0,  # 15 seconds for demo
        channel_config=channel_config,
        optical_table_config=optical_config,
        detector_configs=detector_configs,
        basis_selection_seed=123
    )
    
    # Initialize Alice and Bob
    print("Initializing Alice and Bob...")
    alice = AliceCPU(alice_config)
    bob = BobCPU(bob_config)
    
    # Start Bob's measurement first
    print("Starting Bob's measurement system...")
    if not bob.start_measurement():
        print("Failed to start Bob's measurement")
        return
    
    # Start Alice's transmission
    print("Starting Alice's transmission...")
    if not alice.start_transmission():
        print("Failed to start Alice's transmission")
        bob.stop_measurement()
        return
    
    # Simulate QKD protocol
    print("Running QKD protocol...")
    pulse_count = 0
    
    try:
        while alice.is_running() and bob.is_running() and pulse_count < alice_config.num_pulses:
            # Get pulses from Alice
            alice_pulses = alice.get_output_pulses(max_pulses=10)
            
            for pulse in alice_pulses:
                # Send pulse to Bob through the system
                bob.receive_pulse(pulse)
                pulse_count += 1
            
            # Show progress
            if pulse_count % 20 == 0:
                alice_progress = alice.get_progress()
                bob_efficiency = bob.get_detection_efficiency()
                print(f"Progress: {alice_progress:.1f}%, Detection efficiency: {bob_efficiency:.3f}")
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\\nDemo interrupted by user")
    
    # Stop systems
    print("Stopping Alice and Bob...")
    alice.stop_transmission()
    bob.stop_measurement()
    
    # Show results
    print("\\n" + "=" * 60)
    print("QKD Demo Results")
    print("=" * 60)
    
    # Alice statistics
    alice_stats = alice.get_statistics()
    alice_data = alice.get_transmission_data()
    
    print(f"\\nAlice Results:")
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
    
    print(f"\\nBob Results:")
    print(f"  Total measurements: {bob_stats.total_measurements}")
    print(f"  Successful detections: {bob_stats.successful_detections}")
    print(f"  Detection efficiency: {bob.get_detection_efficiency():.3f}")
    print(f"  Basis selections: Z={bob_stats.basis_selections['Z']}, X={bob_stats.basis_selections['X']}")
    print(f"  Detector counts: {bob_stats.detector_counts}")
    print(f"  Errors: {len(bob_stats.errors)}")
    
    # Component information
    print(f"\\nComponent Information:")
    bob_info = bob.get_component_info()
    channel_info = bob_info['quantum_channel']
    print(f"  Channel transmission: {channel_info['performance']['average_transmission_efficiency']:.4f}")
    print(f"  Channel weather: {channel_info['config']['weather_condition']}")
    
    # Key matching simulation (simplified)
    if len(alice_data.bases) > 0 and len(bob_data.measurement_bases) > 0:
        print(f"\\nBasis Matching Analysis:")
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


def run_full_simulation_demo():
    """Run a demonstration with the full simulation Alice CPU."""
    
    print("\\n" + "=" * 60)
    print("Full Simulation Demo (with VOA)")
    print("=" * 60)
    
    # Configure full simulation
    sim_config = SimulationConfig(
        pulses_total=500,
        use_hardware=False,  # Pure simulation
        use_hardware_qrng=False,
        use_hardware_laser=False,
        use_hardware_voa=False,
        use_hardware_polarization=False,
        pulse_period_seconds=0.01,  # 100 Hz
        random_seed=42,
        
        # Enhanced configurations
        laser=LaserInfo(
            central_wavelength_nm=1550.0,
            max_power_mW=1.0,
            polarization_extinction_ratio_dB=25.0
        ),
        decoy_scheme=DecoyInfo(
            intensities={"signal": 0.5, "weak": 0.1, "vacuum": 0.0},
            probabilities={"signal": 0.7, "weak": 0.2, "vacuum": 0.1}
        )
    )
    
    # Initialize full Alice
    print("Initializing full simulation Alice...")
    alice_full = AliceCPUGeneral(sim_config)
    
    print("Starting full simulation...")
    alice_full.start_transmission()
    
    # Let it run for a few seconds
    time.sleep(5.0)
    
    # Stop and show results
    alice_full.stop_transmission()
    
    results = alice_full.get_simulation_results()
    print(f"\\nFull Simulation Results:")
    print(f"  Pulses sent: {results.alice_state.pulses_sent}")
    print(f"  Runtime: {results.alice_state.total_runtime_s:.2f}s")
    print(f"  Component types: {results.statistics['component_types']}")
    print(f"  Queue sizes: {results.statistics['queue_sizes']}")


if __name__ == "__main__":
    # Run the simple demo
    run_simple_qkd_demo()
    
    # Run the full simulation demo
    run_full_simulation_demo()
    
    print("\\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
