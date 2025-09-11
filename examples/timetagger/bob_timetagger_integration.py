"""
Comprehensive BobCPU Time Tagger Integration Example.

This example demonstrates how to use BobCPU with the integrated time tagger system
for realistic QKD detection and timing analysis.
"""

import logging
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

# Import Bob components
from src.bob import (
    BobCPU, BobConfig, BobMode,
    TimeTaggerControllerConfig, TimeTaggerConfig, 
    ChannelConfig as TTChannelConfig, SimulatorConfig
)

# Import Alice components for simulation
from src.alice import AliceCPUGeneral, SimulationConfig as AliceSimConfig
from src.alice.qrng import QRNGConfig
from src.alice.laser import LaserConfig
from src.alice.voa import VOAConfig  
from src.alice.polarization import PolarizationConfig

# Import data structures
from src.utils.data_structures import Basis, Bit, Pulse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_bob_with_timetagger(use_hardware: bool = False) -> BobCPU:
    """
    Create a BobCPU instance with configured time tagger.
    
    Args:
        use_hardware: Whether to use hardware time tagger (requires Swabian Instruments device)
    
    Returns:
        Configured BobCPU instance
    """
    
    # Configure time tagger
    timetagger_config = TimeTaggerControllerConfig(
        use_hardware=use_hardware,
        auto_fallback=True,  # Fallback to simulator if hardware fails
        timetagger_config=TimeTaggerConfig(
            resolution_ps=1000,  # 1 ps resolution
            buffer_size=100000,
            max_count_rate_hz=10000000,  # 10 MHz
            channels={
                0: TTChannelConfig(
                    enabled=True,
                    trigger_level_v=0.5,
                    dead_time_ps=50000,  # 50 ns dead time
                    input_delay_ps=0
                ),
                1: TTChannelConfig(
                    enabled=True,
                    trigger_level_v=0.5,
                    dead_time_ps=50000,
                    input_delay_ps=0
                ),
                2: TTChannelConfig(
                    enabled=True,
                    trigger_level_v=0.5,
                    dead_time_ps=50000,
                    input_delay_ps=0
                ),
                3: TTChannelConfig(
                    enabled=True,
                    trigger_level_v=0.5,
                    dead_time_ps=50000,
                    input_delay_ps=0
                )
            }
        ),
        simulator_config=SimulatorConfig(
            timing_jitter_ps=10.0,
            dark_count_rate_hz=100.0,
            enable_crosstalk=True,
            crosstalk_probability=0.001,
            enable_afterpulsing=True,
            afterpulsing_probability=0.01
        )
    )
    
    # Configure Bob
    bob_config = BobConfig(
        mode=BobMode.PASSIVE,
        measurement_duration_s=30.0,
        basis_selection_seed=42,
        timetagger_config=timetagger_config,
        enable_gated_detection=True,
        gate_width_ns=10.0
    )
    
    return BobCPU(bob_config)


def create_alice_simulator() -> AliceCPUGeneral:
    """Create Alice CPU for generating test pulses."""
    
    # Alice simulation configuration
    alice_sim_config = AliceSimConfig(
        use_hardware_qrng=False,
        use_hardware_laser=False,
        use_hardware_voa=False,
        use_hardware_polarization=False,
        qrng_config=QRNGConfig(method="pseudo", seed=123),
        laser_config=LaserConfig(power_mw=1.0, wavelength_nm=850),
        voa_config=VOAConfig(min_attenuation_db=0, max_attenuation_db=60),
        polarization_config=PolarizationConfig(basis=Basis.Z)
    )
    
    return AliceCPUGeneral(alice_sim_config)


def run_timetagger_integration_demo():
    """
    Demonstrate comprehensive time tagger integration with BobCPU.
    """
    logger.info("=== Bob CPU Time Tagger Integration Demo ===")
    
    # Create Alice and Bob
    logger.info("Creating Alice and Bob systems...")
    alice = create_alice_simulator()
    bob = create_bob_with_timetagger(use_hardware=False)  # Use simulator
    
    # Display system information
    logger.info("\n=== System Information ===")
    logger.info(f"Alice: {alice.__class__.__name__}")
    logger.info(f"Bob: {bob.__class__.__name__}")
    
    # Display time tagger status
    tt_status = bob.get_timetagger_status()
    logger.info(f"Time Tagger Status:")
    logger.info(f"  Using Hardware: {tt_status['using_hardware']}")
    logger.info(f"  Device: {tt_status['device_info']['device_type']}")
    logger.info(f"  Resolution: {tt_status['device_info']['resolution_ps']} ps")
    logger.info(f"  Channels: {list(tt_status['device_info']['channels'].keys())}")
    
    try:
        # Start measurements
        logger.info("\n=== Starting Measurements ===")
        
        # Start Bob measurement (includes time tagger)
        if not bob.start_measurement():
            logger.error("Failed to start Bob measurement")
            return
        
        # Start Alice pulse generation
        alice.start_operation()
        
        # Generate and send test pulses
        logger.info("Generating test pulses...")
        test_pulses = []
        
        for i in range(50):
            # Generate pulse from Alice
            pulse_data = alice.generate_pulse()
            if pulse_data:
                pulse = Pulse(
                    pulse_id=i,
                    timestamp=time.time(),
                    basis=pulse_data['basis'],
                    bit=pulse_data['bit'],
                    intensity=pulse_data.get('intensity', 1.0),
                    wavelength_nm=850,
                    duration_ns=1.0
                )
                
                # Send to Bob
                bob.receive_pulse(pulse)
                test_pulses.append(pulse)
                
                logger.debug(f"Sent pulse {i}: basis={pulse.basis.value}, bit={pulse.bit.value}")
            
            # Small delay between pulses
            time.sleep(0.1)
        
        logger.info(f"Generated {len(test_pulses)} test pulses")
        
        # Let measurements run for a while
        logger.info("Collecting measurements for 5 seconds...")
        time.sleep(5.0)
        
        # Analyze results
        logger.info("\n=== Analysis Results ===")
        
        # Get Bob's measurement data
        stats = bob.stats
        logger.info(f"Total measurements: {stats.total_measurements}")
        logger.info(f"Successful detections: {stats.successful_detections}")
        logger.info(f"Basis selections: {stats.basis_selections}")
        logger.info(f"Detector counts: {stats.detector_counts}")
        
        # Get time tag data
        logger.info("\n=== Time Tag Analysis ===")
        timestamps = bob.get_timetag_data()
        logger.info(f"Total time tag events: {len(timestamps)}")
        
        if timestamps:
            # Analyze by channel
            channel_counts = {}
            for ts in timestamps:
                channel_counts[ts.channel] = channel_counts.get(ts.channel, 0) + 1
            
            logger.info(f"Events per channel: {channel_counts}")
            
            # Calculate timing statistics
            if len(timestamps) > 1:
                time_diffs = []
                for i in range(1, len(timestamps)):
                    diff_ps = timestamps[i].time_ps - timestamps[i-1].time_ps
                    time_diffs.append(diff_ps)
                
                if time_diffs:
                    avg_interval_ps = np.mean(time_diffs)
                    std_interval_ps = np.std(time_diffs)
                    logger.info(f"Average interval: {avg_interval_ps:.2f} ± {std_interval_ps:.2f} ps")
        
        # Analyze coincidences
        logger.info("\n=== Coincidence Analysis ===")
        coincidences = bob.analyze_coincidences(time_window_ps=10000)  # 10 ns window
        logger.info(f"Coincidence events found: {len(coincidences)}")
        
        if coincidences:
            logger.info("Sample coincidences:")
            for i, (ts1, ts2) in enumerate(coincidences[:5]):
                time_diff_ps = abs(ts2.time_ps - ts1.time_ps)
                logger.info(f"  {i+1}: Ch{ts1.channel} & Ch{ts2.channel}, Δt = {time_diff_ps} ps")
        
        # Get count rates
        count_rates = bob.get_count_rates()
        logger.info(f"\nCount rates: {count_rates}")
        
        # Display final time tagger status
        final_status = bob.get_timetagger_status()
        logger.info(f"\nFinal statistics:")
        logger.info(f"  Total events: {final_status['statistics']['total_events']}")
        logger.info(f"  Coincidences: {final_status['statistics']['coincidences']}")
        logger.info(f"  Coincidence rate: {final_status['statistics']['coincidence_rate_hz']:.2f} Hz")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        logger.info("\n=== Cleanup ===")
        alice.stop_operation()
        bob.stop_measurement()
        logger.info("Demo completed")


def demonstrate_timetagger_features():
    """Demonstrate specific time tagger features."""
    logger.info("\n=== Time Tagger Features Demo ===")
    
    bob = create_bob_with_timetagger(use_hardware=False)
    
    try:
        # Start time tagger
        bob.start_measurement()
        
        # Test channel configuration
        logger.info("Testing channel configuration...")
        success = bob.timetagger.set_trigger_level(0, 0.3)  # Set channel 0 trigger to 0.3V
        logger.info(f"Channel 0 trigger level set: {success}")
        
        # Test individual channel enable/disable
        success = bob.timetagger.enable_channel(2, False)  # Disable channel 2
        logger.info(f"Channel 2 disabled: {success}")
        
        success = bob.timetagger.enable_channel(2, True)   # Re-enable channel 2
        logger.info(f"Channel 2 re-enabled: {success}")
        
        # Inject test events (simulator only)
        if hasattr(bob.timetagger.driver, 'inject_test_event'):
            logger.info("Injecting test events...")
            for ch in range(4):
                current_time_ps = int(time.time() * 1e12)
                success = bob.timetagger.driver.inject_test_event(ch, current_time_ps + ch * 1000)
                logger.info(f"Test event injected on channel {ch}: {success}")
        
        # Wait and collect events
        time.sleep(2.0)
        
        # Retrieve and analyze events
        timestamps = bob.get_timetag_data()
        logger.info(f"Retrieved {len(timestamps)} events")
        
        # Test buffer management
        buffer_cleared = bob.timetagger.clear_buffer()
        logger.info(f"Buffer cleared: {buffer_cleared}")
        
        # Test reset functionality
        reset_success = bob.reset_timetagger()
        logger.info(f"Time tagger reset: {reset_success}")
        
    except Exception as e:
        logger.error(f"Error in features demo: {e}")
        
    finally:
        bob.stop_measurement()


if __name__ == "__main__":
    # Run the demonstrations
    run_timetagger_integration_demo()
    demonstrate_timetagger_features()
    
    logger.info("\n=== All demonstrations completed ===")
