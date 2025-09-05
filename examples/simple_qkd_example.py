"""
Simple QKD System Example.

Demonstrates the simplified QKD components:
- Simple quantum channel (pass-through or attenuation)
- Simple optical table (perfect or with deviations)
- Simple detectors (photon count to detector number)
- Simple Bob CPU (streamlined processing)
"""

import logging
import time
import numpy as np
from typing import List, Dict, Any

# Import simple components
from src.bob import (
    SimpleBobCPU, SimpleBobConfig, SimpleBobMode,
    SimpleOpticalConfig, SimpleDetectorConfig
)
from src.quantum_channel import SimpleQuantumChannel, SimpleChannelConfig

# Import Alice for pulse generation
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


def create_simple_bob(mode: SimpleBobMode = SimpleBobMode.PERFECT) -> SimpleBobCPU:
    """
    Create a Simple Bob CPU instance.
    
    Args:
        mode: Operation mode (PERFECT, REALISTIC, or PASSIVE)
    
    Returns:
        Configured Simple Bob CPU
    """
    
    # Configure based on mode
    if mode == SimpleBobMode.PERFECT:
        channel_config = SimpleChannelConfig(
            pass_through_mode=True,
            apply_attenuation=False,
            attenuation_db=0.0,
            random_loss_probability=0.0
        )
        
        optical_config = SimpleOpticalConfig(
            perfect_measurement=True,
            apply_angular_deviation=False,
            angular_deviation_degrees=0.0
        )
        
        detector_config = SimpleDetectorConfig(
            perfect_detection=True,
            quantum_efficiency=1.0,
            dark_count_rate_hz=0.0
        )
        
    elif mode == SimpleBobMode.REALISTIC:
        channel_config = SimpleChannelConfig(
            pass_through_mode=False,
            apply_attenuation=True,
            attenuation_db=3.0,  # 3 dB channel loss
            random_loss_probability=0.05  # 5% random loss
        )
        
        optical_config = SimpleOpticalConfig(
            perfect_measurement=False,
            apply_angular_deviation=True,
            angular_deviation_degrees=2.0,  # 2° angular error
            random_angular_deviation=True,
            max_random_deviation_degrees=1.0
        )
        
        detector_config = SimpleDetectorConfig(
            perfect_detection=False,
            quantum_efficiency=0.8,  # 80% efficiency
            dark_count_rate_hz=100.0,  # 100 Hz dark counts
            photon_threshold=1
        )
        
    else:  # PASSIVE mode
        channel_config = SimpleChannelConfig(
            pass_through_mode=False,
            apply_attenuation=True,
            attenuation_db=1.0  # 1 dB loss
        )
        
        optical_config = SimpleOpticalConfig(
            perfect_measurement=False,
            apply_angular_deviation=True,
            angular_deviation_degrees=1.0
        )
        
        detector_config = SimpleDetectorConfig(
            perfect_detection=False,
            quantum_efficiency=0.9,
            dark_count_rate_hz=50.0
        )
    
    # Create Bob configuration
    bob_config = SimpleBobConfig(
        mode=mode,
        measurement_duration_s=30.0,
        basis_selection_seed=42,
        channel_config=channel_config,
        optical_config=optical_config,
        detector_config=detector_config,
        basis_z_probability=0.5
    )
    
    return SimpleBobCPU(bob_config)


def create_alice_for_testing() -> AliceCPUGeneral:
    """Create Alice CPU for generating test pulses."""
    
    alice_config = AliceSimConfig(
        use_hardware_qrng=False,
        use_hardware_laser=False,
        use_hardware_voa=False,
        use_hardware_polarization=False,
        qrng_config=QRNGConfig(method="pseudo", seed=123),
        laser_config=LaserConfig(power_mw=1.0, wavelength_nm=850),
        voa_config=VOAConfig(min_attenuation_db=0, max_attenuation_db=60),
        polarization_config=PolarizationConfig(basis=Basis.Z)
    )
    
    return AliceCPUGeneral(alice_config)


def demonstrate_perfect_mode():
    """Demonstrate perfect measurement mode."""
    logger.info("\n=== Perfect Mode Demonstration ===")
    
    alice = create_alice_for_testing()
    bob = create_simple_bob(SimpleBobMode.PERFECT)
    
    try:
        # Start systems
        alice.start_operation()
        bob.start_measurement()
        
        # Generate test pulses with known properties
        test_results = []
        
        for i in range(20):
            pulse_data = alice.generate_pulse()
            if pulse_data:
                pulse = Pulse(
                    pulse_id=i,
                    timestamp=time.time(),
                    basis=pulse_data['basis'],
                    bit=pulse_data['bit'],
                    intensity=1.0,
                    wavelength_nm=850,
                    duration_ns=1.0
                )
                
                # Send to Bob
                bob.receive_pulse(pulse)
                
                test_results.append({
                    'sent_basis': pulse.basis,
                    'sent_bit': pulse.bit,
                    'pulse_id': i
                })
                
                time.sleep(0.05)  # 50ms between pulses
        
        # Allow processing
        time.sleep(2.0)
        
        # Analyze results
        summary = bob.get_measurement_summary()
        logger.info(f"Perfect mode results:")
        logger.info(f"  Total pulses: {summary['statistics']['total_pulses']}")
        logger.info(f"  Total detections: {summary['statistics']['total_detections']}")
        logger.info(f"  Detection efficiency: {summary['statistics']['detection_efficiency']:.3f}")
        logger.info(f"  Detector events: {summary['statistics']['detector_events']}")
        logger.info(f"  Channel efficiency: {summary['channel_efficiency']:.3f}")
        
        # In perfect mode, we should have 100% detection efficiency
        assert summary['statistics']['detection_efficiency'] >= 0.9, "Perfect mode should have high efficiency"
        
    finally:
        alice.stop_operation()
        bob.stop_measurement()
        del alice, bob


def demonstrate_realistic_mode():
    """Demonstrate realistic measurement mode with losses."""
    logger.info("\n=== Realistic Mode Demonstration ===")
    
    alice = create_alice_for_testing()
    bob = create_simple_bob(SimpleBobMode.REALISTIC)
    
    try:
        # Start systems
        alice.start_operation()
        bob.start_measurement()
        
        # Generate more pulses to account for losses
        for i in range(50):
            pulse_data = alice.generate_pulse()
            if pulse_data:
                pulse = Pulse(
                    pulse_id=i,
                    timestamp=time.time(),
                    basis=pulse_data['basis'],
                    bit=pulse_data['bit'],
                    intensity=1.0,
                    wavelength_nm=850,
                    duration_ns=1.0
                )
                
                bob.receive_pulse(pulse)
                time.sleep(0.02)  # 20ms between pulses
        
        # Allow processing
        time.sleep(2.0)
        
        # Analyze results
        summary = bob.get_measurement_summary()
        logger.info(f"Realistic mode results:")
        logger.info(f"  Total pulses: {summary['statistics']['total_pulses']}")
        logger.info(f"  Total detections: {summary['statistics']['total_detections']}")
        logger.info(f"  Detection efficiency: {summary['statistics']['detection_efficiency']:.3f}")
        logger.info(f"  Channel efficiency: {summary['channel_efficiency']:.3f}")
        logger.info(f"  Detector events: {summary['statistics']['detector_events']}")
        logger.info(f"  Detector statistics: {summary['detector_stats']}")
        
        # In realistic mode, efficiency should be lower due to losses
        assert summary['statistics']['detection_efficiency'] < 0.9, "Realistic mode should have losses"
        
    finally:
        alice.stop_operation()
        bob.stop_measurement()
        del alice, bob


def demonstrate_batch_processing():
    """Demonstrate batch processing of pulses."""
    logger.info("\n=== Batch Processing Demonstration ===")
    
    alice = create_alice_for_testing()
    bob = create_simple_bob(SimpleBobMode.PASSIVE)
    
    try:
        alice.start_operation()
        
        # Generate batch of pulses
        pulse_batch = []
        for i in range(30):
            pulse_data = alice.generate_pulse()
            if pulse_data:
                pulse = Pulse(
                    pulse_id=i,
                    timestamp=time.time(),
                    basis=pulse_data['basis'],
                    bit=pulse_data['bit'],
                    intensity=1.0,
                    wavelength_nm=850,
                    duration_ns=1.0
                )
                pulse_batch.append(pulse)
        
        logger.info(f"Generated batch of {len(pulse_batch)} pulses")
        
        # Process batch
        start_time = time.time()
        detector_numbers = bob.process_pulse_batch(pulse_batch)
        processing_time = time.time() - start_time
        
        # Analyze batch results
        detections = [d for d in detector_numbers if d is not None]
        logger.info(f"Batch processing results:")
        logger.info(f"  Processing time: {processing_time:.3f} seconds")
        logger.info(f"  Successful detections: {len(detections)}/{len(pulse_batch)}")
        logger.info(f"  Detection rate: {len(detections)/len(pulse_batch):.3f}")
        
        # Count detections per detector
        detector_counts = {}
        for det_num in detections:
            detector_name = bob.get_detector_name(det_num)
            detector_counts[detector_name] = detector_counts.get(detector_name, 0) + 1
        
        logger.info(f"  Detections per detector: {detector_counts}")
        
    finally:
        alice.stop_operation()
        del alice, bob


def demonstrate_parameter_tuning():
    """Demonstrate real-time parameter tuning."""
    logger.info("\n=== Parameter Tuning Demonstration ===")
    
    alice = create_alice_for_testing()
    bob = create_simple_bob(SimpleBobMode.PASSIVE)
    
    try:
        alice.start_operation()
        bob.start_measurement()
        
        # Test different attenuation levels
        attenuation_levels = [0.0, 1.0, 3.0, 6.0, 10.0]
        results = {}
        
        for attenuation_db in attenuation_levels:
            logger.info(f"Testing {attenuation_db} dB attenuation...")
            
            # Set attenuation
            bob.set_channel_attenuation(attenuation_db)
            
            # Reset measurements
            bob.reset_measurements()
            
            # Send test pulses
            for i in range(20):
                pulse_data = alice.generate_pulse()
                if pulse_data:
                    pulse = Pulse(
                        pulse_id=i,
                        timestamp=time.time(),
                        basis=pulse_data['basis'],
                        bit=pulse_data['bit'],
                        intensity=1.0,
                        wavelength_nm=850,
                        duration_ns=1.0
                    )
                    bob.receive_pulse(pulse)
                    time.sleep(0.01)
            
            # Allow processing
            time.sleep(1.0)
            
            # Record results
            summary = bob.get_measurement_summary()
            results[attenuation_db] = summary['statistics']['detection_efficiency']
            
            logger.info(f"  Detection efficiency: {results[attenuation_db]:.3f}")
        
        # Display attenuation curve
        logger.info(f"\nAttenuation vs Detection Efficiency:")
        for att_db, efficiency in results.items():
            logger.info(f"  {att_db:4.1f} dB → {efficiency:.3f}")
        
        # Test detector efficiency tuning
        logger.info(f"\nTesting detector efficiency tuning...")
        bob.set_pass_through_mode(True)  # Remove channel effects
        
        efficiency_levels = [0.5, 0.7, 0.9, 1.0]
        for efficiency in efficiency_levels:
            bob.set_detector_efficiency(efficiency)
            bob.reset_measurements()
            
            # Send test pulses
            for i in range(15):
                pulse_data = alice.generate_pulse()
                if pulse_data:
                    pulse = Pulse(
                        pulse_id=i,
                        timestamp=time.time(),
                        basis=pulse_data['basis'],
                        bit=pulse_data['bit'],
                        intensity=1.0,
                        wavelength_nm=850,
                        duration_ns=1.0
                    )
                    bob.receive_pulse(pulse)
                    time.sleep(0.01)
            
            time.sleep(0.5)
            summary = bob.get_measurement_summary()
            measured_efficiency = summary['statistics']['detection_efficiency']
            
            logger.info(f"  Set: {efficiency:.1f} → Measured: {measured_efficiency:.3f}")
        
    finally:
        alice.stop_operation()
        bob.stop_measurement()
        del alice, bob


def run_comprehensive_test():
    """Run comprehensive test of all simple components."""
    logger.info("\n=== Comprehensive Simple QKD Test ===")
    
    alice = create_alice_for_testing()
    
    # Test all three modes
    modes = [SimpleBobMode.PERFECT, SimpleBobMode.REALISTIC, SimpleBobMode.PASSIVE]
    mode_results = {}
    
    try:
        alice.start_operation()
        
        for mode in modes:
            logger.info(f"\nTesting {mode.value} mode...")
            
            bob = create_simple_bob(mode)
            bob.start_measurement()
            
            # Send test pulses
            for i in range(25):
                pulse_data = alice.generate_pulse()
                if pulse_data:
                    pulse = Pulse(
                        pulse_id=i,
                        timestamp=time.time(),
                        basis=pulse_data['basis'],
                        bit=pulse_data['bit'],
                        intensity=1.0,
                        wavelength_nm=850,
                        duration_ns=1.0
                    )
                    bob.receive_pulse(pulse)
                    time.sleep(0.02)
            
            time.sleep(1.5)
            
            # Record results
            summary = bob.get_measurement_summary()
            mode_results[mode.value] = {
                'detection_efficiency': summary['statistics']['detection_efficiency'],
                'channel_efficiency': summary['channel_efficiency'],
                'total_detections': summary['statistics']['total_detections']
            }
            
            bob.stop_measurement()
            del bob
        
        # Display comparison
        logger.info(f"\n=== Mode Comparison ===")
        logger.info(f"{'Mode':<12} {'Detection':<12} {'Channel':<12} {'Detections'}")
        logger.info(f"{'':12} {'Efficiency':<12} {'Efficiency':<12} {'Count'}")
        logger.info("-" * 50)
        
        for mode_name, results in mode_results.items():
            logger.info(f"{mode_name:<12} {results['detection_efficiency']:<12.3f} "
                       f"{results['channel_efficiency']:<12.3f} {results['total_detections']}")
        
    finally:
        alice.stop_operation()
        del alice


if __name__ == "__main__":
    logger.info("=== Simple QKD Components Demonstration ===")
    
    # Run all demonstrations
    demonstrate_perfect_mode()
    demonstrate_realistic_mode()
    demonstrate_batch_processing()
    demonstrate_parameter_tuning()
    run_comprehensive_test()
    
    logger.info("\n=== All demonstrations completed successfully ===")
    logger.info("Simple components provide streamlined QKD functionality:")
    logger.info("- Simple quantum channel: pass-through or attenuation modes")
    logger.info("- Simple optical table: perfect or angular deviation modes")
    logger.info("- Simple detectors: photon count → detector number mapping")
    logger.info("- Simple Bob CPU: streamlined processing pipeline")
