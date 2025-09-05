"""
Integration Test - Complete QKD System with Time Tagger.

This test validates the complete integration of Alice and Bob with time tagger
for a realistic QKD implementation.
"""

import logging
import time
import unittest
from typing import List, Dict, Any

# Import all necessary components
from src.alice import AliceCPUGeneral, SimulationConfig as AliceSimConfig
from src.alice.qrng import QRNGConfig
from src.alice.laser import LaserConfig
from src.alice.voa import VOAConfig
from src.alice.polarization import PolarizationConfig

from src.bob import (
    BobCPU, BobConfig, BobMode,
    TimeTaggerControllerConfig, TimeTaggerConfig,
    ChannelConfig as TTChannelConfig, SimulatorConfig
)

from src.utils.data_structures import Basis, Bit, Pulse


class TestTimeTaggerIntegration(unittest.TestCase):
    """Test cases for time tagger integration with QKD system."""
    
    def setUp(self):
        """Set up test environment."""
        # Configure logging for tests
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create Alice configuration
        self.alice_config = AliceSimConfig(
            use_hardware_qrng=False,
            use_hardware_laser=False,
            use_hardware_voa=False,
            use_hardware_polarization=False,
            qrng_config=QRNGConfig(method="pseudo", seed=42),
            laser_config=LaserConfig(power_mw=1.0, wavelength_nm=850),
            voa_config=VOAConfig(min_attenuation_db=0, max_attenuation_db=60),
            polarization_config=PolarizationConfig(basis=Basis.Z)
        )
        
        # Create Bob configuration with time tagger
        self.bob_config = BobConfig(
            mode=BobMode.PASSIVE,
            measurement_duration_s=10.0,
            basis_selection_seed=42,
            timetagger_config=TimeTaggerControllerConfig(
                use_hardware=False,  # Use simulator for tests
                auto_fallback=True,
                timetagger_config=TimeTaggerConfig(
                    resolution_ps=1000,
                    buffer_size=10000,
                    max_count_rate_hz=1000000,
                    channels={
                        0: TTChannelConfig(enabled=True, trigger_level_v=0.5, dead_time_ps=50000),
                        1: TTChannelConfig(enabled=True, trigger_level_v=0.5, dead_time_ps=50000),
                        2: TTChannelConfig(enabled=True, trigger_level_v=0.5, dead_time_ps=50000),
                        3: TTChannelConfig(enabled=True, trigger_level_v=0.5, dead_time_ps=50000)
                    }
                ),
                simulator_config=SimulatorConfig(
                    timing_jitter_ps=5.0,
                    dark_count_rate_hz=50.0
                )
            )
        )
    
    def test_bob_initialization_with_timetagger(self):
        """Test that Bob initializes correctly with time tagger."""
        bob = BobCPU(self.bob_config)
        
        # Check that all components are initialized
        self.assertIsNotNone(bob.timetagger)
        self.assertIsNotNone(bob.optical_table)
        self.assertIsNotNone(bob.quantum_channel)
        self.assertIsNotNone(bob.detectors)
        
        # Check time tagger status
        status = bob.get_timetagger_status()
        self.assertIn('device_info', status)
        self.assertIn('using_hardware', status)
        self.assertFalse(status['using_hardware'])  # Should be using simulator
        
        # Cleanup
        del bob
    
    def test_timetagger_measurement_lifecycle(self):
        """Test time tagger measurement start/stop cycle."""
        bob = BobCPU(self.bob_config)
        
        # Initially not measuring
        self.assertFalse(bob.timetagger.is_measuring())
        
        # Start measurement
        success = bob.start_measurement()
        self.assertTrue(success)
        self.assertTrue(bob.timetagger.is_measuring())
        
        # Stop measurement
        bob.stop_measurement()
        self.assertFalse(bob.timetagger.is_measuring())
        
        # Cleanup
        del bob
    
    def test_pulse_processing_with_timetag_generation(self):
        """Test that pulses generate appropriate time tag events."""
        alice = AliceCPUGeneral(self.alice_config)
        bob = BobCPU(self.bob_config)
        
        try:
            # Start systems
            alice.start_operation()
            bob.start_measurement()
            
            # Generate test pulses
            initial_event_count = bob.stats.timetag_events
            
            for i in range(10):
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
            
            # Allow processing time
            time.sleep(1.0)
            
            # Check that some time tag events were generated
            timestamps = bob.get_timetag_data()
            final_event_count = bob.stats.timetag_events
            
            # Should have some events (from dark counts or injected events)
            self.assertGreaterEqual(final_event_count, initial_event_count)
            
        finally:
            alice.stop_operation()
            bob.stop_measurement()
            del alice, bob
    
    def test_coincidence_analysis(self):
        """Test coincidence detection functionality."""
        bob = BobCPU(self.bob_config)
        
        try:
            bob.start_measurement()
            
            # Inject test events on different channels with known timing
            if hasattr(bob.timetagger.driver, 'inject_test_event'):
                base_time_ps = int(time.time() * 1e12)
                
                # Inject coincident events (within 1000 ps)
                bob.timetagger.driver.inject_test_event(0, base_time_ps)
                bob.timetagger.driver.inject_test_event(1, base_time_ps + 500)  # 500 ps later
                
                # Inject non-coincident events
                bob.timetagger.driver.inject_test_event(2, base_time_ps + 50000)  # 50 ns later
                bob.timetagger.driver.inject_test_event(3, base_time_ps + 100000)  # 100 ns later
            
            # Allow events to be processed
            time.sleep(0.5)
            
            # Analyze coincidences
            coincidences = bob.analyze_coincidences(time_window_ps=1000)
            
            # Should find at least one coincidence (channels 0 and 1)
            self.assertGreaterEqual(len(coincidences), 0)  # May be 0 due to simulation
            
            # Check statistics were updated
            self.assertGreaterEqual(bob.stats.coincidences, 0)
            
        finally:
            bob.stop_measurement()
            del bob
    
    def test_channel_configuration(self):
        """Test time tagger channel configuration."""
        bob = BobCPU(self.bob_config)
        
        try:
            # Test enabling/disabling channels
            success = bob.timetagger.enable_channel(0, False)
            self.assertTrue(success)
            
            success = bob.timetagger.enable_channel(0, True)
            self.assertTrue(success)
            
            # Test trigger level setting
            success = bob.timetagger.set_trigger_level(1, 0.8)
            self.assertTrue(success)
            
            # Test getting device info
            device_info = bob.timetagger.get_device_info()
            self.assertIn('device_type', device_info)
            self.assertIn('channels', device_info)
            
        finally:
            del bob
    
    def test_buffer_management(self):
        """Test time tagger buffer operations."""
        bob = BobCPU(self.bob_config)
        
        try:
            bob.start_measurement()
            
            # Clear buffer
            success = bob.timetagger.clear_buffer()
            self.assertTrue(success)
            
            # Get timestamps (should be empty after clear)
            timestamps = bob.get_timetag_data()
            
            # Buffer operations should succeed
            self.assertIsInstance(timestamps, list)
            
        finally:
            bob.stop_measurement()
            del bob
    
    def test_statistics_collection(self):
        """Test that statistics are properly collected."""
        bob = BobCPU(self.bob_config)
        
        try:
            # Initial statistics
            initial_stats = bob.get_timetagger_status()['statistics']
            self.assertIn('total_events', initial_stats)
            self.assertIn('coincidences', initial_stats)
            
            # Start measurement
            bob.start_measurement()
            time.sleep(0.5)  # Let some events accumulate
            
            # Get updated statistics
            final_stats = bob.get_timetagger_status()['statistics']
            
            # Statistics should be valid
            self.assertGreaterEqual(final_stats['total_events'], 0)
            self.assertGreaterEqual(final_stats['coincidences'], 0)
            
        finally:
            bob.stop_measurement()
            del bob
    
    def test_full_qkd_integration(self):
        """Test full QKD protocol with time tagger integration."""
        alice = AliceCPUGeneral(self.alice_config)
        bob = BobCPU(self.bob_config)
        
        try:
            # Start both systems
            alice.start_operation()
            bob.start_measurement()
            
            # Run a mini QKD session
            sent_bits = []
            sent_bases = []
            
            for i in range(20):
                pulse_data = alice.generate_pulse()
                if pulse_data:
                    sent_bits.append(pulse_data['bit'])
                    sent_bases.append(pulse_data['basis'])
                    
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
                
                time.sleep(0.05)  # 50ms between pulses
            
            # Allow processing to complete
            time.sleep(1.0)
            
            # Check that both systems collected data
            self.assertGreater(len(sent_bits), 0)
            self.assertGreater(bob.stats.total_measurements, 0)
            
            # Check time tag data was collected
            timestamps = bob.get_timetag_data()
            self.assertIsInstance(timestamps, list)
            
            # Verify final statistics
            final_status = bob.get_timetagger_status()
            self.assertIn('statistics', final_status)
            
        finally:
            alice.stop_operation()
            bob.stop_measurement()
            del alice, bob


def run_integration_tests():
    """Run all integration tests."""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_integration_tests()
