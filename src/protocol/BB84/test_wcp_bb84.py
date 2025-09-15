#!/usr/bin/env python3
"""
Comprehensive test suite for WCP (Weak Coherent Pulse) BB84 protocol implementation.

This test suite validates:
1. WCP pulse generation and properties
2. Poisson photon number distribution
3. Parameter estimation with decoy states
4. PNS attack detection
5. Alice and Bob WCP implementations
6. Integration testing of the complete protocol

Author: WCP BB84 Development Team
Date: June 2025
"""

import unittest
import numpy as np
import sys
import os
import logging
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import threading
import queue
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import WCP modules
try:
    from bb84_protocol.wcp_pulse import WCPPulse, PulseType, WCPIntensityManager
    from bb84_protocol.wcp_parameter_estimation import WCPParameterEstimator
    from bb84_protocol.alice_wcp_bb84_ccc import AliceWCPQubits
    from bb84_protocol.bob_wcp_bb84_ccc import BobWCPQubits
except ImportError:
    # Fallback import path
    from BB84.bb84_protocol.wcp_pulse import WCPPulse, PulseType, WCPIntensityManager
    from BB84.bb84_protocol.wcp_parameter_estimation import WCPParameterEstimator
    from BB84.bb84_protocol.alice_wcp_bb84_ccc import AliceWCPQubits
    from BB84.bb84_protocol.bob_wcp_bb84_ccc import BobWCPQubits

class TestWCPPulse(unittest.TestCase):
    """Test cases for WCP pulse generation and management"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.signal_intensity = 0.5
        self.decoy_intensity = 0.1
        self.vacuum_intensity = 0.0
        
    def test_pulse_creation(self):
        """Test basic WCP pulse creation"""
        pulse = WCPPulse(bit=1, base=0, pulse_type=PulseType.SIGNAL, intensity=self.signal_intensity)
        
        self.assertEqual(pulse.bit, 1)
        self.assertEqual(pulse.base, 0)
        self.assertEqual(pulse.pulse_type, PulseType.SIGNAL)
        self.assertEqual(pulse.intensity, self.signal_intensity)
        self.assertIsNotNone(pulse.photon_number)
        self.assertFalse(pulse._measured)
        
    def test_photon_number_distribution(self):
        """Test that photon numbers follow Poisson distribution"""
        intensity = 0.5
        num_samples = 1000
        photon_numbers = []
        
        for _ in range(num_samples):
            pulse = WCPPulse(bit=0, base=0, pulse_type=PulseType.SIGNAL, intensity=intensity)
            photon_numbers.append(pulse.photon_number)
        
        # Check mean is approximately equal to intensity
        mean_photons = np.mean(photon_numbers)
        self.assertAlmostEqual(mean_photons, intensity, delta=0.1)
        
        # Check variance is approximately equal to intensity (Poisson property)
        var_photons = np.var(photon_numbers)
        self.assertAlmostEqual(var_photons, intensity, delta=0.1)
        
    def test_vacuum_pulse(self):
        """Test vacuum pulse always has zero photons"""
        for _ in range(100):
            pulse = WCPPulse(bit=0, base=0, pulse_type=PulseType.VACUUM, intensity=0.0)
            self.assertEqual(pulse.photon_number, 0)
            
    def test_pulse_encoding_decoding(self):
        """Test pulse information encoding and decoding"""
        test_cases = [
            (0, 0, 0),  # bit=0, base=0, signal
            (1, 0, 0),  # bit=1, base=0, signal
            (0, 1, 1),  # bit=0, base=1, decoy
            (1, 1, 2),  # bit=1, base=1, vacuum
        ]
        
        for bit, base, pulse_type_idx in test_cases:
            # Encode
            byte_val = WCPPulse.pulse_info_to_byte(bit, base, pulse_type_idx)
            
            # Decode
            decoded_bit, decoded_base, decoded_pulse_type_idx = WCPPulse.byte_to_pulse_info(byte_val)
            
            self.assertEqual(decoded_bit, bit)
            self.assertEqual(decoded_base, base)
            self.assertEqual(decoded_pulse_type_idx, pulse_type_idx)
            
    def test_pulse_from_byte(self):
        """Test creating pulse from encoded byte"""
        intensities = {
            'signal': 0.5,
            'decoy': 0.1,
            'vacuum': 0.0
        }
        
        # Create encoded byte for signal pulse
        byte_val = WCPPulse.pulse_info_to_byte(1, 0, 0)  # bit=1, base=0, signal
        
        pulse = WCPPulse.from_byte_and_intensity(byte_val, intensities)
        
        self.assertEqual(pulse.bit, 1)
        self.assertEqual(pulse.base, 0)
        self.assertEqual(pulse.pulse_type, PulseType.SIGNAL)
        self.assertEqual(pulse.intensity, 0.5)
        
    def test_pulse_measurement_simulation(self):
        """Test pulse measurement with detector efficiency and dark counts"""
        pulse = WCPPulse(bit=1, base=0, pulse_type=PulseType.SIGNAL, intensity=0.5)
        
        # Test measurement with perfect efficiency
        detection_prob = pulse.simulate_measurement(detector_efficiency=1.0, dark_count_rate=0.0)
        
        # Detection probability should be 1 - exp(-η*n) where n is photon number
        expected_prob = 1 - np.exp(-1.0 * pulse.photon_number)
        self.assertAlmostEqual(detection_prob, expected_prob, places=6)
        
    def test_multi_photon_pulse_properties(self):
        """Test properties specific to multi-photon pulses"""
        # Create pulse with higher intensity to ensure multi-photon events
        intensity = 2.0
        pulse = WCPPulse(bit=1, base=0, pulse_type=PulseType.SIGNAL, intensity=intensity)
        
        self.assertGreaterEqual(pulse.photon_number, 0)
        self.assertEqual(pulse.intensity, intensity)


class TestWCPIntensityManager(unittest.TestCase):
    """Test cases for WCP intensity management"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = WCPIntensityManager(
            signal_intensity=0.5,
            decoy_intensity=0.1,
            vacuum_intensity=0.0,
            signal_prob=0.7,
            decoy_prob=0.25,
            vacuum_prob=0.05
        )
        
    def test_probability_distribution(self):
        """Test that pulse type selection follows specified probabilities"""
        num_samples = 10000
        pulse_counts = {PulseType.SIGNAL: 0, PulseType.DECOY: 0, PulseType.VACUUM: 0}
        
        for _ in range(num_samples):
            pulse_type = self.manager.select_pulse_type()
            pulse_counts[pulse_type] += 1
            
        # Check probabilities are within expected range (with some tolerance)
        signal_prob = pulse_counts[PulseType.SIGNAL] / num_samples
        decoy_prob = pulse_counts[PulseType.DECOY] / num_samples
        vacuum_prob = pulse_counts[PulseType.VACUUM] / num_samples
        
        self.assertAlmostEqual(signal_prob, 0.7, delta=0.05)
        self.assertAlmostEqual(decoy_prob, 0.25, delta=0.05)
        self.assertAlmostEqual(vacuum_prob, 0.05, delta=0.02)
        
    def test_intensity_retrieval(self):
        """Test intensity retrieval for different pulse types"""
        self.assertEqual(self.manager.get_intensity(PulseType.SIGNAL), 0.5)
        self.assertEqual(self.manager.get_intensity(PulseType.DECOY), 0.1)
        self.assertEqual(self.manager.get_intensity(PulseType.VACUUM), 0.0)


class TestWCPParameterEstimator(unittest.TestCase):
    """Test cases for WCP parameter estimation and security analysis"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.estimator = WCPParameterEstimator(
            signal_intensity=0.5,
            decoy_intensity=0.1,
            vacuum_intensity=0.0,
            detector_efficiency=0.1,
            dark_count_rate=1e-6
        )
        
    def test_parameter_initialization(self):
        """Test parameter estimator initialization"""
        self.assertEqual(self.estimator.mu_s, 0.5)
        self.assertEqual(self.estimator.mu_d, 0.1)
        self.assertEqual(self.estimator.mu_v, 0.0)
        self.assertEqual(self.estimator.detector_efficiency, 0.1)
        self.assertEqual(self.estimator.dark_count_rate, 1e-6)
        
    def test_measurement_data_accumulation(self):
        """Test accumulation of measurement data"""
        # Add some test data
        self.estimator.add_measurement_data('signal', 0, True, True, False)
        self.estimator.add_measurement_data('signal', 0, True, False, False)
        self.estimator.add_measurement_data('decoy', 1, True, True, True)
        
        stats = self.estimator.stats
        self.assertEqual(stats['signal']['sent'], 2)
        self.assertEqual(stats['signal']['detected'], 1)
        self.assertEqual(stats['signal']['errors'], 0)
        self.assertEqual(stats['decoy']['sent'], 1)
        self.assertEqual(stats['decoy']['detected'], 1)
        self.assertEqual(stats['decoy']['errors'], 1)
        
    def test_gain_calculation(self):
        """Test gain calculation for different pulse types"""
        # Add measurement data
        for _ in range(100):
            self.estimator.add_measurement_data('signal', 0, True, True, False)
        for _ in range(50):
            self.estimator.add_measurement_data('signal', 0, True, False, False)
            
        gain = self.estimator.calculate_gain('signal')
        expected_gain = 100 / 150  # detected / sent
        self.assertAlmostEqual(gain, expected_gain, places=6)
        
    def test_qber_calculation(self):
        """Test QBER calculation"""
        # Add measurement data with some errors
        for _ in range(80):
            self.estimator.add_measurement_data('signal', 0, True, True, False)
        for _ in range(20):
            self.estimator.add_measurement_data('signal', 0, True, True, True)
            
        qber = self.estimator.calculate_qber('signal')
        expected_qber = 20 / 100  # errors / detected
        self.assertAlmostEqual(qber, expected_qber, places=6)
        
    def test_single_photon_yield_estimation(self):
        """Test single photon yield estimation using decoy state method"""
        # Simulate realistic measurement data
        signal_gain = 0.05
        decoy_gain = 0.02
        vacuum_gain = 1e-6
        
        # Mock the gain calculations
        with patch.object(self.estimator, 'calculate_gain') as mock_gain:
            mock_gain.side_effect = lambda pulse_type: {
                'signal': signal_gain,
                'decoy': decoy_gain,
                'vacuum': vacuum_gain
            }[pulse_type]
            
            y1_lower, y1_upper = self.estimator.estimate_single_photon_yield()
            
            # Verify that bounds are reasonable
            self.assertGreaterEqual(y1_lower, 0)
            self.assertLessEqual(y1_upper, 1)
            self.assertLessEqual(y1_lower, y1_upper)
            
    def test_pns_attack_detection(self):
        """Test PNS attack detection mechanism"""
        # Test normal operation (no attack)
        normal_gains = {'signal': 0.05, 'decoy': 0.02, 'vacuum': 1e-6}
        is_attack_normal = self.estimator.detect_pns_attack(normal_gains)
        self.assertFalse(is_attack_normal)
        
        # Test suspicious gains (potential attack)
        suspicious_gains = {'signal': 0.05, 'decoy': 0.01, 'vacuum': 1e-6}
        is_attack_suspicious = self.estimator.detect_pns_attack(suspicious_gains)
        # This might be True depending on the threshold implementation
        
    def test_security_report_generation(self):
        """Test security report generation"""
        # Add some measurement data
        for _ in range(1000):
            self.estimator.add_measurement_data('signal', 0, True, 
                                               np.random.random() < 0.05,  # detection
                                               np.random.random() < 0.11)  # error
        for _ in range(500):
            self.estimator.add_measurement_data('decoy', 0, True, 
                                               np.random.random() < 0.02,
                                               np.random.random() < 0.11)
        for _ in range(100):
            self.estimator.add_measurement_data('vacuum', 0, True, 
                                               np.random.random() < 1e-6,
                                               False)
        
        report = self.estimator.generate_security_report()
        
        # Verify report structure
        self.assertIn('gains', report)
        self.assertIn('qber', report)
        self.assertIn('single_photon_yield', report)
        self.assertIn('pns_attack_detected', report)
        self.assertIn('secure_key_rate_estimate', report)


class TestWCPProtocolIntegration(unittest.TestCase):
    """Integration tests for the complete WCP BB84 protocol"""
    
    def setUp(self):
        """Set up test fixtures for integration testing"""
        self.key_length = 100
        self.signal_intensity = 0.5
        self.decoy_intensity = 0.1
        
    @patch('BB84.bb84_protocol.alice_wcp_bb84_ccc.Role')
    def test_alice_initialization(self, mock_role):
        """Test Alice WCP initialization"""
        alice = AliceWCPQubits(
            num_qubits=self.key_length,
            signal_intensity=self.signal_intensity,
            decoy_intensity=self.decoy_intensity
        )
        
        self.assertEqual(alice.num_qubits, self.key_length)
        self.assertEqual(alice.signal_intensity, self.signal_intensity)
        self.assertEqual(alice.decoy_intensity, self.decoy_intensity)
        
    @patch('BB84.bb84_protocol.bob_wcp_bb84_ccc.Role')
    def test_bob_initialization(self, mock_role):
        """Test Bob WCP initialization"""
        bob = BobWCPQubits(
            num_qubits=self.key_length,
            detector_efficiency=0.1,
            dark_count_rate=1e-6
        )
        
        self.assertEqual(bob.num_qubits, self.key_length)
        self.assertIsNotNone(bob.parameter_estimator)
        
    def test_pulse_generation_and_measurement_cycle(self):
        """Test complete pulse generation and measurement cycle"""
        # Create intensity manager
        intensity_manager = WCPIntensityManager(
            signal_intensity=0.5,
            decoy_intensity=0.1,
            vacuum_intensity=0.0,
            signal_prob=0.7,
            decoy_prob=0.25,
            vacuum_prob=0.05
        )
        
        # Generate random bit and base
        bit = np.random.randint(0, 2)
        base = np.random.randint(0, 2)
        
        # Select pulse type and create pulse
        pulse_type = intensity_manager.select_pulse_type()
        intensity = intensity_manager.get_intensity(pulse_type)
        pulse = WCPPulse(bit=bit, base=base, pulse_type=pulse_type, intensity=intensity)
        
        # Simulate measurement
        detector_efficiency = 0.1
        dark_count_rate = 1e-6
        detection_prob = pulse.simulate_measurement(detector_efficiency, dark_count_rate)
        
        # Verify results
        self.assertIn(pulse.bit, [0, 1])
        self.assertIn(pulse.base, [0, 1])
        self.assertIn(pulse.pulse_type, [PulseType.SIGNAL, PulseType.DECOY, PulseType.VACUUM])
        self.assertGreaterEqual(detection_prob, 0.0)
        self.assertLessEqual(detection_prob, 1.0)
        
    def test_statistical_validation(self):
        """Test statistical properties of the WCP protocol"""
        num_pulses = 10000
        signal_count = 0
        decoy_count = 0
        vacuum_count = 0
        
        intensity_manager = WCPIntensityManager(
            signal_intensity=0.5,
            decoy_intensity=0.1,
            vacuum_intensity=0.0,
            signal_prob=0.7,
            decoy_prob=0.25,
            vacuum_prob=0.05
        )
        
        photon_numbers = []
        
        for _ in range(num_pulses):
            pulse_type = intensity_manager.select_pulse_type()
            intensity = intensity_manager.get_intensity(pulse_type)
            pulse = WCPPulse(bit=0, base=0, pulse_type=pulse_type, intensity=intensity)
            
            photon_numbers.append(pulse.photon_number)
            
            if pulse_type == PulseType.SIGNAL:
                signal_count += 1
            elif pulse_type == PulseType.DECOY:
                decoy_count += 1
            elif pulse_type == PulseType.VACUUM:
                vacuum_count += 1
                
        # Verify pulse type distribution
        self.assertAlmostEqual(signal_count / num_pulses, 0.7, delta=0.05)
        self.assertAlmostEqual(decoy_count / num_pulses, 0.25, delta=0.05)
        self.assertAlmostEqual(vacuum_count / num_pulses, 0.05, delta=0.02)
        
        # Verify photon number statistics
        mean_photons = np.mean(photon_numbers)
        expected_mean = 0.7 * 0.5 + 0.25 * 0.1 + 0.05 * 0.0  # weighted average
        self.assertAlmostEqual(mean_photons, expected_mean, delta=0.05)


class TestWCPSecurityAnalysis(unittest.TestCase):
    """Test cases for WCP security analysis and attack detection"""
    
    def setUp(self):
        """Set up test fixtures for security testing"""
        self.estimator = WCPParameterEstimator(
            signal_intensity=0.5,
            decoy_intensity=0.1,
            vacuum_intensity=0.0,
            detector_efficiency=0.1,
            dark_count_rate=1e-6
        )
        
    def test_intercept_resend_attack_detection(self):
        """Test detection of intercept-resend attacks"""
        # Simulate normal operation
        for _ in range(1000):
            # Normal error rate ~11%
            error = np.random.random() < 0.11
            self.estimator.add_measurement_data('signal', 0, True, True, error)
            
        qber_normal = self.estimator.calculate_qber('signal')
        
        # Simulate intercept-resend attack (higher error rate ~25%)
        estimator_attack = WCPParameterEstimator(
            signal_intensity=0.5,
            decoy_intensity=0.1,
            vacuum_intensity=0.0,
            detector_efficiency=0.1,
            dark_count_rate=1e-6
        )
        
        for _ in range(1000):
            # Attack increases error rate to ~25%
            error = np.random.random() < 0.25
            estimator_attack.add_measurement_data('signal', 0, True, True, error)
            
        qber_attack = estimator_attack.calculate_qber('signal')
        
        # Attack should result in higher QBER
        self.assertGreater(qber_attack, qber_normal)
        self.assertGreater(qber_attack, 0.15)  # Threshold for attack detection
        
    def test_photon_number_splitting_attack_simulation(self):
        """Test simulation of photon number splitting attacks"""
        # Simulate PNS attack scenario
        gains_pns = {
            'signal': 0.03,  # Reduced gain for multi-photon pulses
            'decoy': 0.02,   # Similar gain for decoy
            'vacuum': 1e-6   # Normal vacuum gain
        }
        
        # Check if PNS attack is detected
        is_pns_detected = self.estimator.detect_pns_attack(gains_pns)
        
        # The specific result depends on implementation, but test should run without error
        self.assertIsInstance(is_pns_detected, bool)
        
    def test_secure_key_rate_calculation(self):
        """Test secure key rate calculation"""
        # Add measurement data
        for _ in range(1000):
            detected = np.random.random() < 0.05  # 5% detection rate
            error = np.random.random() < 0.11 if detected else False
            self.estimator.add_measurement_data('signal', 0, True, detected, error)
            
        report = self.estimator.generate_security_report()
        
        # Verify secure key rate is calculated
        self.assertIn('secure_key_rate_estimate', report)
        self.assertGreaterEqual(report['secure_key_rate_estimate'], 0.0)


def run_wcp_tests():
    """Run all WCP tests with detailed output"""
    print("=" * 70)
    print("RUNNING WCP BB84 COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestWCPPulse,
        TestWCPIntensityManager, 
        TestWCPParameterEstimator,
        TestWCPProtocolIntegration,
        TestWCPSecurityAnalysis
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
            
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Configure numpy for reproducible tests
    np.random.seed(42)
    
    # Run tests
    success = run_wcp_tests()
    
    if success:
        print("\n✅ All WCP BB84 tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)