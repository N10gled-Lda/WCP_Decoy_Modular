import numpy as np
import math
from typing import Dict, Tuple, List
from enum import Enum
import logging
from dataclasses import dataclass

from BB84.bb84_protocol.wcp_pulse import PulseType, WCPPulse

# Configure logging
logger = logging.getLogger(__name__)

@dataclass(slots=True)
class DecoyStats:
    """Aggregate counts *after* sifting for both Z and X bases."""

    # Z-basis sent counts
    N_mu_Z: int = 0   # signal
    N_nu_Z: int = 0   # decoy
    N_vac_Z: int = 0  # vacuum

    # Z-basis detection counts
    n_mu_Z: int = 0
    n_nu_Z: int = 0
    n_vac_Z: int = 0

    # Z-basis error counts
    e_mu_Z: int = 0
    e_nu_Z: int = 0 # Error for decoy in Z-basis

    # X-basis sent counts
    N_mu_X: int = 0   # signal
    N_nu_X: int = 0   # decoy
    N_vac_X: int = 0  # vacuum

    # X-basis detection counts
    n_mu_X: int = 0
    n_nu_X: int = 0
    n_vac_X: int = 0

    # X-basis error counts
    e_mu_X: int = 0
    e_nu_X: int = 0 # Error for decoy in X-basis

    # helper – increment in one line
    def add(self, tag: str, detected: bool, error: bool, basis: int): # basis: 0 for Z, 1 for X
        if basis == 0:  # Z-basis
            if tag == "S":
                self.N_mu_Z += 1
                if detected:
                    self.n_mu_Z += 1
                    if error:
                        self.e_mu_Z += 1
            elif tag == "D":
                self.N_nu_Z += 1
                if detected:
                    self.n_nu_Z += 1
                    if error:
                        self.e_nu_Z += 1
            else:  # vacuum
                self.N_vac_Z += 1
                if detected:
                    self.n_vac_Z += 1
        elif basis == 1:  # X-basis
            if tag == "S":
                self.N_mu_X += 1
                if detected:
                    self.n_mu_X += 1
                    if error:
                        self.e_mu_X += 1
            elif tag == "D":
                self.N_nu_X += 1
                if detected:
                    self.n_nu_X += 1
                    if error:
                        self.e_nu_X += 1
            else:  # vacuum
                self.N_vac_X += 1
                if detected:
                    self.n_vac_X += 1


class WCPParameterEstimator:
    """
    Parameter estimation for WCP BB84 protocol including decoy-state analysis
    and PNS attack detection.
    """

    def __init__(self, signal_intensity: float=0.5, decoy_intensity: float=0.1, vacuum_intensity: float=0.0,
                 detector_efficiency: float=0.1, dark_count_rate: float=1e-6, use_simplified_stats: bool=False):
        """
        Initialize parameter estimator with pulse parameters.
        
        :param signal_intensity: Mean photon number for signal pulses (μs)
        :param decoy_intensity: Mean photon number for decoy pulses (μd)
        :param vacuum_intensity: Mean photon number for vacuum pulses (μv)
        :param detector_efficiency: Detector efficiency
        :param dark_count_rate: Dark count rate
        :param use_simplified_stats: Use DecoyStats instead of full statistics
        """
        self.mu_s = signal_intensity  # Signal intensity
        self.mu_d = decoy_intensity   # Decoy intensity
        self.mu_v = vacuum_intensity  # Vacuum intensity (should be 0)
        self.detector_efficiency = detector_efficiency
        self.dark_count_rate = dark_count_rate
        self.use_simplified_stats = use_simplified_stats

        # Save values that are calculated
        self.Y_1 = None  # Single photon yield lower bound
        self.e_1 = None  # Single photon error rate upper bound
        self.attack_detected = False  # Flag for PNS attack detection
        self.attack_message = ""  # Message for PNS attack detection
        self.results_dict = {} # Results dictionary for gains and QBERs
        self.skr_dict = {}  # Secret key rate dictionary
        
        if use_simplified_stats:
            self.decoy_stats = DecoyStats()
        else:
            # Storage for measurement statistics
            self.stats = {
                PulseType.SIGNAL: {'sent': 0, 'detected': 0, 'errors': 0, 'basis_Z': {'sent': 0, 'detected': 0, 'errors': 0}, 'basis_X': {'sent': 0, 'detected': 0, 'errors': 0}},
                PulseType.DECOY: {'sent': 0, 'detected': 0, 'errors': 0, 'basis_Z': {'sent': 0, 'detected': 0, 'errors': 0}, 'basis_X': {'sent': 0, 'detected': 0, 'errors': 0}},
                PulseType.VACUUM: {'sent': 0, 'detected': 0, 'errors': 0, 'basis_Z': {'sent': 0, 'detected': 0, 'errors': 0}, 'basis_X': {'sent': 0, 'detected': 0, 'errors': 0}}
            }

    def add_measurement_data(self, pulse_type: str, basis: int, sent: bool, detected: bool, error: bool):
        """
        Add measurement data for parameter estimation.
        
        :param pulse_type: Type of pulse ('signal', 'decoy', 'vacuum') or tag ('S', 'D', 'V')
        :param basis: Measurement basis (0 for Z, 1 for X)
        :param sent: Whether pulse was sent
        :param detected: Whether pulse was detected
        :param error: Whether measurement resulted in error
        """
        if self.use_simplified_stats:
            # Convert pulse_type to tag format if needed
            tag_map = {'signal': 'S', 'decoy': 'D', 'vacuum': 'V'}
            tag = tag_map.get(pulse_type.name if isinstance(pulse_type, Enum) else pulse_type,
                              pulse_type.name if isinstance(pulse_type, Enum) else pulse_type)  # Handle Enum or string

            # Only process if sent
            if sent:
                self.decoy_stats.add(tag, detected, error, basis)  # Pass the basis
        else:
            # Original implementation
            pulse_enum_type = pulse_type if isinstance(pulse_type, PulseType) else PulseType(pulse_type) # Ensure PulseType
            basis_str = 'basis_Z' if basis == 0 else 'basis_X'
            basis_str2 = 'Z' if basis == 0 else 'X'
            
            if sent:
                self.stats[pulse_enum_type]['sent'] += 1
                self.stats[pulse_enum_type][basis_str]['sent'] += 1
                
            if detected:
                self.stats[pulse_enum_type]['detected'] += 1
                self.stats[pulse_enum_type][basis_str]['detected'] += 1

            if error and detected: # Error only counts if detected
                self.stats[pulse_enum_type]['errors'] += 1
                self.stats[pulse_enum_type][basis_str]['errors'] += 1

    def add_pulse(self, pulse: WCPPulse, sent: bool = True, detected: bool = True, error: bool = False):
        """
        Add a WCPPulse object to the parameter estimator.
        
        :param pulse: WCPPulse object containing pulse data
        """
        if not isinstance(pulse, WCPPulse):
            raise TypeError("Expected WCPPulse object")
        
        # Add measurement data from the pulse
        self.add_measurement_data(pulse.pulse_type, pulse.base, sent, detected, error)

    def calculate_gains_and_qber(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate overall gain (Q) and QBER (E) for each pulse type.
        
        :return results (Dict[str, Dict[str, float]]): Dictionary containing gains and QBERs
        """
        if self.use_simplified_stats:
            results = {}
            
            # Signal
            # Overall gain Q_μ = n_μ / N_μ
            Q_s = self.decoy_stats.n_mu_Z / self.decoy_stats.N_mu_Z if self.decoy_stats.N_mu_Z > 0 else 0.0
            # Overall QBER E_μ = e_μ / n_μ
            E_s = self.decoy_stats.e_mu_Z / self.decoy_stats.n_mu_Z if self.decoy_stats.n_mu_Z > 0 else 0.0
            
            results['signal'] = {
                'gain': Q_s, 'qber': E_s,
                'sent': self.decoy_stats.N_mu_Z,
                'detected': self.decoy_stats.n_mu_Z,
                'errors': self.decoy_stats.e_mu_Z
            }
            
            # Decoy
            Q_d = self.decoy_stats.n_nu_Z / self.decoy_stats.N_nu_Z if self.decoy_stats.N_nu_Z > 0 else 0.0
            E_d = self.decoy_stats.e_nu_Z / self.decoy_stats.n_nu_Z if self.decoy_stats.n_nu_Z > 0 else 0.0
            
            results['decoy'] = {
                'gain': Q_d, 'qber': E_d,
                'sent': self.decoy_stats.N_nu_Z,
                'detected': self.decoy_stats.n_nu_Z,
                'errors': self.decoy_stats.e_nu_Z
            }
            
            # Vacuum
            Q_v = self.decoy_stats.n_vac_Z / self.decoy_stats.N_vac_Z if self.decoy_stats.N_vac_Z > 0 else 0.0
            
            results['vacuum'] = {
                'gain': Q_v, 'qber': 0.0,  # No error tracking for vacuum
                'sent': self.decoy_stats.N_vac_Z,
                'detected': self.decoy_stats.n_vac_Z,
                'errors': 0
            }
            
            self.results_dict = results  # Store results in instance variable
            return results
        else:
            # Original implementation
            results = {}
            
            for pulse_type in ['signal', 'decoy', 'vacuum']:
                stats = self.stats[pulse_type]
                
                # Overall gain Q_μ = n_μ / N_μ
                if stats['sent'] > 0:
                    gain = stats['detected'] / stats['sent']
                else:
                    gain = 0.0
                
                # Overall QBER E_μ = e_μ / n_μ
                if stats['detected'] > 0:
                    qber = stats['errors'] / stats['detected']
                else:
                    qber = 0.0
                
                results[pulse_type] = {
                    'gain': gain,
                    'qber': qber,
                    'sent': stats['sent'],
                    'detected': stats['detected'],
                    'errors': stats['errors']
                }
                
                # Basis-specific calculations
                for basis in ['basis_Z', 'basis_X']:
                    basis_stats = stats[basis]
                    if basis_stats['sent'] > 0:
                        basis_gain = basis_stats['detected'] / basis_stats['sent']
                    else:
                        basis_gain = 0.0
                    
                    if basis_stats['detected'] > 0:
                        basis_qber = basis_stats['errors'] / basis_stats['detected']
                    else:
                        basis_qber = 0.0
                    
                    results[pulse_type][basis] = {
                        'gain': basis_gain,
                        'qber': basis_qber,
                        'sent': basis_stats['sent'],
                        'detected': basis_stats['detected'],
                        'errors': basis_stats['errors']
                    }
            
            self.results_dict = results  # Store results in instance variable
            return results
    
    def estimate_single_photon_parameters(self) -> Tuple[float, float]:
        """
        Estimate single-photon yield (Y1) and error rate (e1) using decoy-state method.
        
        Uses the formula:
        Y_1^{L,v,0} ≥ (μ/(μν - ν²)) * (Q_v*e^v - Q_μ*e^μ*(ν²/μ²) - (μ² - ν²)/μ² * Y_0)
        e_1 ≤ (E_v * Q_v * e^v - e_0 * Y_0) / (Y_1^L * v)
        Where μ = signal intensity, v = decoy intensity
        
        :return Y1, e1 (Tuple[float, float]): Tuple of (Y1_lower_bound, e1_upper_bound)
        """
        gains_qber = self.calculate_gains_and_qber()
        
        Q_s = gains_qber['signal']['gain']  # Signal gain
        Q_d = gains_qber['decoy']['gain']   # Decoy gain
        Q_v = gains_qber['vacuum']['gain']  # Vacuum gain (background)
        
        E_s = gains_qber['signal']['qber']  # Signal QBER
        E_d = gains_qber['decoy']['qber']   # Decoy QBER
        E_v = gains_qber['vacuum']['qber']  # Vacuum QBER
        
        # Background yield Y0 (estimated from vacuum pulses)
        Y_0 = Q_v
        
        # Single photon yield lower bound using decoy-state formula
        # Y_1^{L, v, 0} ≥ (μ/(μν - ν²)) * (Q_v * e^v - Q_μ * e^μ * (ν²/μ²) - ((μ² - ν²)/μ²) * Y_0)
        mu = self.mu_s  # Signal intensity
        nu = self.mu_d  # Decoy intensity
        
        if mu * nu - nu**2 <= 0:
            logger.warning("Invalid intensity relationship for decoy-state estimation")
            return 0.0, 1.0
        
        try:
            coefficient = mu / (mu * nu - nu**2)
            
            term1 = Q_d * math.exp(nu)
            term2 = Q_s * math.exp(mu) * (nu**2 / mu**2)
            term3 = ((mu**2 - nu**2) / mu**2) * Y_0
            
            Y_1_lower = coefficient * (term1 - term2 - term3)
            Y_1_lower = max(Y_1_lower, 0.0)  # Ensure non-negative
            
        except (ZeroDivisionError, OverflowError):
            logger.warning("Mathematical error in Y1 calculation")
            Y_1_lower = 0.0
        
        # Single photon error rate upper bound
        # e_1 ≤ (E_v * Q_v * e^v - e_0 * Y_0) / (Y_1^L * v)
        if Y_1_lower > 0 and nu > 0:
            try:
                e_0 = E_v  # Background error rate
                numerator = E_d * Q_d * math.exp(nu) - e_0 * Y_0
                e_1_upper = numerator / (Y_1_lower * nu)
                e_1_upper = min(max(e_1_upper, 0.0), 1.0)  # Bound between 0 and 1
            except (ZeroDivisionError, OverflowError):
                logger.warning("Mathematical error in e1 calculation")
                e_1_upper = 1.0
        else:
            e_1_upper = 1.0
        
        logger.info(f"Single photon yield (lower bound): {Y_1_lower:.6f}")
        logger.info(f"Single photon error rate (upper bound): {e_1_upper:.6f}")

        self.Y_1 = Y_1_lower  # Store in instance variable
        self.e_1 = e_1_upper  # Store in instance variable
        return Y_1_lower, e_1_upper
    
    def detect_pns_attack(self, margin=0.1) -> Tuple[bool, str]:
        """
        Detect potential PNS (Photon Number Splitting) attacks by analyzing
        gain ratios between different pulse intensities.

        :param margin: Safety margin for gain ratio comparison
        :return attack_detected, description (Tuple[bool, str]): Tuple indicating 
        if an attack was detected and a description
        """
        gains_qber = self.calculate_gains_and_qber()

        Q_s = gains_qber['signal']['gain']  # Signal gain
        Q_d = gains_qber['decoy']['gain']   # Decoy gain
        Q_v = gains_qber['vacuum']['gain']  # Vacuum gain
        
        warnings = []
        attack_detected = False
        
        # Check 1: Decoy-to-signal gain ratio
        if Q_s > 0:
            gain_ratio = Q_d / Q_s
            expected_ratio = self.mu_d / self.mu_s  # Approximate expected ratio
            
            if gain_ratio > expected_ratio * (1 + margin):
                warnings.append(f"Decoy-signal gain ratio ({gain_ratio:.3f}) exceeds threshold; potential PNS attack")
                attack_detected = True
        
        # Check 2: Vacuum gain should be very low
        vacuum_threshold = 0.01  # 1% threshold
        if Q_v > vacuum_threshold:
            warnings.append(f"Vacuum gain ({Q_v:.4f}) suspiciously high; potential tampering")
            attack_detected = True
        
        # Check 3: QBER consistency check
        E_s = gains_qber['signal']['qber']
        E_d = gains_qber['decoy']['qber']
        
        if abs(E_s - E_d) > 0.1:  # 10% difference threshold
            warnings.append(f"Significant QBER difference between signal ({E_s:.3f}) and decoy ({E_d:.3f})")
            attack_detected = True
        
        # Check 4: Basis-dependent gain anomalies (only for full stats mode)
        if not self.use_simplified_stats:
            for pulse_type in ['signal', 'decoy']:
                if 'basis_Z' in gains_qber[pulse_type] and 'basis_X' in gains_qber[pulse_type]:
                    gain_Z = gains_qber[pulse_type]['basis_Z']['gain']
                    gain_X = gains_qber[pulse_type]['basis_X']['gain']
                    
                    if gain_Z > 0 and gain_X > 0:
                        basis_ratio = abs(gain_Z - gain_X) / max(gain_Z, gain_X)
                        if basis_ratio > 0.2:  # 20% difference threshold
                            warnings.append(f"Significant basis-dependent gain difference for {pulse_type} pulses")
                            attack_detected = True
        
        warning_message = "; ".join(warnings) if warnings else "No anomalies detected"
        
        if attack_detected:
            logger.warning(f"Potential security threat detected: {warning_message}")
        else:
            logger.info("Security check passed: No anomalies detected")
        
        self.attack_detected = attack_detected  # Store in instance variable
        self.attack_message = warning_message  # Store in instance variable

        return attack_detected, warning_message

    def calculate_secret_key_rate(self, signal_fraction: float = 0.7, error_corr_efficiency: float = 1.2, Y_1: float = None, e_1: float = None) -> Dict[str, float]:
        """
        Calculate the secret key rate using WCP BB84 parameters.
        
        Uses the formula:
        R ≈ Qs * [Y_1 * μs * e^(-μs) * (1 - h(e_1)) - f * Es * h(Es)]

        :param signal_fraction: Fraction of signal pulses (default 0.7)
        :param error_corr_efficiency: Error correction efficiency factor (default 1.2)
        :param Y_1: Single photon yield (optional, will be estimated if not provided)
        :param e_1: Single photon error rate (optional, will be estimated if not provided)
        :return SKR (Dict[str, float]): Dictionary with key rate information
        """
        if Y_1 is None or e_1 is None:
            Y_1, e_1 = self.estimate_single_photon_parameters()

        gains_qber = self.calculate_gains_and_qber()
        # Shannon entropy for error correction
        if e_1 > 0 and e_1 < 1:
            h_e1 = -e_1 * math.log2(e_1) - (1 - e_1) * math.log2(1 - e_1)
        else:
            h_e1 = 0
        
        # Secret key rate per signal pulse (simplified formula)
        # R ≈ Qs * [Y_1 * μs * e^(-μs) * (1 - h(e_1)) - f * Es * h(Es)]
        Q_s = gains_qber['signal']['gain']
        E_s = gains_qber['signal']['qber']
        
        if E_s > 0 and E_s < 1:
            h_Es = -E_s * math.log2(E_s) - (1 - E_s) * math.log2(1 - E_s)
        else:
            h_Es = 0
        
        f = error_corr_efficiency  # Error correction efficiency factor
        
        try:
            single_photon_contribution = Y_1 * self.mu_s * math.exp(-self.mu_s) * (1 - h_e1)
            error_correction_cost = f * E_s * h_Es
            
            rate_per_signal = Q_s * (single_photon_contribution - error_correction_cost)
            rate_per_signal = max(rate_per_signal, 0.0)  # Ensure non-negative
            
        except (ValueError, OverflowError):
            rate_per_signal = 0.0
        
        # Estimate total key rate
        total_rate = rate_per_signal * signal_fraction

        self.skr_dict = {
            'rate_per_signal_pulse': rate_per_signal,
            'total_rate': total_rate,
            'single_photon_yield': Y_1,
            'single_photon_error_rate': e_1,
            'signal_gain': Q_s,
            'signal_qber': E_s,
            'error_correction_cost': error_correction_cost
        }
        return self.skr_dict

    def get_summary_report(self) -> str:
        """
        Generate a comprehensive summary report of the WCP parameter estimation.
        
        :return report (str): Formatted summary report string
        """
        if self.results_dict is None:
            self.results_dict = self.calculate_gains_and_qber()
        if self.skr_dict is None:
            self.skr_dict = self.calculate_secret_key_rate()
        if self.Y_1 is None or self.e_1 is None:
            self.Y_1, self.e_1 = self.estimate_single_photon_parameters()
        if self.attack_detected is False or self.attack_message == "":
            self.attack_detected, self.attack_message = self.detect_pns_attack()
        
        gains_qber = self.results_dict
        Y_1 = self.Y_1
        e_1 = self.e_1
        attack_detected = self.attack_detected
        attack_message = self.attack_message
        skr_info = self.skr_dict

        report = []
        report.append("=== WCP BB84 Parameter Estimation Report ===")
        report.append("")
        
        # Pulse statistics
        report.append("Pulse Statistics:")
        for pulse_type in ['signal', 'decoy', 'vacuum']:
            stats = gains_qber[pulse_type]
            report.append(f"  {pulse_type.capitalize()}:")
            report.append(f"    Sent: {stats['sent']}, Detected: {stats['detected']}, Errors: {stats['errors']}")
            report.append(f"    Gain (Q): {stats['gain']:.6f}, QBER (E): {stats['qber']:.6f}")
        
        report.append("")
        
        # Single photon parameters
        report.append("Single Photon Parameters:")
        report.append(f"  Yield (Y₁) lower bound: {Y_1:.6f}")
        report.append(f"  Error rate (e₁) upper bound: {e_1:.6f}")
        report.append("")
        
        # Security analysis
        report.append("Security Analysis:")
        report.append(f"  PNS Attack Detection: {'⚠️  THREAT DETECTED' if attack_detected else '✅ SECURE'}")
        report.append(f"  Details: {attack_message}")
        report.append("")
        
        # Key rate estimation
        report.append("Key Rate Estimation:")
        report.append(f"  Rate per signal pulse: {skr_info['rate_per_signal_pulse']:.6f}")
        report.append(f"  Total estimated rate: {skr_info['total_rate']:.6f}")
        report.append(f"  Error correction cost: {skr_info['error_correction_cost']:.6f}")
        
        return "\n".join(report)
    
    def reset_statistics(self):
        """Reset all measurement statistics"""
        if self.use_simplified_stats:
            self.decoy_stats = DecoyStats() # Re-initialize the extended DecoyStats
        else:
            for pulse_type_enum in PulseType: # Iterate through Enum members
                self.stats[pulse_type_enum] = {
                    'sent': 0, 'detected': 0, 'errors': 0,
                    'basis_Z': {'sent': 0, 'detected': 0, 'errors': 0},
                    'basis_X': {'sent': 0, 'detected': 0, 'errors': 0}
                }
