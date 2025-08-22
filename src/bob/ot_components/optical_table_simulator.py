"""Optical Table Simulator - Physics-Based BB84 Implementation."""
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import time

from ...utils.data_structures import Pulse, Basis, Bit


class MeasurementOutcome(Enum):
    """Possible measurement outcomes."""
    NO_DETECTION = "no_detection"
    H_DETECTION = "h_detection"      # Horizontal polarization
    V_DETECTION = "v_detection"      # Vertical polarization
    D_DETECTION = "d_detection"      # Diagonal polarization  
    A_DETECTION = "a_detection"      # Anti-diagonal polarization


@dataclass
class OpticalTableConfig:
    """Configuration for the optical table."""
    # Beam splitter parameters
    beam_splitter_ratio: float = 0.5  # 50:50 beam splitter
    beam_splitter_loss_db: float = 0.1
    
    # Polarization optics
    polarizer_extinction_ratio_db: float = 30.0
    waveplate_accuracy_deg: float = 0.1
    
    # Alignment and stability
    alignment_drift_deg_per_hour: float = 0.05
    thermal_stability_deg_per_celsius: float = 0.01
    
    # Measurement basis control
    basis_switching_time_ms: float = 10.0
    basis_switching_accuracy: float = 0.999
    
    # System losses
    coupling_efficiency: float = 0.95
    total_optical_loss_db: float = 2.0


@dataclass
class OpticalTableStatistics:
    """Statistics for optical table performance."""
    total_measurements: int = 0
    measurements_per_basis: Dict[str, int] = field(default_factory=lambda: {"Z": 0, "X": 0})
    detection_counts: Dict[str, int] = field(default_factory=lambda: {
        "H": 0, "V": 0, "D": 0, "A": 0, "none": 0
    })
    basis_switching_errors: int = 0
    alignment_corrections: int = 0
    thermal_drifts: List[float] = field(default_factory=list)


class OpticalTableSimulator:
    """
    Comprehensive optical table simulator for BB84 polarization measurements.
    
    Simulates:
    - Polarization measurement in Z and X bases
    - Beam splitter operations
    - Polarizer extinction ratios
    - Optical losses and coupling efficiency
    - Basis switching dynamics
    - Alignment drift and thermal effects
    """
    
    def __init__(self, config: OpticalTableConfig):
        """
        Initialize the optical table simulator.

        Args:
            config: Optical table configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.stats = OpticalTableStatistics()
        
        # Current state
        self.current_basis = Basis.Z
        self.last_basis_switch_time = time.time()
        self.current_alignment_error_deg = 0.0
        self.current_temperature_c = 20.0  # Room temperature
        
        # Optical elements state
        self.beam_splitter_transmission = np.sqrt(self.config.beam_splitter_ratio)
        self.beam_splitter_reflection = np.sqrt(1 - self.config.beam_splitter_ratio)
        
        # Calculate extinction coefficient from dB ratio
        self.extinction_ratio = 10**(self.config.polarizer_extinction_ratio_db / 10)
        
        self.logger.info("Optical table simulator initialized")
        self.logger.info(f"Beam splitter ratio: {self.config.beam_splitter_ratio}")
        self.logger.info(f"Polarizer extinction ratio: {self.config.polarizer_extinction_ratio_db} dB")

    def set_measurement_basis(self, basis: Basis) -> bool:
        """
        Set the measurement basis for polarization analysis.

        Args:
            basis: Measurement basis (Z for rectilinear, X for diagonal)

        Returns:
            bool: True if basis switch successful
        """
        current_time = time.time()
        switching_time = (current_time - self.last_basis_switch_time) * 1000  # ms
        
        # Check if enough time has passed for basis switching
        if switching_time < self.config.basis_switching_time_ms:
            self.logger.warning(f"Basis switching too fast: {switching_time:.1f} ms < {self.config.basis_switching_time_ms} ms")
            return False
        
        # Simulate basis switching accuracy
        switch_success = np.random.random() < self.config.basis_switching_accuracy
        
        if switch_success:
            self.current_basis = basis
            self.last_basis_switch_time = current_time
            self.logger.debug(f"Measurement basis set to {basis.value}")
            return True
        else:
            self.stats.basis_switching_errors += 1
            self.logger.warning(f"Basis switching failed for {basis.value}")
            return False

    def measure_polarization(self, pulse: Pulse, measurement_basis: Optional[Basis] = None) -> Tuple[MeasurementOutcome, float]:
        """
        Perform polarization measurement on incoming pulse.

        Args:
            pulse: Input pulse from the quantum channel
            measurement_basis: Basis to measure in (if None, use current basis)

        Returns:
            Tuple of (measurement outcome, detection probability)
        """
        if pulse.photons == 0:
            return MeasurementOutcome.NO_DETECTION, 0.0
        
        # Use specified basis or current basis
        basis = measurement_basis if measurement_basis is not None else self.current_basis
        
        # Update statistics
        self.stats.total_measurements += 1
        self.stats.measurements_per_basis[basis.value] += 1
        
        # Apply optical table effects
        pulse = self._apply_coupling_losses(pulse)
        pulse = self._apply_optical_losses(pulse)
        pulse = self._apply_alignment_errors(pulse)
        
        if pulse.photons == 0:
            self.stats.detection_counts["none"] += 1
            return MeasurementOutcome.NO_DETECTION, 0.0
        
        # Perform polarization analysis
        outcome, detection_prob = self._analyze_polarization(pulse, basis)
        
        # Update detection statistics
        outcome_key = outcome.value.split('_')[0] if outcome != MeasurementOutcome.NO_DETECTION else "none"
        self.stats.detection_counts[outcome_key] += 1
        
        self.logger.debug(f"Measurement: {pulse.polarization:.1f}° in {basis.value} basis → {outcome.value}")
        
        return outcome, detection_prob

    def _apply_coupling_losses(self, pulse: Pulse) -> Pulse:
        """Apply fiber-to-free-space coupling losses."""
        surviving_photons = np.random.binomial(pulse.photons, self.config.coupling_efficiency)
        pulse.photons = surviving_photons
        return pulse

    def _apply_optical_losses(self, pulse: Pulse) -> Pulse:
        """Apply total optical losses in the system."""
        loss_factor = 10**(-self.config.total_optical_loss_db / 10)
        surviving_photons = np.random.binomial(pulse.photons, loss_factor)
        pulse.photons = surviving_photons
        return pulse

    def _apply_alignment_errors(self, pulse: Pulse) -> Pulse:
        """Apply alignment drift effects."""
        # Simulate slow alignment drift
        current_time = time.time()
        time_hours = (current_time - getattr(self, '_start_time', current_time)) / 3600
        
        if not hasattr(self, '_start_time'):
            self._start_time = current_time
        
        # Accumulate alignment error
        drift_error = self.config.alignment_drift_deg_per_hour * time_hours
        thermal_error = self.config.thermal_stability_deg_per_celsius * (self.current_temperature_c - 20.0)
        
        self.current_alignment_error_deg = drift_error + thermal_error
        
        # Apply random component
        random_error = np.random.normal(0, 0.1)  # Small random fluctuations
        total_error = self.current_alignment_error_deg + random_error
        
        # Modify polarization angle slightly
        pulse.polarization += total_error
        
        return pulse

    def _analyze_polarization(self, pulse: Pulse, basis: Basis) -> Tuple[MeasurementOutcome, float]:
        """
        Analyze polarization in the specified basis.

        Args:
            pulse: Input pulse
            basis: Measurement basis

        Returns:
            Tuple of (measurement outcome, detection probability)
        """
        # Convert polarization angle to radians
        pol_angle_rad = np.radians(pulse.polarization)
        
        if basis == Basis.Z:
            # Rectilinear basis: measure H/V (0°/90°)
            return self._measure_rectilinear(pol_angle_rad, pulse.photons)
        elif basis == Basis.X:
            # Diagonal basis: measure D/A (45°/135°)
            return self._measure_diagonal(pol_angle_rad, pulse.photons)
        else:
            raise ValueError(f"Unknown measurement basis: {basis}")

    def _measure_rectilinear(self, pol_angle_rad: float, photons: int) -> Tuple[MeasurementOutcome, float]:
        """Measure in rectilinear basis (H/V)."""
        # Probability of measuring horizontal (H)
        prob_h = np.cos(pol_angle_rad)**2
        
        # Apply polarizer extinction ratio
        prob_h_corrected = self._apply_extinction_ratio(prob_h)
        prob_v_corrected = self._apply_extinction_ratio(1 - prob_h)
        
        # Normalize probabilities
        total_prob = prob_h_corrected + prob_v_corrected
        if total_prob > 0:
            prob_h_corrected /= total_prob
            prob_v_corrected /= total_prob
        
        # Apply beam splitter (photons go to H or V detector)
        if photons > 0:
            h_photons = np.random.binomial(photons, prob_h_corrected)
            v_photons = photons - h_photons
            
            if h_photons > 0 and v_photons == 0:
                return MeasurementOutcome.H_DETECTION, prob_h_corrected
            elif v_photons > 0 and h_photons == 0:
                return MeasurementOutcome.V_DETECTION, prob_v_corrected
            elif h_photons > 0 and v_photons > 0:
                # Multiple detection (rare but possible)
                # Choose the detector with more photons
                if h_photons >= v_photons:
                    return MeasurementOutcome.H_DETECTION, prob_h_corrected
                else:
                    return MeasurementOutcome.V_DETECTION, prob_v_corrected
        
        return MeasurementOutcome.NO_DETECTION, 0.0

    def _measure_diagonal(self, pol_angle_rad: float, photons: int) -> Tuple[MeasurementOutcome, float]:
        """Measure in diagonal basis (D/A)."""
        # Probability of measuring diagonal (D) - 45°
        prob_d = np.cos(pol_angle_rad - np.pi/4)**2
        
        # Apply polarizer extinction ratio
        prob_d_corrected = self._apply_extinction_ratio(prob_d)
        prob_a_corrected = self._apply_extinction_ratio(1 - prob_d)
        
        # Normalize probabilities
        total_prob = prob_d_corrected + prob_a_corrected
        if total_prob > 0:
            prob_d_corrected /= total_prob
            prob_a_corrected /= total_prob
        
        # Apply beam splitter (photons go to D or A detector)
        if photons > 0:
            d_photons = np.random.binomial(photons, prob_d_corrected)
            a_photons = photons - d_photons
            
            if d_photons > 0 and a_photons == 0:
                return MeasurementOutcome.D_DETECTION, prob_d_corrected
            elif a_photons > 0 and d_photons == 0:
                return MeasurementOutcome.A_DETECTION, prob_a_corrected
            elif d_photons > 0 and a_photons > 0:
                # Multiple detection
                if d_photons >= a_photons:
                    return MeasurementOutcome.D_DETECTION, prob_d_corrected
                else:
                    return MeasurementOutcome.A_DETECTION, prob_a_corrected
        
        return MeasurementOutcome.NO_DETECTION, 0.0

    def _apply_extinction_ratio(self, probability: float) -> float:
        """Apply polarizer extinction ratio to measurement probability."""
        # Finite extinction ratio means some leakage
        leakage = 1.0 / self.extinction_ratio
        
        # Correct probability accounting for leakage
        corrected_prob = probability * (1 - leakage) + (1 - probability) * leakage
        
        return max(0.0, min(1.0, corrected_prob))

    def outcome_to_bit(self, outcome: MeasurementOutcome, basis: Basis) -> Optional[Bit]:
        """
        Convert measurement outcome to bit value.

        Args:
            outcome: Measurement outcome
            basis: Measurement basis

        Returns:
            Bit value or None if no detection
        """
        if outcome == MeasurementOutcome.NO_DETECTION:
            return None
        
        if basis == Basis.Z:
            # Rectilinear basis: H = 0, V = 1
            if outcome == MeasurementOutcome.H_DETECTION:
                return Bit.ZERO
            elif outcome == MeasurementOutcome.V_DETECTION:
                return Bit.ONE
        elif basis == Basis.X:
            # Diagonal basis: D = 0, A = 1
            if outcome == MeasurementOutcome.D_DETECTION:
                return Bit.ZERO
            elif outcome == MeasurementOutcome.A_DETECTION:
                return Bit.ONE
        
        return None

    def set_temperature(self, temperature_c: float) -> None:
        """Set ambient temperature for thermal drift simulation."""
        self.current_temperature_c = temperature_c
        self.logger.debug(f"Temperature set to {temperature_c}°C")

    def perform_alignment_correction(self) -> float:
        """Perform alignment correction and return correction angle."""
        correction_angle = -self.current_alignment_error_deg
        self.current_alignment_error_deg = 0.0
        self.stats.alignment_corrections += 1
        
        self.logger.info(f"Alignment corrected by {correction_angle:.3f}°")
        return correction_angle

    def get_statistics(self) -> OpticalTableStatistics:
        """Get optical table performance statistics."""
        return self.stats

    def get_table_info(self) -> Dict[str, Any]:
        """Get comprehensive optical table information."""
        total_detections = sum(self.stats.detection_counts.values()) - self.stats.detection_counts["none"]
        detection_efficiency = total_detections / max(self.stats.total_measurements, 1)
        
        return {
            "config": {
                "beam_splitter_ratio": self.config.beam_splitter_ratio,
                "extinction_ratio_db": self.config.polarizer_extinction_ratio_db,
                "coupling_efficiency": self.config.coupling_efficiency,
                "total_optical_loss_db": self.config.total_optical_loss_db
            },
            "current_state": {
                "measurement_basis": self.current_basis.value,
                "alignment_error_deg": self.current_alignment_error_deg,
                "temperature_c": self.current_temperature_c
            },
            "performance": {
                "total_measurements": self.stats.total_measurements,
                "detection_efficiency": detection_efficiency,
                "basis_switching_errors": self.stats.basis_switching_errors,
                "detection_counts": self.stats.detection_counts.copy()
            }
        }

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.stats = OpticalTableStatistics()
        self.logger.info("Optical table statistics reset")

    def calibrate_table(self) -> Dict[str, float]:
        """Perform optical table calibration."""
        self.logger.info("Performing optical table calibration...")
        
        calibration_results = {}
        
        # Test polarization measurements with known states
        test_angles = [0, 45, 90, 135]  # H, D, V, A
        test_bases = [Basis.Z, Basis.X]
        
        for basis in test_bases:
            self.set_measurement_basis(basis)
            
            for angle in test_angles:
                test_pulse = Pulse(polarization=angle, photons=100)
                
                # Perform multiple measurements
                outcomes = []
                for _ in range(100):
                    outcome, prob = self.measure_polarization(test_pulse)
                    outcomes.append(outcome)
                
                # Analyze results
                detection_rate = sum(1 for o in outcomes if o != MeasurementOutcome.NO_DETECTION) / len(outcomes)
                calibration_results[f"{basis.value}_{angle}deg"] = detection_rate
        
        self.logger.info("Optical table calibration completed")
        return calibration_results
