"""
Hardware Configuration Examples for WCP Decoy-State QKD Simulator.

This file provides example configurations for different hardware setups
and integration scenarios. Each configuration includes detailed comments
about parameter selection and validation requirements.

WARNING: All hardware parameters in these examples are placeholders and
MUST be validated against your specific hardware specifications before use.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

# =============================================================================
# Example 1: Laboratory Setup with Cooled SPADs
# =============================================================================

LABORATORY_SETUP = {
    "simulation": {
        "pulses_total": 10000000,
        "num_threads": 8,
        "random_seed": 12345,
        "logging_level": "INFO",
        "use_hardware": False,  # Set to True for actual hardware
        "output_dir": "lab_simulation_output"
    },
    
    "clock": {
        "repetition_rate_Hz": 10000000,  # 10 MHz
        "duty_cycle": 0.05,
        "frame_length": 1048576,
        "timing_stability_ps": 1.0  # WARNING: Verify with clock specs
    },
    
    "laser": {
        "central_wavelength_nm": 1550.12,  # WARNING: Measure actual wavelength
        "linewidth_Hz": 100000,  # WARNING: Verify with laser specs
        "max_power_mW": 50.0,  # WARNING: Check laser safety limits
        "pulse_width_fwhm_ps": 50.0,  # WARNING: Measure with autocorrelator
        "timing_jitter_ps": 2.0,  # WARNING: Measure with timing analyzer
        "polarization_extinction_ratio_dB": 25.0,  # WARNING: Measure with analyzer
        "relative_intensity_noise_dB": -155.0,  # WARNING: Measure with RIN meter
        "wavelength_stability_pm": 5.0,  # WARNING: Monitor with wavemeter
        
        # Advanced physics model (WARNING: All parameters need validation)
        "enable_advanced_physics": True,
        "temperature_coefficient_nm_per_C": 0.1,
        "power_coefficient_nm_per_mW": 0.01,
        "aging_coefficient_per_hour": 1e-6
    },
    
    "decoy_scheme": {
        "intensities": {
            "signal": 0.8,      # WARNING: Optimize for your system
            "weak": 0.1,        # WARNING: Optimize for your system  
            "vacuum": 0.001     # WARNING: Measure actual vacuum level
        },
        "probabilities": {
            "signal": 0.7,
            "weak": 0.25,
            "vacuum": 0.05
        }
    },
    
    "voa": {
        # WARNING: All VOA parameters need validation with datasheet
        "insertion_loss_dB": 1.5,
        "temperature_coefficient_dB_per_C": 0.01,
        "wavelength_coefficient_dB_per_nm": 0.001,
        "polarization_dependent_loss_dB": 0.2,
        "switching_time_ms": 10.0,
        "attenuation_range_dB": 60.0,
        "attenuation_accuracy_dB": 0.1
    },
    
    "polarization": {
        # WARNING: Verify with polarization controller specs
        "extinction_ratio_dB": 30.0,
        "insertion_loss_dB": 0.5,
        "wavelength_dependence_dB_per_nm": 0.001,
        "temperature_stability_deg_per_C": 0.1,
        "voltage_range_V": 20.0,
        "response_time_ms": 1.0,
        
        # Hardware interface (if using stepper motor system)
        "use_hardware": False,  # Set to True for actual hardware
        "stepper_port": "COM3",  # WARNING: Check actual port
        "stepper_baudrate": 9600,
        "calibration_points": 36  # Every 10 degrees
    },
    
    "channel": {
        "type": "free_space",
        "length_km": 1.0,  # Short lab distance
        "atmospheric_attenuation_db_km": 0.1,  # WARNING: Measure conditions
        "turbulence_strength": 0.01,  # WARNING: Characterize setup
        "background_rate_Hz": 1000,  # WARNING: Measure background
        "polarization_drift_deg_rms": 1.0,  # WARNING: Monitor stability
        "beam_wander_mrad_rms": 0.1  # WARNING: Measure with position sensor
    },
    
    "detectors": {
        "type": "SPAD_array",
        "number": 4,
        "quantum_efficiency": 0.85,  # WARNING: Measure at operating wavelength
        "dark_count_rate_Hz": 50,  # WARNING: Measure at operating temperature
        "timing_jitter_ps": 25.0,  # WARNING: Measure with timing analyzer
        "dead_time_ns": 25.0,  # WARNING: Check datasheet
        "afterpulse_probability": 0.005,  # WARNING: Measure carefully
        "afterpulse_decay_ns": 500.0,  # WARNING: Characterize time constant
        
        # Operating conditions
        "operating_temperature_C": -40.0,  # WARNING: Use TEC controller
        "bias_voltage_V": 52.5,  # WARNING: Optimize for efficiency/dark counts
        "gate_width_ns": 2.0,  # WARNING: Optimize for timing
        
        # Advanced physics (WARNING: All need validation)
        "enable_advanced_physics": True,
        "efficiency_temp_coefficient_per_C": -0.002,
        "dark_count_temp_coefficient_per_C": 0.15,
        "saturation_photon_number": 15.0
    },
    
    "time_tagger": {
        "resolution_ps": 4.0,  # WARNING: Check instrument specs
        "channel_count": 8,
        "max_count_rate_MHz": 80,
        "time_window_ns": 1000,
        "trigger_threshold_mV": 50,  # WARNING: Optimize for detectors
        "input_impedance_ohm": 50,
        
        # Hardware interface
        "use_hardware": False,  # Set to True for actual hardware
        "device_serial": "1234567890",  # WARNING: Check actual serial
        "external_clock": True
    },
    
    "post_processing": {
        "bb84_sifting": {
            "basis_reconciliation_method": "public_announcement",
            "error_estimation_sample_size": 10000
        },
        "error_reconciliation": {
            "protocol": "cascade",
            "iterations": 4,
            "block_size_initial": 1000,
            "efficiency_target": 1.1
        },
        "privacy_amplification": {
            "hash_function": "universal",
            "security_parameter": 64,
            "finite_size_effects": True
        }
    }
}

# =============================================================================
# Example 2: Field Deployment Setup
# =============================================================================

FIELD_DEPLOYMENT_SETUP = {
    "simulation": {
        "pulses_total": 100000000,  # Longer run for statistics
        "num_threads": 4,  # Conservative for field computer
        "random_seed": 54321,
        "logging_level": "WARNING",  # Reduce log volume
        "use_hardware": True,  # Actually use hardware
        "output_dir": "/var/qkd/field_data"
    },
    
    "clock": {
        "repetition_rate_Hz": 1000000,  # Lower rate for stability
        "duty_cycle": 0.1,
        "frame_length": 2097152,  # Larger frames
        "timing_stability_ps": 10.0  # WARNING: Monitor with GPS
    },
    
    "laser": {
        "central_wavelength_nm": 1550.0,
        "linewidth_Hz": 1000000,  # Broader for DFB stability
        "max_power_mW": 10.0,  # Lower power for eye safety
        "pulse_width_fwhm_ps": 100.0,
        "timing_jitter_ps": 10.0,  # More conservative
        "polarization_extinction_ratio_dB": 20.0,
        "relative_intensity_noise_dB": -150.0,
        "wavelength_stability_pm": 50.0,  # Expect more drift
        
        # Environmental compensation
        "enable_temperature_compensation": True,
        "temperature_sensor_channel": 0,
        "power_stabilization": True
    },
    
    "channel": {
        "type": "free_space",
        "length_km": 10.0,  # Practical field distance
        "atmospheric_attenuation_db_km": 0.5,  # WARNING: Weather dependent
        "turbulence_strength": 0.1,  # WARNING: Atmospheric conditions
        "background_rate_Hz": 10000,  # WARNING: Sunlight/artificial lights
        "polarization_drift_deg_rms": 5.0,  # WARNING: Thermal effects
        "pointing_stability_mrad_rms": 1.0,  # WARNING: Mechanical stability
        
        # Environmental monitoring
        "monitor_weather": True,
        "visibility_threshold_km": 20.0,
        "wind_speed_threshold_mps": 15.0
    },
    
    "detectors": {
        "type": "InGaAs_APD",  # Telecom wavelength
        "number": 4,
        "quantum_efficiency": 0.7,  # WARNING: Typical for InGaAs
        "dark_count_rate_Hz": 1000,  # WARNING: Higher at warmer temps
        "timing_jitter_ps": 100.0,  # WARNING: Typical for APD
        "dead_time_ns": 100.0,  # WARNING: Check datasheet
        "afterpulse_probability": 0.02,  # WARNING: Higher for APD
        
        # Operating conditions (less stringent cooling)
        "operating_temperature_C": -10.0,  # WARNING: Achievable with TEC
        "bias_voltage_V": 95.0,  # WARNING: Below breakdown for InGaAs
        "gate_width_ns": 5.0,
        
        # Environmental protection
        "humidity_protection": True,
        "vibration_isolation": True
    },
    
    # Simplified hardware interfaces for field deployment
    "hardware_interfaces": {
        "laser_controller": {
            "interface": "serial",
            "port": "/dev/ttyUSB0",  # Linux field computer
            "baudrate": 115200,
            "timeout_s": 5.0
        },
        "detector_controller": {
            "interface": "ethernet",
            "ip_address": "192.168.1.100",
            "port": 5025,  # SCPI over TCP
            "timeout_s": 10.0
        }
    }
}

# =============================================================================
# Example 3: Characterization and Testing Setup
# =============================================================================

CHARACTERIZATION_SETUP = {
    "simulation": {
        "pulses_total": 1000000,  # Shorter for parameter sweeps
        "num_threads": 1,  # Single thread for reproducibility
        "random_seed": 11111,
        "logging_level": "DEBUG",
        "use_hardware": False,  # Simulation for characterization
        "output_dir": "characterization_output"
    },
    
    "parameter_sweep": {
        "enable": True,
        "parameters": {
            "laser.max_power_mW": [1, 2, 5, 10, 20, 50],
            "decoy_scheme.intensities.signal": [0.2, 0.5, 0.8, 1.0, 1.2],
            "channel.length_km": [1, 5, 10, 20, 50, 100],
            "detectors.quantum_efficiency": [0.1, 0.3, 0.5, 0.7, 0.9],
            "detectors.dark_count_rate_Hz": [1, 10, 100, 1000, 10000]
        },
        "combinations": "grid",  # or "random" for Monte Carlo
        "output_format": "hdf5"
    },
    
    "validation": {
        "compare_with_theory": True,
        "theoretical_models": ["ideal_bb84", "decoy_state_bounds"],
        "statistical_tests": ["chi_square", "kolmogorov_smirnov"],
        "confidence_level": 0.95
    },
    
    "optimization": {
        "enable": True,
        "objective": "secure_key_rate",  # or "qber", "detection_rate"
        "constraints": {
            "qber_max": 0.11,  # Below security threshold
            "detection_rate_min_Hz": 1000,
            "laser_power_max_mW": 100
        },
        "algorithm": "differential_evolution",
        "iterations": 100
    }
}

# =============================================================================
# Hardware Integration Helper Functions
# =============================================================================

def save_config_template(config: Dict[str, Any], filename: str) -> None:
    """Save configuration template to YAML file."""
    config_path = Path(filename)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
    
    print(f"Configuration template saved to {config_path}")
    print("\nWARNING: This template contains placeholder parameters that MUST be")
    print("validated against your specific hardware before use. Please review:")
    print("- All laser specifications and safety limits")
    print("- Detector characteristics and operating conditions") 
    print("- Channel parameters and environmental conditions")
    print("- Hardware interface settings and communication protocols")


def validate_hardware_config(config: Dict[str, Any]) -> Dict[str, list]:
    """
    Validate hardware configuration and return warnings.
    
    Returns dictionary of validation warnings by category.
    """
    warnings = {
        "laser": [],
        "detectors": [], 
        "channel": [],
        "safety": [],
        "performance": []
    }
    
    # Laser validation
    laser_config = config.get("laser", {})
    if laser_config.get("max_power_mW", 0) > 100:
        warnings["safety"].append("Laser power >100mW requires Class 4 safety measures")
    
    if laser_config.get("wavelength_stability_pm", 0) > 100:
        warnings["performance"].append("Wavelength stability >100pm may affect coherence")
    
    # Detector validation
    detector_config = config.get("detectors", {})
    if detector_config.get("dark_count_rate_Hz", 0) > 10000:
        warnings["detectors"].append("High dark count rate may limit performance")
    
    if detector_config.get("operating_temperature_C", 25) > 0:
        warnings["detectors"].append("Room temperature operation increases dark counts")
    
    # Channel validation
    channel_config = config.get("channel", {})
    if channel_config.get("length_km", 0) > 50:
        warnings["channel"].append("Long distances require atmospheric compensation")
    
    # Add more validation rules as needed...
    
    return warnings


def create_hardware_checklist(config: Dict[str, Any]) -> str:
    """Create hardware setup checklist from configuration."""
    checklist = """
# Hardware Setup Checklist for WCP Decoy-State QKD

## Safety Checklist
- [ ] Laser safety interlocks verified and tested
- [ ] Appropriate laser safety glasses available 
- [ ] Electrical safety: proper grounding and isolation
- [ ] Emergency shutdown procedures posted
- [ ] Personnel trained on laser safety protocols

## Laser System
- [ ] Wavelength calibrated with certified wavemeter
- [ ] Power output measured and verified within specs
- [ ] Pulse width characterized with autocorrelator
- [ ] Timing jitter measured with high-resolution oscilloscope
- [ ] Polarization extinction ratio verified
- [ ] Temperature controller operational and stable

## Detectors
- [ ] Quantum efficiency measured at operating wavelength
- [ ] Dark count rate characterized at operating temperature
- [ ] Timing jitter measured with precision timing analyzer
- [ ] Afterpulsing characterized with appropriate test setup
- [ ] Cooling system operational and stable
- [ ] Bias voltage optimized for efficiency vs. dark counts

## Communication Interfaces
- [ ] All serial/ethernet connections tested
- [ ] Communication protocols verified with test commands
- [ ] Timeout and error handling tested
- [ ] Hardware control software operational

## Environmental Controls
- [ ] Temperature monitoring and logging operational
- [ ] Vibration isolation installed and effective
- [ ] Humidity control (if required) operational
- [ ] Electromagnetic interference shielding verified

## Calibration and Characterization
- [ ] All instruments calibrated with traceable standards
- [ ] System characterization measurements completed
- [ ] Performance baselines established
- [ ] Calibration certificates available and current
"""
    
    return checklist


# =============================================================================
# Configuration Export Functions
# =============================================================================

if __name__ == "__main__":
    # Save all example configurations
    save_config_template(LABORATORY_SETUP, "configs/laboratory_setup.yaml")
    save_config_template(FIELD_DEPLOYMENT_SETUP, "configs/field_deployment.yaml") 
    save_config_template(CHARACTERIZATION_SETUP, "configs/characterization_setup.yaml")
    
    # Generate hardware checklist
    checklist = create_hardware_checklist(LABORATORY_SETUP)
    with open("docs/hardware_checklist.md", "w") as f:
        f.write(checklist)
    
    print("\nAll configuration templates and documentation generated.")
    print("Remember to validate all parameters against your specific hardware!")
