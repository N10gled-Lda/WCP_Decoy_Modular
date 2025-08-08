"""Hardware configuration examples for the complete QKD system."""

# Hardware configuration for Alice's components
ALICE_CONFIG = {
    "laser": {
        "type": "hardware",  # or "simulator"
        "driver": "digilent",
        "device_index": -1,  # -1 for first available device
        "trigger_channel": 0,  # Analog output channel 0
        "pulse_parameters": {
            "amplitude": 3.3,  # Trigger voltage [V]
            "width": 1e-6,     # Pulse width [s]
            "frequency": 1000.0  # Default frequency [Hz]
        },
        "safety": {
            "max_frequency": 10000.0,  # Maximum allowed frequency [Hz]
            "max_amplitude": 5.0,      # Maximum voltage [V]
            "timeout": 10.0            # Communication timeout [s]
        }
    },
    
    "polarization": {
        "type": "hardware",  # or "simulator"
        "driver": "stm32",
        "port": "COM3",      # Serial port
        "baudrate": 115200,
        "timeout": 5.0,
        "queue_size": 1000,  # Input/output queue sizes
        "thread_interval": 0.001,  # Processing interval [s]
        "states": {
            "H": {"channel_1": 0.0, "channel_2": 0.0},    # Horizontal
            "V": {"channel_1": 90.0, "channel_2": 0.0},   # Vertical
            "D": {"channel_1": 45.0, "channel_2": 0.0},   # Diagonal
            "A": {"channel_1": 135.0, "channel_2": 0.0}   # Anti-diagonal
        }
    },
    
    "qrng": {
        "type": "hardware",  # or "simulator"
        "driver": "quantis",
        "device_id": 0,
        "buffer_size": 1024,
        "timeout": 1.0
    },
    
    "voa": {
        "type": "hardware",  # or "simulator"
        "driver": "thorlabs",
        "model": "V1000A",
        "serial_number": None,  # Auto-detect
        "attenuation_range": [0.0, 60.0],  # dB
        "queue_size": 1000,
        "thread_interval": 0.001
    }
}

# Hardware configuration for Bob's components
BOB_CONFIG = {
    "detectors": {
        "type": "hardware",  # or "simulator"
        "driver": "id_quantique",
        "model": "ID230",
        "count": 4,  # Number of detector channels
        "dead_time": 50e-9,  # Dead time [s]
        "dark_count_rate": 100,  # Dark counts per second
        "efficiency": 0.25  # Detection efficiency
    },
    
    "time_tagger": {
        "type": "hardware",  # or "simulator"
        "driver": "swabian",
        "model": "Time Tagger Ultra",
        "channels": [1, 2, 3, 4, 5],  # Channel mapping
        "resolution": 1e-12,  # Time resolution [s]
        "buffer_size": 1000000,
        "trigger_channel": 5  # Sync trigger from Alice
    },
    
    "polarization_analysis": {
        "type": "hardware",  # or "simulator"
        "driver": "motorized_waveplate",
        "port": "COM4",
        "baudrate": 9600,
        "angles": {
            "HV": {"hwp": 0.0, "qwp": 0.0},     # H/V basis
            "DA": {"hwp": 22.5, "qwp": 0.0},    # D/A basis
            "RL": {"hwp": 45.0, "qwp": 45.0}    # R/L basis
        }
    }
}

# Communication and synchronization
SYNC_CONFIG = {
    "clock_source": {
        "type": "external",  # "internal" or "external"
        "frequency": 10e6,   # 10 MHz reference
        "driver": "digilent_shared"  # Share clock between devices
    },
    
    "trigger_distribution": {
        "master": "alice_laser",  # Master trigger source
        "slaves": ["bob_time_tagger", "alice_polarization"],
        "cable_delay_compensation": {
            "bob_time_tagger": 15e-9,      # Cable delay [s]
            "alice_polarization": 5e-9
        }
    },
    
    "communication": {
        "protocol": "tcp",
        "alice_ip": "192.168.1.10",
        "bob_ip": "192.168.1.20",
        "port": 8080,
        "timeout": 30.0
    }
}

# QKD Protocol parameters
PROTOCOL_CONFIG = {
    "type": "bb84_decoy",
    "basis_choices": ["HV", "DA"],  # Measurement bases
    "intensity_levels": {
        "signal": 0.5,     # μ (mean photon number)
        "decoy": 0.1,      # ν (decoy state)
        "vacuum": 0.0      # Vacuum state
    },
    "probabilities": {
        "signal": 0.6,     # P(signal)
        "decoy": 0.3,      # P(decoy)
        "vacuum": 0.1      # P(vacuum)
    },
    "block_size": 10000,   # Pulses per block
    "key_rate_target": 1000,  # Target key rate [bits/s]
    "error_rate_threshold": 0.11  # QBER threshold
}

# Measurement and characterization
CHARACTERIZATION_CONFIG = {
    "laser_characterization": {
        "power_meter": {
            "driver": "thorlabs",
            "model": "PM100D",
            "sensor": "S121C"
        },
        "spectrum_analyzer": {
            "driver": "bristol",
            "model": "771",
            "wavelength_range": [1540, 1560]  # nm
        }
    },
    
    "channel_characterization": {
        "loss_measurement": True,
        "polarization_drift": True,
        "timing_jitter": True,
        "background_measurement": True
    },
    
    "detector_characterization": {
        "dark_count_measurement": True,
        "efficiency_calibration": True,
        "afterpulse_measurement": True,
        "timing_resolution": True
    }
}

# System monitoring and logging
MONITORING_CONFIG = {
    "logging": {
        "level": "INFO",
        "file": "qkd_system.log",
        "max_size": "100MB",
        "backup_count": 5
    },
    
    "metrics": {
        "collection_interval": 1.0,  # seconds
        "storage": "influxdb",
        "retention_policy": "30d"
    },
    
    "alerts": {
        "email_notifications": True,
        "error_threshold": 5,  # Errors per minute
        "performance_threshold": 0.8  # Relative to target
    }
}

# Complete system configuration
SYSTEM_CONFIG = {
    "alice": ALICE_CONFIG,
    "bob": BOB_CONFIG,
    "synchronization": SYNC_CONFIG,
    "protocol": PROTOCOL_CONFIG,
    "characterization": CHARACTERIZATION_CONFIG,
    "monitoring": MONITORING_CONFIG
}

# Example usage functions
def get_hardware_config(component_name):
    """Get hardware configuration for a specific component."""
    parts = component_name.split('.')
    config = SYSTEM_CONFIG
    
    for part in parts:
        if part in config:
            config = config[part]
        else:
            raise KeyError(f"Configuration not found for {component_name}")
    
    return config


def validate_config():
    """Validate the configuration for consistency."""
    errors = []
    
    # Check that all referenced drivers exist
    required_drivers = set()
    
    # Alice components
    if ALICE_CONFIG["laser"]["driver"] == "digilent":
        required_drivers.add("digilent_waveforms")
    
    if ALICE_CONFIG["polarization"]["driver"] == "stm32":
        required_drivers.add("stm32_serial")
    
    # Add more validation as needed
    
    return len(errors) == 0, errors


def create_device_mapping():
    """Create a mapping of logical names to physical devices."""
    return {
        "alice_laser": {
            "type": "digilent_analog_out",
            "device_index": ALICE_CONFIG["laser"]["device_index"],
            "channel": ALICE_CONFIG["laser"]["trigger_channel"]
        },
        "alice_polarization": {
            "type": "stm32_serial",
            "port": ALICE_CONFIG["polarization"]["port"],
            "baudrate": ALICE_CONFIG["polarization"]["baudrate"]
        },
        "bob_detectors": {
            "type": "id_quantique",
            "model": BOB_CONFIG["detectors"]["model"]
        },
        "bob_time_tagger": {
            "type": "swabian_instruments",
            "model": BOB_CONFIG["time_tagger"]["model"]
        }
    }


if __name__ == "__main__":
    """Example configuration validation."""
    print("QKD System Hardware Configuration")
    print("=================================")
    
    # Validate configuration
    is_valid, errors = validate_config()
    if is_valid:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Show device mapping
    print("\nDevice Mapping:")
    device_map = create_device_mapping()
    for name, config in device_map.items():
        print(f"  {name}: {config['type']}")
    
    # Example component access
    print(f"\nLaser configuration: {get_hardware_config('alice.laser')}")
