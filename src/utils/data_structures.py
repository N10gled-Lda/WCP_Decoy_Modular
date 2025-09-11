"""Data structures for the simulation."""
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator

# @dataclass
# class Pulse:
#     """Represents a light pulse."""
#     intensity: float
#     polarization: int
#     basis: str
#     timestamp: float
#     photons: int = 0

@dataclass
class Pulse:
    """Represents a light pulse."""
    polarization: float
    photons: int = 0

    def __post_init__(self):
        if self.photons < 0:
            raise ValueError("Number of photons cannot be negative.")
        
    def __str__(self):
        return f"Pulse(polarization={self.polarization}, photons={self.photons})"

    def copy(self):
        return Pulse(polarization=self.polarization, photons=self.photons)


class DecoyState(IntEnum):
    """Decoy state types for BB84 with three-intensity protocol."""
    SIGNAL = 0
    WEAK   = 1
    VACUUM = 2

    def __str__(self):
        # restore the simple string format
        return self.name.lower()

class Basis(Enum):
    """Measurement bases for BB84 protocol."""
    Z = "Z"  # Rectilinear basis (H/V)
    X = "X"  # Diagonal basis (D/A)

    # Allow for 0 and 1 as integer representations
    @classmethod
    def from_int(cls, value: int) -> "Basis":
        if value == 0:
            return cls.Z
        elif value == 1:
            return cls.X
        else:
            raise ValueError(f"Invalid basis integer: {value}")

class Bit(IntEnum):
    """Binary bit values."""
    ZERO = 0
    ONE = 1

    def __str__(self):
        return "0" if self == Bit.ZERO else "1"
    

class LaserInfo(BaseModel):
    """Configuration for laser parameters."""
    central_wavelength_nm: float = Field(1550.0, ge=800, le=2000)
    linewidth_Hz: float = Field(1e6, ge=1e3, le=1e9)
    max_power_mW: float = Field(10.0, ge=0.1, le=1000000.0)
    pulse_width_fwhm_ns: float = Field(100.0, ge=10, le=1000)
    pulse_energy_nJ: float = Field(0.1, ge=0.01, le=10.0)
    timing_jitter_ps: float = Field(5.0, ge=0, le=100)
    max_repetition_rate_hz: float = Field(1e6, ge=0, le=1e12)
    polarization_extinction_ratio_dB: float = Field(20.0, ge=10, le=50)
    relative_intensity_noise_dB: float = Field(-150.0, le=-100)

    @field_validator('central_wavelength_nm')
    def validate_wavelength(cls, v):
        if not (400 <= v <= 2000):
            raise ValueError("Central wavelength must be between 400 nm and 2000 nm.")
        return v
    @field_validator('linewidth_Hz')
    def validate_linewidth(cls, v):
        if not (1e3 <= v <= 1e9):
            raise ValueError("Linewidth must be between 1 kHz and 1 GHz.")
        return v
    @field_validator('max_power_mW')
    def validate_max_power(cls, v):
        if not (0.1 <= v <= 1000000.0):
            raise ValueError("Max power must be between 0.1 mW and 1_000_000 mW.")
        return v
    @field_validator('pulse_width_fwhm_ns')
    def validate_pulse_width(cls, v):
        if not (10 <= v <= 1000):
            raise ValueError("Pulse width FWHM must be between 10 ps and 1000 ps.")
        return v
    @field_validator('pulse_energy_nJ')
    def validate_pulse_energy(cls, v):
        if not (0.01 <= v <= 10.0):
            raise ValueError("Pulse energy must be between 0.01 nJ and 10 nJ.")
        return v
    @field_validator('timing_jitter_ps')
    def validate_timing_jitter(cls, v):
        if not (0 <= v <= 100):
            raise ValueError("Timing jitter must be between 0 ps and 100 ps.")
        return v
    @field_validator('max_repetition_rate_hz')
    def validate_repetition_rate(cls, v):
        if not (0 <= v <= 1e12):
            raise ValueError("Max repetition rate must be between 0 Hz and 1 THz.")
        return v
    @field_validator('polarization_extinction_ratio_dB')
    def validate_polarization_extinction_ratio(cls, v):
        if not (10 <= v <= 50):
            raise ValueError("Polarization extinction ratio must be between 10 dB and 50 dB.")
        return v
    @field_validator('relative_intensity_noise_dB')
    def validate_relative_intensity_noise(cls, v):
        if not (-150 <= v <= -100):
            raise ValueError("Relative intensity noise must be between -150 dB and -100 dB.")
        return v




class DecoyInfo(BaseModel):
    """Configuration for decoy state parameters."""
    intensities: Dict[str, float] = Field(
        default={"signal": 0.5, "weak": 0.1, "vacuum": 0.0}
    )
    probabilities: Dict[str, float] = Field(
        default={"signal": 0.7, "weak": 0.2, "vacuum": 0.1}
    )
    # intensities: Dict[DecoyState, float] = Field(
    #     default={DecoyState.SIGNAL: 0.5, DecoyState.WEAK: 0.1, DecoyState.VACUUM: 0.0}
    # )
    # probabilities: Dict[DecoyState, float] = Field(
    #     default={DecoyState.SIGNAL: 0.7, DecoyState.WEAK: 0.2, DecoyState.VACUUM: 0.1}
    # )

    @field_validator('probabilities')
    def probabilities_sum_to_one(cls, v):
        if abs(sum(v.values()) - 1.0) > 1e-6:
            raise ValueError("Decoy state probabilities must sum to 1.0")
        return v


class ChannelInfo(BaseModel):
    """Configuration for quantum channel parameters."""
    length_km: float = Field(20.0, ge=0, le=1000)
    fiber_loss_db_km: float = Field(0.2, ge=0, le=10)
    pol_drift_deg_rms: float = Field(2.0, ge=0, le=180)
    # background_rate_Hz: float = Field(5e-6, ge=0, le=1e-3)
    # depolarization_rate: float = Field(0.01, ge=0, le=1)


class DetectorInfo(BaseModel):
    """Configuration for detector parameters."""
    number: int = Field(4, ge=1, le=16)
    efficiency: float = Field(0.75, ge=0, le=1)
    dark_count_rate_Hz: float = Field(4e-6, ge=0, le=1e-3)
    timing_jitter_ps: float = Field(40.0, ge=0, le=1000)
    dead_time_ns: float = Field(60.0, ge=0, le=10000)



# Type aliases for better code readability
PulseList = List[Pulse]