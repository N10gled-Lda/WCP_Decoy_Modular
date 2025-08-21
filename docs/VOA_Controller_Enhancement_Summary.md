# VOA Controller Enhancement Summary

This document summarizes the enhanced VOA (Variable Optical Attenuator) controller implementation
with QRNG integration and probability-based state selection for quantum key distribution systems.

## Key Features

### 1. Configuration Classes

#### IntensityConfig

- Configures the three decoy state intensities:
  - `mu_signal`: Signal state intensity (default: 0.5)
  - `mu_weak`: Weak decoy state intensity (default: 0.1)  
  - `mu_vacuum`: Vacuum state intensity (default: 0.0)

#### ProbabilityConfig

- Configures the selection probabilities for each state:
  - `p_signal`: Signal state probability (default: 0.7)
  - `p_weak`: Weak decoy state probability (default: 0.2)
  - `p_vacuum`: Vacuum state probability (default: 0.1)
- Validates that probabilities sum to 1.0

#### DecoyConfig

- Combined configuration containing both intensity and probability configs
- Automatically creates default probability config if not provided

### 2. Enhanced VOAController Class

#### Initialization Options

The controller supports multiple initialization methods:

```python
# Method 1: Separate configs
voa = VOAController(
    physical=False,
    intensity_config=IntensityConfig(...),
    probability_config=ProbabilityConfig(...),
    n_pulse_initial=1.0
)

# Method 2: Combined config
voa = VOAController(
    physical=False,
    decoy_config=DecoyConfig(...),
    n_pulse_initial=1.0
)

# Method 3: Intensity only (uses default probabilities)
voa = VOAController(
    physical=False,
    intensity_config=IntensityConfig(...),
    n_pulse_initial=1.0
)
```

#### State Selection Methods

1. **Probability-based Selection**: `select_state_by_probability()`
   - Uses 16 QRNG bits to generate high-precision random float
   - Applies inverse transform sampling based on configured probabilities
   - Provides accurate statistical distribution matching target probabilities

2. **Uniform Selection**: `select_random_state()`
   - Uses 2 QRNG bits with fixed mapping:
     - 00 → SIGNAL
     - 01 → SIGNAL  
     - 10 → WEAK
     - 11 → VACUUM
   - Provides uniform distribution (not probability-weighted)

#### Pulse Generation Methods

1. **`generate_pulse_with_probability_selection()`**
   - Uses probability-based state selection
   - Returns VOAOutput with selected state, target intensity, and calculated attenuation

2. **`generate_pulse_with_uniform_selection()`**
   - Uses uniform random state selection
   - Returns VOAOutput with selected state, target intensity, and calculated attenuation

3. **`generate_pulse_with_state_selection(use_probabilities=True)`**
   - Unified method that can switch between probability and uniform selection
   - Default behavior uses probabilities if configured

#### Attenuation Calculation

The controller calculates attenuation using the formula:

```latex
A_dB = 10 * log10(N_pulse / μ)
```

Where:

- `N_pulse`: Initial number of photons per pulse
- `μ`: Target mean photon number for the selected state

Special handling:

- Vacuum state (μ=0): Uses maximum attenuation (60 dB)
- Bounds checking: 0 ≤ A_dB ≤ 60 dB

### 3. VOAOutput Data Structure

The output contains:

- `pulse_type`: Selected DecoyState (SIGNAL, WEAK, or VACUUM)
- `attenuation_db`: Calculated attenuation in dB
- `target_intensity`: Target mean photon number (μ)

### 4. Integration with QRNG

- **Hardware Mode**: Uses QRNGHardware (when physical=True)
- **Simulation Mode**: Uses QRNGSimulator (when physical=False)
- **Bit Generation**: Supports both single bits and multi-bit generation for precision

## Usage Examples

### Basic Setup with Probabilities

```python
from src.alice.voa.voa_controller import (
    VOAController, IntensityConfig, ProbabilityConfig
)

# Configure intensities and probabilities
intensity_config = IntensityConfig(mu_signal=0.5, mu_weak=0.1, mu_vacuum=0.0)
probability_config = ProbabilityConfig(p_signal=0.7, p_weak=0.2, p_vacuum=0.1)

# Initialize controller
voa = VOAController(
    physical=False,
    intensity_config=intensity_config,
    probability_config=probability_config,
    n_pulse_initial=1.0
)

# Generate pulse with probability-based selection
output = voa.generate_pulse_with_probability_selection()
print(f"State: {output.pulse_type}, μ: {output.target_intensity}, A: {output.attenuation_db:.2f}dB")
```

### Custom Probability Distributions

```python
# High signal probability (for testing/debugging)
high_signal = ProbabilityConfig(p_signal=0.9, p_weak=0.08, p_vacuum=0.02)

# Equal probabilities (uniform across states)
equal_probs = ProbabilityConfig(p_signal=0.33, p_weak=0.33, p_vacuum=0.34)

# QKD-optimized (higher decoy rates)
qkd_optimized = ProbabilityConfig(p_signal=0.5, p_weak=0.3, p_vacuum=0.2)
```

### Statistical Analysis

```python
# Generate many samples and analyze distribution
counts = {DecoyState.SIGNAL: 0, DecoyState.WEAK: 0, DecoyState.VACUUM: 0}
n_samples = 1000

for _ in range(n_samples):
    output = voa.generate_pulse_with_probability_selection()
    counts[output.pulse_type] += 1

# Compare actual vs expected distribution
for state in counts:
    actual_pct = (counts[state] / n_samples) * 100
    expected_pct = probability_config.get_probability(state) * 100
    print(f"{state}: {actual_pct:.1f}% (expected {expected_pct:.1f}%)")
```

## Key Advantages

1. **Flexible Configuration**: Support for separate or combined intensity/probability configs
2. **Multiple Selection Methods**: Probability-based for QKD protocols, uniform for testing
3. **High Precision**: 16-bit QRNG for accurate probability distribution
4. **Hardware/Simulation Support**: Seamless switching between physical and simulated components
5. **Automatic Attenuation**: Calculates optimal attenuation for each state
6. **Statistical Validation**: Built-in support for analyzing state distributions
7. **QKD Protocol Ready**: Designed specifically for 3-intensity decoy state protocols

## Core Logic Flow

1. **Initialize** controller with intensity and probability configurations
2. **Generate** quantum random bits using QRNG
3. **Select** decoy state based on configured probabilities
4. **Calculate** required attenuation for target intensity
5. **Set** VOA attenuation (hardware) or store value (simulation)
6. **Return** complete pulse information (state, intensity, attenuation)

This implementation provides a complete solution for VOA control in quantum key distribution systems with support for both hardware and simulation environments.
"""
