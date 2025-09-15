# WCP BB84 Protocol Implementation

A comprehensive implementation of the **Weak Coherent Pulse (WCP) BB84** quantum key distribution protocol with decoy-state analysis and enhanced security features.

<!-- TOC -->
Table of Contents:

- [Overview](#-overview)
- [Key Features](#-key-features)
- [File Structure](#-file-structure)
- [Quick Start](#-quick-start)
- [Technical Details](#-technical-details)
- [Testing](#-testing)
- [Performance Metrics](#-performance-metrics)
- [Advanced Configuration](#-advanced-configuration)
- [Security Considerations](#️-security-considerations)
- [Example Output](#-example-output)

## 🔐 Overview

This implementation extends the standard BB84 protocol to use realistic **Weak Coherent Pulses (WCP)** instead of ideal single-photon sources. The WCP approach addresses real-world challenges in quantum cryptography by:

- Using Poisson-distributed photon numbers instead of perfect single photons
- Implementing decoy-state protocols for enhanced security
- Detecting photon-number-splitting (PNS) attacks
- Providing realistic parameter estimation and key rate calculations

## 🌟 Key Features

### 1. **WCP Pulse Generation**
- **Poisson Distribution**: Photon numbers follow realistic Poisson statistics
- **Multi-Intensity States**: Signal, decoy, and vacuum pulses with different intensities
- **Realistic Modeling**: Accounts for detector efficiency and dark counts

### 2. **Decoy-State Protocol**
- **Three-State System**: Signal (μₛ), decoy (μₐ), and vacuum (μᵥ) states
- **Parameter Estimation**: Single-photon yield (Y₁) and error rate (e₁) calculation
- **Security Analysis**: Bounds on secure key generation rate

### 3. **Attack Detection**
- **PNS Attack Detection**: Identifies photon-number-splitting attacks
- **Gain Analysis**: Monitors detection statistics across pulse types
- **Threshold Alerts**: Automated security violation warnings

### 4. **Enhanced BB84 Implementation**
- **Alice & Bob Classes**: Full protocol implementation with classical communication
- **Multi-threading Support**: Concurrent quantum and classical channels
- **Comprehensive Logging**: Detailed execution and security reports

## 📁 File Structure

```
BB84/
├── bb84_protocol/
│   ├── wcp_pulse.py                 # WCP pulse generation and management
│   ├── wcp_parameter_estimation.py  # Decoy-state parameter estimation
│   ├── alice_wcp_bb84_ccc.py       # Alice WCP implementation
│   ├── bob_wcp_bb84_ccc.py         # Bob WCP implementation
│   └── wcp_qubit.py                # WCP qubit utilities
├── bb84_alice_wcp_mock_qch.py      # Alice main execution script
├── bb84_bob_wcp_mock_qch.py        # Bob main execution script
├── test_wcp_bb84.py                # Comprehensive test suite
└── README_WCP_BB84.md              # This documentation
```

## 🚀 Quick Start

### 1. **Installation**

Ensure you have the required dependencies:

```bash
pip install numpy scipy matplotlib
```

### 2. **Basic Usage**

**Run Alice (Sender):**
```bash
python bb84_alice_wcp_mock_qch.py --key-length 1000 --signal-intensity 0.5 --decoy-intensity 0.1
```

**Run Bob (Receiver):**
```bash
python bb84_bob_wcp_mock_qch.py --detector-efficiency 0.1 --dark-count-rate 1e-6
```

### 3. **Command Line Parameters**

#### Alice Parameters:
- `--key-length`: Number of qubits to send (default: 3500)
- `--signal-intensity`: Mean photon number for signal pulses (default: 0.5)
- `--decoy-intensity`: Mean photon number for decoy pulses (default: 0.1)
- `--signal-prob`: Probability of sending signal pulse (default: 0.7)
- `--decoy-prob`: Probability of sending decoy pulse (default: 0.25)
- `--vacuum-prob`: Probability of sending vacuum pulse (default: 0.05)

#### Bob Parameters:
- `--detector-efficiency`: Single-photon detector efficiency (default: 0.1)
- `--dark-count-rate`: Detector dark count rate (default: 1e-6)
- `--num-threads`: Number of processing threads (default: 4)

## 🔬 Technical Details

### WCP Pulse Model

The WCP model implements realistic photon sources where:

```python
# Photon number follows Poisson distribution
n ~ Poisson(μ)

# Detection probability
P_det = 1 - exp(-η·n) + P_dark

# Where:
# μ = mean photon number (intensity)
# η = detector efficiency  
# P_dark = dark count probability
```

### Decoy-State Analysis

The implementation uses the **GLLP protocol** for parameter estimation:

1. **Single Photon Yield Bounds**:
   ```
   Y₁ᴸ ≤ Y₁ ≤ Y₁ᵁ
   
   where:
   Y₁ᴸ = (μₛQ₀ - μ₀Qₛ)/(μₛ - μ₀)
   Y₁ᵁ = (μ₀Qₐ - μₐQ₀)/(μ₀ - μₐ)
   ```

2. **Single Photon Error Rate**:
   ```
   e₁ = E₁/Y₁
   ```

3. **Secure Key Rate**:
   ```
   R ≥ Y₁[1 - H(e₁)] - fₑcH(E)
   ```

### Security Features

#### PNS Attack Detection
- Monitors gain ratios between pulse types
- Detects anomalous detection patterns
- Implements threshold-based alerting

#### Parameter Validation
- Verifies decoy-state inequalities
- Checks measurement statistics consistency
- Validates secure key generation conditions

## 📊 Testing

Run the comprehensive test suite:

```bash
python test_wcp_bb84.py
```

The test suite validates:
- ✅ Poisson photon number generation
- ✅ Pulse encoding/decoding mechanisms
- ✅ Parameter estimation accuracy
- ✅ Security analysis functions
- ✅ Protocol integration
- ✅ Attack detection capabilities

### Test Categories

1. **`TestWCPPulse`**: Basic pulse generation and properties
2. **`TestWCPIntensityManager`**: Pulse type selection and intensity management
3. **`TestWCPParameterEstimator`**: Decoy-state analysis and security metrics
4. **`TestWCPProtocolIntegration`**: End-to-end protocol testing
5. **`TestWCPSecurityAnalysis`**: Attack detection and security validation

## 📈 Performance Metrics

### Typical Performance Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Signal Intensity (μₛ) | 0.5 | Mean photons per signal pulse |
| Decoy Intensity (μₐ) | 0.1 | Mean photons per decoy pulse |
| Detector Efficiency | 0.1 | Single-photon detection efficiency |
| Dark Count Rate | 1×10⁻⁶ | False detection probability |
| QBER Threshold | 11% | Maximum tolerable error rate |
| Signal Probability | 70% | Fraction of signal pulses |
| Decoy Probability | 25% | Fraction of decoy pulses |

### Expected Outputs

- **Gain**: ~5% for signal, ~2% for decoy, ~10⁻⁶ for vacuum
- **QBER**: ~11% under normal conditions
- **Single Photon Yield**: Depends on channel loss and detector efficiency
- **Secure Key Rate**: Calculated using GLLP bounds

## 🔧 Advanced Configuration

### Custom Intensity Management

```python
from BB84.bb84_protocol.wcp_pulse import WCPIntensityManager

# Create custom intensity manager
manager = WCPIntensityManager(
    signal_intensity=0.8,    # Higher signal intensity
    decoy_intensity=0.15,    # Higher decoy intensity
    vacuum_intensity=0.0,    # Always zero for vacuum
    signal_prob=0.6,         # Custom probability distribution
    decoy_prob=0.35,
    vacuum_prob=0.05
)
```

### Parameter Estimation Settings

```python
from BB84.bb84_protocol.wcp_parameter_estimation import WCPParameterEstimator

# Configure parameter estimator
estimator = WCPParameterEstimator(
    signal_intensity=0.5,
    decoy_intensity=0.1,
    vacuum_intensity=0.0,
    detector_efficiency=0.12,   # Higher detector efficiency
    dark_count_rate=5e-7        # Lower dark count rate
)
```

## 🛡️ Security Considerations

### Attack Resistance

The WCP implementation provides resistance against:

1. **Photon Number Splitting (PNS) Attacks**
   - Detected through gain analysis
   - Mitigated using decoy states

2. **Intercept-Resend Attacks**
   - Detected through elevated QBER
   - Threshold-based detection

3. **Beam Splitting Attacks**
   - Limited by single-photon component analysis
   - Decoy-state bounds provide security

### Security Thresholds

| Attack Type | Detection Method | Threshold |
|-------------|------------------|-----------|
| Intercept-Resend | QBER Analysis | > 11% |
| PNS Attack | Gain Ratio | Deviation > 10% |
| General Eavesdropping | Parameter Bounds | Negative key rate |

## 📝 Example Output

### Alice Output:
```
=== WCP BB84 Alice - Protocol Results ===
Total qubits sent: 3500
Signal pulses: 2450 (70.0%)
Decoy pulses: 875 (25.0%)
Vacuum pulses: 175 (5.0%)
Average photon number: 0.375
Protocol execution time: 45.2 seconds
```

### Bob Output:
```
=== WCP BB84 Bob - Security Analysis ===
Total pulses received: 3500
Detection statistics:
  Signal: 122/2450 (5.0% gain)
  Decoy: 18/875 (2.1% gain)
  Vacuum: 0/175 (0.0% gain)

QBER Analysis:
  Signal: 11.5%
  Decoy: 10.8%

Single Photon Parameters:
  Y₁ (lower bound): 0.089
  Y₁ (upper bound): 0.156
  e₁ (error rate): 12.3%

Security Status: ✅ SECURE
PNS Attack Detected: ❌ NO
Estimated Secure Key Rate: 0.023 bits/pulse
```

## 🤝 Contributing

To contribute to the WCP BB84 implementation:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/enhancement`
3. **Add comprehensive tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with detailed description

### Development Guidelines

- Follow PEP 8 style guidelines
- Include docstrings for all public methods
- Add unit tests for new features
- Update this README for significant changes

## 📚 References

1. **Lo, H.-K., et al.** (2005). Decoy state quantum key distribution. *Physical Review Letters*, 94(23), 230504.

2. **Ma, X., et al.** (2005). Practical decoy state for quantum key distribution. *Physical Review A*, 72(1), 012326.

3. **Gottesman, D., et al.** (2004). Security of quantum key distribution with imperfect devices. *ISIT 2004*.

4. **Lütkenhaus, N.** (2000). Security against individual attacks for realistic quantum key distribution. *Physical Review A*, 61(5), 052304.

## 📄 License

This implementation is provided under the same license as the parent QKD project. See `LICENSE` file for details.

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure parent directory is in Python path
   export PYTHONPATH="${PYTHONPATH}:$(dirname $(pwd))"
   ```

2. **Communication Channel Errors**
   ```bash
   # Check if ports are available
   netstat -an | grep :65432
   ```

3. **Low Detection Rates**
   - Verify detector efficiency settings
   - Check signal intensity parameters
   - Ensure proper channel simulation

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

**Last Updated**: June 2025  
**Version**: 1.0.0  
**Authors**: WCP BB84 Development Team
