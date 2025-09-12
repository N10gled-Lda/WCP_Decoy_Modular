# Simple Alice Hardware Test Documentation

## Overview

This directory contains two simplified Alice hardware test implementations that allow testing the hardware control sequence without Bob communication:

1. **Simple Alice Hardware Test** (`simple_alice_hardware_test.py`) - Basic single-threaded test

## Features

✅ **Hardware Control Testing**: Test laser, QRNG, and polarization components  
✅ **No Bob Communication**: Pure Alice-side testing without network  
✅ **Predetermined Sequences**: Define specific bit/basis sequences for testing  
✅ **Random Generation**: Use QRNG for random bit/basis generation  
✅ **Threading Support**: Multi-threaded packet-based structure (packet version)  
✅ **Synchronization**: Proper waiting for polarization readiness before laser firing  
✅ **Statistics**: Detailed timing and performance metrics  

## Simple Alice Hardware Test

### Basic Usage

```python
from examples.simple_alice_hardware_test import SimpleAliceHardwareTest, AliceTestConfig, AliceTestMode

# Random generation test
config = AliceTestConfig(
    num_pulses=10,
    pulse_period_seconds=1.0,
    use_hardware=False,  # Use simulators
    mode=AliceTestMode.RANDOM,
    qrng_seed=42
)

test = SimpleAliceHardwareTest(config)
results = test.run_test()
```

### Predetermined Sequence Test

```python
# Define specific sequence
predetermined_bits = [0, 1, 1, 0, 1]
predetermined_bases = [0, 0, 1, 1, 0]  # 0=Z, 1=X

config = AliceTestConfig(
    num_pulses=5,
    mode=AliceTestMode.PREDETERMINED,
    predetermined_bits=predetermined_bits,
    predetermined_bases=predetermined_bases
)

test = SimpleAliceHardwareTest(config)
results = test.run_test()
```

### Hardware Testing

```python
# Hardware configuration
config = AliceTestConfig(
    num_pulses=5,
    pulse_period_seconds=2.0,  # Slower for hardware
    use_hardware=True,
    com_port="COM4",  # Your polarization hardware COM port
    laser_channel=8,   # Your laser hardware channel
    mode=AliceTestMode.RANDOM
)
```

## Running the Tests

### Quick Test

```bash
# Run all demos
python examples/simple_alice_hardware_test.py
python examples/alice_packet_hardware_test.py
```

### Custom Test

```python
# Create your own test script
from examples.simple_alice_hardware_test import SimpleAliceHardwareTest, AliceTestConfig

config = AliceTestConfig(
    num_pulses=100,
    pulse_period_seconds=0.1,
    qrng_seed=123
)

test = SimpleAliceHardwareTest(config)
results = test.run_test()

print(f"Generated {len(results.bits)} qubits")
print(f"Sequence: {list(zip(results.bits, results.bases))}")
```

## Hardware Requirements

### For Simulator Mode (Default)

- No hardware required
- Uses simulated laser and polarization components

### For Hardware Mode

- **Polarization Control**: STM32-based polarization controller
  - COM port connection (e.g., "COM4")
  - Supports rotation commands and availability feedback
- **Laser Control**: Digital hardware laser driver
  - Digital channel connection (e.g., channel 8)
  - Supports single pulse triggering
