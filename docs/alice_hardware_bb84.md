# Alice Hardware BB84 Implementation

This implementation integrates Alice's hardware components (laser, QRNG, polarization control) with the BB84 quantum key distribution protocol, replacing the previous simulation-based `AliceQubits` class.

## Features

- **Hardware Integration**: Uses actual laser, polarization, and QRNG hardware components
- **Multiple Test Modes**: Support for random stream, batch, seeded, and predetermined sequences
- **Mock Receiver**: Built-in mock Bob receiver for testing without actual hardware
- **TRL4 Protocol**: Technology Readiness Level 4 implementation for hardware validation
- **Real-time Control**: Sends pulses with configurable timing and proper polarization setup
- **Performance Metrics**: Detailed timing and performance tracking via AliceTestResults
- **Protocol Compatibility**: Integrates with existing BB84 classical communication and error reconciliation
- **Flexible Configuration**: Can switch between hardware and simulation modes
- **Multi-threading Support**: Supports parallel key generation threads

## Architecture

### Components

1. **AliceHardwareQubits**: Main class that coordinates hardware components
2. **Hardware Controllers**:
   - `LaserController`: Controls laser pulse generation
   - `PolarizationController`: Sets polarization based on basis and bit
   - `QRNGSimulator`: Generates random bits and bases
3. **Protocol Integration**: Works with existing `AliceThread` for classical communication

### Operation Flow

```text
1. Initialize hardware components (laser, polarization, QRNG)
2. Configure test mode (RANDOM_STREAM, SEEDED, PREDETERMINED, etc.)
3. Setup quantum channel server and optionally mock receiver
4. For each qubit:
   a. Generate or retrieve basis (Z/X) and bit (0/1) based on mode
   b. Set polarization controller to correct state
   c. Wait for polarization to be ready (with timeout)
   d. Fire laser pulse
   e. Send encoded qubit over quantum channel
   f. Record timing and performance metrics
5. Complete TRL4 protocol quantum transmission
6. Return AliceTestResults with detailed metrics
7. Run classical communication protocol (basis comparison, error correction)
8. Perform privacy amplification if configured
```

## Usage

### Basic Example

```python
from examples.alice_hardware_bb84 import AliceHardwareQubits, HardwareAliceConfig, AliceTestMode

# Configure Alice
config = HardwareAliceConfig(
    num_qubits=1000,
    pulse_period_seconds=1.0,   # 1 Hz pulse rate
    use_hardware=True,
    com_port="COM4",           # Polarization control
    laser_channel=8,           # Laser channel
    mode=AliceTestMode.RANDOM_STREAM,  # Random bit generation mode
    use_mock_receiver=False,   # Use real Bob receiver
    server_host="localhost",
    server_port=12345,
)

# Create and run Alice
alice = AliceHardwareQubits(config)
results = alice.run_trl4_protocol_quantum_part(quantum_channel_connection)
```

### Command Line Interface

Run the full BB84 protocol:

```bash
# With hardware
python examples/alice_hardware_bb84.py --use_hardware --com_port COM4 --laser_channel 8

# With simulation
python examples/alice_hardware_bb84.py -k 1000 -pp 0.5

# With threading
python examples/alice_hardware_bb84.py -nth 2 -k 2048 --use_hardware

# Show all options
python examples/alice_hardware_bb84.py --help
```

### Demo Script

Run a simple demonstration:

```bash
python examples/alice_hardware_demo.py
```

The demo script shows:

- Hardware component initialization
- Mock receiver setup for testing
- TRL4 protocol execution
- Performance metrics display
- Timing analysis (pulse rate, execution time)
- First 10 qubits for verification

## Configuration Options

### HardwareAliceConfig Parameters

- `num_qubits`: Number of qubits to generate (default: 20)
- `pulse_period_seconds`: Time between pulses (default: 1.0)
- `use_hardware`: Use actual hardware vs simulation (default: False)
- `com_port`: COM port for polarization hardware (default: None)
- `laser_channel`: Digital channel for laser (default: 8)
- `mode`: Test mode for bit/basis generation (default: RANDOM_STREAM)
- `qrng_seed`: Seed for reproducible random generation (default: None)
- `predetermined_bits`: Pre-defined bit sequence (for PREDETERMINED mode)
- `predetermined_bases`: Pre-defined basis sequence (for PREDETERMINED mode)
- `use_mock_receiver`: Use mock Bob receiver for testing (default: False)
- `server_host`: Host for quantum channel server (default: "localhost")
- `server_port`: Port for quantum channel server (default: 12345)
- `test_fraction`: Fraction for error rate testing (default: 0.11)
- `loss_rate`: Simulated loss rate (default: 0.0)

### Test Modes (AliceTestMode)

- `RANDOM_STREAM`: Generate random bits using QRNG bit by bit
- `RANDOM_BATCH`: Generate random bits in batch using QRNG all before
- `SEEDED`: Generate bits using a fixed seed using QRNG
- `PREDETERMINED`: Use pre-defined sequence (requires predetermined_bits and predetermined_bases)

### AliceTestResults

The `AliceTestResults` class tracks detailed performance metrics:

- `bits`: Generated bit sequence
- `bases`: Generated basis sequence  
- `polarization_angles`: Actual polarization angles achieved
- `pulse_times`: Timestamp for each pulse
- `rotation_times`: Time taken for polarization rotation
- `wait_times`: Wait times during operations
- `laser_elapsed`: Laser operation timing
- `total_runtime`: Total execution time
- `errors`: List of errors encountered

### Hardware vs Simulation

**Hardware Mode** (`use_hardware=True`):

- Requires actual laser and polarization hardware
- Uses real COM port connections
- Actual physical control of components

**Simulation Mode** (`use_hardware=False`):

- Uses software simulators
- No hardware requirements
- Good for testing and development

## Integration with Existing Protocol

This implementation is designed to be a drop-in replacement for the old `AliceQubits` class:

```python
# Old way (simulation only)
alice_qubit = AliceQubits(num_qubits=1000, qubit_delay_us=1000000, ...)
alice_qubit.run_alice_qubits(connection)

# New way (hardware integrated)
config = HardwareAliceConfig(num_qubits=1000, pulse_period_seconds=1.0, ...)
alice_hardware = AliceHardwareQubits(config)
results = alice_hardware.run_trl4_protocol_quantum_part(connection)

# Both provide the same interface for classical protocol
alice_ccc.set_bits_bases_qubits(
    results.bits,      # Same format as before
    results.bases,     # Same format as before  
    alice_hardware.qubits_bytes  # Same format as before
)
```

### TRL4 Protocol

The new implementation uses the TRL4 (Technology Readiness Level 4) protocol method:

- `run_trl4_protocol_quantum_part()`: Handles the quantum transmission part
- Returns `AliceTestResults` with detailed performance metrics
- Supports both hardware and simulation modes
- Includes mock receiver functionality for testing without Bob

## Hardware Requirements

### Laser Control

- Digital laser controller with channel-based control
- Compatible with `DigitalHardwareLaserDriver`

### Polarization Control

- STM32-based polarization controller
- Serial/COM port communication
- Compatible with `PolarizationHardware`

### Random Number Generation

- Currently uses software QRNG simulator
- Can be extended to use hardware QRNG devices

## Timing and Synchronization

- **Pulse Period**: Configurable time between pulses (default: 1 second)
- **Polarization Setup**: Ensures polarization is ready before laser fires (10s timeout)
- **Performance Tracking**: Detailed timing metrics via AliceTestResults
- **Real-time**: Maintains consistent pulse timing regardless of processing time
- **Protocol Markers**: Handshake, acknowledgment, and end markers for communication

## Networking and Communication

- **Quantum Channel**: TCP socket-based communication to Bob
- **Mock Receiver**: Built-in echo server for testing without Bob
- **Protocol Markers**: Structured communication with handshake and acknowledgment
- **Multi-threading**: Supports concurrent quantum and classical processing
- **Error Handling**: Graceful connection management and error recovery

## Error Handling

- Component initialization failures are caught and reported
- Hardware errors during operation are logged but don't stop the protocol
- Graceful shutdown of hardware components
- Timeout handling for hardware operations

## Threading Model

- Main thread: Protocol coordination and classical communication
- Pulse thread: Hardware control and qubit generation
- Reader thread: Classical channel message processing
- Multiple protocol threads: Parallel key generation (configurable)

## Differences from Original

### Replaced Features

- **Old**: `AliceQubits` with pure simulation
- **New**: `AliceHardwareQubits` with hardware integration

### Enhanced Features

- Real hardware control instead of simulation
- Proper timing control (seconds instead of microseconds)
- Component lifecycle management
- Better error handling and logging
- Configurable hardware/simulation modes

### Maintained Compatibility

- Same data formats (`bits`, `bases`, `qubits_bytes`)
- Same socket-based quantum channel protocol
- Same classical communication integration
- Same error reconciliation and privacy amplification

## Future Enhancements

1. **Hardware QRNG**: Replace simulator with actual quantum RNG
2. **Advanced Test Modes**: Additional test patterns and validation modes
3. **Real-time Monitoring**: Web-based dashboard for system monitoring
4. **Multi-channel Support**: Parallel quantum channels for increased throughput
5. **Adaptive Timing**: Dynamic pulse rate based on system performance and channel conditions
6. **Remote Control**: Network-based hardware control and configuration
7. **Advanced Diagnostics**: Real-time hardware health monitoring and predictive maintenance
8. **Integration Testing**: Automated test suites for different hardware configurations

## Troubleshooting

### Common Issues

1. **COM Port Not Found**: Check that polarization hardware is connected
2. **Laser Channel Error**: Verify laser controller is properly configured
3. **Timing Issues**: Adjust `pulse_period_seconds` for your hardware
4. **Socket Errors**: Ensure Bob is ready to receive on quantum channel
5. **Mock Receiver Issues**: Check that `use_mock_receiver=True` for testing
6. **Test Mode Errors**: Verify predetermined sequences match `num_qubits`

### Debug Mode

Enable debug logging for detailed operation info:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

### Testing Hardware

Use the demo script to test hardware without full protocol:

```bash
python examples/alice_hardware_demo.py
```

### Testing with Mock Receiver

For testing without Bob, enable mock receiver:

```python
config = HardwareAliceConfig(
    use_mock_receiver=True,
    server_host="localhost", 
    server_port=12345,
    # ... other config
)
```
