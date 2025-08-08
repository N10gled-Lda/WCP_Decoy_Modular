# Laser Module Documentation

## Overview

The laser module implements a modular architecture with hardware abstraction, allowing for both simulated and real hardware laser control. The system follows a driver pattern where different implementations can be swapped without changing the controller logic.

## File Structure

### 1. laser_controller.py

**Status**: ✅ Core functionality complete, ⚠️ Some methods need implementation

#### `BaseLaserDriver` (Abstract Base Class)

- **Purpose**: Defines the interface that all laser drivers must implement
- **Methods**:
  - `turn_on()` - Turn on the laser hardware
  - `turn_off()` - Turn off the laser hardware  
  - `stop()` - Safely disable emission
  - `arm(repetition_rate_hz)` - Prepare for pulse sequence
  - `fire(pattern)` - Emit pulses according to pattern
  - `fire_single_pulse()` - Emit a single pulse

#### `LaserController` (Main Controller)

- **Purpose**: High-level laser control of the laser system, either simulated or real hardware
- **Status**: Core methods implemented, some advanced features pending

**Implemented Methods**:

- `turn_on()` - Activates laser and sets internal state
- `turn_off()` - Deactivates laser and clears state
- `is_active()` - Returns current laser status
- `start_continuous(rep_rate_hz)` - Start continuous emission mode
- `stop_continuous()` - Stop continuous emission
- `trigger_once()` - Emit single pulse with default parameters
- `send_frame(n_triggers, rep_rate_hz)` - Send a frame of pulses at specified rate using the driver `fire_single_pulse()` method

**TODO Methods**:

- Driver initialization and configuration - either the driver should be passed in or initialized within the controller

---

### 2. laser_hardware.py

**Status**: ⚠️ Template implementation, needs hardware-specific code

#### `HardwareLaserDriver` (Hardware Implementation)

- **Purpose**: Real hardware interface for laser control
- **Current Status**: Template with placeholder commands

**Methods** (All need hardware-specific implementation):

- `turn_on()` - Sends "OUTP ON" command
- `turn_off()` - Sends "OUTP OFF" command
- `arm(repetition_rate_hz)` - Sets frequency and trigger mode
- `fire(pattern)` - Sends pulse width, power, and shape commands
- `stop()` - Sends "OUTP OFF" command

**Issues**:

- ⚠️ All hardware commands are placeholders
- ⚠️ Missing proper error handling

**TODO**:

- Implement actual hardware communication
- Add error handling and validation
- Add power setting functionality (commented idea)

---

### 3. laser_simulator.py

**Status**: ✅ Core implemented, ⚠️ Interface mismatch

#### `SimulatedLaserDriver` (Simulation Implementation)  

- **Purpose**: Software simulation of laser behavior for testing/development
- **Status**: Most complete implementation

**Implemented Methods**:

- `turn_on()` - Sets simulation state to active
- `turn_off()` - Sets simulation state to inactive  
- `arm(repetition_rate_hz)` - Starts threaded pulse generation
- `fire(pattern)` - Fires pulse pattern in separate thread
- `stop()` - Stops armed/firing threads safely
- `fire_single_pulse()` - Extra method for single pulse emission
- `generate_single_pulse()` - Creates realistic pulse with Poisson photon statistics

**Advanced Features**:

- ✅ Threading for non-blocking operation
- ✅ Queue-based pulse delivery
- ✅ Realistic photon number simulation (Poisson distribution)
- ✅ Polarization simulation
- ✅ Physics-based calculations (photon energy, pulse energy)

**Issues**:

- ⚠️ Interface mismatch with other drivers

---

## Current Issues & TODOs

### Critical Issues

1. **Missing Method**: `send_frame()` calls undefined `trigger_once()` on drivers

### Missing Implementations

1. **Hardware Driver**: All methods are placeholders
2. **Error Handling**: Robust error handling across all modules

### Recommendations

1. Fix import statements in hardware driver
2. Complete hardware driver implementation
3. Add comprehensive error handling
4. Implement missing controller features
5. Add unit tests for all components

## Architecture Strengths

- ✅ Clean separation of concerns
- ✅ Pluggable driver architecture
- ✅ Good logging throughout
- ✅ Realistic simulation capabilities
- ✅ Thread-safe operations in simulator
