# QRNG Module Documentation

## Overview

The QRNG (Quantum Random Number Generator) module provides quantum random number generation capabilities for the QKD system. It implements both simulated and hardware interfaces with different operation modes for testing and production use.

## File Structure

### 1. [`qrng_simulator.py`](qrng_simulator.py)

**Status**: ✅ Core functionality implemented, ⚠️ Missing advanced features

#### [`OperationMode`](qrng_simulator.py) (Enum)

- **Purpose**: Defines different operating modes for the QRNG
- **Values**:
  - `BATCH` - Pre-generate all bits for better performance
  - `STREAMING` - Generate bits on-demand for real-time operation
  - `DETERMINISTIC` - Use PRNG with fixed seed for testing/reproducibility

#### [`QRNGSimulator`](qrng_simulator.py) (Main Simulator Class)

- **Purpose**: Simulated quantum random number generator using numpy PRNG
- **Status**: Core functionality complete

**Implemented Methods**:

- `__init__(seed=None)` - Initialize with optional seed for reproducibility
- `set_seed(seed)` - Update the random seed and reinitialize generator
- `get_random_bit(mode, size=1)` - Generate random bits based on operation mode
- `get_bits_generated()` - Return total count of generated bits
- `reset()` - Reset internal state and bit counter

**Private Methods**:

- `_get_random_bit()` - Generate single random bit (0 or 1)
- `_get_cache_random_bits(size)` - Generate batch of random bits as list

**Features**:

- ✅ Multiple operation modes support
- ✅ Seeded generation for reproducible testing
- ✅ Bit generation tracking
- ✅ Proper logging throughout
- ✅ State reset capability

**Missing Features**:

- ⚠️ Bias probability, entropy estimation, noise simulation ??
- ⚠️ Quality checks not implemented ??

---

### 2. [`qrng_hardware.py`](qrng_hardware.py)

**Status**: ❌ Placeholder implementation only

#### [`QRNGHardware`](qrng_hardware.py) (Hardware Interface)

- **Purpose**: Interface to real quantum random number generator hardware
- **Current Status**: All methods raise NotImplementedError

**Placeholder Methods**:

- `__init__()` - Hardware initialization (not implemented)
- `get_random_bit()` - Get random bit from hardware (not implemented)
- `get_random_bits()` - Get random bits from hardware (not implemented)

**Issues**:

- ❌ No actual hardware communication
- ❌ Missing interface consistency with simulator

---

## Current Issues & TODOs

### Missing Content

1. **Inconsistent Interface**: Hardware and simulator classes have different method signatures
2. **Missing Implementation**: Hardware class is completely unimplemented
3. **Empty Module Init**: No proper module structure or exports
4. **Quality Checks**: Statistical tests for randomness quality
5. **Bias Implementation**: Configurable bias probability not used
6. **Entropy Estimation**: No measure of randomness quality

### Architectural Concerns

1. **No Base Class**: Missing abstract interface for hardware/simulator abstraction
2. **Mode Handling**: Mode parameter passed to method instead of class-level configuration
3. **Factory Pattern**: No factory for creating QRNG instances

---

## Recommendations for Improvement

### 1. **Create Abstract Base Class**

```python
class BaseQRNG(ABC):
    @abstractmethod
    def get_random_bit(self) -> int: pass
    
    @abstractmethod  
    def get_random_bits(self, count: int) -> List[int]: pass

```

### 2. **Implement Missing Features in Simulator**

- Use `bias_probability` from config
- Add quality checks (frequency tests, runs tests, etc.)
- Implement entropy estimation
- Add basis generation for quantum states

### 3. **Add Factory Pattern**

```python
def create_qrng(use_hardware: bool = False, config: QRNGConfig = None) -> BaseQRNG:
    if use_hardware:
        return QRNGHardware(config)
    return QRNGSimulator(config)
```

---

## Architecture Strengths

- ✅ Clean separation between simulation and hardware
- ✅ Flexible operation modes for different use cases
- ✅ Good logging and debugging support
- ✅ Configurable and extensible design
- ✅ Thread-safe operations (numpy RNG is thread-safe)
