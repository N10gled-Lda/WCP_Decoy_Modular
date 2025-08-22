# How to use AliceCPUGeneral

**`AliceCPUGeneral`**: Complete general version (all components, hardware/simulator selection)

## **🎯 Key Features:**

### **1. Complete Component Support:**

- **All Components**: QRNG, Laser, VOA, and Polarization
- **Flexible Hardware/Simulator Selection**: Each component can independently be set to hardware or simulator mode

### **2. Configuration System:**

```python
@dataclass
class SimulationConfig:
    # General settings
    pulses_total: int = 10000
    use_hardware: bool = False
    
    # Component-specific hardware flags
    use_hardware_qrng: bool = False      # QRNG: hardware vs simulator
    use_hardware_laser: bool = False     # Laser: hardware vs simulator  
    use_hardware_voa: bool = False       # VOA: hardware vs simulator
    use_hardware_polarization: bool = False  # Polarization: hardware vs simulator
    
    # Hardware-specific settings
    com_port_polarization: Optional[str] = None
    laser_repetition_rate_hz: float = 1000000
    
    # QRNG operation mode
    qrng_mode: OperationMode = OperationMode.STREAMING
```

### **3. Intelligent Component Initialization:**

- **QRNG**: Chooses between `QRNGHardware` and `QRNGSimulator`
- **Laser**: Chooses between `DigitalHardwareLaserDriver`, `HardwareLaserDriver`, and `SimulatedLaserDriver`
- **VOA**: Chooses between `VOAHardwareDriver` and `VOASimulator`
- **Polarization**: Chooses between `PolarizationHardware` and `PolarizationSimulator`

### **4. Threading Architecture:**

- Queue-based processing
- Proper thread management and cleanup
- Hardware vs simulator thread handling

### **5. Complete Data Collection:**

- Alice's transmission data recording
- Component statistics
- Performance metrics
- Error tracking

## **🔧 Usage Examples:**

### **Full Simulator Mode:**

```python
config = SimulationConfig(
    pulses_total=10000,
    use_hardware=False,
    use_hardware_qrng=False,
    use_hardware_laser=False, 
    use_hardware_voa=False,
    use_hardware_polarization=False,
    qrng_mode=OperationMode.STREAMING,
    random_seed=42
)

alice = AliceCPUGeneral(config)
alice.start_transmission()
```

### **Mixed Hardware/Simulator Mode:**

```python
config = SimulationConfig(
    pulses_total=1000,
    use_hardware_qrng=False,          # Use QRNG simulator
    use_hardware_laser=True,          # Use hardware laser
    use_hardware_voa=False,           # Use VOA simulator  
    use_hardware_polarization=True,   # Use hardware polarization
    com_port_polarization="COM3",
    laser_repetition_rate_hz=1000000
)

alice = AliceCPUGeneral(config)
alice.start_transmission()
```

### **Full Hardware Mode:**

```python
config = SimulationConfig(
    pulses_total=1000,
    use_hardware_qrng=True,           # Use hardware QRNG
    use_hardware_laser=True,          # Use hardware laser
    use_hardware_voa=True,            # Use hardware VOA
    use_hardware_polarization=True,   # Use hardware polarization
    com_port_polarization="COM3"
)

alice = AliceCPUGeneral(config)
alice.start_transmission()
```

## **🏗️ Architecture Highlights:**

1. **Component Independence**: Each component can be hardware or simulator independently
2. **Queue-Based Processing**: Similar to Claude version with proper threading
3. **Comprehensive Configuration**: All parameters configurable through `SimulationConfig`
4. **Error Handling**: Robust error handling and logging
5. **Data Collection**: Complete transmission data and statistics
6. **Calibration Support**: System calibration for all components

## How to use AliceCPU

- **`AliceCPU`**: Simple hardware testing version (laser + polarization hardware, QRNG simulator, no VOA)

## **🎯 Key Features 2:**

### **1. Two Operation Modes:**

- **BATCH Mode**: Pre-generates all random bits before starting transmission
- **STREAMING Mode**: Generates random bits on-demand for each pulse

### **2. Hardware/Simulator Support:**

- **Laser Controller**: Uses either `DigitalHardwareLaserDriver` (hardware) or `SimulatedLaserDriver` (simulation)
- **Polarization Controller**: Uses either `PolarizationHardware` or `PolarizationSimulator`
- **QRNG**: Always uses `QRNGSimulator` (as requested)
- **No VOA**: Completely excluded as per your requirements

### **3. Threading Architecture:**

- Main transmission runs in a separate thread
- Proper timing control to maintain 1-second pulse periods
- Includes rotation time and laser firing time in the processing budget
- Thread-safe pause/resume functionality

### **4. Comprehensive Configuration:**

```python
@dataclass
class AliceConfig:
    num_pulses: int = 1000
    pulse_period_seconds: float = 1.0
    use_hardware: bool = False
    com_port: Optional[str] = None  # For polarization hardware
    qrng_seed: Optional[int] = None
    mode: AliceMode = AliceMode.STREAMING
    laser_repetition_rate_hz: float = 1000000  # 1 MHz for hardware laser
```

### **5. Data Collection & Statistics:**

- Real-time statistics tracking
- Transmission data recording
- Performance metrics (rotation times, laser times)
- Error tracking and logging

### **6. Control Methods:**

- `start_transmission()`: Start QKD transmission
- `stop_transmission()`: Stop transmission
- `pause_transmission()`: Pause transmission
- `resume_transmission()`: Resume transmission
- `get_progress()`: Get transmission progress percentage

## **Usage Example:**

```python
# For hardware setup
config = AliceConfig(
    num_pulses=1000,
    pulse_period_seconds=1.0,
    use_hardware=True,
    com_port="COM3",  # For polarization controller
    mode=AliceMode.BATCH,  # or AliceMode.STREAMING
    qrng_seed=42
)

alice = AliceCPU(config)
alice.start_transmission()

# Monitor progress
while alice.is_running():
    print(f"Progress: {alice.get_progress():.1f}%")
    time.sleep(1)

alice.stop_transmission()
```

## **Process Flow:**

1. **Initialize**: Set up all components (laser, polarization, QRNG)
2. **Generate/Get Random Bits**: Either from pre-generated batch or on-demand
3. **Rotate Polarization**: Set the polarization state based on basis and bit
4. **Fire Laser**: Send pulse with correct polarization
5. **Record Data**: Store transmission data for analysis
6. **Timing Control**: Maintain 1-second periods including processing time

The implementation is production-ready and includes proper error handling, logging, and resource cleanup.
