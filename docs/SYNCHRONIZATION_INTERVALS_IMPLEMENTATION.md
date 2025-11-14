# Synchronization Intervals Implementation

## Overview

This document describes the implementation of periodic synchronization intervals to prevent timing drift during long QKD runs.

## Problem Statement

During long quantum key distribution sessions, clock drift between Alice and Bob causes progressive misalignment:

- Short runs (~100 pulses): QBER ~2% (expected)
- Long runs (>1000 pulses): QBER increases over time
- Eventually: QBER ~50% (random detection, complete misalignment)

## Solution

Implement periodic synchronization intervals where Alice and Bob exchange handshake messages to realign their clocks and reset timing references.

## Implementation Details

### 1. Configuration Changes

#### AliceConfig and BobConfig

Added new parameter:

```python
sync_interval: Optional[int] = None  # Resync every X pulses (None = no resyncing)
```

**Recommended values:**

- `None`: Disable synchronization (original behavior)
- `100-200`: For high precision requirements or unstable clocks
- `200-500`: Standard use case (good balance)
- `500+`: For stable systems with minimal drift

### 2. Protocol Markers

Added new synchronization marker to both Alice and Bob:

```python
self._sync_marker = 175  # New marker for periodic synchronization
```

### 3. Transmission Loop Changes (Alice)

Alice's transmission is now split into intervals:

1. **Interval-based transmission**: Pulses are sent in chunks of `sync_interval` size
2. **Sync handshake**: After each interval (except the first), Alice:
   - Sends sync marker (`_sync_marker = 175`)
   - Waits for acknowledgment from Bob
   - Small delay to ensure Bob is ready
3. **Fresh timing reference**: Each interval starts with a new time reference to prevent drift accumulation

**Key methods:**

- `_synchronized_transmission_loop()`: Main loop with intervals
- `_continuous_transmission_loop()`: Original behavior (no sync)
- `_transmit_pulse_interval()`: Transmit one interval of pulses
- `_send_single_pulse()`: Send a single quantum pulse

### 4. Reception Loop Changes (Bob)

Bob's reception is coordinated with Alice's intervals:

1. **Interval-based measurement**: Bob measures in chunks aligned with Alice
2. **Sync handshake**: Before each interval (except the first), Bob:
   - Waits for sync marker from Alice
   - Acknowledges with `_acknowledge_marker`
   - Prepares for next measurement interval
3. **Data accumulation**: Results from all intervals are combined with correct pulse IDs

**Key methods:**

- `_synchronized_reception_loop()`: Main loop with intervals
- `_continuous_reception_loop()`: Original behavior (no sync)
- `_measure_pulse_interval()`: Measure one interval of pulses
- `set_interval_callback()`: Register callback for GUI updates

### 5. GUI Updates

#### Alice GUI

- Added "Synchronization" section with:
  - Checkbox: "Enable Sync Intervals"
  - Input: "Sync Every (pulses)"
- Config creation includes sync_interval parameter

#### Bob GUI

- Added "Synchronization" section (matching Alice)
- Interval callback system for live updates:
  - `on_interval_complete()`: Called after each interval
  - Updates progress bar
  - Shows detection statistics per interval
  - Logs interval completion messages
- Live view updates after each interval (not just at the end)

### 6. Timing Mechanism

**Without synchronization:**

```text
Alice: [Pulse 0] [Pulse 1] [Pulse 2] ... [Pulse N]
Bob:   [Measure continuously for N * pulse_period seconds]
       └─ Extract pulse-by-pulse data from time bins
```

**With synchronization (interval=3):**

```text
Alice: [P0 P1 P2] --SYNC-- [P3 P4 P5] --SYNC-- [P6 P7 P8]
Bob:   [Measure 3] --ACK--  [Measure 3] --ACK--  [Measure 3]
       └─ Each measurement starts fresh, preventing drift accumulation
```

### 7. Pulse ID Management

Critical: Pulse IDs must be consistent across intervals!

**Alice:**

- Global pulse_id: 0 to num_pulses-1
- Relative timing uses interval start time
- Results stored with global pulse_id

**Bob:**

- Receives pulses with correct global pulse_ids
- Relative measurement index within interval
- Results stored with global pulse_id matching Alice

Example with sync_interval=200, total=600:

- Interval 0: pulses 0-199 (relative 0-199)
- Interval 1: pulses 200-399 (relative 0-199)
- Interval 2: pulses 400-599 (relative 0-199)

## Usage Examples

### Example 1: Enable synchronization for 1000 pulses

```python
config = AliceConfig(
    num_pulses=1000,
    pulse_period_seconds=0.5,
    sync_interval=200,  # Resync every 200 pulses
    # ... other parameters
)
```

### Example 2: Disable synchronization (original behavior)

```python
config = AliceConfig(
    num_pulses=1000,
    pulse_period_seconds=0.5,
    sync_interval=None,  # No synchronization
    # ... other parameters
)
```

### Example 3: Using callback for live monitoring (Bob)

```python
def my_interval_callback(interval_idx, pulse_start, pulse_end):
    print(f"Completed interval {interval_idx}: pulses {pulse_start}-{pulse_end-1}")
    # Update GUI, calculate stats, etc.

bob_cpu = BobCPU(config)
bob_cpu.set_interval_callback(my_interval_callback)
bob_cpu.run_complete_qkd_protocol()
```

## Benefits

1. **Prevents timing drift**: Regular resynchronization eliminates accumulated clock errors
2. **Maintains low QBER**: Keeps quantum bit error rate consistent throughout long runs
3. **Live monitoring**: Bob GUI shows statistics after each interval
4. **Backward compatible**: Setting `sync_interval=None` preserves original behavior
5. **Flexible intervals**: Adjustable based on system requirements and clock stability
6. **Better diagnostics**: Interval-by-interval stats help identify issues

## Testing Recommendations

1. **Short run without sync** (baseline):
   - num_pulses=100, sync_interval=None
   - Expected QBER: ~2%

2. **Long run without sync** (demonstrate problem):
   - num_pulses=2000, sync_interval=None
   - Expected: QBER increases over time

3. **Long run with sync** (solution):
   - num_pulses=2000, sync_interval=200
   - Expected: QBER remains consistent ~2%

4. **Very long run with sync**:
   - num_pulses=5000+, sync_interval=200-300
   - Monitor QBER stability across all intervals

## Performance Considerations

- **Sync overhead**: ~0.2-0.5 seconds per sync handshake
- **For 1000 pulses with sync_interval=200**: 5 sync points = ~1-2.5s overhead
- **Trade-off**: Small overhead vs. maintained accuracy in long runs
- **Recommended**: Use larger intervals (300-500) if clock drift is minimal

## Future Enhancements

1. **Adaptive intervals**: Automatically adjust interval size based on measured drift
2. **Drift monitoring**: Track and report timing drift statistics
3. **Dynamic sync**: Only sync when drift exceeds threshold
4. **Clock synchronization**: Use NTP or PTP for initial clock alignment

## Related Files

- `src/alice/aliceCPU.py`: Alice implementation
- `src/bob/bobCPU.py`: Bob implementation
- `examples/alice/alice_qkd_main_gui.py`: Alice GUI
- `examples/bob/bob_qkd_main_gui.py`: Bob GUI
