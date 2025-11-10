# Multi-Channel Detection Resolution - Implementation Summary

## Overview

This document describes the implementation of multi-channel detection resolution in the QKD system. Previously, when Bob detected photon counts on multiple channels simultaneously, those detections were discarded. Now, these ambiguous detections are kept and resolved during basis sifting by comparing with Alice's transmitted bases.

## Changes Made

### 1. BobCPU (`src/bob/bobCPU.py`)

#### Modified `_convert_counts_to_detection_data()` method

- **Previous behavior**: Discarded detections when multiple channels had counts
- **New behavior**:
  - Keeps multi-channel detections as valid detection indices
  - Marks basis and bit as `-1` for ambiguous detections
  - Returns additional `ambiguous_detections_map` dictionary mapping detection index to channels with counts
  - Single-channel detections are processed as before with direct channel-to-basis-bit mapping

**Channel to Basis/Bit Mapping**:

- Channel 4 (HORIZONTAL): basis=0, bit=0 (0°)
- Channel 3 (VERTICAL): basis=0, bit=1 (90°)
- Channel 2 (DIAGONAL): basis=1, bit=0 (45°)
- Channel 1 (ANTI_DIAGONAL): basis=1, bit=1 (135°)

#### Modified `run_post_processing()` method

- Now receives and passes `ambiguous_detections_map` to the classical communication layer

### 2. QKDBobImplementation (`src/protocol/qkd_bob_implementation_class.py`)

#### Modified `set_bob_detected_quantum_variables()` method

- Added `ambiguous_detections_map` parameter (optional, defaults to empty dict)
- Stores the map as instance variable `_ambiguous_detections_map`

#### Modified `bob_process_base_sifting_classical_steps()` method

- Passes `ambiguous_detections_map` to the BobThread instance

### 3. BobThread (`src/protocol/BB84/bb84_protocol/bob_side_thread_ccc.py`)

#### Modified `__init__()` method

- Added `_ambiguous_detections_map` instance variable

#### Modified `set_bits_bases_qubits_idxs()` method

- Added `ambiguous_detections_map` parameter
- Updated validation to allow `-1` for ambiguous bases and bits
- Stores the ambiguous detections map
- Logs the number of ambiguous detections received

#### Modified `match_bases()` method

This is where the magic happens! The method now:

1. **Resolves ambiguous detections** before normal basis matching:
   - For each ambiguous detection (basis=-1, bit=-1)
   - Gets Alice's transmitted basis for that pulse
   - Checks which of Bob's detected channels match Alice's basis
   - If **exactly one** channel matches:
     - Resolves the ambiguity by setting basis and bit based on that channel
     - Logs the resolution
   - If **zero or multiple** (2 of the same or different basis) channels match:
     - Keeps the detection as ambiguous (will be discarded in next step)
     - Logs the discard

2. **Performs normal basis sifting**:
   - Compares Bob's (now potentially resolved) bases with Alice's bases
   - Excludes any remaining unresolved ambiguous detections (basis=-1)
   - Creates common indices and common bits lists

## Resolution Logic Examples

### Example 1: Successful Resolution

- **Bob's detection**: Channels 3 and 4 have counts (both basis 0)
- **Alice's basis**: 0 (rectilinear)
- **Bob's channels in basis 0**: Channel 3 (bit=1) and Channel 4 (bit=0)
- **Result**: Cannot resolve uniquely → **DISCARDED** (multiple channels in same basis)

### Example 2: Successful Resolution (Cross-basis)

- **Bob's detection**: Channels 1 and 4 have counts (basis 1 and 0)
- **Alice's basis**: 0 (rectilinear)
- **Bob's channels in basis 0**: Only Channel 4 (bit=0)
- **Result**: **RESOLVED** → basis=0, bit=0

### Example 3: No Match

- **Bob's detection**: Channels 1 and 2 have counts (both basis 1)
- **Alice's basis**: 0 (rectilinear)
- **Bob's channels in basis 0**: None
- **Result**: **DISCARDED** (no matching basis)

## Benefits

1. **Increased key rate**: Multi-channel detections are no longer automatically discarded
2. **Better use of available data**: Ambiguous detections that can be resolved contribute to the key
3. **Maintains security**: Only unambiguous detections (after resolution) are kept
4. **Backwards compatible**: Single-channel detections work exactly as before

## Debug Output

The implementation includes comprehensive debug logging:

- Pulse-by-pulse detection status (single/multi-channel)
- Ambiguous detection count
- Resolution statistics (resolved vs. discarded)
- Detailed channel information for each resolution/discard decision

## Testing Recommendations

1. Test with various multi-channel detection scenarios
2. Verify QBER calculations include/exclude resolved detections appropriately
3. Compare key rates before/after this change
4. Verify security: ensure no information leakage through ambiguous detection resolution
