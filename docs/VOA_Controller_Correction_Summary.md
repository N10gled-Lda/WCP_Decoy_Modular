"""
Corrected VOA Controller Implementation Summary
==============================================

You were absolutely right to point out that there was already a `DecoyInfo` class in `data_structures.py`!
I unnecessarily created new dataclasses when I should have used the existing infrastructure.

## What Was Fixed

### ❌ Previous Implementation (Redundant)
- Created new `IntensityConfig`, `ProbabilityConfig`, and `DecoyConfig` classes
- Duplicated functionality that already existed in `DecoyInfo`
- Added unnecessary complexity and inconsistency

### ✅ Corrected Implementation (Using Existing DecoyInfo)
- Uses the existing `DecoyInfo` class from `data_structures.py`
- Created `DecoyInfoExtended` class that extends `DecoyInfo` with helper methods
- Maintains consistency with the existing codebase architecture

## Key Changes Made

### 1. Removed Redundant Classes
```python
# REMOVED these redundant classes:
# - IntensityConfig
# - ProbabilityConfig  
# - DecoyConfig
```

### 2. Extended Existing DecoyInfo
```python
class DecoyInfoExtended(DecoyInfo):
    """Extended DecoyInfo class with helper methods for state-based access."""
    
    def get_intensity(self, state: DecoyState) -> float:
        """Get the intensity for a given decoy state."""
        state_str = str(state)  # Converts SIGNAL -> "signal"
        return self.intensities[state_str]
    
    def get_probability(self, state: DecoyState) -> float:
        """Get the probability for a given decoy state."""
        state_str = str(state)  # Converts SIGNAL -> "signal"
        return self.probabilities[state_str]
    
    def set_intensity(self, state: DecoyState, intensity: float) -> None:
        """Set the intensity for a given decoy state."""
        state_str = str(state)
        self.intensities[state_str] = intensity
    
    def set_probability(self, state: DecoyState, probability: float) -> None:
        """Set the probability for a given decoy state."""
        state_str = str(state)
        self.probabilities[state_str] = probability
        # Re-validate probabilities sum to 1.0
        if abs(sum(self.probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("Probabilities must sum to 1.0 after modification")
```

### 3. Simplified VOAController
```python
class VOAController:
    def __init__(self, 
                 physical: bool = False,
                 decoy_info: Optional[DecoyInfoExtended] = None,
                 n_pulse_initial: float = 1.0):
        # Uses DecoyInfoExtended instead of multiple config objects
        if decoy_info is not None:
            self.decoy_info = decoy_info
        else:
            # Create default configuration using existing DecoyInfo defaults
            self.decoy_info = DecoyInfoExtended()
```

## Benefits of Using Existing DecoyInfo

### 1. **Consistency**: 
   - Uses the same data structure as the rest of the codebase
   - No duplication of functionality
   - Consistent validation (probabilities sum to 1.0)

### 2. **Simplicity**:
   - Single configuration object instead of multiple classes
   - Leverages existing Pydantic validation
   - Maintains backward compatibility

### 3. **Extensibility**:
   - Can easily add more parameters to DecoyInfo in the future
   - Helper methods provide convenient state-based access
   - Preserves the original dictionary-based interface

## Usage Examples

### Basic Usage with Defaults
```python
# Uses default values from DecoyInfo
voa = VOAController(physical=False)
# Default: intensities={"signal": 0.5, "weak": 0.1, "vacuum": 0.0}
# Default: probabilities={"signal": 0.7, "weak": 0.2, "vacuum": 0.1}
```

### Custom Configuration
```python
# Create custom DecoyInfo
custom_decoy_info = DecoyInfoExtended(
    intensities={"signal": 0.6, "weak": 0.15, "vacuum": 0.0},
    probabilities={"signal": 0.8, "weak": 0.15, "vacuum": 0.05}
)

voa = VOAController(physical=False, decoy_info=custom_decoy_info)
```

### Dynamic Updates
```python
voa = VOAController(physical=False)

# Update individual parameters
voa.update_intensity(DecoyState.SIGNAL, 0.8)
voa.update_probabilities(signal=0.6, weak=0.3, vacuum=0.1)
```

### Pulse Generation with Probabilities
```python
# Probability-based selection (70% signal, 20% weak, 10% vacuum)
output = voa.generate_pulse_with_probability_selection()

# Uniform selection (using 2-bit mapping)
output = voa.generate_pulse_with_uniform_selection()
```

## Key Features Retained

✅ **Probability-based state selection** using QRNG  
✅ **Custom probability distributions** (e.g., 70% signal, 20% weak, 10% vacuum)  
✅ **Automatic attenuation calculation** based on target intensities  
✅ **Hardware/simulation support** with seamless switching  
✅ **Statistical validation** and distribution analysis  
✅ **Multiple selection methods** (probability-based vs uniform)  

## Lessons Learned

1. **Always check existing codebase** before creating new data structures
2. **Extend existing classes** rather than duplicating functionality  
3. **Maintain consistency** with established patterns and naming conventions
4. **Leverage existing validation** and infrastructure when possible

The corrected implementation now properly uses the existing `DecoyInfo` class while adding the requested probability-based state selection functionality. This maintains consistency with your codebase architecture and avoids unnecessary duplication.
"""
