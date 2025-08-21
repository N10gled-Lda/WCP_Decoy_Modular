"""
Example usage of the enhanced VOA controller with QRNG integration and probability-based state selection.

This script demonstrates:
1. Setting up intensity and probability configuration using DecoyInfo
2. Using QRNG to randomly select decoy states with custom probabilities
3. Calculating attenuations based on target intensities
4. Generating pulses with both probability-based and uniform selection
5. Comparing different selection methods
"""

from multiprocessing import Queue
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging
from src.alice.qrng.qrng_simulator import OperationMode
from src.alice.voa.voa_controller import (
    VOAController, DecoyInfoExtended, VOAOutput
)
from src.alice.voa.voa_simulator import VOASimulator
from src.utils.data_structures import DecoyState

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate VOA controller functionality with probabilities."""
    
    # Example 1: VOA controller with custom DecoyInfo configuration
    print("\n=== Example 1: VOA Controller with Custom DecoyInfo ===")

    # Create custom DecoyInfo configuration
    custom_decoy_info = DecoyInfoExtended(
        intensities={"signal": 0.5, "weak": 0.1, "vacuum": 0.001},
        probabilities={"signal": 0.7, "weak": 0.2, "vacuum": 0.1}
    )
    
    pulse_queue = Queue()
    attenuated_pulse_queue = Queue()
    voa_custom = VOAController(
        driver=VOASimulator(pulse_queue, attenuated_pulse_queue),
        physical=False, 
        decoy_info=custom_decoy_info
    )
    
    # Initialize the controller
    voa_custom.initialize()
    
    print(f"Custom intensities: {voa_custom.decoy_info.intensities}")
    print(f"Custom probabilities: {voa_custom.decoy_info.probabilities}")

    # Example 2: Manual state selection with automatic attenuation
    print("\n=== Example 2: Manual State Selection ===")
    for state in [DecoyState.SIGNAL, DecoyState.WEAK, DecoyState.VACUUM]:
        voa_custom.set_state(state)
        attenuation = voa_custom.get_attenuation()
        target_mu = custom_decoy_info.get_intensity(state)
        probability = custom_decoy_info.get_probability(state)
        print(f"State: {state:6} | Target μ: {target_mu:4.1f} | "
              f"Probability: {probability:4.1f} | Attenuation: {attenuation:5.2f} dB")

    # Example 3: Probability-based random state selection
    print("\n=== Example 3: Probability-based State Selection ===")
    print("Generating 15 pulses with probability-based selection:")
    print("Pulse | State    | Target μ | Attenuation | Expected %")
    print("------|----------|----------|-------------|----------")
    
    for i in range(15):
        output = voa_custom.generate_pulse_with_state_selection(use_probabilities=True)
        expected_pct = custom_decoy_info.get_probability(output.pulse_type) * 100
        
        print(f"  {i+1:2d}  | {output.pulse_type:8} | "
              f"{output.target_intensity:6.1f}   | {output.attenuation_db:8.2f} dB | "
              f"{expected_pct:6.1f}%")

    # Example 4: Uniform random state selection (bit mapping)
    print("\n=== Example 4: Uniform State Selection (Bit Mapping) ===")
    print("Generating 12 pulses with uniform selection:")
    print("Pulse | Bits | State    | Target μ | Attenuation")
    print("------|------|----------|----------|------------")
    
    for i in range(12):
        # Get bits for demonstration
        bit1 = voa_custom.qrng_driver.get_random_bit(mode=OperationMode.STREAMING)
        bit2 = voa_custom.qrng_driver.get_random_bit(mode=OperationMode.STREAMING)

        state = voa_custom.get_state_from_bits(bit1, bit2)

        print(f"  {i+1:2d}  | {bit1}{bit2}   | {voa_custom.get_current_state():8} | "
              f"{voa_custom.get_current_target_intensity():6.1f}   | {voa_custom.get_current_attenuation():8.2f} dB")

    # Example 5: Comparison of selection methods
    print("\n=== Example 5: Selection Method Comparison ===")
    n_samples = 1000
    
    # Probability-based selection
    prob_counts = {DecoyState.SIGNAL: 0, DecoyState.WEAK: 0, DecoyState.VACUUM: 0}
    for _ in range(n_samples):
        output = voa_custom.generate_pulse_with_state_selection(use_probabilities=True)
        prob_counts[output.pulse_type] += 1
    
    # Uniform selection
    uniform_counts = {DecoyState.SIGNAL: 0, DecoyState.WEAK: 0, DecoyState.VACUUM: 0}
    for _ in range(n_samples):
        output = voa_custom.generate_pulse_with_state_selection(use_probabilities=False)
        uniform_counts[output.pulse_type] += 1
    
    print(f"Distribution comparison over {n_samples} samples:")
    print("State    | Expected | Prob-based | Uniform   ")
    print("---------|----------|------------|--------   ")
    
    for state in [DecoyState.SIGNAL, DecoyState.WEAK, DecoyState.VACUUM]:
        expected_pct = custom_decoy_info.get_probability(state) * 100
        prob_pct = (prob_counts[state] / n_samples) * 100
        uniform_pct = (uniform_counts[state] / n_samples) * 100
        
        print(f"{state:8} | {expected_pct:6.1f}%   | {prob_pct:8.1f}%  | {uniform_pct:7.1f}%")

    # Example 6: Dynamic configuration updates
    print("\n=== Example 6: Dynamic Configuration Updates ===")

    # Create different decoy configurations
    configs = [
        ("Standard", {"signal": 0.7, "weak": 0.2, "vacuum": 0.1}),
        ("High Signal", {"signal": 0.9, "weak": 0.08, "vacuum": 0.02}),
        ("Equal", {"signal": 0.33, "weak": 0.33, "vacuum": 0.34})
    ]
    
    for config_name, prob_dict in configs:
        # Update probabilities
        decoy_info = DecoyInfoExtended(
            intensities={"signal": 0.5, "weak": 0.1, "vacuum": 0.0},
            probabilities=prob_dict
        )
        
        pulse_queue = Queue()
        attenuated_pulse_queue = Queue()
        voa_temp = VOAController(
            driver=VOASimulator(pulse_queue, attenuated_pulse_queue),
            physical=False,
            decoy_info=decoy_info
        )
        
        # Initialize the controller
        voa_temp.initialize()
        
        # Generate samples
        counts = {DecoyState.SIGNAL: 0, DecoyState.WEAK: 0, DecoyState.VACUUM: 0}
        for _ in range(300):
            output = voa_temp.generate_pulse_with_state_selection(use_probabilities=True)
            counts[output.pulse_type] += 1
        
        print(f"\n{config_name} Config:")
        print(f"  Probabilities: {prob_dict}")
        for state, count in counts.items():
            percentage = (count / 300) * 100
            expected = decoy_info.get_probability(state) * 100
            print(f"  {state:8}: {count:3d} ({percentage:5.1f}%, expected {expected:4.1f}%)")

    # Example 7: Individual parameter updates
    print("\n=== Example 7: Individual Parameter Updates ===")

    pulse_queue = Queue()
    attenuated_pulse_queue = Queue()
    voa_update = VOAController(driver=VOASimulator(pulse_queue, attenuated_pulse_queue), physical=False)
    # Initialize the controller
    voa_update.initialize()
    
    print(f"Initial intensities: {voa_update.decoy_info.intensities}")
    print(f"Initial probabilities: {voa_update.decoy_info.probabilities}")
    
    # Update individual intensities
    voa_update.update_intensity(DecoyState.SIGNAL, 0.8)
    voa_update.update_intensity(DecoyState.WEAK, 0.15)
    
    # Update probabilities together to ensure they sum to 1.0
    voa_update.update_probabilities(signal=0.6, weak=0.3, vacuum=0.1)
    
    print(f"Updated intensities: {voa_update.decoy_info.intensities}")
    print(f"Updated probabilities: {voa_update.decoy_info.probabilities}")

    # Example 8: Attenuation calculation for different initial photon numbers
    print("\n=== Example 8: Attenuation vs Initial Photon Number ===")
    target_mu = 0.1  # Weak decoy state
    print(f"Target μ = {target_mu}")
    print("N_pulse | Attenuation")
    print("--------|------------")
    
    for n_pulse in [0.5, 1.0, 2.0, 5.0, 10.0]:
        voa_temp = VOAController(
            driver=VOASimulator(pulse_queue, attenuated_pulse_queue),
            physical=False
        )
        # Initialize the controller
        voa_temp.initialize()
        attenuation = voa_temp.calculate_attenuation_for_intensity(target_mu, n_pulse)
        print(f"  {n_pulse:4.1f}  | {attenuation:8.2f} dB")


if __name__ == "__main__":
    main()
