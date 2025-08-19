"""Example usage of the improved polarization controller with QRNG integration and queue processing."""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging
from queue import Queue
from src.alice.polarization.polarization_controller import create_polarization_controller_with_simulator
from src.alice.polarization.polarization_base import PolarizationState
from src.utils.data_structures import Basis, Bit, Pulse, LaserInfo

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate polarization controller functionality with queue processing."""
    
    # Create queues for pulse processing
    pulses_queue = Queue()
    polarized_pulses_queue = Queue()
    
    # Create laser info
    laser_info = LaserInfo(
        wavelength=1550.0,  # nm
        power=1.0,  # mW
        pulse_width=1e-9  # 1ns
    )
    
    # Initialize polarization controller with simplified interface
    with create_polarization_controller_with_simulator(
        input_queue=pulses_queue,
        output_queue=polarized_pulses_queue,
        laser_info=laser_info
    ) as pol_controller:
        
        logger.info("=== Polarization Controller Demo with Queue Processing ===")
        
        # 1. Generate random polarization using QRNG
        logger.info("\n1. Random polarization generation:")
        for i in range(3):
            output = pol_controller.set_polarization_from_qrng()
            logger.info(f"  Run {i+1}: {output.basis} basis, bit {output.bit} → "
                       f"{output.polarization_state} ({output.angle_degrees}°)")
        
        # 2. Create some test pulses and add them to the input queue
        logger.info("\n2. Adding test pulses to input queue:")
        test_pulses = [
            Pulse(timestamp=1.0, intensity=1000.0, phase=0.0, wavelength=1550.0, photons=1000),
            Pulse(timestamp=2.0, intensity=1200.0, phase=0.5, wavelength=1550.0, photons=1200),
            Pulse(timestamp=3.0, intensity=800.0, phase=1.0, wavelength=1550.0, photons=800),
        ]
        
        for i, pulse in enumerate(test_pulses):
            pulses_queue.put(pulse)
            logger.info(f"  Added pulse {i+1}: {pulse.photons} photons, intensity {pulse.intensity}")
        
        # 3. Set a specific polarization and process the queue
        logger.info("\n3. Setting polarization to Z basis, bit 1 (Vertical - 90°):")
        pol_controller.set_polarization_manually(Basis.Z, Bit.ONE)
        
        # Process all pulses in the queue
        logger.info("\n4. Processing pulses through polarization controller:")
        pol_controller.apply_polarization_to_queue()
        
        # Check queue status
        queue_info = pol_controller.get_queue_info()
        logger.info(f"  Queue status after processing: {queue_info}")
        
        # 5. Retrieve and examine polarized pulses
        logger.info("\n5. Examining polarized pulses:")
        while not polarized_pulses_queue.empty():
            polarized_pulse = polarized_pulses_queue.get()
            logger.info(f"  Polarized pulse: {polarized_pulse.photons} photons, "
                       f"basis={getattr(polarized_pulse, 'basis', 'N/A')}, "
                       f"bit={getattr(polarized_pulse, 'bit_value', 'N/A')}, "
                       f"angle={getattr(polarized_pulse, 'polarization_angle', 'N/A')}°")
        
        # 6. Test different polarization states with queue processing
        logger.info("\n6. Testing different polarization states:")
        test_states = [
            (Basis.Z, Bit.ZERO, "Horizontal (0°)"),
            (Basis.X, Bit.ZERO, "Diagonal (45°)"),
            (Basis.X, Bit.ONE, "Anti-diagonal (135°)")
        ]
        
        for basis, bit, description in test_states:
            # Add a test pulse
            test_pulse = Pulse(timestamp=4.0, intensity=1000.0, phase=0.0, wavelength=1550.0, photons=1000)
            pulses_queue.put(test_pulse)
            
            # Set polarization
            pol_controller.set_polarization_manually(basis, bit)
            logger.info(f"  Set to {description}")
            
            # Process the pulse
            pol_controller.apply_polarization_to_queue()
            
            # Get the result
            if not polarized_pulses_queue.empty():
                result_pulse = polarized_pulses_queue.get()
                logger.info(f"    Result: basis={getattr(result_pulse, 'basis', 'N/A')}, "
                           f"bit={getattr(result_pulse, 'bit_value', 'N/A')}, "
                           f"angle={getattr(result_pulse, 'polarization_angle', 'N/A')}°")
        
        # 7. Current state information
        logger.info("\n7. Final state information:")
        state_info = pol_controller.get_current_state()
        for key, value in state_info.items():
            if key == 'jones_vector':
                logger.info(f"  {key}: [{value[0]:.3f}, {value[1]:.3f}]")
            else:
                logger.info(f"  {key}: {value}")
        
        logger.info("\nPolarization controller demo completed!")


if __name__ == "__main__":
    main()
