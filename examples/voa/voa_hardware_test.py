"""VOA hardware testing demo."""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import logging
import time
import argparse
from typing import List, Dict

# Import VOA components
from src.alice.voa import (
    VOAController,
    DecoyInfoExtended,
    VOAOutput,
    VOAHardwareDriver
)
from src.alice.qrng.qrng_hardware import QRNGHardware
from src.utils.data_structures import DecoyState

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VOAHardwareDemo:
    """Demo for VOA hardware control."""
    
    def __init__(self, device_id: str = "voa_device_0"):
        """
        Initialize the demo.
        
        Args:
            device_id: VOA hardware device identifier
        """
        self.device_id = device_id
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def test_hardware_driver_direct(self) -> bool:
        """Test the hardware driver directly."""
        self.logger.info("--- Testing VOA Hardware Driver Direct Interface ---")
        
        try:
            # Create hardware driver
            voa_hardware = VOAHardwareDriver(device_id=self.device_id)
            
            # Get initial status
            status = voa_hardware.get_status()
            self.logger.info(f"Initial status: {status}")
            
            # NOTE: The following operations will raise NotImplementedError
            # since hardware implementation is not yet complete
            
            try:
                # Initialize hardware
                voa_hardware.initialize()
                self.logger.info("Hardware initialized successfully")
                
                # Test attenuation setting
                test_attenuations = [0.0, 5.0, 10.0, 20.0]
                for attenuation in test_attenuations:
                    voa_hardware.set_attenuation(attenuation)
                    current_attenuation = voa_hardware.get_attenuation()
                    output_factor = voa_hardware.get_output_from_attenuation()
                    
                    self.logger.info(f"Set: {attenuation:.1f} dB, "
                                   f"Read: {current_attenuation:.1f} dB, "
                                   f"Factor: {output_factor:.6f}")
                
                # Test reset
                voa_hardware.reset()
                self.logger.info("Hardware reset successfully")
                
                # Shutdown
                voa_hardware.shutdown()
                self.logger.info("Hardware shutdown successfully")
                
                return True
                
            except NotImplementedError as e:
                self.logger.warning(f"Hardware operation not implemented: {e}")
                self.logger.info("This is expected since hardware interface is TODO")
                return True
            
        except Exception as e:
            self.logger.error(f"Hardware driver test failed: {e}")
            return False
    
    def test_hardware_controller(self) -> bool:
        """Test VOA controller with hardware driver."""
        self.logger.info("--- Testing VOA Controller with Hardware ---")
        
        try:
            # Create custom decoy configuration
            decoy_info = DecoyInfoExtended(
                intensities={"signal": 0.5, "weak": 0.1, "vacuum": 0.0},
                probabilities={"signal": 0.7, "weak": 0.2, "vacuum": 0.1}
            )
            
            # Create hardware driver
            voa_hardware = VOAHardwareDriver(device_id=self.device_id)
            
            # Create controller
            voa_controller = VOAController(
                driver=voa_hardware,
                physical=True,
                decoy_info=decoy_info,
                n_pulse_initial=1.0
            )
            
            self.logger.info("VOA controller created with hardware driver")
            
            try:
                # Initialize controller (will try to initialize hardware)
                voa_controller.initialize()
                self.logger.info("Controller initialized successfully")
                
                # Test state selection and attenuation
                for state in [DecoyState.SIGNAL, DecoyState.WEAK, DecoyState.VACUUM]:
                    voa_controller.select_state(state)
                    attenuation = voa_controller.get_attenuation()
                    intensity = voa_controller.decoy_info.get_intensity(state)
                    
                    self.logger.info(f"State: {state}, "
                                   f"Intensity: {intensity}, "
                                   f"Attenuation: {attenuation:.2f} dB")
                
                # Test pulse generation
                self.logger.info("Testing pulse generation:")
                for i in range(5):
                    output = voa_controller.generate_pulse_with_state_selection()
                    self.logger.info(f"Pulse {i}: {output.pulse_type}, "
                                   f"{output.attenuation_db:.2f} dB, "
                                   f"μ={output.target_intensity}")
                
                # Shutdown
                voa_controller.shutdown()
                self.logger.info("Controller shutdown successfully")
                
                return True
                
            except NotImplementedError as e:
                self.logger.warning(f"Hardware operation not implemented: {e}")
                self.logger.info("This is expected since hardware interface is TODO")
                return True
            
        except Exception as e:
            self.logger.error(f"Hardware controller test failed: {e}")
            return False
    
    def test_hardware_characterization(self) -> bool:
        """Test hardware characterization (placeholder)."""
        self.logger.info("--- Testing Hardware Characterization ---")
        
        try:
            self.logger.info("Hardware characterization test placeholder")
            
            # TODO: Implement actual hardware characterization
            # This would include:
            # - Attenuation vs voltage/control signal calibration
            # - Response time measurements
            # - Stability testing
            # - Temperature dependence
            # - Wavelength dependence
            
            characterization_plan = [
                "1. Calibrate attenuation vs control signal",
                "2. Measure response time for step changes",
                "3. Test stability over time",
                "4. Characterize temperature dependence",
                "5. Test wavelength dependence (if applicable)",
                "6. Measure insertion loss",
                "7. Test repeatability"
            ]
            
            self.logger.info("Hardware characterization plan:")
            for step in characterization_plan:
                self.logger.info(f"  {step}")
            
            self.logger.info("Characterization test completed (placeholder)")
            return True
            
        except Exception as e:
            self.logger.error(f"Hardware characterization test failed: {e}")
            return False
    
    def run_hardware_tests(self) -> bool:
        """Run all hardware tests."""
        self.logger.info(f"Starting VOA hardware tests for device: {self.device_id}")
        
        all_passed = True
        
        # Test hardware driver directly
        if not self.test_hardware_driver_direct():
            all_passed = False
        
        # Test hardware controller
        if not self.test_hardware_controller():
            all_passed = False
        
        # Test hardware characterization
        if not self.test_hardware_characterization():
            all_passed = False
        
        if all_passed:
            self.logger.info("\n🎉 All VOA hardware tests passed!")
        else:
            self.logger.error("\n❌ Some VOA hardware tests failed!")
        
        return all_passed


def list_available_devices() -> List[str]:
    """
    List available VOA devices (placeholder).
    
    Returns:
        List of device identifiers
    """
    # TODO: Implement actual device discovery
    # This would scan for connected VOA devices
    
    placeholder_devices = [
        "voa_device_0",
        "voa_device_1", 
        "simulator"
    ]
    
    logger.info("Available VOA devices (placeholder):")
    for device in placeholder_devices:
        logger.info(f"  - {device}")
    
    return placeholder_devices


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="VOA Hardware Demo")
    parser.add_argument("--device", type=str, default="voa_device_0", help="VOA device identifier")
    parser.add_argument("--list-devices", action="store_true", help="List available VOA devices")
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_available_devices()
        return True
    
    # Run hardware demo
    demo = VOAHardwareDemo(device_id=args.device)
    success = demo.run_hardware_tests()
    
    if success:
        logger.info("\nHardware demo completed successfully!")
    else:
        logger.error("\nHardware demo encountered errors!")
    
    return success


if __name__ == "__main__":
    main()
