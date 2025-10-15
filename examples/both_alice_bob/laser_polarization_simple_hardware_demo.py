"""Simple demo for testing laser synchronized with polarization multiple states."""
import sys
import os


# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import logging
import time
from typing import List

# Import required components
from src.alice.laser.hardware_laser.digilent_digital_interface import DigilentDigitalInterface
from src.alice.polarization.polarization_controller import create_polarization_controller_with_hardware
from src.alice.polarization import PolarizationState

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# def run(states, period_ms: int, device_index: int, digital_channel: int,
#         duty_cycle: float = 0.5, frequency_hz: float = 1000.0, idle_low: bool = True) -> int:
#     """
#     :param states: sequence of integers (labels only; no coupling to polarization controller)
#     :param period_ms: delay between laser triggers for successive states
#     :param device_index: device index for Digilent interface (-1 = auto)
#     :param digital_channel: digital channel index on the device
#     :param duty_cycle: PWM duty cycle for pulse generation (if used by your hardware)
#     :param frequency_hz: PWM frequency (Hz)
#     :param idle_low: whether the digital line idles low (False) or high (True)
#     """

def test_laser_polarization_sync(com_port: str, states_input = None, period_ms: int = 1000, stepper_freq_hz: int = 500, device_index: int = -1, digital_channel: int = 8) -> bool:
    """Test laser synchronized with polarization multiple states.
    Args:
        com_port (str): The COM port to use for the polarization controller.
        states_input (list, optional): List of polarization states to test.
        period_ms (int, optional): Period between states in milliseconds.
        stepper_freq_hz (int, optional): Stepper frequency in Hz.
        device_index (int, optional): Device index for the laser interface.
        digital_channel (int, optional): Digital channel for the laser interface.
    """
    logger.info(f"\n🚀 Testing Laser + Polarization Synchronization ({com_port})")
    
    try:
        # Create polarization controller with hardware
        with create_polarization_controller_with_hardware(com_port=com_port) as controller:
            
            if not controller.is_initialized():
                logger.warning("Trying to initialize polarization controller...")
                controller.initialize()
                if not controller.is_initialized():
                    logger.error("Failed to initialize polarization controller")
                    return False
            
            logger.info("✅ Polarization controller initialized")
            
            # Initialize laser interface
            logger.info("\n🔌 Initializing laser interface...")
            interface = DigilentDigitalInterface(device_index=device_index, digital_channel=digital_channel)
            success = interface.connect()
            
            if not success:
                logger.error("❌ Failed to connect to laser interface")
                return False
                
            logger.info("✅ Laser interface connected")
            
            # Set laser pulse parameters
            interface.set_pulse_parameters(
                duty_cycle=0.5,      # 50% duty cycle
                frequency=1000.0,    # 1 kHz base frequency (pulse width 1 ms * duty cycle)
                idle_state=False     # Idle low
            )
            logger.info("✅ Laser pulse parameters configured")
            

            #############################################################
            # Test single fire at a time
            logger.info("\n🔧 Testing single laser firing after single controller:")
            for state in states:
                controller.driver.set_operation_period(period_ms)
                controller.driver.set_stepper_frequency(stepper_freq_hz)
                start_time = time.time()
                controller.set_polarization_state(PolarizationState(state))
                while not controller.is_available():
                    time.sleep(0.01)  # Small sleep to avoid busy waiting
                    # logger.info(f"  ⏳ Waiting for controller - State: {state_names[state]}")
                logger.info(f"✅ Controller ready after {time.time() - start_time:.2f}s")
                logger.info(f"  🔥 Firing laser - State: {state_names[state]}")
                laser_success = interface.trigger_laser(mode="single")
                result_emoji = "✅" if laser_success else "❌"
                logger.info(f"    {result_emoji} Laser result: {'Success' if laser_success else 'Failed'}")

            logger.info("✅ Single laser firing test completed")
            #############################################################
            # Test multiple polarization states synchronized with laser
            logger.info("\n🔧 Testing synchronized laser + polarization states:")
            
            # Define test sequence
            # states = [0,1,2,3,0,3,1,3,1,3,1,2,3,0,3,1,3,1,3,1,2,3,0,3,1,3,1,3]
            states = [0, 1, 2, 3, 1]
            if states_input is not None:
                states = states_input
            period = period_ms if period_ms is not None else 1000  # 1000 ms period between states
            stepper_freq = stepper_freq_hz if stepper_freq_hz is not None else 500  # 500 Hz stepper frequency


            state_names = {0: "H", 1: "V", 2: "D", 3: "A"}
            logger.info(f"  Sequence: {[state_names[s] for s in states]}")
            logger.info(f"  Period: {period} ms, Stepper freq: {stepper_freq} Hz")
            
            # Start polarization sequence
            success = controller.set_polarization_multiple_states(
                states, 
                period=period, 
                stepper_freq=stepper_freq
            )
            logger.info(f"  Set multiple states {states} with period {period} ms and freq {stepper_freq} Hz: {'✅ Success' if success else '❌ Failed'}")

                
            logger.info("✅ Polarization sequence started")
            
            # Synchronize laser pulses with polarization states
            start_time = time.time()
            count = 1
            
            logger.info("\n⚡ Starting synchronized laser firing:")
            
            while not controller.is_available():
                time.sleep(0.01)  # Small sleep to avoid busy waiting
                
                # Check if it's time to fire the laser
                elapsed_time = time.time() - start_time
                
                if elapsed_time > (period / 1000) * count:
                    # Add slight delay to ensure polarization is stable
                    time.sleep(period / 1000 / 2)  
                    
                    current_state = states[count - 1] if count <= len(states) else states[-1]
                    state_name = state_names[current_state]
                    
                    logger.info(f"  🔥 Firing laser at {elapsed_time:.2f}s - State: {state_name}")
                    
                    # Fire laser
                    laser_success = interface.trigger_laser(mode="single")
                    result_emoji = "✅" if laser_success else "❌"
                    logger.info(f"    {result_emoji} Laser result: {'Success' if laser_success else 'Failed'}")
                    
                    count += 1
                    
                    # Stop if we've fired for all states
                    if count > len(states):
                        break
            
            total_time = time.time() - start_time
            logger.info(f"\n✅ Sequence completed in {total_time:.2f} seconds")
            logger.info("✅ Device is now available")
            
            # Cleanup
            interface.disconnect()
            logger.info("✅ Laser interface disconnected")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Laser polarization sync test failed: {e}")
        return False
    
def list_available_com_ports() -> List[str]:
    """List available COM ports."""
    try:
        import serial.tools.list_ports
        ports = [port.device for port in serial.tools.list_ports.comports()]
        return ports
    except ImportError:
        logger.warning("pyserial not available, cannot list COM ports")
        return []


def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Laser + Polarization Synchronization Demo")
    parser.add_argument("--com-port", "--cp", type=str, required=True, 
                       help="COM port for polarization controller (e.g., COM4)")
    parser.add_argument("--list-com-ports", "--lcp", action="store_true", 
                       help="List available COM ports")

    args = parser.parse_args()
    
    print("Laser + Polarization Synchronization Demo")
    print("=" * 45)
        
    if args.list_com_ports:
        available_ports = list_available_com_ports()
        if available_ports:
            print("Available COM ports:")
            for port in available_ports:
                print(f"  - {port}")
        else:
            print("No COM ports available.")
        return 0
    
    if not args.com_port:
        print("❌ COM port must be specified.")
        return 1
    
    success = test_laser_polarization_sync(args.com_port)
    
    if success:
        print("\n🎉 Demo completed successfully!")
        return 0
    else:
        print("\n💥 Demo failed!")
        return 1


if __name__ == "__main__":
    exit(main())
