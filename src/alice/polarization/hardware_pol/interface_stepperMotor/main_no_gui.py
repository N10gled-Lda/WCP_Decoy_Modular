from time import sleep
from imports.stm32_interface import STM32Interface

def main():
    """
    Demo of STM32 Interface without GUI
    This example demonstrates all available commands:
    - Connection management
    - Polarization control (numbers and device)
    - Angle setting (with offset option)
    - Frequency control (stepper motor and operation period)
    """
    
    # 1. Create and configure the interface
    print("=== STM32 Interface Demo (No GUI) ===")
    print("Initializing STM32 interface on COM8...")
    
    stm = STM32Interface("COM8")
    
    # 2. Define the callback functions
    def handle_connected():
        print("✓ STM32 is connected and ready!")
        print("  Device is now ready to receive commands.")
    
    def handle_available():
        print("✓ STM32 is available for new commands.")
        print("  Ready to process next command...")
    
    # 3. Attach the callbacks
    stm.on_connected = handle_connected
    stm.on_available = handle_available
    
    # 4. Start the interface and initiate connection
    print("Starting STM32 interface threads...")
    stm.start()
    
    print("Attempting to connect to STM32...")
    stm.connect()
    
    # Wait for connection
    print("Waiting for connection...")
    connection_timeout = 10  # seconds
    for i in range(connection_timeout):
        if stm.connected:
            break
        sleep(1)
        print(f"  Waiting... ({i+1}/{connection_timeout})")
    
    if not stm.connected:
        print("❌ Failed to connect to STM32. Check COM port and device.")
        stm.stop()
        return
    
    try:
        # Demo sequence of commands
        print("\n=== Starting Command Demo Sequence ===")
        
        # 5. Test polarization numbers
        print("\n1. Testing Polarization Numbers...")
        polarization_values = [0, 1, 2, 3, 0, 2, 1, 3]  # Example sequence
        print(f"   Sending polarization sequence: {polarization_values}")
        
        success = stm.send_cmd_polarization_numbers(polarization_values)
        if success:
            print("   ✓ Polarization numbers command sent successfully")
        else:
            print("   ❌ Failed to send polarization numbers")
        
        sleep(2)  # Wait for processing
        
        # 6. Test polarization device setting
        print("\n2. Testing Polarization Device Settings...")
        
        # Set to Linear Polarizer
        print("   Setting device to Linear Polarizer (1)...")
        success = stm.send_cmd_polarization_device(1)
        if success:
            print("   ✓ Linear Polarizer selected")
        else:
            print("   ❌ Failed to set Linear Polarizer")
        
        sleep(1)
        
        # Set to Half Wave Plate
        print("   Setting device to Half Wave Plate (2)...")
        success = stm.send_cmd_polarization_device(2)
        if success:
            print("   ✓ Half Wave Plate selected")
        else:
            print("   ❌ Failed to set Half Wave Plate")
        
        sleep(1)
        
        # 7. Test angle settings
        print("\n3. Testing Angle Settings...")
        
        # Set absolute angle
        angle = 45
        print(f"   Setting absolute angle to {angle}°...")
        success = stm.send_cmd_set_angle(angle, is_offset=False)
        if success:
            print(f"   ✓ Absolute angle set to {angle}°")
        else:
            print(f"   ❌ Failed to set absolute angle to {angle}°")
        
        sleep(1)
        
        # Set offset angle
        offset = 90
        print(f"   Setting angle offset to {offset}°...")
        success = stm.send_cmd_set_angle(offset, is_offset=True)
        if success:
            print(f"   ✓ Angle offset set to {offset}°")
        else:
            print(f"   ❌ Failed to set angle offset to {offset}°")
        
        sleep(1)
        
        # 8. Test frequency settings
        print("\n4. Testing Frequency Settings...")
        
        # Set stepper motor frequency
        stepper_freq = 100  # Hz
        print(f"   Setting stepper motor frequency to {stepper_freq} Hz...")
        success = stm.send_cmd_set_frequency(stepper_freq, is_stepper=True)
        if success:
            print(f"   ✓ Stepper motor frequency set to {stepper_freq} Hz")
        else:
            print(f"   ❌ Failed to set stepper motor frequency to {stepper_freq} Hz")
        
        sleep(1)
        
        # Set operation period
        operation_period = 5000  # ms
        print(f"   Setting operation period to {operation_period} ms...")
        success = stm.send_cmd_set_frequency(operation_period, is_stepper=False)
        if success:
            print(f"   ✓ Operation period set to {operation_period} ms")
        else:
            print(f"   ❌ Failed to set operation period to {operation_period} ms")
        
        sleep(2)
        
        # 9. Test a complete polarization sequence
        print("\n5. Testing Complete Polarization Sequence...")
        
        # BB84 states demonstration
        bb84_sequence = [0, 1, 2, 3]  # H, V, D, A
        state_names = ["H (Horizontal)", "V (Vertical)", "D (Diagonal)", "A (Anti-diagonal)"]
        
        for i, (pol_state, state_name) in enumerate(zip(bb84_sequence, state_names)):
            print(f"   Step {i+1}: Setting to {state_name} (state {pol_state})...")
            success = stm.send_cmd_polarization_numbers([pol_state])
            if success:
                print(f"   ✓ Successfully set to {state_name}")
            else:
                print(f"   ❌ Failed to set to {state_name}")
            sleep(1.5)  # Wait between state changes
        
        print("\n=== Demo Sequence Complete ===")
        print("All commands have been tested. The STM32 should have processed all requests.")
        print("Check the STM32 device for proper operation and responses.")
        
        # 10. Keep running to monitor responses
        print("\n6. Monitoring STM32 Responses...")
        print("   Press Ctrl+C to stop monitoring and exit")
        print("   Monitoring for incoming responses and status updates...")
        
        while True:
            sleep(1)
            # You can add additional monitoring or periodic commands here
            # For example, checking connection status or sending periodic heartbeats
            
    except KeyboardInterrupt:
        print("\n\n=== User Interrupted ===")
        print("Stopping STM32 interface...")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
    finally:
        print("Cleaning up and stopping interface...")
        stm.stop()
        print("✓ STM32 interface stopped successfully")
        print("Demo completed.")

if __name__ == "__main__":
    main()