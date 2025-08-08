from time import sleep
from imports.stm32_interface import *

# 1. Create and configure the interface
stm = STM32Interface("COM8")

# 2. Define the callback functions
def handle_connected():
    print("STM32 is connected.")

def handle_available():
    print("STM32 is available. Sending polarization numbers...")
    success = stm.send_polarization_numbers([0, 1, 2, 3])  # Example polarization values
    if success:
        print("Polarization numbers sent.")
    else:
        print("Failed to send polarization numbers.")

def handle_polarization_status(status):
    print(f"Polarization status received: {status}")
    if status == COMMAND_POLARIZATION_SUCCESS:
        print("Polarization set successfully.")
    else:
        print("Polarization failed.")

# 3. Attach the callbacks
stm.on_connected = handle_connected
stm.on_available = handle_available
stm.on_polarization_status = handle_polarization_status

# 4. Start the interface and initiate connection
stm.start()
stm.connect()

# Optional: wait for a while to let the callbacks do their work
try:
    while True:
        sleep(1)  # Keep the main thread alive to allow callbacks to be executed
except KeyboardInterrupt:
    print("Stopping interface...")
    stm.stop()