"""
===============================================================================
File Name: main.py
Description:
    Main file where all modules come together to execute functionality.
    This script initializes the WaveForms SDK, configures the AnalogDiscovery3 device,
    sets up channels, and starts the graphical user interface (GUI).

Author: Pedro Silva
===============================================================================
"""

# Standard library imports
import sys

# External library imports
from ctypes import *                        # Used for handling the WaveForms SDK DLL

# API imports
from dwfconstants import *                  # Constants for the WaveForms SDK

# Project-specific imports
from device_manager import DeviceManager    # Manages device configuration and operations
from channel import Channel                 # Represents individual channels of a device
from gui import Gui                         # GUI interface for interacting with the device

def main():
    """
    Main function to initialize and configure the AnalogDiscovery3 device and start the GUI.

    Raises:
        OSError: If the script is not run on a Windows platform.
    """
    # Load the WaveForms SDK DLL for Windows
    if sys.platform.startswith("win"):
        dwf = cdll.dwf
    else:
        raise OSError("This script is designed for Windows only.")

    # Create a DeviceManager instance for AnalogDiscovery3
    AnalogDiscovery3 = DeviceManager(dwf, "AnalogDiscovery3", DWFUser.AutoConfigure.DISABLED.value)

    # Create and add two analog output channels to AnalogDiscovery3
    channel0 = Channel(dwf, c_int(0))
    channel1 = Channel(dwf, c_int(1))

    AnalogDiscovery3.append_channel(channel0)
    AnalogDiscovery3.append_channel(channel1)

    # Configure the device with the desired behavior
    AnalogDiscovery3.configure_device(DWFUser.TurnOffBehaviour.SHUTDOWN.value)

    # Start the GUI for user interaction
    app = Gui(AnalogDiscovery3)
    app.mainloop()

if __name__ == "__main__":
    main()
