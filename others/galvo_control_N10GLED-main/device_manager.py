"""
================================================================================
File Name: device_manager.py

Description:
    Manages the configuration, operation, and lifecycle of connected devices
    using the WaveForms SDK. Includes functionality for handling channels,
    setting device parameters, and managing device connections.

Author: Pedro Silva
================================================================================
"""

# External library imports
from ctypes import *  # Provides C-compatible data types and function bindings

# API imports
from dwfconstants import *  # Constants for WaveForms SDK

class DeviceManager:
    """
    Represents a manager for controlling WaveForms-compatible devices.

    This class handles device initialization, configuration, and channel management.

    Attributes:
        dwf:                WaveForms SDK instance used for device interaction.
        hdwf:               Handle to the connected device.
        channels_output:    List of channels configured for the device.
        autoConfigure:      Auto Configure setting, determines automatic output updates.
        name:               Name of the device.
        open_state:         Boolean indicating whether the device is currently open.
    
    Usage Example:
        manager = DeviceManager(dwf, "AnalogDiscovery3", DWFUser.AutoConfigure.DISABLED.value)
    """

    def __init__(self, dwf, name, autoConfigure):
        """
        Initializes the DeviceManager instance.

        Args:
            dwf:                    WaveForms SDK DLL instance.
            name (str):             Name of the device.
            autoConfigure (int):    Auto Configure setting.
        """
        self.hdwf               = c_int()       # Device handle
        self.channels_output    = []            # List of configured channels
        self.dwf                = dwf           # WaveForms SDK instance
        self.autoConfigure      = autoConfigure # Auto Configure option
        self.name               = name          # Device name
        self.open_state         = False         # Device open state

        print(f"[DM {self.name}] initialized.\n")

    def append_channel(self, channel: c_int):
        """
        Adds a channel to the device.

        Args:
            channel (c_int): Instance of a channel to append.
        """
        self.channels_output.append(channel)
        print(f"[DM {self.name}] Channel {channel.channel_number} appended.\n")

    def configure_device(self, onCloseAction: c_int):
        """
        Configures the device's behavior upon closing.

        Args:
            onCloseAction (c_int): Using the enum DWFUser.TurnOffBehaviour.
        """
        self.dwf.FDwfParamSet(DwfParamOnClose, onCloseAction)
        print(f"[DM {self.name}] Set the behavior on close to {onCloseAction}.\n")

    def open_device(self):
        """
        Opens the device for operation.

        Returns:
            bool: True if the device is opened successfully, False otherwise.
        """
        print(f"[DM {self.name}] Opening. . .")

        # Open the first device
        self.dwf.FDwfDeviceOpen(DWFUser.Devices.FIRST.value, byref(self.hdwf))

        if self.hdwf.value == hdwfNone.value:
            self.open_state = False

            open_error = create_string_buffer(512)
            self.dwf.FDwfGetLastErrorMsg(open_error)
            print(f"[DM {self.name}] Failed to open device: {open_error.value.decode()}.\n")
            return False

        print(f"[DM {self.name}] Opened successfully!")
        self.open_state = True

        # Set the auto configuration parameter
        self.dwf.FDwfDeviceAutoConfigureSet(self.hdwf, self.autoConfigure)
        print(f"[DM {self.name}] Set the behavior on Auto Configure to {self.autoConfigure}.\n")

        return True

    def close_device(self):
        """
        Closes the device and resets all channels and outputs.

        Returns:
            bool: True if the device is closed successfully.
        """
        # Reset all channels and outputs
        self.dwf.FDwfAnalogOutReset(self.hdwf, DWFUser.ALL_CHANNELS)
        self.dwf.FDwfAnalogOutConfigure(self.hdwf, DWFUser.ALL_CHANNELS, DWFUser.OutputBehaviour.APPLY.value)
        print(f"[DM {self.name}] Closed all channels.")

        # Close the device
        self.dwf.FDwfDeviceClose(self.hdwf)
        print(f"[DM {self.name}] Closed device successfully.\n")

        self.open_state = False
        return True
    
    def get_open_state(self):
        """
        Returns the device open_state.

        Returns:
            open_state
        """
        return self.open_state
    
    def __del__(self):
        """
        Destructor to ensure the device is closed when the object is deleted.
        """
        self.close_device()
