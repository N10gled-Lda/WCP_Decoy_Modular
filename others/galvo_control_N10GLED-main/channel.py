"""
================================================================================
File Name: channel.py

Description:
    Defines the Channel and DCChannel classes, providing an interface for analog
    output channel configuration and management.

Author: Pedro Silva
================================================================================
"""

# External library imports
from ctypes import *  # Used for managing the WaveForms SDK API calls

# API imports
from dwfconstants import *  # Provides constants for WaveForms SDK

class Channel:
    """
    Represents a generic analog output channel.

    This class serves as a base class for specific types of analog output channels,
    such as DC channels.

    Attributes:
        dwf:            WaveForms SDK instance used for device interaction.
        channel_number: Identifier for the channel (e.g., 0 or 1).

    Usage Example:
        channel = Channel(dwf, 0)
    """

    def __init__(self, dwf, channel_number: int):
        """
        Initializes a Channel instance.

        Args:
            dwf:                    WaveForms SDK instance.
            channel_number (int):   Channel number (e.g., 0 or 1).
        """
        self.dwf = dwf  # WaveForms SDK instance
        self.channel_number = channel_number  # Analog output channel identifier

        print(f"[CH] Channel created with number {channel_number}\n")

class DCChannel(Channel):
    """
    Represents a DC analog output channel.

    This class extends the Channel class to manage DC-specific configurations
    and operations.

    Attributes:
        dwf:            WaveForms SDK instance.
        hdwf:           Handle to the connected device.
        channel_number: Identifier for the channel (e.g., 0 or 1).
        DC_MAX:         Maximum allowable DC voltage.
        DC_MIN:         Minimum allowable DC voltage.
        dc_offset:      Current DC offset value (default is 0 V).

    Usage Example:
        dc_channel = DCChannel(dwf, hdwf, 0, -5, 5)
    """

    def __init__(self, dwf, hdwf, channel_number: int, dc_min: float, dc_max: float):
        """
        Initializes a DCChannel instance.

        Args:
            dwf:                    WaveForms SDK instance.
            hdwf:                   Handle to the connected device.
            channel_number (int):   Channel number (e.g., 0 or 1).
            dc_min (float):         Minimum DC voltage.
            dc_max (float):         Maximum DC voltage.
        """
        super().__init__(dwf, channel_number)

        self.hdwf = hdwf
        self.DC_MAX = dc_max
        self.DC_MIN = dc_min
        self.dc_offset = 0  # Default DC offset is 0 V

        print(f"[CH] DC channel created with number {channel_number}")

    def output_set(self, output):
        """
        Sets the DC output channel to the specified configuration.

        Args:
            output : Using the enum DWFUser.OutputBehaviour .
        """
        # Set the DC offset value
        self.dwf.FDwfAnalogOutOffsetSet(self.hdwf, c_int(self.channel_number), c_double(self.dc_offset))

        # Apply the new configuration
        self.dwf.FDwfAnalogOutConfigure(self.hdwf, c_int(self.channel_number), output)

        print(f"[CH] DC channel {self.channel_number} with {self.dc_offset} V configured to output {output}.\n")

    def configure(self):
        """
        Configures the DC channel for operation with the current settings.
        """
        print(f"[CH] Configuring channel {self.channel_number} for DC output: {self.dc_offset} V.\n")

        # Enable the channel
        self.dwf.FDwfAnalogOutEnableSet(self.hdwf, c_int(self.channel_number), True)

        # Set the channel to DC mode
        self.dwf.FDwfAnalogOutFunctionSet(self.hdwf, c_int(self.channel_number), DWFUser.AnalogOutFunctions.DC.value)

        # Apply the DC offset value
        self.dwf.FDwfAnalogOutOffsetSet(self.hdwf, c_int(self.channel_number), c_double(self.dc_offset))

        # Start the channel output
        self.dwf.FDwfAnalogOutConfigure(self.hdwf, c_int(self.channel_number), DWFUser.OutputBehaviour.START.value)

    def dc_offset_set(self, new_value):
        """
        Set the dc_offset.

        Args:
            new_value : New value to set the dc_offset.
        """

        if new_value < self.DC_MIN:
            self.dc_offset = self.DC_MIN
        elif new_value > self.DC_MAX:
            self.dc_offset = self.DC_MAX
        else:
            self.dc_offset = new_value

        return self.dc_offset