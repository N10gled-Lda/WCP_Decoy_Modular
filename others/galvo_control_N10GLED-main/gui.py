"""
================================================================================
File Name: gui.py

Description: 
    Graphical User Interface (GUI) for controlling Galvo devices. 
    Includes manual control for X and Y axes.

Author: Pedro Silva
================================================================================
"""

# External library imports
import customtkinter as ctk
from functools import partial

# API imports
from dwfconstants import *

# Project-specific imports
from channel import *

class GuiOutputChannelDC:
    """
    Represents a DC output channel for the GUI.

    Manages channel settings, GUI elements, and their states.

    Attributes:
        name (str):                         Name of the channel.
        number (int):                       Channel number.
        
        guiButtonEnable (ctk.CTkButton):    Button to enable/disable the channel.
        enabled (ctk.BooleanVar):           Indicates whether the channel is enabled.

        guiValueEntry (ctk.CTkEntry):       Input for channel value.
        value (ctk.DoubleVar):              Current value of the channel.

        guiAdjustEntry (ctk.CTkEntry):      Input for adjustment step.
        adjust (ctk.DoubleVar):             Adjustment step size.
        guiButtonIncrement (ctk.CTkButton): Button to increment value.
        guiButtonDecrement (ctk.CTkButton): Button to decrement value.

        guiOutput (ctk.CTkLabel):           Display for output value.
        output (ctk.DoubleVar):             Current output value.

        guiFrame (ctk.CTkFrame):            Frame containing channel controls.
    """

    def __init__(self, name, number, text1, text2):
        self.name               = name
        self.number             = number

        self.guiButtonEnable    = None
        self.enabled            = ctk.BooleanVar(value=False)

        self.guiValueEntry      = None
        self.value              = ctk.DoubleVar(value=0.0)

        self.guiAdjustEntry     = None
        self.adjust             = ctk.DoubleVar(value=0.5)
        self.guiAdjustText1     = text1
        self.guiAdjustText2     = text2
        self.guiButtonIncrement = None
        self.guiButtonDecrement = None

        self.guiOutput          = None
        self.output             = ctk.DoubleVar(value=0.0)

        self.guiFrame           = None

    def reset_parameters(self):
        """
        Resets the channel's parameters to their initial default values.
        """
        self.enabled.set(False)
        self.value.set(0.0)
        self.adjust.set(0.5)
        self.output.set(0.0)

class Gui(ctk.CTk):
    """
    Main GUI application for controlling Galvo devices.

    Features tabs for manual control and camera usage.

    Attributes:
        device:                         The device object managing hardware interactions.
        x_axis (GuiOutputChannelDC):    Represents the X-axis channel.
        y_axis (GuiOutputChannelDC):    Represents the Y-axis channel.
        tabview (ctk.CTkTabview):       Tabbed interface container.
        manual_tab (ctk.CTkFrame):      Tab for manual control.
        configure_tab (ctk.CTkFrame):   Tab for camera configuration.
        open_button (ctk.CTkButton):    Button to toggle device state.
    """

    def __init__(self, device):
        super().__init__()

        # Window configuration
        self.title("Galvo Control")
        self.geometry("800x600")

        # Themes
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Device interaction
        self.device = device

        # Channels for X and Y axes
        self.x_axis = GuiOutputChannelDC("X axis", int(0), "- (Right)", "+ (Left)")
        self.y_axis = GuiOutputChannelDC("Y axis", int(1), "- (Down)", "- (Up)")

        # Build GUI
        self.build_ui()

    def build_ui(self):
        """
        Builds the main user interface, including tabbed views.
        """
        # Tabbed interface
        self.tabview = ctk.CTkTabview(self, anchor="nw")
        self.tabview.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        # Add tabs
        self.manual_tab = self.tabview.add("Manual")
        self.configure_tab = self.tabview.add("Camera")

        # Build manual tab UI
        self.build_manual_tab()

    def build_manual_tab(self):
        """
        Creates the layout and widgets for the Manual tab.
        """
        main_frame = ctk.CTkFrame(self.manual_tab)
        main_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Configure grid weights
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Frame for control
        control_frame = self.build_control_frame(main_frame)

        # Frame for device
        device_frame = ctk.CTkFrame(main_frame)
        device_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.build_device_controls(device_frame)

    def build_control_frame(self, parent_frame):
        """
        Builds the frame containing X and Y axis controls.

        Args:
            parent_frame (ctk.CTkFrame): Parent frame for the axis controls.
        """
        axis_frame = ctk.CTkFrame(parent_frame)
        axis_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        axis_frame.grid_rowconfigure(0, weight=0)
        axis_frame.grid_rowconfigure(1, weight=0)
        axis_frame.grid_rowconfigure(2, weight=1) 
        axis_frame.grid_columnconfigure(0, weight=1)
        axis_frame.grid_columnconfigure(1, weight=1)

        # Title
        control_label = ctk.CTkLabel(axis_frame, text="Control", font=("Arial", 14, "bold"))
        control_label.grid(row=0, column=0, pady=5, columnspan=2, sticky="nsew")

        # X-axis controls
        self.x_axis.guiFrame = ctk.CTkFrame(axis_frame)
        self.x_axis.guiFrame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(self.x_axis.guiFrame, text="X-axis", font=("Arial", 14, "bold")).grid(row=0, column=0, pady=5, columnspan=2)
        
        self.build_axis_controls(self.x_axis)

        # Y-axis controls
        self.y_axis.guiFrame = ctk.CTkFrame(axis_frame)
        self.y_axis.guiFrame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")  # Use grid for y_axis_frame
        ctk.CTkLabel(self.y_axis.guiFrame, text="Y-axis", font=("Arial", 14, "bold")).grid(row=0, column=0, pady=5, columnspan=2)
        
        self.build_axis_controls(self.y_axis)
        
        return axis_frame

    def build_axis_controls(self, axis):
        """
        Creates control widgets for an axis.

        Args:
            axis (GuiOutputChannelDC): Axis object to configure.
        """

        # Enable button
        axis.guiButtonEnable = ctk.CTkButton(axis.guiFrame)

        ctk.CTkLabel(axis.guiFrame, text="Enable:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        axis.guiButtonEnable.configure(command=partial(self.toggle_enable_channel, axis))
        axis.guiButtonEnable.configure(state="disabled", text="Disabled", fg_color="gray")
        axis.guiButtonEnable.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Value entry
        valueLabel = ctk.CTkLabel(axis.guiFrame, text="Value:")
        valueLabel.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        valueUnitLabel = ctk.CTkLabel(axis.guiFrame, text="V")
        valueUnitLabel.grid(row=2, column=2, padx=5, pady=5, sticky="w")  # "V" for value
        axis.guiValueEntry = ctk.CTkEntry(axis.guiFrame, textvariable=axis.value)
        axis.guiValueEntry.grid(row=2, column=1, padx=5, pady=5)
        axis.guiValueEntry.bind("<Return>", lambda event: self.set_output(axis, axis.number))

        # Adjustment step entry
        adjustLabel = ctk.CTkLabel(axis.guiFrame, text="Adjust:")
        adjustLabel.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        adjustUnitLabel = ctk.CTkLabel(axis.guiFrame, text="V")
        adjustUnitLabel.grid(row=3, column=2, padx=5, pady=5, sticky="w")  # "V" for adjustment

        axis.guiAdjustEntry = ctk.CTkEntry(axis.guiFrame, textvariable=axis.adjust)
        axis.guiAdjustEntry.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        

        # Increment/Decrement buttons
        axis.guiButtonDecrement = ctk.CTkButton(axis.guiFrame, text=axis.guiAdjustText1, command=lambda: self.increment_value(axis, -1, axis.number), width= 60)
        axis.guiButtonDecrement.grid(row=4, column=0, padx=5, pady=5)
        
        axis.guiButtonIncrement = ctk.CTkButton(axis.guiFrame, text=axis.guiAdjustText2, command=lambda: self.increment_value(axis, 1, axis.number), width= 60)
        axis.guiButtonIncrement.grid(row=4, column=1, padx=5, pady=5)

        # Output value
        outputLabel = ctk.CTkLabel(axis.guiFrame, text="Output:")
        outputLabel.grid(row=5, column=0, padx=5, pady=5, sticky="w")
        outputUnitLabel = ctk.CTkLabel(axis.guiFrame, text="V")
        outputUnitLabel.grid(row=5, column=2, padx=5, pady=5, sticky="w")  # "V" for output
        
        axis.guiOutput = ctk.CTkLabel(axis.guiFrame, textvariable = axis.output)
        axis.guiOutput.grid(row=5, column=1, padx=5, pady=5)

    def build_device_controls(self, parent_frame):
        """
        Builds controls for managing the device state.

        Args:
            parent (ctk.CTkFrame): Parent frame for device controls.
        """

        # Title
        title_label = ctk.CTkLabel(parent_frame, text="Device", font=("Arial", 14, "bold"))
        title_label.pack(pady=5, anchor="center")
        
        # Device toggle button
        self.open_button = ctk.CTkButton(parent_frame, text="Device off!", command=self.toggle_open_device, fg_color="red")
        self.open_button.pack(pady=10, anchor="center")

    def toggle_open_device(self):
        """
        Toggles the device's open/close state and updates UI controls.
        """
        print("[GUI Manual] Function toggle_open_device")
        
        if(self.device.get_open_state() == False):
            if(self.device.open_device() == False):
                print("[GUI Manual] Can't open the device!")
            else:
                print("[GUI Manual] Device opened!")
                self.open_button.configure(text="Device on!", fg_color="green")
                self.x_axis.guiButtonEnable.configure(state="normal", text="Disabled", fg_color="red")
                self.y_axis.guiButtonEnable.configure(state="normal", text="Disabled", fg_color="red")
        elif(self.device.get_open_state() == True):
            if(self.device.close_device() == False):
                print("[GUI Manual] Can't close the device!")
            else:
                print("[GUI Manual] Device closed!")
            self.open_button.configure(text="Device off!", fg_color="red")
            self.x_axis.guiButtonEnable.configure(state="disabled", text="Disabled", fg_color="gray")
            self.y_axis.guiButtonEnable.configure(state="disabled", text="Disabled", fg_color="gray")
            self.x_axis.reset_parameters()
            self.y_axis.reset_parameters()
        print("\n")

    def toggle_enable_channel(self, axis):
        """
        Toggles the enable/disable state of a given axis channel.
        
        Parameters:
            axis (GuiOutputChannelDC): The axis object (e.g., X or Y axis).
        """

        current_state = axis.enabled.get()

        axis.reset_parameters()
        if(current_state == False):
            try:
                print(f"[GUI Manual] Try enabling {axis.name}")
                self.device.channels_output[axis.number] = DCChannel(self.device.dwf, self.device.hdwf, axis.number, -5.0, 5.0)
                self.device.channels_output[axis.number].dc_offset_set(0)
                self.device.channels_output[axis.number].configure()
                axis.enabled.set(not current_state)
            except Exception as e:
                # Catch any exceptions and handle them
                print(f"[GUI Manual] Can't enable, error: {e}")
                return
        else:
            try:
                print(f"Disabling {axis.name}")
                self.device.channels_output[axis.number].dc_offset_set(0)
                self.device.channels_output[axis.number].output_set(DWFUser.OutputBehaviour.STOP.value) # DEV: Maybe this here isn't enough
            except Exception as e:
                # Catch any exceptions and handle them
                print(f"[GUI Manual] Can't disable, error: {e}")
                return

        if axis.enabled.get():
            axis.guiButtonEnable.configure(fg_color="green", text="Enabled")  # Green color when enabled
        else:
            axis.guiButtonEnable.configure(fg_color="red", text="Disabled")  # Red color when disabled

    def increment_value(self, axis, delta, channel):
        """
        Adjusts the output value of a channel by a given increment.

        Parameters:
            axis (GuiOutputChannelDC):  The axis object to modify.
            delta (int):                The increment value (-1 for decrement, 1 for increment).
            channel_number (int):       The channel number associated with the axis.
        """
        if(axis.enabled.get() == True):
            new_value = axis.value.get() + (axis.adjust.get() * delta)
            axis.value.set(new_value)
            self.set_output(axis, channel)

    def set_output(self, axis, channel_number, event=None):
        """
        Sets the output value for a specified channel.

        Parameters:
            axis (GuiOutputChannelDC):  The axis object to update.
            channel_number (int):       The channel number to configure.
        """

        channel = self.device.channels_output[channel_number]
        
        # TODO: Improve
        value = axis.value.get()
        if value == "":  
            axis.value.set(0.0)
        else:
            try:
                value = float(value)
            except ValueError:
                value = 0.0

        value = channel.dc_offset_set(value)
        
        axis.value.set(value)

        axis.output.set(value)

        channel.output_set(DWFUser.OutputBehaviour.APPLY.value)
