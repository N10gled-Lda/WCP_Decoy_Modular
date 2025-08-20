"""
GUI Demo for Polarization Controller Hardware Interface using CustomTkinter
Inspired by the STM32 interface main.py
"""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import customtkinter as ctk
import threading
import time
from queue import Queue, Empty
from typing import Optional
import logging

from src.alice.polarization.polarization_controller import PolarizationController, create_polarization_controller_with_hardware
from src.alice.polarization.polarization_base import PolarizationState
from src.utils.data_structures import Basis, Bit, Pulse, LaserInfo
import serial.tools.list_ports

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolarizationControllerGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        # Scrollable window
        self.scrollable_frame = ctk.CTkScrollableFrame(self, width=550, height=800)
        self.scrollable_frame.pack(fill="both", expand=True)
        
        self.title("Polarization Controller Hardware Interface")
        self.geometry("550x800")

        # Controller
        self.pol_controller: Optional[PolarizationController] = None
        
        # Current state
        self.current_basis = Basis.Z
        self.current_bit = Bit.ZERO
        self.is_connected = False
        
        # Setup GUI
        self.setup_gui()
        
        # # Start monitoring thread
        # self.monitoring = True
        # self.monitor_thread = threading.Thread(target=self.monitor_controller, daemon=True)
        # self.monitor_thread.start()

    def setup_gui(self):
        """Setup the GUI layout"""
        
        # Title
        title_label = ctk.CTkLabel(self.scrollable_frame, text="STM32 Polarization Controller Interface", 
                                  font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=(20, 10))
        
        # Connection Frame
        conn_frame = ctk.CTkFrame(self.scrollable_frame)
        conn_frame.pack(pady=10, padx=20, fill="x")
        
        conn_label = ctk.CTkLabel(conn_frame, text="Hardware Connection", 
                                 font=ctk.CTkFont(size=16, weight="bold"))
        conn_label.pack(pady=(10, 5))
        
        # COM port selection
        com_frame = ctk.CTkFrame(conn_frame, fg_color="transparent")
        com_frame.pack(pady=5, padx=10, fill="x")
        
        self.com_label = ctk.CTkLabel(com_frame, text="Select COM Port:")
        self.com_label.pack(side="left", padx=(10, 5))
        
        self.combobox = ctk.CTkComboBox(com_frame, values=self.get_com_ports(), width=150)
        self.combobox.pack(side="left", padx=5, expand=True, fill = "x")
        
        self.refresh_button = ctk.CTkButton(com_frame, text="Refresh", command=self.refresh_com_ports, width=80)
        self.refresh_button.pack(side="left", padx=5, expand=True, fill = "x")
        
        # Connect button
        conn_status_frame = ctk.CTkFrame(conn_frame, fg_color="transparent")
        conn_status_frame.pack(pady=5, padx=10, fill="x")

        self.connect_button = ctk.CTkButton(conn_status_frame, text="Connect", command=self.toggle_connection)
        self.connect_button.pack(pady=0, side="left", expand=True, fill = "x")
        
        # Connection status
        self.status_label = ctk.CTkLabel(conn_status_frame, text="● Disconnected", 
                                           text_color="red", font=ctk.CTkFont(size=14, weight="bold"))
        self.status_label.pack(pady=(0, 0), side="right", expand=True, fill="x")
        status_indicator = ctk.CTkLabel(conn_status_frame, text="Connection Status:", font=ctk.CTkFont(size=14, weight="bold"))
        status_indicator.pack(pady=(0, 0), side="right", expand=True)


        # Polarization Control Frame
        control_frame = ctk.CTkFrame(self.scrollable_frame)
        control_frame.pack(pady=(0,10), padx=20, fill="x")
        
        control_label = ctk.CTkLabel(control_frame, text="Polarization Control", 
                                   font=ctk.CTkFont(size=16, weight="bold"))
        control_label.pack()
        
        # Manual control
        manual_frame = ctk.CTkFrame(control_frame)
        manual_frame.pack(pady=5, padx=10, fill="x")
        man_manual_frame = ctk.CTkFrame(manual_frame, fg_color="transparent")
        man_manual_frame.pack(pady=5, fill="x")


        man_manual_frame.grid_columnconfigure(0, weight=1)
        man_manual_frame.grid_columnconfigure(1, weight=1)

        manual_label = ctk.CTkLabel(man_manual_frame, text="Manual Polarization Control:")
        manual_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        # Set polarization button (top-right)
        self.set_pol_button = ctk.CTkButton(man_manual_frame, text="Set Polarization",
                           command=self.set_polarization_manual, state="disabled")
        self.set_pol_button.grid(row=0, column=1, sticky="we", padx=5, pady=5)

        # Basis selection frame placed on the bottom and spanning both columns
        basis_frame = ctk.CTkFrame(man_manual_frame, fg_color="transparent")
        basis_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=(5, 0))
        
        # ctk.CTkLabel(basis_frame, text="Basis:").pack(side="left", padx=(10, 5))
        # self.basis_var = ctk.StringVar(value="Z")
        # self.basis_menu = ctk.CTkOptionMenu(basis_frame, variable=self.basis_var, 
        #                                   values=["Z", "X"], command=self.on_basis_change)
        # self.basis_menu.pack(side="left", padx=5)
        
        ctk.CTkLabel(basis_frame, text="Basis:").pack(side="left", padx=10)
        self.basis_var = ctk.StringVar(value="Z")
        basis_radio1 = ctk.CTkRadioButton(basis_frame, text="Z (Rectilinear)", 
                                         variable=self.basis_var, value="Z", command=self.update_state_display)
        basis_radio1.pack(side="left", padx=5, expand=True)
        basis_radio2 = ctk.CTkRadioButton(basis_frame, text="X (Diagonal)", 
                                         variable=self.basis_var, value="X", command=self.update_state_display)
        basis_radio2.pack(side="left", padx=5, expand=True)

        # Bit selection
        # ctk.CTkLabel(basis_frame, text="Bit:").pack(side="left", padx=(20, 5))
        # self.bit_var = ctk.StringVar(value="0")
        # self.bit_menu = ctk.CTkOptionMenu(basis_frame, variable=self.bit_var, 
        #                                 values=["0", "1"], command=self.on_bit_change)
        # self.bit_menu.pack(side="left", padx=5)
        
        ctk.CTkLabel(basis_frame, text="Bit:").pack(side="left", padx=10)
        self.bit_var = ctk.StringVar(value="0")
        bit_radio1 = ctk.CTkRadioButton(basis_frame, text="0", width=50,
                                       variable=self.bit_var, value="0", command=self.update_state_display)
        bit_radio1.pack(side="left", padx=5, expand=True)
        bit_radio2 = ctk.CTkRadioButton(basis_frame, text="1", width=50,
                                       variable=self.bit_var, value="1", command=self.update_state_display)
        bit_radio2.pack(side="left", padx=5, expand=True)
        

        # Quick preset buttons
        preset_frame = ctk.CTkFrame(manual_frame, fg_color="transparent")
        preset_frame.pack(pady=(0,5), fill="x", expand=True)
        
        ctk.CTkLabel(preset_frame, text="Quick Presets:").pack(pady=5, padx=(0,5), side = "left", expand=True)
        
        # presets_button_frame = ctk.CTkFrame(preset_frame)
        # presets_button_frame.pack(pady=5)
        
        self.h_button = ctk.CTkButton(preset_frame, text="H (0°)", 
                                    command=lambda: self.set_preset(Basis.Z, Bit.ZERO), 
                                    state="disabled", width=80)
        self.h_button.pack(side="left", padx=2, expand=True)
        
        self.v_button = ctk.CTkButton(preset_frame, text="V (90°)", 
                                    command=lambda: self.set_preset(Basis.Z, Bit.ONE), 
                                    state="disabled", width=80)
        self.v_button.pack(side="left", padx=2, expand=True)
        
        self.d_button = ctk.CTkButton(preset_frame, text="D (45°)", 
                                    command=lambda: self.set_preset(Basis.X, Bit.ZERO), 
                                    state="disabled", width=80)
        self.d_button.pack(side="left", padx=2, expand=True)
        
        self.a_button = ctk.CTkButton(preset_frame, text="A (135°)", 
                                    command=lambda: self.set_preset(Basis.X, Bit.ONE), 
                                    state="disabled", width=80)
        self.a_button.pack(side="left", padx=2, expand=True)
        
        # Random QRNG control
        qrng_frame = ctk.CTkFrame(manual_frame, fg_color="transparent")
        qrng_frame.pack(pady=5, padx=10, fill="x", expand=True)
        
        qrng_label = ctk.CTkLabel(qrng_frame, text="QRNG Control")
        qrng_label.pack(pady=5, side = "left", expand=True)
        
        self.qrng_button = ctk.CTkButton(qrng_frame, text="Set Random Polarization (QRNG)", 
                                       command=self.set_polarization_qrng, state="disabled")
        self.qrng_button.pack(pady=10, side = "left", expand=True)
        

        # Direct STM32 Control (inspired by original main.py)
        stm32_frame = ctk.CTkFrame(control_frame)
        stm32_frame.pack(pady=10, padx=10, fill="x")
        
        stm32_label = ctk.CTkLabel(stm32_frame, text="Direct STM32 Control - Polarization numbers (0-3, comma separated):")
        stm32_label.pack(pady=(10, 0))

        self.polarization_entry = ctk.CTkEntry(stm32_frame, placeholder_text="0,1,2,3")
        self.polarization_entry.pack(pady=10, padx=(10,5), fill="x", side="left", expand=True)
        
        self.send_stm32_button = ctk.CTkButton(stm32_frame, text="Send to STM32", 
                                              command=self.send_polarization_numbers,
                                              state="disabled")
        self.send_stm32_button.pack(pady=10, padx=(5,10), side="right")

        # Advanced STM32 Controls Frame (inspired by main.py)
        advanced_frame = ctk.CTkFrame(self.scrollable_frame)
        advanced_frame.pack(pady=(0,10), padx=20, fill="x")
        
        advanced_label = ctk.CTkLabel(advanced_frame, text="Advanced STM32 Controls", 
                                    font=ctk.CTkFont(size=14, weight="bold"))
        advanced_label.pack()
        
        # Device selection
        device_frame = ctk.CTkFrame(advanced_frame)
        device_frame.pack(pady=5, padx=10, fill="x")
        
        device_label = ctk.CTkLabel(device_frame, text="Device:")
        device_label.pack(side="left", padx=(10, 5))
        
        self.device_var = ctk.StringVar(value="1")
        device_radio1 = ctk.CTkRadioButton(device_frame, text="Linear Polarizer (1)", 
                                         variable=self.device_var, value="1")
        device_radio1.pack(side="left", padx=5)
        device_radio2 = ctk.CTkRadioButton(device_frame, text="Half Wave Plate (2)", 
                                         variable=self.device_var, value="2")
        device_radio2.pack(side="left", padx=5)
        
        self.set_device_button = ctk.CTkButton(device_frame, text="Set Device", 
                                             command=self.set_polarization_device, 
                                             state="disabled", width=100)
        self.set_device_button.pack(side="right", padx=10)
        
        # Angle control
        angle_frame = ctk.CTkFrame(advanced_frame)
        angle_frame.pack(pady=5, padx=10, fill="x")
        
        angle_label = ctk.CTkLabel(angle_frame, text="Angle (0-360°):")
        angle_label.pack(side="left", padx=(10, 5))
        
        self.angle_entry = ctk.CTkEntry(angle_frame, width=80, placeholder_text="45")
        self.angle_entry.pack(side="left", padx=5)
        
        self.offset_switch = ctk.CTkSwitch(angle_frame, text="Set as Offset")
        self.offset_switch.pack(side="left", padx=10)
        
        self.set_angle_button = ctk.CTkButton(angle_frame, text="Set Angle", 
                                            command=self.set_angle_direct, 
                                            state="disabled", width=100)
        self.set_angle_button.pack(side="right", padx=10)
        
        # Frequency controls
        freq_frame = ctk.CTkFrame(advanced_frame)
        freq_frame.pack(pady=5, padx=10, fill="x")
        
        # Stepper frequency
        stepper_frame = ctk.CTkFrame(freq_frame, fg_color="transparent")
        stepper_frame.pack(pady=2, fill="x")
        
        stepper_label = ctk.CTkLabel(stepper_frame, text="Stepper Frequency (1-1000 Hz):")
        stepper_label.pack(side="left", padx=(10, 5))
        
        self.stepper_entry = ctk.CTkEntry(stepper_frame, width=80, placeholder_text="100")
        self.stepper_entry.pack(side="left", padx=5)
        
        self.set_stepper_button = ctk.CTkButton(stepper_frame, text="Set Stepper Freq", 
                                              command=self.set_stepper_frequency, 
                                              state="disabled", width=120)
        self.set_stepper_button.pack(side="right", padx=10)
        
        # Operation period
        period_frame = ctk.CTkFrame(freq_frame, fg_color="transparent")
        period_frame.pack(pady=2, fill="x")
        
        period_label = ctk.CTkLabel(period_frame, text="Operation Period (1-60000 ms):")
        period_label.pack(side="left", padx=(10, 5))
        
        self.period_entry = ctk.CTkEntry(period_frame, width=80, placeholder_text="5000")
        self.period_entry.pack(side="left", padx=5)
        
        self.set_period_button = ctk.CTkButton(period_frame, text="Set Period", 
                                             command=self.set_operation_period, 
                                             state="disabled", width=120)
        self.set_period_button.pack(side="right", padx=10)


        # Current State Frame
        state_frame = ctk.CTkFrame(self.scrollable_frame)
        state_frame.pack(pady=10, padx=20, fill="x")
        
        state_label = ctk.CTkLabel(state_frame, text="Current State", 
                                 font=ctk.CTkFont(size=16, weight="bold"))
        state_label.pack(pady=(10, 5))
        
        # State display
        self.state_text = ctk.CTkTextbox(state_frame, height=120, width=550)
        self.state_text.pack(pady=10, padx=10)
        
        # Log Frame
        log_frame = ctk.CTkFrame(self.scrollable_frame)
        log_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        log_label = ctk.CTkLabel(log_frame, text="Activity Log", 
                               font=ctk.CTkFont(size=16, weight="bold"))
        log_label.pack(pady=(10, 5))
        
        self.log_text = ctk.CTkTextbox(log_frame, height=150)
        self.log_text.pack(pady=10, padx=10, fill="both", expand=True)
        
        # Initial state update
        self.update_state_display()
        self.log_message("GUI initialized. Connect to hardware to begin.")

    def get_com_ports(self):
        """Get available COM ports"""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports] if ports else ["No ports found"]

    def refresh_com_ports(self):
        """Refresh the COM port list"""
        new_ports = self.get_com_ports()
        self.combobox.configure(values=new_ports)
        if new_ports and new_ports[0] != "No ports found":
            self.combobox.set(new_ports[0])
        self.log_message(f"Refreshed COM ports: {len(new_ports)} found")

    def toggle_connection(self):
        """Toggle connection to STM32 hardware."""
        if not self.is_connected:
            self.connect_hardware()
        else:
            self.disconnect_hardware()

    def connect_hardware(self):
        """Connect to the hardware"""

        try:
            com_port = self.combobox.get()
            if not com_port or com_port == "No ports found":
                self.status_label.configure(text="✗ Please select a valid COM port", text_color="red")
                self.log_message("✗ Please select a valid COM port")
                return
            
            self.log_message(f"Connecting to hardware on {com_port}...")
            
            # Create hardware controller
            self.pol_controller = create_polarization_controller_with_hardware(
                com_port=com_port
            )
            
            # Initialize the controller
            self.pol_controller.__enter__()
            
            self.is_connected = True
            self.status_label.configure(text=f"● Connected to {com_port}", text_color="green")
            self.connect_button.configure(text="Disconnect from STM32")
            
            # Enable controls
            self.enable_controls(True)
            
            self.log_message(f"Successfully connected to hardware on {com_port}")
            self.update_state_display()
            
        except Exception as e:
            self.status_label.configure(text=f"Connection failed: {str(e)}", text_color="red")
            self.log_message(f"Connection failed: {str(e)}")
            self.is_connected = False

    def disconnect_hardware(self):
        """Disconnect from hardware"""
        try:
            if self.pol_controller:
                self.pol_controller.__exit__(None, None, None)
                self.pol_controller = None
                
            self.is_connected = False
            self.status_label.configure(text="● Disconnected", text_color="red")
            self.connect_button.configure(text="Connect to STM32")
            
            # Disable controls
            self.enable_controls(False)
            
            self.log_message("Disconnected from hardware")
            
        except Exception as e:
            self.log_message(f"Error during disconnect: {str(e)}")

    def enable_controls(self, enabled: bool):
        """Enable or disable control buttons"""
        state = "normal" if enabled else "disabled"
        
        self.set_pol_button.configure(state=state)
        self.qrng_button.configure(state=state)
        self.h_button.configure(state=state)
        self.v_button.configure(state=state)
        self.d_button.configure(state=state)
        self.a_button.configure(state=state)
        self.send_stm32_button.configure(state=state)
        self.set_device_button.configure(state=state)
        self.set_angle_button.configure(state=state)
        self.set_stepper_button.configure(state=state)
        self.set_period_button.configure(state=state)

    def set_polarization_manual(self):
        """Set polarization manually"""
        if not self.is_connected or not self.pol_controller:
            self.log_message("Not connected to hardware")
            return
            
        try:
            basis = Basis.Z if self.basis_var.get() == "Z" else Basis.X
            bit = Bit.ZERO if self.bit_var.get() == "0" else Bit.ONE
            
            self.pol_controller.set_polarization_manually(basis, bit)
            
            # Update current state
            self.current_basis = basis
            self.current_bit = bit
            
            basis_name = "Z (H/V)" if basis == Basis.Z else "X (D/A)"
            bit_name = "0" if bit == Bit.ZERO else "1"
            angle = self.get_angle_for_state(basis, bit)
            
            self.log_message(f"Set polarization: {basis_name}, bit {bit_name} ({angle}°)")
            self.update_state_display()
            
        except Exception as e:
            self.log_message(f"Error setting polarization: {str(e)}")

    def set_preset(self, basis: Basis, bit: Bit):
        """Set a preset polarization"""
        if not self.is_connected or not self.pol_controller:
            self.log_message("Not connected to hardware")
            return
            
        try:
            self.pol_controller.set_polarization_manually(basis, bit)
            
            # Update GUI
            self.basis_var.set("Z" if basis == Basis.Z else "X")
            self.bit_var.set("0" if bit == Bit.ZERO else "1")
            self.current_basis = basis
            self.current_bit = bit
            
            angle = self.get_angle_for_state(basis, bit)
            state_name = self.get_state_name(basis, bit)
            
            self.log_message(f"Set preset: {state_name} ({angle}°)")
            self.update_state_display()
            
        except Exception as e:
            self.log_message(f"Error setting preset: {str(e)}")

    def set_polarization_qrng(self):
        """Set random polarization using QRNG"""
        if not self.is_connected or not self.pol_controller:
            self.log_message("Not connected to hardware")
            return
            
        try:
            output = self.pol_controller.set_polarization_from_qrng()
            
            # Update GUI to reflect the random choice
            self.basis_var.set("Z" if output.basis == Basis.Z else "X")
            self.bit_var.set("0" if output.bit == Bit.ZERO else "1")
            self.current_basis = output.basis
            self.current_bit = output.bit
            
            self.log_message(f"QRNG set: {output.basis.name} basis, bit {output.bit.value} → "
                           f"{output.polarization_state.name} ({output.angle_degrees}°)")
            self.update_state_display()
            
        except Exception as e:
            self.log_message(f"Error with QRNG: {str(e)}")

    def send_polarization_numbers(self):
        """Send polarization numbers directly to STM32 (inspired by original main.py)."""
        if not self.pol_controller:
            self.log_message("✗ Not connected to STM32", "red")
            return
        
        numbers_str = self.polarization_entry.get()
        try:
            numbers = [int(x.strip()) for x in numbers_str.split(",") if x.strip() != ""]
            if not numbers:
                self.log_message("✗ Enter at least one number", "red")
                return
            if not all(0 <= n <= 3 for n in numbers):
                self.log_message("✗ Numbers must be 0, 1, 2, or 3", "red")
                return
        except ValueError:
            self.log_message("✗ Invalid input. Use comma-separated numbers", "red")
            return

        try:
            # Access the STM32 interface through the hardware driver
            hardware_driver = self.pol_controller.driver
            if hasattr(hardware_driver, 'stm'):
                success = hardware_driver.stm.send_cmd_polarization_numbers(numbers)
                if success:
                    self.log_message(f"✓ Sent polarization numbers to STM32: {numbers}", "green")
                    self.polarization_entry.delete(0, 'end')  # Clear entry
                else:
                    self.log_message("✗ Failed to send numbers to STM32", "red")
            else:
                self.log_message("✗ STM32 interface not available", "red")
        except Exception as e:
            self.log_message(f"✗ Error sending to STM32: {str(e)}", "red")

    def set_polarization_device(self):
        """Set the polarization device (Linear Polarizer or Half Wave Plate)"""
        if not self.pol_controller:
            self.log_message("✗ Not connected to STM32", "red")
            return
        
        device = self.device_var.get()
        try:
            hardware_driver = self.pol_controller.driver
            if hasattr(hardware_driver, 'stm'):
                success = hardware_driver.stm.send_cmd_polarization_device(device)
                if success:
                    self.log_message(f"✓ Set polarization device to: {device}", "green")
                else:
                    self.log_message("✗ Failed to set polarization device", "red")
            else:
                self.log_message("✗ STM32 interface not available", "red")
        except Exception as e:
            self.log_message(f"✗ Error setting device: {str(e)}", "red")

    def set_angle_direct(self):
        """Set angle directly with optional offset"""
        if not self.pol_controller:
            self.log_message("✗ Not connected to STM32", "red")
            return
        
        try:
            angle = float(self.angle_entry.get())
            use_offset = self.offset_var.get()
            
            hardware_driver = self.pol_controller.driver
            if hasattr(hardware_driver, 'stm'):
                if use_offset:
                    success = hardware_driver.stm.send_cmd_set_angle(angle, offset=True)
                    action = "set angle with offset"
                else:
                    success = hardware_driver.stm.send_cmd_set_angle(angle, offset=False)
                    action = "set angle"
                
                if success:
                    self.log_message(f"✓ Successfully {action}: {angle}°", "green")
                    self.angle_entry.delete(0, 'end')
                else:
                    self.log_message(f"✗ Failed to {action}", "red")
            else:
                self.log_message("✗ STM32 interface not available", "red")
        except ValueError:
            self.log_message("✗ Invalid angle value", "red")
        except Exception as e:
            self.log_message(f"✗ Error setting angle: {str(e)}", "red")

    def set_stepper_frequency(self):
        """Set stepper motor frequency"""
        if not self.pol_controller:
            self.log_message("✗ Not connected to STM32", "red")
            return
        
        try:
            frequency = float(self.stepper_entry.get())
            
            hardware_driver = self.pol_controller.driver
            if hasattr(hardware_driver, 'stm'):
                success = hardware_driver.stm.send_cmd_set_frequency(frequency, is_stepper=True)
                if success:
                    self.log_message(f"✓ Set stepper frequency: {frequency} Hz", "green")
                    self.stepper_entry.delete(0, 'end')
                else:
                    self.log_message("✗ Failed to set stepper frequency", "red")
            else:
                self.log_message("✗ STM32 interface not available", "red")
        except ValueError:
            self.log_message("✗ Invalid frequency value", "red")
        except Exception as e:
            self.log_message(f"✗ Error setting stepper frequency: {str(e)}", "red")

    def set_operation_period(self):
        """Set operation period"""
        if not self.pol_controller:
            self.log_message("✗ Not connected to STM32", "red")
            return
        
        try:
            period = float(self.period_entry.get())
            
            hardware_driver = self.pol_controller.driver
            if hasattr(hardware_driver, 'stm'):
                success = hardware_driver.stm.send_cmd_set_frequency(period, is_stepper=False)
                if success:
                    self.log_message(f"✓ Set operation period: {period} s", "green")
                    self.period_entry.delete(0, 'end')
                else:
                    self.log_message("✗ Failed to set operation period", "red")
            else:
                self.log_message("✗ STM32 interface not available", "red")
        except ValueError:
            self.log_message("✗ Invalid period value", "red")
        except Exception as e:
            self.log_message(f"✗ Error setting operation period: {str(e)}", "red")
            
            
    def get_angle_for_state(self, basis: Basis, bit: Bit) -> int:
        """Get angle for polarization state"""
        if basis == Basis.Z:
            return 0 if bit == Bit.ZERO else 90
        else:  # Basis.X
            return 45 if bit == Bit.ZERO else 135

    def get_state_name(self, basis: Basis, bit: Bit) -> str:
        """Get state name"""
        if basis == Basis.Z:
            return "H" if bit == Bit.ZERO else "V"
        else:  # Basis.X
            return "D" if bit == Bit.ZERO else "A"

    def update_state_display(self):
        """Update the state display"""
        try:
            if self.is_connected and self.pol_controller:
                state_info = self.pol_controller.get_current_state()
                
                state_text = "=== Current Polarization State ===\n"
                for key, value in state_info.items():
                    if key == 'jones_vector':
                        state_text += f"{key}: [{value[0]:.3f}, {value[1]:.3f}]\n"
                    else:
                        state_text += f"{key}: {value}\n"
                        
                # Add current GUI settings
                angle = self.get_angle_for_state(self.current_basis, self.current_bit)
                state_name = self.get_state_name(self.current_basis, self.current_bit)
                state_text += f"\nGUI Settings:\n"
                state_text += f"Basis: {self.current_basis.name}\n"
                state_text += f"Bit: {self.current_bit.value}\n"
                state_text += f"State: {state_name} ({angle}°)\n"
                
            else:
                state_text = "=== Not Connected ===\n"
                state_text += "Connect to hardware to see state information.\n\n"
                
                # Show GUI settings anyway
                angle = self.get_angle_for_state(self.current_basis, self.current_bit)
                state_name = self.get_state_name(self.current_basis, self.current_bit)
                state_text += f"GUI Settings:\n"
                state_text += f"Basis: {self.current_basis.name}\n"
                state_text += f"Bit: {self.current_bit.value}\n"
                state_text += f"State: {state_name} ({angle}°)\n"
                
            self.state_text.delete("1.0", "end")
            self.state_text.insert("1.0", state_text)
            
        except Exception as e:
            self.log_message(f"Error updating state display: {str(e)}")


    def log_message(self, message: str):
        """Add a message to the log"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Insert at the end and scroll to bottom
        self.log_text.insert("end", log_entry)
        self.log_text.see("end")

    def on_closing(self):
        """Handle window closing"""
        self.monitoring = False
        if self.is_connected:
            self.disconnect_hardware()
        self.destroy()


if __name__ == "__main__":
    # Set CustomTkinter appearance
    ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
    
    # Create and run the application
    app = PolarizationControllerGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
