"""
Improved GUI Demo for Polarization Controller Hardware Interface using CustomTkinter
Fixed threading issues that cause GUI slowdown after hardware connection
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
from typing import Optional, Tuple
from decimal import Decimal, localcontext, InvalidOperation
import logging
from concurrent.futures import ThreadPoolExecutor
import functools

from src.alice.polarization.polarization_controller import PolarizationController, create_polarization_controller_with_hardware
from src.alice.polarization.polarization_hardware import FREQUENCY_LIMIT, PERIOD_LIMIT
from src.alice.laser.laser_simulator import SimulatedLaserDriver
from src.alice.laser.laser_controller import LaserController, create_laser_controller_with_hardware
from src.alice.laser.hardware_laser.digilent_digital_interface import list_digital_devices
from src.utils.data_structures import Basis, Bit, Pulse, LaserInfo
import serial.tools.list_ports

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USE_SIMULATION = False  # Set to True to use simulated laser driver

def run_in_background(func):
    """Decorator to run hardware operations in background thread"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        def worker():
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                # Schedule GUI update from main thread
                self.after(0, lambda: self.log_message(f"Error in {func.__name__}: {str(e)}"))
        
        # Run in thread pool to avoid blocking GUI
        if hasattr(self, 'thread_pool'):
            self.thread_pool.submit(worker)
        else:
            threading.Thread(target=worker, daemon=True).start()
    
    return wrapper


class PolarizationLaserControllerGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Thread pool for hardware operations
        self.thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="HardwareOp")
        
        # GUI update queue for thread-safe communication
        self.gui_update_queue = Queue()
        
        # Scrollable window
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True)
        self.scrollable_frame = ctk.CTkScrollableFrame(self.main_frame, width=1100, height=600)
        # self.scrollable_frame.pack(fill="both", expand=True)
        
        self.title("Polarization Laser Controller Hardware Interface for Testing Alice")
        self.geometry("1100x600")

        # Controller
        self.pol_controller: Optional[PolarizationController] = None
        self.laser_controller: Optional[LaserController] = None
        
        # Current state
        self.current_basis = Basis.Z
        self.current_bit = Bit.ZERO
        self.current_angle = 0
        self.is_connected = False
        self._connecting = False  # Prevent multiple connection attempts
        self.laser_connected = False
        self._laser_connecting = False
        self.laser_device_mapping: dict[str, int] = {}
        self.laser_in_continuous = False
        
        # Setup GUI
        self.setup_gui()
        
        # Start GUI update processor
        self.process_gui_updates()
        
        # Optional: Start lightweight monitoring
        self.start_status_monitoring()

    def process_gui_updates(self):
        """Process GUI updates from background threads"""
        try:
            while True:
                try:
                    update_func = self.gui_update_queue.get_nowait()
                    update_func()
                except Empty:
                    break
        except Exception as e:
            logger.error(f"Error processing GUI updates: {e}")
        
        # Schedule next update check
        self.after(50, self.process_gui_updates)  # Check every 50ms

    def schedule_gui_update(self, func):
        """Schedule a GUI update from background thread"""
        self.gui_update_queue.put(func)

    def setup_gui(self):
        """Setup the GUI layout"""
        # Configure the scrollable frame to use grid layout for two columns
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=1)
        self.scrollable_frame.grid_rowconfigure(0, weight=1)
        
        # Create left and right container frames
        self.left_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        # self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=10)
        
        self.right_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        # self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)
        
        # Build the UI sections
        self.setup_gui_leftside()
        self.setup_gui_rightside()
        # Initial state update
        self.log_message("GUI initialized. Connect to hardware to begin.")

        # Default values
        self.combobox.set("COM4" if "COM4" in self.get_com_ports() else self.get_com_ports()[0])
        self.laser_channel_entry.insert(0, "8")
        self.laser_duty_entry.insert(0, "10")
        self.laser_freq_entry.insert(0, "1000")
        self.laser_frame_count_entry.insert(0, "10")
        self.laser_frame_time_entry.insert(0, "1")  # Time in milliseconds (1000 Hz = 1 ms period)
        self.laser_cont_freq_entry.insert(0, "1000")
        self.laser_frame_repeat_nb_entry.insert(0, "10")
        self.laser_frame_repeat_interval_entry.insert(0, "1")


        # Bind entry callbacks for frequency/time synchronization
        self.laser_freq_entry.bind('<FocusOut>', self._on_freq_changed)
        self.laser_freq_entry.bind('<Return>', self._on_freq_changed)
        self.laser_frame_time_entry.bind('<FocusOut>', self._on_time_changed)
        self.laser_frame_time_entry.bind('<Return>', self._on_time_changed)
        self.laser_cont_freq_entry.bind('<FocusOut>', self._on_cont_freq_changed)
        self.laser_cont_freq_entry.bind('<Return>', self._on_cont_freq_changed)

    def setup_gui_leftside(self):
        """Setup the left side of the GUI"""
                        
        # Connection Frame
        conn_frame = ctk.CTkFrame(self.left_frame)
        conn_frame.pack(pady=10, padx=10, fill="x")
        
        conn_label = ctk.CTkLabel(conn_frame, text="Polarization Hardware Control", 
                                 font=ctk.CTkFont(size=16, weight="bold"))
        conn_label.pack(pady=(10, 5))

        conn_frame_2 = ctk.CTkFrame(conn_frame)
        conn_frame_2.pack(fill="x")
        
        # COM port selection
        com_frame = ctk.CTkFrame(conn_frame_2, fg_color="transparent")
        com_frame.pack(pady=5, padx=10, fill="x")
        
        self.com_label = ctk.CTkLabel(com_frame, text="Select COM Port:")
        self.com_label.pack(side="left", padx=(10, 5))
        
        self.combobox = ctk.CTkComboBox(com_frame, values=self.get_com_ports(), width=150)
        self.combobox.pack(side="left", padx=5, expand=True, fill="x")

        self.refresh_button = ctk.CTkButton(com_frame, text="Refresh Ports", command=self.refresh_com_ports_async, width=80)
        self.refresh_button.pack(side="left", padx=5, expand=True, fill="x")
        
        # Connect button
        conn_status_frame = ctk.CTkFrame(conn_frame_2, fg_color="transparent")
        conn_status_frame.pack(pady=5, padx=10, fill="x")

        self.connect_button = ctk.CTkButton(conn_status_frame, text="Connect STM", command=self.toggle_connection_async)
        self.connect_button.pack(pady=0, side="left", expand=True, fill="x")
        
        # Connection status
        self.status_label = ctk.CTkLabel(conn_status_frame, text="● Disconnected", 
                                           text_color="red", font=ctk.CTkFont(size=14, weight="bold"))
        self.status_label.pack(pady=(0, 0), side="right", expand=True, fill="x")
        status_indicator = ctk.CTkLabel(conn_status_frame, text="Status:", font=ctk.CTkFont(size=14, weight="bold"))
        status_indicator.pack(pady=(0, 0), side="right", expand=True)

        #####################################################################################
        
        # Polarization Control Frame
        control_frame = ctk.CTkFrame(self.left_frame)
        control_frame.pack(pady=(0,10), padx=10, fill="x")
        
        control_label = ctk.CTkLabel(control_frame, text="Polarization Angle Control", 
                                   font=ctk.CTkFont(size=14, weight="bold"))
        control_label.pack()
        
        # Control frame 2 for a better layout
        control_frame_2 = ctk.CTkFrame(control_frame)
        control_frame_2.pack(fill="x")
        
        # Quick preset buttons
        preset_frame = ctk.CTkFrame(control_frame_2, fg_color="transparent")
        preset_frame.pack(pady=5, padx=10, fill="x", expand=True)
        
        ctk.CTkLabel(preset_frame, text="Presets:").pack(pady=5, padx=(0,5), side="left", expand=True)
        
        self.h_button = ctk.CTkButton(preset_frame, text="H (0°)", 
                                    command=lambda: self.set_preset_async(Basis.Z, Bit.ZERO), 
                                    state="disabled", width=80)
        self.h_button.pack(side="left", padx=2, expand=True)
        
        self.v_button = ctk.CTkButton(preset_frame, text="V (90°)", 
                                    command=lambda: self.set_preset_async(Basis.Z, Bit.ONE), 
                                    state="disabled", width=80)
        self.v_button.pack(side="left", padx=2, expand=True)
        
        self.d_button = ctk.CTkButton(preset_frame, text="D (45°)", 
                                    command=lambda: self.set_preset_async(Basis.X, Bit.ZERO), 
                                    state="disabled", width=80)
        self.d_button.pack(side="left", padx=2, expand=True)
        
        self.a_button = ctk.CTkButton(preset_frame, text="A (135°)", 
                                    command=lambda: self.set_preset_async(Basis.X, Bit.ONE), 
                                    state="disabled", width=80)
        self.a_button.pack(side="left", padx=2, expand=True)
        
        # Random QRNG control
        qrng_frame = ctk.CTkFrame(control_frame_2, fg_color="transparent")
        qrng_frame.pack(pady=5, padx=10, fill="x", expand=True)
        
        qrng_label = ctk.CTkLabel(qrng_frame, text="QRNG Set")
        qrng_label.pack(pady=5, side="left", expand=True)
        
        self.qrng_button = ctk.CTkButton(qrng_frame, text="Set Random Polarization", 
                                       command=self.set_polarization_qrng_async, state="disabled")
        self.qrng_button.pack(pady=5, side="left", expand=True)

        # Small Status display of current state
        status_frame = ctk.CTkFrame(qrng_frame, fg_color="transparent")
        status_frame.pack(side="right", pady=5, padx=10, fill="x", expand=True)
        
        self.last_state_var = ctk.StringVar(
            value=f"Last State: {self.current_basis.name} / {self.current_bit.value} ({self.current_angle}°)"
        )
        self.last_state_label = ctk.CTkLabel(
            status_frame,
            textvariable=self.last_state_var
        )
        self.last_state_label.pack(pady=5, side="left", expand=True)

        # Direct STM32 Control
        stm32_frame = ctk.CTkFrame(control_frame_2, fg_color="transparent")
        stm32_frame.pack(pady=(0,5), padx=10, fill="x")
        
        stm32_label = ctk.CTkLabel(stm32_frame, text="Multiple Polarization numbers (0-3, comma separated):")
        stm32_label.pack(pady=(0, 0))

        self.polarization_entry = ctk.CTkEntry(stm32_frame, placeholder_text="0,1,2,3")
        self.polarization_entry.pack(pady=10, padx=(10,5), fill="x", side="left", expand=True)
        
        self.send_stm32_button = ctk.CTkButton(stm32_frame, text="Send to STM32", 
                                              command=self.send_polarization_numbers_async,
                                              state="disabled")
        self.send_stm32_button.pack(pady=10, padx=(5,10), side="right")

        #####################################################################################

        # Other STM32 Controls Frame (inspired by main.py)
        others_frame = ctk.CTkFrame(self.left_frame)
        others_frame.pack(pady=(0,10), padx=10, fill="x")
        
        others_label = ctk.CTkLabel(others_frame, text="Other STM32 Controls", 
                                    font=ctk.CTkFont(size=14, weight="bold"))
        others_label.pack()

        # advance frame 2 for better layout
        others_frame_2 = ctk.CTkFrame(others_frame)
        others_frame_2.pack(fill="x")
        
        # Device selection
        device_frame = ctk.CTkFrame(others_frame_2, fg_color="transparent")
        device_frame.pack(pady=5, padx=10, fill="x")
        
        device_label = ctk.CTkLabel(device_frame, text="Device:")
        device_label.pack(side="left", padx=(10, 5))
        
        self.device_var = ctk.StringVar(value="1")
        self.device_radio1 = ctk.CTkRadioButton(device_frame, text="Linear (1)", 
                                         variable=self.device_var, value="1", state="disable")
        self.device_radio1.pack(side="left", padx=5)
        self.device_radio2 = ctk.CTkRadioButton(device_frame, text="HWP (2)", 
                                         variable=self.device_var, value="2", state="disable")
        self.device_radio2.pack(side="left", padx=5)
        
        self.set_device_button = ctk.CTkButton(device_frame, text="Set Device", 
                                             command=self.set_polarization_device_async, 
                                             state="disabled", width=100)
        self.set_device_button.pack(side="right", padx=10)
        
        # Angle control
        angle_frame = ctk.CTkFrame(others_frame_2, fg_color="transparent")
        angle_frame.pack(pady=5, padx=10, fill="x")
        
        angle_label = ctk.CTkLabel(angle_frame, text="Angle (0-360°):")
        angle_label.pack(side="left", padx=(10, 5))
        
        self.angle_entry = ctk.CTkEntry(angle_frame, width=80, placeholder_text="45")
        self.angle_entry.pack(side="left", padx=5)
        
        self.offset_switch = ctk.CTkSwitch(angle_frame, text="Set as Offset", state="disable")
        self.offset_switch.pack(side="left", padx=10)
        
        self.set_angle_button = ctk.CTkButton(angle_frame, text="Set Angle", 
                                            command=self.set_angle_direct_async, 
                                            state="disabled", width=100)
        self.set_angle_button.pack(side="right", padx=10)
        
        # Frequency controls
        freq_frame = ctk.CTkFrame(others_frame_2, fg_color="transparent")
        freq_frame.pack(pady=5, padx=10, fill="x")
        
        # Stepper frequency
        stepper_frame = ctk.CTkFrame(freq_frame, fg_color="transparent")
        stepper_frame.pack(pady=2, fill="x")
        
        stepper_label = ctk.CTkLabel(stepper_frame, text="Stepper Frequency (1-1000 Hz):")
        stepper_label.pack(side="left", padx=(10, 5))
        
        self.stepper_entry = ctk.CTkEntry(stepper_frame, width=80, placeholder_text="500")
        self.stepper_entry.pack(side="left", padx=5)
        
        self.set_stepper_button = ctk.CTkButton(stepper_frame, text="Set Stepper Freq", 
                                              command=self.set_stepper_frequency_async, 
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
                                             command=self.set_operation_period_async, 
                                             state="disabled", width=120)
        self.set_period_button.pack(side="right", padx=10)

        self.enable_controls(False)

        #####################################################################################

    def setup_gui_rightside(self):
        """Setup the right side of the GUI"""

        # Laser Control Frame
        laser_frame = ctk.CTkFrame(self.right_frame)
        laser_frame.pack(pady=10, padx=10, fill="x")

        laser_label = ctk.CTkLabel(laser_frame, text="Laser Hardware Control",
                                   font=ctk.CTkFont(size=16, weight="bold"))
        laser_label.pack(pady=(10, 5))

        laser_conn_status_frame = ctk.CTkFrame(laser_frame)
        laser_conn_status_frame.pack(pady=5, padx=10, fill="x")

        # Laser device selection
        laser_conn_frame = ctk.CTkFrame(laser_conn_status_frame, fg_color="transparent")
        laser_conn_frame.pack(pady=5, padx=10, fill="x")

        device_values = self.get_laser_device_labels()
        self.laser_device_label = ctk.CTkLabel(laser_conn_frame, text="Digilent Device:")
        self.laser_device_label.pack(side="left", padx=(10, 5))

        self.laser_device_combobox = ctk.CTkComboBox(
            laser_conn_frame,
            values=device_values if device_values else ["No devices found"],
            width=220
        )
        self.laser_device_combobox.pack(side="left", padx=5, expand=True, fill="x")
        if device_values:
            self.laser_device_combobox.set(device_values[0])
        else:
            self.laser_device_combobox.set("No devices found")

        self.laser_refresh_button = ctk.CTkButton(
            laser_conn_frame,
            text="Refresh Devices",
            command=self.refresh_laser_devices_async,
            width=130
        )
        self.laser_refresh_button.pack(side="left", padx=5)

        # Laser connection status and controls
        laser_status_frame = ctk.CTkFrame(laser_conn_status_frame, fg_color="transparent")
        laser_status_frame.pack(pady=5, padx=10, fill="x")

        channel_label = ctk.CTkLabel(laser_status_frame, text="Channel:")
        channel_label.pack(side="left", padx=(10, 5))

        self.laser_channel_entry = ctk.CTkEntry(laser_status_frame, width=60, placeholder_text="8")
        self.laser_channel_entry.pack(side="left", padx=(0, 5))

        self.laser_connect_button = ctk.CTkButton(
            laser_status_frame,
            text="Connect Laser",
            command=self.toggle_laser_connection_async
        )
        self.laser_connect_button.pack(side="left", padx=(0, 5), expand=True, fill="x")

        self.laser_status_label = ctk.CTkLabel(
            laser_status_frame,
            text="● Disconnected",
            text_color="red",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.laser_status_label.pack(side="right", padx=(5, 0))

        #####################################################################################

        laser_label = ctk.CTkLabel(laser_frame, text="Pulse Control",
                                   font=ctk.CTkFont(size=14, weight="bold"))
        laser_label.pack()

        laser_pulse_control_frame = ctk.CTkFrame(laser_frame)
        laser_pulse_control_frame.pack(pady=(0,5), padx=10, fill="x")

        # Pulse parameter controls
        laser_param_frame = ctk.CTkFrame(laser_pulse_control_frame, fg_color="transparent")
        laser_param_frame.pack(pady=5, padx=10, fill="x")

        laser_param_frame.grid_columnconfigure(5, weight=1)

        ctk.CTkLabel(laser_param_frame, text="Duty Cycle (%)").grid(
            row=0, column=0, padx=5, pady=2, sticky="e"
        )
        self.laser_duty_entry = ctk.CTkEntry(laser_param_frame, width=50, placeholder_text="10")
        self.laser_duty_entry.grid(row=0, column=1, padx=5, pady=2, sticky="w")

        ctk.CTkLabel(laser_param_frame, text="Frequency (Hz)").grid(
            row=0, column=2, padx=5, pady=2, sticky="e"
        )
        self.laser_freq_entry = ctk.CTkEntry(laser_param_frame, width=100, placeholder_text="1000")
        self.laser_freq_entry.grid(row=0, column=3, padx=5, pady=2, sticky="w")

        self.laser_set_params_button = ctk.CTkButton(
            laser_param_frame,
            text="Set Parameters",
            command=self.set_laser_pulse_parameters_async,
            state="disabled"
        )
        self.laser_set_params_button.grid(row=0, column=5, padx=(10, 0), pady=2, sticky="ew")

        # Row 0: Pulse Train
        sequences_frame = ctk.CTkFrame(laser_pulse_control_frame, fg_color="transparent")
        sequences_frame.pack(pady=5, padx=10, fill="x")
        sequences_frame.grid_columnconfigure(0, weight=0)
        sequences_frame.grid_columnconfigure(1, weight=0)
        sequences_frame.grid_columnconfigure(2, weight=0)
        sequences_frame.grid_columnconfigure(3, weight=0)
        sequences_frame.grid_columnconfigure(4, weight=1)

        ctk.CTkLabel(sequences_frame, text="Pulse Train -> Nb:").grid(
            row=0, column=0, padx=(0, 5), pady=2, sticky="e"
        )
        self.laser_frame_count_entry = ctk.CTkEntry(sequences_frame, width=70, placeholder_text="5")
        self.laser_frame_count_entry.grid(row=0, column=1, padx=5, pady=2, sticky="w")

        ctk.CTkLabel(sequences_frame, text="Time(ms):").grid(row=0, column=2, padx=0, pady=2, sticky="w")
        self.laser_frame_time_entry = ctk.CTkEntry(sequences_frame, width=90, placeholder_text="1")
        self.laser_frame_time_entry.grid(row=0, column=3, padx=5, pady=2, sticky="w")

        self.laser_frame_repeat_switch = ctk.CTkSwitch(sequences_frame, text="Repeat? Nb:", command=self.on_repeat_switch_toggle)
        self.laser_frame_repeat_switch.grid(row=1, column=0, padx=(0, 5), pady=2, sticky="e")
        self.laser_frame_repeat_nb_entry = ctk.CTkEntry(sequences_frame, width=70, placeholder_text="10")
        self.laser_frame_repeat_nb_entry.grid(row=1, column=1, padx=5, pady=2, sticky="w")

        ctk.CTkLabel(sequences_frame, text="Interval(s):").grid(row=1, column=2, padx=0, pady=2, sticky="w")
        self.laser_frame_repeat_interval_entry = ctk.CTkEntry(sequences_frame, width=90, placeholder_text="1")
        self.laser_frame_repeat_interval_entry.grid(row=1, column=3, padx=5, pady=2, sticky="w")

        self.laser_send_frame_button = ctk.CTkButton(
            sequences_frame,
            text="Send Pulse Train",
            command=self.send_laser_frame_async,
            state="disabled"
        )
        self.laser_send_frame_button.grid(row=0, rowspan=2, column=4, padx=5, pady=2, sticky="ewns")


        # Row 1: Continuous
        sequences_frame_2 = ctk.CTkFrame(laser_pulse_control_frame, fg_color="transparent")
        sequences_frame_2.pack(pady=5, padx=10, fill="x")        
        sequences_frame_2.grid_columnconfigure(0, weight=0)
        sequences_frame_2.grid_columnconfigure(1, weight=0)
        sequences_frame_2.grid_columnconfigure(2, weight=1)
        sequences_frame_2.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(sequences_frame_2, text="Continuous (Freq):").grid(
            row=0, column=0, padx=(0, 5), pady=2, sticky="e"
        )
        self.laser_cont_freq_entry = ctk.CTkEntry(sequences_frame_2, width=90, placeholder_text="500")
        self.laser_cont_freq_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.laser_start_cont_button = ctk.CTkButton(
            sequences_frame_2,
            text="Start",
            command=self.start_laser_continuous_async,
            state="disabled"
        )
        self.laser_start_cont_button.grid(row=0, column=2, padx=5, pady=2, sticky="ew")
        self.laser_stop_cont_button = ctk.CTkButton(
            sequences_frame_2,
            text="Stop",
            command=self.stop_laser_continuous_async,
            state="disabled"
        )
        self.laser_stop_cont_button.grid(row=0, column=3, padx=(5, 0), pady=2, sticky="ew")

        # Row 2: Single Pulse and Status
        sequences_frame_3 = ctk.CTkFrame(laser_pulse_control_frame, fg_color="transparent")
        sequences_frame_3.pack(pady=5, padx=10, fill="x")
        sequences_frame_3.grid_columnconfigure(0, weight=6)
        sequences_frame_3.grid_columnconfigure(1, weight=1)

        self.laser_single_button = ctk.CTkButton(
            sequences_frame_3,
            text="Trigger Single Pulse",
            command=self.trigger_laser_single_async,
            state="disabled"
        )
        self.laser_single_button.grid(row=0, column=0, padx=5, pady=2, sticky="ew")

        self.laser_status_button = ctk.CTkButton(
            sequences_frame_3,
            text="Status",
            command=self.get_laser_status_async,
            state="disabled"
        )
        self.laser_status_button.grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        # Ensure initial laser controls reflect disconnected state
        self.enable_laser_controls(False)
        
        #####################################################################################

        # Log Frame
        log_frame = ctk.CTkFrame(self.right_frame)
        log_frame.pack(pady=(0,10), padx=10, fill="both", expand=True)
        
        # # Log Frame (spans both columns)
        # log_frame = ctk.CTkFrame(self.scrollable_frame)
        # log_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        # Make sure row 1 expands to fill available space
        # self.scrollable_frame.grid_rowconfigure(1, weight=1)
        
        log_label = ctk.CTkLabel(log_frame, text="Activity Log", 
                               font=ctk.CTkFont(size=16, weight="bold"))
        log_label.pack()
        
        self.log_text = ctk.CTkTextbox(log_frame, height=150)
        self.log_text.pack(pady=5, padx=5, fill="both", expand=True)
        

    def get_com_ports(self):
        """Get available COM ports (lightweight, can run on main thread)"""
        try:
            ports = serial.tools.list_ports.comports()
            return [port.device for port in ports] if ports else ["No ports found"]
        except Exception as e:
            logger.error(f"Error getting COM ports: {e}")
            return ["Error getting ports"]

    @run_in_background
    def refresh_com_ports_async(self):
        """Refresh the COM port list in background"""
        new_ports = self.get_com_ports()
        
        def update_gui():
            self.combobox.configure(values=new_ports)
            if new_ports and new_ports[0] != "No ports found":
                self.combobox.set(new_ports[0])
            self.log_message(f"Refreshed COM ports: {len(new_ports)} found")
        
        self.schedule_gui_update(update_gui)

    def toggle_connection_async(self):
        """Toggle connection in background"""
        if self._connecting:
            self.log_message("Connection already in progress...")
            return
            
        if not self.is_connected:
            self.connect_hardware_async()
        else:
            self.disconnect_hardware_async()

    @run_in_background
    def connect_hardware_async(self):
        """Connect to hardware in background thread"""
        if self._connecting:
            return
            
        self._connecting = True
        
        def update_connecting_status():
            self.connect_button.configure(text="Connecting...", state="disabled")
            self.status_label.configure(text="● Connecting...", text_color="orange")
        
        self.schedule_gui_update(update_connecting_status)
        
        try:
            com_port = self.combobox.get()
            if not com_port or com_port == "No ports found":
                def update_error():
                    self.status_label.configure(text="✗ Please select a valid COM port", text_color="red")
                    self.connect_button.configure(text="Connect", state="normal")
                    self.log_message("✗ Please select a valid COM port")
                    self._connecting = False
                
                self.schedule_gui_update(update_error)
                return
            
            def update_connecting():
                self.log_message(f"Connecting to hardware on {com_port}...")
            
            self.schedule_gui_update(update_connecting)
            
            # This is the potentially slow operation - now in background
            self.pol_controller = create_polarization_controller_with_hardware(com_port=com_port)
            self.pol_controller.initialize()
            
            # Connection successful
            def update_success():
                self.is_connected = True
                self._connecting = False
                self.status_label.configure(text=f"● Connected to {com_port}", text_color="green")
                self.connect_button.configure(text="Disconnect", state="normal")
                self.enable_controls(True)
                self.log_message(f"Successfully connected to hardware on {com_port}")
            
            self.schedule_gui_update(update_success)
            
        except Exception as e:
            def update_error():
                self._connecting = False
                self.status_label.configure(text=f"Connection failed: {str(e)}", text_color="red")
                self.connect_button.configure(text="Connect", state="normal")
                self.log_message(f"Connection failed: {str(e)}")
                self.is_connected = False
            
            self.schedule_gui_update(update_error)

    @run_in_background
    def disconnect_hardware_async(self):
        """Disconnect from hardware in background"""
        try:
            if self.pol_controller:
                self.pol_controller.shutdown()
                self.pol_controller = None
            
            def update_disconnected():
                self.is_connected = False
                self._connecting = False
                self.status_label.configure(text="● Disconnected", text_color="red")
                self.connect_button.configure(text="Connect", state="normal")
                self.enable_controls(False)
                self.log_message("Disconnected from hardware")
            
            self.schedule_gui_update(update_disconnected)
            
        except Exception as e:
            def update_error():
                self.log_message(f"Error during disconnect: {str(e)}")
            
            self.schedule_gui_update(update_error)

    def enable_controls(self, enabled: bool):
        """Enable or disable control buttons (runs on main thread)"""
        state = "normal" if enabled else "disabled"
        device_state = "normal" if not enabled else "disabled"
        
        self.refresh_button.configure(state=device_state)
        self.combobox.configure(state=device_state)

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
        self.device_radio1.configure(state=state)
        self.device_radio2.configure(state=state)
        self.angle_entry.configure(state=state)
        self.offset_switch.configure(state=state)
        self.polarization_entry.configure(state=state)
        self.stepper_entry.configure(state=state)
        self.period_entry.configure(state=state)

    @run_in_background
    def set_preset_async(self, basis: Basis, bit: Bit):
        """Set a preset polarization in background"""
        if not self.is_connected or not self.pol_controller:
            self.schedule_gui_update(lambda: self.log_message("Not connected to hardware"))
            return
            
        try:
            # Hardware operation in background
            self.pol_controller.set_polarization_manually(basis, bit)
            
            # Update GUI from main thread
            def update_success():
                self.current_basis = basis
                self.current_bit = bit
                
                angle = self.get_angle_for_state(basis, bit)
                state_name = self.get_state_name(basis, bit)
                
                # Update the last state display
                self.last_state_var.set(f"Last State: {self.current_basis.name} / {self.current_bit.value} ({angle}°)")

                self.log_message(f"✓ Set preset: {state_name} ({angle}°)")
            
            self.schedule_gui_update(update_success)
            
        except Exception as e:
            self.schedule_gui_update(lambda: self.log_message(f"✗ Error setting preset: {str(e)}"))

    @run_in_background
    def set_polarization_qrng_async(self):
        """Set random polarization using QRNG in background"""
        if not self.is_connected or not self.pol_controller:
            self.schedule_gui_update(lambda: self.log_message("Not connected to hardware"))
            return
            
        try:
            # Hardware operation in background
            output = self.pol_controller.set_polarization_from_qrng()
            
            # Update GUI from main thread
            def update_success():
                self.current_basis = output.basis
                self.current_bit = output.bit

                # Update the last state display
                self.last_state_var.set(f"Last State: {self.current_basis.name} / {self.current_bit.value} ({output.angle_degrees}°)")

                self.log_message(f"✓ QRNG set: {output.basis.name} basis, bit {output.bit.value} → "
                               f"{output.polarization_state.name} ({output.angle_degrees}°)")
            
            self.schedule_gui_update(update_success)
            
        except Exception as e:
            self.schedule_gui_update(lambda: self.log_message(f"✗ Error with QRNG: {str(e)}"))

    @run_in_background
    def send_polarization_numbers_async(self):
        """Send polarization numbers directly to STM32 in background"""
        if not self.is_connected or not self.pol_controller:
            self.schedule_gui_update(lambda: self.log_message("Not connected to hardware"))
            return
            
        try:
            text = self.polarization_entry.get()
            if not text:
                self.schedule_gui_update(lambda: self.log_message("✗ Please enter polarization numbers"))
                return
            
            numbers = [int(num.strip()) for num in text.split(",") if num.strip().isdigit()]
            if not numbers:
                self.schedule_gui_update(lambda: self.log_message("✗ Invalid input. Enter comma-separated numbers (0-3)"))
                return
            
            # Hardware operation in background
            self.pol_controller.set_polarization_multiple_states(numbers)
            
            # Update GUI from main thread
            def update_success():
                self.log_message(f"✓ Sent polarization numbers to STM32: {numbers}")
            
            self.schedule_gui_update(update_success)
            
        except Exception as e:
            self.schedule_gui_update(lambda: self.log_message(f"✗ Error sending polarization numbers: {str(e)}"))

    @run_in_background
    def set_polarization_device_async(self):
        """Set polarization device (linear polarizer or half wave plate) in background"""
        if not self.is_connected or not self.pol_controller:
            self.schedule_gui_update(lambda: self.log_message("Not connected to hardware"))
            return
            
        try:
            device = int(self.device_var.get())
            if device not in [1, 2]:
                self.schedule_gui_update(lambda: self.log_message("✗ Invalid device selection"))
                return
            
            # Hardware operation in background
            self.pol_controller.driver.set_polarization_device(device)
            
            # Update GUI from main thread
            def update_success():
                device_name = "Linear Polarizer" if device == 1 else "Half Wave Plate"
                self.log_message(f"✓ Set polarization device: {device_name} ({device})")
            
            self.schedule_gui_update(update_success)
            
        except Exception as e:
            self.schedule_gui_update(lambda: self.log_message(f"✗ Error setting device: {str(e)}"))

    @run_in_background
    def set_angle_direct_async(self):
        """Set angle directly in background"""
        if not self.is_connected or not self.pol_controller:
            self.schedule_gui_update(lambda: self.log_message("Not connected to hardware"))
            return
            
        try:
            angle_text = self.angle_entry.get()
            if not angle_text or not angle_text.isdigit():
                self.schedule_gui_update(lambda: self.log_message("✗ Please enter a valid angle (0-360)"))
                return
            
            angle = int(angle_text)
            if angle < 0 or angle > 360:
                self.schedule_gui_update(lambda: self.log_message("✗ Angle must be between 0 and 360 degrees"))
                return

            set_as_offset = bool(self.offset_switch.get())

            # Hardware operation in background
            self.pol_controller.driver.set_angle_direct(angle, set_as_offset)
            
            # Update GUI from main thread
            def update_success():
                offset_text = " (as offset)" if set_as_offset else ""
                self.log_message(f"✓ Set angle to {angle}°{offset_text}")
            
            self.schedule_gui_update(update_success)
            
        except Exception as e:
            self.schedule_gui_update(lambda: self.log_message(f"✗ Error setting angle: {str(e)}"))

    @run_in_background
    def set_stepper_frequency_async(self):
        """Set stepper frequency in background"""
        if not self.is_connected or not self.pol_controller:
            self.schedule_gui_update(lambda: self.log_message("Not connected to hardware"))
            return
            
        try:
            freq_text = self.stepper_entry.get()
            if not freq_text or not freq_text.isdigit():
                self.schedule_gui_update(lambda: self.log_message(f"✗ Please enter a valid frequency (1-{FREQUENCY_LIMIT} Hz)"))
                return
            
            frequency = int(freq_text)
            if frequency < 1 or frequency > FREQUENCY_LIMIT:
                self.schedule_gui_update(lambda: self.log_message(f"✗ Frequency must be between 1 and {FREQUENCY_LIMIT} Hz"))
                return

            # Hardware operation in background
            self.pol_controller.driver.set_stepper_frequency(frequency)
            
            # Update GUI from main thread
            def update_success():
                self.log_message(f"✓ Set stepper frequency to {frequency} Hz")
            
            self.schedule_gui_update(update_success)
            
        except Exception as e:
            self.schedule_gui_update(lambda: self.log_message(f"✗ Error setting stepper frequency: {str(e)}"))

    @run_in_background
    def set_operation_period_async(self):
        """Set operation period in background"""
        if not self.is_connected or not self.pol_controller:
            self.schedule_gui_update(lambda: self.log_message("Not connected to hardware"))
            return
            
        try:
            period_text = self.period_entry.get()
            if not period_text or not period_text.isdigit():
                self.schedule_gui_update(lambda: self.log_message(f"✗ Please enter a valid period (1-{PERIOD_LIMIT} ms)"))
                return
            
            period = int(period_text)
            if period < 1 or period > PERIOD_LIMIT:
                self.schedule_gui_update(lambda: self.log_message(f"✗ Period must be between 1 and {PERIOD_LIMIT} ms"))
                return

            # Hardware operation in background
            self.pol_controller.driver.set_operation_period(period)
            
            # Update GUI from main thread
            def update_success():
                self.log_message(f"✓ Set operation period to {period} ms")
            
            self.schedule_gui_update(update_success)
            
        except Exception as e:
            self.schedule_gui_update(lambda: self.log_message(f"✗ Error setting operation period: {str(e)}"))
            
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

    #####################################################################################
    # Frequency/Time Synchronization Methods
    #####################################################################################

    def _format_decimal(self, d: Decimal, sci_low: Decimal = Decimal('1e-3'), sci_high: Decimal = Decimal('1e6')) -> str:
        """Format Decimal d using plain string with trimmed zeros, or scientific notation
        when |d| >= sci_high or (0 < |d| < sci_low). Avoid scientific notation otherwise.
        """
        if d.is_nan():
            return ""
        if d.is_zero():
            return "0"
        ad = abs(d)
        # Choose format
        if ad >= sci_high or ad < sci_low:
            # Scientific notation with trimmed mantissa zeros and compact exponent
            s = f"{d:.12e}"
            mantissa, exp = s.split('e')
            mantissa = mantissa.rstrip('0').rstrip('.')
            exp_sign = '-' if exp.startswith('-') else ''
            exp_digits = exp.lstrip('+-').lstrip('0') or '0'
            return f"{mantissa}e{exp_sign}{exp_digits}"
        else:
            # Plain decimal: trim trailing zeros only if there is a fractional part
            s = format(d, 'f')
            if '.' in s:
                s = s.rstrip('0').rstrip('.')
            # # Remove the left 0's before the first non-zero digit
            # s = s.lstrip('0')
            return s or '0'
        
    def _on_freq_changed(self, event=None):
        """Update time entry when frequency changes (freq in Hz -> time in ms)"""
        try:
            freq_text = self.laser_freq_entry.get().strip()
            if not freq_text:
                return
            
            # Use Decimal to parse and compute
            frequency_dec = Decimal(freq_text)
            if frequency_dec <= 0:
                return
            freq_text = self._format_decimal(frequency_dec)  # Normalize input
            
            # Calculate period in milliseconds: T = 1000 / f (use Decimal for precision)
            with localcontext() as ctx:
                ctx.prec = 50  # high precision to avoid rounding artifacts
                time_ms_dec = (Decimal(1000) / frequency_dec)
                time_ms_str = self._format_decimal(time_ms_dec)
            
            self.laser_freq_entry.delete(0, 'end')
            self.laser_freq_entry.insert(0, freq_text)  # Update to normalized input

            # Update time entry (avoid triggering callback)
            self.laser_frame_time_entry.unbind('<FocusOut>')
            self.laser_frame_time_entry.unbind('<Return>')
            self.laser_frame_time_entry.delete(0, 'end')
            self.laser_frame_time_entry.insert(0, time_ms_str)
            self.laser_frame_time_entry.bind('<FocusOut>', self._on_time_changed)
            self.laser_frame_time_entry.bind('<Return>', self._on_time_changed)
            
            # Also update continuous frequency entry
            self.laser_cont_freq_entry.unbind('<FocusOut>')
            self.laser_cont_freq_entry.unbind('<Return>')
            self.laser_cont_freq_entry.delete(0, 'end')
            self.laser_cont_freq_entry.insert(0, freq_text)
            self.laser_cont_freq_entry.bind('<FocusOut>', self._on_cont_freq_changed)
            self.laser_cont_freq_entry.bind('<Return>', self._on_cont_freq_changed)
            
        except (ValueError, InvalidOperation):
            pass  # Invalid input, ignore
    
    def _on_time_changed(self, event=None):
        """Update frequency entry when time changes (time in ms -> freq in Hz)"""
        try:
            time_text = self.laser_frame_time_entry.get().strip()
            if not time_text:
                return
            
            # Parse as Decimal to keep full precision of user input
            time_ms_dec = Decimal(time_text)
            if time_ms_dec <= 0:
                return
            
            time_text = self._format_decimal(time_ms_dec)  # Normalize input

            # Calculate frequency in Hz: f = 1000 / T (use Decimal for precision)
            with localcontext() as ctx:
                ctx.prec = 50
                freq_dec = (Decimal(1000) / time_ms_dec)
                freq_str = self._format_decimal(freq_dec)
            
            # Round freq to the nearest integer, and adapt the time accordingly
            freq_str = str(int(freq_dec.to_integral_value()))
            time_ms_str = self._format_decimal(Decimal(1000) / Decimal(freq_str))
            self.laser_frame_time_entry.delete(0, 'end')
            self.laser_frame_time_entry.insert(0, time_ms_str)

            # Update frequency entry (avoid triggering callback)
            self.laser_freq_entry.unbind('<FocusOut>')
            self.laser_freq_entry.unbind('<Return>')
            self.laser_freq_entry.delete(0, 'end')
            self.laser_freq_entry.insert(0, freq_str)
            self.laser_freq_entry.bind('<FocusOut>', self._on_freq_changed)
            self.laser_freq_entry.bind('<Return>', self._on_freq_changed)
            
            # Also update continuous frequency entry
            self.laser_cont_freq_entry.unbind('<FocusOut>')
            self.laser_cont_freq_entry.unbind('<Return>')
            self.laser_cont_freq_entry.delete(0, 'end')
            self.laser_cont_freq_entry.insert(0, freq_str)
            self.laser_cont_freq_entry.bind('<FocusOut>', self._on_cont_freq_changed)
            self.laser_cont_freq_entry.bind('<Return>', self._on_cont_freq_changed)
            
        except (ValueError, InvalidOperation):
            pass  # Invalid input, ignore
    
    def _on_cont_freq_changed(self, event=None):
        """Update main frequency when continuous frequency changes"""
        try:
            cont_freq_text = self.laser_cont_freq_entry.get().strip()
            if not cont_freq_text:
                return
            
            # Use Decimal to parse and compute
            frequency_dec = Decimal(cont_freq_text)
            if frequency_dec <= 0:
                return
            cont_freq_text = self._format_decimal(frequency_dec)  # Normalize input

            self.laser_cont_freq_entry.delete(0, 'end')
            self.laser_cont_freq_entry.insert(0, cont_freq_text)  # Update to normalized input
            
            # Update main frequency entry (avoid triggering callback)
            self.laser_freq_entry.unbind('<FocusOut>')
            self.laser_freq_entry.unbind('<Return>')
            self.laser_freq_entry.delete(0, 'end')
            self.laser_freq_entry.insert(0, cont_freq_text)
            self.laser_freq_entry.bind('<FocusOut>', self._on_freq_changed)
            self.laser_freq_entry.bind('<Return>', self._on_freq_changed)
            
            # Calculate and update time with high precision
            with localcontext() as ctx:
                ctx.prec = 50
                time_ms_dec = (Decimal(1000) / frequency_dec)
                time_ms_str = self._format_decimal(time_ms_dec)

            self.laser_frame_time_entry.unbind('<FocusOut>')
            self.laser_frame_time_entry.unbind('<Return>')
            self.laser_frame_time_entry.delete(0, 'end')
            self.laser_frame_time_entry.insert(0, time_ms_str)
            self.laser_frame_time_entry.bind('<FocusOut>', self._on_time_changed)
            self.laser_frame_time_entry.bind('<Return>', self._on_time_changed)
            
        except (ValueError, InvalidOperation):
            pass  # Invalid input, ignore

    #####################################################################################
    #####################################################################################
    #####################################################################################

    def get_laser_device_labels(self) -> list[str]:
        """Enumerate Digilent devices and update the internal mapping."""
        try:
            devices = list_digital_devices() or []
        except Exception as e:
            logger.error(f"Error listing Digilent devices: {e}")
            devices = []

        mapping: dict[str, int] = {}
        labels: list[str] = []

        for device in devices:
            index = device.get("index", -1)
            name = device.get("name", "Unknown")
            serial = device.get("serial", "Unknown")
            label = f"{index}: {name} ({serial})"
            mapping[label] = index
            labels.append(label)

        self.laser_device_mapping = mapping
        return labels

    @run_in_background
    def refresh_laser_devices_async(self):
        """Refresh the Digilent device list in background"""
        device_values = self.get_laser_device_labels()

        def update_gui():
            values = device_values if device_values else ["No devices found"]
            self.laser_device_combobox.configure(values=values)
            self.laser_device_combobox.set(values[0])
            self.log_message(f"Laser devices refreshed: {len(device_values)} found")

        self.schedule_gui_update(update_gui)

    def toggle_laser_connection_async(self):
        """Toggle the laser connection state."""
        if self._laser_connecting:
            self.log_message("Laser connection already in progress...")
            return

        if not self.laser_connected:
            self.connect_laser_async()
        else:
            self.disconnect_laser_async()

    @run_in_background
    def connect_laser_async(self):
        """Connect to the laser hardware in a background thread."""
        if self._laser_connecting:
            return

        self._laser_connecting = True

        def update_connecting():
            self.laser_connect_button.configure(text="Connecting...", state="disabled")
            self.laser_status_label.configure(text="● Connecting...", text_color="orange")

        self.schedule_gui_update(update_connecting)

        try:
            selection = self.laser_device_combobox.get()

            if selection not in self.laser_device_mapping:
                # Refresh mapping if selection isn't recognized
                self.get_laser_device_labels()

            device_index = self.laser_device_mapping.get(selection)
            if device_index is None or not selection or selection == "No devices found":
                self.log_message("Unable to resolve device selection. Refresh the device list.")
                self.laser_connect_button.configure(text="Connect Laser", state="normal")
                self.laser_status_label.configure(text="✗ Unable to resolve device", text_color="red")
                self._laser_connecting = False
                self.laser_connected = False
                if not USE_SIMULATION:
                    return

            channel_text = self.laser_channel_entry.get().strip()
            channel = int(channel_text) if channel_text else 8

        except Exception as e:
            def update_error():
                self._laser_connecting = False
                self.laser_status_label.configure(text=f"✗ {e}", text_color="red")
                self.laser_connect_button.configure(text="Connect Laser", state="normal")
                self.log_message(f"✗ Laser connection error: {e}")

            self.schedule_gui_update(update_error)
            return

        try:
            if USE_SIMULATION:
                driver_simulation = SimulatedLaserDriver(pulses_queue=Queue(), laser_info=LaserInfo())
                controller = LaserController(driver=driver_simulation)
            else:
                controller = create_laser_controller_with_hardware(device_index=device_index, digital_channel=channel)

            if not controller.initialize():
                self.log_message("Laser initialization failed.")
                self.laser_connect_button.configure(text="Connect Laser", state="normal")
                self.laser_status_label.configure(text="✗ Initialization failed", text_color="red")
                self._laser_connecting = False
                self.laser_connected = False
                return

            self.laser_controller = controller
            self.laser_connected = True
            self.laser_in_continuous = False

            def update_success():
                self._laser_connecting = False
                self.laser_status_label.configure(
                    text=f"● Laser Connected (ch {channel})",
                    text_color="green"
                )
                self.laser_connect_button.configure(text="Disconnect Laser", state="normal")
                self.enable_laser_controls(True)
                self.log_message(f"✓ Connected to Digilent device #{device_index} on channel {channel}")

            self.schedule_gui_update(update_success)

        except Exception as e:
            if 'controller' in locals():
                try:
                    controller.shutdown()
                except Exception:
                    pass

            if self.laser_controller:
                try:
                    self.laser_controller.shutdown()
                except Exception:
                    pass
                self.laser_controller = None

            def update_failure():
                self._laser_connecting = False
                self.laser_connected = False
                self.laser_status_label.configure(text=f"Connection failed: {e}", text_color="red")
                self.laser_connect_button.configure(text="Connect Laser", state="normal")
                self.enable_laser_controls(False)
                self.log_message(f"✗ Laser connection failed: {e}")

            self.schedule_gui_update(update_failure)

    @run_in_background
    def disconnect_laser_async(self):
        """Disconnect from the laser hardware."""
        try:
            if self.laser_controller:
                self.laser_controller.shutdown()
        except Exception as e:
            self.schedule_gui_update(lambda: self.log_message(f"Error during laser disconnect: {e}"))
        finally:
            def update_disconnected():
                self.laser_connected = False
                self._laser_connecting = False
                self.laser_in_continuous = False
                self.laser_controller = None
                self.laser_status_label.configure(text="● Disconnected", text_color="red")
                self.laser_connect_button.configure(text="Connect Laser", state="normal")
                self.enable_laser_controls(False)
                self.log_message("Laser hardware disconnected")

            self.schedule_gui_update(update_disconnected)

    def enable_laser_controls(self, enabled: bool):
        """Enable or disable laser control widgets."""
        state = "normal" if enabled else "disabled"
        entry_state = "normal" if enabled else "disabled"
        device_state = "disabled" if enabled else "normal"

        # Device selection widgets
        self.laser_device_combobox.configure(state=device_state)
        self.laser_refresh_button.configure(state=device_state)
        self.laser_channel_entry.configure(state=device_state)

        # Parameter entries and switches
        self.laser_duty_entry.configure(state=entry_state)
        self.laser_freq_entry.configure(state=entry_state)

        # Pulse train/continuous parameter entries
        self.laser_frame_count_entry.configure(state=entry_state)
        self.laser_frame_time_entry.configure(state=entry_state)
        self.laser_cont_freq_entry.configure(state=entry_state)
        self.laser_frame_repeat_switch.configure(state=entry_state)
        self.laser_frame_repeat_nb_entry.configure(state=entry_state if self.laser_frame_repeat_switch.get() else "disabled")
        self.laser_frame_repeat_interval_entry.configure(state=entry_state if self.laser_frame_repeat_switch.get() else "disabled")

        # Action buttons
        self.laser_set_params_button.configure(state=state)
        self.laser_single_button.configure(state=state)
        self.laser_send_frame_button.configure(state=state)
        self.laser_start_cont_button.configure(state=state)
        self.laser_status_button.configure(state=state)

        stop_state = "normal" if (enabled and self.laser_in_continuous) else "disabled"
        self.laser_stop_cont_button.configure(state=stop_state)

    def _laser_available(self) -> bool:
        if not self.laser_connected or not self.laser_controller:
            self.schedule_gui_update(lambda: self.log_message("Laser hardware not connected"))
            return False
        return True

    @run_in_background
    def set_laser_pulse_parameters_async(self):
        """Update laser pulse parameters."""
        if not self._laser_available():
            return

        try:
            duty_text = self.laser_duty_entry.get().strip()
            freq_text = self.laser_freq_entry.get().strip()

            if not duty_text:
                self.schedule_gui_update(lambda: self.log_message("✗ Enter a duty cycle (0-100%)"))
                return

            duty_value = float(duty_text)
            if duty_value > 100:
                self.schedule_gui_update(lambda: self.log_message("✗ Duty cycle cannot exceed 100%"))
                return
            duty_cycle = duty_value / 100.0 if duty_value > 0 else duty_value
            if duty_cycle <= 0 or duty_cycle >= 1:
                self.schedule_gui_update(lambda: self.log_message("✗ Duty cycle must be between 0 and 100% (exclusive)"))
                return

            frequency = None
            if freq_text:
                frequency = float(freq_text)
                if frequency <= 0:
                    self.schedule_gui_update(lambda: self.log_message("✗ Frequency must be positive"))
                    return

            success = self.laser_controller.set_pulse_parameters(
                duty_cycle=duty_cycle,
                frequency=frequency
            )

            def update_gui():
                if success:
                    freq_display = f"{frequency:.1f} Hz" if frequency is not None else "current"
                    # Update the continuous frequency entry as well and the time entry
                    self._on_freq_changed()
                    self.log_message(
                        f"✓ Updated laser pulse parameters → duty={duty_cycle*100:.1f}% freq={freq_display}"
                    )
                else:
                    self.log_message("✗ Failed to update laser pulse parameters")

            self.schedule_gui_update(update_gui)

        except NotImplementedError as e:
            self.schedule_gui_update(lambda: self.log_message(f"✗ Pulse parameter update unsupported: {e}"))
        except Exception as e:
            self.schedule_gui_update(lambda: self.log_message(f"✗ Invalid pulse parameters: {e}"))

    @run_in_background
    def trigger_laser_single_async(self):
        """Trigger a single laser pulse."""
        if not self._laser_available():
            return

        try:
            success = self.laser_controller.trigger_once()
            def update_gui():
                if success:
                    self.log_message("✓ Single laser pulse triggered")
                else:
                    self.log_message("✗ Failed to trigger laser pulse")
            self.schedule_gui_update(update_gui)
        except Exception as e:
            self.schedule_gui_update(lambda: self.log_message(f"✗ Error triggering laser pulse: {e}"))

    @run_in_background
    def send_laser_frame_async(self):
        """Send a pulse train using the laser controller."""
        if not self._laser_available():
            return

        try:
            count_text = self.laser_frame_count_entry.get().strip()
            time_text = self.laser_frame_time_entry.get().strip()

            if not count_text:
                self.schedule_gui_update(lambda: self.log_message("✗ Enter number of pulses"))
                return

            count = int(count_text)
            if count <= 0:
                self.schedule_gui_update(lambda: self.log_message("✗ Pulse count must be positive"))
                return

            # Get time in milliseconds, if empty use frequency entry
            if not time_text:
                freq_text = self.laser_freq_entry.get().strip()
                if not freq_text:
                    self.schedule_gui_update(lambda: self.log_message("✗ Enter a time period (ms) or frequency (Hz)"))
                    return
                try:
                    frequency_dec = Decimal(freq_text)
                except InvalidOperation:
                    self.schedule_gui_update(lambda: self.log_message("✗ Invalid frequency value"))
                    return
                frequency = float(frequency_dec)
            else:
                # Convert time (ms) to frequency (Hz): f = 1000 / T
                try:
                    time_ms_dec = Decimal(time_text)
                except InvalidOperation:
                    self.schedule_gui_update(lambda: self.log_message("✗ Invalid time value"))
                    return
                if time_ms_dec <= 0:
                    self.schedule_gui_update(lambda: self.log_message("✗ Time period must be positive"))
                    return
                with localcontext() as ctx:
                    ctx.prec = 50
                    frequency_dec = (Decimal(1000) / time_ms_dec)
                    frequency = float(frequency_dec)

            if frequency <= 0:
                self.schedule_gui_update(lambda: self.log_message("✗ Frequency must be positive"))
                return

            repeat_params = self.get_repeat_parameters()
            if repeat_params:
                nb_repeats, interval = repeat_params
                self.log_message(f"⟳ Repeating pulse train {nb_repeats} times every {interval:.1f} seconds")
                for i in range(nb_repeats):
                    if i > 0:
                        time.sleep(interval)
                    self.log_message(f"→ Sending pulse train {i+1}/{nb_repeats}...")
                    success = self.laser_controller.send_frame(count, frequency)
                    if not success:
                        self.schedule_gui_update(lambda: self.log_message("✗ Failed to send pulse train during repeat"))
                        return
                self.schedule_gui_update(lambda: self.log_message(f"✓ Completed {nb_repeats} pulse train repeats"))
                return
            
            success = self.laser_controller.send_frame(count, frequency)

            def update_gui():
                if success:
                    self.log_message(f"✓ Sent pulse train: {count} pulses at {frequency:.1f} Hz")
                else:
                    self.log_message("✗ Failed to send pulse train")

            self.schedule_gui_update(update_gui)

        except Exception as e:
            self.schedule_gui_update(lambda: self.log_message(f"✗ Pulse train error: {e}"))

    def on_repeat_switch_toggle(self):
        """Enable or disable repeat parameters based on switch state."""
        if self.laser_frame_repeat_switch.get():
            self.laser_frame_repeat_nb_entry.configure(state="normal")
            self.laser_frame_repeat_interval_entry.configure(state="normal")
        else:
            self.laser_frame_repeat_nb_entry.configure(state="disabled")
            self.laser_frame_repeat_interval_entry.configure(state="disabled")
            
    
    def get_repeat_parameters(self) -> Optional[Tuple[int, float]]:
        """Retrieve repeat parameters if enabled."""
        if not self.laser_frame_repeat_switch.get():
            return None

        try:
            nb_text = self.laser_frame_repeat_nb_entry.get().strip()
            interval_text = self.laser_frame_repeat_interval_entry.get().strip()

            if not nb_text or not interval_text:
                self.schedule_gui_update(lambda: self.log_message("✗ Enter repeat count and interval"))
                return None

            nb = int(nb_text)
            interval = float(interval_text)

            if nb <= 0:
                self.schedule_gui_update(lambda: self.log_message("✗ Repeat count must be positive"))
                return None
            if interval <= 0:
                self.schedule_gui_update(lambda: self.log_message("✗ Repeat interval must be positive"))
                return None

            return (nb, interval)

        except Exception as e:
            self.schedule_gui_update(lambda: self.log_message(f"✗ Invalid repeat parameters: {e}"))
            return None

    @run_in_background
    def start_laser_continuous_async(self):
        """Start continuous laser mode."""
        if not self._laser_available():
            return

        try:
            freq_text = self.laser_cont_freq_entry.get().strip()
            if not freq_text:
                freq_text = self.laser_freq_entry.get().strip()
            if not freq_text:
                self.schedule_gui_update(lambda: self.log_message("✗ Enter a frequency for continuous mode"))
                return

            frequency = float(freq_text)
            if frequency <= 0:
                self.schedule_gui_update(lambda: self.log_message("✗ Frequency must be positive"))
                return

            success = self.laser_controller.start_continuous(frequency)

            def update_gui():
                if success:
                    self.laser_in_continuous = True
                    self.laser_start_cont_button.configure(state="disabled")
                    self.laser_stop_cont_button.configure(state="normal")
                    self.log_message(f"✓ Continuous mode started at {frequency:.1f} Hz")
                else:
                    self.log_message("✗ Failed to start continuous mode")

            self.schedule_gui_update(update_gui)

        except Exception as e:
            self.schedule_gui_update(lambda: self.log_message(f"✗ Continuous mode error: {e}"))

    @run_in_background
    def stop_laser_continuous_async(self):
        """Stop continuous laser mode."""
        if not self._laser_available():
            return

        try:
            success = self.laser_controller.stop_continuous()

            def update_gui():
                if success:
                    self.laser_in_continuous = False
                    self.laser_start_cont_button.configure(state="normal")
                    self.laser_stop_cont_button.configure(state="disabled")
                    self.log_message("✓ Continuous mode stopped")
                else:
                    self.log_message("✗ Failed to stop continuous mode")

            self.schedule_gui_update(update_gui)

        except Exception as e:
            self.schedule_gui_update(lambda: self.log_message(f"✗ Error stopping continuous mode: {e}"))

    @run_in_background
    def get_laser_status_async(self):
        """Fetch laser controller status."""
        if not self._laser_available():
            return

        try:
            status = self.laser_controller.get_status()

            def update_gui():
                controller_info = status.get("controller", {})
                driver_info = status.get("driver", {})
                interface_info = driver_info.get("interface_status", {})
                
                # Controller status
                self.log_message("📊 LASER STATUS REPORT")
                self.log_message("🔧 Laser Controller:")
                self.log_message(f"   ├─ Type: {driver_info.get('hardware_type', 'unknown')}")
                self.log_message(f"   ├─ Device Index: {driver_info.get('device_index', 'N/A')}")
                self.log_message(f"   ├─ Channel: {driver_info.get('digital_channel', 'N/A')}")
                self.log_message(f"   ├─ Duty Cycle: {driver_info.get('duty_cycle_percent', 0):.1f}%")
                self.log_message(f"   ├─ Frequency: {driver_info.get('frequency_hz', 0):.1f} Hz")
                self.log_message(f"   ├─ Max Frequency: {driver_info.get('max_frequency_hz', 0)/1e6:.1f} MHz")
                if interface_info:
                    self.log_message(f"   ├─ Connected: {'✓' if interface_info.get('connected') else '✗'}")
                    self.log_message(f"   ├─ Pulse Count: {interface_info.get('pulse_count', 0)}")
                    self.log_message(f"   └─ Error Count: {interface_info.get('error_count', 0)}")
                
                # Controller section
                self.log_message("🎛️ Controller:")
                self.log_message(f"   ├─ Initialized: {'✓' if controller_info.get('initialized') else '✗'}")
                self.log_message(f"   ├─ Active: {'✓' if controller_info.get('active') else '✗'}")
                self.log_message(f"   └─ Continuous Mode: {'ON' if controller_info.get('continuous') else 'OFF'}")
                
            self.schedule_gui_update(update_gui)

        except Exception as e:
            self.schedule_gui_update(lambda: self.log_message(f"✗ Error retrieving laser status: {e}"))



    def log_message(self, message: str):
        """Add a message to the log (thread-safe)"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Insert at the end and scroll to bottom
        self.log_text.insert("end", log_entry)
        self.log_text.see("end")

    def start_status_monitoring(self):
        """Start lightweight status monitoring"""
        def monitor():
            if self.is_connected and self.pol_controller:
                # Lightweight status check every few seconds
                # Only update if there's actually a change
                pass
                
        # Schedule periodic status check (every 5 seconds)
        self.after(5000, self.start_status_monitoring)

    def on_closing(self):
        """Handle window closing"""
        try:
            # Stop thread pool
            self.thread_pool.shutdown(wait=False)
            
            # Disconnect hardware
            if self.is_connected:
                self.disconnect_hardware_async()
                self.disconnect_laser_async()
                
            self.destroy()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            self.destroy()


if __name__ == "__main__":
    # Set CustomTkinter appearance
    ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
    # Load custom theme with primary orange #fe9409
    THEME = "entangled_orange"
    THEME = "green"
    THEME = "dark_blue"
    THEME = "orange"
    try:
        theme_path = os.path.join(os.path.dirname(__file__), 'themes', THEME + '.json')
        if os.path.exists(theme_path):
            ctk.set_default_color_theme(theme_path)
        else:
            ctk.set_default_color_theme("blue")
    except Exception:
        ctk.set_default_color_theme("blue")

    # Optional: slightly increase widget scaling for comfort
    try:
        ctk.set_widget_scaling(1.05)
        ctk.set_window_scaling(1.0)
    except Exception:
        pass
    
    # Create and run the application
    app = PolarizationLaserControllerGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()