"""
GUI for Bob's TimeTagger for testing purposes.
Allows connecting to a TimeTagger (real or simulated), configuring channels,
and viewing live measurement counts.
"""
import customtkinter as ctk
import logging
import functools
import time
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List, Any, Union

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.bob.timetagger.simple_timetagger_controller import SimpleTimeTaggerController
from src.bob.timetagger.simple_timetagger_base_hardware_simulator import SimpleTimeTaggerHardware, SimpleTimeTaggerSimulator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USE_SIMULATION = True  # Set to False to use real hardware by default
POLARIZATIONS = [
    ("H", "Horizontal (0 deg)"),
    ("V", "Vertical (90 deg)"),
    ("D", "Diagonal (45 deg)"),
    ("A", "Anti-diagonal (135 deg)"),
]

def run_in_background(func):
    """Decorator to run hardware operations in a background thread."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'thread_pool') or not isinstance(self.thread_pool, ThreadPoolExecutor):
            # simple fallback with a separate thread if no thread pool is defined
            from threading import Thread
            thread = Thread(target=func, args=(self, *args), kwargs=kwargs, daemon=True)
            thread.start()
            return 

        future = self.thread_pool.submit(func, self, *args, **kwargs)
        
        def on_done(f):
            try:
                f.result()
            except Exception as e:
                logger.error(f"Error in background task '{func.__name__}': {e}", exc_info=True)
        
        future.add_done_callback(on_done)
        return future
    return wrapper


class TimeTaggerControllerGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("TimeTagger Hardware Interface for Testing Bob")
        self.geometry("1000x700")

        # Thread pool for hardware operations
        self.thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="TimeTaggerOp")
        
        # GUI update queue for thread-safe communication
        self.gui_update_queue = Queue()

        # TimeTagger controller
        self.timetagger_controller: Optional[SimpleTimeTaggerController] = None
        self.driver: Optional[Union[SimpleTimeTaggerHardware, SimpleTimeTaggerSimulator]] = None
        self.connected = False

        # Measurement state
        self.is_measuring = False
        self.measurement_task = None
        self.time_check_update_gui_process = 50  # ms
        self.current_max_rows = 0

        # Variables for measurement configuration
        self.status_var = ctk.StringVar(value="Status: Disconnected")
        self.available_channels_var = ctk.StringVar(value="1,2,3,4")
        self.polarization_vars: Dict[str, ctk.StringVar] = {
            pol: ctk.StringVar(value=str(idx + 1)) for idx, (pol, _) in enumerate(POLARIZATIONS)
        }
        self.bin_duration_ms_var = ctk.StringVar(value="1000")
        self.num_rows_var = ctk.StringVar(value="10")
        self.measurement_mode_var = ctk.StringVar(value="continuous")
        self.repeat_count_var = ctk.StringVar(value="10")
        self.use_simulator_var = ctk.BooleanVar(value=True)
        
        # Setup GUI components
        self.setup_gui()
        self.process_gui_updates()

    def process_gui_updates(self):
        """Process items from the GUI update queue."""
        try:
            while not self.gui_update_queue.empty():
                func = self.gui_update_queue.get_nowait()
                func()
        except Empty:
            pass
        finally:
            self.after(self.time_check_update_gui_process, self.process_gui_updates) # Check every 50 ms

    def schedule_gui_update(self, func):
        """Schedule a function to be called in the main GUI thread."""
        self.gui_update_queue.put(func)


    def setup_gui(self):
        """Set up the main GUI layout."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Top frame for connection and configuration
        config_frame = ctk.CTkFrame(self)
        config_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.setup_config_frame(config_frame)

        # Bottom frame for results
        results_frame = ctk.CTkFrame(self)
        results_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.setup_results_frame(results_frame)

    def setup_config_frame(self, parent_frame: ctk.CTkFrame):
        """Setup the connection and measurement configuration widgets."""
        parent_frame.grid_columnconfigure(1, weight=1)

        # --- Connection ---
        connection_frame = ctk.CTkFrame(parent_frame)
        connection_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew", columnspan=4)
        connection_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(connection_frame, text="TimeTagger Connection").grid(row=0, column=0, columnspan=4, pady=(5,10))

        self.connect_button = ctk.CTkButton(connection_frame, text="Connect", command=self.toggle_connection_async)
        self.connect_button.grid(row=1, column=0, padx=10, pady=5)

        self.connection_status_label = ctk.CTkLabel(connection_frame, textvariable=self.status_var, text_color="red")
        self.connection_status_label.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        self.use_sim_switch = ctk.CTkSwitch(connection_frame, text="Use Simulator", onvalue=True, offvalue=False)
        self.use_sim_switch.grid(row=1, column=2, padx=10, pady=5)
        self.use_sim_switch.select() if USE_SIMULATION else self.use_sim_switch.deselect()

        # --- Channel Configuration ---
        channel_frame = ctk.CTkFrame(parent_frame)
        channel_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew", columnspan=4)
        ctk.CTkLabel(channel_frame, text="Channel Polarization Mapping").pack(pady=(5,10))

        self.channel_map_entries = {}
        self.channel_map_labels = {1: "H (0°)", 2: "V (90°)", 3: "D (45°)", 4: "A (135°)"}
        
        channel_grid_frame = ctk.CTkFrame(channel_frame)
        channel_grid_frame.pack(pady=5, padx=10)

        for i in range(4):
            channel = i + 1
            ctk.CTkLabel(channel_grid_frame, text=f"Channel {channel}:").grid(row=i, column=0, padx=5, pady=2, sticky="w")
            entry = ctk.CTkEntry(channel_grid_frame, placeholder_text=f"e.g., {self.channel_map_labels.get(channel, '')}")
            entry.grid(row=i, column=1, padx=5, pady=2)
            entry.insert(0, self.channel_map_labels.get(channel, ''))
            self.channel_map_entries[channel] = entry

        # ctk.CTkLabel(channel_frame, text="Channels (comma separated)").grid(
        #     row=0, column=0, padx=6, pady=6, sticky="w"
        # )
        # channel_entry = ctk.CTkEntry(channel_frame, textvariable=self.available_channels_var)
        # channel_entry.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        # self.channel_entry = channel_entry

        # ctk.CTkLabel(channel_frame, text="Polarization mapping").grid(
        #     row=1, column=0, columnspan=2, padx=6, pady=(12, 6), sticky="w"
        # )

        # for idx, (code, description) in enumerate(POLARIZATIONS):
        #     ctk.CTkLabel(channel_frame, text=f"{code}:").grid(row=2 + idx, column=0, padx=6, pady=4, sticky="w")
        #     option = ctk.CTkOptionMenu(
        #         channel_frame,
        #         variable=self.polarization_vars[code],
        #         values=self.get_available_channels(),
        #     )
        #     option.grid(row=2 + idx, column=1, padx=6, pady=4, sticky="ew")
        #     setattr(self, f"pol_option_{code}", option)

        # --- Measurement Configuration ---
        measure_frame = ctk.CTkFrame(parent_frame)
        measure_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew", columnspan=4)
        measure_frame.grid_columnconfigure((0,1,2,3,4), weight=1)

        ctk.CTkLabel(measure_frame, text="Measurement Control").grid(row=0, column=0, columnspan=5, pady=(5,10))

        ctk.CTkLabel(measure_frame, text="Time per Bin (ms):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.time_bin_entry = ctk.CTkEntry(measure_frame, width=80, textvariable=self.bin_duration_ms_var)
        self.time_bin_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        # self.time_bin_entry.insert(0, "1000")

        ctk.CTkLabel(measure_frame, text="Num Bins to Show:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.num_bins_entry = ctk.CTkEntry(measure_frame, width=80, textvariable=self.num_rows_var)
        self.num_bins_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        # self.num_bins_entry.insert(0, "10")

        self.measure_button = ctk.CTkButton(measure_frame, text="Start Measurement", command=self.toggle_measurement, state="disabled")
        self.measure_button.grid(row=1, column=2, rowspan=2, padx=10, pady=5)

        self.continuous_switch = ctk.CTkSwitch(measure_frame, text="Continuous", onvalue=True, offvalue=False)
        self.continuous_switch.grid(row=1, column=3, padx=10, pady=5)
        self.continuous_switch.select()

        self.fixed_time_entry = ctk.CTkEntry(measure_frame, placeholder_text="Total time (s)", width=100, state="disabled")
        self.fixed_time_entry.grid(row=2, column=3, padx=10, pady=5)
        self.continuous_switch.configure(command=lambda: self.fixed_time_entry.configure(state="normal" if not self.continuous_switch.get() else "disabled"))

        # mode_frame = ctk.CTkFrame(settings_frame)
        # mode_frame.grid(row=2, column=0, columnspan=2, padx=6, pady=(6, 6), sticky="ew")
        # mode_frame.grid_columnconfigure(1, weight=1)
        # ctk.CTkLabel(mode_frame, text="Measurement mode").grid(row=0, column=0, columnspan=2, padx=4, pady=4)
        # ctk.CTkRadioButton(
        #     mode_frame,
        #     text="Continuous",
        #     variable=self.measurement_mode_var,
        #     value="continuous",
        # ).grid(row=1, column=0, padx=4, pady=4, sticky="w")
        # ctk.CTkRadioButton(
        #     mode_frame,
        #     text="Finite",
        #     variable=self.measurement_mode_var,
        #     value="finite",
        # ).grid(row=1, column=1, padx=4, pady=4, sticky="w")

        # ctk.CTkLabel(mode_frame, text="Finite repeats").grid(row=2, column=0, padx=4, pady=(6, 4), sticky="w")
        # self.repeat_entry = ctk.CTkEntry(mode_frame, textvariable=self.repeat_count_var)
        # self.repeat_entry.grid(row=2, column=1, padx=4, pady=(6, 4), sticky="ew")
        

    def setup_results_frame(self, parent_frame: ctk.CTkFrame):
        """Setup the frame for displaying measurement results."""
        parent_frame.grid_columnconfigure(0, weight=1)
        parent_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(parent_frame, text="Measurements", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, pady=5)

        self.results_scrollable_frame = ctk.CTkScrollableFrame(parent_frame, label_text="Counts per Bin")
        self.results_scrollable_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.results_scrollable_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

    def toggle_connection_async(self):
        """Handle the connect/disconnect button click."""
        if self.timetagger_controller and self.timetagger_controller.is_initialized():
            self.disconnect_from_timetagger()
        else:
            self.connect_to_timetagger()

    @run_in_background
    def connect_to_timetagger(self):
        """Connect to the TimeTagger in a background thread."""
        self.schedule_gui_update(lambda: self.connect_button.configure(state="disabled", text="Connecting..."))
        
        use_sim = self.use_sim_switch.get()
        
        channels_int = [int(ch) for ch in self.get_available_channels() if ch.isdigit()]
        
        try:
            if use_sim:
                logger.info("Using TimeTagger simulator.")
                driver = SimpleTimeTaggerSimulator(
                    detector_channels=channels_int,
                    dark_count_rate=50.0,
                    signal_count_rate=200.0,
                    signal_probability=0.1
                )
            else:
                logger.info("Using TimeTagger hardware.")
                driver = SimpleTimeTaggerHardware(detector_channels=channels_int)

            self.timetagger_controller = SimpleTimeTaggerController(driver)
            
            if self.timetagger_controller.initialize():
                logger.info("TimeTagger connected successfully.")
                self.schedule_gui_update(lambda: self.status_var.set("Status: Connected"))
                self.schedule_gui_update(lambda: self.connection_status_label.configure(text_color="green"))
                self.schedule_gui_update(lambda: self.connect_button.configure(text="Disconnect"))
                self.schedule_gui_update(lambda: self.enable_controls(True))
                # self.schedule_gui_update(self.refresh_option_menus) # TODO: UNCOMENT AFTER ADDING THE CHANNEL ENTRY
            else:
                logger.error("Failed to initialize TimeTagger.")
                self.schedule_gui_update(lambda: self.status_var.set("Status: Connection Failed"))
                self.schedule_gui_update(lambda: self.connection_status_label.configure(text_color="red"))
                self.timetagger_controller = None

        except Exception as e:
            logger.error(f"Error connecting to TimeTagger: {e}", exc_info=True)
            self.schedule_gui_update(lambda: self.status_var.set("Status: Error"))
            self.schedule_gui_update(lambda: self.connection_status_label.configure(text_color="red"))
            self.timetagger_controller = None
        finally:
            self.schedule_gui_update(lambda: self.connect_button.configure(state="normal"))

    @run_in_background
    def disconnect_from_timetagger(self):
        """Disconnect from the TimeTagger in a background thread."""
        self.schedule_gui_update(lambda: self.connect_button.configure(state="disabled", text="Disconnecting..."))
        if self.is_measuring:
            self.stop_measurement()

        if self.timetagger_controller:
            try:
                self.timetagger_controller.shutdown()
                logger.info("TimeTagger disconnected successfully.")
                self.schedule_gui_update(lambda: self.status_var.set("Status: Disconnected"))
                self.schedule_gui_update(lambda: self.connection_status_label.configure(text_color="red"))
                self.schedule_gui_update(lambda: self.connect_button.configure(text="Connect"))
                self.schedule_gui_update(lambda: self.enable_controls(False))
            except Exception as e:
                logger.error(f"Error during TimeTagger shutdown: {e}", exc_info=True)
            finally:
                self.timetagger_controller = None
        
        self.schedule_gui_update(lambda: self.connect_button.configure(state="normal"))

    def enable_controls(self, enabled: bool):
        """Enable or disable measurement controls."""
        self.measure_button.configure(state="normal" if enabled else "disabled")
        self.time_bin_entry.configure(state="normal" if enabled else "disabled")
        self.num_bins_entry.configure(state="normal" if enabled else "disabled")
        self.continuous_switch.configure(state="normal" if enabled else "disabled")
        
        is_fixed_time = enabled and not self.continuous_switch.get()
        self.fixed_time_entry.configure(state="normal" if is_fixed_time else "disabled")

        for entry in self.channel_map_entries.values():
            entry.configure(state="normal" if enabled else "disabled")

    def get_available_channels(self) -> List[str]:
        """Get the list of available channels from the entry."""
        channels_str = self.available_channels_var.get().strip()
        if not channels_str:
            return ["1", "2", "3", "4"]
        
        channels = [ch.strip() for ch in channels_str.split(",") if ch.strip().isdigit()]
        return channels

    def refresh_option_menus(self) -> None:
        """Update polarization drop downs to reflect channel list."""
        values = self.get_available_channels()
        for code, _ in POLARIZATIONS:
            option: ctk.CTkOptionMenu = getattr(self, f"pol_option_{code}")
            option.configure(values=values)
            if self.polarization_vars[code].get() not in values:
                self.polarization_vars[code].set(values[0])

    def toggle_measurement(self):
        """Start or stop the measurement process."""
        if self.is_measuring:
            self.stop_measurement()
        else:
            self.start_measurement()

    def start_measurement(self):
        """Start the measurement loop."""
        if not self.timetagger_controller or not self.timetagger_controller.is_initialized():
            logger.error("Cannot start measurement: TimeTagger not connected.")
            return

        # TODO: DO SEPARATE VALIDATION OF PARAMETERS HERE
        # try:
        #     bin_duration_ms = float(self.bin_duration_ms_var.get())
        # except ValueError:
        #     self.status_var.set("Status: invalid bin duration")
        #     return
        # if bin_duration_ms <= 0:
        #     self.status_var.set("Status: bin duration must be positive")
        #     return
        # duration_seconds = bin_duration_ms / 1000.0

        try:
            self.time_per_bin = float(self.bin_duration_ms_var.get()) / 1000.0
            self.num_bins_to_show = int(self.num_rows_var.get())
            self.is_continuous = self.continuous_switch.get()
            
            self.total_measurement_time = None
            if not self.is_continuous:
                self.total_measurement_time = float(self.fixed_time_entry.get())

        except (ValueError, TypeError):
            logger.error("Invalid measurement parameters. Please enter valid numbers.")
            self.log_message("Error: Invalid measurement parameters.", "red")
            return


        # mode = self.measurement_mode_var.get()
        # repeat_target = None
        # if mode == "finite":
        #     try:
        #         repeat_target = int(self.repeat_count_var.get())
        #     except ValueError:
        #         self.status_var.set("Status: invalid repeat count")
        #         return
        #     if repeat_target <= 0:
        #         self.status_var.set("Status: repeat count must be positive")
        #         return



        if self.time_per_bin <= 0:
            logger.error("Time per bin must be positive.")
            return

        # Clear previous results
        for widget in self.results_scrollable_frame.winfo_children():
            widget.destroy()
        self.bin_row_index = 0
        
        # Create headers
        headers = self.get_polarization_labels()
        for i, header in enumerate(["Bin"] + list(headers.values())):
             ctk.CTkLabel(self.results_scrollable_frame, text=header, font=ctk.CTkFont(weight="bold")).grid(row=0, column=i, padx=5, pady=5)

        self.is_measuring = True
        self.measure_button.configure(text="Stop Measurement")
        self.measurement_start_time = time.time()
        self.measurement_task = self.thread_pool.submit(self._measurement_loop)

    def stop_measurement(self):
        """Stop the measurement loop."""
        self.is_measuring = False
        if self.measurement_task:
            # The loop will check self.is_measuring and exit, no need to force future
            pass
        self.measure_button.configure(text="Start Measurement")

    def _measurement_loop(self):
        """Background task for continuous measurement."""
        logger.info("Measurement loop started.")
        
        iteration = 0
        while self.is_measuring:
            iteration += 1
            start_time = time.time()

            # Check for fixed time limit
            if not self.is_continuous and self.total_measurement_time is not None:
                if (start_time - self.measurement_start_time) >= self.total_measurement_time:
                    self.schedule_gui_update(self.stop_measurement)
                    break
            
            try:
                self.timetagger_controller.set_measurement_duration(self.time_per_bin)
                counts = self.timetagger_controller.measure_counts()
                
                if counts:
                    self.schedule_gui_update(lambda c=counts: self.add_result_row(c))

            except Exception as e:
                logger.error(f"Error during measurement: {e}", exc_info=True)
                self.schedule_gui_update(lambda: self.log_message(f"Error: {e}", "red"))
                break
            
            # if repeat_target is not None and iteration >= repeat_target:
            #     break

            # if not self.is_measuring:
            #     break
            
            # Sleep to maintain the measurement interval
            elapsed = time.time() - start_time
            sleep_time = max(0, self.time_per_bin - elapsed)
            # time.sleep(sleep_time)

        logger.info("Measurement loop stopped.")

    def add_result_row(self, counts: Dict[int, int]):
        """Add a new row of results to the GUI."""
        self.bin_row_index += 1
        
        # Manage displayed bins
        if self.bin_row_index > self.num_bins_to_show:
            # Find and remove the oldest row (row 1, since 0 is header)
            for widget in self.results_scrollable_frame.grid_slaves():
                if int(widget.grid_info()["row"]) == 1:
                    widget.destroy()
            # Shift all other rows up
            for widget in self.results_scrollable_frame.grid_slaves():
                 if widget.grid_info()["row"] > 1:
                    widget.grid(row=widget.grid_info()["row"] - 1)
            self.bin_row_index -=1


        row = self.bin_row_index
        ctk.CTkLabel(self.results_scrollable_frame, text=str(row)).grid(row=row, column=0, padx=5)

        pol_labels = self.get_polarization_labels()
        for i, channel in enumerate(pol_labels.keys()):
            count = counts.get(channel, 0)
            ctk.CTkLabel(self.results_scrollable_frame, text=str(count)).grid(row=row, column=i + 1, padx=5)

    def get_polarization_labels(self) -> Dict[int, str]:
        """Get the current polarization labels from the GUI."""
        return {ch: entry.get() for ch, entry in self.channel_map_entries.items()}

    def log_message(self, message: str, color: str = "gray"):
        """Log a message to the GUI."""
        # This could be a status bar at the bottom, for now just prints
        print(f"GUI_LOG: {message}")

    def on_closing(self):
        """Handle window closing event."""
        if self.is_measuring:
            self.stop_measurement()
        if self.timetagger_controller and self.timetagger_controller.is_initialized():
            self.timetagger_controller.shutdown()
        self.thread_pool.shutdown(wait=False)
        self.destroy()

if __name__ == "__main__":
    app = TimeTaggerControllerGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()