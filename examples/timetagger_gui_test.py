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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

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

USE_SIMULATION = False  # Set to False to use real hardware by default
POLARIZATIONS = [
    ("H", "H (0º)"),
    ("V", "V (90º)"),
    ("D", "D (45º)"),
    ("A", "A (135º)"),
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


class HistogramWindow(ctk.CTkToplevel):
    """Window displaying histogram of measurement counts."""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        self.title("Count Histogram & Relations")
        self.geometry("900x700")
        
        # Data storage for histogram
        self.histogram_data: List[int] = []
        
        # Histogram display mode
        # Options: "all_nonzero", "valid_only", "channel_specific"
        self.display_mode = "all_nonzero"  # ACTIVE MODE
        self.filter_channel = None  # For channel_specific mode
        
        # View mode: "histogram" or "relations"
        self.view_mode = "histogram"
        
        # Relations data storage
        self.relations_data: List[Dict[str, float]] = []  # Store relation ratios for each bin
        self.relations_accumulation_mode = "last_bin"  # "last_bin" or "accumulate"
        self.relations_accumulation_bins = 10  # Number of bins to accumulate
        
        # Channel mapping for relations (default mapping)
        self.relation_channels = {
            "H": "4",
            "V": "3",
            "D": "2",
            "A": "1"
        }
        
        self.setup_gui()
        
    def setup_gui(self):
        """Set up the histogram GUI."""
        # Control frame at top
        control_frame = ctk.CTkFrame(self)
        control_frame.pack(side="top", fill="x", padx=10, pady=10)
        
        # Title
        self.title_label = ctk.CTkLabel(control_frame, text="Total Count Histogram", 
                    font=ctk.CTkFont(size=16, weight="bold"))
        self.title_label.pack(side="left", padx=10)
        
        # View mode switch
        self.view_switch = ctk.CTkSwitch(control_frame, text="View Relations", 
                                         command=self.toggle_view_mode,
                                         onvalue=True, offvalue=False)
        self.view_switch.pack(side="left", padx=20)
        
        # Histogram mode selector (for histogram view)
        self.histogram_mode_frame = ctk.CTkFrame(control_frame)
        self.histogram_mode_frame.pack(side="left", padx=20)
        ctk.CTkLabel(self.histogram_mode_frame, text="Mode:").pack(side="left", padx=5)
        self.mode_var = ctk.StringVar(value="all_nonzero")
        mode_menu = ctk.CTkOptionMenu(self.histogram_mode_frame, variable=self.mode_var,
                                      values=["all_nonzero", "valid_only", "channel_specific"],
                                      command=self.on_mode_changed)
        mode_menu.pack(side="left")
        
        # Relations mode selector (for relations view)
        self.relations_mode_frame = ctk.CTkFrame(control_frame)
        # Will be packed when switching to relations view
        ctk.CTkLabel(self.relations_mode_frame, text="Mode:").pack(side="left", padx=5)
        self.accumulation_var = ctk.StringVar(value="last_bin")
        accumulation_menu = ctk.CTkOptionMenu(self.relations_mode_frame, variable=self.accumulation_var,
                                              values=["last_bin", "accumulate"],
                                              command=self.on_accumulation_mode_changed,
                                              width=120)
        accumulation_menu.pack(side="left", padx=5)
        
        # Bins to accumulate entry (only for accumulate mode in relations view)
        self.accumulate_bins_frame = ctk.CTkFrame(control_frame)
        # Will be packed when accumulate mode is selected
        ctk.CTkLabel(self.accumulate_bins_frame, text="Bins:").pack(side="left", padx=5)
        self.accumulate_bins_var = ctk.StringVar(value="10")
        self.accumulate_bins_entry = ctk.CTkEntry(self.accumulate_bins_frame, 
                                                   textvariable=self.accumulate_bins_var,
                                                   width=60)
        self.accumulate_bins_entry.pack(side="left")
        self.accumulate_bins_entry.bind("<Return>", lambda e: self.update_relations_display())
        
        # Channel selector (for channel_specific mode in histogram view)
        self.channel_frame = ctk.CTkFrame(control_frame)
        self.channel_frame.pack(side="left", padx=10)
        ctk.CTkLabel(self.channel_frame, text="Channel:").pack(side="left", padx=5)
        self.channel_var = ctk.StringVar(value="1")
        self.channel_menu = ctk.CTkOptionMenu(self.channel_frame, variable=self.channel_var,
                                              values=["1", "2", "3", "4"],
                                              command=lambda _: self.update_histogram())
        self.channel_menu.pack(side="left")
        self.channel_frame.pack_forget()  # Hidden by default
        
        # Reset button
        reset_btn = ctk.CTkButton(control_frame, text="Reset Data", 
                                  command=self.reset_histogram_data, width=100)
        reset_btn.pack(side="right", padx=10)
        
        # Container for switching between views
        self.content_container = ctk.CTkFrame(self)
        self.content_container.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Setup histogram view
        self.setup_histogram_view()
        
        # Setup relations view
        self.setup_relations_view()
        
        # Show histogram by default
        self.histogram_frame.pack(fill="both", expand=True)
        self.relations_frame.pack_forget()
        self.relations_mode_frame.pack_forget()  # Hide relations mode initially
        self.accumulate_bins_frame.pack_forget()  # Hide bins entry initially
        
        # Info label at bottom
        self.info_label = ctk.CTkLabel(self, text="No data collected yet", 
                                       font=ctk.CTkFont(size=11))
        self.info_label.pack(side="bottom", pady=5)
    
    def setup_histogram_view(self):
        """Setup the histogram matplotlib view."""
        self.histogram_frame = ctk.CTkFrame(self.content_container)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Total Counts per Bin")
        self.ax.set_ylabel("Frequency")
        self.ax.set_title("Distribution of Total Counts")
        self.ax.grid(True, alpha=0.3)
        
        # Embed figure in tkinter
        self.canvas = FigureCanvasTkAgg(self.figure, self.histogram_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def setup_relations_view(self):
        """Setup the relations analysis view."""
        self.relations_frame = ctk.CTkFrame(self.content_container)
        
        # Grid frame for relations display (no more top control frame here)
        self.relations_grid = ctk.CTkFrame(self.relations_frame, fg_color="transparent")
        self.relations_grid.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        
        # Configure grid
        self.relations_grid.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Create 4 columns for H/V, V/H, D/A, A/D
        self.relation_pairs = [
            ("H", "V", "H/(H+V)"),
            ("V", "H", "V/(V+H)"),
            ("D", "A", "D/(D+A)"),
            ("A", "D", "A/(A+D)")
        ]
        
        self.relation_channel_vars = {}
        self.relation_value_labels = {}
        
        for col_idx, (numerator, denominator, label_text) in enumerate(self.relation_pairs):
            # Column frame
            col_frame = ctk.CTkFrame(self.relations_grid, fg_color=("gray85", "gray25"))
            col_frame.grid(row=0, column=col_idx, padx=10, pady=10, sticky="nsew")
            
            # Title
            ctk.CTkLabel(col_frame, text=label_text, 
                        font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5))
            
            # Channel selectors
            channel_frame = ctk.CTkFrame(col_frame, fg_color="transparent")
            channel_frame.pack(pady=10, padx=5)
            
            # Numerator channel selector
            ctk.CTkLabel(channel_frame, text=f"{numerator} Ch:").grid(row=0, column=0, padx=5, pady=2, sticky="e")
            num_var = ctk.StringVar(value=self.relation_channels[numerator])
            num_menu = ctk.CTkOptionMenu(channel_frame, variable=num_var,
                                         values=["1", "2", "3", "4"],
                                         command=lambda _: self.update_relations_display(),
                                         width=70)
            num_menu.grid(row=0, column=1, padx=5, pady=2)
            self.relation_channel_vars[numerator] = num_var
            
            # Denominator channel selector
            ctk.CTkLabel(channel_frame, text=f"{denominator} Ch:").grid(row=1, column=0, padx=5, pady=2, sticky="e")
            denom_var = ctk.StringVar(value=self.relation_channels[denominator])
            denom_menu = ctk.CTkOptionMenu(channel_frame, variable=denom_var,
                                           values=["1", "2", "3", "4"],
                                           command=lambda _: self.update_relations_display(),
                                           width=70)
            denom_menu.grid(row=1, column=1, padx=5, pady=2)
            self.relation_channel_vars[denominator] = denom_var
            
            # Divider
            ctk.CTkFrame(col_frame, height=2, fg_color=("gray70", "gray40")).pack(fill="x", padx=20, pady=10)
            
            # Average/Current ratio display
            ratio_display_frame = ctk.CTkFrame(col_frame, fg_color=("gray90", "gray20"))
            ratio_display_frame.pack(pady=10, padx=10, fill="x")
            
            ctk.CTkLabel(ratio_display_frame, text="Ratio:", 
                        font=ctk.CTkFont(size=11)).pack(pady=(5, 0))
            ratio_label = ctk.CTkLabel(ratio_display_frame, text="—", 
                                      font=ctk.CTkFont(size=20, weight="bold"))
            ratio_label.pack(pady=(0, 5))
            self.relation_value_labels[f"{numerator}/{denominator}"] = ratio_label
        
        self.relations_grid.grid_rowconfigure(0, weight=1)
        
    def toggle_view_mode(self):
        """Toggle between histogram and relations view."""
        if self.view_switch.get():
            # Switch to relations view
            self.view_mode = "relations"
            self.view_switch.configure(text="View Histogram")
            self.title_label.configure(text="Base Relations Analysis")
            self.histogram_frame.pack_forget()
            self.relations_frame.pack(fill="both", expand=True)
            
            # Hide histogram controls, show relations controls
            self.histogram_mode_frame.pack_forget()
            self.channel_frame.pack_forget()
            self.relations_mode_frame.pack(side="left", padx=20, after=self.view_switch)
            
            # Show bins entry if in accumulate mode
            if self.accumulation_var.get() == "accumulate":
                self.accumulate_bins_frame.pack(side="left", padx=10, after=self.relations_mode_frame)
            
            self.update_relations_display()
        else:
            # Switch to histogram view
            self.view_mode = "histogram"
            self.view_switch.configure(text="View Relations")
            self.title_label.configure(text="Total Count Histogram")
            self.relations_frame.pack_forget()
            self.histogram_frame.pack(fill="both", expand=True)
            
            # Hide relations controls, show histogram controls
            self.relations_mode_frame.pack_forget()
            self.accumulate_bins_frame.pack_forget()
            self.histogram_mode_frame.pack(side="left", padx=20, after=self.view_switch)
            
            # Show channel selector if in channel_specific mode
            if self.display_mode == "channel_specific":
                self.channel_frame.pack(side="left", padx=10, after=self.histogram_mode_frame)
            
            self.update_histogram()
    
    def on_accumulation_mode_changed(self, mode: str):
        """Handle accumulation mode change."""
        self.relations_accumulation_mode = mode
        
        if mode == "accumulate":
            self.accumulate_bins_frame.pack(side="left", padx=10, after=self.relations_mode_frame)
        else:
            self.accumulate_bins_frame.pack_forget()
        
        # Trim relations_data to appropriate size for the new mode
        try:
            if mode == "last_bin":
                max_needed = 1
            else:  # accumulate mode
                max_needed = int(self.accumulate_bins_var.get())
            
            if len(self.relations_data) > max_needed:
                self.relations_data = self.relations_data[-max_needed:]
        except (ValueError, AttributeError):
            pass  # Keep current data if there's an error
        
        self.update_relations_display()
    
    def on_mode_changed(self, new_mode: str):
        """Handle display mode change."""
        self.display_mode = new_mode
        
        # Show/hide channel selector
        if new_mode == "channel_specific":
            self.channel_frame.pack(side="left", padx=10)
        else:
            self.channel_frame.pack_forget()
        
        self.update_histogram()
    
    def add_measurement(self, counts: Dict[int, int]):
        """Add a new measurement to histogram data.
        
        Filtering modes:
        - all_nonzero: Include bins where total count > 0 (ACTIVE)
        - valid_only: Include only bins with exactly one channel having counts
        - channel_specific: Include only bins where specified channel has counts
        """
        total_count = sum(counts.values())
        
        if self.display_mode == "all_nonzero":
            # ACTIVE MODE: Include all bins with non-zero total count
            if total_count > 0:
                self.histogram_data.append(total_count)
        
        elif self.display_mode == "valid_only":
            # Include only valid bins (exactly one channel with counts >= 1)
            channels_with_counts = [ch for ch, count in counts.items() if count >= 1]
            if len(channels_with_counts) == 1 and total_count > 0:
                self.histogram_data.append(total_count)
        
        elif self.display_mode == "channel_specific":
            # Include only bins where the specified channel has counts
            try:
                filter_channel = int(self.channel_var.get())
                if counts.get(filter_channel, 0) >= 1 and total_count > 0:
                    self.histogram_data.append(total_count)
            except ValueError:
                pass
        
        # Calculate and store relations for this bin
        self.calculate_bin_relations(counts)
        
        # Update the appropriate view
        if self.view_mode == "histogram":
            self.update_histogram()
        else:
            self.update_relations_display()
    
    def calculate_bin_relations(self, counts: Dict[int, int]):
        """Calculate relation ratios for the current bin."""
        relations = {}
        
        for numerator, denominator, _ in self.relation_pairs:
            try:
                num_channel = int(self.relation_channel_vars[numerator].get())
                denom_channel = int(self.relation_channel_vars[denominator].get())
                
                num_count = counts.get(num_channel, 0)
                denom_count = counts.get(denom_channel, 0)
                
                total = num_count + denom_count
                
                if total > 0:
                    ratio = num_count / total
                    relations[f"{numerator}/{denominator}"] = ratio
                else:
                    relations[f"{numerator}/{denominator}"] = None
            except (ValueError, ZeroDivisionError):
                relations[f"{numerator}/{denominator}"] = None
        
        self.relations_data.append(relations)
        
        # Keep only necessary data based on accumulation mode
        # For "last_bin" mode, we only need 1 entry
        # For "accumulate" mode, we only need the last N entries
        try:
            if self.relations_accumulation_mode == "last_bin":
                max_needed = 1
            else:  # accumulate mode
                max_needed = int(self.accumulate_bins_var.get())
        except (ValueError, AttributeError):
            max_needed = 10  # Default fallback
        
        # Trim the list if it exceeds what we need
        if len(self.relations_data) > max_needed:
            self.relations_data = self.relations_data[-max_needed:]
    
    def update_relations_display(self):
        """Update the relations display based on current mode."""
        if self.relations_accumulation_mode == "last_bin":
            # Show only the last bin's ratios
            if self.relations_data:
                last_relations = self.relations_data[-1]
                for key, value in last_relations.items():
                    if key in self.relation_value_labels:
                        if value is not None:
                            percentage = value * 100  # Convert ratio to percentage
                            self.relation_value_labels[key].configure(text=f"{percentage:.1f}%")
                        else:
                            self.relation_value_labels[key].configure(text="—")
                
                self.info_label.configure(text=f"Showing last bin | Total bins: {len(self.relations_data)}")
            else:
                for label in self.relation_value_labels.values():
                    label.configure(text="—")
                self.info_label.configure(text="No data collected yet")
        
        elif self.relations_accumulation_mode == "accumulate":
            # Calculate mean of last N bins
            try:
                n_bins = int(self.accumulate_bins_var.get())
            except ValueError:
                n_bins = 10
            
            if self.relations_data:
                # Get last N bins (or all if less than N)
                recent_data = self.relations_data[-n_bins:]
                actual_bins = len(recent_data)
                
                # Calculate mean for each relation
                for key in self.relation_value_labels.keys():
                    valid_values = [rel[key] for rel in recent_data if rel.get(key) is not None]
                    
                    if valid_values:
                        mean_ratio = sum(valid_values) / len(valid_values)
                        percentage = mean_ratio * 100  # Convert ratio to percentage
                        self.relation_value_labels[key].configure(text=f"{percentage:.1f}%")
                    else:
                        self.relation_value_labels[key].configure(text="—")
                
                self.info_label.configure(
                    text=f"Mean of last {actual_bins} bins (requested: {n_bins}) | Total bins: {len(self.relations_data)}"
                )
            else:
                for label in self.relation_value_labels.values():
                    label.configure(text="—")
                self.info_label.configure(text="No data collected yet")
    
    def update_histogram(self):
        """Redraw the histogram with current data."""
        self.ax.clear()
        
        if not self.histogram_data:
            self.ax.text(0.5, 0.5, "No data to display", 
                        ha='center', va='center', transform=self.ax.transAxes,
                        fontsize=14, color='gray')
            self.ax.set_xlabel("Total Counts per Bin")
            self.ax.set_ylabel("Frequency")
            self.ax.set_title("Distribution of Total Counts")
            self.info_label.configure(text="No data collected yet")
        else:
            # Calculate histogram
            self.ax.hist(self.histogram_data, bins=20, color='#fe9409', 
                        alpha=0.7, edgecolor='black')
            self.ax.set_xlabel("Total Counts per Bin")
            self.ax.set_ylabel("Frequency")
            self.ax.set_title("Distribution of Total Counts")
            self.ax.grid(True, alpha=0.3)
            
            # Update info label with statistics
            mean_count = sum(self.histogram_data) / len(self.histogram_data)
            max_count = max(self.histogram_data)
            min_count = min(self.histogram_data)
            self.info_label.configure(
                text=f"Bins: {len(self.histogram_data)} | Mean: {mean_count:.1f} | "
                     f"Min: {min_count} | Max: {max_count}"
            )
        
        self.canvas.draw()
    
    def reset_histogram_data(self):
        """Clear all histogram data."""
        self.histogram_data.clear()
        self.relations_data.clear()
        
        if self.view_mode == "histogram":
            self.update_histogram()
        else:
            self.update_relations_display()
        
        logger.info("Histogram and relations data reset")


class TimeTaggerControllerGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("TimeTagger Hardware Interface for Testing Bob")
        self.geometry("600x900")

        # Thread pool for hardware operations
        self.thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="TimeTaggerOp")
        
        # GUI update queue for thread-safe communication
        self.gui_update_queue = Queue()

        # TimeTagger controller
        self.timetagger_controller: Optional[SimpleTimeTaggerController] = None
        self.driver: Optional[Union[SimpleTimeTaggerHardware, SimpleTimeTaggerSimulator]] = None
        self.connected = False
        self.results_history: List[Dict[str, int]] = []
        self.result_cells: List[Dict[str, ctk.CTkLabel]] = []
        
        # Histogram window
        self.histogram_window: Optional[HistogramWindow] = None
        
        # Measurement state
        self.is_measuring = False
        self.measurement_task = None
        self.time_check_update_gui_process = 50  # ms
        self.current_max_rows = 0
        
        # Statistics tracking
        self.stats_bin_count = 20  # Number of VALID bins to track for statistics
        self.recent_counts: List[Dict[int, int]] = []  # Store ALL recent bin counts
        self.valid_counts: List[Dict[int, int]] = []  # Store only VALID bin counts (filtered)
        self.percentage_labels: Dict[str, ctk.CTkLabel] = {}  # Labels for percentages
        self.stats_counter_label: Optional[ctk.CTkLabel] = None  # Label showing valid bins / total bins

        # Variables for measurement configuration
        self.device_name_var = ctk.StringVar(value="Swabian TimeTagger")
        self.status_var = ctk.StringVar(value="● Disconnected")
        # self.available_channels_var = ctk.StringVar(value="1,2,3,4")
        self.available_channels_var = ctk.StringVar(value="4,3,2,1")
        self.polarization_vars: Dict[str, ctk.StringVar] = {
            pol: ctk.StringVar(value=str(len(POLARIZATIONS) - idx)) for idx, (pol, _) in enumerate(POLARIZATIONS)
        }
        self.bin_duration_ms_var = ctk.StringVar(value="1000")
        self.num_rows_var = ctk.StringVar(value="10")
        self.stats_bins_var = ctk.StringVar(value="20")
        self.continuous_var = ctk.BooleanVar(value=True)
        self.repeat_count_var = ctk.StringVar(value="10")
        self.use_simulator_var = ctk.BooleanVar(value=USE_SIMULATION)
        
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

        # Configure grid layout for main window
        self.grid_rowconfigure(0, weight=0)  # Connection frame - fixed size
        self.grid_rowconfigure(1, weight=0)  # Settings frame - fixed size
        self.grid_rowconfigure(2, weight=3)  # Results frame - expandable (higher weight)
        # self.grid_rowconfigure(3, weight=1)  # Log frame - expandable (lower weight)
        self.grid_columnconfigure(0, weight=1)
        
        # Top frame for connection
        connection_frame = ctk.CTkFrame(self)
        connection_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        self.setup_connection_frame(connection_frame)

        # Second top frame for channel and measurement config
        setting_frame = ctk.CTkFrame(self)
        setting_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(10, 0))
        self.setup_channel_measurement_frame(setting_frame)

        # Bottom frame for results
        results_frame = ctk.CTkFrame(self)
        results_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(10, 0))
        self.setup_results_frame(results_frame)
        
        # LOG
        log_frame = ctk.CTkFrame(self)
        # log_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(log_frame, text="Log", font=ctk.CTkFont(size=16, weight="bold")).pack(padx=10, pady=(10, 5))
        self.logbox = ctk.CTkTextbox(log_frame, height=120)
        self.logbox.pack(fill="both", expand=True, padx=10, pady=(0,10))

        # TODO: UNCOMENT AFTER NEW MEASUREMENT RESULTS SECTION
        # self.update_result_table_capacity(int(self.num_rows_var.get()))

    def setup_connection_frame(self, connection_frame: ctk.CTkFrame):
        """Setup the connection configuration widgets."""
        # connection_frame = ctk.CTkFrame(parent_frame, fg_color="red")
        # connection_frame.pack(fill="both", expand=True, padx=10, pady=10)
        connection_frame.grid_columnconfigure((1,2,3), weight=1)

        # --- Connection ---
        # Title
        title_label = ctk.CTkLabel(connection_frame, text="TimeTagger Connection", 
                                   font=ctk.CTkFont(size=16, weight="bold"))
        title_label.grid(row=0, column=0, columnspan=4, padx=10, pady=(5,0))

        # Device name (for display only, hardware auto-detects)
        # TODO: Change this entry to a dropdown menu (combobox) for multiple devices. And subquent device_name_var updates.
        ctk.CTkLabel(connection_frame, text="Device Name:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        self.device_name_entry = ctk.CTkEntry(connection_frame, textvariable=self.device_name_var, 
                                             state="readonly")
        self.device_name_entry.grid(row=1, column=1, columnspan=2, padx=10, pady=5, sticky="we")
 
        # Scan button
        self.scan_button = ctk.CTkButton(connection_frame, text="Scan", command=self.scan_timetagger, width=80)
        self.scan_button.grid(row=1, column=3, padx=10, pady=5, sticky="e")


        # Status label
        ctk.CTkLabel(connection_frame, text="Status:", font=ctk.CTkFont(weight="bold")).grid(row=2, column=0, padx=6, pady=5, sticky="e")
        self.status_label = ctk.CTkLabel(connection_frame, textvariable=self.status_var)
        self.status_label.grid(row=2, column=1, padx=6, pady=5, sticky="w")

        # Use simulator switch
        self.use_sim_switch = ctk.CTkSwitch(connection_frame, text="Use Simulator", onvalue=True, offvalue=False)
        self.use_sim_switch.grid(row=2, column=2, padx=10, pady=5)
        self.use_sim_switch.select() if USE_SIMULATION else self.use_sim_switch.deselect()

        # Connect/Disconnect button
        self.connect_button = ctk.CTkButton(connection_frame, text="Connect", command=self.toggle_connection_async, width=100)
        self.connect_button.grid(row=2, column=3, padx=10, pady=(5,10), sticky="e")

    def setup_channel_measurement_frame(self, parent_frame: ctk.CTkFrame):
        """Setup the channel and measurement configuration widgets."""
        parent_frame.grid_columnconfigure((0,1), weight=1)
        parent_frame.grid_rowconfigure(0, weight=1)
        # --- Channel Configuration ---
        channel_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
        channel_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        # Title
        ctk.CTkLabel(channel_frame, text="Channel Mapping", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(5,0))

        # TODO: CONFIRM THAT THIS IS NOT NEEDED ANYMORE BEFORE REMOVING
        # self.channel_map_entries = {}
        # self.channel_map_labels = {1: "H (0°)", 2: "V (90°)", 3: "D (45°)", 4: "A (135°)"}
        
        # channel_grid_frame = ctk.CTkFrame(channel_frame)
        # channel_grid_frame.pack(pady=5, padx=10)

        # for i in range(4):
        #     channel = i + 1
        #     ctk.CTkLabel(channel_grid_frame, text=f"Channel {channel}:").grid(row=i, column=0, padx=5, pady=2, sticky="w")
        #     entry = ctk.CTkEntry(channel_grid_frame, placeholder_text=f"e.g., {self.channel_map_labels.get(channel, '')}")
        #     entry.grid(row=i, column=1, padx=5, pady=2)
        #     entry.insert(0, self.channel_map_labels.get(channel, ''))
        #     self.channel_map_entries[channel] = entry

        channel_frame_1 = ctk.CTkFrame(channel_frame)
        channel_frame_1.pack(pady=0, padx=0, fill="x", expand=True)
        ctk.CTkLabel(channel_frame_1, text="Channels (comma separated):", font=ctk.CTkFont(size=12, weight="bold")).pack(side="top", padx=0, pady=0, fill="x", expand=True)
        
        self.channel_entry = ctk.CTkEntry(channel_frame_1, textvariable=self.available_channels_var, width=100)
        self.channel_entry.pack(side="left", padx=(0,2), pady=(0,5), fill="x", expand=True)
        refresh_button = ctk.CTkButton(
            channel_frame_1,
            text="Refresh Options",
            command=self.refresh_option_menus,
            width=120,
        )
        refresh_button.pack(side="left", padx=(2,0), pady=(0,5))

        channel_frame_2 = ctk.CTkFrame(channel_frame)
        channel_frame_2.pack(pady=2, padx=0, fill="both", expand=True)
        channel_frame_2.grid_columnconfigure(0, weight=1)
        channel_frame_2.grid_columnconfigure(1, weight=1)
        self.channel_options: Dict[str, ctk.CTkOptionMenu] = {}
        for idx, (code, description) in enumerate(POLARIZATIONS):
            ctk.CTkLabel(channel_frame_2, text=f"{description}:").grid(row=idx, column=0, padx=6, pady=4, sticky="e")
            option = ctk.CTkOptionMenu(
                channel_frame_2,
                variable=self.polarization_vars[code],
                values=self.get_available_channels(),
            )
            option.grid(row=idx, column=1, padx=(6,0), pady=4, sticky="w")
            self.channel_options[code] = option
            setattr(self, f"pol_option_{code}", option)


        # --- Measurement Configuration ---
        measure_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
        measure_frame.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        measure_frame.grid_columnconfigure((0,1), weight=1)
        measure_frame.grid_rowconfigure((0,1,2,3,4), weight=1)

        ctk.CTkLabel(measure_frame, text="Measurement Control", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=5, pady=(5,10))

        ctk.CTkLabel(measure_frame, text="Time per Bin (ms):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.time_bin_entry = ctk.CTkEntry(measure_frame, width=100, textvariable=self.bin_duration_ms_var)
        self.time_bin_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        # self.time_bin_entry.insert(0, "1000")

        ctk.CTkLabel(measure_frame, text="Num Bins to Show:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.num_bins_entry = ctk.CTkEntry(measure_frame, width=100, textvariable=self.num_rows_var)
        self.num_bins_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        # self.num_bins_entry.insert(0, "10")

        # TODO: CONFIRM IF THIS IS NEEDED ANYMORE BEFORE REMOVING
        # self.continuous_switch = ctk.CTkSwitch(measure_frame, text="Continuous", onvalue=True, offvalue=False)
        # self.continuous_switch.grid(row=3, column=0, padx=10, pady=5)
        # self.continuous_switch.select()

        self.radio_continuous = ctk.CTkRadioButton(
            measure_frame,
            text="Continuous",
            variable=self.continuous_var,
            value=True,
            command=self.on_mode_changed,
        )
        self.radio_continuous.grid(row=3, column=0, padx=4, pady=4, sticky="w")
        self.radio_finite = ctk.CTkRadioButton(
            measure_frame,
            text="Finite",
            variable=self.continuous_var,
            value=False,
            command=self.on_mode_changed,
        )
        self.radio_finite.grid(row=3, column=1, padx=4, pady=4, sticky="w")

        ctk.CTkLabel(measure_frame, text="Fixed Time (s):").grid(row=4, column=0, padx=5, pady=5, sticky="e")
        self.fixed_time_entry = ctk.CTkEntry(measure_frame, width=100, state="disabled")
        self.fixed_time_entry.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        # self.continuous_switch.configure(command=lambda: self.fixed_time_entry.configure(state="normal" if not self.continuous_switch.get() else "disabled"))

        # TODO: CHECK IF IS BETTER TO HAVE A REPEAT COUNT FOR FIXED TIME
        # ctk.CTkLabel(mode_frame, text="Finite repeats").grid(row=2, column=0, padx=4, pady=(6, 4), sticky="w")
        # self.repeat_entry = ctk.CTkEntry(mode_frame, textvariable=self.repeat_count_var)
        # self.repeat_entry.grid(row=2, column=1, padx=4, pady=(6, 4), sticky="ew")
        
        # Buttons frame for measurement and histogram
        buttons_frame = ctk.CTkFrame(measure_frame, fg_color="transparent")
        buttons_frame.grid(row=5, column=0, columnspan=2, padx=10, pady=5)
        
        self.measure_button = ctk.CTkButton(buttons_frame, text="Start Measurement", 
                                           command=self.toggle_measurement, state="disabled", width=140)
        self.measure_button.pack(side="left", padx=(0, 5))
        
        self.histogram_button = ctk.CTkButton(buttons_frame, text="Histogram", 
                                             command=self.open_histogram_window, width=100)
        self.histogram_button.pack(side="left")


    def setup_results_frame(self, parent_frame: ctk.CTkFrame):
        """Setup the frame for displaying measurement results."""
        parent_frame.grid_columnconfigure((0,1,2,3,4), weight=1)
        parent_frame.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(parent_frame, text="Measurements Results (Counts per Bin)", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=5, pady=(5,0))

        # Header labels
        bin_header = ctk.CTkLabel(parent_frame, text="Total Count", font=ctk.CTkFont(size=13, weight="bold"))
        bin_header.grid(row=1, column=0, padx=15, pady=(5,0))
        polarizations = [desc for _, desc in POLARIZATIONS]
        for col_idx, pol_name in enumerate(polarizations):
            # Header
            header = ctk.CTkLabel(parent_frame, text=pol_name, 
                                font=ctk.CTkFont(size=13, weight="bold"))
            header.grid(row=1, column=col_idx+1, padx=15, pady=(5,0))

        self.results_scrollable_frame = ctk.CTkScrollableFrame(parent_frame)
        # self.results_scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.results_scrollable_frame.grid(row=2, column=0, columnspan=5, sticky="nsew", padx=10, pady=10)
        # self.results_scrollable_frame.grid_columnconfigure(0, weight=1)
        self.results_scrollable_frame.grid_columnconfigure(tuple(range(len(polarizations) + 1)), weight=1) 
        # columnconfigure: len of POLARIZATIONS + bin index
        
        # Statistics frame (row 3)
        stats_frame = ctk.CTkFrame(parent_frame)
        stats_frame.grid(row=3, column=0, columnspan=5, sticky="ew", padx=10, pady=(0, 10))
        stats_frame.grid_columnconfigure((0,1,2,3,4), weight=1)
        
        # Column 0: Stats bin count entry with counter label
        stats_config_frame = ctk.CTkFrame(stats_frame, fg_color="transparent")
        stats_config_frame.grid(row=0, column=0, padx=0, pady=0, sticky="ew")
        stats_config_frame.grid_columnconfigure(0, weight=1)
        
        self.stats_bins_entry = ctk.CTkEntry(stats_config_frame, textvariable=self.stats_bins_var, width=10, placeholder_text="Stats Bins")
        self.stats_bins_entry.grid(row=0, column=0, padx=(10, 0), pady=5, sticky="ew")
        self.stats_bins_entry.bind("<Return>", lambda e: self.update_stats_bin_count())
        self.stats_bins_entry.bind("<FocusOut>", lambda e: self.update_stats_bin_count())
        
        self.stats_counter_label = ctk.CTkLabel(stats_config_frame, text="0/20", font=ctk.CTkFont(size=11))
        self.stats_counter_label.grid(row=0, column=1, padx=(5, 5), pady=5, sticky="w")
        
        # Columns 1-4: Percentage displays for H, V, D, A
        for col_idx, (code, description) in enumerate(POLARIZATIONS):
            percentage_frame = ctk.CTkFrame(stats_frame, fg_color=("gray85", "gray25"))
            percentage_frame.grid(row=0, column=col_idx+1, padx=10, pady=5, sticky="ew")
            
            # ctk.CTkLabel(percentage_frame, text=f"{code} %", font=ctk.CTkFont(size=11, weight="bold")).pack(pady=(5,2))
            percentage_label = ctk.CTkLabel(percentage_frame, text="0.0%", font=ctk.CTkFont(size=14))
            percentage_label.pack(pady=(5,5))
            self.percentage_labels[code] = percentage_label
        

        # self.count_labels = {}
        # TODO: CHECK IF DEFINE THE COUNT LABELS SO TO UPDATE THEM INSTEAD OF RECREATING EVERY TIME
        # self.restart_initial_display()

    def toggle_connection_async(self):
        """Handle the connect/disconnect button click."""
        if self.timetagger_controller and self.timetagger_controller.is_initialized():
            self.disconnect_from_timetagger()
        else:
            self.connect_to_timetagger()

    def on_mode_changed(self):
        """Handle measurement mode change"""
        self.continuous_mode = self.continuous_var.get()
        
        if self.continuous_mode:
            self.fixed_time_entry.configure(state="disabled")
        else:
            self.fixed_time_entry.configure(state="normal")

    def update_stats_bin_count(self):
        """Update the number of VALID bins to track for statistics."""
        try:
            new_count = int(self.stats_bins_var.get())
            if new_count <= 0:
                logger.warning("Stats bin count must be positive. Reverting to previous value.")
                self.stats_bins_var.set(str(self.stats_bin_count))
                return
            
            old_count = self.stats_bin_count
            self.stats_bin_count = new_count
            
            # Adjust valid_counts list (keep only last N valid bins)
            if new_count < old_count:
                # Remove oldest valid entries if new count is smaller
                self.valid_counts = self.valid_counts[-new_count:]
            # If new count is larger, we just keep existing data and will add more as measurements come
            
            # Recalculate percentages
            self.update_statistics()
            logger.info(f"Statistics bin count updated to {new_count}")
            
        except ValueError:
            logger.warning("Invalid stats bin count. Please enter a valid integer.")
            self.stats_bins_var.set(str(self.stats_bin_count))

    @run_in_background
    def scan_timetagger(self):
        """Scan for connected TimeTagger devices in a background thread."""
        self.schedule_gui_update(lambda: self.connect_button.configure(state="disabled", text="Scanning..."))
        self.device_name_var.set("Scanning...")
        try:
            temp_timetagger = SimpleTimeTaggerHardware()
            devices = temp_timetagger.scan_for_devices(5.0)
            if devices:
                logger.info(f"Found TimeTagger devices: {devices}")
                self.schedule_gui_update(lambda: self.log_message(f"Found devices: {devices}", "green"))
                self.schedule_gui_update(lambda: self.device_name_var.set(devices[0]))
            else:
                logger.info("No TimeTagger devices found.")
                self.schedule_gui_update(lambda: self.log_message("No devices found.", "red"))
                self.schedule_gui_update(lambda: self.device_name_var.set("No devices found"))
        except Exception as e:
            logger.error(f"Error scanning for TimeTagger devices: {e}", exc_info=True)
            self.schedule_gui_update(lambda: self.log_message(f"Error scanning for devices: {e}", "red"))
            self.schedule_gui_update(lambda: self.device_name_var.set("Error during scan"))
        finally:
            self.schedule_gui_update(lambda: self.connect_button.configure(state="normal", text="Connect"))

    @run_in_background
    def connect_to_timetagger(self):
        """Connect to the TimeTagger in a background thread."""
        self.schedule_gui_update(lambda: self.connect_button.configure(state="disabled", text="Connecting..."))
        
        use_sim = self.use_sim_switch.get()
        
        channels_int = [int(ch) for ch in self.get_available_channels() if ch.isdigit()]
        
        try:
            if use_sim:
                logger.info("Using TimeTagger simulator.")
                self.driver = SimpleTimeTaggerSimulator(
                    detector_channels=channels_int,
                    dark_count_rate=50.0,
                    signal_count_rate=100.0,
                    signal_probability=0.1
                )
            else:
                logger.info("Using TimeTagger hardware.")
                self.driver = SimpleTimeTaggerHardware(detector_channels=channels_int)

            self.timetagger_controller = SimpleTimeTaggerController(self.driver)

            if self.timetagger_controller.initialize():
                logger.info("TimeTagger connected successfully.")
                connection_type = "simulator" if use_sim else "hardware"
                self.schedule_gui_update(lambda: self.log_message(f"TimeTagger connected successfully ({connection_type}).", "green"))
                self.connected = True
                self.schedule_gui_update(lambda: self.status_var.set(self.connection_status_text()))
                self.schedule_gui_update(lambda: self.status_label.configure(text_color="green"))
                self.schedule_gui_update(lambda: self.connect_button.configure(text="Disconnect"))
                self.schedule_gui_update(lambda: self.enable_controls(True))
                self.schedule_gui_update(self.refresh_option_menus)
            else:
                logger.error("Failed to initialize TimeTagger.")
                self.schedule_gui_update(lambda: self.status_var.set("● Connection Failed"))
                self.schedule_gui_update(lambda: self.log_message("Failed to initialize TimeTagger.", "red"))
                self.schedule_gui_update(lambda: self.status_label.configure(text_color="red"))
                self.timetagger_controller = None

        except Exception as e:
            logger.error(f"Error connecting to TimeTagger: {e}", exc_info=True)
            self.schedule_gui_update(lambda: self.status_var.set("● Error"))
            self.schedule_gui_update(lambda: self.log_message(f"Error connecting to TimeTagger: {e}", "red"))
            self.schedule_gui_update(lambda: self.status_label.configure(text_color="red"))
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
                self.connected = False
                logger.info("TimeTagger disconnected successfully.")
                self.schedule_gui_update(lambda: self.log_message("TimeTagger disconnected.", "orange"))
                self.schedule_gui_update(lambda: self.status_var.set(self.connection_status_text()))
                self.schedule_gui_update(lambda: self.status_label.configure(text_color="red"))
                self.schedule_gui_update(lambda: self.connect_button.configure(text="Connect"))
                self.schedule_gui_update(lambda: self.enable_controls(False))
            except Exception as e:
                logger.error(f"Error during TimeTagger shutdown: {e}", exc_info=True)
                self.schedule_gui_update(lambda: self.log_message(f"Error during disconnection: {e}", "red"))
            finally:
                self.timetagger_controller = None
                self.connected = False
        if self.driver:
            try:
                self.driver.shutdown()
            except Exception as e:
                logger.error(f"Error during driver shutdown: {e}", exc_info=True)
                self.schedule_gui_update(lambda: self.log_message(f"Error during driver shutdown: {e}", "red"))
            self.driver = None
            
        self.schedule_gui_update(lambda: self.connect_button.configure(state="normal"))

    def enable_controls(self, enabled: bool):
        """Enable or disable measurement controls."""
        self.measure_button.configure(state="normal" if enabled else "disabled")
        self.time_bin_entry.configure(state="normal" if enabled else "disabled")
        self.num_bins_entry.configure(state="normal" if enabled else "disabled")
        self.radio_continuous.configure(state="normal" if enabled else "disabled")
        self.radio_finite.configure(state="normal" if enabled else "disabled")
        # self.continuous_switch.configure(state="normal" if enabled else "disabled")
        
        # is_fixed_time = enabled and not self.continuous_switch.get()
        # is_fixed_time = enabled and not self.continuous_var.get()
        # self.fixed_time_entry.configure(state="normal" if is_fixed_time else "disabled")

        for option in self.channel_options.values():
            option.configure(state="normal" if enabled else "disabled")

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
            self.log_message("Error: TimeTagger not connected.", "red")
            return

        if self.is_measuring:
            logger.warning("Measurement already in progress.")
            self.log_message("Warning: Measurement already in progress.", "orange")
            return
        
        
        # Parse and validate measurement parameters
        try:
            self.time_per_bin = float(self.bin_duration_ms_var.get())
        except ValueError:
            logger.error("Invalid bin duration. Please enter a valid number.")
            self.log_message("Error: Invalid bin duration.", "red")
            return
        if self.time_per_bin <= 0:
            logger.error("Bin duration must be positive.")
            self.log_message("Error: Bin duration must be positive.", "red")
            return
        self.time_per_bin /= 1000.0

        try:
            self.num_bins_to_show = int(self.num_rows_var.get())
        except ValueError:
            logger.error("Invalid number of bins to show. Please enter a valid integer.")
            self.log_message("Error: Invalid number of bins to show.", "red")
            return
        if self.num_bins_to_show <= 0:
            logger.error("Number of bins to show must be positive.")
            self.log_message("Error: Number of bins to show must be positive.", "red")
            return

        self.is_continuous = self.continuous_var.get()
        self.total_measurement_time = None
        if not self.is_continuous:
            try:
                self.total_measurement_time = float(self.fixed_time_entry.get())
            except (ValueError, TypeError):
                logger.error("Invalid measurement parameters. Please enter valid numbers.")
                self.log_message("Error: Invalid measurement parameters.", "red")
                return
            
            if self.total_measurement_time <= 0:
                logger.error("Fixed measurement time must be positive.")
                self.log_message("Error: Fixed measurement time must be positive.", "red")
                return
        
        # Update result table capacity
        # self.update_result_table_capacity(self.num_bins_to_show)


        # self.mapping = self.get_polarization_mapping() # {'H': 1, 'V': 2, 'D': 3, 'A': 4}
        # if not self.mapping:
        #     self.status_var.set("invalid polarization mapping")
            # return
        # print(f"Using polarization mapping: {mapping}")
        # self.log_message(f"Using polarization mapping: {mapping}", "blue")


        # Clear previous results
        for widget in self.results_scrollable_frame.winfo_children():
            widget.destroy()
        
        # Clear statistics
        self.recent_counts = []
        self.valid_counts = []
        self.update_statistics()
        
        # Clear histogram and relations data in the histogram window if it exists
        if self.histogram_window and self.histogram_window.winfo_exists():
            self.histogram_window.histogram_data.clear()
            self.histogram_window.relations_data.clear()
            if self.histogram_window.view_mode == "histogram":
                self.histogram_window.update_histogram()
            else:
                self.histogram_window.update_relations_display()
        
        self.is_measuring = True
        self.timetagger_controller.set_measurement_duration(self.time_per_bin)
        self.measure_button.configure(text="Stop Measurement")
        self.measurement_start_time = time.time()
        self.log_message("Measurement started.")

        self.bin_row_index = 0
        # self.restart_initial_display()
        self.measurement_task = self.thread_pool.submit(self._measurement_loop)

    def stop_measurement(self):
        """Stop the measurement loop."""
        self.is_measuring = False
        if self.measurement_task:
            # The loop will check self.is_measuring and exit, no need to force future
            pass
        self.measure_button.configure(text="Start Measurement")
        self.log_message("Measurement stopped.")

    def _measurement_loop(self):
        """Background task for continuous measurement."""
        logger.info("Measurement loop started.")
        
        iteration = 0
        self.timetagger_controller.set_measurement_duration(self.time_per_bin)
        while self.is_measuring:

            # Check for fixed time limit
            iteration_time = iteration * self.time_per_bin
            if not self.is_continuous and self.total_measurement_time is not None:
                if (iteration_time) >= self.total_measurement_time:
                    self.schedule_gui_update(self.stop_measurement)
                    break
            
            try:
                counts = self.timetagger_controller.measure_counts()
                
                if counts:
                    self.schedule_gui_update(lambda c=counts: self.add_result_row(c))

            except Exception as e:
                logger.error(f"Error during measurement: {e}", exc_info=True)
                self.schedule_gui_update(lambda: self.log_message(f"Error: {e}", "red"))
                break
            
            iteration += 1
            # if repeat_target is not None and iteration >= repeat_target:
            #     self.schedule_gui_update(self.stop_measurement)
            #     break

            # if not self.is_measuring:
            #     self.schedule_gui_update(self.stop_measurement)
            #     break

        logger.info("Measurement loop stopped.")

    def add_result_row(self, counts: Dict[int, int]):
        """Add a new row of results to the GUI."""
        self.bin_row_index += 1
        
        # Save counts to recent_counts (all measurements)
        self.recent_counts.append(counts.copy())
        
        # Update histogram window if open
        if self.histogram_window and self.histogram_window.winfo_exists():
            self.histogram_window.add_measurement(counts)
        
        # Check if this is a valid bin (exactly one channel has counts >= 1)
        channels_with_counts = [ch for ch, count in counts.items() if count >= 1]
        
        if len(channels_with_counts) == 1:
            # Valid bin: add to valid_counts and maintain the limit
            self.valid_counts.append(counts.copy())
            
            # Keep only the last stats_bin_count VALID entries
            if len(self.valid_counts) > self.stats_bin_count:
                self.valid_counts.pop(0)
        
        # Update statistics display
        self.update_statistics()
        
        # Manage displayed bins
        if self.bin_row_index > self.num_bins_to_show:
            # Find and remove the oldest row 
            for widget in self.results_scrollable_frame.grid_slaves():
                if int(widget.grid_info()["row"]) == 0:
                    widget.destroy()
            # Shift all other rows up
            for widget in self.results_scrollable_frame.grid_slaves():
                 if widget.grid_info()["row"] > 0:
                    widget.grid(row=widget.grid_info()["row"] - 1)
            self.bin_row_index -=1


        row = self.bin_row_index-1
        
        # Calculate total counts for this bin
        total_counts = sum(counts.values())
        
        # If only one is non-zero, color the label green, else black
        label_color = "green4" if sum(1 for c in counts.values() if c > 0) == 1 else "black"
        ctk.CTkLabel(self.results_scrollable_frame, text=str(total_counts), text_color=label_color).grid(row=row, column=0, padx=5)
        pol_labels = self.get_polarization_labels()
        for i, channel in enumerate(pol_labels.keys()):
            count = counts.get(channel, 0)
            ctk.CTkLabel(self.results_scrollable_frame, text=str(count), text_color=label_color).grid(row=row, column=i + 1, padx=5)
            # self.count_labels[channel][row].configure(text=str(count))

        # TODO: TRY TO INSTEAD OF DESTROY AND MOVE ALL LABELS JUST MODIFY THE TEXT OF THE EXISTING LABELS
        # try:
        #     for pol_key in ["H", "V", "D", "A"]:
        #         data = self.measurement_data[pol_key]
        #         labels = self.count_labels[pol_key]
                
        #         # Display last N bins (where N is number of visible labels)
        #         display_data = data[-len(labels):]
                
        #         for idx, label in enumerate(labels):
        #             if idx < len(display_data):
        #                 count = display_data[idx]
        #                 label.configure(text=str(count))
        #             else:
        #                 label.configure(text="—")


    def get_polarization_labels(self) -> Dict[int, str]:
        """Get the current polarization labels from the GUI."""
        # return {ch: entry.get() for ch, entry in self.channel_map_entries.items()}
        # TODO: REMOVE THIS
        return {int(self.polarization_vars[code].get()): code for code, _ in POLARIZATIONS}
    
    @run_in_background
    def update_statistics(self):
        """Calculate and update the percentage statistics for each polarization.
        
        Uses only valid bins (where exactly one channel has counts >= 1).
        The stats_bin_count refers to the number of VALID bins to track,
        not the total number of measurements.
        """
        if not self.valid_counts:
            # No valid data yet, show 0%
            for code, _ in POLARIZATIONS:
                self.percentage_labels[code].configure(text="0.0%")
            if self.stats_counter_label:
                self.stats_counter_label.configure(text=f"0/{self.stats_bin_count}")
            return
        
        # Get current polarization mapping (channel -> polarization code)
        pol_labels = self.get_polarization_labels()  # {channel: code}
        
        # Count valid bins for each polarization
        pol_bin_counts = {code: 0 for code, _ in POLARIZATIONS}
        valid_bins = len(self.valid_counts)
        
        for bin_counts in self.valid_counts:
            # Each entry in valid_counts already has exactly one channel with counts
            channels_with_counts = [ch for ch, count in bin_counts.items() if count >= 1]
            
            if len(channels_with_counts) == 1:
                channel = channels_with_counts[0]
                pol_code = pol_labels.get(channel)
                if pol_code:
                    pol_bin_counts[pol_code] += 1
        
        # Update counter label
        if self.stats_counter_label:
            self.stats_counter_label.configure(text=f"{valid_bins}/{self.stats_bin_count}")
        
        # Calculate and update percentage labels
        if valid_bins > 0:
            for code, _ in POLARIZATIONS:
                percentage = (pol_bin_counts[code] / valid_bins) * 100
                self.percentage_labels[code].configure(text=f"{percentage:.1f}%")
        else:
            # No valid bins
            for code, _ in POLARIZATIONS:
                self.percentage_labels[code].configure(text="0.0%")
    
    def get_polarization_mapping(self) -> Dict[str, int]:
        """Build map from polarization to channel."""
        mapping: Dict[str, int] = {}
        for code, _ in POLARIZATIONS:
            value = self.polarization_vars[code].get()
            try:
                mapping[code] = int(value)
            except ValueError:
                return {}
        return mapping
        
    def push_measurement_result(self, result: Dict[str, int], timestamp: str) -> None:
        """Insert new measurement at top of the table."""
        result_with_time = {**result, "timestamp": timestamp}
        self.results_history.insert(0, result_with_time)
        self.results_history = self.results_history[: self.current_max_rows]
        self.refresh_results_table()

    def update_result_table_capacity(self, rows: int) -> None:
        """Rebuild result table when capacity changes."""
        if rows == self.current_max_rows and self.result_cells:
            return
        for widget in self.results_scrollable_frame.winfo_children():
            widget.destroy()


        self.result_cells = []
        for row in range(rows):
            cell_row: Dict[str, ctk.CTkLabel] = {}
            time_label = ctk.CTkLabel(self.results_scrollable_frame, text="-")
            time_label.grid(row=row, column=1, padx=4, pady=2, sticky="ew")
            cell_row["timestamp"] = time_label
            for col, (code, _) in enumerate(POLARIZATIONS, start=1):
                label = ctk.CTkLabel(self.results_scrollable_frame, text="-")
                label.grid(row=row, column=col + 1, padx=4, pady=2, sticky="ew")
                cell_row[code] = label
            self.result_cells.append(cell_row)

        self.current_max_rows = rows
        self.results_history = self.results_history[:rows]
        self.refresh_results_table()

    def refresh_results_table(self) -> None:
        """Update label text with latest history."""
        for row_index in range(self.current_max_rows):
            print(f"Updating row {row_index}, total history length: {len(self.results_history)}")
            if row_index < len(self.results_history):
                data = self.results_history[row_index]
                print(f"Refreshing row {row_index} with data: {data}")
                print(f"Result cells: {self.result_cells[row_index]}")
                for code, label in self.result_cells[row_index].items():
                    print(f"Updating cell for code {code} with label {label}")
                    label.configure(text=str(data.get(code, "-")))
            else:
                for label in self.result_cells[row_index].values():
                    label.configure(text="-")

    def restart_initial_display(self):
        """Update initial results display"""
        # Create the labels for the number of rows specified
        for bin_idx in range(int(self.num_rows_var.get())):  # Max 10 visible bins
            # Bin index label
            bin_label = ctk.CTkLabel(self.results_scrollable_frame, text=f"{bin_idx+1}", 
                                    font=ctk.CTkFont(size=12))
            bin_label.grid(row=bin_idx, column=0, padx=15, pady=2)

        for col_idx, (pol_key, _) in enumerate(POLARIZATIONS):
            # Initialize list for count labels
            self.count_labels[pol_key] = []

            # Create labels for bins (will be populated during measurement)
            for bin_idx in range(int(self.num_rows_var.get())):  # Max 10 visible bins
                count_label = ctk.CTkLabel(self.results_scrollable_frame, text="—", 
                                          font=ctk.CTkFont(size=12))
                count_label.grid(row=bin_idx, column=col_idx+1, padx=15, pady=2)
                self.count_labels[pol_key].append(count_label)

    def clear_results_display(self):
        """Clear results display"""
        for (pol_key, _) in POLARIZATIONS:
            for label in self.count_labels[pol_key]:
                label.configure(text="—")

    def open_histogram_window(self):
        """Open or focus the histogram window."""
        if self.histogram_window is None or not self.histogram_window.winfo_exists():
            self.histogram_window = HistogramWindow(self)
            self.histogram_window.focus()
        else:
            self.histogram_window.focus()
            self.histogram_window.lift()
    
    def connection_status_text(self) -> str:
        """Return a user friendly connection status line."""
        if not self.connected or not self.driver:
            return "● Disconnected"
        driver_kind = "hardware" if isinstance(self.driver, SimpleTimeTaggerHardware) else "simulator"
        return f"● Connected ({driver_kind})"
    
    def log_message(self, message: str, color: str = "white"):
        """Log a message to the GUI."""
        # This could be a status bar at the bottom, for now just prints
        print(f"GUI_LOG: {message}")
        ts = time.strftime("%H:%M:%S")
        tag_name = f"log_{time.time()}"  # Create unique tag for each message
        self.logbox.tag_config(tag_name, foreground=color)
        self.logbox.insert("end", f"[{ts}] {message}\n", tag_name)
        self.logbox.see("end")

    def on_closing(self):
        """Handle window closing event."""
        if self.is_measuring:
            self.stop_measurement()
        if self.timetagger_controller and self.timetagger_controller.is_initialized():
            self.timetagger_controller.shutdown()
        self.thread_pool.shutdown(wait=False)
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
    app = TimeTaggerControllerGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()