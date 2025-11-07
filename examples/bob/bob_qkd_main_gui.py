"""
Main GUI for Bob's Complete QKD Protocol
Provides a comprehensive interface to configure and run the full QKD protocol
with live updates and final results display.
"""
import sys
import os
import logging
import time
import threading
import functools
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List, Any
from dataclasses import asdict
import customtkinter as ctk
from datetime import datetime

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.bob.bobCPU import BobCPU, BobConfig, BobMode, BobResults

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detector mapping (channel to polarization)
DETECTOR_LABELS = {
    1: "H (0°)",
    2: "V (90°)",
    3: "D (45°)",
    4: "A (135°)"
}

def run_in_background(func):
    """Decorator to run protocol operations in background thread"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        future = self.thread_pool.submit(func, self, *args, **kwargs)
        return future
    return wrapper


class BobQKDGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Bob - Complete QKD Protocol")
        self.geometry("900x800")

        # Thread pool for protocol operations
        self.thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="BobQKD")
        
        # GUI update queue for thread-safe communication
        self.gui_update_queue = Queue()

        # Bob CPU instance
        self.bob_cpu: Optional[BobCPU] = None
        self.is_running = False
        self.protocol_thread = None
        
        # Live view state
        self.live_view_enabled = True
        self.max_detections_display = 5
        self.detection_history: List[Dict] = []  # List of {pulse_id: counts_dict}
        self.current_pulses_received = 0
        self.total_pulses = 0
        
        # Results state
        self.results: Optional[BobResults] = None
        self.post_processing_stage = "idle"  # idle, waiting, processing, done
        
        # GUI Variables
        self.setup_variables()
        
        # Setup GUI
        self.setup_gui()
        self.process_gui_updates()

    def setup_variables(self):
        """Initialize all GUI control variables"""
        # Detection parameters
        self.num_pulses_var = ctk.StringVar(value="200")
        self.pulse_period_var = ctk.StringVar(value="0.4")
        self.measurement_fraction_var = ctk.StringVar(value="0.8")
        
        # Hardware
        self.use_hardware_var = ctk.BooleanVar(value=True)
        self.detector_channels_var = ctk.StringVar(value="1,2,3,4")
        self.dark_count_rate_var = ctk.StringVar(value="1.0")
        self.mode_var = ctk.StringVar(value="continuous")
        
        # Network
        self.use_mock_transmitter_var = ctk.BooleanVar(value=False)
        self.listen_qch_host_var = ctk.StringVar(value="10.127.1.178")
        self.listen_qch_port_var = ctk.StringVar(value="12345")
        self.alice_ip_var = ctk.StringVar(value="10.127.1.178")
        self.alice_port_var = ctk.StringVar(value="65432")
        self.bob_ip_var = ctk.StringVar(value="10.127.1.177")
        self.bob_port_var = ctk.StringVar(value="65433")
        self.shared_key_var = ctk.StringVar(value="IzetXlgAnY4oye56")
        
        # Post-processing
        self.enable_post_processing_var = ctk.BooleanVar(value=True)
        self.test_fraction_var = ctk.StringVar(value="0.25")
        self.error_threshold_var = ctk.StringVar(value="0.6")
        self.pa_compression_var = ctk.StringVar(value="0.5")
        
        # Display
        self.show_live_var = ctk.BooleanVar(value=True)

    def process_gui_updates(self):
        """Process items from the GUI update queue."""
        try:
            while True:
                func = self.gui_update_queue.get_nowait()
                func()
        except Empty:
            pass
        finally:
            self.after(50, self.process_gui_updates)

    def schedule_gui_update(self, func):
        """Schedule a function to be called in the main GUI thread."""
        self.gui_update_queue.put(func)

    def setup_gui(self):
        """Set up the main GUI layout."""
        # Create main scrollable frame
        main_frame = ctk.CTkScrollableFrame(self, width=850, height=750)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Input Area (Top)
        self.setup_input_area(main_frame)
        
        # Control Buttons
        self.setup_control_area(main_frame)
        
        # Live View Section
        self.setup_live_view_area(main_frame)
        
        # Results Section
        self.setup_results_area(main_frame)

    def setup_input_area(self, parent):
        """Setup the configuration input area"""
        input_frame = ctk.CTkFrame(parent)
        input_frame.pack(fill="x", padx=5, pady=5)
        
        title = ctk.CTkLabel(input_frame, text="Configuration", font=("Arial", 16, "bold"))
        title.grid(row=0, column=0, columnspan=4, pady=5, sticky="w")
        
        # Detection Parameters
        row = 1
        ctk.CTkLabel(input_frame, text="Detection Parameters", font=("Arial", 12, "bold")).grid(
            row=row, column=0, columnspan=4, pady=(10, 5), sticky="w")
        row += 1
        
        ctk.CTkLabel(input_frame, text="Expected Pulses:").grid(row=row, column=0, sticky="w", padx=5)
        ctk.CTkEntry(input_frame, textvariable=self.num_pulses_var, width=100).grid(row=row, column=1, padx=5)
        
        ctk.CTkLabel(input_frame, text="Pulse Period (s):").grid(row=row, column=2, sticky="w", padx=5)
        ctk.CTkEntry(input_frame, textvariable=self.pulse_period_var, width=100).grid(row=row, column=3, padx=5)
        row += 1
        
        ctk.CTkLabel(input_frame, text="Measurement Fraction:").grid(row=row, column=0, sticky="w", padx=5)
        ctk.CTkEntry(input_frame, textvariable=self.measurement_fraction_var, width=100).grid(row=row, column=1, padx=5)
        row += 1
        
        # Hardware Parameters
        ctk.CTkLabel(input_frame, text="Hardware", font=("Arial", 12, "bold")).grid(
            row=row, column=0, columnspan=4, pady=(10, 5), sticky="w")
        row += 1
        
        ctk.CTkCheckBox(input_frame, text="Use Hardware", variable=self.use_hardware_var).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5)
        row += 1
        
        ctk.CTkLabel(input_frame, text="Detector Channels:").grid(row=row, column=0, sticky="w", padx=5)
        ctk.CTkEntry(input_frame, textvariable=self.detector_channels_var, width=150).grid(
            row=row, column=1, columnspan=2, padx=5, sticky="w")
        row += 1
        
        ctk.CTkLabel(input_frame, text="Dark Count Rate:").grid(row=row, column=0, sticky="w", padx=5)
        ctk.CTkEntry(input_frame, textvariable=self.dark_count_rate_var, width=100).grid(row=row, column=1, padx=5)
        
        ctk.CTkLabel(input_frame, text="Mode:").grid(row=row, column=2, sticky="w", padx=5)
        mode_menu = ctk.CTkOptionMenu(input_frame, variable=self.mode_var,
                                       values=["continuous", "random_basis"])
        mode_menu.grid(row=row, column=3, padx=5, sticky="w")
        row += 1
        
        # Network
        ctk.CTkLabel(input_frame, text="Network Configuration", font=("Arial", 12, "bold")).grid(
            row=row, column=0, columnspan=4, pady=(10, 5), sticky="w")
        row += 1
        
        ctk.CTkCheckBox(input_frame, text="Use Mock Transmitter", 
                       variable=self.use_mock_transmitter_var).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5)
        row += 1
        
        ctk.CTkLabel(input_frame, text="Listen Host:").grid(row=row, column=0, sticky="w", padx=5)
        ctk.CTkEntry(input_frame, textvariable=self.listen_qch_host_var, width=100).grid(row=row, column=1, padx=5)
        
        ctk.CTkLabel(input_frame, text="Listen Port:").grid(row=row, column=2, sticky="w", padx=5)
        ctk.CTkEntry(input_frame, textvariable=self.listen_qch_port_var, width=100).grid(row=row, column=3, padx=5)
        row += 1
        
        ctk.CTkLabel(input_frame, text="Alice IP:").grid(row=row, column=0, sticky="w", padx=5)
        ctk.CTkEntry(input_frame, textvariable=self.alice_ip_var, width=100).grid(row=row, column=1, padx=5)
        
        ctk.CTkLabel(input_frame, text="Alice Port:").grid(row=row, column=2, sticky="w", padx=5)
        ctk.CTkEntry(input_frame, textvariable=self.alice_port_var, width=100).grid(row=row, column=3, padx=5)
        row += 1
        
        ctk.CTkLabel(input_frame, text="Bob IP:").grid(row=row, column=0, sticky="w", padx=5)
        ctk.CTkEntry(input_frame, textvariable=self.bob_ip_var, width=100).grid(row=row, column=1, padx=5)
        
        ctk.CTkLabel(input_frame, text="Bob Port:").grid(row=row, column=2, sticky="w", padx=5)
        ctk.CTkEntry(input_frame, textvariable=self.bob_port_var, width=100).grid(row=row, column=3, padx=5)
        row += 1
        
        ctk.CTkLabel(input_frame, text="Shared Key:").grid(row=row, column=0, sticky="w", padx=5)
        ctk.CTkEntry(input_frame, textvariable=self.shared_key_var, width=300).grid(
            row=row, column=1, columnspan=3, padx=5, sticky="ew")
        row += 1
        
        # Post-processing
        ctk.CTkLabel(input_frame, text="Post-Processing", font=("Arial", 12, "bold")).grid(
            row=row, column=0, columnspan=4, pady=(10, 5), sticky="w")
        row += 1
        
        ctk.CTkCheckBox(input_frame, text="Enable Post-Processing", 
                       variable=self.enable_post_processing_var).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5)
        row += 1
        
        ctk.CTkLabel(input_frame, text="Test Fraction:").grid(row=row, column=0, sticky="w", padx=5)
        ctk.CTkEntry(input_frame, textvariable=self.test_fraction_var, width=100).grid(row=row, column=1, padx=5)
        
        ctk.CTkLabel(input_frame, text="Error Threshold:").grid(row=row, column=2, sticky="w", padx=5)
        ctk.CTkEntry(input_frame, textvariable=self.error_threshold_var, width=100).grid(row=row, column=3, padx=5)
        row += 1
        
        ctk.CTkLabel(input_frame, text="PA Compression:").grid(row=row, column=0, sticky="w", padx=5)
        ctk.CTkEntry(input_frame, textvariable=self.pa_compression_var, width=100).grid(row=row, column=1, padx=5)

    def setup_control_area(self, parent):
        """Setup control buttons and switches"""
        control_frame = ctk.CTkFrame(parent)
        control_frame.pack(fill="x", padx=5, pady=10)
        
        # Run Protocol Button
        self.run_button = ctk.CTkButton(control_frame, text="Run Protocol", 
                                        command=self.toggle_protocol,
                                        width=200, height=40,
                                        font=("Arial", 14, "bold"))
        self.run_button.pack(side="left", padx=10)
        
        # Live View Switch
        live_switch = ctk.CTkSwitch(control_frame, text="Show Live View",
                                    variable=self.show_live_var,
                                    command=self.on_live_view_toggle)
        live_switch.pack(side="left", padx=20)
        
        # Status Label
        self.status_label = ctk.CTkLabel(control_frame, text="● Ready", 
                                         font=("Arial", 12, "bold"))
        self.status_label.pack(side="right", padx=10)

    def setup_live_view_area(self, parent):
        """Setup live view section"""
        live_frame = ctk.CTkFrame(parent)
        live_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        title = ctk.CTkLabel(live_frame, text="Live View", font=("Arial", 14, "bold"))
        title.pack(pady=5)
        
        # Progress
        progress_frame = ctk.CTkFrame(live_frame)
        progress_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(progress_frame, text="Receiving:").pack(side="left", padx=5)
        self.progress_label = ctk.CTkLabel(progress_frame, text="0 / 0", 
                                          font=("Arial", 12, "bold"))
        self.progress_label.pack(side="left", padx=5)
        
        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=400)
        self.progress_bar.pack(side="left", padx=10)
        self.progress_bar.set(0)
        
        # Detector counts display (last 5 pulses)
        det_frame = ctk.CTkFrame(live_frame)
        det_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(det_frame, text="Recent Detector Counts:", 
                    font=("Arial", 11, "bold")).pack(anchor="w", padx=5)
        
        self.detection_labels = []
        for i in range(self.max_detections_display):
            label = ctk.CTkLabel(det_frame, text="---", font=("Arial", 10))
            label.pack(anchor="w", padx=20, pady=2)
            self.detection_labels.append(label)
        
        # Post-processing status
        pp_frame = ctk.CTkFrame(live_frame)
        pp_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(pp_frame, text="Post-Processing:", 
                    font=("Arial", 11, "bold")).pack(side="left", padx=5)
        self.pp_status_label = ctk.CTkLabel(pp_frame, text="Idle", 
                                           font=("Arial", 11))
        self.pp_status_label.pack(side="left", padx=5)

    def setup_results_area(self, parent):
        """Setup results display section"""
        results_frame = ctk.CTkFrame(parent)
        results_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        title = ctk.CTkLabel(results_frame, text="Results", font=("Arial", 14, "bold"))
        title.pack(pady=5)
        
        # Create scrollable text widget for results
        self.results_text = ctk.CTkTextbox(results_frame, height=300, width=850)
        self.results_text.pack(fill="both", expand=True, padx=10, pady=5)
        self.results_text.configure(state="disabled")

    def toggle_protocol(self):
        """Start or stop the protocol"""
        if not self.is_running:
            self.start_protocol()
        else:
            self.stop_protocol()

    @run_in_background
    def start_protocol(self):
        """Start the complete QKD protocol"""
        def update_ui_starting():
            self.run_button.configure(text="Stop Protocol", fg_color="red")
            self.status_label.configure(text="● Running", text_color="green")
            self.clear_live_view()
            self.clear_results()
            self.post_processing_stage = "idle"
            self.is_running = True
        
        self.schedule_gui_update(update_ui_starting)
        
        try:
            # Create configuration from GUI inputs
            config = self.create_config_from_gui()
            
            # Create Bob CPU
            self.bob_cpu = BobCPU(config)
            
            # Initialize system
            if not self.bob_cpu.initialize_system():
                raise Exception("Failed to initialize Bob system")
            
            self.log_message("Bob system initialized successfully")
            
            # Update total pulses
            self.total_pulses = config.num_expected_pulses
            self.schedule_gui_update(lambda: self.update_progress(0, self.total_pulses))
            
            # Start monitoring thread for live updates
            if self.show_live_var.get():
                monitor_thread = threading.Thread(target=self.monitor_live_updates, daemon=True)
                monitor_thread.start()
            
            # Run complete protocol
            success = self.bob_cpu.run_complete_qkd_protocol()
            
            if success:
                self.log_message("QKD protocol completed successfully!")
                self.results = self.bob_cpu.get_results()
                self.display_final_results()
            else:
                self.log_message("QKD protocol failed", "error")
            
        except Exception as e:
            self.log_message(f"Error: {str(e)}", "error")
            logger.exception("Error in protocol execution")
        finally:
            self.cleanup_protocol()

    def stop_protocol(self):
        """Stop the running protocol"""
        self.is_running = False
        if self.bob_cpu:
            # Bob CPU doesn't have explicit stop, but we set the flag
            self.log_message("Stopping protocol...")
        self.schedule_gui_update(lambda: self.run_button.configure(
            text="Run Protocol", fg_color=["#3B8ED0", "#1F6AA5"]))
        self.schedule_gui_update(lambda: self.status_label.configure(
            text="● Stopped", text_color="orange"))

    def cleanup_protocol(self):
        """Clean up after protocol completion"""
        if self.bob_cpu:
            self.bob_cpu.shutdown_components()
            self.bob_cpu.cleanup_network_resources()
        
        self.is_running = False
        
        def update_ui():
            self.run_button.configure(text="Run Protocol", 
                                     fg_color=["#3B8ED0", "#1F6AA5"])
            self.status_label.configure(text="● Complete", text_color="blue")
        
        self.schedule_gui_update(update_ui)

    def monitor_live_updates(self):
        """Monitor Bob CPU for live updates"""
        last_pulse_count = 0
        reception_started = False
        
        while self.is_running and self.bob_cpu:
            try:
                cpu_running = self.bob_cpu.is_running()
                if cpu_running:
                    reception_started = True
                
                # Get current results
                current_results = self.bob_cpu.get_results()
                current_count = len(current_results.pulse_counts)
                
                # Update if new pulses were detected
                if current_count > last_pulse_count:
                    # Get new detections
                    for i in range(last_pulse_count, current_count):
                        counts = current_results.pulse_counts[i]
                        pulse_id = i + 1
                        
                        self.schedule_gui_update(
                            lambda p=pulse_id, c=counts: 
                            self.add_detection_to_live_view(p, c))
                    
                    last_pulse_count = current_count
                    self.schedule_gui_update(
                        lambda: self.update_progress(current_count, self.total_pulses))
                
                # Check post-processing stage
                if hasattr(self.bob_cpu, 'qkd_impl') and self.bob_cpu.qkd_impl:
                    # Update post-processing status based on Bob's state
                    if current_count >= self.total_pulses:
                        if self.post_processing_stage == "idle":
                            self.post_processing_stage = "waiting"
                            self.schedule_gui_update(
                                lambda: self.pp_status_label.configure(text="Waiting..."))
                        if reception_started and not cpu_running:
                            break
                elif reception_started and not cpu_running and current_count >= self.total_pulses:
                    break
                
                time.sleep(0.1)  # Poll every 100ms
                
            except Exception as e:
                logger.error(f"Error in monitor thread: {e}")
                break

    def add_detection_to_live_view(self, pulse_id: int, counts: Dict[int, int]):
        """Add a detection to the live view"""
        if not self.show_live_var.get():
            return
        
        # Add to history
        self.detection_history.append({"pulse_id": pulse_id, "counts": counts})
        
        # Keep only last N
        if len(self.detection_history) > self.max_detections_display:
            self.detection_history.pop(0)
        
        # Update display
        for i, label in enumerate(self.detection_labels):
            if i < len(self.detection_history):
                detection = self.detection_history[-(i+1)]
                pid = detection["pulse_id"]
                counts_dict = detection["counts"]
                
                # Format counts for all detectors
                counts_str = ", ".join([
                    f"{DETECTOR_LABELS.get(ch, f'Ch{ch}')}: {count}"
                    for ch, count in sorted(counts_dict.items())
                ])
                
                text = f"Pulse #{pid}: [{counts_str}]"
                label.configure(text=text)
            else:
                label.configure(text="---")

    def update_progress(self, current: int, total: int):
        """Update progress display"""
        self.progress_label.configure(text=f"{current} / {total}")
        if total > 0:
            self.progress_bar.set(current / total)

    def clear_live_view(self):
        """Clear the live view displays"""
        self.detection_history = []
        for label in self.detection_labels:
            label.configure(text="---")
        self.progress_label.configure(text="0 / 0")
        self.progress_bar.set(0)
        self.pp_status_label.configure(text="Idle")

    def display_final_results(self):
        """Display the final results"""
        if not self.results:
            return
        
        # Update post-processing status
        self.schedule_gui_update(lambda: self.pp_status_label.configure(text="Processing..."))
        self.post_processing_stage = "processing"
        
        # Wait a moment then show done
        time.sleep(0.5)
        self.schedule_gui_update(lambda: self.pp_status_label.configure(text="Done"))
        self.post_processing_stage = "done"
        
        # Build results text
        results_text = self.build_results_text()
        
        # Display in text widget
        def update_results():
            self.results_text.configure(state="normal")
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", results_text)
            self.results_text.configure(state="disabled")
        
        self.schedule_gui_update(update_results)

    def build_results_text(self) -> str:
        """Build formatted results text"""
        if not self.results:
            return "No results available"
        
        text = "=" * 80 + "\n"
        text += "QKD PROTOCOL RESULTS - BOB\n"
        text += "=" * 80 + "\n\n"
        
        # Detection statistics
        text += "DETECTOR COUNTS:\n"
        text += f"  Total pulses received: {len(self.results.pulse_counts)}\n"
        text += f"  Pulses with counts: {self.results.pulses_with_counts}\n\n"
        
        # Show first few pulse counts
        text += "First 10 pulse detections:\n"
        for i, counts in enumerate(self.results.pulse_counts[:10]):
            counts_str = ", ".join([
                f"{DETECTOR_LABELS.get(ch, f'Ch{ch}')}: {count}"
                for ch, count in sorted(counts.items())
            ])
            text += f"  Pulse #{i+1}: [{counts_str}]\n"
        text += "\n"
        
        # Get post-processing results if available
        if hasattr(self.bob_cpu, 'classical_channel_participant_for_pp') and self.bob_cpu.classical_channel_participant_for_pp:
            qkd_impl = self.bob_cpu.classical_channel_participant_for_pp
            
            text += "BITS RECEIVED:\n"
            if hasattr(qkd_impl.bob_ccc, '_detected_qubits_bytes') and qkd_impl.bob_ccc._detected_qubits_bytes:
                text += f"  Raw key length: {len(qkd_impl.bob_ccc._detected_qubits_bytes)}\n"
                text += f"  First 20 bits: {qkd_impl.bob_ccc._detected_qubits_bytes[:20]}\n\n"
            else:
                text += "  Not available\n\n"
            
            text += "BITS AFTER BASIS SIFT:\n"
            if hasattr(qkd_impl.bob_ccc, 'final_key') and qkd_impl.bob_ccc.final_key:
                text += f"  Sifted key length: {len(qkd_impl.bob_ccc.final_key)}\n"
                text += f"  First 20 sifted bits: {qkd_impl.bob_ccc.final_key[:20]}\n\n"
            else:
                text += "  Not available\n\n"
            
            text += "BITS AFTER POST-BASIS SIFT:\n"
            if hasattr(qkd_impl, 'get_corrected_key') and qkd_impl.get_corrected_key():
                text += f"  After error correction: {qkd_impl.get_corrected_key().get_size()}\n"
                text += f"  First 20 bits: {qkd_impl.get_corrected_key().generate_array()[:20]}\n\n"
            else:
                text += "  Not available\n\n"
        
        # Final key
        text += "FINAL KEY:\n"
        if hasattr(self.bob_cpu, 'classical_channel_participant_for_pp') and self.bob_cpu.classical_channel_participant_for_pp.get_secured_key().any():
            qkd_impl = self.bob_cpu.classical_channel_participant_for_pp
            if hasattr(qkd_impl, 'get_secured_key') and qkd_impl.get_secured_key().any():
                text += f"  Length: {len(qkd_impl.get_secured_key())} bits\n"
                text += f"  Key: {qkd_impl.get_secured_key()}\n\n"
            else:
                text += "  Not generated\n\n"
        else:
            text += "  Not available\n\n"
        
        # QBER
        text += "QUANTUM BIT ERROR RATE (QBER):\n"
        if hasattr(self.bob_cpu, 'classical_channel_participant_for_pp') and self.bob_cpu.classical_channel_participant_for_pp:
            qkd_impl = self.bob_cpu.classical_channel_participant_for_pp
            if hasattr(qkd_impl, 'get_qber'):
                text += f"  QBER: {qkd_impl.get_qber():.4f} ({qkd_impl.get_qber()*100:.2f}%)\n\n"
            else:
                text += "  Not calculated\n\n"
        else:
            text += "  Not available\n\n"
        
        # Timing information
        text += "TIMING INFORMATION:\n"
        text += f"  Total runtime: {self.results.total_runtime_seconds:.2f} seconds\n"
        text += f"  Pulses received: {self.results.pulses_received}\n"
        text += f"  Average detection rate: {self.results.average_detection_rate_hz:.2f} Hz\n\n"
        
        if len(self.results.measurement_times) > 0:
            avg_measurement = sum(self.results.measurement_times) / len(self.results.measurement_times)
            text += f"  Average measurement time: {avg_measurement*1000:.2f} ms\n"
        
        text += "\n" + "=" * 80 + "\n"
        
        return text

    def clear_results(self):
        """Clear the results display"""
        self.results = None
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.configure(state="disabled")

    def create_config_from_gui(self) -> BobConfig:
        """Create BobConfig from GUI inputs"""
        # Parse detector channels
        channels_str = self.detector_channels_var.get()
        detector_channels = [int(ch.strip()) for ch in channels_str.split(",")]
        
        return BobConfig(
            num_expected_pulses=int(self.num_pulses_var.get()),
            pulse_period_seconds=float(self.pulse_period_var.get()),
            measurement_fraction=float(self.measurement_fraction_var.get()),
            use_hardware=self.use_hardware_var.get(),
            detector_channels=detector_channels,
            dark_count_rate=float(self.dark_count_rate_var.get()),
            mode=BobMode(self.mode_var.get()),
            use_mock_transmitter=self.use_mock_transmitter_var.get(),
            listen_qch_host=self.listen_qch_host_var.get(),
            listen_qch_port=int(self.listen_qch_port_var.get()),
            alice_ip=self.alice_ip_var.get(),
            alice_port=int(self.alice_port_var.get()),
            bob_ip=self.bob_ip_var.get(),
            bob_port=int(self.bob_port_var.get()),
            shared_secret_key=self.shared_key_var.get(),
            enable_post_processing=self.enable_post_processing_var.get(),
            test_fraction=float(self.test_fraction_var.get()),
            error_threshold=float(self.error_threshold_var.get()),
            pa_compression_rate=float(self.pa_compression_var.get()),
        )

    def on_live_view_toggle(self):
        """Handle live view toggle"""
        if not self.show_live_var.get():
            self.clear_live_view()

    def log_message(self, message: str, level: str = "info"):
        """Log a message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_text = f"[{timestamp}] {message}"
        
        if level == "error":
            logger.error(message)
        else:
            logger.info(message)
        
        print(log_text)

    def on_closing(self):
        """Handle window closing"""
        self.is_running = False
        if self.bob_cpu:
            try:
                self.bob_cpu.shutdown_components()
                self.bob_cpu.cleanup_network_resources()
            except:
                pass
        self.thread_pool.shutdown(wait=False)
        self.destroy()


if __name__ == "__main__":
    # Set CustomTkinter appearance
    ctk.set_appearance_mode("System")
    
    # Load theme
    THEME = "dark_blue"
    try:
        theme_path = os.path.join(project_root, "examples", "themes", f"{THEME}.json")
        if os.path.exists(theme_path):
            ctk.set_default_color_theme(theme_path)
    except Exception as e:
        logger.warning(f"Could not load theme: {e}")
    
    # Create and run the application
    app = BobQKDGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
