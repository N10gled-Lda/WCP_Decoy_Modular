#!/usr/bin/env python3
"""
Bob TimeTagger GUI

Purpose
  - Scan and connect to a Swabian TimeTagger (fallback to simulator)
  - Map detector channels to polarizations (H, V, D, A)
  - Set bin duration and number of displayed bins
  - Run continuous or finite-repeat measurements
  - Display the most recent counts per polarization in a scrollable grid

Notes
  - Interacts only with the TimeTagger via SimpleTimeTaggerController
  - Does not run the full Bob protocol
"""

import os
import sys
import threading
import time
import logging
from collections import deque
from typing import Dict, List, Optional

try:
	import tkinter as tk
	from tkinter import ttk, messagebox
except Exception:  # pragma: no cover
	raise


# Add project root to path so we can import from src
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from src.bob.timetagger.simple_timetagger_controller import (
	SimpleTimeTaggerController,
	SimpleTimeTaggerHardware,
	SimpleTimeTaggerSimulator,
)


logger = logging.getLogger("BobTimeTaggerGUI")
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


POLARIZATIONS = ["H (0°)", "V (90°)", "D (45°)", "A (135°)"]
POL_KEYS = ["H", "V", "D", "A"]


class ScrollableFrame(ttk.Frame):
	"""A simple scrollable frame for the counts grid."""

	def __init__(self, container, *args, **kwargs):
		super().__init__(container, *args, **kwargs)
		self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, height=260)
		self.vscroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
		self.inner = ttk.Frame(self.canvas)

		self.inner.bind(
			"<Configure>",
			lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
		)
		self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
		self.canvas.configure(yscrollcommand=self.vscroll.set)

		self.canvas.grid(row=0, column=0, sticky="nsew")
		self.vscroll.grid(row=0, column=1, sticky="ns")
		self.grid_rowconfigure(0, weight=1)
		self.grid_columnconfigure(0, weight=1)


class TimeTaggerGUI(tk.Tk):
	def __init__(self) -> None:
		super().__init__()
		self.title("Bob TimeTagger Measurement GUI")
		self.geometry("1024x720")

		# Measurement/controller state
		self.controller: Optional[SimpleTimeTaggerController] = None
		self.driver_type: Optional[str] = None  # "hardware" or "simulator"
		self.connected: bool = False

		# UI state
		self.available_devices: List[str] = []  # e.g., ["Swabian TimeTagger"], always include "Simulator"
		self.selected_device = tk.StringVar(value="Simulator")
		self.status_text = tk.StringVar(value="Disconnected")

		# Channel mapping H/V/D/A -> timetagger channel number
		self.channel_vars: Dict[str, tk.IntVar] = {
			"H": tk.IntVar(value=1),
			"V": tk.IntVar(value=2),
			"D": tk.IntVar(value=3),
			"A": tk.IntVar(value=4),
		}

		# Measurement config
		self.bin_ms = tk.DoubleVar(value=1000.0)  # ms per bin
		self.num_bins_display = tk.IntVar(value=5)  # rows
		self.mode = tk.StringVar(value="continuous")  # "continuous" or "finite"
		self.finite_repeats = tk.IntVar(value=20)  # used when mode == finite

		# Live data buffers (per polarization)
		self.buffers: Dict[str, deque] = {k: deque(maxlen=self.num_bins_display.get()) for k in POL_KEYS}

		# Grid label widgets [row][col]
		self.grid_labels: List[List[ttk.Label]] = []

		# Measurement thread handling
		self._run_thread: Optional[threading.Thread] = None
		self._running = threading.Event()

		self._build_gui()
		self._update_controls_state()

		# Attempt an initial scan
		self.after(200, self.scan_devices)

		# Proper close handling
		self.protocol("WM_DELETE_WINDOW", self.on_close)

	# ----------------------------
	# GUI Layout
	# ----------------------------
	def _build_gui(self) -> None:
		root = self

		# Top: connection area
		connection = ttk.LabelFrame(root, text="Connection")
		connection.pack(fill="x", padx=10, pady=8)

		ttk.Label(connection, text="Device:").grid(row=0, column=0, padx=6, pady=6, sticky="w")
		self.device_combo = ttk.Combobox(connection, textvariable=self.selected_device, state="readonly", width=28)
		self.device_combo.grid(row=0, column=1, padx=6, pady=6, sticky="w")
		ttk.Button(connection, text="Scan", command=self.scan_devices).grid(row=0, column=2, padx=6, pady=6)
		self.connect_btn = ttk.Button(connection, text="Connect", command=self.toggle_connection)
		self.connect_btn.grid(row=0, column=3, padx=6, pady=6)
		ttk.Label(connection, text="Status:").grid(row=0, column=4, padx=(18, 6), pady=6, sticky="e")
		ttk.Label(connection, textvariable=self.status_text, width=18).grid(row=0, column=5, padx=6, pady=6, sticky="w")

		# Channel mapping area
		channels = ttk.LabelFrame(root, text="Channels → Polarizations")
		channels.pack(fill="x", padx=10, pady=8)

		for i, (key, label) in enumerate(zip(POL_KEYS, POLARIZATIONS)):
			ttk.Label(channels, text=f"{label}").grid(row=0, column=i, padx=8, pady=4)
			spin = tk.Spinbox(channels, from_=1, to=8, width=6, textvariable=self.channel_vars[key])
			spin.grid(row=1, column=i, padx=8, pady=4)

		# Bin settings
		bins = ttk.LabelFrame(root, text="Measurement Bins")
		bins.pack(fill="x", padx=10, pady=8)

		ttk.Label(bins, text="Time per bin (ms):").grid(row=0, column=0, padx=6, pady=6, sticky="e")
		tk.Entry(bins, textvariable=self.bin_ms, width=10).grid(row=0, column=1, padx=6, pady=6, sticky="w")
		ttk.Label(bins, text="Nb bins (displayed):").grid(row=0, column=2, padx=18, pady=6, sticky="e")
		tk.Entry(bins, textvariable=self.num_bins_display, width=6).grid(row=0, column=3, padx=6, pady=6, sticky="w")
		ttk.Button(bins, text="Apply", command=self._apply_bins_config).grid(row=0, column=4, padx=10, pady=6)

		# Mode selection
		mode = ttk.LabelFrame(root, text="Mode")
		mode.pack(fill="x", padx=10, pady=8)

		tk.Radiobutton(mode, text="Continuous", variable=self.mode, value="continuous", command=self._update_controls_state).grid(
			row=0, column=0, padx=6, pady=6, sticky="w"
		)
		tk.Radiobutton(mode, text="Finite repeat", variable=self.mode, value="finite", command=self._update_controls_state).grid(
			row=0, column=1, padx=6, pady=6, sticky="w"
		)
		ttk.Label(mode, text="Repeats:").grid(row=0, column=2, padx=(20, 6), pady=6, sticky="e")
		self.repeats_entry = tk.Entry(mode, textvariable=self.finite_repeats, width=8)
		self.repeats_entry.grid(row=0, column=3, padx=6, pady=6, sticky="w")

		# Start/Stop
		actions = ttk.LabelFrame(root, text="Measure")
		actions.pack(fill="x", padx=10, pady=8)
		self.start_btn = ttk.Button(actions, text="Start", command=self.start_measurement)
		self.start_btn.grid(row=0, column=0, padx=8, pady=6)
		self.stop_btn = ttk.Button(actions, text="Stop", command=self.stop_measurement, state="disabled")
		self.stop_btn.grid(row=0, column=1, padx=8, pady=6)

		# Results grid header
		results_box = ttk.LabelFrame(root, text="Results (latest at top)")
		results_box.pack(fill="both", expand=True, padx=10, pady=8)

		header = ttk.Frame(results_box)
		header.pack(fill="x")
		ttk.Label(header, text="Bin #", width=8, anchor="center").grid(row=0, column=0, padx=4, pady=4)
		for i, label in enumerate(POLARIZATIONS, start=1):
			ttk.Label(header, text=label, width=14, anchor="center").grid(row=0, column=i, padx=4, pady=4)

		# Scrollable grid
		self.scroll = ScrollableFrame(results_box)
		self.scroll.pack(fill="both", expand=True)

		# Build initial grid
		self._rebuild_grid()

	# ----------------------------
	# Device handling
	# ----------------------------
	def scan_devices(self) -> None:
		"""Scan for available devices. Always list Simulator. Add hardware if detected."""
		logger.info("Scanning for TimeTagger devices...")

		found = ["Simulator"]

		# Try to bring up a hardware driver and initialize; if it works, list it
		try:
			channels = self._current_mapped_channels()
			hw = SimpleTimeTaggerHardware(detector_channels=channels)
			try:
				if hw.initialize():
					found.append("Swabian TimeTagger")
					hw.shutdown()
					logger.info("Hardware TimeTagger detected")
				else:
					logger.info("Hardware driver present but not initialized")
			except Exception as e:
				logger.info(f"Hardware not available: {e}")
			finally:
				# Ensure hardware shutdown in case initialize partially succeeded
				try:
					hw.shutdown()
				except Exception:
					pass
		except Exception as e:
			logger.info(f"Hardware driver not importable or failed to construct: {e}")

		self.available_devices = found
		self.device_combo.configure(values=self.available_devices)
		if self.selected_device.get() not in self.available_devices:
			self.selected_device.set(self.available_devices[0])

		self.status_text.set("Scan complete")

	def toggle_connection(self) -> None:
		if not self.connected:
			self._connect()
		else:
			self._disconnect()

	def _connect(self) -> None:
		if self.connected:
			return

		# Build driver from selection
		channels = self._current_mapped_channels()

		device = self.selected_device.get()
		driver = None
		self.driver_type = None
		if device == "Swabian TimeTagger":
			try:
				driver = SimpleTimeTaggerHardware(detector_channels=channels)
				self.driver_type = "hardware"
			except Exception as e:
				messagebox.showerror("Connection Error", f"Failed to create hardware driver: {e}")
				return
		else:
			# Simulator defaults; tuned for clarity
			driver = SimpleTimeTaggerSimulator(
				detector_channels=channels,
				dark_count_rate=20.0,
				signal_count_rate=120.0,
				signal_probability=0.08,
			)
			self.driver_type = "simulator"

		self.controller = SimpleTimeTaggerController(driver)
		ok = self.controller.initialize()
		if not ok:
			self.controller = None
			self.driver_type = None
			messagebox.showerror("Connection Error", "Failed to initialize TimeTagger.")
			return

		self.connected = True
		self.status_text.set(f"Connected ({self.driver_type})")
		self._update_controls_state()

	def _disconnect(self) -> None:
		self.stop_measurement()
		if self.controller:
			try:
				self.controller.shutdown()
			except Exception:
				pass
		self.controller = None
		self.driver_type = None
		self.connected = False
		self.status_text.set("Disconnected")
		self._update_controls_state()

	# ----------------------------
	# Measurement
	# ----------------------------
	def start_measurement(self) -> None:
		if not self.connected or not self.controller:
			messagebox.showwarning("Not connected", "Please connect to a device first.")
			return

		# Validate and apply measurement duration
		try:
			bin_ms = float(self.bin_ms.get())
			if bin_ms <= 0:
				raise ValueError
		except Exception:
			messagebox.showerror("Invalid value", "Time per bin must be a positive number (ms).")
			return

		duration_s = bin_ms / 1000.0
		if not self.controller.set_measurement_duration(duration_s):
			messagebox.showerror("Configuration Error", "Failed to set measurement duration on device.")
			return

		# Clear buffers and grid if starting fresh
		self._clear_buffers()
		self._refresh_grid_values()

		# Start background thread
		self._running.set()
		self._run_thread = threading.Thread(target=self._measure_loop, daemon=True)
		self._run_thread.start()
		self.status_text.set("Measuring…")
		self._update_controls_state()

	def stop_measurement(self) -> None:
		if self._running.is_set():
			self._running.clear()
			# Wait briefly for thread to finish
			t = self._run_thread
			if t and t.is_alive():
				try:
					t.join(timeout=1.5)
				except Exception:
					pass
			self._run_thread = None
			if self.connected:
				self.status_text.set(f"Connected ({self.driver_type})")
			else:
				self.status_text.set("Disconnected")
			self._update_controls_state()

	def _measure_loop(self) -> None:
		"""Run measurement cycles according to the chosen mode and update UI."""
		repeats_target = None
		if self.mode.get() == "finite":
			try:
				repeats_target = int(self.finite_repeats.get())
				if repeats_target <= 0:
					repeats_target = 1
			except Exception:
				repeats_target = 1

		repeats_done = 0
		while self._running.is_set():
			# Measure once for the configured duration
			if not self.controller:
				break
			try:
				counts = self.controller.measure_counts()  # Dict[channel->count]
			except Exception as e:
				logger.error(f"Measurement error: {e}")
				break

			# Map to polarizations
			pol_counts = self._map_counts_to_pols(counts)

			# Update buffers and UI
			self._push_counts(pol_counts)
			self.after(0, self._refresh_grid_values)

			repeats_done += 1
			if repeats_target is not None and repeats_done >= repeats_target:
				# Auto-stop in finite mode
				self.after(0, self.stop_measurement)
				break

			# Sleep a tiny moment to allow UI refresh if the hardware is very fast
			# The controller blocks for the bin duration, so this is mainly for simulator
			time.sleep(0.01)

	# ----------------------------
	# Helpers: grid and buffers
	# ----------------------------
	def _apply_bins_config(self) -> None:
		"""Apply changes in displayed number of bins and rebuild grid/buffers."""
		try:
			n = int(self.num_bins_display.get())
			if n <= 0:
				raise ValueError
		except Exception:
			messagebox.showerror("Invalid value", "Nb bins must be a positive integer.")
			return

		# Recreate buffers and grid
		for k in POL_KEYS:
			self.buffers[k] = deque(list(self.buffers[k])[-n:], maxlen=n)
		self._rebuild_grid()
		self._refresh_grid_values()

	def _rebuild_grid(self) -> None:
		# Clear existing inner frame
		for child in self.scroll.inner.winfo_children():
			child.destroy()
		self.grid_labels.clear()

		# Build N rows, 4 columns + bin index column
		n = self.num_bins_display.get()
		for r in range(n):
			# Show bin index where 0 is latest
			idx_lbl = ttk.Label(self.scroll.inner, text=f"{r}", width=8, anchor="e")
			idx_lbl.grid(row=r, column=0, padx=4, pady=2, sticky="e")
			row_labels: List[ttk.Label] = []
			for c in range(4):
				lbl = ttk.Label(self.scroll.inner, text="", width=14, anchor="center", relief="groove")
				lbl.grid(row=r, column=c + 1, padx=2, pady=2, sticky="nsew")
				row_labels.append(lbl)
			self.grid_labels.append(row_labels)

		# Expand columns
		for col in range(5):
			self.scroll.inner.grid_columnconfigure(col, weight=1)

	def _refresh_grid_values(self) -> None:
		"""Refresh cell texts based on buffers. Row 0 = latest values."""
		n = self.num_bins_display.get()

		# Build a matrix rows x 4 ordered by latest at top
		latest_lists = []
		for key in POL_KEYS:
			# deque oldest->newest, we want newest first
			buf = list(self.buffers[key])
			ordered = list(reversed(buf))  # newest-first
			latest_lists.append(ordered)

		# For each row, read value or blank
		for r in range(n):
			for c, key in enumerate(POL_KEYS):
				try:
					val = latest_lists[c][r]
				except IndexError:
					val = ""
				self.grid_labels[r][c].configure(text=str(val))

	def _push_counts(self, pol_counts: Dict[str, int]) -> None:
		for k in POL_KEYS:
			self.buffers[k].append(pol_counts.get(k, 0))

	def _clear_buffers(self) -> None:
		n = self.num_bins_display.get()
		for k in POL_KEYS:
			self.buffers[k] = deque(maxlen=n)

	def _map_counts_to_pols(self, counts: Dict[int, int]) -> Dict[str, int]:
		mapping = {
			"H": self.channel_vars["H"].get(),
			"V": self.channel_vars["V"].get(),
			"D": self.channel_vars["D"].get(),
			"A": self.channel_vars["A"].get(),
		}
		result = {}
		for k, ch in mapping.items():
			result[k] = counts.get(int(ch), 0)
		return result

	def _current_mapped_channels(self) -> List[int]:
		# Unique channel list gathered from mapping inputs
		channels = [
			int(self.channel_vars["H"].get()),
			int(self.channel_vars["V"].get()),
			int(self.channel_vars["D"].get()),
			int(self.channel_vars["A"].get()),
		]
		# preserve order but unique
		seen = set()
		ordered_unique = []
		for ch in channels:
			if ch not in seen:
				ordered_unique.append(ch)
				seen.add(ch)
		return ordered_unique

	# ----------------------------
	# UI state and lifecycle
	# ----------------------------
	def _update_controls_state(self) -> None:
		# Connection buttons
		self.connect_btn.configure(text="Disconnect" if self.connected else "Connect")
		self.device_combo.configure(state="readonly" if not self.connected else "disabled")

		# Channel and config fields disabled while running
		running = self._running.is_set()
		state = "disabled" if running else "normal"

		for key in POL_KEYS:
			# Find the Spinbox widgets in the channels frame
			# We can enable/disable by traversing children of the parent labelframe
			pass

		# Simpler: disable entire window sections by toggling start/stop
		self.start_btn.configure(state=("disabled" if running or not self.connected else "normal"))
		self.stop_btn.configure(state=("normal" if running else "disabled"))

		# Mode controls
		finite = self.mode.get() == "finite"
		self.repeats_entry.configure(state=("normal" if finite and not running else "disabled"))

	def on_close(self) -> None:
		try:
			self.stop_measurement()
		except Exception:
			pass
		try:
			self._disconnect()
		except Exception:
			pass
		self.destroy()


def main() -> int:
	app = TimeTaggerGUI()
	app.mainloop()
	return 0


if __name__ == "__main__":
	sys.exit(main())

