import customtkinter as ctk
import serial.tools.list_ports
from imports.stm32_interface import STM32Interface

class STM32GUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("STM32 Connect")
        self.geometry("800x500")
        self.stm = None
        self.com_port = None
        self.available = False

        # --- COM Port & Connect Section ---
        com_section = ctk.CTkFrame(self)
        com_section.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(com_section, text="COM Port & Connect", font=("Arial", 16, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=5, columnspan=3)
        ctk.CTkLabel(com_section, text="Select COM Port:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.combobox = ctk.CTkComboBox(com_section, values=self.get_com_ports(), width=180)
        self.combobox.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.set_port_button = ctk.CTkButton(com_section, text="Set Port", command=self.set_com_port, fg_color="blue", width=120)
        self.set_port_button.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        self.connect_button = ctk.CTkButton(com_section, text="Send Connect", command=self.send_connect, fg_color="red", width=120)
        self.connect_button.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        # --- Polarization Section ---
        pol_section = ctk.CTkFrame(self)
        pol_section.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(pol_section, text="Polarization", font=("Arial", 16, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=5, columnspan=3)
        ctk.CTkLabel(pol_section, text="Polarizations (0,1,2,3,...) max 95:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.pol_entry = ctk.CTkEntry(pol_section, width=180)
        self.pol_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.pol_button = ctk.CTkButton(pol_section, text="Send Polarizations", command=self.send_polarizations, fg_color="purple", state="disabled", width=120)
        self.pol_button.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        legend = "Device: 1 = Linear Polarizer, 2 = Half Wave Plate"
        ctk.CTkLabel(pol_section, text=legend).grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.poldev_entry = ctk.CTkEntry(pol_section, width=60)
        self.poldev_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.poldev_button = ctk.CTkButton(pol_section, text="Set Polarizer Device", command=self.send_polarization_device, fg_color="orange", state="disabled", width=120)
        self.poldev_button.grid(row=2, column=2, padx=10, pady=5, sticky="w")

        # --- Angle Section ---
        angle_section = ctk.CTkFrame(self)
        angle_section.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(angle_section, text="Angle", font=("Arial", 16, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=5, columnspan=4)
        ctk.CTkLabel(angle_section, text="Angle (0-360):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.angle_entry = ctk.CTkEntry(angle_section, width=80)
        self.angle_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.offset_switch = ctk.CTkSwitch(angle_section, text="Set as Offset")
        self.offset_switch.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        self.angle_button = ctk.CTkButton(angle_section, text="Send Angle", command=self.send_angle, fg_color="teal", state="disabled", width=120)
        self.angle_button.grid(row=1, column=3, padx=10, pady=5, sticky="w")

        # --- Frequency Section ---
        freq_section = ctk.CTkFrame(self)
        freq_section.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(freq_section, text="Frequency", font=("Arial", 16, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=5, columnspan=3)
        ctk.CTkLabel(freq_section, text="Stepper Motor Frequency (1-1000 Hz):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.freq_entry = ctk.CTkEntry(freq_section, width=80)
        self.freq_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.freq_button = ctk.CTkButton(freq_section, text="Set Stepper Frequency", command=self.send_stepper_frequency, fg_color="blue", state="disabled", width=160)
        self.freq_button.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        ctk.CTkLabel(freq_section, text="Operation Period (1-60000 ms):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.period_entry = ctk.CTkEntry(freq_section, width=80)
        self.period_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.period_button = ctk.CTkButton(freq_section, text="Set Operation Period", command=self.send_operation_period, fg_color="green", state="disabled", width=160)
        self.period_button.grid(row=2, column=2, padx=10, pady=5, sticky="w")

    def get_com_ports(self):
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]

    def set_com_port(self):
        self.com_port = self.combobox.get()
        if self.com_port:
            if self.stm is not None and self.stm.running:
                self.stm.running = False  # Stop previous threads if any
            self.stm = STM32Interface(port=self.com_port)
            self.stm.on_connected = self.on_connected
            self.stm.on_available = self.on_available
            self.stm.start()
            print(f"COM port set to: {self.com_port}")
            self.set_port_button.configure(state="disabled")
            self.combobox.configure(state="disabled")

    def on_connected(self):
        self.connect_button.configure(fg_color="green", state="disabled")
        print("Connected! Connect button is now green and disabled.")
        self.available = True
        self.pol_button.configure(state="normal")
        self.poldev_button.configure(state="normal")
        self.angle_button.configure(state="normal")
        self.freq_button.configure(state="normal")
        self.period_button.configure(state="normal")

    def on_available(self):
        if self.stm.available:
            self.pol_button.configure(state="normal")
            self.poldev_button.configure(state="normal")
            self.angle_button.configure(state="normal")
            self.freq_button.configure(state="normal")
            self.period_button.configure(state="normal")
            print("All command buttons enabled (available=True).")
        else:
            self.pol_button.configure(state="disabled")
            self.poldev_button.configure(state="disabled")
            self.angle_button.configure(state="disabled")
            self.freq_button.configure(state="disabled")
            self.period_button.configure(state="disabled")
            print("All command buttons disabled (available=False).")

    def send_connect(self):
        if not self.stm or not self.stm.running:
            print("Please set and start a COM port first.")
            return
        self.stm.connect()
        print("Connect command sent.")

    def send_polarizations(self):
        if not self.available:
            print("Polarization command not available.")
            return
        entry = self.pol_entry.get()
        try:
            nums = [int(x.strip()) for x in entry.split(",") if x.strip() != ""]
        except ValueError:
            print("Invalid input: Only numbers 0,1,2,3 allowed, separated by commas.")
            return
        if any(n not in [0,1,2,3] for n in nums):
            print("Invalid polarization: Only 0, 1, 2, or 3 allowed.")
            return
        if len(nums) > 95:
            print("Too many polarizations! Max is 95.")
            return
        print(f"Sending polarizations: {nums}")
        self.stm.send_cmd_polarization_numbers(nums)

    def send_polarization_device(self):
        if not self.available:
            print("Polarization device command not available.")
            return
        entry = self.poldev_entry.get()
        try:
            device = int(entry.strip())
        except ValueError:
            print("Invalid input: Device must be 1 or 2.")
            return
        if device not in [1, 2]:
            print("Invalid device: Only 1 (Linear Polarizer) or 2 (Half Wave Plate) allowed.")
            return
        print(f"Sending polarization device: {device}")
        self.stm.send_cmd_polarization_device(device)

    def send_angle(self):
        if not self.available:
            print("Angle command not available.")
            return
        entry = self.angle_entry.get()
        try:
            value = int(entry.strip())
        except ValueError:
            print("Invalid input: Angle must be an integer between 0 and 360.")
            return
        if not (0 <= value <= 360):
            print("Invalid angle: Must be between 0 and 360.")
            return
        is_offset = self.offset_switch.get()
        print(f"Sending angle: {value}, as {'offset' if is_offset else 'angle'}")
        self.stm.send_cmd_set_angle(value, is_offset=is_offset)

    def send_stepper_frequency(self):
        if not self.available:
            print("Stepper frequency command not available.")
            return
        entry = self.freq_entry.get()
        try:
            freq = int(entry.strip())
        except ValueError:
            print("Invalid input: Frequency must be an integer between 1 and 1000 Hz.")
            return
        if not isinstance(freq, int) or not (1 <= freq <= 1000):
            print("Invalid stepper motor frequency. Must be an integer between 1 and 1000 Hz.")
            return
        print(f"Sending stepper motor frequency: {freq} Hz")
        self.stm.send_cmd_set_frequency(freq, is_stepper=True)

    def send_operation_period(self):
        if not self.available:
            print("Operation period command not available.")
            return
        entry = self.period_entry.get()
        try:
            period = int(entry.strip())
        except ValueError:
            print("Invalid input: Period must be an integer between 1 and 60000 ms.")
            return
        if not (1 <= period <= 60000):
            print("Invalid period: Must be between 1 and 60000 ms.")
            return
        print(f"Sending operation period: {period} ms")
        self.stm.send_cmd_set_frequency(period, is_stepper=False)

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = STM32GUI()
    app.mainloop()