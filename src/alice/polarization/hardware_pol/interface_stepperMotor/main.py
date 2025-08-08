import customtkinter as ctk
from imports.stm32_interface import STM32Interface
import serial.tools.list_ports

class STM32GUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("STM32 Polarization Sender")
        self.geometry("400x250")
        self.stm = None

        # COM port selection
        self.com_label = ctk.CTkLabel(self, text="Select COM Port:")
        self.com_label.pack(pady=(20, 0))
        self.combobox = ctk.CTkComboBox(self, values=self.get_com_ports())
        self.combobox.pack()

        # Connect button
        self.connect_button = ctk.CTkButton(self, text="Connect", command=self.connect_to_com)
        self.connect_button.pack(pady=(10, 0))

        # Polarization entry
        self.entry_label = ctk.CTkLabel(self, text="Polarization numbers (comma separated):")
        self.entry_label.pack(pady=(20, 0))
        self.entry = ctk.CTkEntry(self)
        self.entry.pack()

        # Send button (disabled until connected)
        self.send_button = ctk.CTkButton(self, text="Send", command=self.send_polarizations, state="disabled")
        self.send_button.pack(pady=20)

        # Status label
        self.status_label = ctk.CTkLabel(self, text="")
        self.status_label.pack()

    def get_com_ports(self):
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]

    def connect_to_com(self):
        com_port = self.combobox.get()
        if not com_port:
            self.status_label.configure(text="Please select a COM port.", text_color="red")
            return
        try:
            self.stm = STM32Interface(com_port)
            self.stm.start()
            self.stm.connect()
            self.status_label.configure(text=f"Connected to {com_port}", text_color="green")
            self.send_button.configure(state="normal")
        except Exception as e:
            self.status_label.configure(text=f"Error: {e}", text_color="red")
            self.send_button.configure(state="disabled")

    def send_polarizations(self):
        numbers_str = self.entry.get()
        try:
            numbers = [int(x.strip()) for x in numbers_str.split(",") if x.strip() != ""]
            if not numbers:
                self.status_label.configure(text="Enter at least one number.", text_color="red")
                return
            if not all(0 <= n <= 3 for n in numbers):
                self.status_label.configure(text="Numbers must be 0, 1, 2, or 3.", text_color="red")
                return
        except ValueError:
            self.status_label.configure(text="Invalid input. Use comma-separated numbers.", text_color="red")
            return

        if self.stm is None:
            self.status_label.configure(text="Not connected to any COM port.", text_color="red")
            return

        success = self.stm.send_polarization_numbers(numbers)
        if success:
            self.status_label.configure(text="Polarization numbers sent!", text_color="green")
        else:
            self.status_label.configure(text="Failed to send numbers.", text_color="red")

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = STM32GUI()
    app.mainloop()