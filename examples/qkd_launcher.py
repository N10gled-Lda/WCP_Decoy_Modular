"""Launcher for Alice/Bob QKD GUIs without opening extra consoles."""
import sys
import os
import subprocess
import customtkinter as ctk
from pathlib import Path

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class QKDLauncher(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("QKD Protocol Launcher")
        self.geometry("400x300")
        
        # Store process handles
        self.alice_process = None
        self.bob_process = None
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the launcher GUI"""
        # Title
        title = ctk.CTkLabel(self, text="QKD Protocol Launcher", 
                            font=("Arial", 20, "bold"))
        title.pack(pady=20)
        
        # Description
        desc = ctk.CTkLabel(self, 
                           text="Launch Alice and Bob QKD Protocol GUIs\n" +
                                "Tip: Start Bob first, then Alice",
                           font=("Arial", 12))
        desc.pack(pady=10)
        
        # Buttons frame
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(pady=20)
        
        # Alice button
        self.alice_button = ctk.CTkButton(button_frame, 
                                         text="Launch Alice GUI",
                                         command=self.launch_alice,
                                         width=150,
                                         height=40)
        self.alice_button.grid(row=0, column=0, padx=10, pady=10)
        
        # Bob button
        self.bob_button = ctk.CTkButton(button_frame, 
                                       text="Launch Bob GUI",
                                       command=self.launch_bob,
                                       width=150,
                                       height=40)
        self.bob_button.grid(row=0, column=1, padx=10, pady=10)
        
        # Both button
        both_button = ctk.CTkButton(self, 
                                    text="Launch Both (Bob first, then Alice)",
                                    command=self.launch_both,
                                    width=300,
                                    height=40,
                                    fg_color="green",
                                    hover_color="darkgreen")
        both_button.pack(pady=10)
        
        # Status
        self.status_label = ctk.CTkLabel(self, text="Ready to launch", 
                                        font=("Arial", 11))
        self.status_label.pack(pady=10)
        
        # Close all button
        close_button = ctk.CTkButton(self, 
                                    text="Close All Windows",
                                    command=self.close_all,
                                    width=200,
                                    fg_color="red",
                                    hover_color="darkred")
        close_button.pack(pady=10)
    
    def _launch_script(self, script_path: Path):
        """Helper to launch a script in a new process with visible console."""
        # Use regular python.exe so output is visible in a console window
        if sys.platform == "win32":
            # CREATE_NEW_CONSOLE creates a new console window for the process
            return subprocess.Popen(
                [sys.executable, str(script_path)],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            # On Unix-like systems, just run in background
            return subprocess.Popen([sys.executable, str(script_path)])

    def _monitor_process(self, attr_name: str, button: ctk.CTkButton, label_name: str):
        """Monitor launched process and re-enable button when it exits."""
        process = getattr(self, attr_name)
        if not process:
            return
        if process.poll() is None:
            self.after(500, lambda: self._monitor_process(attr_name, button, label_name))
        else:
            setattr(self, attr_name, None)
            button.configure(fg_color=["#3B8ED0", "#1F6AA5"], state="normal")
            self.status_label.configure(text=f"{label_name} GUI closed")

    def launch_alice(self):
        """Launch Alice's GUI"""
        try:
            if self.alice_process and self.alice_process.poll() is None:
                self.status_label.configure(text="Alice GUI already running!")
                return
            
            alice_script = Path(project_root) / "examples" / "alice" / "alice_qkd_main_gui.py"
            if not alice_script.exists():
                self.status_label.configure(text="Error: Alice GUI script not found!")
                return

            self.alice_process = self._launch_script(alice_script)
            
            self.status_label.configure(text="Alice GUI launched!")
            self.alice_button.configure(fg_color="gray", state="disabled")
            self._monitor_process("alice_process", self.alice_button, "Alice")
            
        except Exception as e:
            self.status_label.configure(text=f"Error launching Alice: {str(e)}")
    
    def launch_bob(self):
        """Launch Bob's GUI"""
        try:
            if self.bob_process and self.bob_process.poll() is None:
                self.status_label.configure(text="Bob GUI already running!")
                return
            
            bob_script = Path(project_root) / "examples" / "bob" / "bob_qkd_main_gui.py"
            if not bob_script.exists():
                self.status_label.configure(text="Error: Bob GUI script not found!")
                return

            self.bob_process = self._launch_script(bob_script)
            
            self.status_label.configure(text="Bob GUI launched!")
            self.bob_button.configure(fg_color="gray", state="disabled")
            self._monitor_process("bob_process", self.bob_button, "Bob")
            
        except Exception as e:
            self.status_label.configure(text=f"Error launching Bob: {str(e)}")
    
    def launch_both(self):
        """Launch both GUIs (Bob first)"""
        self.launch_bob()
        # Small delay to ensure Bob starts first
        self.after(1000, self.launch_alice)
    
    def close_all(self):
        """Close all launched windows"""
        closed = []
        
        if self.alice_process and self.alice_process.poll() is None:
            self.alice_process.terminate()
            closed.append("Alice")
            self.alice_button.configure(fg_color=["#3B8ED0", "#1F6AA5"], state="normal")
            self.alice_process = None

        if self.bob_process and self.bob_process.poll() is None:
            self.bob_process.terminate()
            closed.append("Bob")
            self.bob_button.configure(fg_color=["#3B8ED0", "#1F6AA5"], state="normal")
            self.bob_process = None
        
        if closed:
            self.status_label.configure(text=f"Closed: {', '.join(closed)}")
        else:
            self.status_label.configure(text="No active windows to close")
    
    def on_closing(self):
        """Handle window closing"""
        self.close_all()
        self.destroy()


if __name__ == "__main__":
    # Set CustomTkinter appearance
    ctk.set_appearance_mode("System")
    
    # Try to load theme
    THEME = "dark_blue"
    try:
        theme_path = os.path.join(project_root, "examples", "themes", f"{THEME}.json")
        if os.path.exists(theme_path):
            ctk.set_default_color_theme(theme_path)
    except Exception:
        pass
    
    # Create and run the launcher
    app = QKDLauncher()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
