from ctypes import *
from dwfconstants import *
import time

dwf = cdll.dwf
hdwf = c_int()
if dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf)) == 0:
    raise RuntimeError("Failed to open device.")

ch = c_int(8)  # DIO0

dwf.FDwfDigitalOutEnableSet(hdwf, ch, c_int(1))
dwf.FDwfDigitalOutDividerSet(hdwf, ch, c_int(1))      # fastest clock
dwf.FDwfDigitalOutCounterInitSet(hdwf, ch, c_int(0))  # start LOW
dwf.FDwfDigitalOutCounterSet(hdwf, ch, c_int(999), c_int(1))  # 1 tick low, 1 tick high

# Key lines to avoid continuous square wave:
try:
    # Per-channel repetition (some SDK versions call this RepetitionSet)
    dwf.FDwfDigitalOutRepetitionSet(hdwf, ch, c_int(1))
except AttributeError:
    pass  # Older runtimes may not have this; global repeat still helps.

dwf.FDwfDigitalOutRepeatSet(hdwf, c_int(1))           # global: run the programmed sequence once
dwf.FDwfDigitalOutIdleSet(hdwf, ch, DwfDigitalOutIdleLow)

dwf.FDwfDigitalOutConfigure(hdwf, c_int(1))           # fire it
time.sleep(0.001)
dwf.FDwfDigitalOutConfigure(hdwf, c_int(0))
dwf.FDwfDeviceCloseAll()