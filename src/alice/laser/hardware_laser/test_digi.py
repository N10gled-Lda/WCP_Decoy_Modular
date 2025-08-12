# single_min_pulse_custom.py
# Generates exactly one minimum-width pulse on DIO0 using a custom pattern.
# Requires Digilent WaveForms installed and dwfconstants.py in the same folder.

from ctypes import *
import time

try:
    from dwfconstants import *
except ImportError:
    raise SystemExit("Put dwfconstants.py from the WaveForms SDK next to this script.")

dwf = cdll.dwf

def chk(ok, what):
    if ok == 0:
        err = c_int()
        dwf.FDwfGetLastError(byref(err))
        raise RuntimeError(f"{what} failed (err={err.value})")

# Open device
hdwf = c_int()
chk(dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf)), "FDwfDeviceOpen")

# 1) Query internal DigitalOut clock to compute one-tick time
hz = c_double()
chk(dwf.FDwfDigitalOutInternalClockInfo(hdwf, byref(hz)), "InternalClockInfo")
f_clk = hz.value
if f_clk <= 0:
    raise RuntimeError("DigitalOut internal clock returned 0 Hz.")

# Channel 0 = DIO0
ch = c_int(8)

# 2) Configure channel for a custom pattern
#    Pattern: 2 bits total: bit0=0, bit1=1  (LSB-first)
#    That yields: 1 tick LOW, then 1 tick HIGH -> single minimum-width pulse.
chk(dwf.FDwfDigitalOutEnableSet(hdwf, ch, c_int(1)), "EnableSet")

freq = 1000
nb_pulses = 2  # only one pulse
divider = c_int(int(f_clk / freq / 2))  # 100 kHz sample rate
t_tick = 1.0 / freq / 2  # 10 ms per tick

print(f"Configuring DIO{ch.value} for {freq} Hz, {nb_pulses} pulse(s) of 1 tick each of {1000/freq}ms...")

chk(dwf.FDwfDigitalOutDividerSet(hdwf, ch, divider), "DividerSet")  # fastest tick
chk(dwf.FDwfDigitalOutTypeSet(hdwf, ch, DwfDigitalOutTypeCustom), "TypeSet")
data = (c_ubyte * 1)(0b00000010)  # LSB-first: 0, then 1
chk(dwf.FDwfDigitalOutDataSet(hdwf, ch, data, c_int(2)), "DataSet")
chk(dwf.FDwfDigitalOutIdleSet(hdwf, ch, DwfDigitalOutIdleLow), "IdleSet")

# 3) Run exactly for the pattern length (2 ticks) and do it once.
chk(dwf.FDwfDigitalOutWaitSet(hdwf, c_double(0.0)), "WaitSet")
chk(dwf.FDwfDigitalOutRunSet(hdwf, c_double(2 * t_tick * nb_pulses)), "RunSet")   # 2 bits -> 2 ticks
chk(dwf.FDwfDigitalOutRepeatSet(hdwf, c_int(1)), "RepeatSet")         # play once



# 4) Fire the output, then stop and close
chk(dwf.FDwfDigitalOutConfigure(hdwf, c_int(1)), "Configure(start)")

# check status until done
sts = c_int()
dwf.FDwfDigitalOutStatus(hdwf, byref(sts))
print(f"Status: {sts.value} (1=armed, 2=done, 3=running, 4=config, 5=prefill, 6=not done, 7=wait, 0 = ready)")
nb_cycles = 0
while True:
    dwf.FDwfDigitalOutStatus(hdwf, byref(sts))
    if sts.value == 2:  # 2 means done
        break
    nb_cycles += 1
    # time.sleep(0.0000001)  # small sleep to prevent busy waiting
print(f"Status: {sts.value} (1=armed, 2=done, 3=running, 4=config, 5=prefill, 6=not done, 7=wait, 0 = ready)")
print(f"Cycles: {nb_cycles}")

# time.sleep(1)

chk(dwf.FDwfDigitalOutConfigure(hdwf, c_int(0)), "Configure(stop)")

dwf.FDwfDeviceCloseAll()

print("Done: one minimum-width pulse generated on DIO0 (0→1, one tick each).")