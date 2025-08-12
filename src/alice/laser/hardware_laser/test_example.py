"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2023-01-09

   Requires:                       
       Python 2.7, 3
   Generate pulses on trigger
"""

from ctypes import *
import time
from dwfconstants import *
import math
import sys

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

hdwf = c_int()
sts = c_byte()

dwf.FDwfParamSet(DwfParamOnClose, c_int(0)) # 0 = run, 1 = stop, 2 = shutdown

version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: "+str(version.value))

print("Opening first device")
dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

if hdwf.value == 0:
    print("failed to open device")
    szerr = create_string_buffer(512)
    dwf.FDwfGetLastErrorMsg(szerr)
    print(str(szerr.value))
    quit()

# the device will only be configured when FDwf###Configure is called
dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(0)) 

iChannel = 8
hzFreq = 1e3 # freq Hz
prcDuty = 50.0 # duty %
fPol = 0 # low or high, 0 or 1
cPulses = 2
sWait = 0

hzSys = c_double()
maxCnt = c_uint()
dwf.FDwfDigitalOutInternalClockInfo(hdwf, byref(hzSys))
dwf.FDwfDigitalOutCounterInfo(hdwf, c_int(0), 0, byref(maxCnt))

# for low frequencies use divider as pre-scaler to satisfy counter limitation of 32k
print(f"hzSys: {hzSys.value}, maxCnt: {maxCnt.value}")
# cDiv = int(math.ceil(hzSys.value/hzFreq/2)) # can use 2 instead of maxCnt.value?
cDiv = int(math.ceil(hzSys.value/hzFreq/maxCnt.value)) # can use 2 instead of maxCnt.value?
# count steps to generate the give frequency
cPulse = int(round(hzSys.value/hzFreq/cDiv))
# duty
cHigh = int(cPulse*prcDuty/100)
cLow = int(cPulse-cHigh)

print("Generate: "+str(hzSys.value/cPulse/cDiv)+"Hz duty: "+str(100.0*cHigh/cPulse)+"% divider: "+str(cDiv)+"High")
print(f"Time per pulse: {cPulse*cDiv/hzSys.value:.6f} s, high: {cHigh*cDiv/hzSys.value:.6f} s, low: {cLow*cDiv/hzSys.value:.6f} s")
print(f"Total time: {cPulses*(cLow+cHigh)*cDiv/hzSys.value:.6f} s")
print(f"High: {cHigh}, Low: {cLow}, Pulse: {cPulse}, Divider: {cDiv}")

dwf.FDwfDigitalOutEnableSet(hdwf, c_int(iChannel), c_int(1)) # 
dwf.FDwfDigitalOutTypeSet(hdwf, c_int(iChannel), DwfDigitalOutTypePulse) # Pulse type - freq = internal clock / divider / counter
dwf.FDwfDigitalOutDividerSet(hdwf, c_int(iChannel), c_int(cDiv)) # max 2147483649, for counter limitation or custom sample rate
dwf.FDwfDigitalOutCounterSet(hdwf, c_int(iChannel), c_int(cLow), c_int(cHigh)) # max 32768
dwf.FDwfDigitalOutCounterInitSet(hdwf, c_int(iChannel), c_int(fPol), c_int(0)) 
dwf.FDwfDigitalOutRunSet(hdwf, c_double(1.0*cPulses*(cLow+cHigh)*cDiv/hzSys.value)) # seconds to run
dwf.FDwfDigitalOutWaitSet(hdwf, c_double(sWait)) # wait after trigger
dwf.FDwfDigitalOutRepeatSet(hdwf, c_int(1)) # infinite
# dwf.FDwfDigitalOutRepeatTriggerSet(hdwf, c_int(1))

# trigger on Trigger IO 
# dwf.FDwfDigitalOutTriggerSourceSet(hdwf, trigsrcExternal1)

# trigger on DIOs
#dwf.FDwfDigitalOutTriggerSourceSet(hdwf, trigsrcDetectorDigitalIn)
#dwf.FDwfDigitalInTriggerSet(hdwf, c_int(0), c_int(0), c_int(1<<1), c_int(0)) # DIO/DIN-1 rise
#dwf.FDwfDigitalInConfigure(hdwf, c_int(1), c_int(1))


print("Armed")
dwf.FDwfDigitalOutConfigure(hdwf, c_int(1))


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

# time.sleep(1)  # wait for a second to see the output

dwf.FDwfDeviceCloseAll()
