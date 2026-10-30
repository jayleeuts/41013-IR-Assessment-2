# ------------------------------------PLC WIRING TEST---------------------------------------------------------
# Run this first with the real PLC plugged in (no simulation needed). It switches the stack light outputs
# one by one, finds which input the e-stop and reset button are wired to (press each when asked) and prints
# the INPUT_ROLES lines for Pool_PLC.py, then shows every input and light change live.

import time
import serial
from Pool_PLC import (COM_PORT, BAUD_RATE, INPUT_NAMES, OUTPUT_NAMES, LIGHTS, Read_Bits, Write_Coil)


def Read_Inputs(plc):
    return dict(zip(INPUT_NAMES, Read_Bits(plc, 2, len(INPUT_NAMES))))


def Show(values):
    return "  ".join(f"{k}={int(v)}" for k, v in values.items())


def Light_Test(plc):
    print("\n--- 1. LIGHT TEST ---")
    for colour, name in LIGHTS.items():
        for other in LIGHTS.values():
            Write_Coil(plc, OUTPUT_NAMES.index(other), other == name)
        time.sleep(0.3)
        y = dict(zip(OUTPUT_NAMES, Read_Bits(plc, 1, len(OUTPUT_NAMES))))
        stuck = "" if y[name] else "   <-- the PLC program overwrites it (fine: keep LIGHT_MODE = \"PLC\")"
        input(f"{name} should be {colour.upper()} now{stuck}. Press Enter for the next one...")
    for other in LIGHTS.values():
        Write_Coil(plc, OUTPUT_NAMES.index(other), False)


def Find_Input(plc, role, how):
    input(f"\n{role}: leave everything normal, then press Enter...")
    before = Read_Inputs(plc)
    print(f"   now {how} (you have 8 s)...")
    end = time.time() + 8
    while time.time() < end:
        now = Read_Inputs(plc)
        changed = [k for k in INPUT_NAMES if now[k] != before[k]]
        if changed:
            name = changed[0]
            print(f"   -> {name} changed {int(before[name])} -> {int(now[name])}")
            print("   now release / reset it...")
            back = time.time() + 15
            while Read_Inputs(plc)[name] != before[name] and time.time() < back:
                time.sleep(0.05)
            return f'    "{role}": ("{name}", {int(now[name])}),'
        time.sleep(0.05)
    print("   no input changed - not wired (or check the wiring)")
    return f'    "{role}": (None, 1),'


if __name__ == "__main__":
    with serial.Serial(port=COM_PORT, baudrate=BAUD_RATE, bytesize=serial.EIGHTBITS,
                       parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0.5) as plc:
        print("Inputs now:", Show(Read_Inputs(plc)))
        Light_Test(plc)

        print("\n--- 2. FIND INPUTS ---")
        lines = [Find_Input(plc, "estop", "PRESS the e-stop"),
                 Find_Input(plc, "reset", "press the RESET button")]
        print("\nPaste this into INPUT_ROLES in Pool_PLC.py:\n")
        print("INPUT_ROLES = {")
        print("\n".join(lines))
        print("}")

        print("\n--- 3. LIVE VIEW (Ctrl+C to stop) ---")
        print("Press the e-stop, release it, press reset... and watch what the light does.")
        colour_of = {name: colour for colour, name in LIGHTS.items()}
        last = {**Read_Inputs(plc), **dict(zip(OUTPUT_NAMES, Read_Bits(plc, 1, len(OUTPUT_NAMES))))}
        print(Show(last))
        try:
            while True:
                now = {**Read_Inputs(plc), **dict(zip(OUTPUT_NAMES, Read_Bits(plc, 1, len(OUTPUT_NAMES))))}
                for k in now:
                    if now[k] != last[k]:
                        what = f" ({colour_of[k]} light {'ON' if now[k] else 'off'})" if k in colour_of else ""
                        print(f"{time.strftime('%H:%M:%S')}  {k}: {int(last[k])} -> {int(now[k])}{what}")
                last = now
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\ndone")
