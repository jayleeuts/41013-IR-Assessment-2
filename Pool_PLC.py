# ------------------------------------REAL PLC LINK-----------------------------------------------------------
# Connects the real Mitsubishi FX3U PLC to the simulation over USB (Modbus RTU), using the same messages as
# the tutor's fx3u_read_io.py. A background thread reads the PLC every POLL_TIME seconds so the simulation
# never waits on it. If the PLC stops answering for COMMS_TIMEOUT seconds a comms fault is reported and the
# simulation stops (fail safe); if no PLC is plugged in, the simulation simply runs on its own.

import struct
import threading
import time

try:
    import serial
except ImportError:
    serial = None


# ------------------------------------SETTINGS----------------------------------------------------------------
# Port and timing, the PLC's input and output names, which input is the e-stop (check with PLC_IO_Test.py),
# and which output drives each lamp. In PLC mode the PLC's own ladder program drives the stack light and the
# simulation follows it: red = e-stop, yellow = disengaged, green = engaged.

COM_PORT = "COM5"
BAUD_RATE = 57600
SLAVE_ID = 1
POLL_TIME = 0.1
COMMS_TIMEOUT = 1.0
RETRY_TIME = 3.0

INPUT_NAMES = tuple([f"X{i}" for i in range(8)] + [f"X1{i}" for i in range(6)])
OUTPUT_NAMES = tuple([f"Y{i}" for i in range(8)] + ["Y10", "Y11"])

INPUT_ROLES = {
    "estop": ("X14", 1),
    "reset": (None, 1),
}

LIGHTS = {"red": "Y1", "amber": "Y2", "green": "Y3"}

LIGHT_MODE = "PLC"
LIGHT_MEANING = {"red": "ESTOP", "amber": "DISENGAGED", "green": "ENGAGED"}


# ------------------------------------MODBUS RTU MESSAGES-----------------------------------------------------
# Builds each request, checks the reply's address, function and checksum, and unpacks the bits. Function 2
# reads the inputs, function 1 reads the outputs and function 5 switches one output.

def Crc16(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def Read_Exact(port, length):
    received = bytearray()
    while len(received) < length:
        chunk = port.read(length - len(received))
        if not chunk:
            break
        received.extend(chunk)
    if len(received) != length:
        raise RuntimeError(f"Timeout: received {len(received)} of {length} expected bytes")
    return bytes(received)


def Check_Reply(response, function):
    if response[0] != SLAVE_ID or response[1] != function:
        raise RuntimeError("Unexpected Modbus response")
    if Crc16(response[:-2]) != struct.unpack("<H", response[-2:])[0]:
        raise RuntimeError("Modbus CRC check failed")


def Read_Bits(port, function, quantity):
    request = struct.pack(">BBHH", SLAVE_ID, function, 0, quantity)
    request += struct.pack("<H", Crc16(request))
    port.reset_input_buffer()
    port.write(request)
    port.flush()
    n_bytes = (quantity + 7) // 8
    response = Read_Exact(port, 3 + n_bytes + 2)
    Check_Reply(response, function)
    if response[2] != n_bytes:
        raise RuntimeError("Unexpected number of data bytes")
    packed = response[3:-2]
    return tuple(bool(packed[i // 8] & (1 << (i % 8))) for i in range(quantity))


def Write_Coil(port, index, value):
    request = struct.pack(">BBHH", SLAVE_ID, 5, index, 0xFF00 if value else 0x0000)
    request += struct.pack("<H", Crc16(request))
    port.reset_input_buffer()
    port.write(request)
    port.flush()
    response = Read_Exact(port, 8)
    Check_Reply(response, 5)


# ------------------------------------THE LINK (BACKGROUND THREAD)--------------------------------------------
# Keeps the latest inputs and outputs, reopens the port if it is unplugged, and answers the simple questions
# the control panel asks: is it connected, is the e-stop active, which colour is the PLC showing.

class PLC_Link:
    def __init__(self, port_name=COM_PORT):
        self.port_name = port_name
        self.port = None
        self.inputs = {}
        self.outputs = {}
        self.want_light = None
        self.lit = None
        self.last_ok = 0.0
        self.ever_connected = False
        self.error = "not started"
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._Run, daemon=True).start()

    def Connected(self):
        return self.port is not None and time.time() - self.last_ok < COMMS_TIMEOUT

    def Comms_Fault(self):
        return self.ever_connected and not self.Connected()

    def Active(self, role):
        name, active_value = INPUT_ROLES.get(role, (None, 1))
        if name is None or not self.Connected():
            return False
        with self.lock:
            return name in self.inputs and int(self.inputs[name]) == active_value

    def Status_Text(self):
        if self.Connected():
            return f"PLC: connected on {self.port_name}"
        if self.Comms_Fault():
            return "PLC: COMMS FAULT - no reply"
        return "PLC: not connected - simulation only"

    def Set_Light(self, colour):
        self.want_light = colour

    def PLC_Light(self):
        if not self.Connected():
            return None
        with self.lock:
            on = [colour for colour, name in LIGHTS.items() if self.outputs.get(name)]
        return on[0] if on else None

    def Stop(self):
        self.running = False

    def _Open(self):
        if serial is None:
            if not self.error.startswith("pyserial"):
                print("[PLC] pyserial not installed (pip install pyserial) - running without the PLC")
            self.error = "pyserial not installed"
            return
        if self.error.startswith("can't open"):
            try:
                from serial.tools import list_ports
                names = [p.device for p in list_ports.comports()]
                if names and self.port_name not in names:
                    self.port_name = names[0] if len(names) == 1 else next(
                        (p.device for p in list_ports.comports()
                         if any(k in (p.description or "") for k in ("USB", "CH340", "Serial"))), self.port_name)
            except Exception:
                pass
        try:
            self.port = serial.Serial(port=self.port_name, baudrate=BAUD_RATE, bytesize=serial.EIGHTBITS,
                                      parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0.3)
            self.error = ""
        except Exception as err:
            if self.error != f"can't open {self.port_name}":
                print(f"[PLC] can't open {self.port_name} - running without the PLC ({err})")
            self.port, self.error = None, f"can't open {self.port_name}"

    def _Close(self):
        try:
            self.port.close()
        except Exception:
            pass
        self.port = None

    def _Run(self):
        next_try = 0.0
        while self.running:
            if self.port is None:
                if time.time() >= next_try:
                    self._Open()
                    next_try = time.time() + RETRY_TIME
                time.sleep(0.2)
                continue
            try:
                x = Read_Bits(self.port, 2, len(INPUT_NAMES))
                y = Read_Bits(self.port, 1, len(OUTPUT_NAMES))
                with self.lock:
                    self.inputs = dict(zip(INPUT_NAMES, x))
                    self.outputs = dict(zip(OUTPUT_NAMES, y))
                want = self.want_light
                if LIGHT_MODE == "PYTHON" and want is not None and (want != self.lit or not self._Light_Matches(want)):
                    for colour, name in LIGHTS.items():
                        Write_Coil(self.port, OUTPUT_NAMES.index(name), colour == want)
                    self.lit = want
                self.last_ok = time.time()
                self.ever_connected = True
                self.error = ""
            except Exception as err:
                self.error = str(err)
                if time.time() - self.last_ok > COMMS_TIMEOUT:
                    self._Close()
            time.sleep(POLL_TIME)
        if self.port is not None and LIGHT_MODE == "PYTHON":
            try:
                for name in LIGHTS.values():
                    Write_Coil(self.port, OUTPUT_NAMES.index(name), False)
            except Exception:
                pass
        if self.port is not None:
            self._Close()

    def _Light_Matches(self, want):
        with self.lock:
            return all(bool(self.outputs.get(name)) == (colour == want) for colour, name in LIGHTS.items())
