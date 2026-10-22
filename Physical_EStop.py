"""
Physical_EStop - reads the Arduino E-STOP / RESET box (Pool_EStop_Arduino.ino)
in a background thread, so the Swift main loop never blocks on the serial port.

    box = Physical_EStop()          # finds the Arduino by itself (or Physical_EStop("COM5"))
    box.Start()
    ...
    every frame:
        for event in box.Poll():    # "estop", "reset", "reset_refused", "link_lost", "link_back"
            ...
        box.latched                 # True while the Arduino says the stop is latched
        box.link_ok                 # True while heartbeats are arriving

If pyserial is not installed or no Arduino is plugged in, Start() returns False,
prints why, and Poll() just returns nothing - the GUI buttons keep working.

Fail safe: once the Arduino has been heard from, losing its heartbeat for more
than HEARTBEAT_TIMEOUT seconds (cable pulled, board reset) raises "link_lost",
which the menu treats exactly like pressing the E-STOP. The thread keeps trying
to reconnect, and raises "link_back" when the heartbeat returns.

Needs:  pip install pyserial
"""
import queue
import threading
import time

try:
    import serial
    import serial.tools.list_ports
except ImportError:                         # the menu still runs without it
    serial = None

BAUD = 115200
HEARTBEAT_TIMEOUT = 0.5                     # s without a heartbeat -> treat as e-stop
ARDUINO_HINTS = ("arduino", "ch340", "ch341", "usb serial", "usb-serial", "cp210", "ftdi", "wch")
ARDUINO_VIDS = {0x2341, 0x2A03, 0x1A86, 0x10C4, 0x0403}   # Arduino, Arduino.org, CH340, CP210x, FTDI


def Find_Arduino_Port():
    """Return the first serial port that looks like an Arduino, or None."""
    if serial is None:
        return None
    for p in serial.tools.list_ports.comports():
        text = f"{p.description} {p.manufacturer or ''}".lower()
        if (p.vid in ARDUINO_VIDS) or any(h in text for h in ARDUINO_HINTS):
            return p.device
    return None


class Physical_EStop:
    def __init__(self, port=None, baud=BAUD):
        self.port, self.baud = port, baud
        self.events = queue.Queue()
        self.latched = False                # Arduino's own latch (from the heartbeat)
        self.link_ok = False                # heartbeats arriving right now
        self.ever_connected = False         # only watch the heartbeat once we have heard one
        self._last_beat = 0.0
        self._stop = threading.Event()
        self._thread = None

    # ---------------------------------------------------------------- public
    def Start(self):
        """Open the port and start the reader thread. False if there is no Arduino."""
        if serial is None:
            print("[E-STOP box] pyserial not installed (pip install pyserial) - using GUI buttons only")
            return False
        if self.port is None:
            self.port = Find_Arduino_Port()
        if self.port is None:
            print("[E-STOP box] no Arduino found - using GUI buttons only")
            return False
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.1)
        except (serial.SerialException, OSError) as err:
            print(f"[E-STOP box] could not open {self.port}: {err} - using GUI buttons only")
            return False
        print(f"[E-STOP box] listening on {self.port} (board restarts on connect, ~2 s)")
        self._thread = threading.Thread(target=self._Run, args=(ser,), daemon=True)
        self._thread.start()
        return True

    def Poll(self):
        """All events since the last call (call once per frame from the main loop)."""
        self._Check_Heartbeat()
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def Close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    # ---------------------------------------------------------------- internals
    def _Check_Heartbeat(self):
        if self.ever_connected and self.link_ok and time.monotonic() - self._last_beat > HEARTBEAT_TIMEOUT:
            self.link_ok = False
            self.events.put("link_lost")

    def _Handle_Line(self, line):
        if line.startswith("HB,"):
            self._last_beat = time.monotonic()
            self.latched = (line == "HB,ESTOP")
            if not self.link_ok:
                self.link_ok = True
                if self.ever_connected:
                    self.events.put("link_back")
                self.ever_connected = True
            if self.latched:                 # a latched stop always wins, even if an event was missed
                self.events.put("estop_level")
        elif line == "EVT,ESTOP":
            self.latched = True
            self.events.put("estop")
        elif line == "EVT,RESET":
            self.latched = False
            self.events.put("reset")
        elif line == "EVT,RESET_HELD":
            self.events.put("reset_refused")

    def _Run(self, ser):
        while not self._stop.is_set():
            try:
                if ser is None:              # try to reconnect after the cable was pulled
                    time.sleep(0.5)
                    port = Find_Arduino_Port() or self.port
                    ser = serial.Serial(port, self.baud, timeout=0.1)
                    self.port = port
                    print(f"[E-STOP box] reconnected on {port}")
                raw = ser.readline()
                if raw:
                    self._Handle_Line(raw.decode("ascii", errors="ignore").strip())
            except (serial.SerialException, OSError):
                try:
                    if ser is not None:
                        ser.close()
                except Exception:
                    pass
                ser = None                   # heartbeat stops -> Poll() raises link_lost
        if ser is not None:
            ser.close()


if __name__ == "__main__":
    # quick bench test: python Physical_EStop.py   then press the buttons
    box = Physical_EStop()
    if box.Start():
        try:
            while True:
                for e in box.Poll():
                    if e != "estop_level":
                        print("event:", e)
                time.sleep(0.02)
        except KeyboardInterrupt:
            box.Close()
