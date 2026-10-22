/*
  Pool_EStop_Arduino - physical E-STOP + RESET for Pool_Robot_Menu_EStop.py

  WIRING (no resistors needed - the Arduino's internal pull-ups are used)
    E-STOP button : one leg -> pin 2, other leg -> GND
    RESET  button : one leg -> pin 3, other leg -> GND
    (optional) LED: pin 13 is the on-board LED - lit while the e-stop is latched
    USB cable to the PC (this is also the serial link, 115200 baud)

  With INPUT_PULLUP a pin reads HIGH when the button is up and LOW when pressed.

  BEHAVIOUR
    * Pressing E-STOP LATCHES the stop inside the Arduino. Letting go of the
      button does NOT clear it - only a RESET press does.
    * RESET is ignored while the E-STOP button is still held down.
    * The Arduino sends its state 10 times a second ("HB,ESTOP" or "HB,OK").
      This is a heartbeat: if the PC stops hearing it (cable pulled, board
      crashed) the Python side treats that as an E-STOP too.
    * Each press is also sent once as an event ("EVT,ESTOP" / "EVT,RESET") so
      the menu reacts on the very next frame.

  Serial messages (one per line):
    HB,OK        heartbeat, no stop latched
    HB,ESTOP     heartbeat, stop latched
    EVT,ESTOP    e-stop button just pressed
    EVT,RESET    reset button just pressed (and the latch was cleared)
    EVT,RESET_HELD  reset pressed but refused - e-stop button is still held
    READY        sent once after power-up
*/

const int ESTOP_PIN = 2;
const int RESET_PIN = 3;
const int LED_PIN   = 13;

const unsigned long DEBOUNCE_MS  = 30;    // a reading must be steady this long to count
const unsigned long HEARTBEAT_MS = 100;   // send state every 100 ms

struct Button {
  int pin;
  bool stable;              // debounced level: true = pressed
  bool lastRaw;
  unsigned long changedAt;
};

Button estopBtn = {ESTOP_PIN, false, false, 0};
Button resetBtn = {RESET_PIN, false, false, 0};

bool latched = false;
unsigned long lastBeat = 0;

// Returns true once, on the debounced press edge
bool Pressed_Edge(Button &b) {
  bool raw = (digitalRead(b.pin) == LOW);       // LOW = pressed (button pulls the pin to GND)
  unsigned long now = millis();
  if (raw != b.lastRaw) {                       // still bouncing - restart the timer
    b.lastRaw = raw;
    b.changedAt = now;
  }
  if (raw != b.stable && now - b.changedAt >= DEBOUNCE_MS) {
    b.stable = raw;
    return b.stable;                            // only report the press, not the release
  }
  return false;
}

void Send_Heartbeat() {
  Serial.println(latched ? "HB,ESTOP" : "HB,OK");
}

void setup() {
  pinMode(ESTOP_PIN, INPUT_PULLUP);
  pinMode(RESET_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(115200);

  // If the e-stop is already held down at power-up, start latched (fail safe)
  delay(DEBOUNCE_MS);
  estopBtn.stable = estopBtn.lastRaw = (digitalRead(ESTOP_PIN) == LOW);
  resetBtn.stable = resetBtn.lastRaw = (digitalRead(RESET_PIN) == LOW);
  latched = estopBtn.stable;

  Serial.println("READY");
  Send_Heartbeat();
}

void loop() {
  if (Pressed_Edge(estopBtn)) {
    latched = true;
    Serial.println("EVT,ESTOP");
    Send_Heartbeat();
  }

  // holding the e-stop down keeps it latched, whatever else happens
  if (estopBtn.stable) latched = true;

  if (Pressed_Edge(resetBtn)) {
    if (estopBtn.stable) {
      Serial.println("EVT,RESET_HELD");         // can't reset while the e-stop is still pressed
    } else {
      latched = false;
      Serial.println("EVT,RESET");
    }
    Send_Heartbeat();
  }

  digitalWrite(LED_PIN, latched ? HIGH : LOW);

  if (millis() - lastBeat >= HEARTBEAT_MS) {
    lastBeat = millis();
    Send_Heartbeat();
  }
}
