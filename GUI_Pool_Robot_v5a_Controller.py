import math
import tkinter as tk

import numpy as np

import Pool_Robot_Menu_v2_Controller as Menu

BG = "#20232a"
FG = "#e6e6e6"
FONT = "Segoe UI"

# ------------------------------------THE WINDOW-----------------------------------------------------

window = tk.Tk()
window.title("Cue-T Interface")
window.geometry("380x770+0+0")
window.attributes("-topmost", True)
window.configure(bg=BG)

# ------------------------------------STATE-----------------------------------------------------

state = {"safety": "DISENGAGED", "mode": "none"}

# ------------------------------------STATUS AREA (always at the top)----------------------------------

Title_Label = tk.Label(window, text="9-Ball Command", font=(FONT, 16, "bold"), fg="white", bg=BG)
Title_Label.pack(pady=(10, 2))

Status_Row = tk.Frame(window, bg=BG)
Status_Row.pack(pady=4)

Light_Canvas = tk.Canvas(Status_Row, width=30, height=78, bg=BG, highlightthickness=0)
Light_Canvas.grid(row=0, column=0, rowspan=2, padx=(0, 10))
Light_Canvas.create_rectangle(13, 0, 17, 78, fill="#555555", outline="")
Lamps = {
    "red": Light_Canvas.create_rectangle(4, 2, 26, 24, fill="#4a1414", outline="#111111"),
    "amber": Light_Canvas.create_rectangle(4, 28, 26, 50, fill="#4a3a10", outline="#111111"),
    "green": Light_Canvas.create_rectangle(4, 54, 26, 76, fill="#103a14", outline="#111111"),
}
LAMP_ON = {"red": "#ff2a2a", "amber": "#ffb000", "green": "#2ee05a"}
LAMP_OFF = {"red": "#4a1414", "amber": "#4a3a10", "green": "#103a14"}

Safety_Label = tk.Label(Status_Row, text="", font=(FONT, 12, "bold"), fg=FG, bg=BG, anchor="w")
Safety_Label.grid(row=0, column=1, sticky="w")

Status_Label = tk.Label(Status_Row, text="", font=(FONT, 10), fg="#bbbbbb", bg=BG, anchor="w",
                        wraplength=240, justify="left")
Status_Label.grid(row=1, column=1, sticky="w")


def Show_Light(colour):
    for name, lamp in Lamps.items():
        Light_Canvas.itemconfig(lamp, fill=LAMP_ON[name] if name == colour else LAMP_OFF[name])


def Set_Status(text):
    Status_Label.config(text=text)

# ------------------------------------PAGES-----------------------------------------------------

Page_Area = tk.Frame(window, bg=BG)
Page_Area.pack(fill="x", pady=6)

Main_Page = tk.Frame(Page_Area, bg=BG)
Sequence_Page = tk.Frame(Page_Area, bg=BG)
current_page = {"frame": None}


def Show_Page(page):
    if current_page["frame"] is not None:
        current_page["frame"].pack_forget()

    page.pack(fill="x")
    current_page["frame"] = page

    if page in (Teach_Page, Controller_Page):
        window.geometry("380x880+0+0")
    else:
        window.geometry("380x770+0+0")

    if page is Main_Page:
        Back_Button.pack_forget()
    else:
        Back_Button.pack(side="left", padx=10)


def Big_Button(parent, text, command):
    return tk.Button(parent, text=text, command=command, width=22, height=1,
                     font=(FONT, 11), bg="#3a3f4b", fg="white",
                     activebackground="#50586a", activeforeground="white", relief="flat")

# ---- main page: the three modes ----


def On_Exhibition():
    Show_Page(Sequence_Page)
    Set_Status("Exhibition: pick a step, or Run all")


def On_Teach():
    if state["safety"] != "ARMED":
        Set_Status("Can't start Teach - press ARM first")
        return

    state["mode"] = "teach"

    # Re-sync Teach with the UR3's current joint angles.
    # This is important after using Controller mode.
    Menu.teach["q"] = Menu.np.array(Menu.ur3_p.q, dtype=float)
    Menu.teach["result"] = ""
    Menu.teach["ideal"] = None

    teach_ui["loaded"] = False

    Menu.control["request"] = "teach"

    Show_Page(Teach_Page)
    Set_Status("Teach mode: setting up the UR3...")

def On_Controller():
    if state["safety"] != "ARMED":
        Set_Status("Can't start Controller - press ARM first")
        return

    state["mode"] = "controller"

    # Start Controller mode in Pool_Robot_Menu.
    Menu.control["request"] = "controller"

    # Show the controller joint-angle readout page.
    Show_Page(Controller_Page)

    Set_Status("Controller mode: use the gamepad sticks")


tk.Label(Main_Page, text="Modes", font=(FONT, 12, "bold"), fg=FG, bg=BG).pack(pady=(0, 4))
Big_Button(Main_Page, "1. Exhibition", On_Exhibition).pack(pady=3)
Big_Button(Main_Page, "2. Teach", On_Teach).pack(pady=3)
Big_Button(Main_Page, "3. Controller", On_Controller).pack(pady=3)

Big_Button(Main_Page, "Person opens the door (sensor test)",
           lambda: Menu.control.update(request="door")).pack(pady=(10, 3))
Big_Button(Main_Page, "Person presses the E-STOP by the door",
           lambda: Menu.control.update(request="wall_estop")).pack(pady=3)

# ---- sequence page: the exhibition steps ----

SETUP_STEPS = ["Assembly", "Rack", "Chalk", "Break"]
BALL_COLOURS = {
    1: "#f7c600", 2: "#1f4fd8", 3: "#d62828", 4: "#6a1b9a", 5: "#f77f00",
    6: "#1b8a3a", 7: "#7a1f1f", 8: "#111111", 9: "#f7c600",
}
STRIPES = {9}
RACK_LAYOUT = {
    1: (0, -2), 2: (-0.5, -1), 3: (0.5, -1), 4: (-1, 0), 9: (0, 0),
    5: (1, 0), 6: (-0.5, 1), 7: (0.5, 1), 8: (0, 2),
}
BALL_R = 20
GAP = 2 * BALL_R + 4
done_balls = set()


def Go_To(step, name):
    if state["safety"] != "ARMED":
        Set_Status(f"Can't start {name} - press ARM first")
        return

    Menu.control["request"] = ("goto", step)
    Set_Status(Menu.Step_Name(step))


def Setup_Step(i):
    name = SETUP_STEPS[i]
    first = {"Assembly": 0, "Rack": 1}.get(name)

    if first is None:
        first = next(k for k, n in enumerate(Menu.STEP_NAMES) if n.startswith(name))

    Go_To(first, name)


def Run_All():
    Go_To(0, "the whole sequence")


def On_Ball(n):
    Go_To(Menu.BALL_STEP[n], f"the {n} ball")


def Draw_Ball(canvas, n, x, y, r):
    tag = f"ball{n}"
    colour = BALL_COLOURS[n]
    dim = n in done_balls

    if n in STRIPES:
        canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="", tags=tag)
        h = r * 0.55
        band = []

        for i in range(21):
            a = -math.asin(h / r) + i * 2 * math.asin(h / r) / 20
            band += [x + r * math.cos(a), y + r * math.sin(a)]

        for i in range(21):
            a = math.pi - math.asin(h / r) + i * 2 * math.asin(h / r) / 20
            band += [x + r * math.cos(a), y + r * math.sin(a)]

        canvas.create_polygon(band, fill=colour, outline="", tags=tag)
    else:
        canvas.create_oval(x - r, y - r, x + r, y + r, fill=colour, outline="", tags=tag)

    canvas.create_oval(x - r * 0.5, y - r * 0.5, x + r * 0.5, y + r * 0.5,
                       fill="white", outline="", tags=tag)
    canvas.create_text(x, y, text=str(n), font=(FONT, 9, "bold"), fill="black", tags=tag)
    canvas.create_oval(x - r, y - r, x + r, y + r, outline="#000000", width=1, tags=tag)

    if dim:
        canvas.create_oval(x - r, y - r, x + r, y + r, fill=BG, stipple="gray50", outline="", tags=tag)
        canvas.create_line(x - r * 0.6, y, x - r * 0.1, y + r * 0.5, x + r * 0.6, y - r * 0.5,
                           fill=LAMP_ON["green"], width=3, tags=tag)

    canvas.tag_bind(tag, "<Button-1>", lambda event, n=n: On_Ball(n))


def Draw_Rack():
    Rack_Canvas.delete("all")
    cx, cy = RACK_W / 2, RACK_H / 2

    for n, (col, row) in RACK_LAYOUT.items():
        Draw_Ball(Rack_Canvas, n, cx + col * GAP, cy + row * GAP * 0.87, BALL_R)


tk.Label(Sequence_Page, text="Sequence", font=(FONT, 12, "bold"), fg=FG, bg=BG).pack(pady=(0, 4))
Step_Label = tk.Label(Sequence_Page, text="", font=(FONT, 10, "bold"), fg=LAMP_ON["green"], bg=BG)
Step_Label.pack(pady=(0, 4))

Step_Row = tk.Frame(Sequence_Page, bg=BG)
Step_Row.pack()

for i, name in enumerate(SETUP_STEPS):
    number = i + 1
    tk.Button(Step_Row, text=f"{number}. {name}", width=14,
              command=lambda i=i: Setup_Step(i),
              font=(FONT, 10), bg="#3a3f4b", fg="white", activebackground="#50586a",
              relief="flat").grid(row=i // 2, column=i % 2, padx=3, pady=3)

RACK_W, RACK_H = 2 * GAP + 2 * BALL_R + 10, 4 * GAP * 0.87 + 2 * BALL_R + 10
Rack_Canvas = tk.Canvas(Sequence_Page, width=RACK_W, height=RACK_H, bg="#0b6e3b",
                        highlightthickness=2, highlightbackground="#5a3a1a", cursor="hand2")
Rack_Canvas.pack(pady=(6, 2))
Draw_Rack()

Big_Button(Sequence_Page, "Run all", Run_All).pack(pady=(6, 0))

# ---- teach page: the UR3 and one straight shot ----
# Sliders set the UR3's six joints directly (like the joint-slider labs). Below them: the best
# configuration for this shot, and the checks - each warning says what is wrong and which joint
# to move. SHOOT rocks back once and strikes along the cue; the physics uses the real cue line.

SMALL = (FONT, 8)
Teach_Page = tk.Frame(Page_Area, bg=BG)
tk.Label(Teach_Page, text="Teach - UR3 straight shot", font=(FONT, 11, "bold"), fg=FG, bg=BG).pack()
Slider_Box = tk.Frame(Teach_Page, bg=BG)
Slider_Box.pack()

teach_ui = {"sync": False, "tick": 0}
Joint_Sliders = []


def On_Slider(_=None):
    if teach_ui["sync"] or Menu.teach["q"] is None:
        return

    Menu.teach["q"] = Menu.np.radians([s.get() for s in Joint_Sliders])
    Menu.teach["result"] = ""  # a new configuration - old result no longer applies


for j in range(6):
    tk.Label(Slider_Box, text=f"J{j + 1}", font=SMALL, fg=FG, bg=BG).grid(
        row=j, column=0, sticky="se", padx=(0, 4))

    s = tk.Scale(Slider_Box, from_=-180, to=180, resolution=0.1, orient="horizontal", length=250, width=10,
                 sliderlength=16, font=SMALL, command=On_Slider, bg=BG, fg=FG, troughcolor="#3a3f4b",
                 highlightthickness=0)
    s.grid(row=j, column=1)
    Joint_Sliders.append(s)

tk.Label(Slider_Box, text="m/s", font=SMALL, fg=FG, bg=BG).grid(row=6, column=0, sticky="se", padx=(0, 4))
Speed_Slider = tk.Scale(Slider_Box, from_=0.3, to=1.5, resolution=0.1, orient="horizontal", length=250, width=10,
                        sliderlength=16, font=SMALL, command=lambda v: Menu.teach.update(speed=float(v)), bg=BG,
                        fg=FG, troughcolor="#3a3f4b", highlightthickness=0)
Speed_Slider.set(0.7)
Speed_Slider.grid(row=6, column=1)


def Set_Sliders(q):
    teach_ui["sync"] = True

    for s, v in zip(Joint_Sliders, Menu.np.degrees(q)):
        s.set(round((v + 180) % 360 - 180, 1))

    teach_ui["sync"] = False
    On_Slider()


def Use_Example():
    if Menu.teach["ideal"] is not None:
        Set_Sliders(Menu.teach["ideal"])
        Menu.teach["q"] = Menu.np.array(Menu.teach["ideal"], dtype=float)  # exact, not rounded to the sliders


def Shoot():
    if state["safety"] != "ARMED":
        Set_Status("Press ARM first")
        return

    Menu.teach["result"] = ""
    Menu.control["request"] = "teach_shoot"


Button_Row = tk.Frame(Teach_Page, bg=BG)
Button_Row.pack(pady=3)

tk.Button(Button_Row, text="SHOOT", width=8, command=Shoot, font=(FONT, 10, "bold"), bg="#1f6f3a", fg="white",
          activebackground="#2e8f4e", relief="flat").pack(side="left", padx=2)
tk.Button(Button_Row, text="Use example", width=10, command=Use_Example, font=SMALL, bg="#3a3f4b", fg="white",
          relief="flat").pack(side="left", padx=2)
tk.Button(Button_Row, text="Reset balls", width=9, command=lambda: Menu.control.update(request="teach_reset"),
          font=SMALL, bg="#3a3f4b", fg="white", relief="flat").pack(side="left", padx=2)

Example_Label = tk.Label(Teach_Page, text="", font=("Consolas", 8), fg="#7fc8ff", bg=BG, justify="left")
Example_Label.pack(anchor="w", padx=8)

Check_Labels = [tk.Label(Teach_Page, text="", font=SMALL, bg=BG, justify="left", anchor="w", wraplength=340)
                for _ in range(5)]

for lab in Check_Labels:
    lab.pack(anchor="w", padx=8)

Result_Label = tk.Label(Teach_Page, text="", font=(FONT, 9, "bold"), fg="#ffd166", bg=BG, wraplength=340,
                        justify="left")
Result_Label.pack(anchor="w", padx=8, pady=(2, 0))


def Teach_Update():
    """Refresh the example and the checks (a few times a second)."""
    t = Menu.teach

    if not t.get("ready") or t["q"] is None:
        return

    if not teach_ui.get("loaded"):
        Set_Sliders(t["q"])
        teach_ui["loaded"] = True

    if t["ideal"] is not None:
        v = Menu.np.degrees(t["ideal"])
        Example_Label.config(
            text="Best (example): " + " ".join(f"J{j + 1} {v[j]:.0f}" for j in range(3)) + "\n"
                 "                    " + " ".join(f"J{j + 1} {v[j]:.0f}" for j in range(3, 6)))

    checks, _ = Menu.Teach_Checks(t["q"])

    for lab, item in zip(Check_Labels, checks + [None] * 5):
        if item is None:
            lab.config(text="")
        else:
            ok, text = item
            lab.config(text=("OK " if ok else "! ") + text,
                       fg="#2ee05a" if ok else "#ffb000")

    if all(ok for ok, _ in checks):
        Result_Label.config(text=t["result"] or "Ready - press SHOOT",
                            fg="#2ee05a" if not t["result"] else "#ffd166")
    else:
        Result_Label.config(text=t["result"] or "Not ideal for a straight shot yet - see the warnings",
                            fg="#ffb000")

# ---- controller page: live UR3 joint-angle readout ----

Controller_Page = tk.Frame(Page_Area, bg=BG)

tk.Label(Controller_Page, text="Controller - UR3 joint angles",
         font=(FONT, 11, "bold"), fg=FG, bg=BG).pack(pady=(0, 6))

Joint_Labels = []

for j in range(6):
    row = tk.Frame(Controller_Page, bg=BG)
    row.pack(fill="x", padx=10, pady=2)

    tk.Label(row, text=f"J{j + 1}", font=SMALL, fg=FG, bg=BG,
             width=4, anchor="w").pack(side="left")

    value_label = tk.Label(row, text="0.0 deg", font=("Consolas", 10),
                           fg="#7fc8ff", bg=BG, width=14, anchor="e")
    value_label.pack(side="right")

    Joint_Labels.append(value_label)


def Update_Joint_Labels():
    """Show the UR3's current joint angles as text."""

    q = Menu.ur3_p.q

    if q is None:
        return

    q_deg = np.degrees(np.asarray(q, dtype=float))

    for label, value in zip(Joint_Labels, q_deg):
        label.config(text=f"{value:8.1f} deg")

# ------------------------------------SAFETY PANEL (always visible, middle of the window)-------------------

Safety_Panel = tk.Frame(window, bg=BG)
Safety_Panel.pack(expand=True)


def On_EStop_Button():
    if state["safety"] == "ESTOP":
        state["safety"] = "DISENGAGED"
        Menu.control["request"] = "reset"
        Set_Status("E-stop released. Nothing moves until you press ARM.")
    else:
        state["safety"] = "ESTOP"
        Menu.Handle_Request("estop")
        Set_Status("E-STOP - all motion stopped")

    Update_Safety()


def On_Arm_Button():
    if state["safety"] == "DISENGAGED":
        state["safety"] = "ARMED"
        Menu.control["request"] = "resume"
        Set_Status("Armed - robots may move")
    elif state["safety"] == "ARMED":
        state["safety"] = "DISENGAGED"
        Menu.control["request"] = "pause"
        Set_Status("Disarmed - robots hold still")
    else:
        Set_Status("Release the E-STOP first (press DISENGAGE)")

    Update_Safety()


EStop_Button = tk.Button(Safety_Panel, command=On_EStop_Button, width=14, height=3,
                         font=(FONT, 16, "bold"), fg="white", relief="raised", bd=4)
EStop_Button.pack(pady=(4, 6))

Arm_Button = tk.Button(Safety_Panel, command=On_Arm_Button, width=14, font=(FONT, 11, "bold"),
                       relief="flat")
Arm_Button.pack()


def Update_Safety():
    s = state["safety"]

    if s == "ESTOP":
        EStop_Button.config(text="DISENGAGE", bg="#8b0000", activebackground="#a00000")
        Arm_Button.config(text="ARM", bg="#444444", fg="#888888", state="disabled")
        Safety_Label.config(text="E-STOP", fg=LAMP_ON["red"])
        Show_Light("red")
    elif s == "DISENGAGED":
        EStop_Button.config(text="E-STOP", bg="#d62828", activebackground="#ff4444")
        Arm_Button.config(text="ARM", bg="#c98a00", fg="white", state="normal")
        Safety_Label.config(text="DISENGAGED", fg=LAMP_ON["amber"])
        Show_Light("amber")
    else:
        EStop_Button.config(text="E-STOP", bg="#d62828", activebackground="#ff4444")
        Arm_Button.config(text="DISARM", bg="#2e7d32", fg="white", state="normal")
        Safety_Label.config(text="ARMED", fg=LAMP_ON["green"])
        Show_Light("green")

# ------------------------------------BOTTOM BAR: Back + Quit (always at the very bottom)----------------

Bottom_Bar = tk.Frame(window, bg=BG)
Bottom_Bar.pack(side="bottom", fill="x", pady=10)

Back_Button = tk.Button(Bottom_Bar, text="< Back", width=10, command=lambda: Show_Page(Main_Page))
Quit_Button = tk.Button(Bottom_Bar, text="Quit", width=10, command=lambda: Menu.control.update(request="quit"))
Quit_Button.pack(side="right", padx=10)

# ------------------------------------START-----------------------------------------------------

Show_Page(Main_Page)
Update_Safety()
Set_Status("Ready - press ARM, then pick a mode")

last = {"text": Menu.control.get("text")}


def Tick():
    if not Menu.Step_Once():
        window.destroy()
        return

    Menu.env.step(Menu.DT)

    sim = Menu.control["state"]

    if sim == "ESTOPPED" and state["safety"] != "ESTOP":  # e-stopped from outside the panel (wall e-stop)
        state["safety"] = "ESTOP"
        Update_Safety()

    if current_page["frame"] is Teach_Page:
        teach_ui["tick"] += 1

        if teach_ui["tick"] % 10 == 0:
            Teach_Update()

    if current_page["frame"] is Controller_Page:
        Update_Joint_Labels()

    if state["safety"] == "ARMED":
        Show_Light({"RUNNING": "green", "DOOR_OPEN": "red", "ESTOPPED": "red"}.get(sim, "amber"))

    Step_Label.config(text=Menu.Step_Name() if Menu.state.get("step") is not None else "")

    if Menu.control.get("text") != last["text"]:
        last["text"] = Menu.control.get("text")
        Set_Status(last["text"])

    potted = Menu.Potted_Balls()

    if potted != done_balls:
        done_balls.clear()
        done_balls.update(potted)
        Draw_Rack()

    window.after(1, Tick)


Tick()
window.mainloop()