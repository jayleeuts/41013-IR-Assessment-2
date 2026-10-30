# ------------------------------------CONTROL PANEL-----------------------------------------------------------
# The pop-up control panel (tkinter) for the pool workcell in Pool_Robot_Menu.py. It never moves robots
# itself: every button becomes a request for the simulation, and a timer (Tick) advances the simulation and
# refreshes the panel. It also links to the real PLC (Pool_PLC.py) so the real e-stop and stack light work
# with the simulation.

import math
import numpy as np
import tkinter as tk

import Pool_Robot_Menu as Menu
import Pool_PLC

BG = "#20232a"
FG = "#e6e6e6"
FONT = "Segoe UI"


# ------------------------------------THE WINDOW--------------------------------------------------------------
# A narrow window pinned to the top-left corner of the screen and kept on top, so it never hides the Swift
# view.

window = tk.Tk()
window.title("Cue-T Interface")
window.geometry("380x790+0+0")
window.attributes("-topmost", True)
window.configure(bg=BG)


# ------------------------------------STATE-------------------------------------------------------------------
# The panel's own safety state (DISENGAGED, ENGAGED or ESTOP) and the link to the real PLC. When the PLC's own
# program drives the stack light (PLC_MASTER), the panel follows the PLC rather than the other way round.

state = {"safety": "DISENGAGED"}
plc = Pool_PLC.PLC_Link()
plc_prev = {"reset": False, "text": "", "light": None, "local_estop": False}
PLC_MASTER = Pool_PLC.LIGHT_MODE == "PLC"


# ------------------------------------TOP BAR-----------------------------------------------------------------
# Back and Quit sit beside the title, so they are always on screen whatever page is open.

Top_Bar = tk.Frame(window, bg=BG)
Top_Bar.pack(fill="x", pady=(8, 2))
Top_Bar.columnconfigure(1, weight=1)
Top_Bar.columnconfigure(0, minsize=80)
Top_Bar.columnconfigure(2, minsize=80)
Back_Button = tk.Button(Top_Bar, text="< Back", width=6, command=lambda: Show_Page(Main_Page))
Back_Button.grid(row=0, column=0, padx=8)
Title_Label = tk.Label(Top_Bar, text="9-Ball Command", font=(FONT, 13, "bold"), fg="white", bg=BG)
Title_Label.grid(row=0, column=1)
Quit_Button = tk.Button(Top_Bar, text="Quit", width=6, command=lambda: Menu.control.update(request="quit"))
Quit_Button.grid(row=0, column=2, padx=8)


# ------------------------------------STATUS AREA-------------------------------------------------------------
# A small stack light, the safety state, the latest status message and the PLC connection line.

Status_Row = tk.Frame(window, bg=BG)
Status_Row.pack(pady=4)

Light_Canvas = tk.Canvas(Status_Row, width=30, height=78, bg=BG, highlightthickness=0)
Light_Canvas.grid(row=0, column=0, rowspan=2, padx=(0, 10))
Light_Canvas.create_rectangle(13, 0, 17, 78, fill="#555555", outline="")
Lamps = {"red":   Light_Canvas.create_rectangle(4, 2, 26, 24, fill="#4a1414", outline="#111111"),
         "amber": Light_Canvas.create_rectangle(4, 28, 26, 50, fill="#4a3a10", outline="#111111"),
         "green": Light_Canvas.create_rectangle(4, 54, 26, 76, fill="#103a14", outline="#111111")}
LAMP_ON = {"red": "#ff2a2a", "amber": "#ffb000", "green": "#2ee05a"}
LAMP_OFF = {"red": "#4a1414", "amber": "#4a3a10", "green": "#103a14"}

Safety_Label = tk.Label(Status_Row, text="", font=(FONT, 12, "bold"), fg=FG, bg=BG, anchor="w")
Safety_Label.grid(row=0, column=1, sticky="w")
Status_Label = tk.Label(Status_Row, text="", font=(FONT, 10), fg="#bbbbbb", bg=BG, anchor="w",
                        wraplength=240, justify="left")
Status_Label.grid(row=1, column=1, sticky="w")
PLC_Label = tk.Label(window, text="", font=(FONT, 9), fg="#8a8a8a", bg=BG, wraplength=360)
PLC_Label.pack()


def Show_Light(colour):
    plc.Set_Light(colour)
    for name, lamp in Lamps.items():
        Light_Canvas.itemconfig(lamp, fill=LAMP_ON[name] if name == colour else LAMP_OFF[name])


def Set_Status(text):
    Status_Label.config(text=text)


# ------------------------------------PAGES-------------------------------------------------------------------
# The middle of the window swaps between pages (main, sequence, teach, controller). The window resizes to fit
# each page but never grows past the screen: on a small screen the page scrolls (scroll bar or mouse wheel)
# while the top bar and the safety buttons stay fixed, so Back, Quit and E-STOP are always reachable.

Page_Holder = tk.Frame(window, bg=BG)
Page_Holder.pack(fill="both", expand=True, pady=6)
Page_Canvas = tk.Canvas(Page_Holder, bg=BG, highlightthickness=0, width=360)
Page_Scroll = tk.Scrollbar(Page_Holder, orient="vertical", command=Page_Canvas.yview)
Page_Canvas.configure(yscrollcommand=Page_Scroll.set)
Page_Canvas.pack(side="left", fill="both", expand=True)
Page_Area = tk.Frame(Page_Canvas, bg=BG)
Page_Canvas.create_window((0, 0), window=Page_Area, anchor="nw", tags="page")
Page_Area.bind("<Configure>", lambda e: Page_Canvas.configure(scrollregion=Page_Canvas.bbox("all")))
Page_Canvas.bind("<Configure>", lambda e: Page_Canvas.itemconfig("page", width=e.width))
window.bind_all("<MouseWheel>", lambda e: Page_Canvas.yview_scroll(int(-e.delta / 120), "units"))

Main_Page = tk.Frame(Page_Area, bg=BG)
Sequence_Page = tk.Frame(Page_Area, bg=BG)
current_page = {"frame": None}


def Show_Page(page):
    if current_page["frame"] is not None:
        current_page["frame"].pack_forget()
    page.pack(fill="x")
    current_page["frame"] = page
    if page is Main_Page:
        Back_Button.grid_remove()
    else:
        Back_Button.grid()
    Page_Canvas.yview_moveto(0)
    window.update_idletasks()
    Fit_Window()


def Fit_Window():
    needed = window.winfo_reqheight() - Page_Canvas.winfo_reqheight() + Page_Area.winfo_reqheight()
    if needed == current_page.get("needed"):
        return
    current_page["needed"] = needed
    room = window.winfo_screenheight() - 110
    window.geometry(f"380x{min(needed + 6, room)}+0+0")
    if needed + 6 > room:
        Page_Scroll.pack(side="right", fill="y")
    else:
        Page_Scroll.pack_forget()


def Big_Button(parent, text, command):
    return tk.Button(parent, text=text, command=command, width=22, height=1,
                     font=(FONT, 11), bg="#3a3f4b", fg="white", activebackground="#50586a",
                     activeforeground="white", relief="flat")


# ------------------------------------MAIN PAGE---------------------------------------------------------------
# The three modes plus two safety demonstrations (a person opens the door, or presses the wall e-stop).
# Teach and Controller only start when the cell is ENGAGED. Picking a mode completely drops the mode that was
# running (Exhibition drops Teach / Controller as soon as its page opens).

def On_Exhibition():
    if Menu.control["mode"] in ("teach", "controller"):
        Menu.control["request"] = "drop_mode"
    Show_Page(Sequence_Page)
    Set_Status("Exhibition: pick a step, or Run all")


def On_Teach():
    if state["safety"] != "ENGAGED":
        Set_Status("Can't start Teach - press ENGAGE first")
        return
    teach_ui["loaded"] = False
    Menu.control["request"] = "teach"
    Show_Page(Teach_Page)
    Set_Status("Teach mode: setting up the UR3...")


def On_Controller():
    if state["safety"] != "ENGAGED":
        Set_Status("Can't start Controller - press ENGAGE first")
        return

    Menu.control["request"] = "controller"

    Show_Page(Controller_Page)

    Set_Status("Controller mode: use the gamepad sticks")


tk.Label(Main_Page, text="Modes", font=(FONT, 12, "bold"), fg=FG, bg=BG).pack(pady=(0, 4))
Big_Button(Main_Page, "1. Exhibition", On_Exhibition).pack(pady=3)
Big_Button(Main_Page, "2. Teach", On_Teach).pack(pady=3)
Big_Button(Main_Page, "3. Controller", On_Controller).pack(pady=3)
Big_Button(Main_Page, "Person opens door", lambda: Menu.control.update(request="door")).pack(pady=(10, 3))
Big_Button(Main_Page, "Person presses E-STOP", lambda: Menu.control.update(request="wall_estop")).pack(pady=3)


# ------------------------------------SEQUENCE PAGE-----------------------------------------------------------
# Jumps to any part of the exhibition: the four set-up steps, any ball (click it in the rack to jump to the
# shot that pots it) or Run all. Potted balls are ticked off as the game goes on.

SETUP_STEPS = ["Assembly", "Rack", "Chalk", "Break"]
BALL_COLOURS = {1: "#f7c600", 2: "#1f4fd8", 3: "#d62828", 4: "#6a1b9a", 5: "#f77f00",
                6: "#1b8a3a", 7: "#7a1f1f", 8: "#111111", 9: "#f7c600"}
STRIPES = {9}
RACK_LAYOUT = {1: (0, -2), 2: (-0.5, -1), 3: (0.5, -1), 4: (-1, 0), 9: (0, 0),
               5: (1, 0), 6: (-0.5, 1), 7: (0.5, 1), 8: (0, 2)}
BALL_R = 20
GAP = 2 * BALL_R + 4
done_balls = set()


def Go_To(step, name):
    if state["safety"] != "ENGAGED":
        Set_Status(f"Can't start {name} - press ENGAGE first")
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
    canvas.create_oval(x - r * 0.5, y - r * 0.5, x + r * 0.5, y + r * 0.5, fill="white", outline="", tags=tag)
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
              activeforeground="white", relief="flat").grid(row=i // 2, column=i % 2, padx=3, pady=3)

RACK_W, RACK_H = 2 * GAP + 2 * BALL_R + 10, 4 * GAP * 0.87 + 2 * BALL_R + 10
Rack_Canvas = tk.Canvas(Sequence_Page, width=RACK_W, height=RACK_H, bg="#0b6e3b",
                        highlightthickness=2, highlightbackground="#5a3a1a", cursor="hand2")
Rack_Canvas.pack(pady=(6, 2))
Draw_Rack()
Big_Button(Sequence_Page, "Run all", Run_All).pack(pady=(6, 0))


# ------------------------------------TEACH PAGE--------------------------------------------------------------
# Six joint sliders and a speed slider for the UR3, SHOOT / Use example / Reset balls, the best
# configuration and the live checks with their suggested fixes.

SMALL = (FONT, 8)
Teach_Page = tk.Frame(Page_Area, bg=BG)
tk.Label(Teach_Page, text="Teach - UR3 straight shot", font=(FONT, 11, "bold"), fg=FG, bg=BG).pack()
Slider_Box = tk.Frame(Teach_Page, bg=BG)
Slider_Box.pack()

teach_ui = {"sync": False, "tick": 0}


def On_Slider(_=None):
    if teach_ui["sync"] or Menu.teach["q"] is None:
        return
    Menu.teach["q"] = np.radians([s.get() for s in Joint_Sliders])
    Menu.teach["result"] = ""


def Slider(row, name, low, high, command):
    value = tk.DoubleVar()
    tk.Label(Slider_Box, text=name, font=SMALL, fg=FG, bg=BG).grid(row=row, column=0, sticky="e", padx=(0, 4))
    s = tk.Scale(Slider_Box, from_=low, to=high, resolution=0.1, orient="horizontal", length=240, width=10,
                 sliderlength=16, showvalue=0, variable=value, command=command, bg=BG, fg=FG,
                 troughcolor="#3a3f4b", highlightthickness=0)
    s.grid(row=row, column=1, pady=3)
    tk.Label(Slider_Box, textvariable=value, font=SMALL, fg=FG, bg=BG, width=6, anchor="e").grid(row=row, column=2)
    return s


Joint_Sliders = [Slider(j, f"J{j + 1}", -180, 180, On_Slider) for j in range(6)]
Speed_Slider = Slider(6, "m/s", 0.3, 1.5, lambda v: Menu.teach.update(speed=float(v)))
Speed_Slider.set(0.7)


def Set_Sliders(q):
    teach_ui["sync"] = True
    for s, v in zip(Joint_Sliders, np.degrees(q)):
        s.set(round((v + 180) % 360 - 180, 1))
    teach_ui["sync"] = False
    On_Slider()


def Use_Example():
    if Menu.teach["ideal"] is not None:
        Set_Sliders(Menu.teach["ideal"])
        Menu.teach["q"] = np.array(Menu.teach["ideal"], dtype=float)


def Shoot():
    if state["safety"] != "ENGAGED":
        Set_Status("Press ENGAGE first")
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
    t = Menu.teach
    if not t.get("ready") or t["q"] is None:
        return
    if not teach_ui.get("loaded"):
        Set_Sliders(t["q"])
        teach_ui["loaded"] = True
    if t["ideal"] is not None:
        v = np.degrees(t["ideal"])
        Example_Label.config(text="Best (example): " + "  ".join(f"J{j + 1} {v[j]:.0f}" for j in range(3)) + "\n"
                                  "                " + "  ".join(f"J{j + 1} {v[j]:.0f}" for j in range(3, 6)))
    checks, _ = Menu.Teach_Checks(t["q"])
    for lab, item in zip(Check_Labels, checks + [None] * 5):
        if item is None:
            lab.config(text="")
        else:
            ok, text = item
            lab.config(text=("OK  " if ok else "!  ") + text, fg="#2ee05a" if ok else "#ffb000")
    if all(ok for ok, _ in checks):
        Result_Label.config(text=t["result"] or "Ready - press SHOOT", fg="#2ee05a" if not t["result"] else "#ffd166")
    else:
        Result_Label.config(text=t["result"] or "Not ideal for a straight shot yet - see the warnings", fg="#ffb000")


# ------------------------------------CONTROLLER PAGE---------------------------------------------------------
# Shows the UR3's joint angles live while it is driven with the gamepad.

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
    q = Menu.ur3_p.q

    if q is None:
        return

    q_deg = np.degrees(np.asarray(q, dtype=float))

    for label, value in zip(Joint_Labels, q_deg):
        label.config(text=f"{value:8.1f} deg")


# ------------------------------------SAFETY PANEL------------------------------------------------------------
# Always visible. E-STOP stops everything at once; RESET acknowledges the stop (only once any real e-stop
# is released); ENGAGE / DISENGAGE lets the robots move or holds them still. When the PLC is in charge, ENGAGE
# only works while the PLC shows green.

Safety_Panel = tk.Frame(window, bg=BG)
Safety_Panel.pack(side="bottom", pady=(4, 12), before=Page_Holder)


def PLC_Hold():
    if plc.Active("estop"):
        return "PLC E-STOP pressed"
    if PLC_MASTER and Pool_PLC.LIGHT_MEANING.get(plc.PLC_Light()) == "ESTOP":
        return "PLC in E-STOP (red light)"
    if plc.Comms_Fault():
        return "PLC COMMS FAULT - no reply from the PLC"
    return None


def On_EStop_Button():
    if state["safety"] == "ESTOP" and PLC_Hold():
        Set_Status(PLC_Hold() + " - clear it before resetting")
        return
    if state["safety"] == "ESTOP":
        plc_prev["local_estop"] = False
        state["safety"] = "DISENGAGED"
        Menu.Handle_Request("reset")
        Set_Status("E-stop reset. Nothing moves until you press ENGAGE.")
    else:
        plc_prev["local_estop"] = True
        state["safety"] = "ESTOP"
        Menu.Handle_Request("estop")
        Set_Status("E-STOP - all motion stopped")
    Update_Safety()


def On_Engage_Button(from_plc=False):
    if (PLC_MASTER and plc.Connected() and not from_plc and state["safety"] == "DISENGAGED"
            and Pool_PLC.LIGHT_MEANING.get(plc.PLC_Light()) != "ENGAGED"):
        Set_Status("The PLC isn't ready (not green) - press RESET on the PLC panel first")
        return
    if state["safety"] == "DISENGAGED":
        state["safety"] = "ENGAGED"
        Menu.Handle_Request("resume")
        Set_Status("Engaged - robots may move")
    elif state["safety"] == "ENGAGED":
        state["safety"] = "DISENGAGED"
        Menu.Handle_Request("pause")
        Set_Status("Disengaged - robots hold still")
    else:
        Set_Status("Reset the E-STOP first (press RESET)")
    Update_Safety()


EStop_Button = tk.Button(Safety_Panel, command=On_EStop_Button, width=14, height=2,
                         font=(FONT, 15, "bold"), fg="white", relief="raised", bd=4)
EStop_Button.pack(pady=(0, 6))
Engage_Button = tk.Button(Safety_Panel, command=On_Engage_Button, width=14, font=(FONT, 11, "bold"), relief="flat")
Engage_Button.pack()


def Update_Safety():
    s = state["safety"]
    if s == "ESTOP":
        EStop_Button.config(text="RESET", bg="#8b0000", activebackground="#a00000")
        Engage_Button.config(text="ENGAGE", bg="#444444", fg="#888888", state="disabled")
        Safety_Label.config(text="E-STOP", fg=LAMP_ON["red"])
        Show_Light("red")
    elif s == "DISENGAGED":
        EStop_Button.config(text="E-STOP", bg="#d62828", activebackground="#ff4444")
        Engage_Button.config(text="ENGAGE", bg="#c98a00", fg="white", state="normal")
        Safety_Label.config(text="DISENGAGED", fg=LAMP_ON["amber"])
        Show_Light("amber")
    else:
        EStop_Button.config(text="E-STOP", bg="#d62828", activebackground="#ff4444")
        Engage_Button.config(text="DISENGAGE", bg="#2e7d32", fg="white", state="normal")
        Safety_Label.config(text="ENGAGED", fg=LAMP_ON["green"])
        Show_Light("green")


# ------------------------------------START-------------------------------------------------------------------
# Opens on the main page, DISENGAGED, so nothing can move until ENGAGE is pressed.

Show_Page(Main_Page)
Update_Safety()
Set_Status("Ready - press ENGAGE, then pick a mode")


last = {"text": Menu.control.get("text")}


# ------------------------------------REAL PLC LINK-----------------------------------------------------------
# Every update the real PLC's inputs and stack light are read. A pressed real e-stop, a red light or a lost
# connection stops the cell; a change of the PLC's light to yellow or green is copied to the panel
# (acknowledge or arm). Only changes are followed, so the panel's own buttons still work in between.

def Follow_PLC_Light():
    colour = plc.PLC_Light()
    before, plc_prev["light"] = plc_prev["light"], colour
    if colour is None or colour == before:
        return
    want = Pool_PLC.LIGHT_MEANING.get(colour)
    if want != "ESTOP" and state["safety"] == "ESTOP" and not plc_prev["local_estop"]:
        state["safety"] = "DISENGAGED"
        Menu.Handle_Request("reset")
        Update_Safety()
        Menu.Set_Status("PLC reset - e-stop acknowledged")
    if want == "ENGAGED" and state["safety"] == "DISENGAGED" and before is not None:
        On_Engage_Button(from_plc=True)
        Menu.Set_Status("PLC green - engaged, robots may move")
    elif want == "DISENGAGED" and state["safety"] == "ENGAGED":
        On_Engage_Button(from_plc=True)
        Menu.Set_Status("PLC yellow - disengaged, robots hold still")


def PLC_Check():
    hold = PLC_Hold()
    if hold and state["safety"] != "ESTOP":
        state["safety"] = "ESTOP"
        Menu.Handle_Request("estop")
        Update_Safety()
        Menu.Set_Status(hold + " - all motion stopped. Clear it, then press RESET.")
    pressed = plc.Active("reset")
    if pressed and not plc_prev["reset"] and state["safety"] == "ESTOP":
        On_EStop_Button()
    plc_prev["reset"] = pressed
    if PLC_MASTER:
        Follow_PLC_Light()
    text = plc.Status_Text()
    if PLC_MASTER and plc.Connected():
        text += f"  |  PLC light: {plc.PLC_Light() or 'off'}"
    if text != plc_prev["text"]:
        plc_prev["text"] = text
        PLC_Label.config(text=text, fg="#2ee05a" if plc.Connected() else
                         ("#ff2a2a" if plc.Comms_Fault() else "#8a8a8a"))


# ------------------------------------MAIN LOOP---------------------------------------------------------------
# Runs about every millisecond: one simulation update, the PLC check, then the panel is refreshed (safety
# state, light, status, teach checks, joint readout, potted balls).

def Tick():
    if not Menu.Step_Once():
        plc.Stop()
        window.after(400, window.destroy)
        return
    PLC_Check()
    Menu.env.step(Menu.DT)
    sim = Menu.control["state"]
    if sim == "ESTOPPED" and state["safety"] != "ESTOP":
        state["safety"] = "ESTOP"
        Update_Safety()
    if current_page["frame"] is Teach_Page:
        teach_ui["tick"] += 1
        if teach_ui["tick"] % 10 == 0:
            Teach_Update()
            Fit_Window()
    if current_page["frame"] is Controller_Page:
        Update_Joint_Labels()
    if state["safety"] == "ENGAGED":
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
