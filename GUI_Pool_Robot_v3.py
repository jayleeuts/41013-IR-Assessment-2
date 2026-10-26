import math
import tkinter as tk

BG = "#20232a"
FG = "#e6e6e6"
FONT = "Segoe UI"


# ------------------------------------THE WINDOW-----------------------------------------------------

window = tk.Tk()
window.title("Cue-T Interface")
window.geometry("380x720+50+50")
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


def Show_Light(colour):
    for name, lamp in Lamps.items():
        Light_Canvas.itemconfig(lamp, fill=LAMP_ON[name] if name == colour else LAMP_OFF[name])


def Set_Status(text):
    Status_Label.config(text=text)
    print(text)


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
    if page is Main_Page:
        Back_Button.pack_forget()
    else:
        Back_Button.pack(side="left", padx=10)


def Big_Button(parent, text, command):
    return tk.Button(parent, text=text, command=command, width=22, height=1,
                     font=(FONT, 11), bg="#3a3f4b", fg="white", activebackground="#50586a",
                     activeforeground="white", relief="flat")


# ---- main page: the three modes ----

def On_Exhibition():
    Show_Page(Sequence_Page)
    Set_Status("Exhibition: pick a step, or Run all")


def On_Teach():
    state["mode"] = "teach"
    Set_Status("Teach mode - coming next")


def On_Controller():
    state["mode"] = "controller"
    Set_Status("Controller mode - coming next")


tk.Label(Main_Page, text="Modes", font=(FONT, 12, "bold"), fg=FG, bg=BG).pack(pady=(0, 4))
Big_Button(Main_Page, "1. Exhibition", On_Exhibition).pack(pady=3)
Big_Button(Main_Page, "2. Teach", On_Teach).pack(pady=3)
Big_Button(Main_Page, "3. Controller", On_Controller).pack(pady=3)


# ---- sequence page: the exhibition steps ----

SETUP_STEPS = ["Assembly", "Rack", "Chalk"]
BALL_COLOURS = {1: "#f7c600", 2: "#1f4fd8", 3: "#d62828", 4: "#6a1b9a", 5: "#f77f00",
                6: "#1b8a3a", 7: "#7a1f1f", 8: "#111111", 9: "#f7c600"}
STRIPES = {9}
RACK_LAYOUT = {1: (0, -2), 2: (-0.5, -1), 3: (0.5, -1), 4: (-1, 0), 9: (0, 0),
               5: (1, 0), 6: (-0.5, 1), 7: (0.5, 1), 8: (0, 2)}
BALL_R = 20
GAP = 2 * BALL_R + 4
done_balls = set()


def Run_Step(number, name):
    if state["safety"] != "ARMED":
        Set_Status(f"Can't start {name} - press ARM first")
        return False
    state["mode"] = name
    Set_Status(f"Running step {number}. {name}")
    return True


def Run_All():
    Run_Step(0, "the whole sequence")


def On_Ball(n):
    if Run_Step(len(SETUP_STEPS) + n, f"{n} Ball"):
        done_balls.add(n)
        Draw_Rack()


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


def Reset_Rack():
    done_balls.clear()
    Draw_Rack()
    Set_Status("Rack display reset")


tk.Label(Sequence_Page, text="Sequence", font=(FONT, 12, "bold"), fg=FG, bg=BG).pack(pady=(0, 4))
Step_Row = tk.Frame(Sequence_Page, bg=BG)
Step_Row.pack()
for i, name in enumerate(SETUP_STEPS):
    number = i + 1
    tk.Button(Step_Row, text=f"{number}. {name}", width=10,
              command=lambda n=number, s=name: Run_Step(n, s),
              font=(FONT, 10), bg="#3a3f4b", fg="white", activebackground="#50586a",
              activeforeground="white", relief="flat").grid(row=0, column=i, padx=3, pady=3)

RACK_W, RACK_H = 2 * GAP + 2 * BALL_R + 10, 4 * GAP * 0.87 + 2 * BALL_R + 10
Rack_Canvas = tk.Canvas(Sequence_Page, width=RACK_W, height=RACK_H, bg="#0b6e3b",
                        highlightthickness=2, highlightbackground="#5a3a1a", cursor="hand2")
Rack_Canvas.pack(pady=(6, 2))
Draw_Rack()
Bottom_Row = tk.Frame(Sequence_Page, bg=BG)
Bottom_Row.pack(pady=(6, 0))
for col, text, cmd in ((0, "Run all", Run_All), (1, "Reset rack", Reset_Rack)):
    tk.Button(Bottom_Row, text=text, width=10, command=cmd, font=(FONT, 10), bg="#3a3f4b", fg="white",
              activebackground="#50586a", activeforeground="white", relief="flat").grid(row=0, column=col, padx=3)


# ------------------------------------SAFETY PANEL (always visible, middle of the window)-------------------

Safety_Panel = tk.Frame(window, bg=BG)
Safety_Panel.pack(expand=True)


def On_EStop_Button():
    if state["safety"] == "ESTOP":
        state["safety"] = "DISENGAGED"
        Set_Status("E-stop released. Nothing moves until you press ARM.")
    else:
        state["safety"] = "ESTOP"
        Set_Status("E-STOP - all motion stopped")
    Update_Safety()


def On_Arm_Button():
    if state["safety"] == "DISENGAGED":
        state["safety"] = "ARMED"
        Set_Status("Armed - robots may move")
    elif state["safety"] == "ARMED":
        state["safety"] = "DISENGAGED"
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
Quit_Button = tk.Button(Bottom_Bar, text="Quit", width=10, command=window.destroy)
Quit_Button.pack(side="right", padx=10)


# ------------------------------------START-----------------------------------------------------

Show_Page(Main_Page)
Update_Safety()
Set_Status("Ready - press ARM, then pick a mode")


def Tick():
    window.after(20, Tick)


Tick()
window.mainloop()
