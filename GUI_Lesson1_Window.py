"""
GUI_Lesson1_Window.py - LESSON 1: your own pop-up control window (tkinter).

Run it on its own:   python GUI_Lesson1_Window.py
It does NOT touch the robots yet. It only teaches the 5 building blocks you need
for the real control panel (lesson 2 plugs this into Pool_Robot_Menu.py).

    1. the WINDOW        Tk()                 - the pop-up itself
    2. WIDGETS           Label, Button, Canvas - the things you see in it
    3. LAYOUT            .grid(row, column)   - where each thing goes
    4. CALLBACKS         command=Some_Function - what happens when you click
    5. THE TIMER LOOP    window.after(ms, fn) - code that runs every few ms
                                               (this is where the robot sim will step)

tkinter comes with Python - nothing to pip install.
"""
import tkinter as tk


# ------------------------------------1. THE WINDOW-----------------------------------------------------
# Tk() makes the window. Everything else is put INSIDE it.

window = tk.Tk()
window.title("Pool Robot Control")          # text in the window's title bar
window.geometry("250x700+0+0")            # width x height + x + y  (where it pops up on screen)
window.attributes("-topmost", True)         # stay on top of the Swift browser tab
window.configure(bg="#4f6d7a")              # background colour (any hex colour)


# ------------------------------------2. WIDGETS-----------------------------------------------------
# A widget = one thing in the window. You create it with the window as its PARENT
# (the first argument), then place it with .grid() (section 3).

Title_Label = tk.Label(window, text="9-Ball Workcell", font=("Segoe UI", 14, "bold"),
                       fg="white", bg="#4f6d7a")

Status_Label = tk.Label(window, text="State: IDLE", font=("Segoe UI", 10),
                        fg="#cccccc", bg="#20232a")

# A Canvas is a small drawing area - we draw a round "lamp" on it.
State_Canvas = tk.Canvas(window, width=40, height=40, bg="#20232a", highlightthickness=0)
State = State_Canvas.create_oval(10, 10, 60, 60, fill="orange")     # x1, y1, x2, y2 of the circle


# ------------------------------------4. CALLBACKS (what the buttons do)----------------------------------
# A callback is just a normal function. You give its NAME to a button (no brackets!),
# and tkinter calls it when the button is clicked.

state = {"name": "IDLE"}                    # a dictionary so every function can change it


def Set_State(new_state, colour):
    """Change the state, the status text and the lamp colour."""
    state["name"] = new_state
    Status_Label.config(text=f"State: {new_state}")          # .config() changes a widget after it's made
    Lamp_Canvas.itemconfig(Lamp, fill=colour)                # change the circle's colour


def On_Exhibition():
    Set_State("RUNNING", "green")


def On_EStop():
    Set_State("ESTOPPED", "red")


def On_Reset():
    if state["name"] == "ESTOPPED":          # only does something after an e-stop
        Set_State("RESET", "orange")


Exhibition_Button = tk.Button(window, text="1. Exhibition", width=14, command=On_Exhibition)
EStop_Button = tk.Button(window, text="E-STOP", width=14, height=2, command=On_EStop,
                         bg="#d62828", fg="white", activebackground="#ff4444", font=("Segoe UI", 11, "bold"))
Reset_Button = tk.Button(window, text="RESET", width=14, command=On_Reset)
Quit_Button = tk.Button(window, text="Quit", width=14, command=window.destroy)   # destroy = close the window


# ------------------------------------3. LAYOUT-----------------------------------------------------
# .grid() puts widgets in rows and columns, like a table. padx / pady = space around it.
# columnspan=2 lets a widget stretch across 2 columns.

Title_Label.grid(row=0, column=0, columnspan=2, pady=(10, 4))
State_Canvas.grid(row=1, column=0, padx=10)
Status_Label.grid(row=1, column=1, sticky="w")              # sticky="w" = stick to the left (west)
Exhibition_Button.grid(row=2, column=0, columnspan=2, pady=4)
EStop_Button.grid(row=3, column=0, columnspan=2, pady=4)
Reset_Button.grid(row=4, column=0, columnspan=2, pady=4)
Quit_Button.grid(row=5, column=0, columnspan=2, pady=4)


# ------------------------------------5. THE TIMER LOOP-----------------------------------------------------
# A GUI can't use a normal  while True:  loop - the window would freeze, because the
# window needs time to redraw itself and notice clicks.
# Instead, window.after(20, Tick) says "run Tick again in 20 ms". Tick re-books itself
# every time, so it runs forever - and the window stays responsive in between.
# In lesson 2, Tick is where we advance the robots one frame and call env.step().

frame = {"count": 0}


def Tick():
    if state["name"] == "RUNNING":          # only "moves" while running - just like the robots will
        frame["count"] += 1
        Status_Label.config(text=f"State: RUNNING   frame {frame['count']}")
    window.after(20, Tick)                  # book the next tick in 20 ms  (= 50 times a second)


Tick()                                      # start the timer loop
window.mainloop()                           # hand control to tkinter: shows the window, waits for clicks
