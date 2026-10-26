import tkinter as tk

#The window
window = tk.Tk()
window.title("Cue-T Interface")
window.geometry("250x700+0+0")
window.attributes("-topmost", True)
window.configure(bg="#4f6d7a")

#The widgets
Title_Label = tk.Label(window, text="9-Ball System", font=("Segoe UI", 14, "bold"), fg="white", bg="#4f6d7a")
Status_Label = tk.Label(window, text="State: IDLE", font=("Segoe UI", 10), fg="#cccccc", bg="#4af6d74")

State_Canvas = tk.Canvas(window, width=40, height=40, bg="#4af6d74", highlightthickness=0)
State = State_Canvas.create_oval(5, 5, 35, 35, fill="orange")

#The Layout
Title_Label.grid(row=0, column=0, columnspan=2, pady=(10, 4))
State_Canvas.grid(row=1, column=0, padx=10)
Status_Label.grid(row=1, column=1, sticky="w")              # sticky="w" = stick to the left (west)
Exhibition_Button.grid(row=2, column=0, columnspan=2, pady=4)
EStop_Button.grid(row=3, column=0, columnspan=2, pady=4)
Reset_Button.grid(row=4, column=0, columnspan=2, pady=4)
Quit_Button.grid(row=5, column=0, columnspan=2, pady=4)