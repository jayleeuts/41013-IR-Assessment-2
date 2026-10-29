import sys
import json
from pathlib import Path
import swift
import numpy as np
import spatialgeometry as sg
from math import pi
from spatialmath import SE3
import roboticstoolbox as rtb
from Gamepad_Controller import GamepadController

from Doosan_Robot import Doosan_Robot
from ir_support_extra_robots.robots import Turtlebot3Waffle
from Pool_Parts import Part_Path, Part_Mesh
from Pool_Objects import Cue_Ball, Cue_Stick
from Pool_Motion import Holder_Tool, Solve_IK, Cue_Pose, Punch_Stroke_Profile, _Rotation_From_Vector
from Pool_Scene import (TRACK_NAME, TRACK_POSE, CHANNEL_Y, INNER_X, INNER_Y, RIM_TOP_Z,
                        TABLE_Z, FELT_Z, PLAY_AREA, GROOVE_HEIGHT, POCKETS, POCKET_RADIUS,
                        POCKET_FLOOR_Z, Turtle_Pose, Robot_Base_On, Drive_Path)
from Jbot_Racker import Jbot_Mobile, Ball_Rack, Rack_Slots, Tool_Down, Jbot_IK
from Pool_Players import UR3_Player, Meca_Player
from Meca_Jumper import Short_Stroke_Profile, CARRY_Q as MECA_CARRY_Q
from Pool_Game import GAME
from Pool_Scene import _Loop, _Loop_S, _Loop_Point
from Pool_Trick_Shots import (Pool_Ball, Table_Step, Break_Script, LAYOUT, ORANGE, ORANGE_POCKET,
                              POCKET_NAMES, MU_ROLL, SHOTS, Shot_Poses)
from Pool_Station import (Station, Q_REST, Chalk_Present_Pose, Assembly_Pose, HANDLE_PIN as PIN, N_TURNS, LINE_UP,
                          PRESENT, ASM_PRESENT)
import test_meca500_shaft_grip as SG
import Pool_Arm as PA


# ------------------------------------CONSTANTS-----------------------------------------------------

DT = 0.02
SPEED_UP = 5                          # playback speed: robots AND balls move this many times faster on screen
PHYSICS_SUBSTEPS = 5                  # ball physics steps per frame (fast balls need small steps)
JOINT_SPEED = 2.0                     # rad/s cap for joint-space moves (both robots)

# --- Doosan ---
HANDLE_PIN = 0.033
GRIP_FROM_BUTT = 0.55
CARRY_GRIP_POINT = (0.40, 0.0, 0.65)  # cue held upright in front of the robot (robot-base frame)
CARRY_AIM_DEG = 180.0
DOOSAN_LOOP = (1.72, CHANNEL_Y)       # outer side of the side lanes (corners clear the outer rim)
DOOSAN_WAIT = (-1.72, 0.45, -90.0)    # waits on the LEFT lane - Jbot never drives there
BREAK_TURTLE = (-1.72, -0.40, None)   # breaks from here
CUE_BALL_SPOT = (-0.60, 0.20)         # behind the head string
BREAK_ELEV_DEG, BREAK_ROLL_DEG = 8.0, -90.0
BREAK_TIP_SPEED = 1.6                 # m/s at contact
ADDRESS_GAP = 0.02
APPROACH = 0.15
CUE_TO_BALL = 1.4

# --- Jbot ---
JBOT_SPEED = 0.8                      # turtlebot driving speed (m/s)
RACK_SPEED_UP = 5                          # playback speed: robots AND balls move this many times faster on screen
TRAY_PARK = (INNER_X, 0.0, -90.0)     # Jbot fills / slides the rack from here
RACK_START_APEX = 0.83                # rack starts near the foot cushion (apex ball x)
FOOT_SPOT = 0.534                     # rack apex ends here (1/4 of the table from the foot)
CARRY_LOCAL = SE3(0.25, 0.0, 0.20) * SE3.Rx(pi)   # cup above the turtlebot, pointing down
ABOVE = RIM_TOP_Z + 0.10              # cup height when moving over a pocket

# where Jbot parks for each pocket (x, y, yaw) - checked reachable.
# The head-end pockets (2, 3) are reached from the long lanes, so Jbot never
# uses the left lane - that is where the Doosan waits.
# Turtle always faces along its lane (it reverses rather than turning mid-lane).
POCKET_PARKS = {0: (INNER_X, 0.501, -90.0), 1: (INNER_X, -0.499, -90.0),
                2: (-1.096, INNER_Y, 0.0), 3: (-1.096, -INNER_Y, 180.0),
                4: (0.004, INNER_Y, 0.0), 5: (0.004, -INNER_Y, 180.0)}

# which balls start in which pocket (bottom first), and the order Jbot collects them.
# Collected in this order into rack slots apex -> back, so 1 is at the apex and 9 in the middle.
POCKET_STACKS = {0: [2, 1], 1: [4, 3], 4: [9], 5: [5], 2: [7, 6], 3: [8]}
PICK_ORDER = [1, 2, 3, 4, 9, 5, 6, 7, 8]

BALL_COLOURS = {1: (1.0, 0.85, 0.0), 2: (0.0, 0.2, 0.8), 3: (0.85, 0.0, 0.0), 4: (0.4, 0.0, 0.5),
                5: (1.0, 0.45, 0.0), 6: (0.0, 0.5, 0.2), 7: (0.5, 0.05, 0.05), 8: (0.05, 0.05, 0.05),
                9: (1.0, 0.95, 0.6)}


# ------------------------------------SCENE-----------------------------------------------------

HEADLESS = "--make-snapshots" in sys.argv          # build the time-skip snapshots without opening Swift


class _No_Swift(swift.Swift):
    """Stand-in for Swift when making snapshots (nothing is shown, no browser opens)."""
    def __init__(self): pass
    def launch(self, **kw): pass
    def add(self, *a, **kw): pass
    def step(self, *a, **kw): pass
    def set_camera_pose(self, *a, **kw): pass
    def close(self): pass


env = _No_Swift() if HEADLESS else swift.Swift()
env.launch(realtime=True)
CAMERA_OVERVIEW = ([3.6, -4.6, 3.6], [0.4, -0.6, 0.8])           # the whole cell
env.set_camera_pose(*CAMERA_OVERVIEW)

env.add(Part_Mesh(TRACK_NAME, pose=TRACK_POSE))
env.add(Part_Mesh("Pool_Table", pose=SE3(0, 0, TABLE_Z)))

# Doosan on its turtlebot, holding the assembled cue in the holder
d_turtle = Turtlebot3Waffle()
d_xyyaw = DOOSAN_WAIT
d_turtle.base = Turtle_Pose(*d_xyyaw)
d_turtle.add_to_env(env)
doosan = Doosan_Robot()
doosan.base = Robot_Base_On(d_turtle.base)
doosan.add_to_env(env)
holder = Part_Mesh("Cue_Holder")
env.add(holder)
cue = Cue_Stick(str(Part_Path("Cuestick_Shaft")), str(Part_Path("Cuestick_Handle")), handle_insert=HANDLE_PIN)
cue.Add_To_Env(env)
D_TOOL = Holder_Tool(GROOVE_HEIGHT, cue.length - GRIP_FROM_BUTT)

# Jbot on its turtlebot
jbot = Jbot_Mobile(TRAY_PARK)
jbot.Add_To_Env(env)

# the opponents: UR3 (same holder + cue) and the Meca500 (jump cue) - both wait at the head end
UR3_HOME = (-1.72, -0.45, 90.0)
MECA_HOME = (-INNER_X, 0.0, 90.0)
ur3_p = UR3_Player(UR3_HOME)
ur3_p.Add_To_Env(env)
meca_p = Meca_Player(MECA_HOME)
meca_p.Add_To_Env(env)

# the service Meca500 at the head end: assembles the cue and chalks every cue (Pool_Station.py)
station = Station()
station.Add_To_Env(env)
cue_state = {"shaft": None}            # None = cue assembled; otherwise the shaft's own world pose

# balls: cue ball (mesh with markings) + 9 coloured object balls
cue_ball_mesh = Cue_Ball(str(Part_Path("Cue_Ball")))
R = cue_ball_mesh.radius
cue_ball_mesh.Add_To_Env(env)
ball_meshes = {0: cue_ball_mesh}
for n, colour in BALL_COLOURS.items():          # numbered balls (Pool_Parts/1Ball.dae ... 9Ball.dae)
    try:
        ball_meshes[n] = Cue_Ball(str(Part_Path(f"{n}Ball")))
        ball_meshes[n].Add_To_Env(env)
    except ValueError:                           # mesh missing - plain coloured sphere instead
        ball_meshes[n] = sg.Sphere(radius=R, color=[*colour, 1.0])
        env.add(ball_meshes[n])
physics = {n: Pool_Ball([0, 0, FELT_Z + R], R, mu_roll=MU_ROLL, jaws=True, bounds=PLAY_AREA, pockets=POCKETS,
                        pocket_radius=POCKET_RADIUS, pocket_floor_z=POCKET_FLOOR_Z + R)
           for n in ball_meshes}
BALL_LIST = [physics[n] for n in sorted(physics)]          # fixed order -> same result every run

rack = Ball_Rack(R, FELT_Z)
rack.Add_To_Env(env)

# ------------------------------------CONTROL: MENU + E-STOP-----------------------------------------------------
#
#   RUNNING  --E-STOP-->  ESTOPPED  --RESET-->  RESET  --RESUME-->  RUNNING
#                                                 \--(pick a mode)--> starts that mode from a clean scene
#
#   * E-STOP freezes every robot and ball on the very next frame (the running
#     sequence is simply not advanced any more).
#   * Nothing restarts by itself: RESET only acknowledges the stop, RESUME is a
#     separate, deliberate action. RESUME is refused until RESET has been pressed.
#   * While stopped the main loop keeps calling env.step() so Swift stays live -
#     no blocking busy-wait.

control = {"state": "IDLE", "sequence": None, "mode": None, "request": None}

class _Text:
    desc = ""


status = _Text()                       # status text (the control panel shows it)


def Set_Status(text):
    control["text"] = text
    status.desc = f"[{control['state']}] {text}"
    if not HEADLESS:
        print(f"[{control['state']}] {text}")


# stack light beside the table: green = running, amber = idle / reset, red = e-stop
LIGHT_XY = (1.98, -1.30)
env.add(sg.Cylinder(radius=0.015, length=1.2, color=[0.3, 0.3, 0.3, 1.0], pose=SE3(*LIGHT_XY, 0.6)))
_lamp_z = {"red": 1.32, "amber": 1.26, "green": 1.20}
# colours must be floats (1.0 not 1): an all-integer colour makes spatialgeometry store
# the opacity as a numpy int64, which Swift cannot send to the browser (it crashes)
_lamp_rgb = {"red": (1.0, 0.0, 0.0), "amber": (1.0, 0.6, 0.0), "green": (0.0, 0.9, 0.0)}
_lamps_dim, _lamps_lit = {}, {}
for name, z in _lamp_z.items():
    r, g, b = _lamp_rgb[name]
    _lamps_dim[name] = sg.Cylinder(radius=0.035, length=0.05, color=[0.25 * r + 0.1, 0.25 * g + 0.1, 0.25 * b + 0.1, 1.0],
                                   pose=SE3(*LIGHT_XY, z))
    _lamps_lit[name] = sg.Cylinder(radius=0.038, length=0.052, color=[r, g, b, 1.0], pose=SE3(*LIGHT_XY, -5))
    env.add(_lamps_dim[name])
    env.add(_lamps_lit[name])


def Set_Light(lit):
    """Light one lamp of the stack light (the others are hidden below the floor)."""
    for name, lamp in _lamps_lit.items():
        lamp.T = SE3(*LIGHT_XY, _lamp_z[name] if name == lit else -5).A


# ------------------------------------ROOM, DOOR + DOOR SENSOR (safety interlock)-------------------------
#
# The whole workcell is inside a room. The only way in is a door with a door sensor on its
# frame. Door opens -> the sensor trips -> every robot and ball PAUSES (same freeze as the
# E-STOP). Door shut again -> operations RESUME on their own (no reset needed - it is an
# interlock, not an emergency stop). A person (from the toolbox's ir_support_extra_parts)
# walks up and opens the door a few seconds into the Exhibition; the "Person opens the
# door" button makes it happen again whenever you like.

ROOM_X, ROOM_Y, ROOM_H, WALL_T = 2.6, 1.9, 2.4, 0.05
DOOR_HINGE, DOOR_W, DOOR_H = (1.0, -ROOM_Y), 0.8, 2.05
DOOR_DEMO_AT = 3.0                     # s after the Exhibition starts that the person arrives
WALL_COLOUR = [0.85, 0.85, 0.8, 0.25]                          # see-through, so the camera can look in
for cx, cy, sx, sy in [(-ROOM_X, 0, WALL_T, 2 * ROOM_Y), (ROOM_X, 0, WALL_T, 2 * ROOM_Y),
                       (0, ROOM_Y, 2 * ROOM_X, WALL_T),
                       ((-ROOM_X + DOOR_HINGE[0]) / 2, -ROOM_Y, DOOR_HINGE[0] + ROOM_X, WALL_T),       # wall left of the door
                       ((DOOR_HINGE[0] + DOOR_W + ROOM_X) / 2, -ROOM_Y, ROOM_X - DOOR_HINGE[0] - DOOR_W, WALL_T)]:
    env.add(sg.Cuboid(scale=[sx, sy, ROOM_H], color=WALL_COLOUR, pose=SE3(cx, cy, ROOM_H / 2)))
env.add(sg.Cuboid(scale=[DOOR_W, WALL_T, ROOM_H - DOOR_H], color=WALL_COLOUR,                  # above the door
                  pose=SE3(DOOR_HINGE[0] + DOOR_W / 2, -ROOM_Y, (ROOM_H + DOOR_H) / 2)))
door = sg.Cuboid(scale=[DOOR_W, 0.04, DOOR_H], color=[0.55, 0.35, 0.2, 1.0])
env.add(door)
sensor_ok = sg.Cuboid(scale=[0.06, 0.03, 0.06], color=[0.0, 0.9, 0.0, 1.0])     # green = door shut
sensor_trip = sg.Cuboid(scale=[0.06, 0.03, 0.06], color=[1.0, 0.0, 0.0, 1.0])   # red = door open
env.add(sensor_ok)
env.add(sensor_trip)
try:   # person, sensor and sign meshes from the toolbox (ir_support_extra_parts) - copies are in Pool_Parts
    try:
        Part_Mesh("personMaleConstruction")
        Toolbox_Part = Part_Mesh
    except Exception:
        from ir_support_extra_parts.parts import part_mesh as Toolbox_Part
    person = Toolbox_Part("personMaleConstruction")
    env.add(Toolbox_Part("ProximitySensor", pose=SE3(DOOR_HINGE[0] + DOOR_W + 0.12, -ROOM_Y + 0.08, DOOR_H - 0.38)))
    env.add(Toolbox_Part("CautionSign", pose=SE3(DOOR_HINGE[0] - 0.4, -ROOM_Y - 0.15, 0.0)))
except Exception:                                              # stand-in person made of simple shapes
    person = sg.Cylinder(radius=0.18, length=1.75, color=[0.95, 0.6, 0.1, 1.0])
door_state = {"angle": 0.0, "script": [], "t": 0.0, "demo_done": False}
PERSON_FAR, PERSON_AT_DOOR, PERSON_IN_DOOR = (1.95, -3.4), (1.95, -2.65), (1.4, -2.0)


def Place_Door():
    a = np.radians(door_state["angle"])                       # swings OUT of the room (towards -y)
    door.T = (SE3(DOOR_HINGE[0], DOOR_HINGE[1] - 0.03, DOOR_H / 2) * SE3.Rz(-a) * SE3(DOOR_W / 2, 0, 0)).A
    tripped = door_state["angle"] > DOOR_SWITCH_DEG
    at = SE3(DOOR_HINGE[0] + DOOR_W + 0.12, -ROOM_Y - 0.05, DOOR_H - 0.45)
    sensor_trip.T = (at if tripped else SE3(0, 0, -5)).A
    sensor_ok.T = (SE3(0, 0, -5) if tripped else at).A


def Place_Person(xy, yaw_deg=90.0):
    z = 0.875 if isinstance(person, sg.Cylinder) else 0.0
    person.T = (SE3(xy[0], xy[1], z) * SE3.Rz(np.radians(yaw_deg))).A


def Door_Visit():
    """Person walks up, opens the door, steps into the doorway, steps back, shuts it, walks off."""
    def walk(a, b, secs):
        n = int(secs / DT)
        return [("p", tuple(np.array(a) + (np.array(b) - np.array(a)) * k / n)) for k in range(1, n + 1)]
    def swing(a, b, secs):
        n = int(secs / DT)
        return [("d", a + (b - a) * k / n) for k in range(1, n + 1)]
    return (walk(PERSON_FAR, PERSON_AT_DOOR, 1.5) + swing(0, 85, 1.0) + walk(PERSON_AT_DOOR, PERSON_IN_DOOR, 0.8)
            + [("w", None)] * int(1.5 / DT) + walk(PERSON_IN_DOOR, PERSON_AT_DOOR, 0.8) + swing(85, 0, 1.0)
            + walk(PERSON_AT_DOOR, PERSON_FAR, 1.5))


def Door_Step():
    """Advance the person/door animation one screen update (always real time, not sped up)."""
    if door_state["script"]:
        kind, value = door_state["script"].pop(0)
        if kind == "p":
            Place_Person(value)
        elif kind == "d":
            door_state["angle"] = value
        elif kind == "e":                                     # the person hits the wall e-stop
            Handle_Request("estop")
            Set_Status("WALL E-STOP PRESSED (outside the door) - all motion stopped. Press Reset to acknowledge.")
    Place_Door()


DOOR_SWITCH_DEG = 3.0                  # door sensor trips once the door is open more than this
Place_Door()
Place_Person(PERSON_FAR)
env.add(person)


# ------------------------------------WALL E-STOP OUTSIDE THE DOOR-----------------------------------------------------
# The toolbox's wall-mounted emergency stop (ir_support_extra_parts "emergencyStopWallMounted",
# copy in Pool_Parts), on the outside of the front wall next to the door. It is the SAME e-stop
# as the control panel's: a person can walk up and hit it ("Person presses the E-STOP by the
# door" on the panel), and either one stops the whole cell (the stack light turns red).

WALL_ESTOP_XY = (0.45, -ROOM_Y - WALL_T / 2)            # outside face of the front wall, left of the door
WALL_ESTOP_Z = 1.30
PERSON_AT_ESTOP = (0.45, -2.35)
try:
    wall_estop = Part_Mesh("emergencyStopWallMounted", pose=SE3(*WALL_ESTOP_XY, WALL_ESTOP_Z) * SE3.Rz(-pi / 2))
except Exception:                                              # stand-in: red mushroom on a yellow box
    wall_estop = sg.Cuboid(scale=[0.12, 0.08, 0.12], color=[1.0, 0.85, 0.0, 1.0],
                           pose=SE3(WALL_ESTOP_XY[0], WALL_ESTOP_XY[1] - 0.04, WALL_ESTOP_Z))
env.add(wall_estop)


def Wall_EStop_Visit():
    """Person walks up to the wall e-stop, hits it, then steps back."""
    def walk(a, b, secs):
        n = int(secs / DT)
        return [("p", tuple(np.array(a) + (np.array(b) - np.array(a)) * k / n)) for k in range(1, n + 1)]
    return (walk(PERSON_FAR, PERSON_AT_ESTOP, 1.8) + [("w", None)] * int(0.4 / DT) + [("e", None)]
            + [("w", None)] * int(1.0 / DT) + walk(PERSON_AT_ESTOP, PERSON_FAR, 1.8))


# ------------------------------------STATE + DRAWING-----------------------------------------------------

state = {"physics": False, "script": [], "fast": False}
held = {"what": None, "T_rel": None}          # what Jbot's cup is holding


def Doosan_Carry_Pose(base):
    R_up = Cue_Pose([0, 0, 0], CARRY_AIM_DEG, -90.0).R
    grip = base * SE3(*CARRY_GRIP_POINT)
    Rw = base.R @ R_up
    return SE3.Rt(Rw, grip.t + (cue.length - GRIP_FROM_BUTT) * Rw[:, 2])


def Draw(dq=None, jq=None):
    """Put everything where it should be this frame (Swift shows it on the next env.step).
    dq / jq = new joint angles for the Doosan / Jbot (None = keep)."""
    d_turtle.base = Turtle_Pose(*d_xyyaw)
    doosan.base = Robot_Base_On(d_turtle.base)
    if dq is not None:
        doosan.q = dq
    T_flange = doosan.fkine(doosan.q)
    holder.T = T_flange.A
    T_tip = T_flange * D_TOOL
    cue.Place(T_tip, shaft_offset=None if cue_state["shaft"] is None else T_tip.inv() * cue_state["shaft"])
    station.Draw()

    jbot.Set_Base()
    if jq is not None:
        jbot.robot.q = jq
    jbot.Draw_Tool()
    for p in (ur3_p, meca_p):
        p.Set_Base()
        p.Draw()
    if held["what"] is not None:                                   # carried object follows the cup
        T_obj = jbot.Tool_Pose() * held["T_rel"]
        if held["what"] == "rack":
            rack.Place(T_obj)
            for n, T_rel in held.get("balls", {}).items():
                physics[n].pos = (T_obj * T_rel).t
        else:
            physics[held["what"]].pos = T_obj.t

    if state["script"]:                                            # hard-coded break: balls follow a script
        Play_Script_Frame(state["script"].pop(0))
    elif state["physics"]:
        for _ in range(PHYSICS_SUBSTEPS):
            Table_Step(BALL_LIST, DT / PHYSICS_SUBSTEPS)
    for n, mesh in ball_meshes.items():
        if hasattr(mesh, "Place"):                               # meshes roll with the ball (numbers turn)
            mesh.Place(physics[n].pos, physics[n].R)
        else:
            mesh.T = SE3(*physics[n].pos).A


def Frame(dq=None, jq=None):
    """ONE frame of motion. Every sequence is a generator built from these: the main
    loop asks for the next frame only while the system is running, so the E-STOP
    simply stops asking - every robot and ball freezes exactly where it is."""
    Draw(dq, jq)
    yield


def Steps_For(q_from, q_goal, steps):
    biggest = np.max(np.abs(np.asarray(q_goal) - np.asarray(q_from)))
    return max(steps, int(1.9 * biggest / (JOINT_SPEED * DT)) + 1)


def Doosan_Joints(q_goal, steps=80):
    for q in rtb.jtraj(doosan.q, q_goal, Steps_For(doosan.q, q_goal, steps)).q:
        yield from Frame(dq=q)


def Jbot_Joints(q_goal, steps=30):
    for q in rtb.jtraj(jbot.robot.q, q_goal, Steps_For(jbot.robot.q, q_goal, steps)).q:
        yield from Frame(jq=q)


def Jbot_Line(T_from, T_to, steps=20, keep_spin=False):
    """Cup in a straight line. keep_spin=True also holds the cup's spin (used when carrying the rack)."""
    q = jbot.robot.q
    mask = (1, 1, 1, 1, 1, 1) if keep_spin else (1, 1, 1, 1, 1, 0)
    for s in rtb.quintic(0, 1, steps).q:
        T = T_from.interp(T_to, s)
        q_new = Jbot_IK(jbot.robot, T * jbot.tool.inv(), q, mask=mask)
        if q_new is None:
            Set_Status(f"Jbot IK failed at {np.round(T.t, 3)} - stopping this move")
            return False
        q = q_new
        yield from Frame(jq=q)
    return True


def Jbot_Drive(goal):
    """Jbot drives in the convoy too, so any robot parked in its way is moved on ahead of it."""
    yield from Drive_Convoy("J", goal)


def Doosan_Drive(goal):
    global d_xyyaw
    for pose in Drive_Path(d_xyyaw, goal, DT, loop=DOOSAN_LOOP):
        d_xyyaw = pose
        yield from Frame()


# ------------------------------------RESET-----------------------------------------------------

def Reset_Scene():
    """Balls back in the pockets, rack on the felt, robots at their start spots."""
    global d_xyyaw
    Teach_Unhide()
    state["physics"], state["script"], state["fast"] = False, [], False
    state["score"], state["in_play"] = {"D": 0, "U": 0}, False
    station.Reset()
    cue_state["shaft"] = station.Shaft_Pose(SG.Q_GRIP)                 # shaft waits on its rests, Doosan has the handle
    held["what"] = None
    for pocket, stack in POCKET_STACKS.items():
        px, py = POCKETS[pocket]
        for level, n in enumerate(stack):
            physics[n].Set([px, py, POCKET_FLOOR_Z + R + level * 2 * R])
            physics[n].floor_z = FELT_Z + R
    physics[0].Set([*CUE_BALL_SPOT, FELT_Z + R])
    rack.Place(SE3(RACK_START_APEX, 0, FELT_Z + R))

    d_xyyaw = DOOSAN_WAIT
    jbot.xyyaw = TRAY_PARK
    for p, home in ((ur3_p, UR3_HOME), (meca_p, MECA_HOME)):
        p.xyyaw = home
        p.Set_Base()
        q_carry = MECA_CARRY_Q.copy() if p is meca_p else p.IK(p.Carry_Tip(), p.q)
        if q_carry is not None:
            p.q = q_carry
    d_base = Robot_Base_On(Turtle_Pose(*d_xyyaw))
    doosan.base = d_base
    doosan.q = Solve_IK(doosan, Doosan_Carry_Pose(d_base), D_TOOL)
    jbot.Set_Base()
    jbot.robot.q = jbot.IK(jbot.robot.base * CARRY_LOCAL)
    Draw()


# ------------------------------------JBOT: RACK THE BALLS-----------------------------------------------------

def Jbot_Carry_q():
    return jbot.IK(jbot.robot.base * CARRY_LOCAL, jbot.robot.q)


def Pick_From_Pocket(n, pocket, level):
    yield from Jbot_Drive(POCKET_PARKS[pocket])
    px, py = POCKETS[pocket]
    T_above = Tool_Down(px, py, ABOVE)
    T_pick = Tool_Down(px, py, POCKET_FLOOR_Z + 2 * R + level * 2 * R)     # cup on top of the ball
    yield from Jbot_Joints(jbot.IK(T_above, jbot.robot.q))
    yield from Jbot_Line(T_above, T_pick)
    held["what"], held["T_rel"] = n, jbot.Tool_Pose().inv() * SE3(*physics[n].pos)
    yield from Jbot_Line(T_pick, T_above)
    yield from Jbot_Joints(Jbot_Carry_q())


def Drop_In_Rack(n, slot_xy):
    yield from Jbot_Drive(TRAY_PARK)
    T_above = Tool_Down(slot_xy[0], slot_xy[1], FELT_Z + 2 * R + 0.08)
    T_drop = Tool_Down(slot_xy[0], slot_xy[1], FELT_Z + 2 * R)
    yield from Jbot_Joints(jbot.IK(T_above, jbot.robot.q))
    yield from Jbot_Line(T_above, T_drop)
    held["what"] = None
    physics[n].pos = np.array([slot_xy[0], slot_xy[1], FELT_Z + R])
    yield from Jbot_Line(T_drop, T_above)
    yield from Jbot_Joints(Jbot_Carry_q())


def Slide_And_Lift_Rack():
    """Grip the rack's pad, slide the full rack to the foot spot, lift it off, keep it."""
    top = rack.Handle_Top()
    T_above = Tool_Down(top[0], top[1], top[2] + 0.08)
    T_grip = Tool_Down(top[0], top[1], top[2])
    yield from Jbot_Joints(jbot.IK(T_above, jbot.robot.q))
    yield from Jbot_Line(T_above, T_grip)
    T_cup = jbot.Tool_Pose()
    held["what"], held["T_rel"] = "rack", T_cup.inv() * rack.T
    held["balls"] = {n: rack.T.inv() * SE3(*physics[n].pos) for n in PICK_ORDER}
    Set_Status("Jbot: sliding the rack to the foot spot")
    yield from Jbot_Line(T_cup, SE3(FOOT_SPOT - RACK_START_APEX, 0, 0) * T_cup, steps=80, keep_spin=True)
    held.pop("balls")                                              # balls stay racked
    Set_Status("Jbot: lifting the rack off")
    T_cup = jbot.Tool_Pose()
    yield from Jbot_Line(T_cup, SE3(0, 0, 0.08) * T_cup, steps=20, keep_spin=True)
    yield from Jbot_Joints(Jbot_Carry_q())


def Rack_The_Balls():
    state["fast"] = True                                       # racking plays RACK_SPEED_UP times faster
    slots = [np.array([RACK_START_APEX, 0.0]) + s[:2] for s in Rack_Slots(R)]
    where = {n: (pocket, level) for pocket, stack in POCKET_STACKS.items() for level, n in enumerate(stack)}
    for k, n in enumerate(PICK_ORDER):
        pocket, level = where[n]
        Set_Status(f"Jbot: collecting ball {n} from pocket {pocket} ({k + 1}/9)")
        yield from Pick_From_Pocket(n, pocket, level)
        Set_Status(f"Jbot: placing ball {n} in the rack")
        yield from Drop_In_Rack(n, slots[k])
    yield from Slide_And_Lift_Rack()
    state["fast"] = False


# ------------------------------------HARD-CODED BREAK (balls land on the photo layout)--------------------

def Rack_Spots():
    """Where each ball sits in the rack after it has been slid to the foot spot."""
    slots = [np.array([FOOT_SPOT, 0.0]) + s[:2] for s in Rack_Slots(R)]
    return {n: slots[k] for k, n in enumerate(PICK_ORDER)}


def Break_Frames(cue_xy, cue_speed):
    """Frames for the break: the cue ball runs into the apex ball, then every ball
    spreads to its spot in the photo layout (orange into its pocket)."""
    start = Rack_Spots()
    apex = start[PICK_ORDER[0]]
    u = (apex - cue_xy) / np.linalg.norm(apex - cue_xy)
    hit_xy = apex - 2 * R * u                                  # cue ball touching the apex ball
    n_run = max(1, int(np.linalg.norm(hit_xy - cue_xy) / (cue_speed * DT)))
    frames = [{0: cue_xy + (hit_xy - cue_xy) * k / n_run} for k in range(1, n_run + 1)]
    start[0] = hit_xy
    goal = dict(LAYOUT)
    goal[ORANGE] = np.array(POCKETS[ORANGE_POCKET])
    return frames + Break_Script(start, goal, n_frames=150, radius=R)


def Play_Script_Frame(frame):
    """Move the balls to one frame of the script (the cue ball rolls as it moves)."""
    hole = np.array(POCKETS[ORANGE_POCKET])
    for n, xy in frame.items():
        p = physics[n]
        z = FELT_Z + R
        if n == ORANGE:                                        # drops as it reaches the hole
            sink = np.clip((0.09 - np.linalg.norm(xy - hole)) / 0.06, 0, 1)
            z -= sink * (FELT_Z - POCKET_FLOOR_Z)
        new = np.array([xy[0], xy[1], z])
        step = new - p.pos
        p.R = _Rotation_From_Vector(np.cross([0, 0, 1.0], step) / R) @ p.R
        p.pos = new


def Set_Layout(layout=None):
    """Balls exactly on the layout, at rest; the orange in its pocket."""
    for n, p in physics.items():
        if n == ORANGE:
            continue
        xy = (layout or LAYOUT).get(n)
        if xy is None:
            continue
        keep = p.R
        p.Set([xy[0], xy[1], FELT_Z + R])
        p.R = keep
    o = physics[ORANGE]
    o.Set([*POCKETS[ORANGE_POCKET], POCKET_FLOOR_Z + R])
    o.floor_z = FELT_Z + R
    o.potted = ORANGE_POCKET


# ------------------------------------DOOSAN: BREAK-----------------------------------------------------

def Break_Shot():
    Set_Status("Doosan: driving to the head of the table")
    yield from Doosan_Drive(BREAK_TURTLE)
    ball = physics[0].pos.copy()
    aim = np.degrees(np.arctan2(0.0 - ball[1], FOOT_SPOT - ball[0]))       # at the apex ball
    d = Cue_Pose([0, 0, 0], aim, BREAK_ELEV_DEG, BREAK_ROLL_DEG).R[:, 2]
    contact = -R * d
    T_address = Cue_Pose(ball + contact - ADDRESS_GAP * d, aim, BREAK_ELEV_DEG, BREAK_ROLL_DEG)
    T_ready = T_address * SE3.Tz(-APPROACH)
    q_carry = doosan.q.copy()
    Set_Status("Doosan: lining up the break")
    q_ready = Solve_IK(doosan, T_ready, D_TOOL, Solve_IK(doosan, T_address, D_TOOL, quiet=True))
    if q_ready is None:
        Set_Status("Doosan can't reach the break from here")
        return
    yield from Doosan_Joints(q_ready)
    q = q_ready
    for s in rtb.quintic(0, 1, 40).q:
        q = Solve_IK(doosan, T_ready.interp(T_address, s), D_TOOL, q)
        yield from Frame(dq=q)
    Set_Status("Doosan: BREAK!")
    prev_tip, hit = T_address.t, False
    for s in Punch_Stroke_Profile(DT, contact_s=-ADDRESS_GAP, tip_speed=BREAK_TIP_SPEED, follow=0.05):
        T_tip = T_address * SE3.Tz(-s)
        q = Solve_IK(doosan, T_tip, D_TOOL, q)
        if q is None:
            break
        if not hit and np.dot(T_tip.t - (ball + contact), d) >= -1e-6:
            state["script"] = Break_Frames(ball[:2], CUE_TO_BALL * BREAK_TIP_SPEED)
            hit = True
        prev_tip = T_tip.t
        yield from Frame(dq=q)
    T_now = doosan.fkine(doosan.q) * D_TOOL
    for s in rtb.quintic(0, 1, 30).q:
        q = Solve_IK(doosan, T_now.interp(T_now * SE3.Tz(-APPROACH), s), D_TOOL, doosan.q)
        yield from Frame(dq=q)
    yield from Doosan_Joints(q_carry)
    while state["script"]:
        yield from Frame()
    Set_Layout()
    Set_Status("Break done - the orange went down (top-right corner)")


# ------------------------------------DOOSAN: TRICK-SHOT RUN-OUT-----------------------------------------------

LIFT = 0.15                           # cue lifted this far above its line before swinging in / out
FOOT_END_KEEP_OUT = 1.05              # the Doosan never drives past this x (Jbot is parked at the foot end)
HEAD_WAYPOINT = (-1.72, 0.0, 90.0)


def Doosan_Drive_Safe(goal):
    """Drive round the loop, but never through the foot end where Jbot is parked."""
    path = Drive_Path(d_xyyaw, goal, DT, loop=DOOSAN_LOOP)
    if max(p[0] for p in path) > FOOT_END_KEEP_OUT:
        yield from Doosan_Drive(HEAD_WAYPOINT)                 # go the long way, round the head end
    yield from Doosan_Drive(goal)


def Lift_From(T, q_seed):
    """Highest lift (up to LIFT) the Doosan can move straight up to from cue pose T
    without the arm flipping. Returns (lift, joints at the top)."""
    for lift in (LIFT, 0.10, 0.05):
        q, ok = q_seed, True
        for s in np.linspace(0, 1, 16)[1:]:
            q_new = Solve_IK(doosan, SE3(0, 0, lift * s) * T, D_TOOL, q, quiet=True)
            if q_new is None or np.max(np.abs(q_new - q)) > 0.3:
                ok = False
                break
            q = q_new
        if ok:
            return lift, q
    return 0.0, q_seed


DOOSAN_VMAX = np.radians([150, 150, 180, 225, 225, 225])   # M0609 joint speed limits


def Doosan_Path(T_from, T_to, q_start, steps=40):
    """Joint angles for the cue tip moving in a straight line T_from -> T_to,
    or None if the arm would have to flip part way."""
    qs, q = [], q_start
    for s in rtb.quintic(0, 1, steps).q[1:]:
        q_new = Solve_IK(doosan, T_from.interp(T_to, s), D_TOOL, q, quiet=True)
        if q_new is None or np.max(np.abs(q_new - q)) > 0.3:
            return None
        qs.append(q_new)
        q = q_new
    return qs


def Doosan_Move(T_from, T_to, q_goal, joint_fallback=True):
    """Cue tip in a straight line, slowed down so no joint goes over its speed limit.
    If the straight line can't be followed, a joint move to q_goal is used instead."""
    qs = Doosan_Path(T_from, T_to, doosan.q)
    if qs is None:
        if q_goal is None or not joint_fallback:
            qs = Doosan_Path(T_from, T_to, doosan.q, steps=120) or []
        else:
            yield from Doosan_Joints(q_goal)
            return
    q_prev = doosan.q.copy()
    for q in qs:                                               # extra frames wherever a joint would be too fast
        n = max(1, int(np.ceil(np.max(np.abs(q - q_prev) / (0.9 * DOOSAN_VMAX * DT)))))
        for t in np.linspace(0, 1, n + 1)[1:]:
            yield from Frame(dq=q_prev + t * (q - q_prev))
        q_prev = q


def Settle_Table(layout):
    """Every ball still on the table comes to rest exactly on the planned layout
    (it is already there to well under a millimetre - this just removes rounding)."""
    for n, xy in layout.items():
        p = physics[n]
        keep = p.R
        p.Set([xy[0], xy[1], FELT_Z + R])
        p.R = keep


def Take_Shot(k, shot):
    target, pocket = shot["ball"], shot["pocket"]
    name = f"Shot {k}/{len(SHOTS)} ({shot['kind']}): ball {target} -> {POCKET_NAMES[pocket]} pocket"
    Set_Status(f"Doosan driving to {name}")
    yield from Doosan_Drive_Safe(tuple(shot["turtle"]))
    Settle_Table(shot["layout"])
    if shot.get("guide"):                                      # hard-coded: this ball is steered into its pocket
        physics[target].guide = tuple(shot["guide"])
    ball = physics[0].pos.copy()
    d, contact, T_address = Shot_Poses(ball, shot, R, ADDRESS_GAP)
    T_ready = T_address * SE3.Tz(-APPROACH)
    q_carry = doosan.q.copy()
    q_ready = Solve_IK(doosan, T_ready, D_TOOL, np.array(shot["q_ready"]))
    if q_ready is None:
        Set_Status(f"Doosan can't reach {name} - skipping it")
        return
    Set_Status(f"Lining up {name}")
    lift, q_above = Lift_From(T_ready, q_ready)                 # come down onto the line from above,
    T_above = SE3(0, 0, lift) * T_ready                          # so the cue never sweeps through a rail
    T_carry = doosan.fkine(q_carry) * D_TOOL
    yield from Doosan_Move(T_carry, T_above, q_above)            # carry -> above the shot
    yield from Doosan_Move(T_above, T_ready, q_ready, joint_fallback=False)
    yield from Doosan_Move(T_ready, T_address, None, joint_fallback=False)
    Set_Status(name)
    state["physics"] = True
    profile = Punch_Stroke_Profile(DT, contact_s=-ADDRESS_GAP, tip_speed=shot["speed"],
                                   back=shot["back"], follow=shot["follow"])
    hit_at = int(np.argmax(profile <= -ADDRESS_GAP + 1e-9))      # the sample where the tip reaches the ball
    q = doosan.q
    for i, s in enumerate(profile):
        q = Solve_IK(doosan, T_address * SE3.Tz(-s), D_TOOL, q)
        if i == hit_at:
            physics[0].Hit(CUE_TO_BALL * shot["speed"] * d, contact)
        yield from Frame(dq=q)
    T_now = doosan.fkine(doosan.q) * D_TOOL
    T_back = T_now * SE3.Tz(-APPROACH)
    yield from Doosan_Move(T_now, T_back, None, joint_fallback=False)
    lift, _ = Lift_From(T_back, doosan.q)
    T_up = SE3(0, 0, lift) * T_back                              # straight up, clear of the rails
    yield from Doosan_Move(T_back, T_up, None, joint_fallback=False)
    yield from Doosan_Move(T_up, T_carry, q_carry)               # back to carrying the cue upright
    while any(p.Moving() for p in BALL_LIST):
        yield from Frame()
    p = physics[target]
    if p.potted == pocket:
        Set_Status(f"{name}: POTTED")
    else:
        Set_Status(f"{name}: missed")
    for _ in range(25):                                        # short pause between shots
        yield from Frame()


def Run_Out():
    """The trick-shot run-out from the photo layout: 8 shots, all by the Doosan."""
    for k, shot in enumerate(SHOTS, start=1):
        yield from Take_Shot(k, shot)
    left = [n for n, p in physics.items() if p.potted is None and n != 0]
    Set_Status("Run-out complete - table cleared!" if not left else f"Run-out finished - left on the table: {left}")


# ------------------------------------THE GAME: Doosan + the Meca500 vs the UR3-----------------------------
#
# The shots come from Pool_Game.GAME (planned offline with this exact ball physics). Every ball
# moves by physics alone - nothing is steered. Before each shot the table is "settled" onto the
# planned layout (it already matches to well under a millimetre - this only removes rounding).
#
# Traffic: all four turtlebots share the track and cannot pass each other. Before a robot
# drives, any robot in its way (or within GAP of where it is going) is first driven on ahead
# of it, out of the way - so two turtlebots never come closer than GAP along the track.

GAP = 0.75                                                     # m along the track between two turtlebots
SHOT_ROOM = 1.6                                                # m kept free either side of a robot taking a shot (its cue)
OUT_LOOP = _Loop(*DOOSAN_LOOP)
NAMES = {"D": "Doosan", "U": "UR3", "M": "Meca500", "J": "Jbot"}


def Get_XY(k):
    return {"D": lambda: d_xyyaw, "U": lambda: ur3_p.xyyaw, "M": lambda: meca_p.xyyaw, "J": lambda: jbot.xyyaw}[k]()


def Loop_Of(k):
    return (INNER_X, INNER_Y) if k in ("M", "J") else DOOSAN_LOOP


def Track_S(xy, k="D"):
    """Position along the track (0 .. track length), measured on robot k's own loop and scaled
    to the Doosan's loop - so the inner and outer loops line up, corner for corner."""
    L = _Loop(*Loop_Of(k))
    return _Loop_S(np.asarray(xy[:2], dtype=float), L) * OUT_LOOP[2] / L[2]


def On_Loop(k, s):
    """Point on robot k's own loop at track position s."""
    L = _Loop(*Loop_Of(k))
    p = _Loop_Point((s % OUT_LOOP[2]) * L[2] / OUT_LOOP[2], L)
    return (float(p[0]), float(p[1]), None)


def Drive_Robot(k, goal):
    global d_xyyaw
    path = Drive_Path(Get_XY(k), goal, DT, speed=0.8 if k == "J" else 0.6, loop=Loop_Of(k),
                      allow_reverse=k in ("M", "J"))
    for pose in path:
        if k == "D":
            d_xyyaw = pose
        elif k == "U":
            ur3_p.xyyaw = pose
        elif k == "M":
            meca_p.xyyaw = pose
        else:
            jbot.xyyaw = pose
        yield from Frame()


TURTLE_GAP = 1.1                                               # turtlebot centres never closer than this


def Set_XY(k, pose):
    global d_xyyaw
    if k == "D":
        d_xyyaw = pose
    elif k == "U":
        ur3_p.xyyaw = pose
    elif k == "M":
        meca_p.xyyaw = pose
    else:
        jbot.xyyaw = pose


def _Too_Close(k, xy):
    return [o for o in ("D", "U", "M", "J")
            if o != k and np.linalg.norm(np.array(Get_XY(o)[:2]) - xy) < TURTLE_GAP]


def _Nudge(o, direction, ds, depth=0):
    """Move robot o a little way along its own loop (the way the robot behind it is going),
    nudging on any robot in front of it first - a convoy, so nobody is ever driven into."""
    if depth > 4:
        return
    old = np.array(Get_XY(o)[:2])
    for _ in range(20):
        new = On_Loop(o, Track_S(old, o) + direction * ds)
        if np.linalg.norm(np.array(new[:2]) - old) > 1e-4:
            break
        ds += 0.02                                             # stuck at an inner corner - step further
    xy = np.array(new[:2])
    for other in _Too_Close(o, xy):
        if np.linalg.norm(np.array(Get_XY(other)[:2]) - xy) < np.linalg.norm(np.array(Get_XY(other)[:2]) - old) + 1e-9:
            for _ in range(30):                                # keep it a real TURTLE_GAP away (corners!)
                _Nudge(other, direction, ds, depth + 1)
                if np.linalg.norm(np.array(Get_XY(other)[:2]) - xy) >= TURTLE_GAP:
                    break
    yaw = Get_XY(o)[2]
    Set_XY(o, (float(xy[0]), float(xy[1]), yaw))


def Drive_Convoy(k, goal):
    """Drive robot k to goal; any robot in front of it is pushed along ahead of it."""
    path = Drive_Path(Get_XY(k), goal, DT, speed=0.8 if k == "J" else 0.6, loop=Loop_Of(k),
                      allow_reverse=k in ("M", "J"))
    pushed = False
    for pose in path:
        old = np.array(Get_XY(k)[:2])
        xy = np.array(pose[:2])
        step = np.linalg.norm(xy - old)
        if step > 1e-6:
            direction = 1 if ((Track_S(xy, k) - Track_S(old, k) + OUT_LOOP[2] / 2) % OUT_LOOP[2] - OUT_LOOP[2] / 2) >= 0 else -1
            for other in _Too_Close(k, xy):
                o_xy = np.array(Get_XY(other)[:2])
                if np.linalg.norm(o_xy - xy) < np.linalg.norm(o_xy - old):   # getting closer -> push it on
                    for _ in range(30):
                        _Nudge(other, direction, step + 0.005)
                        if np.linalg.norm(np.array(Get_XY(other)[:2]) - xy) >= TURTLE_GAP:
                            break
                    if not pushed:
                        Set_Status(f"{NAMES[k]} driving - {NAMES[other]} moves along ahead of it")
                        pushed = True
        Set_XY(k, pose)
        yield from Frame()


def Make_Room(k, room=1.0):
    """Before a shot: robots closer than `room` to the shooter drive away along the track."""
    me = np.array(Get_XY(k)[:2])
    for other in ("D", "U", "M", "J"):
        if other == k:
            continue
        for _ in range(400):
            o = np.array(Get_XY(other)[:2])
            if np.linalg.norm(o - me) >= room:
                break
            away = 1 if ((Track_S(o, other) - Track_S(me, k) + OUT_LOOP[2] / 2) % OUT_LOOP[2] - OUT_LOOP[2] / 2) >= 0 else -1
            _Nudge(other, away, 0.35 * DT * 2)
            if np.linalg.norm(np.array(Get_XY(k)[:2]) - me) > 1e-6:      # never nudge the shooter
                Set_XY(k, (*me, Get_XY(k)[2]))
            yield from Frame()


class Arm:
    """Uniform access to the three shooting arms (joint angles, IK, cue tip, carry pose, speed limits)."""

    def __init__(self, k):
        self.k = k
        self.p = {"U": ur3_p, "M": meca_p}.get(k)
        self.vmax = DOOSAN_VMAX if k == "D" else self.p.VMAX

    @property
    def q(self):
        return doosan.q if self.k == "D" else self.p.q

    def Robot(self):
        return doosan if self.k == "D" else self.p.robot

    def IK(self, T, seed=None):
        if self.k == "D":
            return Solve_IK(doosan, T, D_TOOL, seed, quiet=True)
        return self.p.IK(T, seed)

    def Shape_IK(self, T):
        """IK from the ONE arm shape (Pool_Arm.ARM_SHAPE): base turned to face T, the other joints
        from the fixed elbow-up shape. Only that seed is tried, and the answer must keep the shape."""
        if self.k == "M":
            return self.IK(T)
        seed = PA.Shape_Seed(self.k, self.Robot(), T)
        if self.k == "D":
            sol = doosan.ikine_LM(T * D_TOOL.inv(), q0=seed, joint_limits=False, slimit=1)
        else:
            T_local = ur3_p.robot.base.inv() * T * ur3_p.tool.inv()
            sol = ur3_p.kin.ikine_LM(T_local, q0=seed,  joint_limits=True, slimit=1)
        return sol.q if sol.success and PA.Same_Shape(sol.q, seed) else None

    def Tip(self, q=None):
        if self.k == "D":
            return doosan.fkine(doosan.q if q is None else q) * D_TOOL
        return self.p.Tip(q)

    def Carry(self):
        return Doosan_Carry_Pose(doosan.base) if self.k == "D" else self.p.Carry_Tip()

    def Frame(self, q):
        if self.k == "D":
            yield from Frame(dq=q)
        else:
            self.p.q = q
            yield from Frame()


def Arm_Play(arm, qs):
    """Play joint angles, adding frames wherever a joint would go over its speed limit."""
    q_prev = np.array(arm.q, dtype=float)
    for q in qs:
        q = np.asarray(q, dtype=float)
        n = max(1, int(np.ceil(np.max(np.abs(q - q_prev) / (0.9 * arm.vmax * DT)))))
        for t in np.linspace(0, 1, n + 1)[1:]:
            yield from arm.Frame(q_prev + t * (q - q_prev))
        q_prev = q


def In_Limits(arm, qs):
    if arm.k != "M":
        return True
    lo, hi = arm.Robot().qlim
    return all(np.all(q >= lo - 1e-6) and np.all(q <= hi + 1e-6) for q in qs)


def RMRC_Path(arm, poses):
    """RMRC (Pool_Arm.RMRC): joint angles that carry the cue tip through `poses`, one per frame.
    None if the arm can't follow them closely (5 mm) or would leave its joint limits."""
    qs, err = PA.RMRC(arm.Tip, np.array(arm.q, dtype=float), poses, DT)
    return qs if err < 0.005 and In_Limits(arm, qs) else None


def Arm_Line(arm, T_from, T_to, q_goal=None, steps=40):
    """Cue tip in a straight line, driven by RMRC (speed-limited). If RMRC can't follow the line,
    or the cue would touch the arm on the way, a collision-checked joint move to q_goal."""
    q0 = np.array(arm.q, dtype=float)
    poses = [T_from.interp(T_to, s) for s in rtb.quintic(0, 1, steps).q]
    qs = RMRC_Path(arm, poses)
    if qs is None:                                   # RMRC can't (joint limits): IK at every step of the line
        qs = IK_Line(arm, poses)
    if q_goal is None and qs is None:
        q_goal = arm.IK(T_to, q0)
    if qs is None or min(Cue_Free_Gap(arm, q) for q in qs) < PA.CLEARANCE:
        if q_goal is None:
            if qs is None:
                return
            qs = Clear_Joint_Path(arm, q0, np.asarray(qs[-1], dtype=float), straight=np.array(qs))
        else:
            qs = Clear_Joint_Path(arm, q0, np.asarray(q_goal, dtype=float),
                                  straight=None if qs is None else np.array(qs))
    yield from Arm_Play(arm, qs)


def IK_Line(arm, poses):
    """Straight cue-tip line by IK at every step (used when RMRC would leave the joint limits).
    None if any step has no solution or the arm would jump."""
    qs, q = [], np.array(arm.q, dtype=float)
    for T in poses[1:]:
        q_new = arm.IK(T, q)
        if q_new is None or np.max(np.abs(q_new - q)) > 0.3:
            return None
        qs.append(q_new)
        q = q_new
    return qs


def Cue_Clearance(arm, q):
    """Air gap between the arm's own cue and its own links (Pool_Arm.Cue_Gap: links as capsules
    of their real thickness; the flange the holder bolts onto is left out)."""
    T, L = Real_Cue(arm, q)
    return PA.Cue_Gap(arm.k, arm.Robot(), q, T, L)


def Real_Cue(arm, q):
    """Tip pose and length of the part of the cue the arm really holds (before the assembly
    the Doosan only has the handle)."""
    T = arm.Tip(q)
    if arm.k == "D" and cue_state["shaft"] is not None:
        return T * SE3.Tz(-cue.shaft_length), cue.length - cue.shaft_length
    return T, (1.127 if arm.k == "M" else cue.length)


def Nearest_Turn(arm, q0, q1):
    """Same arm pose, but each joint that can turn all the way round (Doosan, UR3) goes the SHORT
    way (+-180 deg) instead of nearly a full circle."""
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    if arm.k == "M":
        return q1
    lo, hi = (np.full(6, -2 * pi), np.full(6, 2 * pi)) if arm.k == "D" else ur3_p.robot.qlim
    q = q0 + (q1 - q0 + pi) % (2 * pi) - pi
    return np.where((q >= lo) & (q <= hi), q, q1)


def Clear_Joint_Path(arm, q0, q1, steps=50, straight=None):
    """Joint move q0 -> q1 that keeps the cue off the robot's own arm. Tries, in order: the
    straight-line move (if given), a plain joint move, moving some joints first (base, wrist,
    shoulder) and going through the upright carry pose - and takes the first that stays clear
    (Pool_Arm.CLEARANCE), or else the one that stays clearest."""
    q1 = Nearest_Turn(arm, q0, q1)                               # never spin a joint the long way round

    def Two(q_mid):
        return np.vstack([rtb.jtraj(q0, q_mid, steps // 2).q, rtb.jtraj(q_mid, q1, steps // 2).q[1:]])
    options = [] if straight is None else [straight]
    options.append(rtb.jtraj(q0, q1, steps).q)
    for first in ([0], [1, 2, 3, 4, 5], [3, 4, 5], [0, 3, 4, 5], [1, 2]):
        q_mid = q0.copy()
        q_mid[first] = q1[first]
        options.append(Two(q_mid))
    q_carry = arm.IK(arm.Carry(), q0)
    if q_carry is not None:
        options.append(np.vstack([rtb.jtraj(q0, q_carry, steps // 2).q, rtb.jtraj(q_carry, q1, steps // 2).q[1:]]))
    best, best_d = options[0], -1.0
    for qs in options:
        d = min(Cue_Free_Gap(arm, q) for q in qs)
        if d >= PA.CLEARANCE:
            return qs
        if d > best_d:
            best, best_d = qs, d
    # no simple move stays clear: plan one (RRT-Connect in joint space, Pool_Arm.py)
    free = lambda q: Cue_Free_Gap(arm, q) >= PA.CLEARANCE
    if free(q0) and free(q1):
        path = PA.RRT_Connect(q0, q1, free)
        if path is not None:
            dense = [q0]
            for a, b in zip(path[:-1], path[1:]):
                n = max(4, int(np.ceil(np.max(np.abs(b - a)) / 0.04)))
                dense += list(rtb.jtraj(a, b, n).q[1:])
            return np.array(dense)
    return best


# ------------------------------------COLLISION MODEL (every robot, turtlebot and cue)----------------------------
# Every link is a capsule (a tube with round ends) as thick as the real link, every turtlebot
# body is a capsule, every cue is a 15 mm tube. One model is used for planning the arm moves and
# for the full-run check, so what the check passes is what you see.

ARM_RADII = {"D": [0.11, 0.085, 0.08, 0.07, 0.065, 0.06],                    # Doosan M0609
             "U": [0.0, 0.065, 0.055, 0.05, 0.045, 0.045, 0.04, 0.035],        # UR3 (first 9 URDF frames)
             "M": [0.08, 0.067, 0.06, 0.053, 0.0, 0.045],                      # Meca500 x2
             "J": [0.092] * 12,                                                # Jbot x3.0
             "S": [0.04] * 6}                                                  # station Meca500 (real size)
TURTLE_RADIUS = 0.17


def Robot_Of(k):
    return {"D": doosan, "U": ur3_p.robot, "M": meca_p.robot, "J": jbot.robot, "S": station.robot}[k]


def Arm_Capsules(k, q=None):
    rob = Robot_Of(k)
    try:
        P = np.array([T.t for T in rob.fkine_all(rob.q if q is None else q)])
    except Exception:
        P = np.array([rob.base.t, rob.fkine(rob.q if q is None else q).t])
    if k == "U":
        P = P[:9]
    radii = ARM_RADII[k]
    return [(P[i], P[i + 1], radii[min(i, len(radii) - 1)]) for i in range(len(P) - 1)
            if radii[min(i, len(radii) - 1)] > 0 and np.linalg.norm(P[i + 1] - P[i]) > 1e-6]


def Turtle_Capsule(k):
    tb = Turtle_Pose(*Get_XY(k))
    return [((tb * SE3(-0.05, 0, 0.07)).t, (tb * SE3(0.05, 0, 0.07)).t, TURTLE_RADIUS)]


def Cue_Capsule(k, q=None):
    if k == "J":
        return []
    arm = Arm(k)
    T, L = Real_Cue(arm, arm.q if q is None else q)
    return [(T.t, (T * SE3.Tz(-L)).t, PA.CUE_RADIUS)]


def Others(k):
    """Everything that is NOT robot k, as capsules: other arms, turtlebots and cues (and the
    station, unless its hand is meant to touch the cue)."""
    out = []
    for o in "DUMJ":
        if o == k:
            continue
        out += Arm_Capsules(o) + Turtle_Capsule(o) + Cue_Capsule(o)
    if avoid["station"] and k != "S":
        out += Arm_Capsules("S")
    return out


def Caps_Gap(A, B):
    return min((PA.Seg_Dist(a1, a2, b1, b2) - ra - rb for a1, a2, ra in A for b1, b2, rb in B), default=9.0)


# capsule radii kept for the older helpers
OTHER_RADII = ARM_RADII


_obstacles = {"key": None, "caps": []}
avoid = {"station": True}              # off while the Doosan's cue is being assembled in the station's hand


def Cue_Free_Gap(arm, q):
    """Smallest air gap for arm k at joints q: its cue to its own links and turtlebot, and its
    cue AND its links to every other robot, turtlebot and cue. (The first 6 cm of the cue are
    left out - the tip is meant to touch the ball / chalk.)"""
    key = (arm.k, avoid["station"], tuple(np.round([c for k in "DUMJ" for c in Get_XY(k)[:2]], 3)),
           tuple(np.round(doosan.q, 3)), tuple(np.round(ur3_p.q, 3)), tuple(np.round(meca_p.q, 3)),
           tuple(np.round(jbot.robot.q, 3)), tuple(np.round(station.robot.q, 3)), cue_state["shaft"] is None)
    if _obstacles["key"] != key:
        _obstacles["key"], _obstacles["caps"] = key, Others(arm.k)
    T, L = Real_Cue(arm, q)
    a, b = T.t, (T * SE3.Tz(-L)).t
    a = a + 0.06 * (b - a) / np.linalg.norm(b - a)
    cue_cap = [(a, b, PA.CUE_RADIUS)]
    gap = min(Cue_Clearance(arm, q), Caps_Gap(cue_cap, Turtle_Capsule(arm.k)))
    gap = min(gap, Caps_Gap(cue_cap + Arm_Capsules(arm.k, q), _obstacles["caps"]))
    return gap


def Arm_Lift(arm, T, q_seed):
    """Highest lift (up to LIFT) the cue can rise straight up from pose T (by RMRC) without
    the cue touching the arm."""
    for lift in (LIFT, 0.10, 0.05):
        poses = [SE3(0, 0, lift * s) * T for s in np.linspace(0, 1, 12)]
        qs, err = PA.RMRC(arm.Tip, np.array(q_seed, dtype=float), poses, DT)
        if err >= 0.003 or not In_Limits(arm, qs):
            qs, q = [], np.array(q_seed, dtype=float)
            for Tp in poses[1:]:                                     # IK at each step instead
                q_new = arm.IK(Tp, q)
                if q_new is None or np.max(np.abs(q_new - q)) > 0.3:
                    qs = None
                    break
                qs.append(q_new)
                q = q_new
            if qs is None:
                continue
        if min(Cue_Clearance(arm, q) for q in qs) >= PA.CLEARANCE:
            return lift, qs[-1]
    return 0.0, np.array(q_seed, dtype=float)


def Game_Shot(k, shot):
    """One shot, done the same way every time:
       1. one joint move from carrying the cue upright to just above the shot line, in the arm's
          one shape (collision-checked joint path);
       2. RMRC straight down onto the line, then RMRC along it to the ball;
       3. the stroke - RMRC along the cue's own axis;
       4. RMRC back and up, then one joint move back to carrying the cue."""
    arm = Arm(k)
    layout = shot["layout"]
    ball = np.array([*layout[0], FELT_Z + R])
    d, contact, T_address = Shot_Poses(ball, shot, R, ADDRESS_GAP)
    steep = shot["elev"] > 30
    approach = 0.08 if steep else APPROACH
    T_ready = T_address * SE3.Tz(-approach)
    q_carry = np.array(arm.q, dtype=float)
    q_ready = arm.IK(T_ready, np.array(shot["q_ready"])) if k == "M" else arm.Shape_IK(T_ready)
    if q_ready is None:
        q_ready = arm.IK(T_ready, np.array(shot["q_ready"]))
    if q_ready is None:
        Set_Status(f"{NAMES[k]} can't reach this shot - skipping it")
        return
    lift, q_above = Arm_Lift(arm, T_ready, q_ready)
    T_above = SE3(0, 0, lift) * T_ready
    yield from Arm_Play(arm, Clear_Joint_Path(arm, q_carry, q_above))         # carry -> above the shot
    yield from Arm_Line(arm, T_above, T_ready, q_ready)                         # RMRC down onto the line
    yield from Arm_Line(arm, T_ready, T_address)                                # RMRC along it
    if k == "M":                                                       # short jump / masse stroke
        profile = np.array(Short_Stroke_Profile(DT, contact_s=-ADDRESS_GAP, tip_speed=shot["speed"],
                                                follow=0.015, backswing=0.05))
    else:
        profile = Punch_Stroke_Profile(DT, contact_s=-ADDRESS_GAP, tip_speed=shot["speed"],
                                       back=0.20 if k == "D" else 0.15, follow=0.03)
    hit_at = int(np.argmax(profile <= -ADDRESS_GAP + 1e-9))
    T_start = arm.Tip()
    qs = RMRC_Path(arm, [T_start] + [T_address * SE3.Tz(-s) for s in profile])     # the stroke, by RMRC
    if qs is None:                                                     # RMRC would leave the joint limits:
        qs = IK_Line(arm, [T_start] + [T_address * SE3.Tz(-s) for s in profile])   # IK at every sample
    if qs is None:
        qs, q = [], np.array(arm.q, dtype=float)
        for s in profile:
            q_new = arm.IK(T_address * SE3.Tz(-s), q)
            q = q if q_new is None else q_new
            qs.append(q)
    for i, q in enumerate(qs):
        if i == hit_at:
            # the ball gets the planned cue speed. (Strokes are played back within the joints'
            # speed limits, so on screen a fast stroke's cue moves slower than the ball is hit.)
            physics[0].Hit(CUE_TO_BALL * shot["speed"] * d, contact)
        yield from Arm_Play(arm, [q])
    T_now = arm.Tip()
    T_back = T_now * SE3.Tz(-approach)
    yield from Arm_Line(arm, T_now, T_back)
    lift, _ = Arm_Lift(arm, T_back, arm.q)
    yield from Arm_Line(arm, T_back, SE3(0, 0, lift) * T_back)
    yield from Arm_Play(arm, Clear_Joint_Path(arm, np.array(arm.q, dtype=float), q_carry))   # back to carrying


def Station_Play(qs, shaft=False):
    """The station Meca500 follows a list of joint angles (the shaft follows it if it holds it)."""
    for q in qs:
        station.robot.q = q
        if shaft:
            cue_state["shaft"] = station.Shaft_Pose(q)
        yield from Frame()


def Assemble():
    """Assembly: the station Meca500 lifts the shaft off its rests and holds it level
    (test_meca500_shaft_grip), the Doosan brings the handle in along the same line and the
    shaft screws on (DoosanAssemblyRightPosition: HANDLE_PIN, N_TURNS, LINE_UP_HEIGHT)."""
    arm = Arm("D")
    avoid["station"] = False
    spot, roll, q_seeds = ASM_PRESENT
    Set_Status("Assembly: Doosan drives to the Meca station with the handle")
    yield from Drive_Convoy("D", tuple(spot))
    Set_Status("Assembly: the station Meca picks up the shaft")
    reach, lift = station.Pick_Shaft_Path()
    yield from Station_Play(reach)
    station.holding = "shaft"
    yield from Station_Play(lift, shaft=True)
    Set_Status("Assembly: Doosan lines the handle up with the shaft")
    q_carry = np.array(arm.q, dtype=float)
    T_line, T_pin = Assembly_Pose(LINE_UP + PIN, roll), Assembly_Pose(PIN, roll)
    q_line = arm.IK(T_line, np.array(q_seeds[2]))
    T_above = SE3(0, 0, 0.12) * T_line
    yield from Arm_Line(arm, arm.Tip(), T_above, arm.IK(T_above, q_line))
    yield from Arm_Line(arm, T_above, T_line, q_line)
    Set_Status("Assembly: sliding the handle onto the shaft's pin")
    yield from Arm_Line(arm, T_line, T_pin, arm.IK(T_pin, q_line))
    Set_Status("Assembly: screwing the shaft on")
    S = station.Shaft_Pose(SG.Q_HOLD)
    q = np.array(arm.q, dtype=float)
    for s in rtb.quintic(0, 1, 150).q:
        q_new = arm.IK(Assembly_Pose(PIN * (1 - s), roll), q)
        q = q if q_new is None else q_new
        cue_state["shaft"] = S * SE3.Rz(N_TURNS * 2 * pi * s)
        yield from arm.Frame(q)
    cue_state["shaft"] = None                                       # one cue now - it moves with the handle
    station.holding = None
    for s in np.linspace(0, 1, 15):
        station.gripper.opening = s
        yield from Frame()
    yield from Station_Play(rtb.jtraj(station.robot.q, station.Release_Up(), 30).q)     # hand lifts off the cue
    Set_Status("Assembly: cue assembled - Doosan lifts it upright")
    T_now = arm.Tip()
    yield from Arm_Line(arm, T_now, SE3(0, 0, 0.12) * T_now)
    yield from Arm_Line(arm, arm.Tip(), arm.Carry(), q_carry)
    yield from Station_Play(rtb.jtraj(station.robot.q, Q_REST, 80).q)
    avoid["station"] = True


CHALK_ROOM = 1.6                       # the cue lies along the rim while it is chalked - keep the others well clear


def Chalk_Cue(k):
    """Chalking (Test_Meca500_v1_Chalk): robot k holds its cue tip still at the
    station and the station Meca500 chalks it."""
    arm = Arm(k)
    spot, roll, q_seed = PRESENT[k]
    Set_Status(f"Chalk: {NAMES[k]} drives to the chalking station")
    yield from Drive_Convoy(k, tuple(spot))
    yield from Make_Room(k, CHALK_ROOM)
    q_carry = np.array(arm.q, dtype=float)
    if roll is None:                                                 # joints given (Meca500)
        q_p = np.array(q_seed)
        T_p = arm.Tip(q_p)
    else:
        T_p = Chalk_Present_Pose(roll)
        q_p = arm.IK(T_p, np.array(q_seed))
    T_back = T_p * SE3.Tz(-0.10)                                     # 10 cm back along the cue
    T_up = SE3(0, 0, 0.10) * T_back
    # the way out (tip -> back -> up) worked out by RMRC from the presenting pose, so the way in
    # is the same path played backwards - always the same arm shape, nothing to guess
    qs_back, _ = PA.RMRC(arm.Tip, q_p, [T_p.interp(T_back, s) for s in rtb.quintic(0, 1, 30).q], DT)
    qs_up, _ = PA.RMRC(arm.Tip, qs_back[-1], [T_back.interp(T_up, s) for s in rtb.quintic(0, 1, 30).q], DT)
    yield from Arm_Play(arm, Clear_Joint_Path(arm, q_carry, qs_up[-1]))      # carry -> above the station
    yield from Arm_Play(arm, qs_up[::-1][1:] + [qs_back[-1]])
    yield from Arm_Play(arm, qs_back[::-1][1:] + [q_p])
    Set_Status(f"Chalk: the station Meca chalks the cue tip of the {NAMES[k]}")
    for q, opening, hold in station.Chalk_Path():
        station.robot.q = q
        if opening is not None:
            station.gripper.opening = opening
        if hold and station.holding != "chalk":
            station.holding = "chalk"
        elif not hold and station.holding == "chalk":
            station.holding, station.T_chalk = None, station.T_chalk_home      # chalk back on its stand
        yield from Frame()
    yield from Arm_Play(arm, qs_back)
    yield from Arm_Play(arm, qs_up)
    yield from Arm_Play(arm, Clear_Joint_Path(arm, np.array(arm.q, dtype=float), q_carry))   # back to carrying


def Game_Shot_Step(i):
    """Shot i of the planned game (Pool_Game.GAME)."""
    shot = GAME[i]
    who = shot["who"]
    what = {"break": "the break", "pot": "a pot", "miss": "a shot", "jump": "a JUMP shot", "masse": "a MASSE"}[shot["kind"]]
    Set_Status(f"{NAMES[who]} drives in for {what}")
    yield from Drive_Convoy(who, tuple(shot["turtle"]))
    yield from Make_Room(who, SHOT_ROOM)
    Settle_Table(shot["layout"])
    state["physics"], state["in_play"] = True, True
    Set_Status(f"{NAMES[who]} - {what}")
    yield from Game_Shot(who, shot)
    while any(p.Moving() for p in BALL_LIST):
        yield from Frame()
    potted = [n for n in shot["pots"] if physics[n].potted is not None]
    side = "U" if who == "U" else "D"
    state["score"][side] += len(potted)
    sc = state["score"]
    Set_Status(f"{NAMES[who]} " + (f"POTS {potted}" if potted else "no pot")
               + f"   (Doosan team {sc['D']} - UR3 {sc['U']})")
    for _ in range(30):
        yield from Frame()
    if i == len(GAME) - 1:
        Set_Status(f"9 is down - the DOOSAN team wins!   (Doosan team {sc['D']} - UR3 {sc['U']})")


def Shot_Name(i, shot):
    who = {"D": "Doosan", "U": "UR3", "M": "Meca500"}[shot["who"]]
    if shot["kind"] == "break":
        return "Break"
    kind = {"pot": "pots " + ", ".join(str(n) for n in shot["pots"]), "miss": "misses",
            "jump": "JUMP shot", "masse": "MASSE"}[shot["kind"]]
    if shot["kind"] in ("jump", "masse") and shot["pots"]:
        kind += " - pots " + ", ".join(str(n) for n in shot["pots"])
    return f"Shot {i + 1}: {who} {kind}"


def Build_Steps():
    """The whole exhibition as a numbered list of steps. Chalk before the break and every
    time the player changes."""
    steps = [("Assembly", Assemble), ("Rack", Rack_The_Balls)]
    previous = None
    for i, shot in enumerate(GAME):
        if shot["who"] != previous:
            steps.append((f"Chalk - {NAMES[shot['who']]}", lambda k=shot["who"]: Chalk_Cue(k)))
        steps.append((Shot_Name(i, shot), lambda i=i: Game_Shot_Step(i)))
        previous = shot["who"]
    return steps


STEPS = Build_Steps()
STEP_NAMES = [name for name, _ in STEPS]
# which step pots each ball (the GUI's ball buttons jump straight to it)
BALL_STEP = {n: STEP_NAMES.index(Shot_Name(i, shot)) for i, shot in enumerate(GAME) for n in shot["pots"]}
state["step"] = None


def Step_Name(i=None):
    i = state["step"] if i is None else i
    return "-" if i is None else f"{i + 1}. {STEP_NAMES[i]}"


def Run_From(first):
    """Run the exhibition from step `first` to the end."""
    for i in range(first, len(STEPS)):
        state["step"] = i
        Set_Status(Step_Name(i))
        yield from Snapshot_Hook(i)
        yield from STEPS[i][1]()
    state["step"] = None


def Potted_Balls():
    """Object balls pocketed in this game so far (for the ticks on the control panel)."""
    if not state.get("in_play"):
        return set()
    return {n for n in range(1, 10) if physics[n].potted is not None}


# ------------------------------------TIME-SKIP: SNAPSHOTS-----------------------------------------------------
#
# A snapshot is everything that moves, saved at the very start of a step: every turtlebot and
# arm, every ball, the rack, the chalk, the shaft. They are made once with
#     python Pool_Robot_Menu.py --make-snapshots
# (it plays the whole exhibition without drawing it) and saved in Pool_Snapshots.json. Jumping to
# a step puts everything back exactly as it was at that moment and carries on from there.

SNAPSHOT_FILE = Path(__file__).resolve().parent / "Pool_Snapshots.json"
snapshots = {}


def _Plain(v):
    if isinstance(v, np.ndarray):
        return {"nd": v.tolist()}
    if isinstance(v, SE3):
        return {"se3": v.A.tolist()}
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_Plain(x) for x in v]
    if isinstance(v, dict):
        return {"dict": [[_Plain(k), _Plain(x)] for k, x in v.items()]}
    return v


def _Back(v):
    if isinstance(v, dict) and "nd" in v:
        return np.array(v["nd"], dtype=float)
    if isinstance(v, dict) and "se3" in v:
        return SE3(np.array(v["se3"]), check=False)
    if isinstance(v, dict) and "dict" in v:
        return {_Back(k): _Back(x) for k, x in v["dict"]}
    if isinstance(v, list):
        return [_Back(x) for x in v]
    return v


BALL_KEEP = ("pos", "v", "w", "R", "_vz", "potted", "floor_z", "vz", "airborne", "cushion_hits", "guide", "homing")


def Take_Snapshot():
    return _Plain({
        "d": [list(d_xyyaw), doosan.q], "u": [list(ur3_p.xyyaw), ur3_p.q], "m": [list(meca_p.xyyaw), meca_p.q],
        "j": [list(jbot.xyyaw), jbot.robot.q],
        "station": [station.robot.q, station.holding, station.gripper.opening, station.T_chalk],
        "shaft": cue_state["shaft"], "rack": rack.T,
        "held": [held.get("what"), held.get("T_rel"), held.get("balls")],
        "balls": {n: {k: getattr(p, k) for k in BALL_KEEP if hasattr(p, k)} for n, p in physics.items()},
        "state": {k: state[k] for k in ("physics", "fast", "score", "in_play")}})


def Restore_Snapshot(i):
    """Put the whole cell back to the start of step i. False if there is no snapshot for it."""
    global d_xyyaw
    if not snapshots:
        Load_Snapshots()
    if i not in snapshots:
        if i == 0:                                          # start of the show = a fresh scene anyway
            Reset_Scene()
            return True
        return False
    S = _Back(snapshots[i])
    d_xyyaw = tuple(S["d"][0])
    doosan.q = S["d"][1]
    for p, key in ((ur3_p, "u"), (meca_p, "m")):
        p.xyyaw = tuple(S[key][0])
        p.Set_Base()
        p.q = S[key][1]
    jbot.xyyaw = tuple(S["j"][0])
    jbot.Set_Base()
    jbot.robot.q = S["j"][1]
    station.robot.q, station.holding, station.gripper.opening, station.T_chalk = S["station"]
    cue_state["shaft"] = S["shaft"]
    rack.Place(S["rack"])
    held.clear()
    held["what"], held["T_rel"] = S["held"][0], S["held"][1]
    if S["held"][2] is not None:
        held["balls"] = S["held"][2]
    for n, attrs in S["balls"].items():
        for k, v in attrs.items():
            setattr(physics[n], k, v)
    state.update(S["state"])
    state["script"] = []
    door_state["script"], door_state["angle"] = [], 0.0
    avoid["station"] = True
    Teach_Unhide()
    Draw()
    return True


def Load_Snapshots():
    try:
        data = json.load(open(SNAPSHOT_FILE))
        if data.get("steps") == STEP_NAMES:
            snapshots.update({int(k): v for k, v in data["snaps"].items()})
        else:
            print("Pool_Snapshots.json is for a different sequence - make it again with --make-snapshots")
    except (FileNotFoundError, ValueError):
        pass


def Snapshot_Hook(i):
    """While making snapshots: save one at the start of each step."""
    if HEADLESS:
        snapshots[i] = Take_Snapshot()
    return
    yield


def Make_Snapshots():
    import time
    t0 = time.time()
    Reset_Scene()
    frames = 0
    for _ in Run_From(0):
        frames += 1
    tmp = SNAPSHOT_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"steps": STEP_NAMES, "snaps": {str(k): v for k, v in snapshots.items()}}, f)
    tmp.replace(SNAPSHOT_FILE)
    print(f"saved {len(snapshots)} snapshots ({frames} frames, {time.time() - t0:.0f} s) -> {SNAPSHOT_FILE.name}")
    print("final score", state["score"], "potted", sorted(Potted_Balls()))


# ------------------------------------MODES-----------------------------------------------------

def Exhibition():
    Set_Status("Exhibition: resetting the table")
    Reset_Scene()
    yield
    yield from Run_From(0)


# ------------------------------------TEACH MODE: the UR3 and one straight shot-----------------------------
#
# Only the UR3, the cue ball and the 8 ball. The 8 ball sits on a straight line from the cue ball
# into the bottom-right corner pocket. The joints are set by hand with the sliders on the control
# panel (GUI_Pool_Robot_v4.py). Every update the panel shows:
#   * the BEST configuration for this shot (IK of the ideal address pose, the UR3's one arm shape),
#   * checks on the cue as it is now - aim, tilt, where on the ball it would hit, how far the
#     tip is from the ball, whether the cue touches the arm - each with a warning and a fix
#     ("joint 1 needs to be about -37 deg"),
# and SHOOT plays a short rock back and a stroke straight along the cue's own axis (RMRC). The
# ball is hit where the cue line really meets it, in the direction the cue really points, so
# the physics follows the configuration: a bad aim misses, an off-centre hit spins, a steep cue
# jumps, a cue that doesn't reach the ball does nothing.

TEACH_POCKET = 1                                               # bottom-right corner (1.104, -0.499)
TEACH_8 = np.array([0.857, -0.252])                           # 35 cm from the corner pocket, on its diagonal
TEACH_GAP = 0.543                                              # cue ball this far behind the 8 ball (gap between them)
TEACH_ELEV = 8.0                                               # ideal cue tilt (deg)
TEACH_MAX_REACH = 0.15                                         # longest stroke - tip must start closer than this
_u = np.array(POCKETS[TEACH_POCKET]) - TEACH_8
_u /= np.linalg.norm(_u)
TEACH_CUE = TEACH_8 - (TEACH_GAP + 2 * R) * _u
TEACH_AIM = float(np.degrees(np.arctan2(_u[1], _u[0])))
teach = {"q": None, "ideal": None, "shot": None, "speed": 0.7, "result": "", "ready": False}


TEACH_FAR = {"D": (40.0, 0.0, 0.0), "M": (44.0, 0.0, 0.0), "J": (48.0, 0.0, 0.0)}   # out of sight
hidden = {"teach": False}


def Teach_Hide():
    """Teach mode shows only the UR3 and the two balls: everything else goes far out of sight,
    and the camera moves in close on the shot."""
    global d_xyyaw
    for k, far in TEACH_FAR.items():
        Set_XY(k, far)
    cue_state["shaft"] = None                                  # (the Doosan's cue goes with it)
    station.Hide(True)
    rack.Place(SE3(52.0, 0.0, 0.0))
    for n, p in physics.items():
        if n not in (0, 8):
            p.Set([56.0, float(n), 0.0])
            p.potted = 0
    hidden["teach"] = True
    me = np.array(ur3_p.xyyaw[:2])
    look = np.array([(me[0] + TEACH_CUE[0]) / 2, (me[1] + TEACH_CUE[1]) / 2, 0.95])
    side = np.array([-_u[1], _u[0], 0.0])                       # square to the shot line,
    if side @ np.array([0.0, -1.0, 0.0]) < 0:                    # on the camera's usual (front) side
        side = -side
    env.set_camera_pose(list(look + 1.7 * side + np.array([0.0, 0.0, 0.85])), list(look))


def Teach_Unhide():
    if hidden.get("teach"):
        station.Hide(False)
        env.set_camera_pose(*CAMERA_OVERVIEW)
        hidden["teach"] = False


def Teach_Address():
    """Ideal cue-tip pose: straight at the cue ball's centre, ADDRESS_GAP behind it, tilted TEACH_ELEV."""
    shot = dict(aim=TEACH_AIM, elev=TEACH_ELEV, side=0.0, height=0.0, roll=PA.CUE_ROLL["U"])
    return Shot_Poses(np.array([*TEACH_CUE, FELT_Z + R]), shot, R, ADDRESS_GAP)[2]


def Teach_Balls():
    for n, p in physics.items():                               # every other ball is out of sight
        if n not in (0, 8):
            p.Set([56.0, float(n), 0.0])
            p.potted = 0
    physics[0].Set([*TEACH_CUE, FELT_Z + R])
    physics[8].Set([*TEACH_8, FELT_Z + R])


def Teach_Setup():
    """UR3 to the best spot behind the shot line; everything else hidden; camera in close."""
    global d_xyyaw
    Reset_Scene()
    state["physics"], state["in_play"] = True, False
    Teach_Balls()
    arm = Arm("U")
    T_addr = Teach_Address()
    from Make_Arm_Plan import Spots
    home_xyyaw = ur3_p.xyyaw
    best = None
    for spot in Spots(TEACH_CUE, _u):
        ur3_p.xyyaw = spot
        ur3_p.Set_Base()
        q = arm.Shape_IK(T_addr)
        if q is None:
            q = arm.IK(T_addr)
        if q is None or Cue_Clearance(arm, q) < PA.CLEARANCE:
            continue
        prof = Punch_Stroke_Profile(DT, contact_s=-ADDRESS_GAP, tip_speed=1.0, back=0.12, follow=0.03,
                                    n_rocks=1, rock_amp=0.05)              # the stroke SHOOT will play must fit too
        qs, err = PA.RMRC(arm.Tip, q, [T_addr] + [T_addr * SE3.Tz(-s) for s in prof], DT)
        if err < 0.005 and min(Cue_Clearance(arm, x) for x in qs[::3]) >= PA.CLEARANCE:
            best = (spot, q)
            break
    ur3_p.xyyaw, q_ideal = best if best else (ur3_p.xyyaw, None)
    
    q_carry = ur3_p.IK(ur3_p.Carry_Tip(), ur3_p.q)
    if q_carry is not None:
        ur3_p.q = q_carry
    else:
        # No valid teach spot was found: put the UR3 back where it started.
        ur3_p.xyyaw = home_xyyaw
        q_ideal = None
    ur3_p.xyyaw = (0.0, 0.84, -90.0)
    ur3_p.Set_Base()
    Teach_Hide()
    teach.update(q=np.array(ur3_p.q, dtype=float), ideal=q_ideal, shot=None, result="", ready=True)
    Draw()


def Teach_Checks(q):
    """[(ok, text)] for the cue as the sliders have it now, plus the joint values the arm needs."""
    arm = Arm("U")
    q = np.asarray(q, dtype=float)
    T = arm.Tip(q)
    d, tip = T.R[:, 2], T.t
    ball = np.array([*TEACH_CUE, FELT_Z + R])
    out = []
    aim = np.degrees(np.arctan2(d[1], d[0]))
    elev = np.degrees(np.arcsin(np.clip(-d[2], -1, 1)))
    aim_err = (aim - TEACH_AIM + 180) % 360 - 180
    rel = ball - tip
    along = rel @ d                                            # how far ahead of the tip the ball centre is
    offset = np.linalg.norm(rel - along * d)                   # how far the cue line passes from the centre
    reach = along - np.sqrt(max(R ** 2 - offset ** 2, 0.0))    # tip -> ball surface along the cue
    # suggestion: the nearest configuration that IS the ideal address pose
    sol = ur3_p.kin.ikine_LM(ur3_p.robot.base.inv() * Teach_Address() * ur3_p.tool.inv(), q0=q,
                             joint_limits=True, slimit=1)
    q_fix = sol.q if sol.success else teach["ideal"]

    def Fix(*joints):
        if q_fix is None:
            return ""
        tips = [f"J{j + 1} about {np.degrees(q_fix[j]):.0f} deg (now {np.degrees(q[j]):.0f})"
                for j in joints if abs((q_fix[j] - q[j] + pi) % (2 * pi) - pi) > np.radians(3)]
        return ("  ->  set " + ", ".join(tips)) if tips else ""

    if elev < -2:
        out.append((False, f"Cue points UP ({-elev:.0f} deg) - the tip would pass over the ball. Tilt it down to about {TEACH_ELEV:.0f} deg." + Fix(1, 2, 3, 4)))
    elif elev > 20:
        out.append((False, f"Cue tilt {elev:.0f} deg is steep - the ball will jump / swerve. Keep it under 15 deg (best {TEACH_ELEV:.0f})." + Fix(1, 2, 3, 4)))
    else:
        out.append((True, f"Cue tilt {elev:.1f} deg (best {TEACH_ELEV:.0f}, keep 0-15)"))
    if abs(aim_err) > 1.0:
        out.append((False, f"Aim is {aim_err:+.1f} deg off the line to the 8 ball - it needs to be within 1 deg." + Fix(0, 5)))
    else:
        out.append((True, f"Aim {aim_err:+.1f} deg off the line - good"))
    if along <= 0 or offset >= R:
        out.append((False, f"The cue line misses the cue ball by {max(offset - R, 0) * 100:.1f} cm - nothing would be hit." + Fix(0, 1, 2)))
    elif offset > 0.3 * R:
        out.append((False, f"Tip would hit {offset * 1000:.0f} mm off centre - the ball will spin and drift. Keep it under {0.3 * R * 1000:.0f} mm." + Fix(1, 2, 3)))
    else:
        out.append((True, f"Hits {offset * 1000:.0f} mm from the centre of the cue ball - good"))
    if along > 0 and offset < R:
        if reach < 0:
            out.append((False, "Tip is INSIDE the cue ball - pull the cue back." + Fix(1, 2)))
        elif reach > TEACH_MAX_REACH:
            out.append((False, f"Tip is {reach * 100:.0f} cm from the ball - the stroke only reaches {TEACH_MAX_REACH * 100:.0f} cm. Move it closer." + Fix(1, 2)))
        else:
            out.append((True, f"Tip {reach * 100:.1f} cm behind the ball (the stroke reaches {TEACH_MAX_REACH * 100:.0f} cm)"))
    gap = Cue_Clearance(arm, q)
    if gap < PA.CLEARANCE:
        out.append((False, f"The cue goes through the robot's own arm ({gap * 100:.1f} cm) - change the wrist (J4-J6)." + Fix(3, 4, 5)))
    return out, (reach, offset, along)


def Teach_Shot():
    """Rock back once and stroke straight along the cue's own axis (RMRC); the ball is hit where
    the cue line actually meets it, in the direction the cue actually points."""
    arm = Arm("U")
    q0 = np.array(ur3_p.q, dtype=float)
    T0 = arm.Tip(q0)
    d = T0.R[:, 2]
    _, (reach, offset, along) = Teach_Checks(q0)
    hits = along > 0 and offset < R and 0 <= reach <= TEACH_MAX_REACH
    contact_s = -reach if hits else -TEACH_MAX_REACH            # (no ball in reach: full-length stroke)
    profile = Punch_Stroke_Profile(DT, contact_s=contact_s, tip_speed=teach["speed"], back=0.12, follow=0.03,
                                   n_rocks=1, rock_amp=0.05)
    poses = [T0] + [T0 * SE3.Tz(-s) for s in profile]
    qs = RMRC_Path(arm, poses)
    if qs is None:                                              # RMRC can't follow - IK sample by sample instead
        qs, q = [], q0
        for T in poses[1:]:
            q = arm.IK(T, q)
            if q is None:
                teach["result"] = "The arm can't make a straight stroke from this pose (out of reach / joint limits)."
                Set_Status("Teach: " + teach["result"])
                return
            qs.append(q)
    hit_at = int(np.argmax(profile <= contact_s + 1e-9))
    Set_Status(f"Teach: UR3 shoots at {teach['speed']:.1f} m/s")
    for i, q in enumerate(qs):
        if hits and i == hit_at:
            tip = arm.Tip(np.asarray(q)).t
            physics[0].Hit(CUE_TO_BALL * teach["speed"] * d, tip - physics[0].pos)
        yield from Arm_Play(arm, [q])
    yield from Arm_Play(arm, rtb.jtraj(np.array(ur3_p.q, dtype=float), q0, 40).q)
    while any(p.Moving() for p in BALL_LIST):
        yield from Frame()
    if not hits:
        teach["result"] = "Missed - the cue never touched the cue ball."
    elif physics[0].potted is not None and physics[8].potted is not None:
        teach["result"] = "8 ball in - but the cue ball went in too (scratch)."
    elif physics[8].potted == TEACH_POCKET:
        teach["result"] = "8 ball POTTED in the corner - straight shot!"
    elif physics[8].potted is not None:
        teach["result"] = "8 ball potted - in a different pocket than aimed."
    elif physics[0].potted is not None:
        teach["result"] = "Scratch - the cue ball went in."
    else:
        e = np.linalg.norm(physics[8].pos[:2] - np.array(POCKETS[TEACH_POCKET]))
        teach["result"] = f"8 ball missed - it stopped {e * 100:.0f} cm from the pocket. Check the aim / hit point."
    Set_Status("Teach: " + teach["result"])


def Teach():
    Set_Status("Teach mode: setting up the UR3 straight shot...")
    Teach_Setup()
    Set_Status("Teach mode: move the joint sliders, then SHOOT")
    yield
    while True:
        if teach["shot"] is not None:
            yield from teach["shot"]
            teach["shot"] = None
            teach["q"] = np.array(ur3_p.q, dtype=float)
        elif teach.pop("reset_balls", False):
            Teach_Balls()
            teach["result"] = ""
            yield from Frame()
        else:
            yield from Teach_Hold()


def Teach_Hold():
    ur3_p.q = np.array(teach["q"], dtype=float)
    yield from Frame()

def Controller():
    """Controller mode: UR3 controlled directly by a games console controller."""

    Set_Status("Controller mode: use the gamepad to move the UR3")

    # Same isolated scene setup as Teach mode.
    Reset_Scene()
    state["physics"], state["inplay"] = True, False
    Teach_Balls()

    # Place the UR3 near the top-middle pocket.
    ur3_p.xyyaw = (0.0, 0.84, -90.0)
    ur3_p.Set_Base()

    Teach_Hide()

    # Fixed overhead operator camera.
    # Directly above the robot-ball line, looking down the table.
    camera_position = np.array([0.0, 0.9, 2.3])
    camera_look_at = np.array([0.0, 0.1, 0.0])

    env.set_camera_pose(camera_position, look_at=camera_look_at)

    # Start from the current Teach configuration if available.
    if teach.get("q") is None:
        teach["q"] = np.array(ur3_p.q, dtype=float)

    ur3_p.q = np.array(teach["q"], dtype=float)

    teach.update(
        q=np.array(ur3_p.q, dtype=float),
        ideal=None,
        shot=None,
        result="Controller mode ready",
        ready=True,
    )

    Draw()

    # ---- Create the gamepad controller ----
    try:
        pad = GamepadController(
            joystick_index=0,
            deadzone=0.15,
            joint_speed=np.radians(45.0),
            smoothing=0.25,
        )
    except RuntimeError as e:
        Set_Status(f"Controller mode: {e}")
        print(e)

        while True:
            yield from Frame()

    shoot_request = {"requested": False}

    def reset_ur3():
        """Return the UR3 to its upright carry pose."""
        q_reset = ur3_p.IK(ur3_p.Carry_Tip(), ur3_p.q)

        if q_reset is not None:
            ur3_p.q = np.array(q_reset, dtype=float)

    def request_shot():
        """Set the shoot request; the main Controller loop performs the shot."""
        shoot_request["requested"] = True

    def Controller_Shot(power=1.0):
        """
        Shoot straight along the cue's current direction.

        This does not check whether the shot is ideal; it simply strokes
        along the cue line and hits the cue ball if the cue reaches it.
        """
        arm = Arm("U")

        q0 = np.array(ur3_p.q, dtype=float)
        T0 = arm.Tip(q0)

        # Cue direction and current cue-tip position.
        d = T0.R[:, 2]
        tip0 = T0.t

        # Cue-ball position.
        ball = np.array(physics[0].pos, dtype=float)

        # Geometry of the cue line relative to the ball.
        rel = ball - tip0
        along = float(np.dot(rel, d))
        perp = rel - along * d
        offset = float(np.linalg.norm(perp))

        # Distance from the cue tip to the ball surface along the cue line.
        reach = along - np.sqrt(max(R ** 2 - offset ** 2, 0.0))

        if along > 0 and offset < R and reach > 0:
            contacts = -reach
        else:
            contacts = -0.30

        # Backswing, then forward stroke.
        profile = Punch_Stroke_Profile(DT, contacts, power)

        poses = [T0] + [T0 * SE3.Tz(-float(s)) for s in profile]

        # Follow the cue line with RMRC where possible.
        qs = RMRC_Path(arm, poses)

        if qs is None:
            qs = IK_Line(arm, poses)

        if qs is None:
            Set_Status("Controller: cannot make a straight shot from this pose")
            return

        Set_Status(f"Controller: shooting at {power:.1f} m/s")

        hit = False

        for q in qs:
            q = np.asarray(q, dtype=float)

            # Current cue-tip position.
            tip = arm.Tip(q).t

            # Distance from the tip to the ball centre along the cue line.
            along_now = float(np.dot(ball - tip, d))

            # Contact occurs when the tip reaches the ball surface.
            if not hit and along_now <= R + 1e-6:
                physics[0].Hit(power * d, tip - ball)
                hit = True

            yield from arm.Frame(q)

        # Let the balls settle.
        while any(p.Moving for p in BALL_LIST):
            yield from Frame()

        Set_Status("Controller: shot complete")
    while True:
        # Process pygame events and read the controller.
        pad.update()

        # 1. Direct joint velocities from the gamepad.
        dq = pad.read_joint_velocities()

        # 2. Current UR3 joint angles.
        q_now = np.array(ur3_p.q, dtype=float)

        # 3. Integrate joint velocities.
        q_next = q_now + dq * DT

        # 4. Keep the UR3 inside its joint limits.
        q_lo, q_hi = ur3_p.robot.qlim
        q_next = np.clip(q_next, q_lo, q_hi)

        ur3_p.q = q_next

        # 5. Controller buttons.
        pad.handle_buttons(
            reset_callback=reset_ur3,
            shoot_callback=request_shot,
        )

        # 6. Perform the shot if Start was pressed.
        if shoot_request["requested"]:
            shoot_request["requested"] = False
            yield from Controller_Shot(power=1.0)

        # 7. Advance and draw one frame.
        yield from Frame()

MODES = {
    "exhibition": Exhibition,
    "teach": Teach,
    "controller": Controller,
}
LIGHT_FOR = {"IDLE": "amber", "RUNNING": "green", "ESTOPPED": "red", "RESET": "amber", "DOOR_OPEN": "red"}


def Handle_Request(req):
    """Act on one button press. Returns False when the program should end."""
    st = control["state"]
    if req == "estop":
        if st != "ESTOPPED":
            control["state"] = "ESTOPPED"
            Set_Status("E-STOP PRESSED - all motion stopped. Press Reset to acknowledge.")
    elif req == "reset":
        if st == "ESTOPPED":
            control["state"] = "RESET"
            Set_Status("e-stop acknowledged. Press Resume to continue, or pick a mode to start again.")
        else:
            Set_Status("Reset only does something after an E-STOP.")
    elif req == "resume":
        if st == "ESTOPPED":
            Set_Status("Press Reset first - Resume is locked until the e-stop is acknowledged.")
        elif st == "RESET" and control["sequence"] is not None:
            control["state"] = "RUNNING"
            Set_Status(f"resuming {control['mode']}")
        elif st == "RESET":
            control["state"] = "IDLE"
            Set_Status("nothing to resume - pick a mode")
    elif req in MODES:
        if st == "ESTOPPED":
            Set_Status("E-STOP active - press Reset first.")
        elif st == "DOOR_OPEN" or (st == "RUNNING" and control["mode"] != "teach"):
            Set_Status(f"busy with {control['mode']} - press E-STOP to stop it first")
        else:                                                   # IDLE or RESET: start fresh
            control["state"], control["mode"] = "RUNNING", req
            control["sequence"] = MODES[req]()
            door_state["t"], door_state["demo_done"] = 0.0, False
            Set_Status(f"starting {req}")
    elif isinstance(req, tuple) and req[0] == "goto":            # time-skip
        i = req[1]
        if st in ("ESTOPPED", "DOOR_OPEN"):
            Set_Status("Can't skip while the cell is stopped")
        elif not Restore_Snapshot(i):
            Set_Status("No snapshots yet - run:  python Pool_Robot_Menu.py --make-snapshots")
        else:
            control["state"], control["mode"] = "RUNNING", "exhibition"
            control["sequence"] = Run_From(i)
            state["step"] = i
            Set_Status(Step_Name(i))
            door_state["t"], door_state["demo_done"] = 0.0, i > 0          # the door demo only on a full run
    elif req == "pause":                                         # disarmed from the control panel
        if st == "RUNNING":
            control["state"] = "RESET"
            Set_Status(f"paused - {Step_Name()}")
    elif req == "door":
        if not door_state["script"]:
            door_state["script"] = Door_Visit()
    elif req == "teach_shoot":
        if control["mode"] == "teach" and st == "RUNNING" and teach["shot"] is None:
            teach["shot"] = Teach_Shot()
        else:
            Set_Status("SHOOT works in Teach mode while the cell is ARMED")
    elif req == "teach_reset":
        if control["mode"] == "teach":
            teach["reset_balls"] = True
    elif req == "wall_estop":
        if not door_state["script"]:
            door_state["script"] = Wall_EStop_Visit()
    elif req == "controller":
        if st == "ESTOPPED":
            Set_Status("E-STOP active - press Reset first")

        elif st == "DOOR_OPEN":
            Set_Status("Door open - shut the door before starting Controller mode")

        elif st == "RUNNING":
            Set_Status(f"Busy with {control['mode']} - press E-STOP to stop it first")

        else:
            # IDLE or RESET: start Controller mode
            control["state"] = "RUNNING"
            control["mode"] = "controller"
            control["sequence"] = Controller()

            door_state["t"] = 0.0
            door_state["demo_done"] = False

            Set_Status("Starting Controller mode")    
    elif req == "quit":
        Set_Status("closing")
        env.close()
        return False
    return True


Reset_Scene()
if not HEADLESS:
    Load_Snapshots()
control["state"] = "IDLE"
Set_Status("ready - click 1. Exhibition, 2. Teach or 3. Controller")


def Step_Once():
    """One screen update: button requests, the door interlock, then (only while RUNNING) the
    next SPEED_UP frames of whatever is running. Returns False when the program should end.
    The tkinter control panel (GUI_Pool_Robot_v4.py) calls this every screen update."""
    req, control["request"] = control["request"], None
    if req is not None and not Handle_Request(req):
        return False
    # door sensor interlock: pause everything while the door is open, carry on when it shuts
    Door_Step()
    door_open = door_state["angle"] > DOOR_SWITCH_DEG
    if control["state"] == "RUNNING" and door_open:
        control["state"] = "DOOR_OPEN"
        Set_Status("DOOR OPEN - door sensor tripped, all robots paused until the door is shut")
    elif control["state"] == "DOOR_OPEN" and not door_open:
        control["state"] = "RUNNING"
        Set_Status(f"door shut - resuming {Step_Name()}")
    if control["state"] == "RUNNING" and control["mode"] == "exhibition" and not door_state["demo_done"]:
        door_state["t"] += DT                                  # the person turns up a few seconds in
        if door_state["t"] >= DOOR_DEMO_AT:
            door_state["demo_done"] = True
            door_state["script"] = Door_Visit()
    if control["state"] == "RUNNING" and control["sequence"] is not None:
        try:
            for _ in range(SPEED_UP * (RACK_SPEED_UP if state.get("fast") else 1)):   # frames per screen update
                next(control["sequence"])
        except StopIteration:
            control["sequence"] = None
            control["state"] = "IDLE"
            Set_Status(f"{control['mode']} finished - pick a mode")
    Set_Light(LIGHT_FOR[control["state"]])
    return True


if __name__ == "__main__":
    if HEADLESS:
        Make_Snapshots()
    else:                                               # the pop-up control panel (GUI_Pool_Robot_v4.py)
        import runpy
        sys.modules["Pool_Robot_Menu"] = sys.modules["__main__"]   # the panel uses THIS running cell
        runpy.run_path(str(Path(__file__).resolve().parent / "GUI_Pool_Robot_v4.py"), run_name="__main__")
