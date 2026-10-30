# ------------------------------------POOL ROBOT MENU--------------------------------------------------------
# The main file of the 9-ball pool workcell. It builds the whole scene in Swift (track, table, the four
# robots on their turtlebots, the service station, balls, room, door and e-stops) and runs the three modes:
# Exhibition, Teach and Controller. Run this file: it opens the pop-up control panel (GUI_Pool_Robot_v4.py),
# which drives everything here.

import sys
import copy
import pickle
from pathlib import Path
import swift
import numpy as np
import spatialgeometry as sg
from math import pi
from spatialmath import SE3
import roboticstoolbox as rtb

from Doosan_Robot import Doosan_Robot
from ir_support_extra_robots.robots import Turtlebot3Waffle
from Pool_Parts import Part_Path, Part_Mesh
from Pool_Objects import Cue_Ball, Cue_Stick
from Pool_Motion import Holder_Tool, Solve_IK, Punch_Stroke_Profile
from Pool_Scene import (TRACK_NAME, TRACK_POSE, CHANNEL_Y, INNER_X, INNER_Y, RIM_TOP_Z,
                        TABLE_Z, FELT_Z, PLAY_AREA, GROOVE_HEIGHT, POCKETS, POCKET_RADIUS,
                        POCKET_FLOOR_Z, Turtle_Pose, Robot_Base_On, Drive_Path, _Loop, _Loop_S, _Loop_Point)
from Jbot_Racker import Jbot_Mobile, Ball_Rack, Rack_Slots, Tool_Down, Jbot_IK
from Pool_Players import UR3_Player, Meca_Player, Carry_Pose, HANDLE_PIN, GRIP_FROM_BUTT
from Meca_Jumper import CARRY_Q as MECA_CARRY_Q, Short_Stroke_Profile
from Pool_Game import GAME
from Pool_Trick_Shots import Pool_Ball, Table_Step, MU_ROLL, Shot_Poses
from Pool_Station import (Station, Q_REST, Q_GRIP, Q_HOLD, Chalk_Present_Pose, Assembly_Pose, HANDLE_PIN as PIN, N_TURNS, LINE_UP,
                          PRESENT, ASM_PRESENT)
import Pool_Arm as PA


# ------------------------------------CONSTANTS--------------------------------------------------------------
# Timing, speeds and fixed positions shared by the whole file. DT is one animation frame; SPEED_UP plays
# several frames per screen update so the show runs faster without changing the motion itself. The Doosan
# values describe how it holds and carries its cue; the Jbot values describe where it parks and how the
# balls start in the pockets before racking.

DT = 0.02
SPEED_UP = 5
PHYSICS_SUBSTEPS = 5
JOINT_SPEED = 2.0

CARRY_GRIP_POINT = (0.40, 0.0, 0.65)
DOOSAN_CARRY_Q = np.array([0.0387056095326721, -0.23887131728965105, 1.3924895203856824, -0.09528474100458828,
                           0.41886485571446075, -3.0545016228680386])
UR3_CARRY_Q = np.array([-1.0193767201825428, -2.1884716506699915, 2.3575292657865123, 2.972535017943372,
                        -0.551419605786224, -1.5707963012163848])
JBOT_CARRY_Q = np.array([1.5707547347319735, 0.1829386853718597, -1.9277979711070055, 0.0, 7.243522129218264e-05,
                         1.3966525106800218, -1.1312840113480187])
DOOSAN_LOOP = (1.72, CHANNEL_Y)
DOOSAN_WAIT = (-1.72, 0.45, -90.0)
CUE_BALL_SPOT = (-0.60, 0.20)
ADDRESS_GAP = 0.02
APPROACH = 0.15
CUE_TO_BALL = 1.4
LIFT = 0.15
DOOSAN_VMAX = np.radians([150, 150, 180, 225, 225, 225])

RACK_SPEED_UP = 5
TRAY_PARK = (INNER_X, 0.0, -90.0)
RACK_START_APEX = 0.83
FOOT_SPOT = 0.534
CARRY_LOCAL = SE3(0.25, 0.0, 0.20) * SE3.Rx(pi)
ABOVE = RIM_TOP_Z + 0.10

POCKET_PARKS = {0: (INNER_X, 0.501, -90.0), 1: (INNER_X, -0.499, -90.0),
                2: (-1.096, INNER_Y, 0.0), 3: (-1.096, -INNER_Y, 180.0),
                4: (0.004, INNER_Y, 0.0), 5: (0.004, -INNER_Y, 180.0)}

POCKET_STACKS = {0: [2, 1], 1: [4, 3], 4: [9], 5: [5], 2: [7, 6], 3: [8]}
PICK_ORDER = [1, 2, 3, 4, 9, 5, 6, 7, 8]


# ------------------------------------SCENE------------------------------------------------------------------
# Creates every object once. When making snapshots (--make-snapshots) a stand-in for Swift is used so the
# whole show can be played without opening a browser. Each ball has a mesh (what you see) and a physics
# object (what moves it).

HEADLESS = "--make-snapshots" in sys.argv


class _No_Swift(swift.Swift):
    def __init__(self): pass
    def launch(self, **kw): pass
    def add(self, *a, **kw): pass
    def step(self, *a, **kw): pass
    def set_camera_pose(self, *a, **kw): pass
    def close(self): pass


env = _No_Swift() if HEADLESS else swift.Swift()
env.launch(realtime=True)
CAMERA_OVERVIEW = ([3.6, -4.6, 3.6], [0.4, -0.6, 0.8])
env.set_camera_pose(*CAMERA_OVERVIEW)

env.add(Part_Mesh(TRACK_NAME, pose=TRACK_POSE))
env.add(Part_Mesh("Pool_Table", pose=SE3(0, 0, TABLE_Z)))

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

jbot = Jbot_Mobile(TRAY_PARK)
jbot.Add_To_Env(env)

UR3_HOME = (-1.72, -0.45, 90.0)
MECA_HOME = (-INNER_X, 0.0, 90.0)
ur3_p = UR3_Player(UR3_HOME)
ur3_p.Add_To_Env(env)
meca_p = Meca_Player(MECA_HOME)
meca_p.Add_To_Env(env)

station = Station()
station.Add_To_Env(env)
cue_state = {"shaft": None}

ball_meshes = {n: Cue_Ball(str(Part_Path(f"{n}Ball" if n else "Cue_Ball"))) for n in range(10)}
for mesh in ball_meshes.values():
    mesh.Add_To_Env(env)
R = ball_meshes[0].radius
physics = {n: Pool_Ball([0, 0, FELT_Z + R], R, mu_roll=MU_ROLL, jaws=True, bounds=PLAY_AREA, pockets=POCKETS,
                        pocket_radius=POCKET_RADIUS, pocket_floor_z=POCKET_FLOOR_Z + R)
           for n in ball_meshes}
BALL_LIST = [physics[n] for n in sorted(physics)]

rack = Ball_Rack(R, FELT_Z)
rack.Add_To_Env(env)


# ------------------------------------CONTROL STATE AND STATUS-----------------------------------------------
# One dictionary holds the cell's state (IDLE, RUNNING, ESTOPPED, RESET, DOOR_OPEN), the sequence being
# played and the next button request. Set_Status keeps the message the control panel shows.

control = {"state": "IDLE", "sequence": None, "mode": None, "request": None}


def Set_Status(text):
    control["text"] = text
    if not HEADLESS:
        print(f"[{control['state']}] {text}")


# ------------------------------------STACK LIGHT IN THE CELL------------------------------------------------
# A red / amber / green light tower beside the table. Only one lamp is lit at a time; the others are moved
# below the floor, which is cheaper than recolouring meshes every frame.

LIGHT_XY = (1.98, -1.30)
env.add(sg.Cylinder(radius=0.015, length=1.2, color=[0.3, 0.3, 0.3, 1.0], pose=SE3(*LIGHT_XY, 0.6)))
_lamp_z = {"red": 1.32, "amber": 1.26, "green": 1.20}
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
    for name, lamp in _lamps_lit.items():
        lamp.T = SE3(*LIGHT_XY, _lamp_z[name] if name == lit else -5).A


# ------------------------------------ROOM, DOOR AND DOOR SENSOR---------------------------------------------
# The cell sits inside a see-through room whose only entrance is a door with a sensor. Opening the door
# pauses every robot (an interlock), and shutting it lets them carry on. A person walks up and opens it a
# few seconds into the Exhibition to show this, played as a short script of walk / swing / wait steps.

ROOM_X, ROOM_Y, ROOM_H, WALL_T = 2.6, 1.9, 2.4, 0.05
DOOR_HINGE, DOOR_W, DOOR_H = (1.0, -ROOM_Y), 0.8, 2.05
DOOR_DEMO_AT = 3.0
WALL_COLOUR = [0.85, 0.85, 0.8, 0.25]
for cx, cy, sx, sy in [(-ROOM_X, 0, WALL_T, 2 * ROOM_Y), (ROOM_X, 0, WALL_T, 2 * ROOM_Y),
                       (0, ROOM_Y, 2 * ROOM_X, WALL_T),
                       ((-ROOM_X + DOOR_HINGE[0]) / 2, -ROOM_Y, DOOR_HINGE[0] + ROOM_X, WALL_T),
                       ((DOOR_HINGE[0] + DOOR_W + ROOM_X) / 2, -ROOM_Y, ROOM_X - DOOR_HINGE[0] - DOOR_W, WALL_T)]:
    env.add(sg.Cuboid(scale=[sx, sy, ROOM_H], color=WALL_COLOUR, pose=SE3(cx, cy, ROOM_H / 2)))
env.add(sg.Cuboid(scale=[DOOR_W, WALL_T, ROOM_H - DOOR_H], color=WALL_COLOUR,
                  pose=SE3(DOOR_HINGE[0] + DOOR_W / 2, -ROOM_Y, (ROOM_H + DOOR_H) / 2)))
door = sg.Cuboid(scale=[DOOR_W, 0.04, DOOR_H], color=[0.55, 0.35, 0.2, 1.0])
env.add(door)
sensor_ok = sg.Cuboid(scale=[0.06, 0.03, 0.06], color=[0.0, 0.9, 0.0, 1.0])
sensor_trip = sg.Cuboid(scale=[0.06, 0.03, 0.06], color=[1.0, 0.0, 0.0, 1.0])
env.add(sensor_ok)
env.add(sensor_trip)
person = Part_Mesh("personMaleConstruction")
env.add(Part_Mesh("ProximitySensor", pose=SE3(DOOR_HINGE[0] + DOOR_W + 0.12, -ROOM_Y + 0.08, DOOR_H - 0.38)))
env.add(Part_Mesh("CautionSign", pose=SE3(DOOR_HINGE[0] - 0.4, -ROOM_Y - 0.15, 0.0)))
door_state = {"angle": 0.0, "script": [], "t": 0.0, "demo_done": False}
PERSON_FAR, PERSON_AT_DOOR, PERSON_IN_DOOR = (1.95, -3.4), (1.95, -2.65), (1.4, -2.0)


def Place_Door():
    a = np.radians(door_state["angle"])
    door.T = (SE3(DOOR_HINGE[0], DOOR_HINGE[1] - 0.03, DOOR_H / 2) * SE3.Rz(-a) * SE3(DOOR_W / 2, 0, 0)).A
    tripped = door_state["angle"] > DOOR_SWITCH_DEG
    at = SE3(DOOR_HINGE[0] + DOOR_W + 0.12, -ROOM_Y - 0.05, DOOR_H - 0.45)
    sensor_trip.T = (at if tripped else SE3(0, 0, -5)).A
    sensor_ok.T = (SE3(0, 0, -5) if tripped else at).A


def Place_Person(xy, yaw_deg=90.0):
    person.T = (SE3(xy[0], xy[1], 0.0) * SE3.Rz(np.radians(yaw_deg))).A


def Walk(a, b, secs):
    n = int(secs / DT)
    return [("p", tuple(np.array(a) + (np.array(b) - np.array(a)) * k / n)) for k in range(1, n + 1)]


def Swing(a, b, secs):
    n = int(secs / DT)
    return [("d", a + (b - a) * k / n) for k in range(1, n + 1)]


def Wait(secs):
    return [("w", None)] * int(secs / DT)


def Door_Visit():
    return (Walk(PERSON_FAR, PERSON_AT_DOOR, 1.5) + Swing(0, 85, 1.0) + Walk(PERSON_AT_DOOR, PERSON_IN_DOOR, 0.8)
            + Wait(1.5) + Walk(PERSON_IN_DOOR, PERSON_AT_DOOR, 0.8) + Swing(85, 0, 1.0)
            + Walk(PERSON_AT_DOOR, PERSON_FAR, 1.5))


def Door_Step():
    if door_state["script"]:
        kind, value = door_state["script"].pop(0)
        if kind == "p":
            Place_Person(value)
        elif kind == "d":
            door_state["angle"] = value
        elif kind == "e":
            Handle_Request("estop")
            Set_Status("Wall E-STOP pressed - press RESET")
    Place_Door()


DOOR_SWITCH_DEG = 3.0
Place_Door()
Place_Person(PERSON_FAR)
env.add(person)


# ------------------------------------WALL E-STOP OUTSIDE THE DOOR -------------------------------------------
# A wall-mounted emergency stop outside the room. A person can walk up and press it; it does exactly what
# the panel's E-STOP does, so either one stops the whole cell.

WALL_ESTOP_XY = (0.45, -ROOM_Y - WALL_T / 2)
WALL_ESTOP_Z = 1.30
PERSON_AT_ESTOP = (0.45, -2.35)
wall_estop = Part_Mesh("emergencyStopWallMounted", pose=SE3(*WALL_ESTOP_XY, WALL_ESTOP_Z) * SE3.Rz(-pi / 2))
env.add(wall_estop)


def Wall_EStop_Visit():
    return (Walk(PERSON_FAR, PERSON_AT_ESTOP, 1.8) + Wait(0.4) + [("e", None)]
            + Wait(1.0) + Walk(PERSON_AT_ESTOP, PERSON_FAR, 1.8))


# ------------------------------------STATE AND DRAWING------------------------------------------------------
# Draw puts every robot, cue, ball and the rack where it belongs for the current frame. Frame draws once and
# then waits (yield), so every motion is a generator of frames: the main loop simply stops asking for frames
# when the cell is stopped, which freezes everything exactly where it is.

state = {"physics": False, "fast": False}
held = {"what": None, "T_rel": None}


def Doosan_Carry_Pose(base):
    return Carry_Pose(base, CARRY_GRIP_POINT, cue.length)


def Draw(dq=None, jq=None):
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
    if held["what"] is not None:
        T_obj = jbot.Tool_Pose() * held["T_rel"]
        if held["what"] == "rack":
            rack.Place(T_obj)
            for n, T_rel in held.get("balls", {}).items():
                physics[n].pos = (T_obj * T_rel).t
        else:
            physics[held["what"]].pos = T_obj.t

    if state["physics"]:
        for _ in range(PHYSICS_SUBSTEPS):
            Table_Step(BALL_LIST, DT / PHYSICS_SUBSTEPS)
    for n, mesh in ball_meshes.items():
        mesh.Place(physics[n].pos, physics[n].R)


def Frame(dq=None, jq=None):
    Draw(dq, jq)
    yield


# ------------------------------------JBOT JOINT AND STRAIGHT-LINE MOVES--------------------------------------
# Small building blocks for the Jbot: a smooth joint move whose length is set by the biggest joint change (so
# no joint goes too fast) and a straight-line move of the suction cup by IK at every step.

def Steps_For(q_from, q_goal, steps):
    biggest = np.max(np.abs(np.asarray(q_goal) - np.asarray(q_from)))
    return max(steps, int(1.9 * biggest / (JOINT_SPEED * DT)) + 1)


def Jbot_Joints(q_goal, steps=30):
    for q in rtb.jtraj(jbot.robot.q, q_goal, Steps_For(jbot.robot.q, q_goal, steps)).q:
        yield from Frame(jq=q)


def Jbot_Line(T_from, T_to, steps=20, keep_spin=False):
    q = jbot.robot.q
    mask = (1, 1, 1, 1, 1, 1) if keep_spin else (1, 1, 1, 1, 1, 0)
    for s in rtb.quintic(0, 1, steps).q:
        T = T_from.interp(T_to, s)
        q_new = Jbot_IK(jbot.robot, T * jbot.tool.inv(), q, mask=mask)
        if q_new is None:
            Set_Status(f"Jbot IK failed at {np.round(T.t, 3)} - stopping this move")
            return
        q = q_new
        yield from Frame(jq=q)


# ------------------------------------RESET-------------------------------------------------------------------
# Puts the whole cell back to the start of the show: balls stacked in the pockets, the rack on the felt, the
# shaft waiting at the station and every robot at its home spot carrying its cue upright (the carry joint
# angles are fixed values, worked out once by IK).

def Reset_Scene():
    global d_xyyaw
    Teach_Unhide()
    state["physics"], state["fast"] = False, False
    state["score"], state["in_play"] = {"D": 0, "U": 0}, False
    station.Reset()
    cue_state["shaft"] = station.Shaft_Pose(Q_GRIP)
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
    for p, home, q in ((ur3_p, UR3_HOME, UR3_CARRY_Q), (meca_p, MECA_HOME, MECA_CARRY_Q)):
        p.xyyaw = home
        p.Set_Base()
        p.q = q.copy()
    doosan.q = DOOSAN_CARRY_Q.copy()
    jbot.robot.q = JBOT_CARRY_Q.copy()
    Draw()


# ------------------------------------JBOT RACKS THE BALLS----------------------------------------------------
# The Jbot drives to each pocket, goes down and up with its suction cup to lift a ball out, drives back and goes
# down and up again to drop it into the rack in order, then slides the full rack to the foot spot and lifts it
# off. Racking is played faster (RACK_SPEED_UP) because it is long.

def Jbot_Carry_q():
    return jbot.IK(jbot.robot.base * CARRY_LOCAL, jbot.robot.q)


def Down_And_Up(x, y, z_above, z_low, at_bottom):
    T_above, T_low = Tool_Down(x, y, z_above), Tool_Down(x, y, z_low)
    yield from Jbot_Joints(jbot.IK(T_above, jbot.robot.q))
    yield from Jbot_Line(T_above, T_low)
    at_bottom()
    yield from Jbot_Line(T_low, T_above)
    yield from Jbot_Joints(Jbot_Carry_q())


def Pick_From_Pocket(n, pocket, level):
    yield from Drive_Convoy("J", POCKET_PARKS[pocket])
    yield from Down_And_Up(*POCKETS[pocket], ABOVE, POCKET_FLOOR_Z + 2 * R + level * 2 * R,
                           lambda: held.update(what=n, T_rel=jbot.Tool_Pose().inv() * SE3(*physics[n].pos)))


def Drop_In_Rack(n, slot_xy):
    def Let_Go():
        held["what"] = None
        physics[n].pos = np.array([slot_xy[0], slot_xy[1], FELT_Z + R])
    yield from Drive_Convoy("J", TRAY_PARK)
    yield from Down_And_Up(slot_xy[0], slot_xy[1], FELT_Z + 2 * R + 0.08, FELT_Z + 2 * R, Let_Go)


def Slide_And_Lift_Rack():
    top = rack.Handle_Top()
    T_above = Tool_Down(top[0], top[1], top[2] + 0.08)
    T_grip = Tool_Down(top[0], top[1], top[2])
    yield from Jbot_Joints(jbot.IK(T_above, jbot.robot.q))
    yield from Jbot_Line(T_above, T_grip)
    T_cup = jbot.Tool_Pose()
    held["what"], held["T_rel"] = "rack", T_cup.inv() * rack.T
    held["balls"] = {n: rack.T.inv() * SE3(*physics[n].pos) for n in PICK_ORDER}
    Set_Status("Jbot is racking")
    yield from Jbot_Line(T_cup, SE3(FOOT_SPOT - RACK_START_APEX, 0, 0) * T_cup, steps=80, keep_spin=True)
    held.pop("balls")
    T_cup = jbot.Tool_Pose()
    yield from Jbot_Line(T_cup, SE3(0, 0, 0.08) * T_cup, steps=20, keep_spin=True)
    yield from Jbot_Joints(Jbot_Carry_q())


def Rack_The_Balls():
    state["fast"] = True
    slots = [np.array([RACK_START_APEX, 0.0]) + s[:2] for s in Rack_Slots(R)]
    where = {n: (pocket, level) for pocket, stack in POCKET_STACKS.items() for level, n in enumerate(stack)}
    for k, n in enumerate(PICK_ORDER):
        pocket, level = where[n]
        Set_Status("Jbot is racking")
        yield from Pick_From_Pocket(n, pocket, level)
        yield from Drop_In_Rack(n, slots[k])
    yield from Slide_And_Lift_Rack()
    state["fast"] = False


# ------------------------------------TRAFFIC ON THE TRACK---------------------------------------------------
# All four turtlebots share one track and cannot pass each other. When a robot drives, any robot in its way
# is pushed on ahead of it (a convoy), and before a shot the others move away so the shooter's cue has room.
# Positions are measured along the track so the inner and outer loops line up corner for corner.

def Settle_Table(layout):
    for n, xy in layout.items():
        p = physics[n]
        keep = p.R
        p.Set([xy[0], xy[1], FELT_Z + R])
        p.R = keep


SHOT_ROOM = 1.6
OUT_LOOP = _Loop(*DOOSAN_LOOP)
NAMES = {"D": "Doosan", "U": "UR3", "M": "Meca500", "J": "Jbot"}


def Get_XY(k):
    return d_xyyaw if k == "D" else {"U": ur3_p, "M": meca_p, "J": jbot}[k].xyyaw


def Loop_Of(k):
    return (INNER_X, INNER_Y) if k in ("M", "J") else DOOSAN_LOOP


def Track_S(xy, k="D"):
    L = _Loop(*Loop_Of(k))
    return _Loop_S(np.asarray(xy[:2], dtype=float), L) * OUT_LOOP[2] / L[2]


def On_Loop(k, s):
    L = _Loop(*Loop_Of(k))
    p = _Loop_Point((s % OUT_LOOP[2]) * L[2] / OUT_LOOP[2], L)
    return (float(p[0]), float(p[1]), None)


TURTLE_GAP = 1.1


def Set_XY(k, pose):
    global d_xyyaw
    if k == "D":
        d_xyyaw = pose
    else:
        {"U": ur3_p, "M": meca_p, "J": jbot}[k].xyyaw = pose


def XY(k):
    return np.array(Get_XY(k)[:2])


def _Too_Close(k, xy):
    return [o for o in "DUMJ" if o != k and np.linalg.norm(XY(o) - xy) < TURTLE_GAP]


def _Way(s_to, s_from):
    L = OUT_LOOP[2]
    return 1 if ((s_to - s_from + L / 2) % L - L / 2) >= 0 else -1


def _Push(other, xy, direction, ds, depth=0):
    for _ in range(30):
        _Nudge(other, direction, ds, depth)
        if np.linalg.norm(XY(other) - xy) >= TURTLE_GAP:
            break


def _Nudge(o, direction, ds, depth=0):
    if depth > 4:
        return
    old = XY(o)
    for _ in range(20):
        new = On_Loop(o, Track_S(old, o) + direction * ds)
        if np.linalg.norm(np.array(new[:2]) - old) > 1e-4:
            break
        ds += 0.02
    xy = np.array(new[:2])
    for other in _Too_Close(o, xy):
        if np.linalg.norm(XY(other) - xy) < np.linalg.norm(XY(other) - old) + 1e-9:
            _Push(other, xy, direction, ds, depth + 1)
    Set_XY(o, (float(xy[0]), float(xy[1]), Get_XY(o)[2]))


def Drive_Convoy(k, goal):
    path = Drive_Path(Get_XY(k), goal, DT, speed=0.8 if k == "J" else 0.6, loop=Loop_Of(k),
                      allow_reverse=k in ("M", "J"))
    pushed = False
    for pose in path:
        old, xy = XY(k), np.array(pose[:2])
        step = np.linalg.norm(xy - old)
        if step > 1e-6:
            direction = _Way(Track_S(xy, k), Track_S(old, k))
            for other in _Too_Close(k, xy):
                if np.linalg.norm(XY(other) - xy) < np.linalg.norm(XY(other) - old):
                    _Push(other, xy, direction, step + 0.005)
                    if not pushed:
                        Set_Status(f"{NAMES[k]} is driving, {NAMES[other]} moves ahead")
                        pushed = True
        Set_XY(k, pose)
        yield from Frame()


def Make_Room(k, room=1.0):
    me = XY(k)
    for other in "DUMJ":
        if other == k:
            continue
        for _ in range(400):
            o = XY(other)
            if np.linalg.norm(o - me) >= room:
                break
            _Nudge(other, _Way(Track_S(o, other), Track_S(me, k)), 0.35 * DT * 2)
            if np.linalg.norm(XY(k) - me) > 1e-6:
                Set_XY(k, (*me, Get_XY(k)[2]))
            yield from Frame()


# ------------------------------------SHOOTING ARMS: ONE INTERFACE-------------------------------------------
# The Doosan, UR3 and Meca500 are driven through one Arm class (joints, IK, cue tip, carry pose, speed
# limits), so every shot, chalking and assembly move is written once. Straight cue moves use RMRC; when RMRC
# can't follow the line, IK at every step is used instead. Joint moves are checked against the collision
# model and, if the simple ones all touch something, a collision-free path is planned with RRT-Connect.

class Arm:
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
        if self.k == "M":
            return self.IK(T)
        seed = PA.Shape_Seed(self.k, self.Robot(), T)
        if self.k == "D":
            sol = doosan.ikine_LM(T * D_TOOL.inv(), q0=seed, joint_limits=False, slimit=1)
        else:
            T_local = ur3_p.robot.base.inv() * T * ur3_p.tool.inv()
            sol = ur3_p.kin.ikine_LM(T_local, q0=seed, end=ur3_p.END, joint_limits=True, slimit=1)
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
    qs, err = PA.RMRC(arm.Tip, np.array(arm.q, dtype=float), poses, DT)
    return qs if err < 0.005 and In_Limits(arm, qs) else None


def Arm_Line(arm, T_from, T_to, q_goal=None, steps=40):
    q0 = np.array(arm.q, dtype=float)
    poses = [T_from.interp(T_to, s) for s in rtb.quintic(0, 1, steps).q]
    qs = RMRC_Path(arm, poses)
    if qs is None:
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


def IK_Line(arm, poses, q0=None):
    qs, q = [], np.array(arm.q if q0 is None else q0, dtype=float)
    for T in poses[1:]:
        q_new = arm.IK(T, q)
        if q_new is None or np.max(np.abs(q_new - q)) > 0.3:
            return None
        qs.append(q_new)
        q = q_new
    return qs


def Cue_Clearance(arm, q):
    T, L = Real_Cue(arm, q)
    return PA.Cue_Gap(arm.k, arm.Robot(), q, T, L)


def Real_Cue(arm, q):
    T = arm.Tip(q)
    if arm.k == "D" and cue_state["shaft"] is not None:
        return T * SE3.Tz(-cue.shaft_length), cue.length - cue.shaft_length
    return T, (1.127 if arm.k == "M" else cue.length)


def Nearest_Turn(arm, q0, q1):
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    if arm.k == "M":
        return q1
    lo, hi = (np.full(6, -2 * pi), np.full(6, 2 * pi)) if arm.k == "D" else ur3_p.robot.qlim
    q = q0 + (q1 - q0 + pi) % (2 * pi) - pi
    return np.where((q >= lo) & (q <= hi), q, q1)


def Clear_Joint_Path(arm, q0, q1, steps=50, straight=None):
    q1 = Nearest_Turn(arm, q0, q1)

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


# ------------------------------------COLLISION MODEL--------------------------------------------------------
# Every arm link, turtlebot body and cue is modelled as a capsule (a tube with round ends) of its real
# thickness. The same model is used to plan the moves and to check the finished run, so what passes the
# check is what you see. The obstacles are cached until something else in the cell moves.

ARM_RADII = {"D": [0.11, 0.085, 0.08, 0.07, 0.065, 0.06],
             "U": [0.0, 0.065, 0.055, 0.05, 0.045, 0.045, 0.04, 0.035],
             "M": [0.08, 0.067, 0.06, 0.053, 0.0, 0.045],
             "J": [0.092] * 12,
             "S": [0.04] * 6}
TURTLE_RADIUS = 0.17


def Robot_Of(k):
    return {"D": doosan, "U": ur3_p.robot, "M": meca_p.robot, "J": jbot.robot, "S": station.robot}[k]


def Arm_Capsules(k, q=None):
    rob = Robot_Of(k)
    P = np.array([T.t for T in rob.fkine_all(rob.q if q is None else q)])[:9 if k == "U" else None]
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


_obstacles = {"key": None, "caps": []}
avoid = {"station": True}


def Cue_Free_Gap(arm, q):
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


# ------------------------------------ONE SHOT---------------------------------------------------------------
# Every shot is played the same way: a collision-free joint move from carrying the cue to just above the
# shot line, RMRC down onto the line and along it to the ball, the stroke along the cue's own axis, then
# back and up and back to carrying. The ball is hit on the sample where the tip reaches it.

def Arm_Lift(arm, T, q_seed):
    for lift in (LIFT, 0.10, 0.05):
        poses = [SE3(0, 0, lift * s) * T for s in np.linspace(0, 1, 12)]
        qs, err = PA.RMRC(arm.Tip, np.array(q_seed, dtype=float), poses, DT)
        if err >= 0.003 or not In_Limits(arm, qs):
            qs = IK_Line(arm, poses, q_seed)
            if qs is None:
                continue
        if min(Cue_Clearance(arm, q) for q in qs) >= PA.CLEARANCE:
            return lift, qs[-1]
    return 0.0, np.array(q_seed, dtype=float)


def Game_Shot(k, shot):
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
        Set_Status(f"{NAMES[k]} can't reach - skipped")
        return
    lift, q_above = Arm_Lift(arm, T_ready, q_ready)
    T_above = SE3(0, 0, lift) * T_ready
    yield from Arm_Play(arm, Clear_Joint_Path(arm, q_carry, q_above))
    yield from Arm_Line(arm, T_above, T_ready, q_ready)
    yield from Arm_Line(arm, T_ready, T_address)
    if k == "M":
        profile = np.array(Short_Stroke_Profile(DT, contact_s=-ADDRESS_GAP, tip_speed=shot["speed"],
                                                follow=0.015, backswing=0.05))
    else:
        profile = Punch_Stroke_Profile(DT, contact_s=-ADDRESS_GAP, tip_speed=shot["speed"],
                                       back=0.20 if k == "D" else 0.15, follow=0.03)
    hit_at = int(np.argmax(profile <= -ADDRESS_GAP + 1e-9))
    T_start = arm.Tip()
    qs = RMRC_Path(arm, [T_start] + [T_address * SE3.Tz(-s) for s in profile])
    if qs is None:
        qs = IK_Line(arm, [T_start] + [T_address * SE3.Tz(-s) for s in profile])
    if qs is None:
        qs, q = [], np.array(arm.q, dtype=float)
        for s in profile:
            q_new = arm.IK(T_address * SE3.Tz(-s), q)
            q = q if q_new is None else q_new
            qs.append(q)
    for i, q in enumerate(qs):
        if i == hit_at:
            physics[0].Hit(CUE_TO_BALL * shot["speed"] * d, contact)
        yield from Arm_Play(arm, [q])
    T_now = arm.Tip()
    T_back = T_now * SE3.Tz(-approach)
    yield from Arm_Line(arm, T_now, T_back)
    lift, _ = Arm_Lift(arm, T_back, arm.q)
    yield from Arm_Line(arm, T_back, SE3(0, 0, lift) * T_back)
    yield from Arm_Play(arm, Clear_Joint_Path(arm, np.array(arm.q, dtype=float), q_carry))


# ------------------------------------ASSEMBLY---------------------------------------------------------------
# The station Meca500 lifts the shaft off its rests and holds it level, the Doosan brings the handle in
# along the same line, and the shaft is screwed on. Collisions with the station are ignored only here,
# because its hand is meant to touch the cue.

def Station_Play(qs, shaft=False):
    for q in qs:
        station.robot.q = q
        if shaft:
            cue_state["shaft"] = station.Shaft_Pose(q)
        yield from Frame()


def Assemble():
    arm = Arm("D")
    avoid["station"] = False
    spot, roll, q_seeds = ASM_PRESENT
    Set_Status("Doosan is driving")
    yield from Drive_Convoy("D", tuple(spot))
    Set_Status("Station Meca is moving")
    reach, lift = station.Pick_Shaft_Path()
    yield from Station_Play(reach)
    station.holding = "shaft"
    yield from Station_Play(lift, shaft=True)
    Set_Status("Doosan is assembling")
    q_carry = np.array(arm.q, dtype=float)
    T_line, T_pin = Assembly_Pose(LINE_UP + PIN, roll), Assembly_Pose(PIN, roll)
    q_line = arm.IK(T_line, np.array(q_seeds[2]))
    T_above = SE3(0, 0, 0.12) * T_line
    yield from Arm_Line(arm, arm.Tip(), T_above, arm.IK(T_above, q_line))
    yield from Arm_Line(arm, T_above, T_line, q_line)
    yield from Arm_Line(arm, T_line, T_pin, arm.IK(T_pin, q_line))
    S = station.Shaft_Pose(Q_HOLD)
    q = np.array(arm.q, dtype=float)
    for s in rtb.quintic(0, 1, 150).q:
        q_new = arm.IK(Assembly_Pose(PIN * (1 - s), roll), q)
        q = q if q_new is None else q_new
        cue_state["shaft"] = S * SE3.Rz(N_TURNS * 2 * pi * s)
        yield from arm.Frame(q)
    cue_state["shaft"] = None
    station.holding = None
    for s in np.linspace(0, 1, 15):
        station.gripper.opening = s
        yield from Frame()
    yield from Station_Play([Q_HOLD] * 30)
    Set_Status("Doosan is lifting the cue")
    T_now = arm.Tip()
    yield from Arm_Line(arm, T_now, SE3(0, 0, 0.12) * T_now)
    yield from Arm_Line(arm, arm.Tip(), arm.Carry(), q_carry)
    yield from Station_Play(rtb.jtraj(station.robot.q, Q_REST, 80).q)
    avoid["station"] = True


# ------------------------------------CHALKING---------------------------------------------------------------
# Before the break and whenever the player changes, that robot drives to the station and holds its cue tip
# still while the station Meca500 chalks it. The way in is the way out (worked out by RMRC) played
# backwards, so it always uses the same arm shape.

CHALK_ROOM = 1.6


def Chalk_Cue(k):
    arm = Arm(k)
    spot, roll, q_seed = PRESENT[k]
    Set_Status(f"{NAMES[k]} is driving")
    yield from Drive_Convoy(k, tuple(spot))
    yield from Make_Room(k, CHALK_ROOM)
    q_carry = np.array(arm.q, dtype=float)
    if roll is None:
        q_p = np.array(q_seed)
        T_p = arm.Tip(q_p)
    else:
        T_p = Chalk_Present_Pose(roll)
        q_p = arm.IK(T_p, np.array(q_seed))
    T_back = T_p * SE3.Tz(-0.10)
    T_up = SE3(0, 0, 0.10) * T_back
    qs_back, _ = PA.RMRC(arm.Tip, q_p, [T_p.interp(T_back, s) for s in rtb.quintic(0, 1, 30).q], DT)
    qs_up, _ = PA.RMRC(arm.Tip, qs_back[-1], [T_back.interp(T_up, s) for s in rtb.quintic(0, 1, 30).q], DT)
    yield from Arm_Play(arm, Clear_Joint_Path(arm, q_carry, qs_up[-1]))
    yield from Arm_Play(arm, qs_up[::-1][1:] + [qs_back[-1]])
    yield from Arm_Play(arm, qs_back[::-1][1:] + [q_p])
    Set_Status("Station Meca is chalking")
    for q, opening, hold in station.Chalk_Path():
        station.robot.q = q
        if opening is not None:
            station.gripper.opening = opening
        if hold and station.holding != "chalk":
            station.holding = "chalk"
        elif not hold and station.holding == "chalk":
            station.holding, station.T_chalk = None, station.T_chalk_home
        yield from Frame()
    yield from Arm_Play(arm, qs_back)
    yield from Arm_Play(arm, qs_up)
    yield from Arm_Play(arm, Clear_Joint_Path(arm, np.array(arm.q, dtype=float), q_carry))


# ------------------------------------THE EXHIBITION STEPS---------------------------------------------------
# The planned game (Pool_Game.GAME) is turned into a numbered list of steps: assembly, racking, then a chalk
# step whenever the player changes and one step per shot. The control panel lists these steps and can jump
# to any of them.

def Game_Shot_Step(i):
    shot = GAME[i]
    who = shot["who"]
    Set_Status(f"{NAMES[who]} is driving")
    yield from Drive_Convoy(who, tuple(shot["turtle"]))
    yield from Make_Room(who, SHOT_ROOM)
    Settle_Table(shot["layout"])
    state["physics"], state["in_play"] = True, True
    Set_Status(f"{NAMES[who]} is shooting")
    yield from Game_Shot(who, shot)
    while any(p.Moving() for p in BALL_LIST):
        yield from Frame()
    potted = [n for n in shot["pots"] if physics[n].potted is not None]
    side = "U" if who == "U" else "D"
    state["score"][side] += len(potted)
    sc = state["score"]
    Set_Status(f"{NAMES[who]} " + (f"pots {', '.join(map(str, potted))}" if potted else "misses")
               + f"  ({sc['D']} - {sc['U']})")
    for _ in range(30):
        yield from Frame()
    if i == len(GAME) - 1:
        Set_Status(f"Doosan team wins {sc['D']} - {sc['U']}")


def Shot_Name(i, shot):
    who = NAMES[shot["who"]]
    if shot["kind"] == "break":
        return "Break"
    kind = {"pot": "pots " + ", ".join(str(n) for n in shot["pots"]), "miss": "misses",
            "jump": "JUMP shot", "masse": "MASSE"}[shot["kind"]]
    if shot["kind"] in ("jump", "masse") and shot["pots"]:
        kind += " - pots " + ", ".join(str(n) for n in shot["pots"])
    return f"Shot {i + 1}: {who} {kind}"


def Build_Steps():
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
BALL_STEP = {n: STEP_NAMES.index(Shot_Name(i, shot)) for i, shot in enumerate(GAME) for n in shot["pots"]}
state["step"] = None


def Step_Name(i=None):
    i = state["step"] if i is None else i
    return "-" if i is None else f"{i + 1}. {STEP_NAMES[i]}"


def Run_From(first):
    for i in range(first, len(STEPS)):
        state["step"] = i
        yield from Snapshot_Hook(i)
        yield from STEPS[i][1]()
    state["step"] = None


def Potted_Balls():
    if not state.get("in_play"):
        return set()
    return {n for n in range(1, 10) if physics[n].potted is not None}


# ------------------------------------TIME-SKIP SNAPSHOTS ----------------------------------------------------
# A snapshot is everything that moves, saved at the start of each step. They are made once with python
# Pool_Robot_Menu.py --make-snapshots (the whole show is played without drawing) and saved to
# Pool_Snapshots.pkl. Jumping to a step restores its snapshot and carries on from there, exactly as the
# full run would.

SNAPSHOT_FILE = Path(__file__).resolve().parent / "Pool_Snapshots.pkl"
snapshots = {}


BALL_KEEP = ("pos", "v", "w", "R", "_vz", "potted", "floor_z", "vz", "airborne")


def Take_Snapshot():
    return copy.deepcopy({
        "d": [list(d_xyyaw), doosan.q], "u": [list(ur3_p.xyyaw), ur3_p.q], "m": [list(meca_p.xyyaw), meca_p.q],
        "j": [list(jbot.xyyaw), jbot.robot.q],
        "station": [station.robot.q, station.holding, station.gripper.opening, station.T_chalk],
        "shaft": cue_state["shaft"], "rack": rack.T,
        "held": [held.get("what"), held.get("T_rel"), held.get("balls")],
        "balls": {n: {k: getattr(p, k) for k in BALL_KEEP if hasattr(p, k)} for n, p in physics.items()},
        "state": {k: state[k] for k in ("physics", "fast", "score", "in_play")}})


def Restore_Snapshot(i):
    global d_xyyaw
    if not snapshots:
        Load_Snapshots()
    if i not in snapshots:
        if i == 0:
            Reset_Scene()
            return True
        return False
    S = copy.deepcopy(snapshots[i])
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
    door_state["script"], door_state["angle"] = [], 0.0
    avoid["station"] = True
    Teach_Unhide()
    Draw()
    return True


def Load_Snapshots():
    try:
        with open(SNAPSHOT_FILE, "rb") as f:
            data = pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError):
        return
    if data["steps"] == STEP_NAMES:
        snapshots.update(data["snaps"])
    else:
        print("Pool_Snapshots.pkl is for a different sequence - make it again with --make-snapshots")


def Snapshot_Hook(i):
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
    with open(SNAPSHOT_FILE, "wb") as f:
        pickle.dump({"steps": STEP_NAMES, "snaps": snapshots}, f)
    print(f"saved {len(snapshots)} snapshots ({frames} frames, {time.time() - t0:.0f} s) -> {SNAPSHOT_FILE.name}")
    print("final score", state["score"], "potted", sorted(Potted_Balls()))


# ------------------------------------EXHIBITION MODE--------------------------------------------------------
# Resets the table and plays every step from the start.

def Exhibition():
    Set_Status("Resetting the table")
    Reset_Scene()
    yield
    yield from Run_From(0)


# ------------------------------------TEACH MODE-------------------------------------------------------------
# Only the UR3, the cue ball and the 8 ball are shown. The joints are set by hand with the sliders, and
# every update the panel shows the best configuration for the shot and checks on the cue as it is now (aim,
# tilt, hit point, reach, clearance from the arm), each with a suggested fix. SHOOT rocks back and strokes
# along the cue's own axis, and the ball is hit where the cue line really meets it, so a bad setup really
# misses. Where the UR3 stands and its best configuration are fixed values, found once by searching the track
# behind the shot line for a spot where the arm keeps its one shape and the stroke clears the arm.

TEACH_POCKET = 1
TEACH_8 = np.array([0.857, -0.252])
TEACH_CUE = np.array([0.432594509931834, 0.17240549006816575])
TEACH_DIR = np.array([0.7071067811865477, -0.7071067811865474])
TEACH_AIM = -44.99999999999999
TEACH_ELEV = 8.0
TEACH_MAX_REACH = 0.15
TEACH_SPOT = (0.09499999999999997, 0.9375, 0.0)
TEACH_IDEAL = np.array([3.0117894601442154, -0.1544603573032326, 1.611556408943784, -3.1388395609131545,
                        1.4858499698773429, 0.9199213058132543])
teach = {"q": None, "ideal": None, "shot": None, "speed": 0.7, "result": "", "ready": False}


TEACH_FAR = {"D": (40.0, 0.0, 0.0), "M": (44.0, 0.0, 0.0), "J": (48.0, 0.0, 0.0)}
hidden = {"teach": False}


def Teach_Hide():
    for k, far in TEACH_FAR.items():
        Set_XY(k, far)
    cue_state["shaft"] = None
    station.Hide(True)
    rack.Place(SE3(52.0, 0.0, 0.0))
    Hide_Other_Balls()
    hidden["teach"] = True
    me = np.array(ur3_p.xyyaw[:2])
    look = np.array([(me[0] + TEACH_CUE[0]) / 2, (me[1] + TEACH_CUE[1]) / 2, 0.95])
    side = np.array([-TEACH_DIR[1], TEACH_DIR[0], 0.0])
    if side @ np.array([0.0, -1.0, 0.0]) < 0:
        side = -side
    env.set_camera_pose(list(look + 1.7 * side + np.array([0.0, 0.0, 0.85])), list(look))


def Teach_Unhide():
    if hidden.get("teach"):
        station.Hide(False)
        env.set_camera_pose(*CAMERA_OVERVIEW)
        hidden["teach"] = False


def Teach_Address():
    shot = dict(aim=TEACH_AIM, elev=TEACH_ELEV, side=0.0, height=0.0, roll=PA.CUE_ROLL["U"])
    return Shot_Poses(np.array([*TEACH_CUE, FELT_Z + R]), shot, R, ADDRESS_GAP)[2]


def Hide_Other_Balls():
    for n, p in physics.items():
        if n not in (0, 8):
            p.Set([56.0, float(n), 0.0])
            p.potted = 0


def Teach_Balls():
    Hide_Other_Balls()
    physics[0].Set([*TEACH_CUE, FELT_Z + R])
    physics[8].Set([*TEACH_8, FELT_Z + R])


def Teach_Setup():
    Reset_Scene()
    state["physics"], state["in_play"] = True, False
    Teach_Balls()
    ur3_p.xyyaw = TEACH_SPOT
    ur3_p.Set_Base()
    Teach_Hide()
    q_carry = ur3_p.IK(ur3_p.Carry_Tip(), ur3_p.q)
    if q_carry is not None:
        ur3_p.q = q_carry
    teach.update(q=np.array(ur3_p.q, dtype=float), ideal=TEACH_IDEAL.copy(), shot=None, result="", ready=True)
    Draw()


def Teach_Checks(q):
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
    along = rel @ d
    offset = np.linalg.norm(rel - along * d)
    reach = along - np.sqrt(max(R ** 2 - offset ** 2, 0.0))
    sol = ur3_p.kin.ikine_LM(ur3_p.robot.base.inv() * Teach_Address() * ur3_p.tool.inv(), q0=q, end=ur3_p.END,
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
    arm = Arm("U")
    q0 = np.array(ur3_p.q, dtype=float)
    T0 = arm.Tip(q0)
    d = T0.R[:, 2]
    _, (reach, offset, along) = Teach_Checks(q0)
    hits = along > 0 and offset < R and 0 <= reach <= TEACH_MAX_REACH
    contact_s = -reach if hits else -TEACH_MAX_REACH
    profile = Punch_Stroke_Profile(DT, contact_s=contact_s, tip_speed=teach["speed"], back=0.12, follow=0.03,
                                   n_rocks=1, rock_amp=0.05)
    poses = [T0] + [T0 * SE3.Tz(-s) for s in profile]
    qs = RMRC_Path(arm, poses)
    if qs is None:
        qs, q = [], q0
        for T in poses[1:]:
            q = arm.IK(T, q)
            if q is None:
                teach["result"] = "The arm can't make a straight stroke from this pose (out of reach / joint limits)."
                Set_Status("Teach: " + teach["result"])
                return
            qs.append(q)
    hit_at = int(np.argmax(profile <= contact_s + 1e-9))
    Set_Status("UR3 is shooting")
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
    Set_Status("UR3 is moving into place")
    Teach_Setup()
    Set_Status("Set the sliders, then SHOOT")
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


# ------------------------------------CONTROLLER MODE--------------------------------------------------------
# The UR3 is driven joint by joint with a gamepad (Gamepad_Controller.py) from a fixed spot at the
# top-middle pocket, with an overhead camera. Select returns the UR3 to its carry pose and Start shoots
# straight along the cue.

def Controller():
    Set_Status("Controller mode: use the gamepad to move the UR3")
    Reset_Scene()
    state["physics"], state["in_play"] = True, False
    Teach_Balls()
    ur3_p.xyyaw = (0.0, 0.84, -90.0)
    ur3_p.Set_Base()
    Teach_Hide()
    env.set_camera_pose(np.array([0.0, 0.9, 2.3]), look_at=np.array([0.0, 0.1, 0.0]))
    if teach.get("q") is None:
        teach["q"] = np.array(ur3_p.q, dtype=float)
    ur3_p.q = np.array(teach["q"], dtype=float)
    teach.update(q=np.array(ur3_p.q, dtype=float), ideal=None, shot=None, result="Controller mode ready", ready=True)
    Draw()
    try:
        from Gamepad_Controller import GamepadController
        pad = GamepadController(joystick_index=0, deadzone=0.15, joint_speed=np.radians(45.0), smoothing=0.25)
    except (RuntimeError, ImportError) as e:
        Set_Status(f"Controller mode: {e}")
        print(e)
        while True:
            yield from Frame()
    shoot_request = {"requested": False}

    def Reset_UR3():
        q_reset = ur3_p.IK(ur3_p.Carry_Tip(), ur3_p.q)
        if q_reset is not None:
            ur3_p.q = np.array(q_reset, dtype=float)

    def Controller_Shot(power=1.0):
        arm = Arm("U")
        T0 = arm.Tip(np.array(ur3_p.q, dtype=float))
        d = T0.R[:, 2]
        ball = np.array(physics[0].pos, dtype=float)
        rel = ball - T0.t
        along = float(np.dot(rel, d))
        offset = float(np.linalg.norm(rel - along * d))
        reach = along - np.sqrt(max(R ** 2 - offset ** 2, 0.0))
        contact = -reach if along > 0 and offset < R and 0 < reach <= TEACH_MAX_REACH else -TEACH_MAX_REACH
        profile = Punch_Stroke_Profile(DT, contact, power, back=0.12, follow=0.03, n_rocks=1, rock_amp=0.05)
        poses = [T0] + [T0 * SE3.Tz(-float(s)) for s in profile]
        qs = RMRC_Path(arm, poses)
        if qs is None:
            qs = IK_Line(arm, poses)
        if qs is None:
            Set_Status("Controller: cannot make a straight shot from this pose")
            return
        Set_Status("UR3 is shooting")
        hit = False
        for q in qs:
            q = np.asarray(q, dtype=float)
            tip = arm.Tip(q).t
            if not hit and float(np.dot(ball - tip, d)) <= R + 1e-6:
                physics[0].Hit(power * d, tip - ball)
                hit = True
            yield from arm.Frame(q)
        while any(p.Moving() for p in BALL_LIST):
            yield from Frame()
        Set_Status("Controller: shot complete")

    while True:
        pad.update()
        ur3_p.q = np.clip(np.array(ur3_p.q, dtype=float) + pad.read_joint_velocities() * DT, *ur3_p.robot.qlim)
        pad.handle_buttons(reset_callback=Reset_UR3, shoot_callback=lambda: shoot_request.update(requested=True))
        if shoot_request["requested"]:
            shoot_request["requested"] = False
            yield from Controller_Shot(power=1.0)
        yield from Frame()


# ------------------------------------BUTTON REQUESTS--------------------------------------------------------
# Every button on the control panel becomes a request handled here. The rules make the cell safe: E-STOP
# always wins, RESET only acknowledges a stop and RESUME is a separate deliberate action. Picking another
# mode (or an exhibition step) completely drops the mode that was running and starts the new one from a
# clean scene - except while e-stopped or with the door open.

MODES = {"exhibition": Exhibition, "teach": Teach, "controller": Controller}
LIGHT_FOR = {"IDLE": "amber", "RUNNING": "green", "ESTOPPED": "red", "RESET": "amber", "DOOR_OPEN": "red"}


def Drop_Mode():
    control["sequence"], control["mode"] = None, None
    teach["shot"] = None
    teach.pop("reset_balls", None)
    state["step"] = None
    avoid["station"] = True
    held.pop("balls", None)


def Handle_Request(req):
    st = control["state"]
    if req == "estop":
        if st != "ESTOPPED":
            control["state"] = "ESTOPPED"
            Set_Status("E-STOP pressed - press RESET")
    elif req == "reset":
        if st == "ESTOPPED":
            control["state"] = "RESET"
            Set_Status("Reset - press ENGAGE to carry on")
        else:
            Set_Status("Reset only does something after an E-STOP.")
    elif req == "resume":
        if st == "ESTOPPED":
            Set_Status("Press RESET first")
        elif st == "RESET" and control["sequence"] is not None:
            control["state"] = "RUNNING"
            Set_Status("Carrying on")
        elif st == "RESET":
            control["state"] = "IDLE"
            Set_Status("nothing to resume - pick a mode")
    elif req in MODES:
        if st == "ESTOPPED":
            Set_Status("E-STOP active - press RESET first.")
        elif st == "DOOR_OPEN":
            Set_Status("Door open - shut the door first")
        else:
            Drop_Mode()
            control["state"], control["mode"] = "RUNNING", req
            control["sequence"] = MODES[req]()
            door_state["t"], door_state["demo_done"] = 0.0, False
            Set_Status(f"starting {req}")
    elif isinstance(req, tuple) and req[0] == "goto":
        i = req[1]
        if st in ("ESTOPPED", "DOOR_OPEN"):
            Set_Status("Can't skip while the cell is stopped")
            return True
        Drop_Mode()
        if not Restore_Snapshot(i):
            control["state"] = "IDLE"
            Set_Status("No snapshots yet - run:  python Pool_Robot_Menu.py --make-snapshots")
        else:
            control["state"], control["mode"] = "RUNNING", "exhibition"
            control["sequence"] = Run_From(i)
            state["step"] = i
            Set_Status("Skipping ahead")
            door_state["t"], door_state["demo_done"] = 0.0, i > 0
    elif req == "pause":
        if st == "RUNNING":
            control["state"] = "RESET"
            Set_Status("Paused")
    elif req == "door":
        if not door_state["script"]:
            door_state["script"] = Door_Visit()
    elif req == "drop_mode":
        if st not in ("ESTOPPED", "DOOR_OPEN") and control["mode"] in ("teach", "controller"):
            Set_Status(f"{control['mode']} mode stopped - pick an exhibition step")
            Drop_Mode()
            Reset_Scene()
            control["state"] = "IDLE"
    elif req == "teach_shoot":
        if control["mode"] == "teach" and st == "RUNNING" and teach["shot"] is None:
            teach["shot"] = Teach_Shot()
        else:
            Set_Status("SHOOT works in Teach mode while the cell is ENGAGED")
    elif req == "teach_reset":
        if control["mode"] == "teach":
            teach["reset_balls"] = True
    elif req == "wall_estop":
        if not door_state["script"]:
            door_state["script"] = Wall_EStop_Visit()
    elif req == "quit":
        Set_Status("closing")
        env.close()
        return False
    return True


# ------------------------------------START-UP AND THE MAIN LOOP---------------------------------------------
# Step_Once is called by the control panel on every screen update: it handles the next request, checks the
# door sensor, and only while RUNNING plays the next frames of the sequence. Running this file opens the
# control panel; --make-snapshots builds the time-skip file instead.

Reset_Scene()
if not HEADLESS:
    Load_Snapshots()
control["state"] = "IDLE"
Set_Status("ready - click 1. Exhibition, 2. Teach or 3. Controller")


def Step_Once():
    req, control["request"] = control["request"], None
    if req is not None and not Handle_Request(req):
        return False
    Door_Step()
    door_open = door_state["angle"] > DOOR_SWITCH_DEG
    if control["state"] == "RUNNING" and door_open:
        control["state"] = "DOOR_OPEN"
        Set_Status("Door open - robots paused")
    elif control["state"] == "DOOR_OPEN" and not door_open:
        control["state"] = "RUNNING"
        Set_Status("Door shut - carrying on")
    if control["state"] == "RUNNING" and control["mode"] == "exhibition" and not door_state["demo_done"]:
        door_state["t"] += DT
        if door_state["t"] >= DOOR_DEMO_AT:
            door_state["demo_done"] = True
            door_state["script"] = Door_Visit()
    if control["state"] == "RUNNING" and control["sequence"] is not None:
        try:
            for _ in range(SPEED_UP * (RACK_SPEED_UP if state.get("fast") else 1)):
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
    else:
        import runpy
        sys.modules["Pool_Robot_Menu"] = sys.modules["__main__"]
        runpy.run_path(str(Path(__file__).resolve().parent / "GUI_Pool_Robot_v4.py"), run_name="__main__")
