"""
Pool_Robot_Menu - the group workcell with a menu in Swift.

  1. Exhibition : Jbot (Jaylee) collects the 9 balls from the pockets into the rack,
                  slides the rack to the foot spot and lifts it off; then the
                  Doosan (Nicole) drives round and breaks. The balls land on the
                  layout from the photo (orange potted), and the Doosan runs the
                  table with the trick shots in Pool_Trick_Shots.py:
                    jump -> yellow top-left, bank -> blue middle-right, then red,
                    purple, green, brown, black, nine (one of them a masse).
  2. Teach      : (coming next)
  3. Experiment : (coming next)

Click the buttons in the Swift browser tab. The status line shows what is happening.

E-STOP: freezes every robot and ball immediately (stack light turns red).
        Reset acknowledges it, then Resume continues - nothing restarts on its own.
        After Reset you can also pick a mode to start again from a clean scene.

Scene: Robot_Track_v2 with the pool table in its centre hole. Two turtlebots on
the blue deck - Jbot's hugs the inner rim (it reaches into the table), the
Doosan's uses the outer side of the lanes so they never meet.
"""
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
from Pool_Motion import Holder_Tool, Solve_IK, Cue_Pose, Punch_Stroke_Profile, _Rotation_From_Vector
from Pool_Scene import (TRACK_NAME, TRACK_POSE, CHANNEL_Y, INNER_X, INNER_Y, RIM_TOP_Z,
                        TABLE_Z, FELT_Z, PLAY_AREA, GROOVE_HEIGHT, POCKETS, POCKET_RADIUS,
                        POCKET_FLOOR_Z, Turtle_Pose, Robot_Base_On, Drive_Path)
from Jbot_Racker import Jbot_Mobile, Ball_Rack, Rack_Slots, Tool_Down, Jbot_IK
from Pool_Trick_Shots import (Pool_Ball, Table_Step, Break_Script, LAYOUT, ORANGE, ORANGE_POCKET,
                              POCKET_NAMES, MU_ROLL, SHOTS, Shot_Poses)


# ------------------------------------CONSTANTS-----------------------------------------------------

DT = 0.02
PHYSICS_SUBSTEPS = 5                  # ball physics steps per frame (fast balls need small steps)
JOINT_SPEED = 2.0                     # rad/s cap for joint-space moves (both robots)

# --- Doosan (Nicole) ---
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

# --- Jbot (Jaylee) ---
JBOT_SPEED = 0.5                      # turtlebot driving speed (m/s)
TRAY_PARK = (INNER_X, 0.0, -90.0)     # Jbot fills / slides the rack from here
RACK_START_APEX = 0.78                # rack starts near the foot cushion (apex ball x)
FOOT_SPOT = 0.534                     # rack apex ends here (1/4 of the table from the foot)
CARRY_LOCAL = SE3(0.30, 0.0, 0.30) * SE3.Rx(pi)   # cup above the turtlebot, pointing down
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

env = swift.Swift()
env.launch(realtime=True)
env.set_camera_pose([3.2, -3.0, 3.0], [0.0, 0.0, 0.8])

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

# balls: cue ball (mesh with markings) + 9 coloured object balls
cue_ball_mesh = Cue_Ball(str(Part_Path("Cue_Ball")))
R = cue_ball_mesh.radius
cue_ball_mesh.Add_To_Env(env)
ball_meshes = {0: cue_ball_mesh}
for n, colour in BALL_COLOURS.items():
    ball_meshes[n] = sg.Sphere(radius=R, color=[*colour, 1.0])
    env.add(ball_meshes[n])
physics = {n: Pool_Ball([0, 0, FELT_Z + R], R, mu_roll=MU_ROLL, bounds=PLAY_AREA, pockets=POCKETS,
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

status = swift.Label(desc="Status: starting")
env.add(status)


def Set_Status(text):
    status.desc = f"[{control['state']}] {text}"
    print(f"[{control['state']}] {text}")


def Press(action):
    def callback(_):
        control["request"] = action
    return callback


for action, text in [("exhibition", "1. Exhibition"), ("teach", "2. Teach"), ("experiment", "3. Experiment"),
                     ("estop", "E-STOP"), ("reset", "RESET"), ("resume", "Resume"),
                     ("quit", "Quit")]:
    env.add(swift.Button(Press(action), desc=text))

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


# ------------------------------------STATE + DRAWING-----------------------------------------------------

state = {"physics": False, "script": []}
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
    cue.Place(T_flange * D_TOOL)

    jbot.Set_Base()
    if jq is not None:
        jbot.robot.q = jq
    jbot.Draw_Tool()
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
        if n == 0:
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


def Jbot_Joints(q_goal, steps=60):
    for q in rtb.jtraj(jbot.robot.q, q_goal, Steps_For(jbot.robot.q, q_goal, steps)).q:
        yield from Frame(jq=q)


def Jbot_Line(T_from, T_to, steps=40, keep_spin=False):
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
    for pose in Drive_Path(jbot.xyyaw, goal, DT, speed=JBOT_SPEED, loop=(INNER_X, INNER_Y),
                           allow_reverse=True):
        jbot.xyyaw = pose
        yield from Frame()


def Doosan_Drive(goal):
    global d_xyyaw
    for pose in Drive_Path(d_xyyaw, goal, DT, loop=DOOSAN_LOOP):
        d_xyyaw = pose
        yield from Frame()


# ------------------------------------RESET-----------------------------------------------------

def Reset_Scene():
    """Balls back in the pockets, rack on the felt, robots at their start spots."""
    global d_xyyaw
    state["physics"], state["script"] = False, []
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
    yield from Jbot_Line(T_cup, SE3(0, 0, 0.08) * T_cup, steps=40, keep_spin=True)
    yield from Jbot_Joints(Jbot_Carry_q())


def Rack_The_Balls():
    slots = [np.array([RACK_START_APEX, 0.0]) + s[:2] for s in Rack_Slots(R)]
    where = {n: (pocket, level) for pocket, stack in POCKET_STACKS.items() for level, n in enumerate(stack)}
    for k, n in enumerate(PICK_ORDER):
        pocket, level = where[n]
        Set_Status(f"Jbot: collecting ball {n} from pocket {pocket} ({k + 1}/9)")
        yield from Pick_From_Pocket(n, pocket, level)
        Set_Status(f"Jbot: placing ball {n} in the rack")
        yield from Drop_In_Rack(n, slots[k])
    yield from Slide_And_Lift_Rack()


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


# ------------------------------------MODES-----------------------------------------------------

def Exhibition():
    Set_Status("Exhibition: resetting the table")
    Reset_Scene()
    yield
    yield from Rack_The_Balls()
    yield from Break_Shot()
    yield from Run_Out()


def Teach():
    Set_Status("Teach mode - coming next (joint + Cartesian jogging)")
    yield


def Experiment():
    Set_Status("Experiment mode - coming next")
    yield


# ------------------------------------RUN-----------------------------------------------------

MODES = {"exhibition": Exhibition, "teach": Teach, "experiment": Experiment}
LIGHT_FOR = {"IDLE": "amber", "RUNNING": "green", "ESTOPPED": "red", "RESET": "amber"}


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
        elif st == "RUNNING":
            Set_Status(f"busy with {control['mode']} - press E-STOP to stop it first")
        else:                                                   # IDLE or RESET: start fresh
            control["state"], control["mode"] = "RUNNING", req
            control["sequence"] = MODES[req]()
            Set_Status(f"starting {req}")
    elif req == "quit":
        Set_Status("closing")
        env.close()
        return False
    return True


Reset_Scene()
control["state"] = "IDLE"
Set_Status("ready - click 1. Exhibition, 2. Teach or 3. Experiment")

running = True
while running:
    req, control["request"] = control["request"], None
    if req is not None:
        running = Handle_Request(req)
        if not running:
            break
    if control["state"] == "RUNNING" and control["sequence"] is not None:
        try:
            next(control["sequence"])                      # advance the sequence by ONE frame
        except StopIteration:
            control["sequence"] = None
            control["state"] = "IDLE"
            Set_Status(f"{control['mode']} finished - pick a mode")
    Set_Light(LIGHT_FOR[control["state"]])
    env.step(DT)
