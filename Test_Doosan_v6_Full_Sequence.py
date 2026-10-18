"""
Test_Doosan_v6_Full_Sequence - the whole pool demo in one script.

Scene: Robot_Track on the floor, Pool_Table in the middle of it, Turtlebot in
the white channel with the Doosan on top, Cue_Holder on the flange holding
the cue handle, two white balls on the felt.

Keys (type in the terminal, then Enter):
  1 = attach   : shaft comes down and screws into the handle in the holder
  2 = straight : turtlebot drives to the end of the table, normal shot
  3 = masse    : turtlebot drives to the side of the table, masse (ball curves)
  x = quit Swift
"""
import swift
import numpy as np
from math import pi
from spatialmath import SE3
import roboticstoolbox as rtb

from Doosan_Robot import Doosan_Robot
from ir_support_extra_robots.robots import Turtlebot3Waffle
from Pool_Parts import Part_Path, Part_Mesh
from Pool_Objects import Cue_Ball, Cue_Stick
from Pool_Motion import (Holder_Tool, Solve_IK, Cue_Pose, Stroke_Profile,
                         Punch_Stroke_Profile, Ball_Physics, Balls_Collide)
from Pool_Scene import (CHANNEL_Y, CHANNEL_X, TABLE_Z, FELT_Z, PLAY_AREA,
                        GROOVE_HEIGHT, Turtle_Pose, Robot_Base_On, Drive_Path)


# ------------------------------------CONSTANTS-----------------------------------------------------

DT = 0.02

HANDLE_PIN = 0.033                    # handle's joint pin = how far the shaft screws down
N_TURNS = 6                           # turns to screw the shaft on
GRIP_FROM_BUTT = 0.55                 # where the handle sits in the holder (from the butt)

# carry pose: holder in front of the robot, cue straight up (robot-base frame)
CARRY_GRIP_POINT = (0.40, 0.0, 0.65)
CARRY_AIM_DEG = 180.0

START_TURTLE = (0.0, CHANNEL_Y, 180.0)          # (x, y, yaw) on the channel

# straight shot: turtlebot at the end of the table, cue along the table
STRAIGHT_TURTLE = (-CHANNEL_X, 0.0, None)       # None = keep the yaw it arrives with
BALL_1_SPOT = (0.00, 0.20)
STRAIGHT_AIM_DEG, STRAIGHT_ELEV_DEG, STRAIGHT_ROLL_DEG = 0.0, 8.0, 180.0
STRAIGHT_STRIKE_TIME = 0.65           # s, back-swing -> follow-through (shorter = harder hit)

# masse: turtlebot at the side of the table, steep cue
MASSE_TURTLE = (-0.30, -CHANNEL_Y, None)
BALL_2_SPOT = (0.30, -0.15)
MASSE_AIM_DEG, MASSE_ELEV_DEG, MASSE_ROLL_DEG = 45.0, 45.0, -90.0
SIDE_HIT = 0.6                        # off-centre hit, fraction of the ball radius
CURVE_LEFT = True
MASSE_TIP_SPEED = 1.4                 # m/s at contact

ADDRESS_GAP = 0.02                    # tip -> ball surface before the stroke
APPROACH = 0.15                       # cue slides in along its own axis from this far back
CUE_TO_BALL = 1.4                     # ball speed / tip speed at impact
JOINT_SPEED = 2.0                     # rad/s - top joint speed for joint-space moves


# ------------------------------------SET UP THE SCENE-----------------------------------------------------

env = swift.Swift()
env.launch(realtime=True)

env.set_camera_pose([3.0, -2.6, 2.6], [0.0, 0.0, 0.8])

track = Part_Mesh("Robot_Track", pose=SE3())
table = Part_Mesh("Pool_Table", pose=SE3(0, 0, TABLE_Z))
env.add(track)
env.add(table)

turtle = Turtlebot3Waffle()
turtle_xyyaw = START_TURTLE
turtle.base = Turtle_Pose(*turtle_xyyaw)
turtle.add_to_env(env)

robot = Doosan_Robot()
robot.base = Robot_Base_On(turtle.base)
robot.add_to_env(env)

holder = Part_Mesh("Cue_Holder")
env.add(holder)

cue = Cue_Stick(str(Part_Path("Cuestick_Shaft")), str(Part_Path("Cuestick_Handle")),
                handle_insert=HANDLE_PIN)
cue.Add_To_Env(env)
GRIP_FROM_TIP = cue.length - GRIP_FROM_BUTT
TOOL = Holder_Tool(GROOVE_HEIGHT, GRIP_FROM_TIP)

balls, physics, spots = [], [], [BALL_1_SPOT, BALL_2_SPOT]
for spot in spots:
    b = Cue_Ball(str(Part_Path("Cue_Ball")))
    p = Ball_Physics([spot[0], spot[1], FELT_Z + b.radius], b.radius, bounds=PLAY_AREA)
    b.Place(p.pos, p.R)
    b.Add_To_Env(env)
    balls.append(b)
    physics.append(p)
BALL_Z = FELT_Z + balls[0].radius

state = {"attached": False, "shaft_world": None}


# ------------------------------------HELPERS-----------------------------------------------------

def Carry_Tip_Pose(base):
    """Cue held straight up in front of the robot (moves with the turtlebot)."""
    R = Cue_Pose([0, 0, 0], CARRY_AIM_DEG, -90.0).R                  # pointing up
    grip = base * SE3(*CARRY_GRIP_POINT)
    return SE3.Rt(base.R @ R, grip.t + GRIP_FROM_TIP * (base.R @ R)[:, 2])


def Update(q=None):
    """Draw one frame: turtlebot, robot, holder, cue, balls."""
    turtle.base = Turtle_Pose(*turtle_xyyaw)
    robot.base = Robot_Base_On(turtle.base)
    if q is not None:
        robot.q = q
    T_flange = robot.fkine(robot.q)
    holder.T = T_flange.A
    T_tip = T_flange * TOOL
    shaft = None if state["shaft_world"] is None else T_tip.inv() * state["shaft_world"]
    cue.Place(T_tip, shaft_offset=shaft)

    for p in physics:
        if p.Moving():
            p.Step(DT)
    Balls_Collide(physics[0], physics[1])
    for b, p in zip(balls, physics):
        b.Place(p.pos, p.R)
    env.step(DT)


def Move_Joints(q_goal, steps):
    """Joint-space move. Takes longer if a joint has far to go, so no joint
    goes faster than JOINT_SPEED (a jtraj peaks at ~1.9x its average speed)."""
    biggest = np.max(np.abs(np.asarray(q_goal) - robot.q))
    steps = max(steps, int(1.9 * biggest / (JOINT_SPEED * DT)) + 1)
    for q in rtb.jtraj(robot.q, q_goal, steps).q:
        Update(q)


def Move_Line(T_from, T_to, steps):
    """Cue tip in a straight line (IK every step). Returns False if IK fails."""
    q = robot.q
    for s in rtb.quintic(0, 1, steps).q:
        q = Solve_IK(robot, T_from.interp(T_to, s), TOOL, q)
        if q is None:
            return False
        Update(q)
    return True


def Drive_To(goal):
    global turtle_xyyaw
    for pose in Drive_Path(turtle_xyyaw, goal, DT):
        turtle_xyyaw = pose
        Update()


def Balls_Settle():
    while any(p.Moving() for p in physics):
        Update()


def Respot(i):
    """Put ball i back on its spot if it has moved."""
    spot = np.array([spots[i][0], spots[i][1], BALL_Z])
    if np.linalg.norm(physics[i].pos - spot) > 0.005:
        print(f"    re-spotting ball {i + 1}")
        physics[i].pos = spot.copy()
        physics[i].Stop()
        Update()


# ------------------------------------ACTIONS-----------------------------------------------------

def Attach():
    """Shaft comes down from above and screws into the handle in the holder."""
    if state["attached"]:
        print("Cue is already assembled.")
        return
    T_tip = robot.fkine(robot.q) * TOOL
    lined_up = T_tip * SE3.Tz(HANDLE_PIN + 0.15)
    print("[1] Shaft lining up above the handle...")
    start = state["shaft_world"]
    for s in rtb.quintic(0, 1, 80).q:
        state["shaft_world"] = start.interp(lined_up, s)
        Update()
    print("    lowering onto the pin...")
    for z in rtb.quintic(HANDLE_PIN + 0.15, HANDLE_PIN, 50).q:
        state["shaft_world"] = T_tip * SE3.Tz(z)
        Update()
    print("    screwing in...")
    for s in rtb.quintic(0, 1, 150).q:
        state["shaft_world"] = T_tip * SE3.Tz(HANDLE_PIN * (1 - s)) * SE3.Rz(N_TURNS * 2 * pi * s)
        Update()
    state["shaft_world"] = None
    state["attached"] = True
    Update()
    print("    cue assembled.")


def Shot(kind):
    if not state["attached"]:
        print("Attach the shaft first (press 1).")
        return
    i = 0 if kind == "straight" else 1
    Respot(i)
    ball = physics[i].pos.copy()

    if kind == "straight":
        print("[2] Driving to the end of the table...")
        Drive_To(STRAIGHT_TURTLE)
        aim, elev, roll = STRAIGHT_AIM_DEG, STRAIGHT_ELEV_DEG, STRAIGHT_ROLL_DEG
        d = Cue_Pose([0, 0, 0], aim, elev, roll).R[:, 2]
        contact = -physics[i].r * d                                   # centre-ball hit
        profile = Stroke_Profile(DT, follow=0.12, strike_time=STRAIGHT_STRIKE_TIME)
    else:
        print("[3] Driving to the side of the table...")
        Drive_To(MASSE_TURTLE)
        aim, elev, roll = MASSE_AIM_DEG, MASSE_ELEV_DEG, MASSE_ROLL_DEG
        d = Cue_Pose([0, 0, 0], aim, elev, roll).R[:, 2]
        left = np.array([-np.sin(np.radians(aim)), np.cos(np.radians(aim)), 0.0])
        b = SIDE_HIT * physics[i].r * (1 if CURVE_LEFT else -1)
        contact = b * left - np.sqrt(physics[i].r ** 2 - b ** 2) * d  # off-centre hit
        profile = Punch_Stroke_Profile(DT, contact_s=-ADDRESS_GAP, tip_speed=MASSE_TIP_SPEED, follow=0.025)

    T_address = Cue_Pose(ball + contact - ADDRESS_GAP * d, aim, elev, roll)
    T_ready = T_address * SE3.Tz(-APPROACH)                           # pulled back along the cue
    q_carry = robot.q.copy()

    print("    lining up...")
    q_ready = Solve_IK(robot, T_ready, TOOL, Solve_IK(robot, T_address, TOOL, quiet=True))
    if q_ready is None:
        print("    can't reach this shot from here - skipping it.")
        return
    Move_Joints(q_ready, 100)
    if not Move_Line(T_ready, T_address, 40):
        Move_Joints(q_carry, 100)
        return

    print("    rocking and striking...")
    q, prev_tip, hit = robot.q, T_address.t, False
    for s in profile:
        T_tip = T_address * SE3.Tz(-s)
        q = Solve_IK(robot, T_tip, TOOL, q)
        if q is None:
            print("    lost the shot (IK failed) - backing off.")
            break
        if not hit and np.dot(T_tip.t - (ball + contact), d) >= -1e-6:
            physics[i].Hit(CUE_TO_BALL * (T_tip.t - prev_tip) / DT, contact)
            hit = True
            print(f"    hit! ball speed {np.linalg.norm(physics[i].v):.2f} m/s")
        prev_tip = T_tip.t
        Update(q)

    print("    standing back up...")
    T_now = robot.fkine(robot.q) * TOOL
    Move_Line(T_now, T_now * SE3.Tz(-APPROACH), 30)                   # slide back out
    Move_Joints(q_carry, 100)
    Balls_Settle()
    print("    done.")


# ------------------------------------START-----------------------------------------------------

# robot holds the handle upright; the shaft waits above it
q_carry = Solve_IK(robot, Carry_Tip_Pose(robot.base), TOOL)
robot.q = q_carry
state["shaft_world"] = (robot.fkine(q_carry) * TOOL) * SE3.Tz(0.60) * SE3.Tx(-0.30)
Update()


import time

print("Running Sequence: Attach -> Straight Shot -> Masse Shot")

# 1. Attach
Attach()

# 2. Straight Shot
Shot("straight")

# 3. Masse Shot
Shot("masse")

print("Sequence complete. Closing Swift in 5 seconds...")
time.sleep(5)
env.close()
