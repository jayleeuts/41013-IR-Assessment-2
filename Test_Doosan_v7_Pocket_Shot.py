"""
Test_Doosan_v7_Pocket_Shot - straight DRAW shot: cue ball pots the object ball
in a corner pocket, then spins back towards the robot.

Runs on its own (no keys):
  1. The robot starts holding the assembled cue upright on the turtlebot.
  2. The turtlebot drives round the track to the near long side.
  3. The robot lines up behind the cue ball and hits it BELOW centre.
  4. The cue ball slides forward with backspin and hits the object ball dead on.
     The object ball rolls into the far corner pocket.
  5. The cue ball stops at the collision, then its backspin pulls it back (the draw).

How the draw works: hitting below centre makes the cue ball spin backwards while
it slides forward. In a straight-on hit, two equal balls swap their speed, so the
cue ball stops dead - but it keeps its backspin, which grips the cloth and drags
it back the way it came.

Why the shot goes along the diagonal: a corner pocket's opening faces into the
table at roughly 45-58 deg to the rails. A ball arriving at a shallower angle
hits the cushion before it reaches the hole (just like a real table).
"""
import time
import swift
import numpy as np
from spatialmath import SE3
import roboticstoolbox as rtb

from Doosan_Robot import Doosan_Robot
from ir_support_extra_robots.robots import Turtlebot3Waffle
from Pool_Parts import Part_Path, Part_Mesh
from Pool_Objects import Cue_Ball, Cue_Stick
from Pool_Motion import (Holder_Tool, Solve_IK, Cue_Pose, Punch_Stroke_Profile,
                         Ball_Physics, Balls_Collide)
from Pool_Scene import (TRACK_NAME, TRACK_POSE, CHANNEL_Y, TABLE_Z, FELT_Z, PLAY_AREA, GROOVE_HEIGHT,
                        POCKETS, POCKET_RADIUS, POCKET_FLOOR_Z,
                        Turtle_Pose, Robot_Base_On, Drive_Path)


# ------------------------------------CONSTANTS-----------------------------------------------------

DT = 0.02

HANDLE_PIN = 0.033
GRIP_FROM_BUTT = 0.55                 # where the handle sits in the holder (from the butt)

CARRY_GRIP_POINT = (0.40, 0.0, 0.65)  # cue held upright in front of the robot (robot-base frame)
CARRY_AIM_DEG = 180.0

START_TURTLE = (0.0, CHANNEL_Y, 180.0)          # (x, y, yaw) on the channel
SHOT_TURTLE = (0.50, -CHANNEL_Y, None)          # near long side; None = keep arrival yaw

TARGET_POCKET = POCKETS[0]            # far right corner (x = +1.104, y = +0.501)
APPROACH_DEG = 48.0                   # direction the balls travel (into the corner's opening)
CUE_BALL_TO_POCKET = 0.80             # m, cue ball starts this far from the pocket centre
OBJECT_BALL_TO_POCKET = 0.45          # m, object ball sits on the same line, closer to the pocket

ELEVATION_DEG = 8.0
GRIP_ROLL_DEG = 180.0
ADDRESS_GAP = 0.02                    # tip -> ball surface before the stroke
APPROACH = 0.15                       # cue slides in along its own axis from this far back
DRAW_LOW = 0.55                       # hit this far BELOW centre (fraction of ball radius) = backspin
TIP_SPEED = 1.40                      # m/s, cue speed at contact (more = more draw)
FOLLOW = 0.03                         # short follow-through - the tip is only ~13 mm above the felt
CUE_TO_BALL = 1.4                     # ball speed / tip speed at impact
JOINT_SPEED = 2.0                     # rad/s - top joint speed for joint-space moves

# both balls sit on the line that runs straight into the pocket (a straight shot)
_dir = np.array([np.cos(np.radians(APPROACH_DEG)), np.sin(np.radians(APPROACH_DEG))])
CUE_BALL_SPOT = np.array(TARGET_POCKET) - CUE_BALL_TO_POCKET * _dir
OBJECT_BALL_SPOT = np.array(TARGET_POCKET) - OBJECT_BALL_TO_POCKET * _dir
AIM_DEG = APPROACH_DEG                # cue points the way the balls should go


# ------------------------------------SET UP THE SCENE-----------------------------------------------------

env = swift.Swift()
env.launch(realtime=True)

env.add(Part_Mesh(TRACK_NAME, pose=TRACK_POSE))
env.add(Part_Mesh("Pool_Table", pose=SE3(0, 0, TABLE_Z)))

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

balls, physics = [], []
for spot in (CUE_BALL_SPOT, OBJECT_BALL_SPOT):
    b = Cue_Ball(str(Part_Path("Cue_Ball")))
    p = Ball_Physics([spot[0], spot[1], FELT_Z + b.radius], b.radius,
                     bounds=PLAY_AREA, pockets=POCKETS, pocket_radius=POCKET_RADIUS,
                     pocket_floor_z=POCKET_FLOOR_Z + b.radius)
    b.Place(p.pos, p.R)
    b.Add_To_Env(env)
    balls.append(b)
    physics.append(p)
cue_ball, object_ball = physics            # physics[0] = cue ball, physics[1] = object ball

env.set_camera_pose([1.9, -1.9, 2.2], [0.75, 0.15, 0.8])


# ------------------------------------HELPERS-----------------------------------------------------

def Carry_Tip_Pose(base):
    """Cue held straight up in front of the robot (moves with the turtlebot)."""
    R = Cue_Pose([0, 0, 0], CARRY_AIM_DEG, -90.0).R
    grip = base * SE3(*CARRY_GRIP_POINT)
    return SE3.Rt(base.R @ R, grip.t + GRIP_FROM_TIP * (base.R @ R)[:, 2])


def Update(q=None):
    """Draw one frame: turtlebot, robot, holder, cue, both balls."""
    turtle.base = Turtle_Pose(*turtle_xyyaw)
    robot.base = Robot_Base_On(turtle.base)
    if q is not None:
        robot.q = q
    T_flange = robot.fkine(robot.q)
    holder.T = T_flange.A
    cue.Place(T_flange * TOOL)
    for p in physics:
        if p.Moving():
            p.Step(DT)
    if cue_ball.potted is None and object_ball.potted is None:
        if Balls_Collide(cue_ball, object_ball):
            hits.append(cue_ball.pos.copy())
    for b, p in zip(balls, physics):
        b.Place(p.pos, p.R)
    env.step(DT)


hits = []                                  # where the cue ball was when it hit the object ball


def Move_Joints(q_goal, steps):
    """Joint-space move, slowed down if needed so no joint passes JOINT_SPEED."""
    biggest = np.max(np.abs(np.asarray(q_goal) - robot.q))
    steps = max(steps, int(1.9 * biggest / (JOINT_SPEED * DT)) + 1)
    for q in rtb.jtraj(robot.q, q_goal, steps).q:
        Update(q)


def Move_Line(T_from, T_to, steps):
    """Cue tip in a straight line (IK every step)."""
    q = robot.q
    for s in rtb.quintic(0, 1, steps).q:
        q = Solve_IK(robot, T_from.interp(T_to, s), TOOL, q)
        Update(q)


def Drive_To(goal):
    global turtle_xyyaw
    for pose in Drive_Path(turtle_xyyaw, goal, DT):
        turtle_xyyaw = pose
        Update()


# ------------------------------------THE SHOT-----------------------------------------------------

def Pocket_Shot():
    print("[1] Driving to the side of the table...")
    Drive_To(SHOT_TURTLE)

    d = Cue_Pose([0, 0, 0], AIM_DEG, ELEVATION_DEG, GRIP_ROLL_DEG).R[:, 2]
    ball_pos = cue_ball.pos.copy()
    # contact point BELOW centre: move down (square to the cue) by DRAW_LOW * radius
    down = -(np.array([0, 0, 1.0]) - d[2] * d)
    down /= np.linalg.norm(down)
    b = DRAW_LOW * cue_ball.r
    contact = b * down - np.sqrt(cue_ball.r ** 2 - b ** 2) * d
    T_address = Cue_Pose(ball_pos + contact - ADDRESS_GAP * d, AIM_DEG, ELEVATION_DEG, GRIP_ROLL_DEG)
    T_ready = T_address * SE3.Tz(-APPROACH)
    q_carry = robot.q.copy()

    print(f"[2] Lining up on the corner pocket at {TARGET_POCKET} ...")
    q_ready = Solve_IK(robot, T_ready, TOOL, Solve_IK(robot, T_address, TOOL, quiet=True))
    Move_Joints(q_ready, 100)
    Move_Line(T_ready, T_address, 40)

    print("[3] Rocking and striking low (draw)...")
    q, prev_tip, hit = robot.q, T_address.t, False
    for s in Punch_Stroke_Profile(DT, contact_s=-ADDRESS_GAP, tip_speed=TIP_SPEED, follow=FOLLOW):
        T_tip = T_address * SE3.Tz(-s)
        q = Solve_IK(robot, T_tip, TOOL, q)
        if not hit and np.dot(T_tip.t - (ball_pos + contact), d) >= -1e-6:
            cue_ball.Hit(CUE_TO_BALL * (T_tip.t - prev_tip) / DT, contact)
            hit = True
            print(f"    hit! cue ball {np.linalg.norm(cue_ball.v):.2f} m/s, backspin {np.linalg.norm(cue_ball.w):.0f} rad/s")
        prev_tip = T_tip.t
        Update(q)

    print("[4] Standing back up, balls rolling...")
    T_now = robot.fkine(robot.q) * TOOL
    Move_Line(T_now, T_now * SE3.Tz(-APPROACH), 30)
    Move_Joints(q_carry, 100)
    while any(p.Moving() for p in physics):
        Update()

    if object_ball.potted is not None:
        print(f"    POTTED - object ball dropped into the pocket at {POCKETS[object_ball.potted]}")
    else:
        print(f"    missed - object ball stopped at ({object_ball.pos[0]:.2f}, {object_ball.pos[1]:.2f})")
    if cue_ball.potted is not None:
        print("    scratch - the cue ball went in too")
    elif hits:
        back = -np.dot(cue_ball.pos[:2] - hits[0][:2], _dir)
        print(f"    draw - cue ball came back {back:.2f} m from where it hit the object ball")


# ------------------------------------RUN-----------------------------------------------------

q_carry = Solve_IK(robot, Carry_Tip_Pose(robot.base), TOOL)
robot.q = q_carry
Update()

print("Running: drive -> line up -> straight draw shot into the corner pocket")
Pocket_Shot()

print("Sequence complete. Closing Swift in 5 seconds...")
time.sleep(5)
env.close()
