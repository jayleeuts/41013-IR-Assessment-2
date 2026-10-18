"""
Test_Doosan_v5_Masse - masse shot: steep cue, hit off-centre, ball curves.

How the curve happens (same physics a real cue ball follows):
  1. The cue comes down steeply and hits the ball to one SIDE of centre.
  2. That gives the ball a lot of spin about a near-horizontal axis.
  3. The ball's bottom is slipping sideways on the cloth, so friction pushes
     it sideways -> the path bends.
  4. Once the slipping stops, the ball rolls straight on.
The red markings on the ball show it spinning the whole way.
"""
import time
import swift
import numpy as np
from spatialmath import SE3
import roboticstoolbox as rtb

from Doosan_Robot import Doosan_Robot
from Pool_Parts import Part_Path
from Pool_Objects import Cue_Ball, Cue_Stick
from Pool_Motion import (Side_Grip_Tool, Simple_Gripper, Solve_IK, Cue_Pose,
                         Punch_Stroke_Profile, Ball_Physics)


# ------------------------------------CONSTANTS-----------------------------------------------------

DT = 0.02

HANDLE_PIN = 0.033

GRIP_OFFSET = 0.08                   # flange face -> cue centre line
GRIP_FROM_TIP = 0.35                 # grip point -> cue tip (shorter = easier to reach steeply)

TABLE_TOP_Z = 0.0
BALL_XY = (0.65, 0.50)
AIM_DEG = 0.0                        # direction the cue points (world Z rotation)
ELEVATION_DEG = 60.0                 # steep cue = masse
GRIP_ROLL_DEG = -90.0                # gripper beside the cue

SIDE_HIT = 0.6                       # how far off-centre, as a fraction of the ball radius
CURVE_LEFT = True                    # True = hit left of centre -> curves left
ADDRESS_GAP = 0.02                   # tip -> ball surface before the stroke
TIP_SPEED = 1.5                      # cue speed at contact (m/s)
FOLLOW = 0.025                       # braking distance after contact (table is just below!)

CUE_TO_BALL = 1.4                    # ball impulse / tip speed (cue is heavier)
MU_SLIDE = 0.12                      # cloth friction while the ball slips (bends the path)
MU_ROLL = 0.05                       # rolling resistance once it rolls

TOOL = Side_Grip_Tool(GRIP_OFFSET, GRIP_FROM_TIP)


# ------------------------------------MAIN-----------------------------------------------------

def main():
    env = swift.Swift()
    env.launch(realtime=True)

    robot = Doosan_Robot()
    robot.base = SE3(0, 0, 0)
    robot.add_to_env(env)

    gripper = Simple_Gripper(GRIP_OFFSET)
    gripper.Add_To_Env(env)

    ball = Cue_Ball(str(Part_Path("Cue_Ball")))
    ball_start = np.array([BALL_XY[0], BALL_XY[1], TABLE_TOP_Z + ball.radius])
    physics = Ball_Physics(ball_start, ball.radius, mu_slide=MU_SLIDE, mu_roll=MU_ROLL)
    ball.Place(physics.pos, physics.R)
    ball.Add_To_Env(env)

    cue = Cue_Stick(str(Part_Path("Cuestick_Shaft")), str(Part_Path("Cuestick_Handle")),
                    handle_insert=HANDLE_PIN)
    cue.Add_To_Env(env)

    trail = []                                     # where the ball has been, for the summary

    def Update(q):
        robot.q = q
        T_flange = robot.fkine(q)
        gripper.Place(T_flange)
        cue.Place(T_flange * TOOL)
        if physics.Moving():
            physics.Step(DT)
            ball.Place(physics.pos, physics.R)
            trail.append(physics.pos.copy())
        env.step(DT)
        time.sleep(DT)

    Update(robot.q)
    env.set_camera_pose([1.6, -0.9, 1.3], [1.0, 0.8, 0.0])

    # 1. Where the tip meets the ball: off to one side of centre
    cue_dir = Cue_Pose([0, 0, 0], AIM_DEG, ELEVATION_DEG, GRIP_ROLL_DEG).R[:, 2]
    left = np.array([-np.sin(np.radians(AIM_DEG)), np.cos(np.radians(AIM_DEG)), 0.0])
    b = SIDE_HIT * ball.radius * (1 if CURVE_LEFT else -1)
    contact = b * left - np.sqrt(ball.radius ** 2 - b ** 2) * cue_dir   # on the ball surface

    tip = ball_start + contact - ADDRESS_GAP * cue_dir
    T_address = Cue_Pose(tip, AIM_DEG, ELEVATION_DEG, GRIP_ROLL_DEG)
    q_address = Solve_IK(robot, T_address, TOOL)

    print("[1] Lining up the masse...")
    for q in rtb.jtraj(robot.q, q_address, 120).q:
        Update(q)

    # 2. Rock, back-swing, then punch down through the contact point
    print("[2] Rocking and striking...")
    q, prev_tip, hit = q_address, T_address.t, False
    for s in Punch_Stroke_Profile(DT, contact_s=-ADDRESS_GAP, tip_speed=TIP_SPEED, follow=FOLLOW):
        T_tip = T_address * SE3.Tz(-s)
        q = Solve_IK(robot, T_tip, TOOL, q)
        if not hit and np.dot(T_tip.t - (ball_start + contact), cue_dir) >= -1e-6:
            tip_vel = (T_tip.t - prev_tip) / DT
            physics.Hit(CUE_TO_BALL * tip_vel, contact)
            hit = True
            print(f"[3] Hit! tip speed {np.linalg.norm(tip_vel):.2f} m/s, "
                  f"spin {np.linalg.norm(physics.w):.0f} rad/s")
        prev_tip = T_tip.t
        Update(q)

    # 3. Pull the cue back up out of the way, ball keeps going
    q_up = Solve_IK(robot, T_address * SE3.Tz(-0.15), TOOL, q)
    for q in rtb.jtraj(q, q_up, 40).q:
        Update(q)
    while physics.Moving():
        Update(q)

    if trail:
        trail = np.array(trail)
        far = np.where(np.linalg.norm(trail - trail[-1], axis=1) > 0.2)[0]   # last 20 cm of travel
        ref = trail[far[-1]] if len(far) else trail[0]
        heading = np.degrees(np.arctan2(trail[-1][1] - ref[1], trail[-1][0] - ref[0]))
        print(f"Shot complete - ball ended at ({trail[-1][0]:.2f}, {trail[-1][1]:.2f}), "
              f"curved to a heading of {heading:.0f} deg (aimed at {AIM_DEG:.0f} deg)")
    env.hold()


if __name__ == "__main__":
    main()
