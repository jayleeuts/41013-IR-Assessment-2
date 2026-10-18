"""
Test_Doosan_v3_5 - normal shot, cue held from the side.

The gripper holds the cue from the SIDE (cue lies across the fingers), and the
robot stands beside the shot line like a player, so the cue never passes
through the arm. Rock 3 times, strike, ball slides then rolls - and you can
see it rotate from the red markings.
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
                         Stroke_Profile, Ball_Physics)


# ------------------------------------CONSTANTS-----------------------------------------------------

DT = 0.02

HANDLE_PIN = 0.033                   # handle's joint pin slides into the shaft

GRIP_OFFSET = 0.08                   # flange face -> cue centre line (cue sits across the fingers)
GRIP_FROM_TIP = 0.50                 # grip point -> cue tip, along the cue

TABLE_TOP_Z = 0.0                    # table surface height (robot base at z = 0)
BALL_XY = (0.60, 0.45)               # ball off to the side, so the cue line misses the robot
AIM_DEG = 0.0                        # shot direction about world Z (0 = +X)
ELEVATION_DEG = 8.0                  # cue tilted down, like a real player
GRIP_ROLL_DEG = -90.0                # gripper beside the cue
ADDRESS_GAP = 0.02                   # tip sits this far from the ball's surface

CUE_TO_BALL = 1.4                    # ball speed / tip speed at impact (cue is heavier)

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
    physics = Ball_Physics(ball_start, ball.radius)
    ball.Place(physics.pos, physics.R)
    ball.Add_To_Env(env)

    cue = Cue_Stick(str(Part_Path("Cuestick_Shaft")), str(Part_Path("Cuestick_Handle")),
                    handle_insert=HANDLE_PIN)
    cue.Add_To_Env(env)

    def Update(q):
        # robot + gripper + cue follow the joints; ball follows its physics
        robot.q = q
        T_flange = robot.fkine(q)
        gripper.Place(T_flange)
        cue.Place(T_flange * TOOL)
        if physics.Moving():
            physics.Step(DT)
            ball.Place(physics.pos, physics.R)
        env.step(DT)
        time.sleep(DT)

    Update(robot.q)
    env.set_camera_pose([1.4, -1.0, 1.0], [0.6, 0.4, 0.1])

    # 1. Address pose: tip just behind the ball, cue along the aim direction
    cue_dir = Cue_Pose([0, 0, 0], AIM_DEG, ELEVATION_DEG, GRIP_ROLL_DEG).R[:, 2]
    tip = ball_start - (ball.radius + ADDRESS_GAP) * cue_dir
    T_address = Cue_Pose(tip, AIM_DEG, ELEVATION_DEG, GRIP_ROLL_DEG)
    q_address = Solve_IK(robot, T_address, TOOL)

    print("[1] Lining up the shot...")
    for q in rtb.jtraj(robot.q, q_address, 100).q:
        Update(q)

    # 2. Rock + strike along the cue axis (IK every step)
    print("[2] Rocking and striking...")
    q, prev_tip, hit = q_address, T_address.t, False
    contact = -ball.radius * cue_dir                  # centre-ball hit: tip meets the ball here

    for s in Stroke_Profile(DT, follow=0.12, strike_time=0.45):
        T_tip = T_address * SE3.Tz(-s)
        q = Solve_IK(robot, T_tip, TOOL, q)
        if not hit and np.dot(T_tip.t - (ball_start + contact), cue_dir) >= -1e-6:
            tip_vel = (T_tip.t - prev_tip) / DT
            physics.Hit(CUE_TO_BALL * tip_vel, contact)
            hit = True
            print(f"[3] Hit! tip speed {np.linalg.norm(tip_vel):.2f} m/s")
        prev_tip = T_tip.t
        Update(q)

    # 3. Ball slides, starts rolling, stops
    while physics.Moving():
        Update(q)

    print(f"Shot complete - ball travelled {np.linalg.norm(physics.pos - ball_start):.2f} m")

    print("\nSequence complete.")
    input("Press Enter to close\n")
    env.close()


if __name__ == "__main__":
    main()
