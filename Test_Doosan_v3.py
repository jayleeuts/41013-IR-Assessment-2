import time
import swift
import numpy as np
from math import pi
from spatialmath import SE3
import roboticstoolbox as rtb

from Doosan_Robot import Doosan_Robot
from Pool_Parts import Part_Path
from Pool_Objects import Cue_Ball, Cue_Stick


# ------------------------------------CONSTANTS-----------------------------------------------------

DT = 0.02

# your .dae files in the Pool_Parts folder, by name
BALL_DAE = str(Part_Path("Cue_Ball"))
SHAFT_DAE = str(Part_Path("Cuestick_Shaft"))
HANDLE_DAE = str(Part_Path("Cuestick_Handle"))
HANDLE_PIN = 0.033                   # handle's joint pin length - slides into the shaft

# where the robot holds the cue: distance from the gripper (flange) to the cue TIP.
# Smaller = easier to reach; the rest of the cue sticks out behind the wrist.
GRIP_FROM_TIP = 0.40

TABLE_TOP_Z = 0.0                    # table surface height (robot base is at z = 0)
BALL_XY = (0.85, 0.0)                # cue ball position on the table
AIM_DEG = 0.0                        # shot direction about world Z (0 = +X)
TILT_DEG = 8.0                       # cue tilted down, like a real player
ADDRESS_GAP = 0.02                   # tip sits this far from the ball's surface

# stroke shape
N_ROCKS, ROCK_AMP, ROCK_TIME = 3, 0.08, 1.0     # practice rocks
BACK, BACK_TIME = 0.12, 0.8                     # final back-swing
FOLLOW, STRIKE_TIME = 0.10, 0.25                # follow-through past the address point

BALL_FRICTION = 0.97                            # speed kept per step after impact

# flange -> cue tip (cue frame +Z = flange +Z)
TOOL = SE3.Tz(GRIP_FROM_TIP)


# ------------------------------------HELPERS-----------------------------------------------------

def Cue_Tip_Pose(robot, q):
    # forward kinematics: where the cue tip is for joint angles q
    return robot.fkine(q) * TOOL


def Solve_IK(robot, T_tip, q_seed):
    # inverse kinematics for a desired cue-tip pose
    sol = robot.ikine_LM(T_tip * TOOL.inv(), q0=q_seed, joint_limits=False)
    if not sol.success:
        print(f"IK failed at tip position {np.round(T_tip.t, 3)} - target may be out of reach")
    return sol.q


def Address_Pose(ball_centre, ball_radius):
    # cue tip just behind the ball, pointing along the aim direction, tilted down
    R = SE3.Rz(np.deg2rad(AIM_DEG)) * SE3.Ry(pi / 2 + np.deg2rad(TILT_DEG))
    cue_dir = R.R[:, 2]
    tip = ball_centre - (ball_radius + ADDRESS_GAP) * cue_dir
    return SE3(*tip) * R, cue_dir


def Stroke_Profile():
    # how far the tip is pulled back from the address point at each step
    s = [ROCK_AMP * (1 - np.cos(2 * pi * t / ROCK_TIME)) / 2
         for t in np.arange(0, N_ROCKS * ROCK_TIME, DT)]
    s += list(rtb.quintic(0, BACK, int(BACK_TIME / DT)).q)
    s += list(rtb.quintic(BACK, -FOLLOW, int(STRIKE_TIME / DT)).q)
    return np.array(s)


# ------------------------------------MAIN-----------------------------------------------------

def main():
    # 1. Launch Swift
    env = swift.Swift()
    env.launch(realtime=True)

    # 2. Robot
    robot = Doosan_Robot()
    robot.base = SE3(0, 0, 0)
    robot.add_to_env(env)

    # 3. Ball and cue from your .dae files
    ball = Cue_Ball(BALL_DAE)
    ball_pos = np.array([BALL_XY[0], BALL_XY[1], TABLE_TOP_Z + ball.radius])
    ball.Place(ball_pos)
    ball.Add_To_Env(env)

    cue = Cue_Stick(SHAFT_DAE, HANDLE_DAE, handle_insert=HANDLE_PIN)
    cue.Add_To_Env(env)
    cue.Place(Cue_Tip_Pose(robot, robot.q))       # cue starts in the gripper

    env.set_camera_pose([1.8, 1.6, 1.0], [0.5, 0.0, 0.2])
    env.step()
    print("Robot, cue and ball loaded into Swift!")

    # ball state (it only starts moving once the tip reaches it)
    ball_vel = np.zeros(3)

    def Update(q):
        # move robot, carry the cue on the gripper, roll the ball
        nonlocal ball_pos, ball_vel
        robot.q = q
        cue.Place(Cue_Tip_Pose(robot, q))
        if np.linalg.norm(ball_vel) > 1e-3:
            ball_pos = ball_pos + ball_vel * DT
            ball_vel = ball_vel * BALL_FRICTION
            ball.Place(ball_pos)
        env.step(DT)
        time.sleep(DT)

    # 4. Line up behind the ball (IK once, then a joint-space move)
    T_address, cue_dir = Address_Pose(ball_pos, ball.radius)
    q_seed = np.array([0, pi / 4, pi / 2, 0, pi / 4, 0])
    q_address = Solve_IK(robot, T_address, q_seed)

    print("[1] Lining up the shot...")
    for q in rtb.jtraj(robot.q, q_address, 100).q:
        Update(q)

    # 5. Rock back and forth, then strike (straight line along the cue, IK every step)
    print("[2] Rocking and striking...")
    flat_dir = np.array([cue_dir[0], cue_dir[1], 0.0])
    flat_dir /= np.linalg.norm(flat_dir)
    hit = False
    q = q_address
    prev_tip = T_address.t

    for s in Stroke_Profile():
        T_tip = T_address * SE3.Tz(-s)
        q = Solve_IK(robot, T_tip, q)

        # contact: tip has reached the ball's surface -> ball takes the tip's speed
        tip = T_tip.t
        if not hit and np.dot(tip - ball_pos, cue_dir) >= -ball.radius:
            tip_speed = np.linalg.norm(tip - prev_tip) / DT
            ball_vel = flat_dir * tip_speed
            hit = True
            print(f"[3] Hit! ball speed {tip_speed:.2f} m/s")
        prev_tip = tip

        Update(q)

    # 6. Let the ball roll to a stop
    while np.linalg.norm(ball_vel) > 0.01:
        Update(q)

    print("Shot complete.")
    env.hold()


if __name__ == "__main__":
    main()
