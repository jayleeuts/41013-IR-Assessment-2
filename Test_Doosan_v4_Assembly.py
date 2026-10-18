"""
Test_Doosan_v4_Assembly - robot holds the HANDLE straight up; the SHAFT flies
in, lines up above it, lowers onto the joint pin and screws itself on.

The cue frame is the same as the other scripts (origin at the tip, +Z along
the cue). Here the cue points straight UP, so "+Z" = up. The shaft's motion is
just an extra transform in that frame:  SE3.Tz(height) * SE3.Rz(spin angle).
"""
import time
import swift
import numpy as np
from math import pi
from spatialmath import SE3
import roboticstoolbox as rtb

from Doosan_Robot import Doosan_Robot
from Pool_Parts import Part_Path
from Pool_Objects import Cue_Stick
from Pool_Motion import Side_Grip_Tool, Simple_Gripper, Solve_IK, Cue_Pose


# ------------------------------------CONSTANTS-----------------------------------------------------

DT = 0.02

HANDLE_PIN = 0.033                    # joint pin length = how far the shaft screws down
N_TURNS = 6                           # turns to screw the shaft fully on

GRIP_OFFSET = 0.08                    # flange face -> cue centre line
GRIP_BELOW_JOINT = 0.25               # robot holds the handle this far below the joint

HOLD_GRIP_POINT = (0.45, 0.0, 0.70)   # where the gripper holds the handle (world)
HOLD_AIM_DEG = 180.0                  # which way the gripper faces around the upright cue

SHAFT_COMES_FROM = (0.35, 0.0, 0.30)  # shaft start, relative to its lined-up spot (world, m)
LINE_UP_HEIGHT = 0.15                 # pauses this far above the pin before lowering


# ------------------------------------MAIN-----------------------------------------------------

def main():
    env = swift.Swift()
    env.launch(realtime=True)

    robot = Doosan_Robot()
    robot.base = SE3(0, 0, 0)
    robot.add_to_env(env)

    gripper = Simple_Gripper(GRIP_OFFSET)
    gripper.Add_To_Env(env)

    cue = Cue_Stick(str(Part_Path("Cuestick_Shaft")), str(Part_Path("Cuestick_Handle")),
                    handle_insert=HANDLE_PIN)
    cue.Add_To_Env(env)

    # flange -> (virtual) cue tip; the tip is where it WILL be once assembled
    grip_from_tip = cue.shaft_length + GRIP_BELOW_JOINT
    TOOL = Side_Grip_Tool(GRIP_OFFSET, grip_from_tip)

    # work out the upright holding pose first (cue +Z pointing straight up)
    T_up = Cue_Pose([0, 0, 0], HOLD_AIM_DEG, -90.0)                 # elevation -90 = pointing up
    tip_xyz = np.array(HOLD_GRIP_POINT) + grip_from_tip * T_up.R[:, 2]
    T_hold = Cue_Pose(tip_xyz, HOLD_AIM_DEG, -90.0)
    q_hold = Solve_IK(robot, T_hold, TOOL)

    # the shaft's pose in the WORLD. T_hold = exactly where it ends up when assembled.
    lined_up = T_hold * SE3.Tz(HANDLE_PIN + LINE_UP_HEIGHT)          # straight above the pin
    start = SE3(*SHAFT_COMES_FROM) * lined_up                        # off to the side
    shaft_world = start

    def Update(q):
        robot.q = q
        T_flange = robot.fkine(q)
        T_tip = T_flange * TOOL
        gripper.Place(T_flange)
        cue.Place(T_tip, shaft_offset=T_tip.inv() * shaft_world)     # shaft placed in world
        env.step(DT)
        time.sleep(DT)

    Update(robot.q)
    env.set_camera_pose([1.6, -1.2, 1.4], [0.45, 0.0, 1.0])

    # 1. Robot brings the handle upright (shaft waits off to the side)
    print("[1] Holding the handle upright...")
    for q in rtb.jtraj(robot.q, q_hold, 120).q:
        Update(q)

    # 2. Shaft flies in and lines up above the pin
    print("[2] Shaft coming in...")
    for s in rtb.quintic(0, 1, 100).q:
        shaft_world = start.interp(lined_up, s)
        Update(q_hold)

    # 3. Lower straight down until the pin touches the shaft's thread
    print("[3] Lowering onto the pin...")
    for z in rtb.quintic(HANDLE_PIN + LINE_UP_HEIGHT, HANDLE_PIN, 60).q:
        shaft_world = T_hold * SE3.Tz(z)
        Update(q_hold)

    # 4. Screw on: turn and move down together (like a real thread)
    print("[4] Screwing on...")
    for s in rtb.quintic(0, 1, 150).q:
        shaft_world = T_hold * SE3.Tz(HANDLE_PIN * (1 - s)) * SE3.Rz(N_TURNS * 2 * pi * s)
        Update(q_hold)

    shaft_world = T_hold                                             # fully assembled
    Update(q_hold)
    print(f"Cue assembled - {cue.length:.3f} m long.")
    env.hold()


if __name__ == "__main__":
    main()
