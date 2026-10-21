"""
Test_Doosan_v4_Assembly - robot holds the HANDLE horizontally facing RIGHT;
positioned forward to prevent self-collision, with the gripper placed ON TOP of 
the handle and the shaft approaching cleanly from the right.
"""
import time
import swift
import numpy as np
from math import pi
from spatialmath import SE3
import roboticstoolbox as rtb

from Doosan_Robot import Doosan_Robot
from meca500_robot import Meca500_Robot
from Pool_Parts import Part_Path
from Pool_Objects import Cue_Stick
from Pool_Motion import Side_Grip_Tool, Simple_Gripper, Solve_IK, Cue_Pose


# ------------------------------------CONSTANTS-----------------------------------------------------

DT = 0.02

HANDLE_PIN = 0.033                    # joint pin length = how far the shaft screws down
N_TURNS = 6                           # turns to screw the shaft fully on

GRIP_OFFSET = 0.08                    # flange face -> cue centre line
GRIP_BELOW_JOINT = 0.25               # robot holds the handle this far below the joint

# 1. SHIFT HOLD POINT FORWARD & UP: Keeps the cue away from the main robot body
HOLD_GRIP_POINT = (0.75, 0.0, 0.47)   

# 2. HORIZONTAL AIM: Points the cue towards the right (+X direction)
HOLD_AIM_DEG = 0.0                    

# 3. APPROACH FROM THE RIGHT: Offset along +Y and +X so it flies in cleanly
SHAFT_COMES_FROM = (0.0, 0.0, 0.0)  

LINE_UP_HEIGHT = 0.15                 # pauses this far along the cue axis before sliding on


# ------------------------------------MAIN-----------------------------------------------------

def main():
    env = swift.Swift()
    env.launch(realtime=True)

    robot = Doosan_Robot()
    robot.base = SE3(0, 0, 0)
    robot.add_to_env(env)

    meca = Meca500_Robot()
    meca.base = SE3(1.5, 0.0, 0.0)
    meca.add_to_env(env)

    gripper = Simple_Gripper(GRIP_OFFSET)
    gripper.Add_To_Env(env)

    cue = Cue_Stick(str(Part_Path("Cuestick_Shaft")), str(Part_Path("Cuestick_Handle")),
                    handle_insert=HANDLE_PIN)
    cue.Add_To_Env(env)

    # flange -> (virtual) cue tip; the tip is where it WILL be once assembled
    grip_from_tip = cue.shaft_length + GRIP_BELOW_JOINT
    
    # 4. FLIP GRIPPER ORIENTATION (ROLL 180°): Forces the gripper to sit ON TOP of the handle
    TOOL = Side_Grip_Tool(GRIP_OFFSET, grip_from_tip)

    # Horizontal pose facing right (Elevation = 0.0deg, Aim = 0.0deg)
    T_up = Cue_Pose([0, 0, 0], HOLD_AIM_DEG, 0.0)
    tip_xyz = np.array(HOLD_GRIP_POINT) + grip_from_tip * T_up.R[:, 2]
    T_hold = Cue_Pose(tip_xyz, HOLD_AIM_DEG, 0.0)
    
    # Solve IK with the inverted tool transform
    q_hold = Solve_IK(robot, T_hold, TOOL)

    # Define shaft path in world frame relative to T_hold
    lined_up = T_hold * SE3.Tz(HANDLE_PIN + LINE_UP_HEIGHT)          
    start = SE3(*SHAFT_COMES_FROM) * lined_up                        
    shaft_world = start

    def Update(q):
        robot.q = q
        T_flange = robot.fkine(q)
        T_tip = T_flange * TOOL
        gripper.Place(T_flange)
        cue.Place(T_tip, shaft_offset=T_tip.inv() * shaft_world)     
        env.step(DT)
        time.sleep(DT)

    Update(robot.q)

    # Set camera view from an elevated angle to observe top-down grip and clearance
    env.set_camera_pose([1.8, -1.2, 1.2], [0.55, 0.0, 0.55])

    # 1. Robot holds handle horizontally out front with gripper on top
    print("[1] Holding the handle horizontally with top-down gripper...")
    for q in rtb.jtraj(robot.q, q_hold, 120).q:
        Update(q)

    # 2. Shaft flies in from the right and aligns
    print("[2] Shaft coming in from the right...")
    for s in rtb.quintic(0, 1, 100).q:
        shaft_world = start.interp(lined_up, s)
        Update(q_hold)

    # 3. Slide horizontally onto the pin
    print("[3] Sliding onto the pin...")
    for z in rtb.quintic(HANDLE_PIN + LINE_UP_HEIGHT, HANDLE_PIN, 60).q:
        shaft_world = T_hold * SE3.Tz(z)
        Update(q_hold)

    # 4. Screw on along the joint thread
    print("[4] Screwing on...")
    for s in rtb.quintic(0, 1, 150).q:
        shaft_world = T_hold * SE3.Tz(HANDLE_PIN * (1 - s)) * SE3.Rz(N_TURNS * 2 * pi * s)
        Update(q_hold)

    shaft_world = T_hold                                             
    Update(q_hold)
    print(f"Cue assembled horizontally without collision - {cue.length:.3f} m long.")
    env.hold()


if __name__ == "__main__":
    main()