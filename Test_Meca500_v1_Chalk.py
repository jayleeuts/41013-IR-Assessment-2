"""
Test_Meca500_v1_Chalk - the Meca500 picks up the chalk and chalks the cue tip.

The cue is held still: it lies level on two rests with its tip 10 cm above the
table, pointing back at the robot. (Nothing in the project holds the cue for
this step, so it is propped up rather than held by the Doosan.)

Sequence:
  1. Come down over the chalk on its stand (cup facing up) and grip its sides.
  2. Lift it and turn it so the cup faces the cue tip.
  3. Push the cup onto the tip.
  4. Rub it left and right in small strokes = chalking.
  5. Back off and put the chalk back on its stand.

Chalk.dae must sit in the same folder as Pool_Parts.py (the project root),
next to Cue_Ball.dae and Cuestick_Shaft.dae, so Part_Path("Chalk") can find it.
The mesh is a 20 mm cube with its origin at the bottom centre and the cup on top (+Z).
"""
import time
import swift
import numpy as np
from math import pi
from spatialmath import SE3
import spatialgeometry as sg
import roboticstoolbox as rtb

from meca500_robot import Meca500_Robot
from Pool_Parts import Part_Path
from Pool_Objects import Cue_Stick
from Pool_Motion import Cue_Pose, Seed_Candidates


# ------------------------------------CONSTANTS-----------------------------------------------------

DT = 0.02

HANDLE_PIN = 0.033

CHALK_SIZE = 0.02                    # chalk is a 20 mm cube
CHALK_CUP_DEPTH = 0.004              # how far the tip sits inside the cup when pressed on
GRIP_OFFSET = 0.04                   # flange face -> chalk centre (clears the palm and the cue)

CHALK_XY = (0.15, -0.18)             # where the chalk stand is (inside the ~0.33 m reach)
STAND_HEIGHT = 0.05                  # chalk sits on top of a small stand / table rail

TIP_XYZ = (0.22, 0.00, 0.10)         # cue tip, held still
CUE_AIM_DEG = 180.0                  # cue points along -X (tip faces the robot)

APPROACH = 0.04                      # straight-line approach / retreat from the cue tip (m)
LIFT = 0.06                          # height above the stand to come down from / lift to (m)

RUB_STROKES = 5                      # left-right strokes
RUB_AMPLITUDE = 0.004                # +/- sideways movement of the chalk (m) - keep it small
RUB_PERIOD = 0.6                     # seconds for one left-right-left stroke
RUB_TWIST_DEG = 8.0                  # small twist of the chalk about the cue, in time with the rub

RETURN_CHALK = True                  # put the chalk back on its stand at the end


# In meca500_robot.py the last DH frame's Z points back INTO the wrist (alpha = pi
# on joint 6): at q = 0 the flange face points along +X but the frame's Z is -X.
# FLANGE_OUT turns it round so +Z points out of the flange face, which is where
# the gripper and the chalk have to be.
FLANGE_OUT = SE3.Rx(pi)

# Flange -> chalk mesh origin. Chalk centre is GRIP_OFFSET out from the flange,
# the cup (+Z of the mesh) points along the flange's +X, fingers grip the chalk's
# sides along the flange's Y.
CHALK_TOOL = FLANGE_OUT * SE3.Tz(GRIP_OFFSET) * SE3.Ry(pi / 2) * SE3.Tz(-CHALK_SIZE / 2)


# ------------------------------------HELPERS-----------------------------------------------------

def Chalk_Pose(centre, cup_dir, flange_side):
    """World pose of the chalk mesh.

    centre      : where the chalk's centre is
    cup_dir     : which way the cup faces
    flange_side : which side of the chalk the robot's flange is on
    """
    z = np.asarray(cup_dir, dtype=float)
    z /= np.linalg.norm(z)
    x = np.asarray(flange_side, dtype=float)
    x = x - np.dot(x, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    origin = np.asarray(centre, dtype=float) - (CHALK_SIZE / 2) * z
    return SE3.Rt(R, origin)


def In_Limits(robot, q):
    return bool(np.all(q >= robot.qlim[0] - 1e-6) and np.all(q <= robot.qlim[1] + 1e-6))


def Solve_IK(robot, T_chalk, tool, q_seed=None, quiet=False):
    """IK for a chalk pose, with joint limits on. Tries q_seed first (smooth
    motion), then the usual spread of seeds. Returns None if nothing works."""
    T_flange = T_chalk * tool.inv()
    seeds = ([np.asarray(q_seed, dtype=float)] if q_seed is not None else []) + Seed_Candidates(robot, T_flange)
    for seed in seeds:
        sol = robot.ikine_LM(T_flange, q0=seed, joint_limits=True, tol=1e-10)
        if sol.success:
            q = (sol.q + pi) % (2 * pi) - pi            # wrap to [-pi, pi]
            if In_Limits(robot, q):
                return q
    if not quiet:
        print(f"IK failed at chalk position {np.round(T_chalk.t, 3)} - target may be out of reach")
    return None


class Chalk_Gripper:
    """Small palm + two fingers that open and close. Purely visual."""

    def __init__(self, grip_offset, width=CHALK_SIZE, color=(0.25, 0.25, 0.28, 1)):
        self.width = width
        self.finger_t = 0.006
        self.finger_len = grip_offset + width / 2 - 0.004       # ends just short of the chalk's far side
        self.palm = sg.Cuboid(scale=[0.04, 0.05, 0.02], color=list(color))
        self.fingers = [sg.Cuboid(scale=[0.016, self.finger_t, self.finger_len], color=list(color))
                        for _ in range(2)]
        self._palm_off = SE3.Tz(0.01)
        self.opening = 1.0                                      # 1 = open, 0 = closed on the chalk

    def Place(self, T_flange):
        self.palm.T = (T_flange * self._palm_off).A
        gap = self.width / 2 + self.finger_t / 2 + 0.008 * self.opening
        for side, finger in zip((-1, 1), self.fingers):
            finger.T = (T_flange * SE3(0, side * gap, self.finger_len / 2)).A

    def Add_To_Env(self, env):
        env.add(self.palm)
        for finger in self.fingers:
            env.add(finger)


# ------------------------------------MAIN-----------------------------------------------------

def main():
    env = swift.Swift()
    env.launch(realtime=True)

    robot = Meca500_Robot()
    robot.base = SE3(0, 0, 0)
    robot.add_to_env(env)

    gripper = Chalk_Gripper(GRIP_OFFSET)
    gripper.Add_To_Env(env)

    # chalk stand
    stand = sg.Cuboid(scale=[0.03, 0.03, STAND_HEIGHT], color=[0.35, 0.2, 0.1, 1],
                      pose=SE3(CHALK_XY[0], CHALK_XY[1], STAND_HEIGHT / 2))
    env.add(stand)

    # chalk, cup up, flange side facing the robot
    radial = np.array([CHALK_XY[0], CHALK_XY[1], 0.0])
    radial /= np.linalg.norm(radial)
    chalk_centre = np.array([CHALK_XY[0], CHALK_XY[1], STAND_HEIGHT + CHALK_SIZE / 2])
    T_chalk_start = Chalk_Pose(chalk_centre, [0, 0, 1], -radial)
    chalk = sg.Mesh(str(Part_Path("Chalk")), pose=T_chalk_start)
    env.add(chalk)

    # cue, held still on two rests
    cue = Cue_Stick(str(Part_Path("Cuestick_Shaft")), str(Part_Path("Cuestick_Handle")),
                    handle_insert=HANDLE_PIN)
    T_tip = Cue_Pose(TIP_XYZ, CUE_AIM_DEG, 0.0)
    cue.Place(T_tip)
    cue.Add_To_Env(env)
    cue_dir = T_tip.R[:, 2]                         # direction the cue points (towards the robot)
    for along, radius in ((0.20, 0.007), (cue.length - 0.10, 0.0145)):
        p = np.array(TIP_XYZ) - along * cue_dir
        h = TIP_XYZ[2] - radius
        env.add(sg.Cuboid(scale=[0.03, 0.06, h], color=[0.35, 0.2, 0.1, 1], pose=SE3(p[0], p[1], h / 2)))

    holding = [False]

    def Update(q):
        if q is None:                               # IK failed - keep the last pose
            q = robot.q
        robot.q = q
        T_flange = robot.fkine(q)
        gripper.Place(T_flange * FLANGE_OUT)
        if holding[0]:
            chalk.T = (T_flange * CHALK_TOOL).A
        env.step(DT)
        time.sleep(DT)

    def Move_Joint(q_to, steps):
        if q_to is None:
            return robot.q
        for q in rtb.jtraj(robot.q, q_to, steps).q:
            Update(q)
        return q_to

    def Move_Line(T_from, T_to, steps):
        """Straight-line move of the chalk frame, IK every step from the last good q."""
        q = robot.q
        for s in np.linspace(0, 1, steps)[1:]:
            T = T_from.interp(T_to, s)
            q_new = Solve_IK(robot, T, CHALK_TOOL, q, quiet=True)
            if q_new is None:
                print(f"    (IK failed at {np.round(T.t, 3)} - holding last pose)")
            else:
                q = q_new
            Update(q)
        return q

    def Set_Gripper(opening, steps=20):
        start = gripper.opening
        for s in np.linspace(0, 1, steps):
            gripper.opening = start + (opening - start) * s
            Update(robot.q)

    Update(robot.q)
    env.set_camera_pose([0.55, -0.55, 0.35], [0.15, -0.05, 0.08])

    # ----- poses -----
    T_pick = T_chalk_start
    T_lift = SE3(0, 0, LIFT) * T_pick                            # straight above the stand
    # Reaching in sideways from the robot's side would need the wrist too close to
    # the base, so the open fingers come straight down over the chalk instead.

    press_centre = np.array(TIP_XYZ) + (CHALK_SIZE / 2 - CHALK_CUP_DEPTH) * cue_dir
    T_press = Chalk_Pose(press_centre, -cue_dir, [0, 0, 1])       # cup faces the tip, flange above
    T_press_pre = SE3(*(APPROACH * cue_dir)) * T_press           # in front of the tip

    # 1. Come down over the chalk and grip it
    print("[1] Reaching for the chalk...")
    q_above = Solve_IK(robot, T_lift, CHALK_TOOL)
    Move_Joint(q_above, 100)
    Move_Line(T_lift, T_pick, 30)
    Set_Gripper(0.0)
    holding[0] = True
    print("    Gripped")

    # 2. Lift, turn the cup towards the cue, line up in front of the tip
    print("[2] Carrying the chalk to the cue tip...")
    Move_Line(T_pick, T_lift, 30)
    q_press_pre = Solve_IK(robot, T_press_pre, CHALK_TOOL, robot.q)
    Move_Joint(q_press_pre, 100)

    # 3. Push the cup onto the tip
    Move_Line(T_press_pre, T_press, 30)

    # 4. Rub left and right (sideways to the cue) with a small twist
    print("[3] Chalking...")
    side = np.cross([0, 0, 1], cue_dir)                           # horizontal, across the cue
    n = int(RUB_STROKES * RUB_PERIOD / DT)
    q = robot.q
    fails = 0
    for i in range(n + 1):
        t = i * DT
        ramp = min(1.0, t / RUB_PERIOD, (n * DT - t) / RUB_PERIOD)   # ease in and out
        phase = np.sin(2 * pi * t / RUB_PERIOD)
        dy = RUB_AMPLITUDE * ramp * phase
        twist = np.radians(RUB_TWIST_DEG) * ramp * phase
        T = SE3(*(dy * side)) * SE3(*TIP_XYZ) * SE3.AngleAxis(twist, cue_dir) * SE3(*(-np.array(TIP_XYZ))) * T_press
        q_new = Solve_IK(robot, T, CHALK_TOOL, q, quiet=True)
        if q_new is None:
            fails += 1
        else:
            q = q_new
        Update(q)
    if fails:
        print(f"    (IK failed on {fails} rub steps - held the last pose; try a smaller RUB_AMPLITUDE)")

    # 5. Back off the tip
    Move_Line(T_press, T_press_pre, 30)

    if RETURN_CHALK:
        print("[4] Putting the chalk back...")
        q_lift = Solve_IK(robot, T_lift, CHALK_TOOL, q_above)
        Move_Joint(q_lift, 100)
        Move_Line(T_lift, T_pick, 30)
        holding[0] = False
        chalk.T = T_pick.A
        Set_Gripper(1.0)
        Move_Line(T_pick, T_lift, 30)
        Move_Joint(np.zeros(6), 100)

    print("Done - cue chalked.")
    env.hold()


if __name__ == "__main__":
    main()
