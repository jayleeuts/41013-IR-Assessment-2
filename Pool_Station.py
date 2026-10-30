# ------------------------------------POOL STATION------------------------------------------------------------
# The service Meca500 at the head end of the table. It assembles the cue (holds the shaft while the Doosan
# screws the handle on) and chalks every cue. The motions themselves come from the group's test scripts
# (shaft grip, chalking, assembly), copied in below, so the station does exactly what those were tested to
# do.

import numpy as np
from math import pi
from spatialmath import SE3
import spatialgeometry as sg
import roboticstoolbox as rtb

from meca500_robot import Meca500_Robot
from Pool_Parts import Part_Path
from Pool_Motion import Cue_Pose, Seed_Candidates


# ------------------------------------TESTED STATION VALUES---------------------------------------------------
# Found by testing each station on its own. Shaft grip: the joint poses for home, hovering over the shaft, gripping it on
# its rests and holding it level. Assembly: how far the handle's pin goes in, how many turns screw it on and
# how far behind the shaft the handle lines up. Chalking: chalk and stand sizes, where the cue tip is held,
# the approach and lift distances, and the rub (strokes, size, period, twist).

Q_HOVER = np.array([0.0, 0.4963, -0.1726, 0.0, 0.5236, 0.0])
Q_GRIP = np.array([0.0, 0.9545, 0.0, 0.0, 0.0, 0.0])
Q_HOLD = np.array([0.0, 0.6272, -0.604, 0.0, 1.0472, 0.0])
HANDLE_PIN, N_TURNS, LINE_UP = 0.033, 6, 0.15
CHALK_SIZE, CHALK_CUP_DEPTH, GRIP_OFFSET = 0.02, 0.004, 0.04
CHALK_XY, STAND_HEIGHT = (0.15, -0.18), 0.05
TIP_XYZ = (0.22, 0.00, 0.10)
APPROACH, LIFT = 0.04, 0.06
RUB_STROKES, RUB_AMPLITUDE, RUB_PERIOD, RUB_TWIST_DEG = 5, 0.004, 0.6, 8.0
FLANGE_OUT = SE3.Rx(pi)
CHALK_TOOL = FLANGE_OUT * SE3.Tz(GRIP_OFFSET) * SE3.Ry(pi / 2) * SE3.Tz(-CHALK_SIZE / 2)


# ------------------------------------CHALK HELPERS-----------------------------------------------------------
# The chalk's pose from its cup direction, IK for the chalk tool that stays inside the joint limits, and the
# two-finger gripper whose fingers open and close.

def Chalk_Pose(centre, cup_dir, flange_side):
    z = np.asarray(cup_dir, dtype=float)
    z /= np.linalg.norm(z)
    x = np.asarray(flange_side, dtype=float)
    x = x - np.dot(x, z) * z
    x /= np.linalg.norm(x)
    return SE3.Rt(np.column_stack([x, np.cross(z, x), z]), np.asarray(centre, dtype=float) - (CHALK_SIZE / 2) * z)


def Chalk_IK(robot, T_chalk, tool, q_seed=None, quiet=False):
    T_flange = T_chalk * tool.inv()
    seeds = ([np.asarray(q_seed, dtype=float)] if q_seed is not None else []) + Seed_Candidates(robot, T_flange)
    for seed in seeds:
        sol = robot.ikine_LM(T_flange, q0=seed, joint_limits=True, tol=1e-10)
        if sol.success:
            q = (sol.q + pi) % (2 * pi) - pi
            if np.all(q >= robot.qlim[0] - 1e-6) and np.all(q <= robot.qlim[1] + 1e-6):
                return q
    return None


class Chalk_Gripper:
    def __init__(self, grip_offset, width=CHALK_SIZE, color=(0.25, 0.25, 0.28, 1)):
        self.width = width
        self.finger_t = 0.006
        self.finger_len = grip_offset + width / 2 - 0.004
        self.palm = sg.Cuboid(scale=[0.04, 0.05, 0.02], color=list(color))
        self.fingers = [sg.Cuboid(scale=[0.016, self.finger_t, self.finger_len], color=list(color)) for _ in range(2)]
        self.opening = 1.0

    def Place(self, T_flange):
        self.palm.T = (T_flange * SE3.Tz(0.01)).A
        gap = self.width / 2 + self.finger_t / 2 + 0.008 * self.opening
        for side, finger in zip((-1, 1), self.fingers):
            finger.T = (T_flange * SE3(0, side * gap, self.finger_len / 2)).A

    def Add_To_Env(self, env):
        env.add(self.palm)
        for finger in self.fingers:
            env.add(finger)


# ------------------------------------WHERE THE STATION IS----------------------------------------------------
# The station's base pose and where a cue tip must be held to be chalked.

STATION_BASE = SE3(-2.225, -0.30, 0.85)
CUE_AIM = -90.0
SHAFT_DIR = np.array([0.0, -1.0, 0.0])

CHALK_TIP = np.array([-2.005, -0.3, 0.95])
Q_REST = np.array([-pi / 2, 0.5, 0.3, 0.0, -0.8, 0.0])
ASSEMBLED_TIP = np.array([-1.9913378437569835, -0.8, 1.0680945322121411])


def Chalk_Present_Pose(roll):
    return Cue_Pose(CHALK_TIP, CUE_AIM, 0.0, roll)


# ------------------------------------THE STATION ROBOT-------------------------------------------------------
# Builds the robot, its gripper, the chalk and the shaft rests; draws the shaft and chalk with the hand when
# held; and gives the joint paths for picking up the shaft, releasing it, and the five chalking steps.

class Station:
    def __init__(self):
        self.robot = Meca500_Robot()
        self.robot.base = STATION_BASE
        self.kin = Meca500_Robot()
        self.robot.q = Q_REST.copy()
        self.gripper = Chalk_Gripper(GRIP_OFFSET)
        self.chalk = sg.Mesh(str(Part_Path("Chalk")))
        radial = np.array([CHALK_XY[0], CHALK_XY[1], 0.0])
        radial /= np.linalg.norm(radial)
        centre = np.array([CHALK_XY[0], CHALK_XY[1], STAND_HEIGHT + CHALK_SIZE / 2])
        self.T_chalk_home = Chalk_Pose(centre, [0, 0, 1], -radial)
        self.T_chalk = self.T_chalk_home
        self.holding = None
        h = STATION_BASE.t[2]
        sx, sy = (STATION_BASE * SE3(CHALK_XY[0], CHALK_XY[1], 0)).t[:2]
        self.parts = [sg.Cuboid(scale=[0.14, 0.14, h], color=[0.3, 0.3, 0.33, 1.0],
                                pose=SE3(STATION_BASE.t[0], STATION_BASE.t[1], h / 2)),
                      sg.Cuboid(scale=[0.03, 0.03, h + STAND_HEIGHT], color=[0.35, 0.2, 0.1, 1.0],
                                pose=SE3(sx, sy, (h + STAND_HEIGHT) / 2))]
        T_rest = self.Shaft_Pose(Q_GRIP)
        for along in (0.15, 0.55):
            p = (T_rest * SE3.Tz(-along)).t
            hh = p[2] - 0.85 - 0.008
            self.parts.append(sg.Cuboid(scale=[0.04, 0.03, hh], color=[0.35, 0.2, 0.1, 1.0],
                                        pose=SE3(p[0], p[1], 0.85 + hh / 2)))
        self._chalk_qs = None
        self.offset = SE3()
        self._part_T = [np.array(p.T) for p in self.parts]

    def Shaft_Pose(self, q):
        ee = (STATION_BASE * self.kin.fkine(q)).t
        return Cue_Pose(ee + 0.5 * SHAFT_DIR, CUE_AIM, 0.0)

    def Draw(self):
        T_flange = self.robot.fkine(self.robot.q)
        self.gripper.Place(T_flange * FLANGE_OUT)
        if self.holding == "chalk":
            self.T_chalk = (self.offset * STATION_BASE).inv() * T_flange * CHALK_TOOL
        self.chalk.T = (self.offset * STATION_BASE * self.T_chalk).A

    def Hide(self, hidden):
        self.offset = SE3(60.0, 0.0, 0.0) if hidden else SE3()
        self.robot.base = self.offset * STATION_BASE
        for part, T in zip(self.parts, self._part_T):
            part.T = (self.offset * SE3(T, check=False)).A
        self.Draw()

    def Add_To_Env(self, env):
        self.robot.add_to_env(env)
        self.gripper.Add_To_Env(env)
        env.add(self.chalk)
        for p in self.parts:
            env.add(p)
        self.Draw()

    def Reset(self):
        self.robot.q = Q_REST.copy()
        self.holding, self.T_chalk, self.gripper.opening = None, self.T_chalk_home, 1.0
        self.Draw()

    def Pick_Shaft_Path(self):
        to_hover = list(rtb.jtraj(Q_REST, Q_HOVER, 80).q) + [Q_HOVER] * 10
        to_grip = list(rtb.jtraj(Q_HOVER, Q_GRIP, 50).q) + [Q_GRIP] * 15
        lift = list(rtb.jtraj(Q_GRIP, Q_HOLD, 100).q)
        return to_hover + to_grip, lift

    def Chalk_Path(self):
        if self._chalk_qs is not None:
            return self._chalk_qs
        robot = self.kin
        out = []
        q_now = [Q_REST.copy()]

        def joint(q_to, steps, hold):
            for q in rtb.jtraj(q_now[0], q_to, steps).q:
                out.append((q, None, hold))
            q_now[0] = q_to

        def line(T_from, T_to, steps, hold):
            q = q_now[0]
            for s in np.linspace(0, 1, steps)[1:]:
                q_new = Chalk_IK(robot, T_from.interp(T_to, s), CHALK_TOOL, q, quiet=True)
                q = q if q_new is None else q_new
                out.append((q, None, hold))
            q_now[0] = q

        def grip(opening, hold):
            start = 1.0 - opening
            for s in np.linspace(0, 1, 20):
                out.append((q_now[0], start + (opening - start) * s, hold))

        T_tip = Cue_Pose(TIP_XYZ, CUE_AIM, 0.0)
        cue_dir = T_tip.R[:, 2]
        T_pick = self.T_chalk_home
        T_lift = SE3(0, 0, LIFT) * T_pick
        press_centre = np.array(TIP_XYZ) + (CHALK_SIZE / 2 - CHALK_CUP_DEPTH) * cue_dir
        T_press = Chalk_Pose(press_centre, -cue_dir, [0, 0, 1])
        T_press_pre = SE3(*(APPROACH * cue_dir)) * T_press

        q_above = Chalk_IK(robot, T_lift, CHALK_TOOL)
        joint(q_above, 100, False)
        line(T_lift, T_pick, 30, False)
        grip(0.0, False)
        line(T_pick, T_lift, 30, True)
        joint(Chalk_IK(robot, T_press_pre, CHALK_TOOL, q_now[0]), 100, True)
        line(T_press_pre, T_press, 30, True)
        side = np.cross([0, 0, 1], cue_dir)
        n = int(RUB_STROKES * RUB_PERIOD / 0.02)
        q = q_now[0]
        for i in range(n + 1):
            t = i * 0.02
            ramp = min(1.0, t / RUB_PERIOD, (n * 0.02 - t) / RUB_PERIOD)
            phase = np.sin(2 * pi * t / RUB_PERIOD)
            T = (SE3(*(RUB_AMPLITUDE * ramp * phase * side)) * SE3(*TIP_XYZ)
                 * SE3.AngleAxis(np.radians(RUB_TWIST_DEG) * ramp * phase, cue_dir)
                 * SE3(*(-np.array(TIP_XYZ))) * T_press)
            q_new = Chalk_IK(robot, T, CHALK_TOOL, q, quiet=True)
            q = q if q_new is None else q_new
            out.append((q, None, True))
        q_now[0] = q
        line(T_press, T_press_pre, 30, True)
        joint(Chalk_IK(robot, T_lift, CHALK_TOOL, q_above), 100, True)
        line(T_lift, T_pick, 30, True)
        grip(1.0, False)
        line(T_pick, T_lift, 30, False)
        joint(Q_REST.copy(), 100, False)
        self._chalk_qs = out
        return out


# ------------------------------------ASSEMBLY AND CHALKING POSITIONS-----------------------------------------
# Where the shaft is held during assembly, the handle's line-up pose, and where each player parks and how it
# presents its cue to be chalked.



def Assembly_Pose(gap, roll):
    return Cue_Pose(ASSEMBLED_TIP - gap * SHAFT_DIR, CUE_AIM, 0.0, roll)


PRESENT = {
    "D": ((-1.72, 0.15, -90.0), 180, [-2.5928, 0.7301, 1.5123, -0.0, 0.8992, 0.5488]),
    "U": ((-1.72, 0.15, 90.0), 45, [0.1789, 0.3257, 0.0766, -1.797, 0.8012, 1.3205]),
    "M": ((-1.44, 0.5, 90.0), None, [2.1983, 1.3733, -0.856, -0.6255, 0.6237, -0.5078]),
}
ASM_PRESENT = ((-1.72, -0.3, -90.0), 180, [[-2.5646, 0.4702, 1.583, 0.0, 1.0883, 0.577], [-2.598, 0.5319, 1.4844, 0.0, 1.1253, 0.5436], [-2.7136, 0.8677, 0.8949, 0.0, 1.3791, 0.428]])
