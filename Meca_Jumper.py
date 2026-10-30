# ------------------------------------MECA JUMPER-------------------------------------------------------------
# The Meca500 that plays the jump and masse shots, scaled up and mounted on its own turtlebot with the jump
# cue.

import struct
import numpy as np
from math import pi
from pathlib import Path
from spatialmath import SE3
import spatialgeometry as sg

from meca500_robot import Meca500_Robot
from ir_support_extra_robots.robots import Turtlebot3Waffle
from Pool_Motion import Seed_Candidates, Rebuild_ETS
from Pool_Objects import _measure_rod, _part_offset, _guess_scale
from Pool_Scene import TURTLE_HEIGHT, Turtle_Pose


# ------------------------------------SETTINGS----------------------------------------------------------------
# The scale, where the gripper holds the jump cue, the tool transform from flange to cue tip and the carry
# pose.

MECA_SIZE = 2.0
GRIP_OFFSET = 0.06 * MECA_SIZE
GRIP_FROM_TIP = 0.50
JUMP_CUE_FILE = Path(__file__).resolve().parent / "Pool_Parts" / "jumpcue.stl"
if not JUMP_CUE_FILE.exists():
    JUMP_CUE_FILE = Path(__file__).resolve().parent / "jumpcue.stl"
FLANGE_OUT = SE3.Rx(pi)
TOOL = FLANGE_OUT * SE3.Tz(GRIP_OFFSET) * SE3.Ry(pi / 2) * SE3.Tz(GRIP_FROM_TIP)
CARRY_Q = np.array([-1.1179, -0.542, -0.5664, -2.3224, -1.2428, 2.8096])
SHAFT_WIDTH = 0.012


# ------------------------------------JUMP CUE, GRIPPER AND STROKE--------------------------------------------
# The jump cue is read from its STL model and placed from the cue tip pose; a simple two-finger gripper holds
# it. The short punchy stroke eases back, pauses, speeds up to the planned speed exactly at the ball and
# brakes quickly, because a steep jump or masse cue has very little room past the ball.

def In_Limits(robot, q):
    return bool(np.all(q >= robot.qlim[0] - 1e-6) and np.all(q <= robot.qlim[1] + 1e-6))


def Stl_Vertices(path):
    data = Path(path).read_bytes()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            tri = np.frombuffer(data[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")]))
            return tri["v"].reshape(-1, 3).astype(float)
    pts = [line.split()[1:4] for line in data.decode(errors="ignore").splitlines()
           if line.strip().startswith("vertex")]
    return np.array(pts, dtype=float)


class Jump_Cue:
    def __init__(self, stl_path, flip=False, color=(0.82, 0.62, 0.36, 1)):
        v = Stl_Vertices(stl_path)
        scale = _guess_scale(float(np.ptp(v, axis=0).max()), 1.0, "jump cue")
        v = v * scale
        self.length, tip_pt, direction = _measure_rod(v, flip)
        self._offset = _part_offset(tip_pt, direction, 0.0)
        self.mesh = sg.Mesh(str(stl_path), scale=[scale] * 3, color=list(color))

    def Place(self, T_tip):
        self.mesh.T = (T_tip * self._offset).A

    def Add_To_Env(self, env):
        env.add(self.mesh)


class Cue_Gripper:
    def __init__(self, grip_offset, width=SHAFT_WIDTH, color=(0.25, 0.25, 0.28, 1)):
        self.finger_t = 0.006
        self.gap = width / 2 + self.finger_t / 2
        self.finger_len = grip_offset + width / 2 - 0.002
        self.palm = sg.Cuboid(scale=[0.04, 0.05, 0.02], color=list(color))
        self.fingers = [sg.Cuboid(scale=[0.016, self.finger_t, self.finger_len], color=list(color))
                        for _ in range(2)]

    def Place(self, T_face):
        self.palm.T = (T_face * SE3.Tz(0.01)).A
        for side, finger in zip((-1, 1), self.fingers):
            finger.T = (T_face * SE3(0, side * self.gap, self.finger_len / 2)).A

    def Add_To_Env(self, env):
        env.add(self.palm)
        for finger in self.fingers:
            env.add(finger)


def Short_Stroke_Profile(dt, contact_s, tip_speed, follow, backswing):
    t = np.linspace(0, 1, 40)
    out = list(backswing * (3 * t ** 2 - 2 * t ** 3)) + [backswing] * 10
    accel = tip_speed ** 2 / (2 * (backswing - contact_s))
    t, t_hit = 0.0, tip_speed / accel
    while t < t_hit:
        t += dt
        out.append(backswing - 0.5 * accel * min(t, t_hit) ** 2)
    decel = tip_speed ** 2 / (2 * follow)
    t, t_stop = 0.0, tip_speed / decel
    while t < t_stop:
        t += dt
        tt = min(t, t_stop)
        out.append(contact_s - (tip_speed * tt - 0.5 * decel * tt ** 2))
    return out


# ------------------------------------THE SCALED MECA500------------------------------------------------------
# The Meca500 with its DH lengths and meshes scaled by MECA_SIZE (and its link geometry rebuilt so IK uses
# the scaled lengths), and IK that stays inside the joint limits.

class Meca500_Scaled(Meca500_Robot):
    def __init__(self, size=MECA_SIZE, base=None):
        super().__init__(base=base)
        for link in self.links:
            link.d *= size
            link.a *= size
        Rebuild_ETS(self)
        q_cad = [-pi / 2, -pi / 2, 0.0, 0.0, 0.0, 0.0]
        T_cad = self._get_transforms(q_cad)
        self._relation_matrices = [np.linalg.inv(T) for T in T_cad]
        for mesh in self.links_3d:
            mesh.scale = [size] * 3
        self._update_3dmodel()


def Meca_IK(robot, T_tip, tool=TOOL, q_seed=None):
    T_flange = T_tip * tool.inv()
    seeds = ([np.asarray(q_seed, dtype=float)] if q_seed is not None else []) + list(Seed_Candidates(robot, T_flange))
    for seed in seeds:
        sol = robot.ikine_LM(T_flange, q0=seed, joint_limits=True, tol=1e-10)
        if sol.success:
            q = (sol.q + pi) % (2 * pi) - pi
            if In_Limits(robot, q):
                return q
    return None


# ------------------------------------MECA500 ON ITS TURTLEBOT------------------------------------------------
# Base pose on the track, cue-tip pose, carry pose and drawing of the gripper and jump cue.

class Meca_Mobile:
    def __init__(self, xyyaw):
        self.robot = Meca500_Scaled()
        self.turtle = Turtlebot3Waffle()
        self.xyyaw = tuple(xyyaw)
        self.cue = Jump_Cue(JUMP_CUE_FILE)
        self.gripper = Cue_Gripper(GRIP_OFFSET)
        self.Set_Base()

    def Set_Base(self):
        self.turtle.base = Turtle_Pose(*self.xyyaw)
        self.robot.base = self.turtle.base * SE3.Tz(TURTLE_HEIGHT)

    def Tip_Pose(self, q=None):
        return self.robot.fkine(self.robot.q if q is None else q) * TOOL

    def Draw_Tool(self):
        T_flange = self.robot.fkine(self.robot.q)
        self.gripper.Place(T_flange * FLANGE_OUT)
        self.cue.Place(T_flange * TOOL)

    def IK(self, T_tip, q_seed=None):
        return Meca_IK(self.robot, T_tip, TOOL, q_seed)

    def Carry_Pose(self):
        return self.robot.fkine(CARRY_Q) * TOOL

    def Add_To_Env(self, env):
        self.turtle.add_to_env(env)
        self.robot.add_to_env(env)
        self.cue.Add_To_Env(env)
        self.gripper.Add_To_Env(env)
        self.Draw_Tool()
