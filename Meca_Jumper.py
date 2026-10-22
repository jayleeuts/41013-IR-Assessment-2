"""
Meca_Jumper.py - Shuban's Meca500 on its own turtlebot, with his jump cue.
Used for the jump and masse shots in the game.

Shuban's work is used as is:
  * meca500_robot.py              : his Meca500 model (DH + meshes)
  * Test_Meca500_v1_Jump_Over.py  : his jump cue (Jump_Cue), gripper (Cue_Gripper),
                                    short jump/masse stroke (Short_Stroke_Profile) and
                                    cue-tip pose (Tip_Pose: gripper on the robot's side)
  * his jump-ball physics is in Pool_Trick_Shots.Pool_Ball (same constants)

What this file adds to use it in the workcell:
  * Size      : the real Meca500 reaches ~0.33 m - from the track it cannot reach the
                table at all. Like Jbot, joints AND meshes are scaled together
                (MECA_SIZE) so they always line up. The jump cue is NOT scaled.
  * Mounting  : base sits on top of its own turtlebot and moves with it.
  * IK        : his IK with joint limits, but for any tool transform.
"""
import numpy as np
from math import pi
from pathlib import Path
from spatialmath import SE3

from meca500_robot import Meca500_Robot
from Test_Meca500_v1_Jump_Over import Jump_Cue, Cue_Gripper, Short_Stroke_Profile, In_Limits
from ir_support_extra_robots.robots import Turtlebot3Waffle
from Pool_Motion import Seed_Candidates
from Pool_Scene import TURTLE_HEIGHT, Turtle_Pose


# ------------------------------------SETTINGS-----------------------------------------------------

MECA_SIZE = 3.0                      # scale of Shuban's Meca500 (joints AND meshes)
GRIP_OFFSET = 0.06 * MECA_SIZE       # flange face -> cue centre line (his 0.06, scaled with the gripper)
GRIP_FROM_TIP = 0.30                 # grip point -> cue tip (along the jump cue)
JUMP_CUE_FILE = Path(__file__).resolve().parent / "jumpcue.stl"
FLANGE_OUT = SE3.Rx(pi)              # (his) last DH frame's Z points back into the wrist
TOOL = FLANGE_OUT * SE3.Tz(GRIP_OFFSET) * SE3.Ry(pi / 2) * SE3.Tz(GRIP_FROM_TIP)


class Meca500_Scaled(Meca500_Robot):
    """Shuban's Meca500 with every length multiplied by `size` (meshes too)."""

    def __init__(self, size=MECA_SIZE, base=None):
        super().__init__(base=base)
        for link in self.links:
            link.d *= size
            link.a *= size
        q_cad = [-pi / 2, -pi / 2, 0.0, 0.0, 0.0, 0.0]          # (his) pose the meshes were exported in
        T_cad = self._get_transforms(q_cad)
        self._relation_matrices = [np.linalg.inv(T) for T in T_cad]
        for mesh in self.links_3d:
            mesh.scale = [size] * 3
        self._update_3dmodel()


def Meca_IK(robot, T_tip, tool=TOOL, q_seed=None):
    """(his Solve_IK) cue-tip pose -> joints, joint limits respected; None if out of reach."""
    T_flange = T_tip * tool.inv()
    seeds = ([np.asarray(q_seed, dtype=float)] if q_seed is not None else []) + list(Seed_Candidates(robot, T_flange))
    for seed in seeds:
        sol = robot.ikine_LM(T_flange, q0=seed, joint_limits=True, tol=1e-10)
        if sol.success:
            q = (sol.q + pi) % (2 * pi) - pi
            if In_Limits(robot, q):
                return q
    return None


class Meca_Mobile:
    """Shuban's Meca500 (scaled) on a turtlebot, holding his jump cue."""

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
        """Jump cue held upright above the turtlebot."""
        return self.robot.base * SE3(0.0, 0.0, 0.55) * SE3.Ry(pi)

    def Add_To_Env(self, env):
        self.turtle.add_to_env(env)
        self.robot.add_to_env(env)
        self.cue.Add_To_Env(env)
        self.gripper.Add_To_Env(env)
        self.Draw_Tool()
