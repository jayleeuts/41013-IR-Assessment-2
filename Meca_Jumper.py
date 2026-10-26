"""
Meca_Jumper.py - the Meca500 on its own turtlebot, with the jump cue.
Used for the jump and masse shots in the game.

The Meca500 test code is used as is:
  * meca500_robot.py              : the Meca500 model (DH + meshes)
  * Test_Meca500_v1_Jump_Over.py  : the jump cue (Jump_Cue), gripper (Cue_Gripper),
                                    short jump/masse stroke (Short_Stroke_Profile) and
                                    cue-tip pose (Tip_Pose: gripper on the robot's side)
  * the jump-ball physics is in Pool_Trick_Shots.Pool_Ball (same constants)

What this file adds to use it in the workcell:
  * Size      : the real Meca500 reaches ~0.33 m - from the track it cannot reach the
                table at all. Like Jbot, joints AND meshes are scaled together
                (MECA_SIZE) so they always line up. The jump cue is NOT scaled.
  * Mounting  : base sits on top of its own turtlebot and moves with it.
  * IK        : the IK with joint limits, but for any tool transform.
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

MECA_SIZE = 2.0                      # scale of the Meca500 (joints AND meshes)
GRIP_OFFSET = 0.06 * MECA_SIZE       # flange face -> cue centre line (0.06, scaled with the gripper)
GRIP_FROM_TIP = 0.50                 # grip point -> cue tip (along the jump cue)
JUMP_CUE_FILE = Path(__file__).resolve().parent / "jumpcue.stl"
FLANGE_OUT = SE3.Rx(pi)              # last DH frame's Z points back into the wrist
TOOL = FLANGE_OUT * SE3.Tz(GRIP_OFFSET) * SE3.Ry(pi / 2) * SE3.Tz(GRIP_FROM_TIP)
# joints for carrying the jump cue while driving (found by a damped-least-squares search: cue
# straight up, tip about 1.3 m above the robot base, the cue kept clear of the arm and the turtlebot)
CARRY_Q = np.array([-1.1179, -0.542, -0.5664, -2.3224, -1.2428, 2.8096])


def Rebuild_ETS(robot):
    """After changing DH lengths, rebuild each link's ETS. The toolbox builds it once when the link
    is made, and ikine / jacob0 use the ETS - without this they would solve for the UNSCALED
    robot while fkine (and the meshes) use the scaled one."""
    for i, link in enumerate(robot.links):
        ets = link._to_ets(link._sigma, link._theta, link.d, link.alpha, link.a, link.offset, link.isflip, link._mdh)
        for et in ets:
            if et.isjoint:
                et.qlim, et.jindex = link.qlim, i
        link._ets = ets


class Meca500_Scaled(Meca500_Robot):
    """the Meca500 with every length multiplied by `size` (meshes too)."""

    def __init__(self, size=MECA_SIZE, base=None):
        super().__init__(base=base)
        for link in self.links:
            link.d *= size
            link.a *= size
        Rebuild_ETS(self)
        q_cad = [-pi / 2, -pi / 2, 0.0, 0.0, 0.0, 0.0]          # pose the meshes were exported in
        T_cad = self._get_transforms(q_cad)
        self._relation_matrices = [np.linalg.inv(T) for T in T_cad]
        for mesh in self.links_3d:
            mesh.scale = [size] * 3
        self._update_3dmodel()


def Meca_IK(robot, T_tip, tool=TOOL, q_seed=None):
    """(Solve_IK) cue-tip pose -> joints, joint limits respected; None if out of reach."""
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
    """the Meca500 (scaled) on a turtlebot, holding the jump cue."""

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
        """Jump cue held upright above the turtlebot (tip up), arm folded in: the pose of CARRY_Q."""
        return self.robot.fkine(CARRY_Q) * TOOL

    def Add_To_Env(self, env):
        self.turtle.add_to_env(env)
        self.robot.add_to_env(env)
        self.cue.Add_To_Env(env)
        self.gripper.Add_To_Env(env)
        self.Draw_Tool()
