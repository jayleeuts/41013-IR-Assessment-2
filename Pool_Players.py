# ------------------------------------POOL PLAYERS------------------------------------------------------------
# The shooting robots on their turtlebots that the simulation steps through: the UR3 holds the cue in the
# holder and the Meca500 holds the jump cue. Both offer the same small set of methods (base, joints, IK, cue
# tip, carry pose, drawing).

import numpy as np
from math import pi
from spatialmath import SE3
import roboticstoolbox as rtb

from ir_support_extra_robots.robots import Turtlebot3Waffle
from Pool_Parts import Part_Path, Part_Mesh
from Pool_Objects import Cue_Stick
from Pool_Motion import Holder_Tool, Cue_Pose
from Pool_Scene import GROOVE_HEIGHT, TURTLE_HEIGHT, Turtle_Pose
from Meca_Jumper import Meca_Mobile, Meca_IK, TOOL as MECA_TOOL


# ------------------------------------SHARED CUE MEASUREMENTS-------------------------------------------------
# Where the handle's pin goes into the shaft and where the holder grips the cue, and the carry pose: the cue
# held upright in front of the robot while it drives.

HANDLE_PIN = 0.033
GRIP_FROM_BUTT = 0.55


def Carry_Pose(base, grip_point, cue_length):
    R_up = Cue_Pose([0, 0, 0], 180.0, -90.0).R
    grip = base * SE3(*grip_point)
    Rw = base.R @ R_up
    return SE3.Rt(Rw, grip.t + (cue_length - GRIP_FROM_BUTT) * Rw[:, 2])


# ------------------------------------UR3---------------------------------------------------------------------
# The UR3 on its turtlebot holding the cue in the holder: joint speed limits, base and cue-tip poses, carry
# pose, drawing and IK. It uses the toolbox's URDF model with the flange frame tool0.

class UR3_Player:
    VMAX = np.radians([180, 180, 180, 360, 360, 360])
    CARRY_GRIP_POINT = (0.25, 0.0, 0.40)
    END = "tool0"

    def __init__(self, xyyaw):
        self.name = "UR3"
        self.robot = rtb.models.UR3()
        self.kin = rtb.models.UR3()
        self.robot.q = np.radians([0, -90, 90, -90, -90, 0])
        self.turtle = Turtlebot3Waffle()
        self.xyyaw = tuple(xyyaw)
        self.holder = Part_Mesh("Cue_Holder")
        self.cue = Cue_Stick(str(Part_Path("Cuestick_Shaft")), str(Part_Path("Cuestick_Handle")),
                             handle_insert=HANDLE_PIN)
        self.tool = Holder_Tool(GROOVE_HEIGHT, self.cue.length - GRIP_FROM_BUTT)
        self.Set_Base()

    def Set_Base(self):
        self.turtle.base = Turtle_Pose(*self.xyyaw)
        self.robot.base = self.turtle.base * SE3.Tz(TURTLE_HEIGHT)

    @property
    def q(self):
        return self.robot.q

    @q.setter
    def q(self, value):
        self.robot.q = value

    def Flange(self, q=None):
        return self.robot.fkine(self.robot.q if q is None else q, end=self.END)

    def Tip(self, q=None):
        return self.Flange(q) * self.tool

    def Carry_Tip(self):
        return Carry_Pose(self.robot.base, self.CARRY_GRIP_POINT, self.cue.length)

    def IK(self, T_tip, q_seed=None):
        T_local = self.robot.base.inv() * T_tip * self.tool.inv()
        q1 = np.arctan2(T_local.t[1], T_local.t[0])
        seeds = ([np.asarray(q_seed, dtype=float)] if q_seed is not None else [])
        seeds += [np.array([q1 + a, b, c, -1.5, e, 0.0]) for a in (0, pi)
                  for b, c in ((-1.2, 1.4), (-0.6, 0.9), (-1.8, 1.8)) for e in (-1.57, 1.57)]
        for seed in seeds:
            sol = self.kin.ikine_LM(T_local, q0=seed, end=self.END, joint_limits=True, slimit=1)
            if sol.success:
                return sol.q
        return None

    def Draw(self):
        T_flange = self.Flange()
        self.holder.T = T_flange.A
        self.cue.Place(T_flange * self.tool)

    def Add_To_Env(self, env):
        self.turtle.add_to_env(env)
        self.robot.add_to_env(env) if hasattr(self.robot, "add_to_env") else env.add(self.robot)
        env.add(self.holder)
        self.cue.Add_To_Env(env)
        self.Draw()


# ------------------------------------MECA500-----------------------------------------------------------------
# Wraps the scaled Meca500 on its turtlebot (Meca_Jumper.py) so it can be used exactly like the UR3.

class Meca_Player:
    VMAX = np.radians([150, 150, 180, 300, 300, 500])

    def __init__(self, xyyaw):
        self.name = "Meca500"
        self.m = Meca_Mobile(xyyaw)
        self.robot, self.turtle = self.m.robot, self.m.turtle

    @property
    def xyyaw(self):
        return self.m.xyyaw

    @xyyaw.setter
    def xyyaw(self, value):
        self.m.xyyaw = tuple(value)

    @property
    def q(self):
        return self.robot.q

    @q.setter
    def q(self, value):
        self.robot.q = value

    def Set_Base(self):
        self.m.Set_Base()

    def Tip(self, q=None):
        return self.m.Tip_Pose(q)

    def IK(self, T_tip, q_seed=None):
        return Meca_IK(self.robot, T_tip, MECA_TOOL, q_seed)

    def Carry_Tip(self):
        return self.m.Carry_Pose()

    def Draw(self):
        self.m.Draw_Tool()

    def Add_To_Env(self, env):
        self.m.Add_To_Env(env)
