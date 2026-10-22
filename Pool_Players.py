"""
Pool_Players.py - the three robots that play, each on its own turtlebot, all
driven by the SAME shot code (Pool_Game.Take_Shot).

  Doosan_Player : Nicole's Doosan M0609 + Cue_Holder + assembled cue
  UR3_Player    : UR3 from the Robotics Toolbox (rtb.models.UR3) + the same holder and cue
  Meca_Player   : Shuban's Meca500 (Meca_Jumper.py) + his jump cue - jump / masse shots

Every player gives the game the same things:
    xyyaw            turtlebot pose on the track
    Set_Base()       put turtlebot + arm where xyyaw says
    q                current joint angles (set it to move the arm)
    Tip(q)           world pose of the cue tip (origin at the tip, +Z along the cue)
    IK(T_tip, seed)  joint angles for a cue-tip pose (None if out of reach)
    Carry_Tip()      cue held upright, used while driving
    Draw()           move the cue / holder meshes to match the arm
    VMAX             joint speed limits (rad/s)
"""
import numpy as np
from math import pi
from spatialmath import SE3
import roboticstoolbox as rtb

from Doosan_Robot import Doosan_Robot
from ir_support_extra_robots.robots import Turtlebot3Waffle
from Pool_Parts import Part_Path, Part_Mesh
from Pool_Objects import Cue_Stick
from Pool_Motion import Holder_Tool, Cue_Pose, Seed_Candidates
from Pool_Scene import GROOVE_HEIGHT, TURTLE_HEIGHT, DOOSAN_FOOT, Turtle_Pose
from Meca_Jumper import Meca_Mobile, Meca_IK, TOOL as MECA_TOOL

HANDLE_PIN = 0.033
GRIP_FROM_BUTT = 0.55                 # where the handle sits in the holder (from the butt)


class _Holder_Player:
    """Common code for a robot holding the assembled cue in the Cue_Holder."""
    CARRY_GRIP_POINT = (0.40, 0.0, 0.65)      # cue held upright here (robot-base frame)
    MOUNT = TURTLE_HEIGHT

    def __init__(self, name, xyyaw):
        self.name = name
        self.turtle = Turtlebot3Waffle()
        self.xyyaw = tuple(xyyaw)
        self.holder = Part_Mesh("Cue_Holder")
        self.cue = Cue_Stick(str(Part_Path("Cuestick_Shaft")), str(Part_Path("Cuestick_Handle")),
                             handle_insert=HANDLE_PIN)
        self.tool = Holder_Tool(GROOVE_HEIGHT, self.cue.length - GRIP_FROM_BUTT)
        self.Set_Base()

    # --- pose ---
    def Set_Base(self):
        self.turtle.base = Turtle_Pose(*self.xyyaw)
        self.robot.base = self.turtle.base * SE3.Tz(self.MOUNT)

    @property
    def q(self):
        return self.robot.q

    @q.setter
    def q(self, value):
        self.robot.q = value

    def Flange(self, q=None):
        return self.robot.fkine(self.robot.q if q is None else q)

    def Tip(self, q=None):
        return self.Flange(q) * self.tool

    def Carry_Tip(self):
        """Cue held straight up in front of the robot (moves with the turtlebot)."""
        R_up = Cue_Pose([0, 0, 0], 180.0, -90.0).R
        base = self.robot.base
        grip = base * SE3(*self.CARRY_GRIP_POINT)
        Rw = base.R @ R_up
        return SE3.Rt(Rw, grip.t + (self.cue.length - GRIP_FROM_BUTT) * Rw[:, 2])

    # --- drawing ---
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


class Doosan_Player(_Holder_Player):
    VMAX = np.radians([150, 150, 180, 225, 225, 225])
    MOUNT = TURTLE_HEIGHT + DOOSAN_FOOT

    def __init__(self, xyyaw):
        self.robot = Doosan_Robot()
        super().__init__("Doosan", xyyaw)

    def IK(self, T_tip, q_seed=None):
        T_flange = T_tip * self.tool.inv()
        seeds = ([np.asarray(q_seed, dtype=float)] if q_seed is not None else []) + Seed_Candidates(self.robot, T_flange)
        for seed in seeds:
            sol = self.robot.ikine_LM(T_flange, q0=seed, joint_limits=False, slimit=1)
            if sol.success:
                return sol.q
        return None


class UR3_Player(_Holder_Player):
    """UR3 from the Robotics Toolbox. Its ikine ignores .base, so IK is solved in the
    robot's own base frame on a second (unmoved) copy of the model."""
    VMAX = np.radians([180, 180, 180, 360, 360, 360])
    CARRY_GRIP_POINT = (0.25, 0.0, 0.40)       # UR3 is smaller - hold the cue closer in
    END = "tool0"

    def __init__(self, xyyaw):
        self.robot = rtb.models.UR3()
        self.kin = rtb.models.UR3()
        self.robot.q = np.radians([0, -90, 90, -90, -90, 0])
        super().__init__("UR3", xyyaw)

    def Flange(self, q=None):
        return self.robot.fkine(self.robot.q if q is None else q, end=self.END)

    def IK(self, T_tip, q_seed=None):
        T_local = self.robot.base.inv() * T_tip * self.tool.inv()
        q1 = np.arctan2(T_local.t[1], T_local.t[0])
        seeds = ([np.asarray(q_seed, dtype=float)] if q_seed is not None else [])
        seeds += [np.array([q1 + a, b, c, d, e, 0.0]) for a in (0, pi) for b, c in ((-1.2, 1.4), (-0.6, 0.9), (-1.8, 1.8))
                  for d in (-1.5,) for e in (-1.57, 1.57)]
        for seed in seeds:
            sol = self.kin.ikine_LM(T_local, q0=seed, end=self.END, joint_limits=True, slimit=1)   # slimit=1: no random restarts
            if sol.success:
                return sol.q
        return None


class Meca_Player:
    """Shuban's Meca500 (scaled) with his jump cue - see Meca_Jumper.py."""
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
