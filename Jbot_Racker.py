"""
Jbot_Racker.py - the racking robot (Jbot, from jbot.py) on its own turtlebot, racking the balls.

Jbot's model (DH parameters, meshes) stays in jbot.py. This file
only adds what is needed to USE it in the pool workcell:

  * Size      : Jbot is a myCobot-280-style arm. At 3.25x it reaches ~0.86 m,
                like an industrial cobot (Doosan M0609 = 0.9 m) - enough to reach
                every pocket AND the rack spot from the track. Joints and meshes
                are scaled together so they always line up.
  * Mounting  : base sits on top of its own turtlebot and moves with it.
  * Tool      : a vacuum cup on a 0.25 m rod, pointing down - it can reach into
                a pocket (55 mm hole) and lift a ball out.
  * Tray      : a 9-ball diamond rack frame that sits on the felt near the foot
                cushion. Jbot fills it, slides it to the foot spot, then lifts
                it off so the balls stay racked.
  * IK        : Jbot's 4th DH link is FIXED (it never moves). The toolbox IK
                treats it as a joint, so Jbot_IK below solves with it locked.
"""
import numpy as np
from math import pi
from spatialmath import SE3
import spatialgeometry as sg

from jbot import MyRobot
from ir_support_extra_robots.robots import Turtlebot3Waffle
from Pool_Scene import TURTLE_HEIGHT, Turtle_Pose


# ------------------------------------SETTINGS-----------------------------------------------------

JBOT_SIZE = 3.0                     # scale of the myCobot-style model (joints AND meshes)
FIXED_JOINT = 3                      # index of Jbot's fixed DH link (never moves)

TOOL_LENGTH = 0.32                   # flange -> vacuum cup face
ROD_RADIUS = 0.012
CUP_RADIUS = 0.022

RACK_BAR = 0.012                     # rack frame bar thickness
RACK_HEIGHT = 0.025                  # rack frame height (sits on the felt)
RACK_GAP = 0.002                     # gap between a ball and the frame
HANDLE_SIZE = 0.04                   # pad on the back of the rack that the vacuum cup grips


def Rebuild_ETS(robot):
    """After changing DH lengths, rebuild each link's ETS (the toolbox builds it once when the link
    is made; jacob0 / ikine use it, fkine and the meshes use the new lengths)."""
    for i, link in enumerate(robot.links):
        ets = link._to_ets(link._sigma, link._theta, link.d, link.alpha, link.a, link.offset, link.isflip, link._mdh)
        for et in ets:
            if et.isjoint:
                et.qlim, et.jindex = link.qlim, i
        link._ets = ets


# ------------------------------------IK (fixed link locked)-----------------------------------------------------

def _Pose_Error(T, Tg):
    return np.r_[Tg[:3, 3] - T[:3, 3],
                 0.5 * (np.cross(T[:3, 0], Tg[:3, 0]) + np.cross(T[:3, 1], Tg[:3, 1])
                        + np.cross(T[:3, 2], Tg[:3, 2]))]


def Jbot_IK(robot, T_goal, q_seed=None, mask=(1, 1, 1, 1, 1, 0), iters=200, tries=20, tol=1e-4):
    """Damped least-squares IK with the fixed link held at 0 and joint limits kept.

    mask (1,1,1,1,1,0) = match position and pointing direction, any spin about
    the world vertical (the vacuum cup is round, so spin doesn't matter).
    Tries q_seed first so motions stay smooth; returns None if nothing works.
    """
    rng = np.random.default_rng(0)
    lo, hi = robot.qlim[0].copy(), robot.qlim[1].copy()
    W = np.diag(np.array(mask, dtype=float))
    Tg = T_goal.A if hasattr(T_goal, "A") else np.asarray(T_goal)
    seeds = [] if q_seed is None else [np.asarray(q_seed, dtype=float)]
    for _ in range(tries):
        s = rng.uniform(lo, hi)
        s[1], s[2] = rng.uniform(-0.3, 1.0), rng.uniform(-2.0, -0.2)
        seeds.append(s)
    for q in seeds:
        q = q.copy()
        q[FIXED_JOINT] = 0.0
        lam = 1e-2
        for _ in range(iters):
            e = W @ _Pose_Error(robot.fkine(q).A, Tg)
            if np.linalg.norm(e) < tol:
                return q
            J = W @ robot.jacob0(q)
            J[:, FIXED_JOINT] = 0.0
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            q_new = np.clip(q + dq, lo, hi)
            q_new[FIXED_JOINT] = 0.0
            if np.linalg.norm(W @ _Pose_Error(robot.fkine(q_new).A, Tg)) < np.linalg.norm(e):
                q, lam = q_new, max(lam / 3, 1e-5)
            else:
                lam *= 4
                if lam > 1e3:
                    break
    return None


def Tool_Down(x, y, z):
    """Vacuum cup face at (x, y, z), pointing straight down."""
    return SE3(x, y, z) * SE3.Rx(pi)


# ------------------------------------THE RACK (TRAY)-----------------------------------------------------

def Rack_Slots(ball_radius):
    """Ball centres in the rack frame (apex ball at the origin, rows going +x): 1-2-3-2-1."""
    h = np.sqrt(3) * ball_radius
    slots = []
    for row, n in enumerate([1, 2, 3, 2, 1]):
        for j in range(n):
            slots.append(np.array([row * h, (j - (n - 1) / 2) * 2 * ball_radius, 0.0]))
    return slots


class Ball_Rack:
    """Diamond-shaped 9-ball rack frame (4 bars) with a pad on the back to grip.

    Rack frame: origin at the APEX ball's centre on the felt, +x towards the back.
    """

    def __init__(self, ball_radius, felt_z, color=(0.55, 0.35, 0.15, 1)):
        self.r, self.felt_z = ball_radius, felt_z
        h = np.sqrt(3) * ball_radius
        # ball-centre diamond, then pushed out so the bars just clear the balls
        A, B, C, D = (np.array(p) for p in [(0, 0), (2 * h, -2 * ball_radius), (4 * h, 0), (2 * h, 2 * ball_radius)])
        off = ball_radius + RACK_GAP + RACK_BAR / 2
        corners = self._Offset([A, B, C, D], off)
        self.bars, self._bar_poses = [], []
        for i in range(4):
            p, q = corners[i], corners[(i + 1) % 4]
            length = np.linalg.norm(q - p) + RACK_BAR
            mid, ang = (p + q) / 2, np.arctan2(q[1] - p[1], q[0] - p[0])
            self.bars.append(sg.Cuboid(scale=[length, RACK_BAR, RACK_HEIGHT], color=list(color)))
            self._bar_poses.append(SE3(mid[0], mid[1], RACK_HEIGHT / 2 - ball_radius) * SE3.Rz(ang))
        back = corners[2]
        self.handle_offset = np.array([back[0] + HANDLE_SIZE / 2, 0.0, RACK_HEIGHT + 0.01 - ball_radius])
        self.handle = sg.Cuboid(scale=[HANDLE_SIZE, HANDLE_SIZE, 0.02], color=[0.2, 0.2, 0.22, 1.0])
        self.length = corners[2][0] - corners[0][0] + HANDLE_SIZE
        self.T = SE3()

    @staticmethod
    def _Offset(poly, d):
        """Push each edge of an anticlockwise convex polygon outward by d; return new corners."""
        lines = []
        n = len(poly)
        for i in range(n):
            p, q = poly[i], poly[(i + 1) % n]
            t = (q - p) / np.linalg.norm(q - p)
            normal = np.array([t[1], -t[0]])                        # outward for anticlockwise order
            lines.append((p + d * normal, t))
        out = []
        for i in range(n):
            (p1, t1), (p2, t2) = lines[i - 1], lines[i]
            s = np.linalg.solve(np.column_stack([t1, -t2]), p2 - p1)
            out.append(p1 + s[0] * t1)
        return out

    def Handle_Top(self):
        """World position of the top of the grip pad."""
        return (self.T * SE3(*self.handle_offset) * SE3.Tz(0.01)).t

    def Place(self, T):
        """T = rack frame in the world (apex ball centre, +x to the back)."""
        self.T = T
        for bar, pose in zip(self.bars, self._bar_poses):
            bar.T = (T * pose).A
        self.handle.T = (T * SE3(*self.handle_offset)).A

    def Add_To_Env(self, env):
        for bar in self.bars:
            env.add(bar)
        env.add(self.handle)


# ------------------------------------JBOT ON ITS TURTLEBOT-----------------------------------------------------

class Jbot_Mobile:
    """Jbot, scaled to JBOT_SIZE, sitting on a turtlebot, with a vacuum-cup tool."""

    def __init__(self, xyyaw):
        self.robot = MyRobot()
        self.robot.set_size(JBOT_SIZE)                  # joints
        Rebuild_ETS(self.robot)                         # so IK / Jacobian use the scaled lengths too
        for mesh in self.robot.links_3d:
            mesh.scale = [JBOT_SIZE] * 3                # meshes (jbot.py draws them at 2x)
        self.turtle = Turtlebot3Waffle()
        self.xyyaw = tuple(xyyaw)
        self.tool = SE3.Tz(TOOL_LENGTH)
        self.rod = sg.Cylinder(radius=ROD_RADIUS, length=TOOL_LENGTH, color=[0.75, 0.75, 0.78, 1.0])
        self.cup = sg.Cylinder(radius=CUP_RADIUS, length=0.015, color=[0.1, 0.1, 0.1, 1.0])
        self.Set_Base()

    def Set_Base(self):
        self.turtle.base = Turtle_Pose(*self.xyyaw)
        self.robot.base = self.turtle.base * SE3.Tz(TURTLE_HEIGHT)

    def Tool_Pose(self, q=None):
        """World pose of the vacuum cup face."""
        return self.robot.fkine(self.robot.q if q is None else q) * self.tool

    def Draw_Tool(self):
        T_flange = self.robot.fkine(self.robot.q)
        self.rod.T = (T_flange * SE3.Tz(TOOL_LENGTH / 2)).A
        self.cup.T = (T_flange * SE3.Tz(TOOL_LENGTH - 0.0075)).A

    def IK(self, T_cup, q_seed=None):
        return Jbot_IK(self.robot, T_cup * self.tool.inv(), q_seed)

    def Add_To_Env(self, env):
        self.turtle.add_to_env(env)
        self.robot.add_to_env(env)
        env.add(self.rod)
        env.add(self.cup)
        self.Draw_Tool()
