"""
Pool_Station.py - Shuban's service Meca500 at the head end of the track.

It does two jobs, both with Shuban's own code and numbers:
  * ASSEMBLY  (test_meca500_shaft_grip.py + DoosanAssemblyRightPosition.py)
      the Meca500 picks the shaft up off its rests (his Q_HOVER / Q_GRIP / Q_HOLD joint
      configs and his shaft_pose_from_ee) and holds it level; the Doosan brings the
      handle in along the same line, the shaft screws on (his HANDLE_PIN, N_TURNS,
      LINE_UP_HEIGHT).
  * CHALKING  (Test_Meca500_v1_Chalk.py)
      before the break and before every change of player the next robot holds its cue
      tip still at CHALK_TIP; the Meca500 picks up the chalk, pushes the cup onto the tip,
      rubs it and puts the chalk back (his Chalk_Pose, Chalk_Gripper, CHALK_TOOL, Solve_IK,
      RUB_* settings).

His tests have the robot base at the origin. Here the same motions are worked out in the
robot's own frame (exactly his numbers) and the robot is simply mounted at STATION_BASE,
on a post just outside the outer rim, so it never stands in a lane. Only change to his
setup: the cue lies along the rim (his cue aim turned by 90 deg, CUE_AIM below) because a
cue pointing straight at the robot would reach over the table.
"""
import numpy as np
from math import pi
from spatialmath import SE3
import spatialgeometry as sg
import roboticstoolbox as rtb

from meca500_robot import Meca500_Robot
from Pool_Parts import Part_Path
from Pool_Motion import Cue_Pose
import Test_Meca500_v1_Chalk as SC                      # Shuban: chalking
import test_meca500_shaft_grip as SG                     # Shuban: shaft grip
import DoosanAssemblyRightPosition as SA                 # Shuban: assembly numbers


# ------------------------------------WHERE THE STATION IS-----------------------------------------------------

STATION_BASE = SE3(-2.225, -0.30, 0.85)                 # on a post outside the head-end rim, facing the track
CUE_AIM = -90.0                                          # cue lies along the rim, tip towards -y
SHAFT_DIR = np.array([0.0, -1.0, 0.0])                   # his shaft_pose_from_ee: tip 0.5 m from the grip, along the rim

CHALK_TIP = (STATION_BASE * SE3(*SC.TIP_XYZ)).t          # where every robot holds its cue tip to be chalked
# resting pose between jobs: his Q_HOME turned to face away from where the cues are held
Q_REST = SG.Q_HOME.copy()
Q_REST[0] = -pi / 2
HOLD_EE = None                                           # filled in below (his Q_HOLD end-effector, world)


def Chalk_Present_Pose(roll):
    """Cue-tip pose a player holds still while it is chalked."""
    return Cue_Pose(CHALK_TIP, CUE_AIM, 0.0, roll)


class Station:
    def __init__(self):
        self.robot = Meca500_Robot()                     # what you see (mounted at the station)
        self.robot.base = STATION_BASE
        self.kin = Meca500_Robot()                       # same robot at the origin = his frame, for IK
        self.robot.q = Q_REST.copy()
        self.gripper = SC.Chalk_Gripper(SC.GRIP_OFFSET)
        self.chalk = sg.Mesh(str(Part_Path("Chalk")))
        radial = np.array([SC.CHALK_XY[0], SC.CHALK_XY[1], 0.0])
        radial /= np.linalg.norm(radial)
        centre = np.array([SC.CHALK_XY[0], SC.CHALK_XY[1], SC.STAND_HEIGHT + SC.CHALK_SIZE / 2])
        self.T_chalk_home = SC.Chalk_Pose(centre, [0, 0, 1], -radial)          # his frame
        self.T_chalk = self.T_chalk_home
        self.holding = None                                                  # None / "chalk" / "shaft"
        self.shaft_spin = 0.0
        # posts from the floor: robot post and his chalk stand
        h = STATION_BASE.t[2]
        sx, sy = (STATION_BASE * SE3(SC.CHALK_XY[0], SC.CHALK_XY[1], 0)).t[:2]
        self.parts = [sg.Cuboid(scale=[0.14, 0.14, h], color=[0.3, 0.3, 0.33, 1.0],
                                pose=SE3(STATION_BASE.t[0], STATION_BASE.t[1], h / 2)),
                      sg.Cuboid(scale=[0.03, 0.03, h + SC.STAND_HEIGHT], color=[0.35, 0.2, 0.1, 1.0],
                                pose=SE3(sx, sy, (h + SC.STAND_HEIGHT) / 2))]
        # shaft rests on the rim (under his Q_GRIP shaft pose)
        T_rest = self.Shaft_Pose(SG.Q_GRIP)
        for along in (0.15, 0.55):
            p = (T_rest * SE3.Tz(-along)).t
            hh = p[2] - 0.85 - 0.008
            self.parts.append(sg.Cuboid(scale=[0.04, 0.03, hh], color=[0.35, 0.2, 0.1, 1.0],
                                        pose=SE3(p[0], p[1], 0.85 + hh / 2)))
        self._chalk_qs = None

    # --- his helpers, in the world ---
    def Shaft_Pose(self, q):
        """(his shaft_pose_from_ee) shaft held at the end-effector, tip 0.5 m along the rim."""
        ee = (STATION_BASE * self.kin.fkine(q)).t
        return Cue_Pose(ee + 0.5 * SHAFT_DIR, CUE_AIM, 0.0)

    def Draw(self):
        T_flange = self.robot.fkine(self.robot.q)
        self.gripper.Place(T_flange * SC.FLANGE_OUT)
        if self.holding == "chalk":
            self.T_chalk = STATION_BASE.inv() * T_flange * SC.CHALK_TOOL
        self.chalk.T = (STATION_BASE * self.T_chalk).A

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

    # --- assembly (his shaft-grip test): joint paths ---
    def Pick_Shaft_Path(self):
        """Home -> hover -> grip (pause) as joint angles, then grip -> hold (shaft follows)."""
        to_hover = list(rtb.jtraj(Q_REST, SG.Q_HOVER, 80).q) + [SG.Q_HOVER] * 10
        to_grip = list(rtb.jtraj(SG.Q_HOVER, SG.Q_GRIP, 50).q) + [SG.Q_GRIP] * 15
        lift = list(rtb.jtraj(SG.Q_GRIP, SG.Q_HOLD, 100).q)
        return to_hover + to_grip, lift

    def Release_Up(self, rise=0.06):
        """After the screw-on: hand rises straight up off the cue (his Q_HOLD, then up)."""
        T_up = SE3(0, 0, rise) * self.kin.fkine(SG.Q_HOLD)
        sol = self.kin.ikine_LM(T_up, q0=SG.Q_HOLD, joint_limits=True)
        return sol.q if sol.success else SG.Q_HOLD.copy()

    # --- chalking (his Test_Meca500_v1_Chalk main(), steps 1-5) ---
    def Chalk_Path(self):
        """Joint angles (and what the gripper holds) for one chalking, worked out once with his
        IK in his frame - every robot holds its tip at the same spot, so it is the same every time."""
        if self._chalk_qs is not None:
            return self._chalk_qs
        robot = self.kin
        out = []                                                   # (q, opening, holding)
        q_now = [Q_REST.copy()]

        def joint(q_to, steps, hold):
            for q in rtb.jtraj(q_now[0], q_to, steps).q:
                out.append((q, None, hold))
            q_now[0] = q_to

        def line(T_from, T_to, steps, hold):
            q = q_now[0]
            for s in np.linspace(0, 1, steps)[1:]:
                q_new = SC.Solve_IK(robot, T_from.interp(T_to, s), SC.CHALK_TOOL, q, quiet=True)
                q = q if q_new is None else q_new
                out.append((q, None, hold))
            q_now[0] = q

        def grip(opening, hold):
            start = 1.0 - opening
            for s in np.linspace(0, 1, 20):
                out.append((q_now[0], start + (opening - start) * s, hold))

        T_tip = Cue_Pose(SC.TIP_XYZ, CUE_AIM, 0.0)
        cue_dir = T_tip.R[:, 2]
        T_pick = self.T_chalk_home
        T_lift = SE3(0, 0, SC.LIFT) * T_pick
        press_centre = np.array(SC.TIP_XYZ) + (SC.CHALK_SIZE / 2 - SC.CHALK_CUP_DEPTH) * cue_dir
        T_press = SC.Chalk_Pose(press_centre, -cue_dir, [0, 0, 1])
        T_press_pre = SE3(*(SC.APPROACH * cue_dir)) * T_press

        q_above = SC.Solve_IK(robot, T_lift, SC.CHALK_TOOL)
        joint(q_above, 100, False)                                 # 1. reach for the chalk
        line(T_lift, T_pick, 30, False)
        grip(0.0, False)
        line(T_pick, T_lift, 30, True)                             # 2. carry it to the tip
        joint(SC.Solve_IK(robot, T_press_pre, SC.CHALK_TOOL, q_now[0]), 100, True)
        line(T_press_pre, T_press, 30, True)                       # 3. cup onto the tip
        side = np.cross([0, 0, 1], cue_dir)                        # 4. rub left / right with a twist
        n = int(SC.RUB_STROKES * SC.RUB_PERIOD / 0.02)
        q = q_now[0]
        for i in range(n + 1):
            t = i * 0.02
            ramp = min(1.0, t / SC.RUB_PERIOD, (n * 0.02 - t) / SC.RUB_PERIOD)
            phase = np.sin(2 * pi * t / SC.RUB_PERIOD)
            T = (SE3(*(SC.RUB_AMPLITUDE * ramp * phase * side)) * SE3(*SC.TIP_XYZ)
                 * SE3.AngleAxis(np.radians(SC.RUB_TWIST_DEG) * ramp * phase, cue_dir)
                 * SE3(*(-np.array(SC.TIP_XYZ))) * T_press)
            q_new = SC.Solve_IK(robot, T, SC.CHALK_TOOL, q, quiet=True)
            q = q if q_new is None else q_new
            out.append((q, None, True))
        q_now[0] = q
        line(T_press, T_press_pre, 30, True)                       # 5. back off, chalk back on its stand
        joint(SC.Solve_IK(robot, T_lift, SC.CHALK_TOOL, q_above), 100, True)
        line(T_lift, T_pick, 30, True)
        grip(1.0, False)
        line(T_pick, T_lift, 30, False)
        joint(Q_REST.copy(), 100, False)
        self._chalk_qs = out
        return out


HOLD_EE = (STATION_BASE * Meca500_Robot().fkine(SG.Q_HOLD)).t
ASSEMBLED_TIP = HOLD_EE + 0.5 * SHAFT_DIR                # shaft tip while the Meca holds it level = cue tip once assembled
HANDLE_PIN, N_TURNS, LINE_UP = SA.HANDLE_PIN, SA.N_TURNS, SA.LINE_UP_HEIGHT


def Assembly_Pose(gap, roll):
    """Doosan's (virtual) cue-tip pose with the handle `gap` m short of fully screwed on."""
    return Cue_Pose(ASSEMBLED_TIP - gap * SHAFT_DIR, CUE_AIM, 0.0, roll)


# ------------------------------------WHERE EACH PLAYER STANDS TO BE CHALKED-----------------------------------
# (turtlebot pose, cue roll, joint angles) - found by searching each lane for a spot where the
# arm can hold its tip at CHALK_TIP (and, for the Doosan, the three assembly poses).

PRESENT = {
    "D": ((-1.72, 0.15, -90.0), 180, [-2.5928, 0.7301, 1.5123, -0.0, 0.8992, 0.5488]),
    "U": ((-1.72, 0.15, 90.0), 45, [0.1789, 0.3257, 0.0766, -1.797, 0.8012, 1.3205]),
    # the Meca500's own IK cannot settle this pose (a joint sits on its limit), so it was found
    # with a damped-least-squares search on tip position + cue direction; roll None = use these joints
    "M": ((-1.44, 0.0, 90.0), None, [1.5708, 1.5708, -1.206, 0.0, 1.796, -1.5708]),
}
ASM_PRESENT = ((-1.72, -0.3, -90.0), 180, [[-2.5646, 0.4702, 1.583, 0.0, 1.0883, 0.577], [-2.598, 0.5319, 1.4844, 0.0, 1.1253, 0.5436], [-2.7136, 0.8677, 0.8949, 0.0, 1.3791, 0.428]])
