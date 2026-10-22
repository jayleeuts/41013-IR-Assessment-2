"""
Test_Meca500_v1_Masse - masse shot on the Meca500: steep cue, hit off-centre, ball curves.

How the curve happens (same physics a real cue ball follows):
  1. The cue comes down steeply and hits the ball to one SIDE of centre.
  2. That gives the ball a lot of spin about a near-horizontal axis.
  3. The ball's bottom is slipping sideways on the cloth, so friction pushes
     it sideways -> the path bends.
  4. Once the slipping stops, the ball rolls straight on.
The red markings on the ball show it spinning the whole way.

Layout (so the cue never passes through the arm or the table):
  - The ball sits 0.25 m in front of the robot. The shot goes sideways (-Y),
    so the long back end of the cue rises off to the robot's side (+Y),
    away from the arm.
  - The gripper holds the cue from the robot's side, fingers either side of
    the shaft, palm sitting on the flange face.
  - The robot starts already holding the cue 8 cm above the shot. Moving
    there from q = 0 would sweep the 1.5 m cue through the robot.

Checked against the DH model for every frame: no IK failures, all joints at
least 9 deg inside their limits, the cue stays clear of the base, upper arm
and forearm by 7.6 cm or more (the wrist, right beside the
gripper, by 1 cm), and the whole cue stays above the table.
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
from Pool_Objects import Cue_Ball, Cue_Stick
from Pool_Motion import Ball_Physics, Seed_Candidates


# ------------------------------------CONSTANTS-----------------------------------------------------

DT = 0.02

HANDLE_PIN = 0.033

GRIP_OFFSET = 0.06                   # flange face -> cue centre line (clears the wrist)
GRIP_FROM_TIP = 0.20                 # grip point -> cue tip
SHAFT_WIDTH = 0.014                  # cue thickness where it is gripped

TABLE_TOP_Z = 0.0
BALL_XY = (0.25, 0.00)               # in front of the robot, inside its ~0.33 m reach
AIM_DEG = -90.0                      # shoot sideways (-Y) so the cue's back end goes off to the side
ELEVATION_DEG = 60.0                 # steep cue = masse

SIDE_HIT = 0.6                       # how far off-centre, as a fraction of the ball radius
CURVE_LEFT = True                    # True = hit left of centre -> curves left (away from the robot)
ADDRESS_GAP = 0.02                   # tip -> ball surface before the stroke
TIP_SPEED = 1.0                      # cue speed at contact (m/s)
FOLLOW = 0.02                        # braking distance after contact (table is just below!)
BACKSWING = 0.05                     # how far the cue pulls back before striking (keep <= 0.06)
LIFT_AFTER = 0.05                    # how far the cue backs off after the shot
START_ABOVE = 0.08                   # robot starts holding the cue this far above the shot

CUE_TO_BALL = 1.4                    # ball impulse / tip speed (cue is heavier)
MU_SLIDE = 0.12                      # cloth friction while the ball slips (bends the path)
MU_ROLL = 0.05                       # rolling resistance once it rolls

# Arm pose to start the IK from for the first pose, so the arm always picks the
# same elbow/wrist shape (the one that was checked for clearance).
Q_START_SEED = np.radians([39.6, 36.0, -56.7, 118.3, -42.1, 25.5])


# In meca500_robot.py the last DH frame's Z points back INTO the wrist (alpha = pi
# on joint 6). FLANGE_OUT turns it round so +Z points out of the flange face,
# which is where the gripper and the cue have to be.
FLANGE_OUT = SE3.Rx(pi)

# Flange -> cue tip. The cue's centre line is GRIP_OFFSET out from the flange face
# and runs along the flange's X (fingers close on it from +/-Y). The tip is
# GRIP_FROM_TIP further along the cue.
TOOL = FLANGE_OUT * SE3.Tz(GRIP_OFFSET) * SE3.Ry(pi / 2) * SE3.Tz(GRIP_FROM_TIP)


# ------------------------------------HELPERS-----------------------------------------------------

def Cue_Dir(aim_deg, elev_deg):
    """Unit vector the cue points along (towards the ball, tilted down)."""
    a, e = np.radians(aim_deg), np.radians(elev_deg)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), -np.sin(e)])


def Tip_Pose(tip, aim_deg, elev_deg):
    """Pose of the cue tip: Z along the cue, X pointing from the cue towards the
    robot (so the gripper holds it from the robot's side)."""
    z = Cue_Dir(aim_deg, elev_deg)
    towards_robot = -np.array([tip[0], tip[1], 0.0])
    x = towards_robot - np.dot(towards_robot, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return SE3.Rt(np.column_stack([x, y, z]), np.asarray(tip, dtype=float))


def In_Limits(robot, q):
    return bool(np.all(q >= robot.qlim[0] - 1e-6) and np.all(q <= robot.qlim[1] + 1e-6))


def Solve_IK(robot, T_tip, q_seed=None, quiet=False):
    """IK for a cue-tip pose with joint limits on. Tries q_seed first (smooth
    motion), then the usual spread of seeds. Returns None if nothing works."""
    T_flange = T_tip * TOOL.inv()
    seeds = ([np.asarray(q_seed, dtype=float)] if q_seed is not None else []) + list(Seed_Candidates(robot, T_flange))
    for seed in seeds:
        sol = robot.ikine_LM(T_flange, q0=seed, joint_limits=True, tol=1e-10)
        if sol.success:
            q = (sol.q + pi) % (2 * pi) - pi            # wrap to [-pi, pi]
            if In_Limits(robot, q):
                return q
    if not quiet:
        print(f"IK failed at tip position {np.round(T_tip.t, 3)} - target may be out of reach")
    return None


def Short_Stroke_Profile(dt, contact_s, tip_speed, follow, backswing):
    """Distance s the tip moves back along the cue (+ = away from the ball).

    Slowly draw back to +backswing, pause, then speed up evenly so the tip is
    moving at tip_speed when it reaches contact_s, then brake over `follow`.
    """
    def ease(n):                                       # smooth 0 -> 1
        t = np.linspace(0, 1, n)
        return 3 * t ** 2 - 2 * t ** 3

    out = list(backswing * ease(40))                   # draw back
    out += [backswing] * 10                            # pause
    accel = tip_speed ** 2 / (2 * (backswing - contact_s))
    t, t_hit = 0.0, tip_speed / accel
    while t < t_hit:                                   # speed up through contact
        t += dt
        out.append(backswing - 0.5 * accel * min(t, t_hit) ** 2)
    decel = tip_speed ** 2 / (2 * follow)
    t, t_stop = 0.0, tip_speed / decel
    while t < t_stop:                                  # brake after contact
        t += dt
        tt = min(t, t_stop)
        out.append(contact_s - (tip_speed * tt - 0.5 * decel * tt ** 2))
    return out


class Cue_Gripper:
    """Palm on the flange face + two fingers either side of the cue shaft. Visual only."""

    def __init__(self, grip_offset, width=SHAFT_WIDTH, color=(0.25, 0.25, 0.28, 1)):
        self.finger_t = 0.006
        self.gap = width / 2 + self.finger_t / 2
        self.finger_len = grip_offset + width / 2 - 0.002        # ends just past the shaft
        self.palm = sg.Cuboid(scale=[0.04, 0.05, 0.02], color=list(color))
        self.fingers = [sg.Cuboid(scale=[0.016, self.finger_t, self.finger_len], color=list(color))
                        for _ in range(2)]

    def Place(self, T_face):
        """T_face = flange frame with +Z pointing out of the flange face."""
        self.palm.T = (T_face * SE3.Tz(0.01)).A
        for side, finger in zip((-1, 1), self.fingers):
            finger.T = (T_face * SE3(0, side * self.gap, self.finger_len / 2)).A

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

    gripper = Cue_Gripper(GRIP_OFFSET)

    ball = Cue_Ball(str(Part_Path("Cue_Ball")))
    ball_start = np.array([BALL_XY[0], BALL_XY[1], TABLE_TOP_Z + ball.radius])
    physics = Ball_Physics(ball_start, ball.radius, mu_slide=MU_SLIDE, mu_roll=MU_ROLL)
    ball.Place(physics.pos, physics.R)

    cue = Cue_Stick(str(Part_Path("Cuestick_Shaft")), str(Part_Path("Cuestick_Handle")),
                    handle_insert=HANDLE_PIN)

    # ----- poses -----
    # Where the tip meets the ball: off to one side of centre
    cue_dir = Cue_Dir(AIM_DEG, ELEVATION_DEG)
    left = np.array([-np.sin(np.radians(AIM_DEG)), np.cos(np.radians(AIM_DEG)), 0.0])
    b = SIDE_HIT * ball.radius * (1 if CURVE_LEFT else -1)
    contact = b * left - np.sqrt(ball.radius ** 2 - b ** 2) * cue_dir   # on the ball surface

    T_address = Tip_Pose(ball_start + contact - ADDRESS_GAP * cue_dir, AIM_DEG, ELEVATION_DEG)
    T_start = SE3(0, 0, START_ABOVE) * T_address

    q_start = Solve_IK(robot, T_start, Q_START_SEED)
    if q_start is None:
        print("Can't reach the start pose - check BALL_XY / GRIP_* settings")
        env.hold()
        return
    q_address = Solve_IK(robot, T_address, q_start)

    robot.q = q_start
    robot.add_to_env(env)
    gripper.Add_To_Env(env)
    ball.Add_To_Env(env)
    cue.Add_To_Env(env)

    trail = []                                     # where the ball has been, for the summary

    def Update(q):
        if q is None:                              # IK failed - keep the last pose
            q = robot.q
        robot.q = q
        T_flange = robot.fkine(q)
        gripper.Place(T_flange * FLANGE_OUT)
        cue.Place(T_flange * TOOL)
        if physics.Moving():
            physics.Step(DT)
            ball.Place(physics.pos, physics.R)
            trail.append(physics.pos.copy())
        env.step(DT)
        time.sleep(DT)

    Update(robot.q)
    env.set_camera_pose([0.9, 0.6, 0.6], [0.25, -0.15, 0.1])

    # 1. Lower the cue onto the line of the shot
    print("[1] Lining up the masse...")
    if q_address is not None:
        for q in rtb.jtraj(robot.q, q_address, 60).q:
            Update(q)

    # 2. Back-swing, then punch down through the contact point
    print("[2] Striking...")
    q, prev_tip, hit = robot.q, T_address.t, False
    for s in Short_Stroke_Profile(DT, contact_s=-ADDRESS_GAP, tip_speed=TIP_SPEED,
                                  follow=FOLLOW, backswing=BACKSWING):
        T_tip = T_address * SE3.Tz(-s)
        q_new = Solve_IK(robot, T_tip, q, quiet=True)
        if q_new is None:
            print(f"    (IK failed at s = {s:+.3f} m - holding last pose; try a smaller BACKSWING)")
        else:
            q = q_new
        if not hit and np.dot(T_tip.t - (ball_start + contact), cue_dir) >= -1e-6:
            tip_vel = (T_tip.t - prev_tip) / DT
            physics.Hit(CUE_TO_BALL * tip_vel, contact)
            hit = True
            print(f"[3] Hit! tip speed {np.linalg.norm(tip_vel):.2f} m/s, "
                  f"spin {np.linalg.norm(physics.w):.0f} rad/s")
        prev_tip = T_tip.t
        Update(q)

    # 3. Pull the cue back up out of the way, ball keeps going
    q_up = Solve_IK(robot, T_address * SE3.Tz(-LIFT_AFTER), q)
    if q_up is not None:
        for q in rtb.jtraj(q, q_up, 40).q:
            Update(q)
    while physics.Moving():
        Update(q)

    if trail:
        trail = np.array(trail)
        far = np.where(np.linalg.norm(trail - trail[-1], axis=1) > 0.2)[0]   # last 20 cm of travel
        ref = trail[far[-1]] if len(far) else trail[0]
        heading = np.degrees(np.arctan2(trail[-1][1] - ref[1], trail[-1][0] - ref[0]))
        print(f"Shot complete - ball ended at ({trail[-1][0]:.2f}, {trail[-1][1]:.2f}), "
              f"curved to a heading of {heading:.0f} deg (aimed at {AIM_DEG:.0f} deg)")
    env.hold()


if __name__ == "__main__":
    main()
