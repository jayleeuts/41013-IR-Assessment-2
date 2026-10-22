"""
Test_Meca500_v1_Jump_Over - jump shot that clears a second cue ball sitting in the way.

Same robot, cue and physics as Test_Meca500_v1_Jump, but:
  - A second (blocker) ball sits on the shot line.
  - The cue is steeper and faster, so the ball jumps higher than a whole ball
    (a ball is 57 mm tall, so the bottom of the jumping ball has to get above that).
  - The blocker is placed automatically where the jump is highest, worked out
    from the cue speed and angle, so it always sits under the top of the jump.
  - If the jump ever falls short, the two balls collide like real balls
    (Balls_Collide from Pool_Motion) and the script says so.

How the jump works:
  1. The cue strikes DOWN through the middle of the ball.
  2. The downward part of the hit pinches the ball into the cloth, which
     throws it back up. The forward part sends it along the shot line.
  3. It flies over the blocker, lands, bounces a couple of times, then the
     normal Ball_Physics takes over: slide, roll, stop.

Layout is the same as the masse and jump scripts: ball 0.25 m in front of the
robot, shot goes sideways (-Y) so the back of the cue rises off to the side.

jumpcue.stl must sit in the same folder as this script (the project root).
"""
import time
import struct
from pathlib import Path

import swift
import numpy as np
from math import pi
from spatialmath import SE3
import spatialgeometry as sg
import roboticstoolbox as rtb

from meca500_robot import Meca500_Robot
from Pool_Parts import Part_Path
from Pool_Objects import Cue_Ball, _measure_rod, _part_offset, _guess_scale
from Pool_Motion import Ball_Physics, Balls_Collide, Seed_Candidates, _Rotation_From_Vector


# ------------------------------------CONSTANTS-----------------------------------------------------

DT = 0.02
PHYSICS_SUBSTEPS = 10                # ball physics steps per frame (bounces need small steps)

JUMP_CUE_FILE = Path(__file__).resolve().parent / "jumpcue.stl"

GRIP_OFFSET = 0.06                   # flange face -> cue centre line (clears the wrist)
GRIP_FROM_TIP = 0.20                 # grip point -> cue tip
SHAFT_WIDTH = 0.012                  # jump cue thickness where it is gripped

TABLE_TOP_Z = 0.0
SHOW_CLOTH = True                    # thin green sheet so the bounces are easy to see
BALL_XY = (0.25, 0.00)               # in front of the robot, inside its ~0.33 m reach
AIM_DEG = -90.0                      # shoot sideways (-Y) so the cue's back end goes off to the side
ELEVATION_DEG = 55.0                 # steeper than the plain jump (50) = more of the hit goes into the jump

ADDRESS_GAP = 0.02                   # tip -> ball surface before the stroke
TIP_SPEED = 2.6                      # cue speed at contact (m/s) - more speed = higher jump
FOLLOW = 0.02                        # braking distance after contact (the table is just below)
BACKSWING = 0.05                     # how far the cue pulls back before striking (keep <= 0.06)
LIFT_AFTER = 0.05                    # how far the cue backs off after the shot
START_ABOVE = 0.08                   # robot starts holding the cue this far above the shot

CUE_TO_BALL = 1.4                    # ball impulse / tip speed (cue is heavier than the ball)
TABLE_BOUNCE = 0.5                   # bounce height kept each landing: speed up / speed down
BOUNCE_FRICTION = 0.2                # cloth grip on each landing (slows it, starts it rolling)
BOUNCE_STOP = 0.15                   # stop bouncing below this upward speed (m/s)
MU_SLIDE = 0.2                       # cloth friction while the ball slips
MU_ROLL = 0.08                       # rolling resistance once it rolls

# Arm pose to start the IK from for the first pose, so the arm always picks the
# same elbow/wrist shape (the one that was checked for clearance).
BLOCKER_GAP = None                   # ball centre -> blocker centre (m); None = put it under the top of the jump
BLOCKER_COLOR = [0.85, 0.1, 0.1, 1]  # so you can tell the two balls apart

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


def Jump_Peak(cue_dir):
    """Predicted (distance along the table, height of the ball's bottom) at the
    top of the jump, from the cue speed and angle. Same maths as Jump_Ball_Physics."""
    J = CUE_TO_BALL * TIP_SPEED * cue_dir
    vz = TABLE_BOUNCE * max(0.0, -J[2])
    t_top = vz / Ball_Physics.G
    return np.linalg.norm(J[:2]) * t_top, vz ** 2 / (2 * Ball_Physics.G)


def Blocker_Position(ball_start, cue_dir):
    """Where the second ball sits: on the shot line, under the top of the jump."""
    along = np.array([cue_dir[0], cue_dir[1], 0.0])
    along /= np.linalg.norm(along)
    gap = BLOCKER_GAP if BLOCKER_GAP is not None else Jump_Peak(cue_dir)[0]
    return ball_start + gap * along


def Plan_Shot(ball_radius):
    """Ball centre, contact point (relative to the centre), cue direction,
    address pose and start pose. Kept separate so it can be checked headlessly."""
    ball_start = np.array([BALL_XY[0], BALL_XY[1], TABLE_TOP_Z + ball_radius])
    cue_dir = Cue_Dir(AIM_DEG, ELEVATION_DEG)
    contact = -ball_radius * cue_dir                   # hit straight through the centre
    T_address = Tip_Pose(ball_start + contact - ADDRESS_GAP * cue_dir, AIM_DEG, ELEVATION_DEG)
    T_start = SE3(0, 0, START_ABOVE) * T_address
    return ball_start, contact, cue_dir, T_address, T_start


# ------------------------------------BALL PHYSICS-----------------------------------------------------

class Jump_Ball_Physics(Ball_Physics):
    """Ball_Physics from Pool_Motion, plus flying and bouncing.

    On the table it behaves exactly like Ball_Physics (slide, roll, stop).
    In the air it follows gravity; each landing keeps TABLE_BOUNCE of its
    upward speed and the cloth's grip trades some forward speed for roll.
    """

    def __init__(self, centre, radius, table_bounce=TABLE_BOUNCE,
                 bounce_friction=BOUNCE_FRICTION, bounce_stop=BOUNCE_STOP, **kw):
        super().__init__(centre, radius, **kw)
        self.floor_z = float(centre[2])                # ball centre height when resting on the cloth
        self.table_bounce = table_bounce
        self.bounce_friction = bounce_friction
        self.bounce_stop = bounce_stop
        self.vz = 0.0
        self.airborne = False
        self.bounces = 0                               # times it bounced back up after landing
        self.max_height = 0.0                          # highest the BOTTOM of the ball got off the cloth

    def Hit(self, impulse_per_mass, contact_rel):
        """Cue strike. The forward part moves the ball along the table; the
        downward part squashes it into the cloth, which throws it back up."""
        J = np.asarray(impulse_per_mass, dtype=float)
        super().Hit(J, contact_rel)                    # forward speed + any spin
        if J[2] < 0:                                   # pushed down into the cloth
            self.vz = -self.table_bounce * J[2]
            self.airborne = self.vz > self.bounce_stop
            if not self.airborne:
                self.vz = 0.0

    def Moving(self):
        return self.airborne or super().Moving()

    def Step(self, dt):
        if not self.airborne:
            super().Step(dt)                           # normal slide / roll on the cloth
            return
        # in the air: no friction, spin carries on, gravity pulls it down
        self.vz -= self.G * dt
        self.pos[:2] += self.v[:2] * dt
        self.pos[2] += self.vz * dt
        self.R = _Rotation_From_Vector(self.w * dt) @ self.R
        self.max_height = max(self.max_height, self.pos[2] - self.floor_z)
        if self.pos[2] <= self.floor_z and self.vz < 0:
            self._Land()

    def _Land(self):
        self.pos[2] = self.floor_z
        vz_in = -self.vz
        # cloth grip while the ball is squashed: opposes the slip of the
        # bottom of the ball, but never more than it takes to make it roll
        slip = self.Contact_Slip()
        slip[2] = 0.0
        slip_speed = np.linalg.norm(slip)
        if slip_speed > 1e-9:
            u = slip / slip_speed
            dv = min(self.bounce_friction * (1 + self.table_bounce) * vz_in, 2.0 / 7.0 * slip_speed)
            self.v[:2] -= dv * u[:2]
            self.w += 5.0 / (2.0 * self.r) * dv * np.cross([0.0, 0.0, 1.0], u)
        self.vz = self.table_bounce * vz_in
        if self.vz < self.bounce_stop:                 # too small to see - it's on the cloth now
            self.vz = 0.0
            self.airborne = False
        else:
            self.bounces += 1


# ------------------------------------JUMP CUE-----------------------------------------------------

def Stl_Vertices(path):
    """All vertex positions of a .stl file (binary or text), in the file's units."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):                   # binary STL
            tri = np.frombuffer(data[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")]))
            return tri["v"].reshape(-1, 3).astype(float)
    pts = [line.split()[1:4] for line in data.decode(errors="ignore").splitlines()
           if line.strip().startswith("vertex")]
    return np.array(pts, dtype=float)


class Jump_Cue:
    """The one-piece jump cue. Same cue frame as Cue_Stick:
    origin = cue tip, +Z = direction the cue points (towards the ball)."""

    def __init__(self, stl_path, flip=False, color=(0.82, 0.62, 0.36, 1)):
        v = Stl_Vertices(stl_path)
        scale = _guess_scale(float(np.ptp(v, axis=0).max()), 1.0, "jump cue")
        v = v * scale
        self.length, tip_pt, direction = _measure_rod(v, flip)
        self._offset = _part_offset(tip_pt, direction, 0.0)
        self.mesh = sg.Mesh(str(stl_path), scale=[scale] * 3, color=list(color))
        print(f"[jump cue] {self.length:.3f} m")

    def Place(self, T_tip):
        self.mesh.T = (T_tip * self._offset).A

    def Add_To_Env(self, env):
        env.add(self.mesh)


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
    ball_start, contact, cue_dir, T_address, T_start = Plan_Shot(ball.radius)
    physics = Jump_Ball_Physics(ball_start, ball.radius, mu_slide=MU_SLIDE, mu_roll=MU_ROLL)
    ball.Place(physics.pos, physics.R)

    cue = Jump_Cue(JUMP_CUE_FILE)

    # the ball in the way
    blocker = Cue_Ball(str(Part_Path("Cue_Ball")), color=BLOCKER_COLOR)
    blocker_start = Blocker_Position(ball_start, cue_dir)
    blocker_physics = Ball_Physics(blocker_start, blocker.radius, mu_slide=MU_SLIDE, mu_roll=MU_ROLL)
    blocker.Place(blocker_physics.pos, blocker_physics.R)
    peak_gap, peak_height = Jump_Peak(cue_dir)
    print(f"[blocker] {np.linalg.norm(blocker_start - ball_start):.2f} m down the line; "
          f"jump should top out {peak_height * 100:.1f} cm off the cloth "
          f"(needs > {2 * ball.radius * 100:.1f} cm to clear)")
    if peak_height < 2 * ball.radius:
        print("    Warning: not enough jump to clear the blocker - raise TIP_SPEED or ELEVATION_DEG")

    q_start = Solve_IK(robot, T_start, Q_START_SEED)
    if q_start is None:
        print("Can't reach the start pose - check BALL_XY / ELEVATION_DEG / GRIP_* settings")
        env.hold()
        return
    q_address = Solve_IK(robot, T_address, q_start)

    if SHOW_CLOTH:
        cloth = sg.Cuboid(scale=[0.6, 3.0, 0.004], color=[0.05, 0.4, 0.15, 1],
                          pose=SE3(BALL_XY[0], BALL_XY[1] - 1.2, TABLE_TOP_Z - 0.002))
        env.add(cloth)

    robot.q = q_start
    robot.add_to_env(env)
    gripper.Add_To_Env(env)
    ball.Add_To_Env(env)
    blocker.Add_To_Env(env)
    cue.Add_To_Env(env)

    trail = []                                     # where the ball has been, for the summary
    closest = [np.inf]                             # closest the two ball centres got
    collided = [False]

    def Step_Balls(dt):
        physics.Step(dt)
        if blocker_physics.Moving():
            blocker_physics.Step(dt)
        d = np.linalg.norm(physics.pos - blocker_physics.pos)
        closest[0] = min(closest[0], d)
        if d < physics.r + blocker_physics.r:      # touching (in 3D) - jump fell short
            if Balls_Collide(physics, blocker_physics) and not collided[0]:
                collided[0] = True
                print("    The jump hit the blocker!")

    def Update(q):
        if q is None:                              # IK failed - keep the last pose
            q = robot.q
        robot.q = q
        T_flange = robot.fkine(q)
        gripper.Place(T_flange * FLANGE_OUT)
        cue.Place(T_flange * TOOL)
        if physics.Moving() or blocker_physics.Moving():
            for _ in range(PHYSICS_SUBSTEPS):
                Step_Balls(DT / PHYSICS_SUBSTEPS)
            ball.Place(physics.pos, physics.R)
            blocker.Place(blocker_physics.pos, blocker_physics.R)
            trail.append(physics.pos.copy())
        env.step(DT)
        time.sleep(DT)

    Update(robot.q)
    env.set_camera_pose([0.9, 0.2, 0.4], [0.25, -0.3, 0.05])

    # 1. Lower the cue onto the line of the shot
    print("[1] Lining up the jump shot over the red ball...")
    if q_address is not None:
        for q in rtb.jtraj(robot.q, q_address, 60).q:
            Update(q)

    # 2. Back-swing, then punch down through the middle of the ball
    print("[2] Striking...")
    q, hit = robot.q, False
    for s in Short_Stroke_Profile(DT, contact_s=-ADDRESS_GAP, tip_speed=TIP_SPEED,
                                  follow=FOLLOW, backswing=BACKSWING):
        T_tip = T_address * SE3.Tz(-s)
        q_new = Solve_IK(robot, T_tip, q, quiet=True)
        if q_new is None:
            print(f"    (IK failed at s = {s:+.3f} m - holding last pose; try a smaller BACKSWING)")
        else:
            q = q_new
        if not hit and np.dot(T_tip.t - (ball_start + contact), cue_dir) >= -1e-6:
            # The stroke is built to be moving at exactly TIP_SPEED here. (Dividing
            # the last step by DT under-reads it: that step is shorter than a frame.)
            tip_vel = TIP_SPEED * cue_dir
            physics.Hit(CUE_TO_BALL * tip_vel, contact)
            hit = True
            print(f"[3] Hit! tip speed {np.linalg.norm(tip_vel):.2f} m/s, "
                  f"ball leaves the cloth at {physics.vz:.2f} m/s upwards")
        Update(q)

    # 3. Pull the cue back up out of the way, ball keeps going
    q_up = Solve_IK(robot, T_address * SE3.Tz(-LIFT_AFTER), q)
    if q_up is not None:
        for q in rtb.jtraj(q, q_up, 40).q:
            Update(q)
    while physics.Moving() or blocker_physics.Moving():
        Update(q)

    if trail:
        travelled = np.linalg.norm(trail[-1][:2] - ball_start[:2])
        print(f"Shot complete - jumped {physics.max_height * 100:.1f} cm off the cloth, "
              f"bounced {physics.bounces} times, stopped {travelled:.2f} m from where it started")
        gap = closest[0] - 2 * ball.radius
        if collided[0]:
            print("It did NOT clear the blocker - raise TIP_SPEED or ELEVATION_DEG")
        else:
            print(f"Cleared the blocker with {gap * 100:.1f} cm to spare")
    env.hold()


if __name__ == "__main__":
    main()
