"""
Pool_Motion.py - shared pieces for the Doosan pool scripts.

  Side_Grip_Tool  : flange -> cue tip transform, cue held BESIDE the flange
  Simple_Gripper  : two-finger gripper drawn on the flange
  Solve_IK        : IK for a cue-tip pose, retrying with several seeds
  Cue_Pose        : cue-tip pose from position, aim, elevation and roll
  Stroke_Profile  : rock back and forth, back-swing, strike
  Ball_Physics    : sliding / rolling ball with spin (curves, visibly rotates),
                    bounces off cushions
  Balls_Collide   : two balls hitting each other
  Holder_Tool     : flange -> cue tip for the 3D-printed cue holder
"""
import numpy as np
from math import pi
import roboticstoolbox as rtb
import spatialgeometry as sg
from spatialmath import SE3


# ------------------------------------GRIP-----------------------------------------------------

def Side_Grip_Tool(grip_offset, grip_from_tip):
    """Flange -> cue-tip transform for a cue held ACROSS the gripper fingers.

    grip_offset   : distance from the flange face out to the cue's centre line
    grip_from_tip : distance along the cue from the grip point to the tip

    The cue lies along the flange's X axis, so it passes beside the robot's
    wrist instead of straight through it.
    """
    return SE3.Tz(grip_offset) * SE3.Ry(pi / 2) * SE3.Tz(grip_from_tip)


def Holder_Tool(groove_height, grip_from_tip):
    """Flange -> cue-tip transform for the Cue_Holder part.

    The holder bolts flat onto the flange; its groove runs along the flange's
    X axis. groove_height = flange face -> cue centre line (cue resting in the groove).
    Same maths as Side_Grip_Tool - the cue lies across the flange, beside the wrist.
    """
    return Side_Grip_Tool(groove_height, grip_from_tip)


class Simple_Gripper:
    """Palm + two fingers either side of the cue. Purely visual."""

    def __init__(self, grip_offset, cue_diameter=0.03, color=(0.25, 0.25, 0.28, 1)):
        self.grip_offset = grip_offset
        gap = cue_diameter / 2 + 0.006
        finger_len = grip_offset + cue_diameter / 2
        self.palm = sg.Cuboid(scale=[0.07, 0.07, 0.02], color=list(color))
        self.fingers = [sg.Cuboid(scale=[0.05, 0.012, finger_len], color=list(color)) for _ in range(2)]
        self._palm_off = SE3.Tz(0.01)
        self._finger_offs = [SE3(0, side * (gap + 0.006), finger_len / 2) for side in (-1, 1)]

    def Place(self, T_flange):
        self.palm.T = (T_flange * self._palm_off).A
        for finger, off in zip(self.fingers, self._finger_offs):
            finger.T = (T_flange * off).A

    def Add_To_Env(self, env):
        env.add(self.palm)
        for finger in self.fingers:
            env.add(finger)


# ------------------------------------IK-----------------------------------------------------

def Seed_Candidates(robot, T_world):
    """A spread of starting joint guesses, all facing the target."""
    target = robot.base.inv() * T_world
    q1 = np.arctan2(target.t[1], target.t[0])
    seeds = []
    for q2, q3 in [(pi / 4, pi / 2), (pi / 6, 2 * pi / 3), (pi / 3, pi / 3),
                   (0.0, pi / 2), (-pi / 6, 2 * pi / 3), (pi / 2, pi / 4)]:
        for q5 in (pi / 2, -pi / 2):
            seeds.append(np.array([q1, q2, q3, 0.0, q5, 0.0]))
    return seeds


def Solve_IK(robot, T_tip, tool, q_seed=None, quiet=False):
    """IK for a cue-tip pose. Tries q_seed first (smooth motion), then other seeds."""
    T_flange = T_tip * tool.inv()
    seeds = ([np.asarray(q_seed, dtype=float)] if q_seed is not None else []) + Seed_Candidates(robot, T_flange)
    for seed in seeds:
        sol = robot.ikine_LM(T_flange, q0=seed, joint_limits=False)
        if sol.success:
            return sol.q
    if not quiet:
        print(f"IK failed at tip position {np.round(T_tip.t, 3)} - target may be out of reach")
    return None


# ------------------------------------CUE POSES-----------------------------------------------------

def Cue_Pose(tip_xyz, aim_deg, elevation_deg, roll_deg=0.0):
    """Cue-tip frame: origin at the tip, +Z along the cue towards the ball.

    aim_deg       : shot direction about world Z (0 = +X)
    elevation_deg : how steeply the cue points down (8 = normal shot, ~70 = masse)
    roll_deg      : spin of the frame about the cue axis - decides which side
                    of the cue the gripper is on (180 = gripper above the cue)
    """
    R = (SE3.Rz(np.deg2rad(aim_deg)) * SE3.Ry(pi / 2 + np.deg2rad(elevation_deg))
         * SE3.Rz(np.deg2rad(roll_deg)))
    return SE3(*tip_xyz) * R


def Stroke_Profile(dt, n_rocks=3, rock_amp=0.08, rock_time=1.0,
                   back=0.12, back_time=0.8, follow=0.05, strike_time=0.35):
    """Distance the tip is pulled back from the address point, one value per step.

    positive = pulled back, negative = pushed past the address point.
    """
    s = [rock_amp * (1 - np.cos(2 * pi * t / rock_time)) / 2
         for t in np.arange(0, n_rocks * rock_time, dt)]
    s += list(rtb.quintic(0, back, int(back_time / dt)).q)
    s += list(rtb.quintic(back, -follow, int(strike_time / dt)).q)
    return np.array(s)


# ------------------------------------BALL PHYSICS-----------------------------------------------------

def _Rotation_From_Vector(w):
    """Rotation matrix for rotating by |w| radians about w (Rodrigues)."""
    angle = np.linalg.norm(w)
    if angle < 1e-12:
        return np.eye(3)
    k = w / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


class Ball_Physics:
    """Cue ball on cloth: slides with friction until it rolls, spin bends its path.

    pos : centre position         v : velocity (m/s)
    w   : spin (rad/s, vector)    R : orientation (so the markings turn)
    """
    G = 9.81

    def __init__(self, centre, radius, mu_slide=0.2, mu_roll=0.03, spin_decay=0.8,
                 bounds=None, cushion_e=0.75,
                 pockets=None, pocket_radius=0.055, pocket_mouth=0.12, pocket_floor_z=None):
        """bounds  = (xmin, xmax, ymin, ymax) of the cushion noses (playing area).
                     The ball bounces off them, losing some speed (cushion_e).
        pockets = list of (x, y) hole centres. Within pocket_mouth of a hole the
                  cushions stop (the pocket's opening); once the ball's centre is
                  over the hole (pocket_radius) it drops to pocket_floor_z."""
        self.bounds = bounds
        self.cushion_e = cushion_e
        self.pockets = [np.array(p, dtype=float) for p in (pockets or [])]
        self.pocket_radius = pocket_radius
        self.pocket_mouth = pocket_mouth
        self.pocket_floor_z = pocket_floor_z
        self.potted = None                 # index of the pocket the ball fell into
        self._vz = 0.0
        self.pos = np.array(centre, dtype=float)
        self.r = radius
        self.v = np.zeros(3)
        self.w = np.zeros(3)
        self.R = np.eye(3)
        self.mu_slide, self.mu_roll, self.spin_decay = mu_slide, mu_roll, spin_decay

    def Hit(self, impulse_per_mass, contact_rel):
        """Cue strike. impulse_per_mass: cue impulse / ball mass (m/s, a vector).
        contact_rel: where the tip touches, relative to the ball centre."""
        J = np.asarray(impulse_per_mass, dtype=float)
        p = np.asarray(contact_rel, dtype=float)
        self.v += np.array([J[0], J[1], 0.0])            # table absorbs the downward part
        self.w += 5.0 / (2.0 * self.r ** 2) * np.cross(p, J)

    def Contact_Slip(self):
        """Velocity of the point touching the cloth (zero when rolling)."""
        return self.v + np.cross(self.w, [0.0, 0.0, -self.r])

    def Moving(self):
        if self.potted is not None:
            return self.pos[2] > self.pocket_floor_z + 1e-6      # still falling
        return np.linalg.norm(self.v) > 0.005 or np.linalg.norm(self.w) > 0.5

    def Step(self, dt):
        if self.potted is not None:
            self._Fall(dt)
            return
        slip = self.Contact_Slip()
        slip[2] = 0.0
        if np.linalg.norm(slip) > 0.01:
            # SLIDING: friction opposes the slip -> pushes the ball sideways = curve
            u = slip / np.linalg.norm(slip)
            self.v += -self.mu_slide * self.G * u * dt
            self.w += (5 * self.mu_slide * self.G / (2 * self.r)) * np.cross([0, 0, 1.0], u) * dt
            if np.dot(self.Contact_Slip()[:2], slip[:2]) < 0:      # overshot -> now rolling
                self.w[0], self.w[1] = -self.v[1] / self.r, self.v[0] / self.r
        else:
            # ROLLING: slow down gently, spin matches the motion
            speed = np.linalg.norm(self.v[:2])
            if speed > 1e-6:
                new_speed = max(0.0, speed - self.mu_roll * self.G * dt)
                self.v[:2] *= new_speed / speed
            self.w[0], self.w[1] = -self.v[1] / self.r, self.v[0] / self.r
        self.w[2] *= np.exp(-self.spin_decay * dt)                  # side spin dies away
        self.v[2] = 0.0

        self.pos += self.v * dt
        self.R = _Rotation_From_Vector(self.w * dt) @ self.R
        self._Cushions()
        self._Check_Pockets()

    def _Near_Pocket(self):
        return any(np.linalg.norm(self.pos[:2] - p) < self.pocket_mouth for p in self.pockets)

    def _Check_Pockets(self):
        for i, p in enumerate(self.pockets):
            if np.linalg.norm(self.pos[:2] - p) < self.pocket_radius:
                self.potted = i
                self._vz = 0.0
                if self.pocket_floor_z is None:
                    self.pocket_floor_z = self.pos[2] - 0.15
                return

    def _Fall(self, dt):
        """Drop into the hole: slide to its centre while falling under gravity."""
        centre = self.pockets[self.potted]
        self.pos[:2] += (centre - self.pos[:2]) * min(1.0, 8 * dt)
        self._vz -= self.G * dt
        self.pos[2] = max(self.pocket_floor_z, self.pos[2] + self._vz * dt)
        self.R = _Rotation_From_Vector(self.w * dt) @ self.R
        self.w *= 0.9
        if self.pos[2] <= self.pocket_floor_z:
            self.v[:] = 0.0

    def _Cushions(self):
        """Bounce off the cushions: flip the speed into the cushion, lose some of it."""
        if self.bounds is None or self._Near_Pocket():      # no cushion at a pocket opening
            return
        xmin, xmax, ymin, ymax = self.bounds
        for axis, lo, hi in ((0, xmin + self.r, xmax - self.r), (1, ymin + self.r, ymax - self.r)):
            if self.pos[axis] < lo and self.v[axis] < 0:
                self.pos[axis] = lo
                self.v[axis] = -self.cushion_e * self.v[axis]
            elif self.pos[axis] > hi and self.v[axis] > 0:
                self.pos[axis] = hi
                self.v[axis] = -self.cushion_e * self.v[axis]

    def Stop(self):
        self.v[:] = 0.0
        self.w[:] = 0.0


def Balls_Collide(a, b, e=0.95):
    """Two equal balls touching -> swap the part of their speed along the line
    between their centres (spin is kept, so follow/draw effects still happen)."""
    gap = b.pos - a.pos
    gap[2] = 0.0
    dist = np.linalg.norm(gap)
    if dist >= a.r + b.r or dist < 1e-9:
        return False
    n = gap / dist
    closing = np.dot(a.v - b.v, n)
    push = (a.r + b.r - dist) / 2                      # separate overlapping balls
    a.pos -= n * push
    b.pos += n * push
    if closing <= 0:
        return False
    j = (1 + e) / 2 * closing
    a.v -= j * n
    b.v += j * n
    return True


def Punch_Stroke_Profile(dt, contact_s, tip_speed, back=0.20, follow=0.04,
                         n_rocks=2, rock_amp=0.06, rock_time=1.0, back_time=0.8):
    """Stroke that is at FULL speed when it reaches the ball, then stops quickly.

    Needed for steep (masse) shots, where the table is just past the ball so
    there is no room for a long follow-through.
    contact_s : value of s where the tip touches the ball (negative = past address)
    tip_speed : cue speed at contact (m/s)
    follow    : how far past contact the tip travels while braking
    """
    s = [rock_amp * (1 - np.cos(2 * pi * t / rock_time)) / 2
         for t in np.arange(0, n_rocks * rock_time, dt)]
    s += list(rtb.quintic(0, back, int(back_time / dt)).q)
    # speed up steadily from rest to tip_speed exactly at contact
    # (the last sample lands EXACTLY on contact_s, so the hit is detected on time)
    run = back - contact_s
    n_accel = max(2, int(np.ceil(2 * run / tip_speed / dt)))
    for tau in np.linspace(0, 1, n_accel + 1)[1:]:
        s.append(back - run * tau ** 2)
    # brake to a stop over the follow distance
    n_brake = max(2, int(np.ceil(2 * follow / tip_speed / dt)))
    for tau in np.linspace(0, 1, n_brake + 1)[1:]:
        s.append(contact_s - follow * (2 * tau - tau ** 2))
    return np.array(s)
