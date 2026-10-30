# ------------------------------------POOL MOTION-------------------------------------------------------------
# Shared pieces for every arm that holds a cue: the cue holder tool, IK with several starting guesses, cue
# poses, the stroke profile and the ball physics.

import numpy as np
from math import pi
import roboticstoolbox as rtb
from spatialmath import SE3


# ------------------------------------GRIP--------------------------------------------------------------------
# The tool transform from the robot's flange to the cue tip for a cue held across the flange in the holder.

def Holder_Tool(grip_offset, grip_from_tip):
    return SE3.Tz(grip_offset) * SE3.Ry(pi / 2) * SE3.Tz(grip_from_tip)


# ------------------------------------IK----------------------------------------------------------------------
# IK is tried from the given seed first (so motions stay smooth), then from a set of standard elbow-up
# seeds. Rebuild_ETS is needed after a robot's DH lengths are scaled: the toolbox keeps an old copy of the
# link geometry, so without it IK and the Jacobian would solve for the unscaled robot.

def Seed_Candidates(robot, T_world):
    target = robot.base.inv() * T_world
    q1 = np.arctan2(target.t[1], target.t[0])
    seeds = []
    for q2, q3 in [(pi / 4, pi / 2), (pi / 6, 2 * pi / 3), (pi / 3, pi / 3),
                   (0.0, pi / 2), (-pi / 6, 2 * pi / 3), (pi / 2, pi / 4)]:
        for q5 in (pi / 2, -pi / 2):
            seeds.append(np.array([q1, q2, q3, 0.0, q5, 0.0]))
    return seeds


def Solve_IK(robot, T_tip, tool, q_seed=None, quiet=False):
    T_flange = T_tip * tool.inv()
    seeds = ([np.asarray(q_seed, dtype=float)] if q_seed is not None else []) + Seed_Candidates(robot, T_flange)
    for seed in seeds:
        sol = robot.ikine_LM(T_flange, q0=seed, joint_limits=False)
        if sol.success:
            return sol.q
    if not quiet:
        print(f"IK failed at tip position {np.round(T_tip.t, 3)} - target may be out of reach")
    return None


def Rebuild_ETS(robot):
    for i, link in enumerate(robot.links):
        ets = link._to_ets(link._sigma, link._theta, link.d, link.alpha, link.a, link.offset, link.isflip, link._mdh)
        for et in ets:
            if et.isjoint:
                et.qlim, et.jindex = link.qlim, i
        link._ets = ets


# ------------------------------------CUE POSES AND STROKES---------------------------------------------------
# A cue pose from its tip position, aim, tilt and roll, and the stroke used for every shot: practice rocks,
# a back-swing, then speeding up steadily so the cue is at full speed exactly when it reaches the ball, and
# a short braking follow-through.

def Cue_Pose(tip_xyz, aim_deg, elevation_deg, roll_deg=0.0):
    R = (SE3.Rz(np.deg2rad(aim_deg)) * SE3.Ry(pi / 2 + np.deg2rad(elevation_deg))
         * SE3.Rz(np.deg2rad(roll_deg)))
    return SE3(*tip_xyz) * R


def Punch_Stroke_Profile(dt, contact_s, tip_speed, back=0.20, follow=0.04,
                         n_rocks=2, rock_amp=0.06, rock_time=1.0, back_time=0.8):
    s = [rock_amp * (1 - np.cos(2 * pi * t / rock_time)) / 2
         for t in np.arange(0, n_rocks * rock_time, dt)]
    s += list(rtb.quintic(0, back, int(back_time / dt)).q)
    run = back - contact_s
    n_accel = max(2, int(np.ceil(2 * run / tip_speed / dt)))
    for tau in np.linspace(0, 1, n_accel + 1)[1:]:
        s.append(back - run * tau ** 2)
    n_brake = max(2, int(np.ceil(2 * follow / tip_speed / dt)))
    for tau in np.linspace(0, 1, n_brake + 1)[1:]:
        s.append(contact_s - follow * (2 * tau - tau ** 2))
    return np.array(s)


# ------------------------------------BALL PHYSICS------------------------------------------------------------
# Each ball slides, then rolls, with friction and spin, bounces off the cushions and drops into a pocket.
# Ball-to-ball collisions push overlapping balls apart and swap their speed along the line between them.

def _Rotation_From_Vector(w):
    angle = np.linalg.norm(w)
    if angle < 1e-12:
        return np.eye(3)
    k = w / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


class Ball_Physics:
    G = 9.81

    def __init__(self, centre, radius, mu_slide=0.2, mu_roll=0.03, spin_decay=0.8,
                 bounds=None, cushion_e=0.75,
                 pockets=None, pocket_radius=0.055, pocket_mouth=0.12, pocket_floor_z=None):
        self.bounds = bounds
        self.cushion_e = cushion_e
        self.pockets = [np.array(p, dtype=float) for p in (pockets or [])]
        self.pocket_radius = pocket_radius
        self.pocket_mouth = pocket_mouth
        self.pocket_floor_z = pocket_floor_z
        self.potted = None
        self._vz = 0.0
        self.pos = np.array(centre, dtype=float)
        self.r = radius
        self.v = np.zeros(3)
        self.w = np.zeros(3)
        self.R = np.eye(3)
        self.mu_slide, self.mu_roll, self.spin_decay = mu_slide, mu_roll, spin_decay

    def Hit(self, impulse_per_mass, contact_rel):
        J = np.asarray(impulse_per_mass, dtype=float)
        p = np.asarray(contact_rel, dtype=float)
        self.v += np.array([J[0], J[1], 0.0])
        self.w += 5.0 / (2.0 * self.r ** 2) * np.cross(p, J)

    def Contact_Slip(self):
        return self.v + np.cross(self.w, [0.0, 0.0, -self.r])

    def Moving(self):
        if self.potted is not None:
            return self.pos[2] > self.pocket_floor_z + 1e-6
        return np.linalg.norm(self.v) > 0.005 or np.linalg.norm(self.w) > 0.5

    def Step(self, dt):
        if self.potted is not None:
            self._Fall(dt)
            return
        slip = self.Contact_Slip()
        slip[2] = 0.0
        if np.linalg.norm(slip) > 0.01:
            u = slip / np.linalg.norm(slip)
            self.v += -self.mu_slide * self.G * u * dt
            self.w += (5 * self.mu_slide * self.G / (2 * self.r)) * np.cross([0, 0, 1.0], u) * dt
            if np.dot(self.Contact_Slip()[:2], slip[:2]) < 0:
                self.w[0], self.w[1] = -self.v[1] / self.r, self.v[0] / self.r
        else:
            speed = np.linalg.norm(self.v[:2])
            if speed > 1e-6:
                new_speed = max(0.0, speed - self.mu_roll * self.G * dt)
                self.v[:2] *= new_speed / speed
            self.w[0], self.w[1] = -self.v[1] / self.r, self.v[0] / self.r
        self.w[2] *= np.exp(-self.spin_decay * dt)
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
        centre = self.pockets[self.potted]
        self.pos[:2] += (centre - self.pos[:2]) * min(1.0, 8 * dt)
        self._vz -= self.G * dt
        self.pos[2] = max(self.pocket_floor_z, self.pos[2] + self._vz * dt)
        self.R = _Rotation_From_Vector(self.w * dt) @ self.R
        self.w *= 0.9
        if self.pos[2] <= self.pocket_floor_z:
            self.v[:] = 0.0

    def _Cushions(self):
        if self.bounds is None or self._Near_Pocket():
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


