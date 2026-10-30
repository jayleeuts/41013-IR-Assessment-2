# ------------------------------------POOL TRICK SHOTS--------------------------------------------------------
# The full table physics used in the game (pocket jaws, airborne jump shots, ball collisions) and the
# geometry that turns a planned shot (aim, tilt, side and height on the ball) into the cue's address pose.

import numpy as np
from Pool_Motion import Ball_Physics, Cue_Pose, _Rotation_From_Vector


# ------------------------------------TABLE CONSTANTS---------------------------------------------------------
# How close to a pocket centre a ball must get to drop, the width of the pocket mouths and jaws, and rolling
# friction.

POCKET_CAPTURE = 0.09
POCKET_MOUTH = 0.16
POCKET_JAW = 0.035
MU_ROLL = 0.012


# ------------------------------------BALL PHYSICS------------------------------------------------------------
# Pool_Ball adds what trick shots need on top of Pool_Motion's ball: a steep cue pushes the ball into the
# table so it jumps, lands with friction and bounces until it settles, and near a pocket the cushion steps
# back to the jaws.

TABLE_BOUNCE = 0.5
BOUNCE_FRICTION = 0.2
BOUNCE_STOP = 0.15


class Pool_Ball(Ball_Physics):
    def __init__(self, centre, radius, capture=POCKET_CAPTURE, jaws=False, **kw):
        super().__init__(centre, radius, **kw)
        self.jaws = jaws
        self.capture = capture
        self.floor_z = float(centre[2])
        self.vz = 0.0
        self.airborne = False

    def Set(self, xyz):
        self.pos = np.array(xyz, dtype=float)
        self.floor_z = float(xyz[2])
        self.Stop()
        self.vz, self.airborne, self.potted = 0.0, False, None

    def Hit(self, impulse_per_mass, contact_rel):
        J = np.asarray(impulse_per_mass, dtype=float)
        super().Hit(J, contact_rel)
        if J[2] < 0:
            self.vz = -TABLE_BOUNCE * J[2]
            self.airborne = self.vz > BOUNCE_STOP
            if not self.airborne:
                self.vz = 0.0

    def Moving(self):
        return self.airborne or super().Moving()

    def Step(self, dt):
        if not self.airborne or self.potted is not None:
            super().Step(dt)
            return
        self.vz -= self.G * dt
        self.pos[:2] += self.v[:2] * dt
        self.pos[2] += self.vz * dt
        self.R = _Rotation_From_Vector(self.w * dt) @ self.R
        self._Cushions()
        if self.pos[2] <= self.floor_z and self.vz < 0:
            self._Land()
        self._Check_Pockets()

    def _Cushions(self):
        if self.bounds is None:
            return
        if not self.jaws:
            super()._Cushions()
            return
        xmin, xmax, ymin, ymax = self.bounds
        if any(np.linalg.norm(self.pos[:2] - p) < POCKET_MOUTH for p in self.pockets):
            xmin, xmax, ymin, ymax = xmin - POCKET_JAW, xmax + POCKET_JAW, ymin - POCKET_JAW, ymax + POCKET_JAW
        for axis, lo, hi in ((0, xmin + self.r, xmax - self.r), (1, ymin + self.r, ymax - self.r)):
            if self.pos[axis] < lo and self.v[axis] < 0:
                self.pos[axis] = lo
                self.v[axis] = -self.cushion_e * self.v[axis]
            elif self.pos[axis] > hi and self.v[axis] > 0:
                self.pos[axis] = hi
                self.v[axis] = -self.cushion_e * self.v[axis]

    def _Land(self):
        self.pos[2] = self.floor_z
        vz_in = -self.vz
        slip = self.Contact_Slip()
        slip[2] = 0.0
        slip_speed = np.linalg.norm(slip)
        if slip_speed > 1e-9:
            u = slip / slip_speed
            dv = min(BOUNCE_FRICTION * (1 + TABLE_BOUNCE) * vz_in, 2.0 / 7.0 * slip_speed)
            self.v[:2] -= dv * u[:2]
            self.w += 5.0 / (2.0 * self.r) * dv * np.cross([0.0, 0.0, 1.0], u)
        self.vz = TABLE_BOUNCE * vz_in
        if self.vz < BOUNCE_STOP:
            self.vz, self.airborne = 0.0, False

    def _Check_Pockets(self):
        if self.airborne and self.pos[2] > self.floor_z + self.r:
            return
        for i, p in enumerate(self.pockets):
            if np.linalg.norm(self.pos[:2] - p) < self.capture:
                self.potted = i
                self._vz = 0.0
                self.airborne = False
                return


# ------------------------------------TABLE STEP--------------------------------------------------------------
# One physics step for every ball: move the moving ones, then find touching pairs all at once (vectorised)
# and collide them in a fixed order so every run gives the same result.

def Collide(a, b, e=0.95):
    gap = b.pos - a.pos
    if gap @ gap >= (a.r + b.r) ** 2:
        return False
    gap = gap.copy()
    gap[2] = 0.0
    dist = np.linalg.norm(gap)
    if dist < 1e-9:
        return False
    n = gap / dist
    push = max(0.0, (a.r + b.r - dist) / 2)
    a.pos -= n * push
    b.pos += n * push
    closing = np.dot(a.v - b.v, n)
    if closing <= 0:
        return False
    j = (1 + e) / 2 * closing
    a.v -= j * n
    b.v += j * n
    return True


def Table_Step(balls, dt):
    for b in balls:
        if b.Moving():
            b.Step(dt)
    live = [b for b in balls if b.potted is None]
    if len(live) < 2:
        return
    P = np.array([b.pos for b in live])
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    close = np.argwhere(np.triu(d2 < (2 * live[0].r + 0.002) ** 2, 1)) if live else []
    for i, j in close:
        Collide(live[i], live[j])


# ------------------------------------CUE GEOMETRY------------------------------------------------------------
# The cue direction, the point on the ball the tip touches (side spin and top/back spin), and the tip pose a
# small gap behind it.

def Cue_Dir(aim_deg, elev_deg):
    a, e = np.radians(aim_deg), np.radians(elev_deg)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), -np.sin(e)])


def Contact_Point(d, side, height, radius):
    up = np.array([0.0, 0.0, 1.0]) - d[2] * d
    up /= np.linalg.norm(up)
    right = np.cross(d, up)
    off = side * radius * right + height * radius * up
    return off - np.sqrt(max(0.0, radius ** 2 - off @ off)) * d


def Shot_Poses(ball_centre, shot, radius, address_gap):
    d = Cue_Dir(shot["aim"], shot["elev"])
    contact = Contact_Point(d, shot["side"], shot["height"], radius)
    T_address = Cue_Pose(np.asarray(ball_centre) + contact - address_gap * d,
                         shot["aim"], shot["elev"], shot["roll"])
    return d, contact, T_address
