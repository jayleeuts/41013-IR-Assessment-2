"""
Pool_Trick_Shots.py - the trick-shot run-out after the break (used by Pool_Robot_Menu).

  * LAYOUT       : where every ball sits after the break - measured from the
                   photo you sent (orange already potted).
  * Pool_Ball    : ball physics for the run-out. Same as Ball_Physics (slide,
                   roll, spin, cushions) plus:
                     - flying / bouncing for the JUMP shot (Shuban's jump model
                       from Test_Meca500_v1_Jump_Over.py),
                     - pocket jaws: a ball whose centre gets within POCKET_CAPTURE
                       of a hole drops in (a ball running along the rail into a
                       corner falls in, like on a real table).
  * Table_Step   : one physics step for every ball + ball-ball collisions (3D, so
                   a jumping ball really does fly over the ball in its way).
  * SHOTS        : the 8 shots. Each one was found by simulating this exact physics
                   (see the planning notes at the bottom), so the robot only has to
                   hit the cue ball the way it says - the balls do the rest, and
                   if they bump into each other on the way, they bump.
  * Break_Script : the break is hard-coded (the rack can't be made to land on
                   the photo layout by physics): every ball slides from the rack
                   to its photo spot, the orange drops into a pocket.
"""
import numpy as np
from spatialmath import SE3
from Pool_Motion import Ball_Physics, Cue_Pose, _Rotation_From_Vector


# ------------------------------------TABLE LAYOUT AFTER THE BREAK (from the photo)------------------------------
#
# Photo top = far (foot) end = +x, photo left = +y.
#   top-left  pocket = 0 (+1.104, +0.501)    top-right    = 1 (+1.104, -0.499)
#   middle-left      = 4 (+0.004, +0.501)    middle-right = 5 (+0.004, -0.499)
#   bottom-left      = 2 (-1.096, +0.501)    bottom-right = 3 (-1.096, -0.499)

POCKET_NAMES = {0: "top-left", 1: "top-right", 2: "bottom-left", 3: "bottom-right",
                4: "middle-left", 5: "middle-right"}

LAYOUT = {0: (-0.920, -0.410),        # cue ball
          1: (-0.763, -0.017),        # yellow
          2: (-0.328, 0.361),         # blue
          3: (-0.820, -0.241),        # red (nudged 2 cm: it half-covers the yellow, so the cue ball must jump it)
          4: (-1.035, -0.273),        # purple
          6: (-0.630, 0.192),         # green
          7: (-0.460, -0.040),        # brown
          8: (-0.046, 0.393),         # black
          9: (-0.166, 0.200)}         # nine (nudged 7 cm so it is not dead in the yellow's path)
ORANGE = 5
ORANGE_POCKET = 1                     # orange goes down in the break (top-right corner)

POCKET_CAPTURE = 0.09                 # ball centre this close to a hole -> it drops in
MU_ROLL = 0.012                       # rolling resistance of the cloth (real cloth is about 0.01)

# ------------------------------------BALL PHYSICS-----------------------------------------------------

TABLE_BOUNCE = 0.5                    # (Shuban) upward speed kept on each landing
BOUNCE_FRICTION = 0.2                 # (Shuban) cloth grip on each landing
BOUNCE_STOP = 0.15                    # (Shuban) stop bouncing below this upward speed (m/s)


class Pool_Ball(Ball_Physics):
    """Ball_Physics + jumping (Shuban's model) + pocket jaws."""

    def __init__(self, centre, radius, capture=POCKET_CAPTURE, **kw):
        super().__init__(centre, radius, **kw)
        self.capture = capture
        self.floor_z = float(centre[2])
        self.vz = 0.0
        self.airborne = False
        self.cushion_hits = 0
        self.guide = None              # (x, y, straight_in): where to send this ball when it is hit (hard-coded shots)
        self.homing = False

    def Set(self, xyz):
        """Put the ball back on the cloth at rest (not potted)."""
        self.pos = np.array(xyz, dtype=float)
        self.floor_z = float(xyz[2])
        self.Stop()
        self.vz, self.airborne, self.potted = 0.0, False, None
        self.cushion_hits = 0
        self.guide, self.homing = None, False

    def Hit(self, impulse_per_mass, contact_rel):
        """Cue strike. The downward part of a steep hit squashes the ball into the
        cloth, which throws it back up (jump shot)."""
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
        self.vz -= self.G * dt                                   # flying: gravity only
        self.pos[:2] += self.v[:2] * dt
        self.pos[2] += self.vz * dt
        self.R = _Rotation_From_Vector(self.w * dt) @ self.R
        self._Cushions()
        if self.pos[2] <= self.floor_z and self.vz < 0:
            self._Land()
        self._Check_Pockets()

    def _Cushions(self):
        if self.homing:                                          # guided straight into its pocket
            return
        v_before = self.v.copy()
        super()._Cushions()
        if np.any(np.sign(self.v[:2]) != np.sign(v_before[:2])):
            self.cushion_hits += 1

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
        if self.airborne and self.pos[2] > self.floor_z + self.r:   # flying over the hole
            return
        for i, p in enumerate(self.pockets):
            if np.linalg.norm(self.pos[:2] - p) < self.capture:
                self.potted = i
                self._vz = 0.0
                self.airborne = False
                return


def Collide(a, b, e=0.95):
    """Two balls touching (in 3D - a ball in the air can pass over another).
    Swap the part of their speed along the line between centres; spin is kept."""
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
    """One physics step: move every moving ball, then sort out collisions.
    balls = list of Pool_Ball (always in the same order -> same result every run)."""
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
    for b in live:                                               # hard-coded shots: steer the hit ball
        if b.guide is not None and np.linalg.norm(b.v[:2]) > 1e-6:
            aim = np.asarray(b.guide[:2]) - b.pos[:2]
            b.v[:2] = np.linalg.norm(b.v[:2]) * aim / np.linalg.norm(aim)
            b.w[:] = 0.0
            b.homing = bool(b.guide[2])                          # True = straight in, cushions ignored
            b.guide = None


# ------------------------------------CUE GEOMETRY-----------------------------------------------------

def Cue_Dir(aim_deg, elev_deg):
    """Unit vector along the cue, towards the ball (tilted down by elev_deg)."""
    a, e = np.radians(aim_deg), np.radians(elev_deg)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), -np.sin(e)])


def Contact_Point(d, side, height, radius):
    """Where the tip touches the ball, relative to its centre.
    side / height = offset from the centre as a fraction of the radius, measured
    square to the cue (side + = right, height + = above centre)."""
    up = np.array([0.0, 0.0, 1.0]) - d[2] * d
    up /= np.linalg.norm(up)
    right = np.cross(d, up)
    off = side * radius * right + height * radius * up
    return off - np.sqrt(max(0.0, radius ** 2 - off @ off)) * d


def Shot_Poses(ball_centre, shot, radius, address_gap):
    """(cue direction, contact point rel. to the ball centre, cue-tip address pose)."""
    d = Cue_Dir(shot["aim"], shot["elev"])
    contact = Contact_Point(d, shot["side"], shot["height"], radius)
    T_address = Cue_Pose(np.asarray(ball_centre) + contact - address_gap * d,
                         shot["aim"], shot["elev"], shot["roll"])
    return d, contact, T_address


# ------------------------------------HARD-CODED BREAK-----------------------------------------------------

def Break_Script(start, goal, n_frames, radius, seed=3):
    """Frames of ball positions from the rack (start) to the photo layout (goal).

    start, goal : {ball: (x, y)} - the orange's goal is its pocket.
    Each ball leaves when the shock reaches it (front of the rack first), slides
    out fast and slows down like a real ball. Balls never pass through each
    other: whenever two get closer than a ball width they are pushed apart, then
    keep heading for their spot. Returns a list (one per frame) of {ball: (x, y)}.
    """
    rng = np.random.default_rng(seed)
    names = list(start)
    p0 = np.array([start[n] for n in names], dtype=float)
    p1 = np.array([goal[n] for n in names], dtype=float)
    apex = p0[np.argmin(p0[:, 0])]
    delay = np.array([0.0 if n == 0 else 0.6 * np.linalg.norm(p - apex) for n, p in zip(names, p0)])
    dur = np.array([1.6 if n == 0 else 1.1 + 0.6 * rng.random() for n in names])
    total = (delay + dur).max() + 0.3
    free = np.array([n != ORANGE for n in names])            # the orange is allowed into its pocket
    pos, out = p0.copy(), []
    for f in range(n_frames):
        t = f / max(1, n_frames - 1) * total
        tau = np.clip((t - delay) / dur, 0, 1)
        target = p0 + (1 - (1 - tau) ** 2)[:, None] * (p1 - p0)
        pos += 0.5 * (target - pos)                           # follow the schedule
        for _ in range(6):                                    # push touching balls apart
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    g = pos[j] - pos[i]
                    d = np.linalg.norm(g)
                    if d < 2 * radius + 0.002 and d > 1e-9:
                        push = (2 * radius + 0.002 - d) / 2 * g / d
                        pos[i] -= push
                        pos[j] += push
        if f == n_frames - 1:
            pos = p1.copy()
        out.append({n: pos[k].copy() for k, n in enumerate(names)})
    return out


# ------------------------------------THE SHOTS-----------------------------------------------------
#
# Filled in by the planner (bottom of this file). Per shot:
#   ball, pocket   : the ball to pot and where
#   kind           : "jump", "bank", "masse" or "plain"
#   aim_deg        : direction the cue points (world, degrees from +x)
#   elev_deg       : cue tilt down from horizontal
#   tip_speed      : cue speed at contact (m/s)
#   side, height   : where the tip hits, as a fraction of the radius
#                    (side + = right of centre, height + = above centre)
#   turtle         : (x, y, yaw) for the Doosan's turtlebot
SHOTS = [
    dict(ball=1, pocket=0, kind='jump', aim=74.8, elev=35.0, speed=1.9, side=0.0, height=0.0,
         guide=None, turtle=(-1.46, -0.9375, 0.0), roll=-90.0, follow=0.015, back=0.12,
         q_ready=(2.347992, -1.085673, 0.600864, 0.583138, 1.985044, -0.700407),
         layout={0: (-0.92, -0.41), 1: (-0.763, -0.017), 2: (-0.328, 0.361), 3: (-0.82, -0.241), 4: (-1.035, -0.273), 6: (-0.63, 0.192), 7: (-0.46, -0.04), 8: (-0.046, 0.393), 9: (-0.166, 0.2)}),
    dict(ball=2, pocket=5, kind='bank', aim=47.8258440979558, elev=8.0, speed=1.2, side=0.0, height=0.4,
         guide=(-0.3135848267934584, 0.4164, False), turtle=(-1.11, -0.9375, 0.0), roll=90.0, follow=0.03, back=0.2,
         q_ready=(-2.881924, 1.127056, 1.260208, 1.129551, 1.95244, -1.042687),
         layout={0: (-0.9616, -0.3098), 2: (-0.328, 0.361), 3: (-0.82, -0.241), 4: (-1.035, -0.273), 6: (-0.63, 0.192), 7: (-0.46, -0.04), 8: (-0.046, 0.393), 9: (-0.166, 0.2)}),
    dict(ball=3, pocket=3, kind='masse', aim=-144.54414592834087, elev=48.0, speed=1.8, side=-0.6, height=0.0,
         guide=None, turtle=(0.695, 0.9375, 0.0), roll=-90.0, follow=0.015, back=0.12,
         q_ready=(-2.542663, 0.330357, 0.832443, 1.57869, 1.589055, -1.895767),
         layout={0: (-0.2663, 0.2831), 3: (-0.82, -0.241), 4: (-1.035, -0.273), 6: (-0.63, 0.192), 7: (-0.46, -0.04), 8: (-0.046, 0.393), 9: (-0.166, 0.2)}),
    dict(ball=4, pocket=3, kind='plain', aim=-171.04204750153087, elev=8.0, speed=1.1, side=0.0, height=0.4,
         guide=None, turtle=(0.545, 0.9375, 0.0), roll=-90.0, follow=0.03, back=0.2,
         q_ready=(-1.099575, 1.09955, 0.636189, -2.037727, 0.354183, 0.6328),
         layout={0: (-0.4083, -0.1186), 4: (-1.035, -0.273), 6: (-0.63, 0.192), 7: (-0.46, -0.04), 8: (-0.046, 0.393), 9: (-0.166, 0.2)}),
    dict(ball=6, pocket=2, kind='plain', aim=138.0966826884882, elev=8.0, speed=1.4, side=0.0, height=0.0,
         guide=None, turtle=(-0.11, -0.9375, 0.0), roll=180.0, follow=0.03, back=0.2,
         q_ready=(0.157191, 0.874412, 0.987962, -0.116471, 1.193168, 0.926864),
         layout={0: (-0.2733, -0.1097), 6: (-0.63, 0.192), 7: (-0.46, -0.04), 8: (-0.046, 0.393), 9: (-0.166, 0.2)}),
    dict(ball=7, pocket=5, kind='plain', aim=-49.58711363626581, elev=8.0, speed=1.6, side=0.0, height=0.4,
         guide=None, turtle=(-1.72, 0.7525, 90.0), roll=90.0, follow=0.03, back=0.2,
         q_ready=(-0.568266, 0.952542, 1.467302, 0.387381, 2.254585, -1.458167),
         layout={0: (-0.7657, 0.3115), 7: (-0.46, -0.04), 8: (-0.046, 0.393), 9: (-0.166, 0.2)}),
    dict(ball=8, pocket=2, kind='plain', aim=111.41725489053977, elev=8.0, speed=1.5, side=0.0, height=0.4,
         guide=(-1.096, 0.501, True), turtle=(0.94, -0.9375, 0.0), roll=90.0, follow=0.03, back=0.2,
         q_ready=(-2.556372, 0.937159, 1.466719, 0.282289, 2.288455, -1.521953),
         layout={0: (0.163, -0.1644), 8: (-0.046, 0.393), 9: (-0.166, 0.2)}),
    dict(ball=9, pocket=0, kind='plain', aim=4.2326082040391695, elev=8.0, speed=0.8, side=0.0, height=0.4,
         guide=None, turtle=(-1.72, 0.1025, 90.0), roll=-90.0, follow=0.03, back=0.2,
         q_ready=(2.25292, 0.637974, 1.59677, -1.165465, 2.037607, 0.9012),
         layout={0: (-1.029, 0.1306), 9: (-0.166, 0.2)}),
]
