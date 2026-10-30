# ------------------------------------POOL SCENE--------------------------------------------------------------
# Measurements of the track, table, pockets and cue holder (taken from the CAD models), and how a turtlebot
# drives round the track. Kept in one file so every other file uses the same numbers.

import numpy as np
from spatialmath import SE3


# ------------------------------------TRACK AND TABLE---------------------------------------------------------
# Heights and sizes: the turtlebots run in a rectangular channel round the table (outer loop) or just inside
# it (inner loop).

TRACK_NAME = "Robot_Track_v2"
TRACK_POSE = SE3(0.005, -0.12, 0.0)

CHANNEL_Z = 0.75
CHANNEL_X = 1.6125
CHANNEL_Y = 0.9375
INNER_X, INNER_Y = 1.44, 0.84

RIM_TOP_Z = 0.85

TABLE_Z = 0.749
FELT_Z = TABLE_Z + 0.0807
PLAY_AREA = (-1.065, 1.070, -0.440, 0.445)

TURTLE_HEIGHT = 0.141
DOOSAN_FOOT = 0.022


# ------------------------------------CUE HOLDER--------------------------------------------------------------
# Where the cue sits in the holder's groove, used to work out the cue tip from the robot's flange.

GROOVE_BOTTOM = 0.025
HANDLE_RADIUS_AT_GRIP = 0.0112
GROOVE_HEIGHT = GROOVE_BOTTOM + HANDLE_RADIUS_AT_GRIP


# ------------------------------------POSES-------------------------------------------------------------------
# A turtlebot's pose on the track, and the pose of the robot mounted on top of it.

def Turtle_Pose(x, y, yaw_deg):
    return SE3(x, y, CHANNEL_Z) * SE3.Rz(np.radians(yaw_deg))


def Robot_Base_On(turtle_pose):
    return turtle_pose * SE3.Tz(TURTLE_HEIGHT + DOOSAN_FOOT)


# ------------------------------------DRIVING AROUND THE TRACK------------------------------------------------
# The track is treated as a rectangle: any point is turned into a distance along it and back. Drive_Path
# goes the short way round, stops to turn at each corner, and can reverse instead of turning round when
# allowed.

def _Loop(half_x, half_y):
    corners = [np.array(c) for c in [(half_x, -half_y), (half_x, half_y),
                                     (-half_x, half_y), (-half_x, -half_y)]]
    seg = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    return corners, seg, sum(seg)


def _Loop_S(xy, L):
    corners, seg, _ = L
    xy = np.asarray(xy, dtype=float)
    best, s0 = None, 0.0
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        t = np.clip(np.dot(xy - a, b - a) / seg[i] ** 2, 0, 1)
        d = np.linalg.norm(a + t * (b - a) - xy)
        if best is None or d < best[0]:
            best = (d, s0 + t * seg[i])
        s0 += seg[i]
    return best[1]


def _Loop_Point(s, L):
    corners, seg, total = L
    s %= total
    for i in range(4):
        if s <= seg[i]:
            a, b = corners[i], corners[(i + 1) % 4]
            return a + (s / seg[i]) * (b - a)
        s -= seg[i]
    return corners[0].copy()


def _Turn(yaw_from, yaw_to, dt, turn_speed):
    diff = (yaw_to - yaw_from + 180) % 360 - 180
    n = max(1, int(abs(diff) / (turn_speed * dt)))
    return [yaw_from + diff * k / n for k in range(1, n + 1)]


def _Angle_Diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def Drive_Path(start, goal, dt, speed=0.35, turn_speed=90.0, loop=(CHANNEL_X, CHANNEL_Y),
               allow_reverse=False):
    L = _Loop(*loop)
    total, seg = L[2], L[1]
    s0, s1 = _Loop_S(start[:2], L), _Loop_S(goal[:2], L)
    forward = (s1 - s0) % total
    sign = 1 if forward <= total / 2 else -1
    dist = forward if sign == 1 else total - forward

    stops = [0.0]
    corner_s = np.cumsum([0] + seg[:-1])
    for c in corner_s:
        along = ((c - s0) * sign) % total
        if 1e-6 < along < dist - 1e-6:
            stops.append(along)
    stops = sorted(stops) + [dist]

    poses, yaw = [], start[2]
    pairs = list(zip(stops[:-1], stops[1:]))
    for k, (a, b) in enumerate(pairs):
        p_a, p_b = _Loop_Point(s0 + sign * a, L), _Loop_Point(s0 + sign * b, L)
        heading = np.degrees(np.arctan2(p_b[1] - p_a[1], p_b[0] - p_a[0]))
        if allow_reverse:
            options = [heading, heading + 180]
            last = k == len(pairs) - 1
            if last and goal[2] is not None and min(_Angle_Diff(o, goal[2]) for o in options) < 1:
                heading = min(options, key=lambda o: _Angle_Diff(o, goal[2]))
            else:
                heading = min(options, key=lambda o: _Angle_Diff(o, yaw))
            heading = (heading + 180) % 360 - 180
        for y in _Turn(yaw, heading, dt, turn_speed):
            poses.append((p_a[0], p_a[1], y))
        yaw = heading
        n = max(1, int((b - a) / (speed * dt)))
        for k in range(1, n + 1):
            p = p_a + (p_b - p_a) * k / n
            poses.append((p[0], p[1], yaw))
    if goal[2] is not None:
        for y in _Turn(yaw, goal[2], dt, turn_speed):
            poses.append((goal[0], goal[1], y))
    return poses


# ------------------------------------POCKETS-----------------------------------------------------------------
# Pocket centres measured from the table model, the capture radius and how deep a potted ball drops.

POCKETS = [(1.104, 0.501), (1.104, -0.499), (-1.096, 0.501), (-1.096, -0.499),
           (0.004, 0.501), (0.004, -0.499)]
POCKET_RADIUS = 0.055
POCKET_FLOOR_Z = TABLE_Z - 0.079
