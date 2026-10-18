"""
Pool_Scene.py - where everything sits, measured from your parts.

World frame: origin on the floor at the centre of the track (and table).

  Robot_Track : origin at its bottom centre. Wooden rings are 0.90 m tall;
                the WHITE channel floor the turtlebot drives on is at z = 0.80.
                Channel centre line: x = +/-1.525 (short ends), y = +/-0.925 (long sides).
  Pool_Table  : origin on the table top; legs reach 0.749 m down, so the table
                is placed at z = 0.749. Felt (playing surface) is then at z = 0.83.
  Turtlebot   : origin at its bottom centre; the Doosan sits on top at 0.141 m.
  Doosan      : base frame is 22 mm above the bottom of its base mesh.
  Cue_Holder  : bolts onto the flange; the cue rests in its groove.
"""
import numpy as np
from math import pi
from spatialmath import SE3


# ------------------------------------TRACK / TABLE-----------------------------------------------------

CHANNEL_Z = 0.80                      # white driving surface
CHANNEL_X = 1.525                     # centre line of the short-end channels
CHANNEL_Y = 0.925                     # centre line of the long-side channels

TABLE_Z = 0.749                       # table origin height (legs on the floor)
FELT_Z = TABLE_Z + 0.0807             # playing surface
PLAY_AREA = (-1.065, 1.070, -0.440, 0.445)   # cushion noses (xmin, xmax, ymin, ymax)

TURTLE_HEIGHT = 0.141                 # top of the turtlebot (where the Doosan base sits)
DOOSAN_FOOT = 0.022                   # Doosan base frame above its mesh underside

# ------------------------------------CUE HOLDER-----------------------------------------------------

GROOVE_BOTTOM = 0.025                 # flange face -> bottom of the holder groove
HANDLE_RADIUS_AT_GRIP = 0.0112        # handle radius where it sits in the groove (0.55 m from the butt)
GROOVE_HEIGHT = GROOVE_BOTTOM + HANDLE_RADIUS_AT_GRIP   # flange face -> cue centre line


# ------------------------------------POSES-----------------------------------------------------

def Turtle_Pose(x, y, yaw_deg):
    """Turtlebot pose on the channel floor."""
    return SE3(x, y, CHANNEL_Z) * SE3.Rz(np.radians(yaw_deg))


def Robot_Base_On(turtle_pose):
    """Doosan base frame when sitting on top of the turtlebot."""
    return turtle_pose * SE3.Tz(TURTLE_HEIGHT + DOOSAN_FOOT)


# ------------------------------------DRIVING AROUND THE TRACK-----------------------------------------------------

# corners of the channel centre line, anticlockwise
_CORNERS = [np.array(c) for c in [(CHANNEL_X, -CHANNEL_Y), (CHANNEL_X, CHANNEL_Y),
                                  (-CHANNEL_X, CHANNEL_Y), (-CHANNEL_X, -CHANNEL_Y)]]
_SEG_LEN = [np.linalg.norm(_CORNERS[(i + 1) % 4] - _CORNERS[i]) for i in range(4)]
_LOOP = sum(_SEG_LEN)


def _Loop_S(xy):
    """Distance along the loop (anticlockwise from the first corner) of a point on it."""
    xy = np.asarray(xy, dtype=float)
    best, s0 = None, 0.0
    for i in range(4):
        a, b = _CORNERS[i], _CORNERS[(i + 1) % 4]
        t = np.clip(np.dot(xy - a, b - a) / _SEG_LEN[i] ** 2, 0, 1)
        d = np.linalg.norm(a + t * (b - a) - xy)
        if best is None or d < best[0]:
            best = (d, s0 + t * _SEG_LEN[i])
        s0 += _SEG_LEN[i]
    return best[1]


def _Loop_Point(s):
    s %= _LOOP
    for i in range(4):
        if s <= _SEG_LEN[i]:
            a, b = _CORNERS[i], _CORNERS[(i + 1) % 4]
            return a + (s / _SEG_LEN[i]) * (b - a)
        s -= _SEG_LEN[i]
    return _CORNERS[0].copy()


def _Turn(yaw_from, yaw_to, dt, turn_speed):
    diff = (yaw_to - yaw_from + 180) % 360 - 180          # shortest way round
    n = max(1, int(abs(diff) / (turn_speed * dt)))
    return [yaw_from + diff * k / n for k in range(1, n + 1)]


def Drive_Path(start, goal, dt, speed=0.35, turn_speed=90.0):
    """Turtlebot poses (x, y, yaw_deg) from start to goal along the channel.

    start, goal : (x, y, yaw_deg), both on the channel centre line.
                  goal yaw may be None = keep facing the way it arrived.
    The turtlebot turns to face where it is going, drives straight, turns on
    the spot at each corner, then turns to the goal's yaw at the end.
    """
    s0, s1 = _Loop_S(start[:2]), _Loop_S(goal[:2])
    forward = (s1 - s0) % _LOOP
    sign = 1 if forward <= _LOOP / 2 else -1                 # shorter way round
    dist = forward if sign == 1 else _LOOP - forward

    # break the route at the corners it passes
    stops = [0.0]
    corner_s = np.cumsum([0] + _SEG_LEN[:-1])
    for c in corner_s:
        along = ((c - s0) * sign) % _LOOP
        if 1e-6 < along < dist - 1e-6:
            stops.append(along)
    stops = sorted(stops) + [dist]

    poses, yaw = [], start[2]
    for a, b in zip(stops[:-1], stops[1:]):
        p_a, p_b = _Loop_Point(s0 + sign * a), _Loop_Point(s0 + sign * b)
        heading = np.degrees(np.arctan2(p_b[1] - p_a[1], p_b[0] - p_a[0]))
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


# ------------------------------------POCKETS-----------------------------------------------------

# hole centres measured from Pool_Table.dae (table is centred on the origin)
POCKETS = [(1.104, 0.501), (1.104, -0.499), (-1.096, 0.501), (-1.096, -0.499),   # corners
           (0.004, 0.501), (0.004, -0.499)]                                     # middles
POCKET_RADIUS = 0.055                 # hole radius
POCKET_FLOOR_Z = TABLE_Z - 0.079      # bottom of the holes (a potted ball rests on this)
