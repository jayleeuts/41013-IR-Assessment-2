"""
Pool_Scene.py - where everything sits, measured from your parts.

World frame: origin on the floor at the centre of the track (and table).

  Robot_Track_v2 : 4.0 x 2.5 m, wooden rims 0.85 m tall, BLUE driving deck at
                z = 0.75 between the outer and inner rims. Its centre hole is not
                at its own origin, so the track is shifted by TRACK_POSE to put
                the hole (2.45 x 1.25 m) around the table at the world origin.
                Lane centre lines: x = +/-1.6125 (short ends), y = +/-0.9375 (long sides).
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

TRACK_NAME = "Robot_Track_v2"         # Pool_Parts mesh
TRACK_POSE = SE3(0.005, -0.12, 0.0)   # centres the track's hole on the table

CHANNEL_Z = 0.75                      # blue driving deck
CHANNEL_X = 1.6125                    # lane centre line, short ends
CHANNEL_Y = 0.9375                    # lane centre line, long sides
INNER_X, INNER_Y = 1.44, 0.84         # loop hugging the inner rim (turtlebot edge 2.5 cm from it)

RIM_TOP_Z = 0.85
# rims as boxes (xmin, xmax, ymin, ymax, zmin, zmax), in world coordinates
INNER_RIM = [(-1.275, -1.225, -0.675, 0.675, 0, RIM_TOP_Z), (1.225, 1.275, -0.675, 0.675, 0, RIM_TOP_Z),
             (-1.275, 1.275, -0.675, -0.625, 0, RIM_TOP_Z), (-1.275, 1.275, 0.625, 0.675, 0, RIM_TOP_Z)]
OUTER_RIM = [(-2.0, -1.95, -1.25, 1.25, 0, RIM_TOP_Z), (1.95, 2.0, -1.25, 1.25, 0, RIM_TOP_Z),
             (-2.0, 2.0, -1.25, -1.2, 0, RIM_TOP_Z), (-2.0, 2.0, 1.2, 1.25, 0, RIM_TOP_Z)]

TABLE_Z = 0.749                       # table origin height (legs on the floor)
FELT_Z = TABLE_Z + 0.0807             # playing surface
PLAY_AREA = (-1.065, 1.070, -0.440, 0.445)   # cushion noses (xmin, xmax, ymin, ymax)

TURTLE_HEIGHT = 0.141                 # top of the turtlebot (where a robot base sits)
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

def _Loop(half_x, half_y):
    """Corners (anticlockwise), side lengths and total length of a rectangular loop."""
    corners = [np.array(c) for c in [(half_x, -half_y), (half_x, half_y),
                                     (-half_x, half_y), (-half_x, -half_y)]]
    seg = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    return corners, seg, sum(seg)




def _Loop_S(xy, L):
    """Distance along loop L (anticlockwise from its first corner) of a point on it."""
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
    diff = (yaw_to - yaw_from + 180) % 360 - 180          # shortest way round
    n = max(1, int(abs(diff) / (turn_speed * dt)))
    return [yaw_from + diff * k / n for k in range(1, n + 1)]


def _Angle_Diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def Drive_Path(start, goal, dt, speed=0.35, turn_speed=90.0, loop=(CHANNEL_X, CHANNEL_Y),
               allow_reverse=False):
    """Turtlebot poses (x, y, yaw_deg) from start to goal around a rectangular loop.

    start, goal : (x, y, yaw_deg), both on the loop.
                  goal yaw may be None = keep facing the way it arrived.
    loop        : (half_x, half_y) of the loop - lane centre lines by default,
                  (INNER_X, INNER_Y) for a robot that has to reach into the table.
    allow_reverse : drive backwards instead of turning round. Turning on the spot
                  is then only needed at corners, where there is room - on the
                  inner loop a turn in the middle of a lane would clip the rim.
    The turtlebot turns to face where it is going, drives straight, turns on
    the spot at each corner, then turns to the goal's yaw at the end.
    """
    L = _Loop(*loop)
    total, seg = L[2], L[1]
    s0, s1 = _Loop_S(start[:2], L), _Loop_S(goal[:2], L)
    forward = (s1 - s0) % total
    sign = 1 if forward <= total / 2 else -1                 # shorter way round
    dist = forward if sign == 1 else total - forward

    # break the route at the corners it passes
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
            options = [heading, heading + 180]                   # forwards or backwards
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


# ------------------------------------POCKETS-----------------------------------------------------

# hole centres measured from Pool_Table.dae (table is centred on the origin)
POCKETS = [(1.104, 0.501), (1.104, -0.499), (-1.096, 0.501), (-1.096, -0.499),   # corners
           (0.004, 0.501), (0.004, -0.499)]                                     # middles
POCKET_RADIUS = 0.055                 # hole radius
POCKET_FLOOR_Z = TABLE_Z - 0.079      # bottom of the holes (a potted ball rests on this)
