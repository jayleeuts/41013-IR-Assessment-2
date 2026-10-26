"""
Pool_Arm.py - how the shooting arms move their cue, the same way for every shot.

  * ONE ARM SHAPE per robot (ARM_SHAPE). The base turns to face the shot, the other five
    joints start from the same elbow-up shape and the cue is always held at the same roll
    (CUE_ROLL). Inverse kinematics is solved from that one seed only, and the answer must
    stay in that shape - if it can't, the robot parks somewhere else on the track instead
    of trying other arm shapes.
  * RMRC (resolved-rate motion control) for every straight cue move - lining up and the
    stroke itself: each frame the joint speeds come from the cue-tip Jacobian,
        q_dot = J^+ (v_wanted + K * pose_error)        (damped least squares)
    so the tip follows the straight line at the speed asked for, without solving IK again.
  * COLLISION CHECK with real thickness: every link is a capsule (a tube with round ends,
    LINK_RADII), the cue is a 15 mm-radius tube. The cue must stay at least CLEARANCE clear
    of the robot's own links (the flange the holder bolts to is left out - the cue rests on it).
"""
import numpy as np
from spatialmath import SE3
from spatialmath.base import trlog

CUE_RADIUS = 0.015
CLEARANCE = 0.01                                    # 1 cm of air between the cue and any link surface

# link capsules: radius of each segment between neighbouring joint frames (flange segment left out)
LINK_RADII = {"D": [0.10, 0.075, 0.075, 0.06, 0.055],                     # Doosan M0609
              "U": [0.0, 0.06, 0.05, 0.045, 0.045, 0.04, 0.035]}          # UR3 (URDF frames, first 9)


def Link_Points(kind, robot, q):
    P = np.array([T.t for T in robot.fkine_all(q)])
    return P[:9] if kind == "U" else P


def Seg_Dist(p1, q1, p2, q2):
    """Shortest distance between line segments p1-q1 and p2-q2."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
    t = (b * s + f) / e if e > 1e-12 else 0.0
    if t < 0:
        t, s = 0.0, np.clip(-c / a, 0, 1)
    elif t > 1:
        t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def Cue_Gap(kind, robot, q, T_tip, cue_length):
    """Air gap (m) between the cue's surface and the nearest of the robot's own link surfaces."""
    if kind not in LINK_RADII:
        return 1.0
    P = Link_Points(kind, robot, q)
    a, b = T_tip.t, (T_tip * SE3.Tz(-cue_length)).t
    gap = 9.0
    for j, r in enumerate(LINK_RADII[kind]):
        if r <= 0 or np.linalg.norm(P[j + 1] - P[j]) < 1e-6:
            continue
        gap = min(gap, Seg_Dist(a, b, P[j], P[j + 1]) - r - CUE_RADIUS)
    return gap


# ------------------------------------RMRC-----------------------------------------------------

def Pose_Error(T_now, T_goal):
    """6-vector (position, rotation) from T_now to T_goal, in the world frame."""
    dp = T_goal.t - T_now.t
    dR = T_goal.R @ T_now.R.T
    w = trlog(dR, twist=True) if not np.allclose(dR, np.eye(3)) else np.zeros(3)
    return np.r_[dp, w]


def Tip_Jacobian(tip, q, h=1e-6):
    """6x6 Jacobian of the cue-tip pose (world frame) by small joint nudges."""
    T0 = tip(q)
    J = np.zeros((6, len(q)))
    for j in range(len(q)):
        dq = np.zeros(len(q))
        dq[j] = h
        J[:, j] = Pose_Error(T0, tip(q + dq)) / h
    return J


def RMRC(tip, q0, poses, dt, gain=0.6, damping=0.002, substeps=4):
    """Joint angles that move the cue tip through `poses` (one per frame) by resolved-rate
    control (each frame is split into a few control steps). Returns (joint list, worst tip
    error in m)."""
    q = np.array(q0, dtype=float)
    qs, worst = [], 0.0
    h = dt / substeps
    for k in range(1, len(poses)):
        for j in range(substeps):
            T_a = poses[k - 1].interp(poses[k], j / substeps)
            T_b = poses[k - 1].interp(poses[k], (j + 1) / substeps)
            v = Pose_Error(T_a, T_b) / h                               # feed-forward: the line's own speed
            e = Pose_Error(tip(q), T_a)                                 # feedback: pull back onto the line
            J = Tip_Jacobian(tip, q)
            q_dot = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), v + gain * e / h)
            q = q + q_dot * h
        qs.append(q.copy())
        worst = max(worst, float(np.linalg.norm(tip(q).t - poses[k].t)))
    return qs, worst


# ------------------------------------THE ONE ARM SHAPE-----------------------------------------------------
# joints 2..6 of the elbow-up shape each robot always uses, and the cue roll it always holds.
# joint 1 (base) is turned to face the shot.  (Chosen by Make_Arm_Plan.py: the shape that keeps
# the cue clear of the arm for every shot of the game.)

ARM_SHAPE = {"D": [-1.6254, 0.6921, 1.7176, -0.5068, 2.2366, 1.38],
             "U": [-0.8396, -1.6232, 2.3283, -2.1943, -1.6842, -2.5158]}
CUE_ROLL = {"D": -90.0, "U": 0.0}


def Facing(robot, T_world):
    """Base angle that points the robot at a world position."""
    p = robot.base.inv() * T_world.t
    return float(np.arctan2(p[1], p[0]))


def Shape_Seed(kind, robot, T_world):
    rel = np.array(ARM_SHAPE[kind], dtype=float)
    return np.r_[Facing(robot, T_world) + rel[0], rel[1:]]


def Same_Shape(q, seed):
    """Is q the same arm configuration as the seed? The three things that make an arm
    'configuration' (the IK branch): shoulder facing the same way (joint 1 within 90 deg),
    elbow up/down (sign of sin q3) and wrist flip (sign of sin q5)."""
    q, seed = np.asarray(q), np.asarray(seed)
    d1 = (q[0] - seed[0] + np.pi) % (2 * np.pi) - np.pi
    return bool(abs(d1) < np.pi / 2 and np.sign(np.sin(q[2])) == np.sign(np.sin(seed[2]))
                and np.sign(np.sin(q[4])) == np.sign(np.sin(seed[4])))


# ------------------------------------PATH PLANNING (RRT-Connect)-----------------------------------------------------
# For moves between two collision-free arm poses where no simple joint move stays clear
# (e.g. carrying the cue upright -> lining up a shot behind the robot): grow two random
# trees of collision-free joint configurations, one from each end, until they meet, then
# shortcut the path. The random numbers are seeded from the start/goal, so the same move
# always gives the same path (the time-skip snapshots stay exact).

def Edge_Free(a, b, free, res=0.05):
    n = max(1, int(np.ceil(np.max(np.abs(b - a)) / res)))
    return all(free(a + (b - a) * t) for t in np.linspace(0, 1, n + 1)[1:])


def RRT_Connect(q0, q1, free, step=0.25, iters=2500, span=1.5):
    """Joint-space path q0 -> q1 (list of waypoints) through configurations where free(q) is
    True, or None if none was found."""
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    rng = np.random.default_rng(int(abs(np.sum(q0 * 1000 + q1 * 7)) * 1000) % (2 ** 32))
    lo, hi = np.minimum(q0, q1) - span, np.maximum(q0, q1) + span
    trees = [([q0], [-1]), ([q1], [-1])]

    def Nearest(nodes, q):
        return int(np.argmin([np.linalg.norm(n - q) for n in nodes]))

    def Extend(tree, q):
        nodes, parent = tree
        i = Nearest(nodes, q)
        d = q - nodes[i]
        dist = np.linalg.norm(d)
        q_new = q if dist <= step else nodes[i] + d * step / dist
        if not Edge_Free(nodes[i], q_new, free):
            return None
        nodes.append(q_new)
        parent.append(i)
        return len(nodes) - 1

    def Branch(tree, i):
        nodes, parent = tree
        out = []
        while i != -1:
            out.append(nodes[i])
            i = parent[i]
        return out

    for it in range(iters):
        a, b = trees[it % 2], trees[(it + 1) % 2]
        q_rand = rng.uniform(lo, hi)
        i_new = Extend(a, q_rand)
        if i_new is None:
            continue
        target = a[0][i_new]
        while True:                                              # connect the other tree to it
            j = Extend(b, target)
            if j is None:
                break
            if np.linalg.norm(b[0][j] - target) < 1e-9:
                pa, pb = Branch(a, i_new), Branch(b, j)
                path = pa[::-1] + pb[1:] if it % 2 == 0 else pb[::-1] + pa[1:]
                return Shortcut(path, free, rng)
    return None


def Shortcut(path, free, rng, tries=80):
    path = list(path)
    for _ in range(tries):
        if len(path) <= 2:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i > 1 and Edge_Free(path[i], path[j], free):
            path = path[:i + 1] + path[j:]
    return path
