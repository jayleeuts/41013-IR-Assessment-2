"""
Test_UR3_Shot - the Robotics Toolbox UR3 on a turtlebot, with the same Cue_Holder
and cue as the Doosan, pots a ball into a corner pocket (real ball physics).

  1. UR3 drives round the track to the side of the table.
  2. Lines up behind the cue ball (comes down onto the shot line from above).
  3. Rocks, strikes; the cue ball hits the object ball, which goes in the pocket.
"""
import swift
import numpy as np
from spatialmath import SE3
import roboticstoolbox as rtb
import spatialgeometry as sg

from Pool_Parts import Part_Path, Part_Mesh
from Pool_Objects import Cue_Ball
from Pool_Motion import Cue_Pose, Punch_Stroke_Profile
from Pool_Scene import (TRACK_NAME, TRACK_POSE, TABLE_Z, FELT_Z, CHANNEL_X, CHANNEL_Y, PLAY_AREA,
                        POCKETS, POCKET_RADIUS, POCKET_FLOOR_Z, Drive_Path)
from Pool_Trick_Shots import Pool_Ball, Table_Step, Cue_Dir, Contact_Point
from Pool_Players import UR3_Player

DT, SUBSTEPS = 0.02, 5
CUE_TO_BALL, ADDRESS_GAP, APPROACH, LIFT = 1.4, 0.02, 0.15, 0.15
CUE_XY, OBJECT_XY, POCKET = (0.30, -0.15), (0.80, 0.22), 0        # pocket 0 = (1.104, 0.501)
TIP_SPEED = 0.9                      # UR3 is small - a softer stroke keeps its joints under their limits

env = swift.Swift()
env.launch(realtime=True)
env.add(Part_Mesh(TRACK_NAME, pose=TRACK_POSE))
env.add(Part_Mesh("Pool_Table", pose=SE3(0, 0, TABLE_Z)))
env.set_camera_pose([-1.2, -2.6, 2.2], [0.0, 0.0, 0.8])

ur3 = UR3_Player((-CHANNEL_X, 0.0, -90.0))
ur3.Add_To_Env(env)
ur3.q = ur3.IK(ur3.Carry_Tip(), ur3.q)
mesh0 = Cue_Ball(str(Part_Path("Cue_Ball")))
R = mesh0.radius
mesh0.Add_To_Env(env)
mesh1 = sg.Sphere(radius=R, color=[1.0, 0.85, 0.0, 1.0])
env.add(mesh1)
balls = [Pool_Ball([*xy, FELT_Z + R], R, jaws=True, bounds=PLAY_AREA, pockets=POCKETS, pocket_radius=POCKET_RADIUS,
                   pocket_floor_z=POCKET_FLOOR_Z + R) for xy in (CUE_XY, OBJECT_XY)]
q_log = []


def Frame(q=None):
    if q is not None:
        ur3.q = q
        q_log.append(np.array(q))
    ur3.Set_Base()
    ur3.Draw()
    for _ in range(SUBSTEPS):
        Table_Step(balls, DT / SUBSTEPS)
    mesh0.Place(balls[0].pos, balls[0].R)
    mesh1.T = SE3(*balls[1].pos).A
    env.step(DT)


def Play(qs):
    """Play a list of joint angles, adding frames wherever a joint would go over its speed limit."""
    q_prev = ur3.q.copy()
    for q in qs:
        n = max(1, int(np.ceil(np.max(np.abs(q - q_prev) / (0.9 * ur3.VMAX * DT)))))
        for t in np.linspace(0, 1, n + 1)[1:]:
            Frame(q_prev + t * (q - q_prev))
        q_prev = q


def Line(T_from, T_to, steps=40):
    """Cue tip in a straight line (IK every step), speed-limited."""
    qs, q = [], ur3.q
    for s in rtb.quintic(0, 1, steps).q:
        q_new = ur3.IK(T_from.interp(T_to, s), q)
        q = q if q_new is None else q_new
        qs.append(q)
    Play(qs)


# ghost ball: where the cue ball must be when it touches the object ball
u = np.array(POCKETS[POCKET]) - np.array(OBJECT_XY)
u /= np.linalg.norm(u)
ghost = np.array(OBJECT_XY) - 2 * R * u
aim = np.degrees(np.arctan2(*(ghost - np.array(CUE_XY))[::-1]))
# park on the track straight behind the shot line: the stroke then points straight
# out from the robot, so the UR3's base joint hardly has to turn while it strikes
back = -np.array([np.cos(np.radians(aim)), np.sin(np.radians(aim))])
k = (-CHANNEL_Y - CUE_XY[1]) / back[1]
SHOT_TURTLE = (CUE_XY[0] + k * back[0], -CHANNEL_Y, 0.0)

d = Cue_Dir(aim, 8.0)
contact = Contact_Point(d, 0.0, 0.0, R)
profile = Punch_Stroke_Profile(DT, contact_s=-ADDRESS_GAP, tip_speed=TIP_SPEED, back=0.15, follow=0.03)


def Stroke_Joints(T, q):
    """Joint angles through the whole stroke, or None if the arm can't follow it."""
    qs = []
    for s in list(np.linspace(APPROACH, 0, 8)) + list(profile):
        q_new = ur3.IK(T * SE3.Tz(-s), q)
        if q_new is None or np.max(np.abs(q_new - q)) > 0.5:
            return None
        qs.append(q_new)
        q = q_new
    return qs


# try turtle spots along the lane, each way of holding the cue (roll) and arm shape;
# keep the one with the slowest joints during the stroke
best = None
for dx in (-0.45, -0.3, -0.15, 0.0, 0.15, 0.3):
  ur3.xyyaw = (SHOT_TURTLE[0] + dx, -CHANNEL_Y, 0.0)
  ur3.Set_Base()
  for roll in range(-180, 180, 45):
    T = Cue_Pose(balls[0].pos + contact - ADDRESS_GAP * d, aim, 8.0, roll)
    for seed in [None] + [np.radians([a, b, c, -90, e, 0]) for a in (0, 90) for b, c in ((-60, 80), (-120, 100))
                          for e in (-90,)]:
        q0 = ur3.IK(T * SE3.Tz(-APPROACH), seed)
        if q0 is None:
            continue
        qs = Stroke_Joints(T, q0)
        if qs is None:
            continue
        ratio = np.max(np.abs(np.diff(qs[8:], axis=0)) / DT / ur3.VMAX)     # the stroke itself
        if best is None or ratio < best[0]:
            best = (ratio, T, qs[0], ur3.xyyaw)
T_addr = best[1]
SHOT_TURTLE = best[3]
print(f"    best: turtle {np.round(SHOT_TURTLE, 2)}, stroke joints at {best[0] * 100:.0f}% of their limits")
ur3.xyyaw = (-CHANNEL_X, 0.0, -90.0)
ur3.Set_Base()
print("[1] driving")
for pose in Drive_Path(ur3.xyyaw, SHOT_TURTLE, DT):
    ur3.xyyaw = pose
    Frame()

print(f"[2] lining up, aim {aim:.1f} deg")
T_ready = T_addr * SE3.Tz(-APPROACH)
T_above = SE3(0, 0, LIFT) * T_ready
q_above = ur3.IK(T_above, best[2])
Play(rtb.jtraj(ur3.q, q_above, 80).q)
Line(T_above, T_ready, 30)
Line(T_ready, T_addr, 40)

print("[3] strike")
hit_at = int(np.argmax(profile <= -ADDRESS_GAP + 1e-9))
q = ur3.q
for i, s in enumerate(profile):
    q_new = ur3.IK(T_addr * SE3.Tz(-s), q)
    q = q if q_new is None else q_new
    if i == hit_at:
        balls[0].Hit(CUE_TO_BALL * TIP_SPEED * d, contact)
    Frame(q)
T_back = T_addr * SE3.Tz(-APPROACH)
Line(ur3.Tip(), T_back, 30)
Line(T_back, SE3(0, 0, LIFT) * T_back, 30)
Play(rtb.jtraj(ur3.q, ur3.IK(ur3.Carry_Tip(), ur3.q), 80).q)
while any(b.Moving() for b in balls):
    Frame()

v = np.abs(np.diff(np.array(q_log), axis=0)) / DT
print(f"    object ball {'POTTED' if balls[1].potted == POCKET else 'missed'}; "
      f"fastest joint {np.max(v / ur3.VMAX) * 100:.0f}% of its limit")
env.hold()
