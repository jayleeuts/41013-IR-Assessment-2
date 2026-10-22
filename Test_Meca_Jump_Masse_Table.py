"""
Test_Meca_Jump_Masse_Table - Shuban's Meca500 (scaled, on a turtlebot) on the real table.

  1. JUMP  : cue ball jumps over a blocker ball sitting on the shot line.
  2. MASSE : steep off-centre hit - the cue ball curves round a blocker ball.

Uses his jump cue, gripper and short stroke (Test_Meca500_v1_Jump_Over.py) and his
jump physics (in Pool_Trick_Shots.Pool_Ball). Runs on its own, no keys.
"""
import swift
import numpy as np
from spatialmath import SE3
import roboticstoolbox as rtb

from Pool_Parts import Part_Path, Part_Mesh
from Pool_Objects import Cue_Ball
from Pool_Motion import Cue_Pose
from Pool_Scene import TRACK_NAME, TRACK_POSE, TABLE_Z, FELT_Z, INNER_Y, PLAY_AREA, POCKETS, POCKET_RADIUS, POCKET_FLOOR_Z
from Pool_Trick_Shots import Pool_Ball, Table_Step, Cue_Dir, Contact_Point
from Meca_Jumper import Meca_Mobile, Short_Stroke_Profile

DT = 0.02
SUBSTEPS = 10
CUE_TO_BALL = 1.4
ADDRESS_GAP = 0.02

# (name, cue ball xy, aim, elevation, tip speed, side, blocker xy)
SHOTS = [("JUMP", (-0.30, -0.33), 60.0, 48.0, 2.4, 0.0, (-0.17, -0.105)),
         ("MASSE", (0.30, -0.33), 60.0, 48.0, 1.4, -0.6, (0.452, -0.187))]   # blocker just right of the line: the ball bends left, away from it

env = swift.Swift()
env.launch(realtime=True)
env.add(Part_Mesh(TRACK_NAME, pose=TRACK_POSE))
env.add(Part_Mesh("Pool_Table", pose=SE3(0, 0, TABLE_Z)))
env.set_camera_pose([0.0, -2.2, 1.9], [0.0, -0.2, 0.85])

meca = Meca_Mobile((-0.30, -INNER_Y, 0.0))
meca.Add_To_Env(env)
cue_mesh = Cue_Ball(str(Part_Path("Cue_Ball")))
R = cue_mesh.radius
cue_mesh.Add_To_Env(env)
import spatialgeometry as sg
blocker_mesh = sg.Sphere(radius=R, color=[0.85, 0.0, 0.0, 1.0])
env.add(blocker_mesh)


def Ball(xy):
    return Pool_Ball([xy[0], xy[1], FELT_Z + R], R, bounds=PLAY_AREA, pockets=POCKETS,
                     pocket_radius=POCKET_RADIUS, pocket_floor_z=POCKET_FLOOR_Z + R)


def Show(balls, q=None):
    if q is not None:
        meca.robot.q = q
    meca.Set_Base()
    meca.Draw_Tool()
    cue_mesh.Place(balls[0].pos, balls[0].R)
    blocker_mesh.T = SE3(*balls[1].pos).A
    env.step(DT)


for name, xy, aim, elev, speed, side, block in SHOTS:
    print(f"--- {name} ---")
    balls = [Ball(xy), Ball(block)]
    meca.xyyaw = (xy[0], -INNER_Y, 0.0)
    meca.Set_Base()
    d = Cue_Dir(aim, elev)
    contact = Contact_Point(d, side, 0.0, R)
    T_addr = None
    for roll in range(-180, 180, 45):
        T = Cue_Pose(np.array([*xy, FELT_Z + R]) + contact - ADDRESS_GAP * d, aim, elev, roll)
        q_addr = meca.IK(T)
        if q_addr is not None:
            T_addr = T
            break
    if T_addr is None:
        print("    can't reach this shot")
        continue
    lift, q_above = 0.0, q_addr
    for h in (0.12, 0.08, 0.04):                     # come down onto the shot from above if it can
        q_try = meca.IK(SE3(0, 0, h) * T_addr, q_addr)
        if q_try is not None:
            lift, q_above = h, q_try
            break
    for q in rtb.jtraj(meca.robot.q, q_above, 60).q:
        Show(balls, q)
    q = q_above
    for s in rtb.quintic(0, 1, 30).q:
        q_new = meca.IK(SE3(0, 0, lift * (1 - s)) * T_addr, q)
        q = q if q_new is None else q_new
        Show(balls, q)
    profile = Short_Stroke_Profile(DT, contact_s=-ADDRESS_GAP, tip_speed=speed, follow=0.015, backswing=0.05)
    hit = False
    for s in profile:
        q_new = meca.IK(T_addr * SE3.Tz(-s), q)
        q = q if q_new is None else q_new
        if not hit and s <= -ADDRESS_GAP + 1e-9:
            balls[0].Hit(CUE_TO_BALL * speed * d, contact)
            hit = True
        for _ in range(SUBSTEPS):
            Table_Step(balls, DT / SUBSTEPS)
        Show(balls, q)
    top = 0.0
    while any(b.Moving() for b in balls):
        for _ in range(SUBSTEPS):
            Table_Step(balls, DT / SUBSTEPS)
        top = max(top, balls[0].pos[2] - balls[0].floor_z)
        Show(balls)
    print(f"    cue ball ended at {np.round(balls[0].pos[:2], 3)}, blocker {'moved' if np.linalg.norm(balls[1].pos[:2] - block) > 1e-3 else 'untouched'}")

print("done")
env.hold()
