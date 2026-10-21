"""
test_meca500_shaft_grip.py
--------------------------
Standalone test: Meca500 reaches forward, grips the cue shaft at its midpoint,
then lifts and holds the shaft horizontally.

No Doosan, no handle - just the Meca500 and the cue shaft.

Sequence
--------
1. Shaft lies horizontally in front of the Meca500 at rest height.
2. Meca500 moves from home to a hover pose (above shaft midpoint).
3. Meca500 descends to grip the shaft midpoint.
4. Meca500 lifts the shaft to a comfortable hold height.
5. Final hold pose.

APPROACH
--------
The IK solver cannot match an arbitrary end-effector orientation while also
reaching a specific position at Meca500's workspace limit.  Instead we use
direct joint-space configurations found by scanning the robot's FK workspace
over a dense grid.  The shaft mesh is placed to match the end-effector position
at each step so the shaft always appears attached to the gripper.
"""
import time
import swift
import numpy as np
from math import pi
from spatialmath import SE3
import roboticstoolbox as rtb

from meca500_robot import Meca500_Robot
from Pool_Parts import Part_Path
from Pool_Objects import Cue_Stick


# ──────────────────────────── CONSTANTS ──────────────────────────────────────

DT = 0.02

MECA_BASE = SE3(1.5, 0.0, 0.0) *SE3.Rz(pi/2)

# These joint configs were found by scanning the Meca500 FK workspace.
# Each one places the end-effector at x≈0.25 m in front of the base,
# at the stated Z height, with |Y| < 0.03 m.

# Home: safe upright pose
Q_HOME = np.array([0.0,  0.5,  0.3,  0.0, -0.8,  0.0])

# Hover: end-effector above the shaft midpoint (z ≈ 0.20 m)
Q_HOVER = np.array([0.0,  0.4963, -0.1726, 0.0,  0.5236, 0.0])

# Grip: end-effector at the shaft midpoint height (z ≈ 0.08 m)
Q_GRIP = np.array([0.0,  0.9545,  0.0,    0.0,  0.0,    0.0])

# Hold: end-effector at the final lifted height (z ≈ 0.22 m)
Q_HOLD = np.array([0.0,  0.6272, -0.604,  0.0,  1.0472, 0.0])

Q_LEFT  = np.array([pi / 2.0,  0.6272, -0.604,  0.0,  1.0472, 0.0])

# ──────────────────────────── SHAFT PLACEMENT ────────────────────────────────

def shaft_pose_from_ee(T_ee, shaft_length):
    """Return the shaft world SE3 so its midpoint sits at the end-effector.

    The shaft lies horizontally.  We place the shaft tip (cue frame origin)
    such that the midpoint (half-way along the shaft's +Z axis) coincides
    with the end-effector position.

    The shaft frame: origin = tip, +Z along the shaft.
    For a horizontal shaft along world +X: apply SE3.Ry(pi/2) so local +Z
    maps to world +X.  Then step -L/2 along world +X to get the tip.
    """
   # ee_pos = T_ee.t.copy()
    
   #return SE3(*tip_pos) * SE3.Rx(-pi / 2)
    
    ee_pos = T_ee.t.copy()

    tip_pos = ee_pos + np.array([0.5, 0.0, 0.0])

    return SE3(*tip_pos) * SE3.Ry(pi / 2)

# ──────────────────────────── MAIN ───────────────────────────────────────────

def main():
    env = swift.Swift()
    env.launch(realtime=True)

    # ── Meca500 ───────────────────────────────────────────────────────────────
    meca = Meca500_Robot()
    meca.base = MECA_BASE
    meca.add_to_env(env)

    # ── Cue shaft mesh ────────────────────────────────────────────────────────
    cue = Cue_Stick(
        str(Part_Path("Cuestick_Shaft")),
        str(Part_Path("Cuestick_Handle")),
    )
    cue.Add_To_Env(env)
    L = cue.shaft_length
    print(f"[cue] shaft length = {L:.3f} m")

    # ── Verify FK positions for our key configs ────────────────────────────────
    meca.q = Q_HOME
    for name, q in [("HOME ", Q_HOME), ("HOVER", Q_HOVER),
                    ("GRIP ", Q_GRIP),  ("HOLD ", Q_HOLD)]:
        T = meca.fkine(q)
        print(f"  {name}: ee pos = {np.round(T.t, 3)}")

    # ── Initial shaft: resting below the hover position ────────────────────────
    # Start with shaft at grip height so it visually makes sense
    T_grip_ee    = meca.fkine(Q_GRIP)
    shaft_resting = shaft_pose_from_ee(T_grip_ee, L)

    # ── Helper: render one frame ───────────────────────────────────────────────
    def update(q, shaft_world_pose):
        meca.q = q
        # shaft_offset=None -> shaft is placed with tip at shaft_w origin
        #cue.Place(shaft_w, shaft_offset=None)
        cue.Place(shaft_world_pose, shaft_offset=None)
        env.step(DT)
        time.sleep(DT)

    # Initial state
    meca.q = Q_HOME
    update(Q_HOME, shaft_resting)

    # Camera: slightly above and to the side
    env.set_camera_pose([0.9, -0.5, 0.45], [0.25, 0.0, 0.15])

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 – Meca500 moves from home to hover above the shaft
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[1] Moving to hover position above shaft midpoint...")
    for q in rtb.jtraj(Q_HOME, Q_HOVER, 80).q:
        update(q, shaft_resting)          # shaft stays still while robot approaches

    # Pause at hover
    for _ in range(25):
        update(Q_HOVER, shaft_resting)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 – Descend to grip the shaft at its midpoint
    # ══════════════════════════════════════════════════════════════════════════
    print("[2] Descending to grip the shaft...")
    for q in rtb.jtraj(Q_HOVER, Q_GRIP, 50).q:
        update(q, shaft_resting)          # shaft still static

    # Pause to show grip
    print("[2] Gripped!")
    for _ in range(30):
        update(Q_GRIP, shaft_resting)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3 – Lift the shaft (shaft moves with the end-effector)
    # ══════════════════════════════════════════════════════════════════════════
    print("[3] Lifting shaft to hold height...")
    lift_traj = rtb.jtraj(Q_GRIP, Q_HOLD, 100).q
    for q in lift_traj:
        # Shaft midpoint tracks the end-effector position each frame
        T_ee  = meca.fkine(q)
        shaft_w = shaft_pose_from_ee(T_ee, L)
        update(q, shaft_w)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 4 – Final hold
    # ══════════════════════════════════════════════════════════════════════════
    T_hold_ee = meca.fkine(Q_HOLD)
    shaft_held = shaft_pose_from_ee(T_hold_ee, L)
    print("[4] Meca500 holding shaft horizontally.")
    for _ in range(50):
        update(Q_HOLD, shaft_held)

    env.hold()
   
   # 4. Hold briefly before moving left
   # for _ in range(20):
    #    update(Q_HOLD, shaft_pose_from_ee(T_ee, L))

    # 5. Swing shaft horizontally to the left (+Y direction)
   # print("[5] Moving shaft horizontally to the left...")
   # for q in rtb.jtraj(Q_HOLD, Q_LEFT, 100).q:
     #   T_ee = meca.fkine(q)
     #   shaft_w = shaft_pose_from_ee(T_ee, L)
    #    update(q, shaft_w)

    # Final hold
   # T_left_ee = meca.fkine(Q_LEFT)
   # shaft_left = shaft_pose_from_ee(T_left_ee, L)
    #print("[6] Sequence complete. Holding position on the left.")
    #for _ in range(50):
    #    update(Q_LEFT, shaft_left)

    env.hold()




if __name__ == "__main__":
    main()
