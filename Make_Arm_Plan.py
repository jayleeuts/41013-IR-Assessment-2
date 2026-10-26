"""
Make_Arm_Plan.py - picks ONE arm shape per robot (Doosan, UR3) that works for every shot of
the game, and where each robot parks for each shot. Run offline:  python Make_Arm_Plan.py
(about 10-20 minutes). It rewrites the turtle / roll / q_ready of each shot in Pool_Game.py
and prints the ARM_SHAPE / CUE_ROLL lines to paste into Pool_Arm.py.

For every shot and every candidate spot on the robot's lane:
  1. IK for the ready pose from the ONE seed (base turned to face the shot + the fixed shape)
     - the answer must still be that shape;
  2. RMRC: lift above -> ready -> address -> the whole stroke, all straight cue-tip lines;
  3. the cue must stay CLEARANCE clear of the robot's own links (capsules) the whole time,
     and the joints under their speed limits (the break is allowed over - see the notes).
"""
import sys
import json
import numpy as np
from math import pi
from spatialmath import SE3

from Pool_Players import Doosan_Player, UR3_Player
from Pool_Motion import Punch_Stroke_Profile
from Pool_Scene import FELT_Z, CHANNEL_Y, _Loop, _Loop_Point
from Pool_Trick_Shots import Shot_Poses
import Pool_Arm as PA
from Pool_Game import GAME

R, DT, ADDRESS_GAP, APPROACH, LIFT = 0.0286, 0.02, 0.02, 0.15, 0.15
LOOP = _Loop(1.72, CHANNEL_Y)


def Spots(ball, d2, step=0.05):
    """Spots on the lane behind the shot, best first: about 30 cm to the side of the shot
    line, so the back of the cue passes BESIDE the robot rather than over its base."""
    out = []
    for s in np.arange(0, LOOP[2], step):
        p = _Loop_Point(s, LOOP)
        rel = p - ball
        if -rel @ d2 < 0.15 or np.linalg.norm(rel) > 1.9:
            continue
        side = abs(rel[0] * d2[1] - rel[1] * d2[0])
        yaw = 0.0 if abs(abs(p[1]) - LOOP[0][1][1]) < 1e-6 else 90.0
        out.append((abs(side - 0.30), (float(p[0]), float(p[1]), yaw)))
    return [s for _, s in sorted(out)][:36]


class Shooter:
    def __init__(self, kind):
        self.kind = kind
        self.p = (Doosan_Player if kind == "D" else UR3_Player)((0, 0, 0))

    def IK1(self, T_tip, seed):
        """IK from ONE seed only."""
        p = self.p
        if self.kind == "D":
            sol = p.robot.ikine_LM(T_tip * p.tool.inv(), q0=seed, joint_limits=False, slimit=1)
        else:
            T_local = p.robot.base.inv() * T_tip * p.tool.inv()
            sol = p.kin.ikine_LM(T_local, q0=seed, end=p.END, joint_limits=True, slimit=1)
        return sol.q if sol.success else None

    def Gap(self, q):
        return PA.Cue_Gap(self.kind, self.p.robot, q, self.p.Tip(q), self.p.cue.length)

    def Try(self, shot, spot, shape, roll):
        p, k = self.p, self.kind
        p.xyyaw = spot
        p.Set_Base()
        sh = dict(shot, roll=roll)
        ball = np.array([*shot["layout"][0], FELT_Z + R])
        d, contact, T_addr = Shot_Poses(ball, sh, R, ADDRESS_GAP)
        T_ready = T_addr * SE3.Tz(-APPROACH)
        seed = np.r_[PA.Facing(p.robot, T_ready) + shape[0], shape[1:]]
        q_ready = self.IK1(T_ready, seed)
        if q_ready is None or not PA.Same_Shape(q_ready, seed) or self.Gap(q_ready) < PA.CLEARANCE:
            return None
        q_addr = self.IK1(T_addr, q_ready)
        if q_addr is None or self.Gap(q_addr) < PA.CLEARANCE:
            return None
        prof = Punch_Stroke_Profile(DT, contact_s=-ADDRESS_GAP, tip_speed=shot["speed"],
                                    back=0.20 if k == "D" else 0.15, follow=0.03)
        line = [T_ready.interp(T_addr, s) for s in np.linspace(0, 1, 30)]
        stroke = [T_addr] + [T_addr * SE3.Tz(-s) for s in prof]
        qs1, e1 = PA.RMRC(p.Tip, q_ready, line, DT)
        if e1 > 0.003:
            return None
        qs2, e2 = PA.RMRC(p.Tip, qs1[-1], stroke, DT)
        if e2 > 0.003:
            return None
        gap = min(self.Gap(q) for q in (qs1 + qs2)[::2])
        if gap < PA.CLEARANCE:
            return None
        ratio = float(np.max(np.abs(np.diff(np.array(qs2), axis=0)) / DT / p.VMAX))
        return dict(turtle=list(spot), roll=roll, q_ready=[float(x) for x in q_ready], gap=gap, ratio=ratio)

    def Best(self, shot, shape, roll):
        ball = np.array(shot["layout"][0])
        a = np.radians(shot["aim"])
        d2 = np.array([np.cos(a), np.sin(a)])
        best = None
        for spot in Spots(ball, d2):
            r = self.Try(shot, spot, shape, roll)
            if r and (best is None or r["ratio"] < best["ratio"]):
                best = r
                if r["ratio"] < 0.9:
                    break
        return best


def Candidates(kind):
    """Arm shapes to try: the shapes the robot already used in each shot (joint 1 measured
    from 'facing the shot'), and the roll it held the cue at."""
    sh = Shooter(kind)
    out = []
    for shot in GAME:
        if shot["who"] != kind:
            continue
        sh.p.xyyaw = shot["turtle"]
        sh.p.Set_Base()
        ball = np.array([*shot["layout"][0], FELT_Z + R])
        _, _, T_addr = Shot_Poses(ball, shot, R, ADDRESS_GAP)
        q = np.array(shot["q_ready"])
        rel = np.r_[(q[0] - PA.Facing(sh.p.robot, T_addr * SE3.Tz(-APPROACH)) + pi) % (2 * pi) - pi, q[1:]]
        out.append((rel, shot["roll"]))
    if kind == "U":                                   # the UR3 is small - also try each shape at every cue roll
        out += [(rel, float(r)) for rel, _ in list(out) for r in range(-180, 180, 45)]
    return out


if __name__ == "__main__":
    kinds = sys.argv[1:] or ["D", "U"]
    result = {}
    for kind in kinds:
        sh = Shooter(kind)
        shots = [i for i, s in enumerate(GAME) if s["who"] == kind]
        best = None
        for shape, roll in Candidates(kind):
            plan, ok = {}, True
            for i in shots:
                r = sh.Best(GAME[i], shape, roll)
                if r is None:
                    ok = False
                    break
                plan[i] = r
            print(kind, "shape", np.round(shape, 3).tolist(), "roll", roll, "->", "ALL SHOTS OK" if ok else f"fails shot {i + 1}", flush=True)
            if ok:
                score = min(r["gap"] for r in plan.values())
                if best is None or score > best[0]:
                    best = (score, shape, roll, plan)
        if best is None:
            print(kind, "no single shape works for every shot")
            continue
        result[kind] = dict(shape=[float(x) for x in best[1]], roll=best[2], plan={str(i): r for i, r in best[3].items()})
        print(kind, "CHOSEN shape", np.round(best[1], 4).tolist(), "roll", best[2], "min gap", round(best[0] * 100, 1), "cm", flush=True)
    json.dump(result, open("arm_plan.json", "w"), indent=1)
