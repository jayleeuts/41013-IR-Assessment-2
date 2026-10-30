# ------------------------------------POOL OBJECTS------------------------------------------------------------
# The cue ball and the two-part cue stick built from our own .dae models. The .dae file is read directly to
# measure each part (length, radius, which end is which), so the meshes can be placed exactly where the
# simulation needs them.

import xml.etree.ElementTree as ET
import numpy as np
import spatialgeometry as sg
from spatialmath import SE3


# ------------------------------------READING A .DAE----------------------------------------------------------
# Collects every vertex in the model with its node transforms applied, in metres, then guesses millimetres
# versus metres from the size and measures a rod-shaped part along its long axis.

def _strip_ns(root):
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _node_matrix(node):
    M = np.eye(4)
    for child in node:
        vals = np.array((child.text or "").split(), dtype=float)
        if child.tag == "matrix" and vals.size == 16:
            M = M @ vals.reshape(4, 4)
        elif child.tag == "translate" and vals.size == 3:
            M = M @ SE3(*vals).A
        elif child.tag == "rotate" and vals.size == 4:
            M = M @ SE3.AngleAxis(np.deg2rad(vals[3]), vals[:3]).A
        elif child.tag == "scale" and vals.size == 3:
            M = M @ np.diag([*vals, 1.0])
    return M


def Dae_Vertices(path):
    root = _strip_ns(ET.parse(path).getroot())

    unit = root.find("./asset/unit")
    metres_per_unit = float(unit.get("meter", 1.0)) if unit is not None else 1.0

    geoms = {}
    for g in root.iter("geometry"):
        mesh = g.find("mesh")
        if mesh is None:
            continue
        sources = {s.get("id"): s for s in mesh.findall("source")}
        pos_input = mesh.find("./vertices/input[@semantic='POSITION']")
        if pos_input is None:
            continue
        src = sources.get(pos_input.get("source").lstrip("#"))
        arr = np.array(src.find("float_array").text.split(), dtype=float)
        geoms[g.get("id")] = arr.reshape(-1, 3)

    out = []

    def walk(node, M_parent):
        M = M_parent @ _node_matrix(node)
        for ig in node.findall("instance_geometry"):
            v = geoms.get(ig.get("url", "").lstrip("#"))
            if v is not None:
                out.append((M[:3, :3] @ v.T).T + M[:3, 3])
        for child in node.findall("node"):
            walk(child, M)

    for scene in root.iter("visual_scene"):
        for node in scene.findall("node"):
            walk(node, np.eye(4))

    verts = np.vstack(out) if out else np.vstack(list(geoms.values()))
    return verts * metres_per_unit


def _guess_scale(size_m, expected_m, name):
    return 0.001 if size_m > 20 * expected_m else 1.0


def _rot_a_to_b(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.isclose(c, -1.0):
        perp = np.eye(3)[np.argmin(np.abs(a))]
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _measure_rod(verts, flip=False):
    lo, hi = verts.min(0), verts.max(0)
    k = int(np.argmax(hi - lo))
    length = hi[k] - lo[k]
    centre = (lo + hi) / 2
    others = [i for i in range(3) if i != k]

    def end_thickness(at_hi):
        band = 0.03 * length
        sel = verts[:, k] > hi[k] - band if at_hi else verts[:, k] < lo[k] + band
        pts = verts[sel][:, others]
        return float(np.ptp(pts, axis=0).max()) if len(pts) else 0.0

    thin_at_hi = end_thickness(True) < end_thickness(False)
    if flip:
        thin_at_hi = not thin_at_hi

    axis = np.zeros(3)
    axis[k] = 1.0
    thin_pt, thick_pt = centre.copy(), centre.copy()
    thin_pt[k] = hi[k] if thin_at_hi else lo[k]
    thick_pt[k] = lo[k] if thin_at_hi else hi[k]
    thick_to_thin = axis if thin_at_hi else -axis
    return length, thin_pt, thick_to_thin


def _part_offset(point, direction, target_z):
    R = _rot_a_to_b(direction, np.array([0.0, 0.0, 1.0]))
    t = np.array([0.0, 0.0, target_z]) - R @ point
    return SE3.Rt(R, t)


# ------------------------------------OBJECTS-----------------------------------------------------------------
# Cue_Ball places and rolls a ball mesh. Cue_Stick places the shaft and handle from the cue tip pose; before
# assembly the shaft can be given its own pose.

class Cue_Ball:
    def __init__(self, dae_path, color=None):
        v = Dae_Vertices(dae_path)
        size = float(np.ptp(v, axis=0).max())
        scale = _guess_scale(size, 0.057, "ball")
        v = v * scale
        self.radius = float(np.ptp(v, axis=0).max()) / 2
        self._centre = (v.min(0) + v.max(0)) / 2
        kw = {"color": color} if color is not None else {}
        self.mesh = sg.Mesh(dae_path, scale=[scale] * 3, **kw)

    def Place(self, centre_xyz, R=None):
        R = np.eye(3) if R is None else np.asarray(R)
        self.mesh.T = (SE3.Rt(R, centre_xyz, check=False) * SE3(*(-self._centre))).A

    def Add_To_Env(self, env):
        env.add(self.mesh)


class Cue_Stick:
    def __init__(self, shaft_path, handle_path, handle_insert=0.0,
                 flip_shaft=False, flip_handle=False,
                 shaft_color=None, handle_color=None):
        vs, vh = Dae_Vertices(shaft_path), Dae_Vertices(handle_path)
        s_scale = _guess_scale(float(np.ptp(vs, axis=0).max()), 1.0, "shaft")
        h_scale = _guess_scale(float(np.ptp(vh, axis=0).max()), 1.0, "handle")
        vs, vh = vs * s_scale, vh * h_scale

        L_s, tip_pt, s_dir = _measure_rod(vs, flip_shaft)
        self._shaft_off = _part_offset(tip_pt, s_dir, 0.0)

        L_h, joint_pt, h_dir = _measure_rod(vh, flip_handle)
        self._handle_off = _part_offset(joint_pt, h_dir, -L_s + handle_insert)
        L_h -= handle_insert

        self.shaft_length = L_s
        self.handle_insert = handle_insert
        self.length = L_s + L_h

        self.shaft = sg.Mesh(shaft_path, scale=[s_scale] * 3,
                             **({"color": shaft_color} if shaft_color else {}))
        self.handle = sg.Mesh(handle_path, scale=[h_scale] * 3,
                              **({"color": handle_color} if handle_color else {}))

    def Place(self, T_tip, shaft_offset=None):
        shaft_T = T_tip if shaft_offset is None else T_tip * shaft_offset
        self.shaft.T = (shaft_T * self._shaft_off).A
        self.handle.T = (T_tip * self._handle_off).A

    def Add_To_Env(self, env):
        env.add(self.shaft)
        env.add(self.handle)
