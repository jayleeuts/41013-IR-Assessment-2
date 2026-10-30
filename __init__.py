# ------------------------------------POOL PARTS--------------------------------------------------------------
# Finds the .dae models in this folder by name (any case, with or without the extension) and loads them as
# Swift meshes.

from pathlib import Path

import numpy as np
import spatialgeometry as geometry


PARTS_DIR = Path(__file__).resolve().parent


def Part_Names():
    return tuple(path.stem for path in sorted(PARTS_DIR.glob("*.dae")))


def Part_Path(name):
    parts = {path.stem.lower(): path for path in sorted(PARTS_DIR.glob("*.dae"))}
    key = str(name)
    if Path(key).suffix.lower() in {".dae", ".ply", ".stl"}:
        key = key[: -len(Path(key).suffix)]
    key = key.lower()
    if key not in parts:
        raise ValueError(f"Unknown pool part '{name}'. Available parts: {', '.join(Part_Names())}")
    return parts[key]


def Part_Mesh(name, pose=None, color=None):
    path = str(Part_Path(name))
    mesh = geometry.Mesh(path, color=color) if color is not None else geometry.Mesh(path)
    if pose is not None:
        mesh.T = pose.A if hasattr(pose, "A") else np.asarray(pose, dtype=float)
    return mesh
