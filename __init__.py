from pathlib import Path

import numpy as np
import spatialgeometry as geometry


PARTS_DIR = Path(__file__).resolve().parent

# Optional source/reference URLs for part meshes. Keep this empty unless
# there is a real, stable source URL for an unmodified or lightly modified asset.
PART_REFERENCE_URLS = {}


def _Part_Index():
    return {path.stem.lower(): path for path in sorted(PARTS_DIR.glob("*.dae"))}


def _Normalise_Name(name):
    name = str(name)
    suffix = Path(name).suffix.lower()
    if suffix in {".dae", ".ply", ".stl"}:
        name = name[: -len(suffix)]
    return name.lower()


def Part_Names():
    """Return the available pool part names without file extensions."""

    return tuple(path.stem for path in sorted(PARTS_DIR.glob("*.dae")))


def Part_Path(name):
    """Return the DAE path for a pool part, e.g. Part_Path("Cue_Ball")."""

    parts = _Part_Index()
    key = _Normalise_Name(name)
    if key not in parts:
        available = ", ".join(Part_Names())
        raise ValueError(f"Unknown pool part '{name}'. Available parts: {available}")
    return parts[key]


def Part_Reference_Url(name):
    """Return a reference URL for a part if one is known."""

    key = _Normalise_Name(name)
    for known_name, url in PART_REFERENCE_URLS.items():
        if _Normalise_Name(known_name) == key:
            return url
    return None


def Part_Mesh(name, pose=None, color=None):
    """Create a spatialgeometry Mesh for a pool part, e.g. Part_Mesh("Cue_Ball")."""

    path = str(Part_Path(name))
    mesh = geometry.Mesh(path, color=color) if color is not None else geometry.Mesh(path)
    if pose is not None:
        mesh.T = pose.A if hasattr(pose, "A") else np.asarray(pose, dtype=float)
    return mesh


PART_NAMES = Part_Names()

__all__ = [
    "PART_NAMES",
    "PART_REFERENCE_URLS",
    "Part_Names",
    "Part_Path",
    "Part_Reference_Url",
    "Part_Mesh",
]
