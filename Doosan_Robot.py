# ------------------------------------DOOSAN M0609------------------------------------------------------------
# The Doosan M0609 as a DH robot with our own CAD meshes; the DH lengths were measured from those meshes so
# the drawing and the kinematics match.

import os
from math import pi
import roboticstoolbox as rtb
from ir_support.robots.UTSMeshRobot import UTSMeshRobot


class Doosan_Robot(UTSMeshRobot):

    def __init__(self, base=None):
        links = [
            rtb.RevoluteDH(d=0.1156, a=0.0, alpha=-pi / 2, offset=0.0, qlim=self._qlim(-360, 360)),

            rtb.RevoluteDH(d=0.0, a=0.4099, alpha=0.0, offset=-pi / 2, qlim=self._qlim(-360, 360)),

            rtb.RevoluteDH(d=-0.1537, a=0.0, alpha=pi / 2, offset=pi / 2, qlim=self._qlim(-360, 360)),

            rtb.RevoluteDH(d=0.3671, a=0.0, alpha=-pi / 2, offset=0.0, qlim=self._qlim(-360, 360)),

            rtb.RevoluteDH(d=0.1449, a=0.0, alpha=pi / 2, offset=0.0, qlim=self._qlim(-360, 360)),

            rtb.RevoluteDH(d=0.1194, a=0.0, alpha=0.0, offset=0.0, qlim=self._qlim(-360, 360)),
        ]

        current_dir = os.path.abspath(os.path.dirname(__file__))
        mesh_dir = os.path.join(current_dir, "Doosan_Parts_Dae")
        if not os.path.isdir(mesh_dir):
            mesh_dir = os.path.join(current_dir, "Doosan_Robot_M0609_SLDWRK", "Doosan_Parts_Dae")

        link3d_names = {
            "link0": "Link_0_Base",
            "link1": "Link_1_Shoulder",
            "link2": "Link_2_UpperArm",
            "link3": "Link_3_Forearm",
            "link4": "Link_4_Wrist1",
            "link5": "Link_5_Wrist2",
            "link6": "Link_6_Flange",
        }

        super().__init__(
            links=links,
            mesh_stem="Doosan_Robot",
            mesh_dir=mesh_dir,
            link3d_names=link3d_names,
            name="Doosan_Robot",
            home_q=[0.0] * 6,
            base=base,
            meshes_are_global_at_home=True,
        )
