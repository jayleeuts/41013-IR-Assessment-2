import os
from math import pi
import roboticstoolbox as rtb
from ir_support.robots.UTSMeshRobot import UTSMeshRobot


class Doosan_Robot(UTSMeshRobot):
    """Doosan 6-DOF Robot model ported for UTS Robotics Toolbox and Swift."""

    def __init__(self, base=None):
        # 6 Links based on Table 6.2 DH Parameters:
        links = [
            # Joint 1: d = 135 mm (base to shoulder axis height)
            rtb.RevoluteDH(d=0.135, a=0.0, alpha=-pi / 2, offset=0.0, qlim=self._qlim(-360, 360)),
            
            # Joint 2: d=6.2mm, a=411mm, alpha=0 deg, offset=-90 deg
            rtb.RevoluteDH(d=0.0062, a=0.411, alpha=0.0, offset=-pi / 2, qlim=self._qlim(-360, 360)),
            
            # Joint 3: d=0, a=0, alpha=90 deg, offset=+90 deg
            rtb.RevoluteDH(d=0.0, a=0.0, alpha=pi / 2, offset=pi / 2, qlim=self._qlim(-360, 360)),
            
            # Joint 4: d=368mm, a=0, alpha=-90 deg, offset=0
            rtb.RevoluteDH(d=0.368, a=0.0, alpha=-pi / 2, offset=0.0, qlim=self._qlim(-360, 360)),
            
            # Joint 5: d=0, a=0, alpha=90 deg, offset=0
            rtb.RevoluteDH(d=0.0, a=0.0, alpha=pi / 2, offset=0.0, qlim=self._qlim(-360, 360)),
            
            # Joint 6: d=121mm, a=0, alpha=0 deg, offset=0
            rtb.RevoluteDH(d=0.121, a=0.0, alpha=0.0, offset=0.0, qlim=self._qlim(-360, 360)),
        ]

        # Path to your exact .dae folder
        current_dir = os.path.abspath(os.path.dirname(__file__))
        mesh_dir = os.path.join(current_dir, "Doosan_Robot_M0609_SLDWRK", "Doosan_Parts_Dae")

        # Map each link to your exact exported filenames:
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