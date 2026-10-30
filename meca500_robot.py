# ------------------------------------MECA500-----------------------------------------------------------------
# The Meca500 as a DH robot with its CAD meshes (MECA500_Mesh_Parts), used by the station and by the jump
# shot robot.

import os
from math import pi
import numpy as np
import roboticstoolbox as rtb
from ir_support.robots.UTSMeshRobot import UTSMeshRobot


class Meca500_Robot(UTSMeshRobot):
    def __init__(self, base=None):
        links = [
            rtb.RevoluteDH(
                d=0.135,
                a=0.0,
                alpha=-pi / 2,
                offset=0.0,
                qlim=self._qlim(-175, 175),
            ),
            rtb.RevoluteDH(
                d=0.0,
                a=0.135,
                alpha=0.0,
                offset=-pi / 2,
                qlim=self._qlim(-70, 90),
            ),
            rtb.RevoluteDH(
                d=0.0,
                a=0.038,
                alpha=-pi / 2,
                offset=0.0,
                qlim=self._qlim(-135, 70),
            ),
            rtb.RevoluteDH(
                d=0.120,
                a=0.0,
                alpha=pi / 2,
                offset=0.0,
                qlim=self._qlim(-170, 170),
            ),
            rtb.RevoluteDH(
                d=0.0,
                a=0.0,
                alpha=-pi / 2,
                offset=0.0,
                qlim=self._qlim(-115, 115),
            ),
            rtb.RevoluteDH(
                d=0.070,
                a=0.0,
                alpha=pi,
                offset=0.0,
                qlim=self._qlim(-175, 175),
            ),
        ]

        current_dir = os.path.abspath(os.path.dirname(__file__))
        mesh_dir = os.path.join(current_dir, "MECA500_Mesh_Parts")

        link3d_names = {
            "link0": "link0",
            "link1": "link1",
            "link2": "link2",
            "link3": "link3",
            "link4": "link4",
            "link5": "link5",
            "link6": "link6",
        }

        super().__init__(
            links=links,
            mesh_stem="Meca500",
            mesh_dir=mesh_dir,
            link3d_names=link3d_names,
            name="Meca500",
            home_q=[0.0] * 6,
            base=base,
            meshes_are_global_at_home=False,
            qtest_transforms=[np.eye(4)] * 7,
        )

        q_cad = [-pi / 2, -pi / 2, 0.0, 0.0, 0.0, 0.0]
        link_transforms_cad = self._get_transforms(q_cad)
        self._relation_matrices = [
            np.linalg.inv(link_transforms_cad[i]) @ np.eye(4)
            for i in range(len(link_transforms_cad))
        ]
        self._update_3dmodel()
