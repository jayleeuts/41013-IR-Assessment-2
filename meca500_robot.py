import os
from math import pi
import numpy as np
import roboticstoolbox as rtb
from ir_support.robots.UTSMeshRobot import UTSMeshRobot


class Meca500_Robot(UTSMeshRobot):
    """Mecademic Meca500 6-DOF Robot model ported for UTS Robotics Toolbox and Swift."""

    def __init__(self, base=None):
        # 6 Links based on standard Meca500 DH Parameters
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

        # Directory containing your assessment meshes
        current_dir = os.path.abspath(os.path.dirname(__file__))
        mesh_dir = os.path.join(current_dir, "MECA500_Mesh_Parts")

        # Standard mesh mapping: Update the right side values to match your exact STL/DAE filenames (without .stl/.dae extension)
        link3d_names = {
            "link0": "link0",  # Base mesh filename
            "link1": "link1",  # Link 1 mesh filename
            "link2": "link2",  # Link 2 mesh filename
            "link3": "link3",  # Link 3 mesh filename
            "link4": "link4",  # Link 4 mesh filename
            "link5": "link5",  # Link 5 mesh filename
            "link6": "link6",  # Link 6 mesh filename
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

        # The CAD STL meshes were exported at joint configuration [-pi/2, -pi/2, 0, 0, 0, 0]
        # (arm reaching along +Y instead of +X, and Link 2 horizontal instead of vertical).
        # We recalibrate the mesh relation matrices to that CAD configuration:
        q_cad = [-pi / 2, -pi / 2, 0.0, 0.0, 0.0, 0.0]
        link_transforms_cad = self._get_transforms(q_cad)
        self._relation_matrices = [
            np.linalg.inv(link_transforms_cad[i]) @ np.eye(4)
            for i in range(len(link_transforms_cad))
        ]
        self._update_3dmodel()
