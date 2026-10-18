import os
from math import pi
import roboticstoolbox as rtb
from ir_support.robots.UTSMeshRobot import UTSMeshRobot


class Doosan_Robot(UTSMeshRobot):
    """Doosan M0609 6-DOF robot for the UTS Robotics Toolbox and Swift.

    DH parameters are measured from OUR CAD meshes (Doosan_Parts_Dae), not the
    datasheet: each joint axis was found from the ring where two neighbouring
    link meshes touch. Every DH joint axis now passes exactly through its CAD
    joint, so the links rotate about the right place and no gaps open up.

    Robot base frame = CAD origin. The base mesh's underside is 22 mm BELOW
    that origin (BASE_FOOT), so mount the robot with base = surface * SE3.Tz(BASE_FOOT).
    """

    BASE_FOOT = 0.022          # base frame sits 22 mm above the bottom of the base mesh

    def __init__(self, base=None):
        links = [
            # Joint 1: vertical. Shoulder axis (J2) is 115.6 mm above the base frame
            rtb.RevoluteDH(d=0.1156, a=0.0, alpha=-pi / 2, offset=0.0, qlim=self._qlim(-360, 360)),

            # Joint 2: shoulder. Upper arm 409.9 mm from shoulder to elbow axis
            rtb.RevoluteDH(d=0.0, a=0.4099, alpha=0.0, offset=-pi / 2, qlim=self._qlim(-360, 360)),

            # Joint 3: elbow. Forearm's S-bend puts J4 153.7 mm to the side
            rtb.RevoluteDH(d=-0.1537, a=0.0, alpha=pi / 2, offset=pi / 2, qlim=self._qlim(-360, 360)),

            # Joint 4: forearm roll. 367.1 mm up to the J5 axis
            rtb.RevoluteDH(d=0.3671, a=0.0, alpha=-pi / 2, offset=0.0, qlim=self._qlim(-360, 360)),

            # Joint 5: wrist bend. J6 axis is 144.9 mm back along J5
            rtb.RevoluteDH(d=0.1449, a=0.0, alpha=pi / 2, offset=0.0, qlim=self._qlim(-360, 360)),

            # Joint 6: flange roll. Flange face 119.4 mm above the J5 axis
            rtb.RevoluteDH(d=0.1194, a=0.0, alpha=0.0, offset=0.0, qlim=self._qlim(-360, 360)),
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
