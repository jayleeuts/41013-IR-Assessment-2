# ------------------------------------JBOT MODEL--------------------------------------------------------------
# The racking robot: a DH robot with a mesh for every link, whose DH lengths and meshes can be scaled
# together (set_size). Jbot_Racker.py puts it on a turtlebot with a suction cup.

import os
from math import pi, radians
import swift
import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as geometry
from spatialmath import SE3
from abc import ABC
from typing import Dict,Iterable,List,Mapping,Optional,Sequence,Tuple,Union


# ------------------------------------DH ROBOT WITH MESHES----------------------------------------------------
# Matches each link's mesh to its DH frame from a test pose, then moves the meshes with the joints.

class DHRobot3D(rtb.DHRobot, ABC):
    def __init__(self, links:List[rtb.DHLink],
                 link3D_names:Dict[str, Union[str, Tuple[float, float, float, float]]],
                 link3d_dir: str,
                 name: Optional[str] = None,
                 qtest: Optional[List[float]] = None,
                 qtest_transforms: Optional[List[np.ndarray]] = None):
        super().__init__(links, name = name)
        self.link3D_names = link3D_names

        if qtest is None:
            qtest = [0 for _ in range(self.n)]
        if qtest_transforms is None:
            qtest_transforms = [np.eye(4) for _ in range(self.n + 1)]

        self._link3D_dir = link3d_dir
        self._qtest = qtest
        self._qtest_transforms = qtest_transforms
        self._apply_3dmodel()

    def _apply_3dmodel(self):
        self.links_3d = []
        for i in range(self.n + 1):
            file_name = None
            for ext in ['.stl', '.dae', '.ply']:
                if os.path.exists(os.path.join(self._link3D_dir, self.link3D_names[f'link{i}'] + ext)):
                    file_name = os.path.join(self._link3D_dir, self.link3D_names[f'link{i}'] + ext)
                    break
            if file_name is not None:
                if f'color{i}' in self.link3D_names:
                    self.links_3d.append(geometry.Mesh(file_name, color = self.link3D_names[f'color{i}'], scale = [2,2,2]))
                else:
                    self.links_3d.append(geometry.Mesh(file_name,scale = [2,2,2]))
            else:
                raise ImportError(f'Cannot get 3D file at link {i}!')

        link_transforms = self._get_transforms(self._qtest)

        self._relation_matrices = [np.linalg.inv(link_transforms[i]) @ self._qtest_transforms[i]
                                   for i in range(len(link_transforms))]

    def _update_3dmodel(self)->None:
        link_transforms = self._get_transforms(self.q)
        for i, link in enumerate(self.links_3d):
            link.T = link_transforms[i] @ self._relation_matrices[i]

    def _get_transforms(self,q)->List[np.ndarray]:
        transforms = [self.base.A]
        L = self.links
        for i in range(self.n):
            transforms.append(transforms[i] @ L[i].A(q[i]).A)
        return transforms

    def add_to_env(self, env:swift.Swift)->None:
        if not isinstance(env, swift.Swift):
            raise TypeError('Environment must be Swift!')
        self._update_3dmodel()
        for link in self.links_3d:
            env.add(link)

    def __setattr__(self, name, value):
        if name == 'q' and hasattr(self, 'q') or name == 'base' and hasattr(self, 'base'):
            self._update_3dmodel()
        super().__setattr__(name, value)

class UTSMeshRobot(DHRobot3D):
    def __init__(
        self,
        links: Sequence[rtb.DHLink],
        mesh_stem: str,
        mesh_dir: str,
        name: Optional[str] = None,
        home_q: Optional[Iterable[float]] = None,
        base: Optional[Union[SE3, np.ndarray]] = None,
        meshes_are_global_at_home: bool = False,
        link3d_names: Optional[
            Mapping[str, Union[str, Tuple[float, float, float, float]]]
        ] = None,
        qtest_transforms: Optional[Sequence[Union[SE3, np.ndarray]]] = None,
    ):
        self.home_q = np.array(
            list(home_q) if home_q is not None else [0.0] * len(links),
            dtype=float,
        )
        if link3d_names is None:
            link3d_names = {
                f"link{i}": f"{mesh_stem}Link{i}" for i in range(len(links) + 1)
            }
        else:
            link3d_names = dict(link3d_names)

        link_colors = getattr(self, "link_colors", None)
        if link_colors is not None:
            for i, colour in enumerate(link_colors):
                link3d_names.setdefault(f"color{i}", tuple(colour))

        if qtest_transforms is not None:
            qtest_transforms = [
                self._as_transform_matrix(transform) for transform in qtest_transforms
            ]
        elif meshes_are_global_at_home:
            qtest_transforms = [np.eye(4) for _ in range(len(links) + 1)]
        else:
            qtest_transforms = self._link_frame_transforms(links, self.home_q)

        if len(qtest_transforms) != len(links) + 1:
            raise ValueError("qtest_transforms must contain one base transform plus one per link")

        super().__init__(
            list(links),
            link3d_names,
            link3d_dir=mesh_dir,
            name=name or mesh_stem,
            qtest=self.home_q.tolist(),
            qtest_transforms=qtest_transforms,
        )

        if base is not None:
            self.base = self._as_se3(base)
        self.q = self.home_q.copy()
        self._update_3dmodel()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in {"q", "base"} and hasattr(self, "links_3d"):
            self._update_3dmodel()

    @staticmethod
    def _link_frame_transforms(links, q):
        transforms = [np.eye(4)]
        for link, qi in zip(links, q):
            transforms.append(transforms[-1] @ link.A(qi).A)
        return transforms

    @staticmethod
    def _as_se3(value):
        if isinstance(value, SE3):
            return value

        matrix = np.asarray(value, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError("base must be an SE3 or a 4x4 homogeneous transform")
        return SE3(matrix, check=False)

    @staticmethod
    def _as_transform_matrix(value):
        if isinstance(value, SE3):
            return value.A

        matrix = np.asarray(value, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError("qtest transform entries must be 4x4 homogeneous transforms")
        return matrix

    @staticmethod
    def _qlim(lower, upper):
        return [radians(lower), radians(upper)]


# ------------------------------------THE JBOT----------------------------------------------------------------
# The Jbot's DH table and joint limits, and scaling of the whole robot.

class MyRobot(UTSMeshRobot):
    def __init__(self, base=None):
        links = self._create_DH()
        self.size_factor = 1.0
        super().__init__(
            links=links,
            mesh_stem="Jbot",
            mesh_dir=(os.path.join(os.path.abspath(os.path.dirname(__file__)), "Pool_Parts")
                      if os.path.exists(os.path.join(os.path.abspath(os.path.dirname(__file__)), "Pool_Parts", "JbotLink0.stl"))
                      else os.path.abspath(os.path.dirname(__file__))),
            name="MyRobot",
            home_q=[0, 0, 0, 0, 0, 0, 0],
            base=base,
        )
        self._original_a = [float(link.a) for link in self.links]
        self._original_d = [float(link.d) for link in self.links]


    def set_size(self, value):
        factor = float(value)

        if not np.isfinite(factor) or factor <= 0:
            return

        print("DH size factor:", factor, flush=True)

        for i, link in enumerate(self.links):
            link.a = self._original_a[i] * factor
            link.d = self._original_d[i] * factor


        self._update_3dmodel()

    def _create_DH(self):
        return [

            rtb.RevoluteDH(d=0.13156, a=0, alpha=pi/2, qlim=[-pi*160/180, pi*160/180], offset = -pi/2),
            rtb.RevoluteDH(d=0, a=0.1104, alpha=0, qlim=[-pi*85/180, pi*90/180], offset=pi/2),
            rtb.RevoluteDH(d=0, a=0.055, alpha=-pi/2 , qlim=[-pi*180/180, pi*45/180]),
            rtb.DHLink(d=0, a=0, alpha=pi/2, offset = pi/2, qlim = [0,0]),
            rtb.RevoluteDH(d=0.05, a=0, alpha=-pi/2, qlim=[-pi*160/180, pi*160/180], offset= -pi/2),
            rtb.RevoluteDH(d=0, a=0, alpha=pi/2, qlim=[-pi*100/180, pi*100/180]),
            rtb.RevoluteDH(d=0.0486, a=0, alpha=0, qlim=None,offset = pi),


        ]
