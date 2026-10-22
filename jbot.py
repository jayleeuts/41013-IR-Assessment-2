import os
from math import pi
import swift
import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as geometry
import spatialmath.base as spb
from spatialmath import SE3
import os
from abc import ABC
from typing import Dict,Iterable,List,Mapping,Optional,Sequence,Tuple,Union

# Useful variables
class DHRobot3D(rtb.DHRobot, ABC):
    """
    This abstract class inherits from the DHRobot class of the `Robotics Toolbox in Python`.
    It represents a 3D Robot defined in standard DH parameters, that can be displayed in the `Swift` simulator.

    Parameters:
    -----------------------------------------------------------
    - `links`: list of DH links `roboticstoolbox.DHLink` to construct the robot
    - `link3D_names`: dictionary for names of the 3D file for each robot link in the directory, e.g. {link0: 'base', link1: 'shoulder'...}. File extension can be ply, dae, stl
    - `link3D_dir`: absolute path to the 3D files
    - `name`: name of the robot
    - `qtest`: an input joint config as a list to calibrate the 3D model. Number of elements in `qtest` is the number of robot joints
    - `qtest_transforms`: transforms of the 3D models to match the config by `qtest`.
                          Number of elements in `qtest_transforms` = number of elements in `qtest` plus 1.
                          The first element is the transform of the global coordinate to the robot.
    """

    def __init__(self, links:List[rtb.DHLink],
                 link3D_names:Dict[str, Union[str, Tuple[float, float, float, float]]],
                 link3d_dir: str,
                 name: Optional[str] = None,
                 qtest: Optional[List[float]] = None,
                 qtest_transforms: Optional[List[np.ndarray]] = None):
        """
        Initialize the 3D robot with the given DH links and 3D models

        Parameters:
        ____________
        `robot`: rtb.DHRobot
            DHRobot object
        `link3D_names`: dict
            Dictionary for names of the 3D file for each robot link in the directory, e.g. {link0: 'base', link1: 'shoulder'...}. File extension can be ply, dae, stl
        `link3D_dir`: str
            Absolute path to the 3D files
        `name`: str, optional
            Name of the robot. Default is None
        `qtest`: list, optional
            An input joint config as a list to calibrate the 3D model. Number of elements in `qtest` is the number of robot joints. Default is None
        `qtest_transforms`: list, optional
            Transforms of the 3D models to match the config by `qtest`. Number of elements in `qtest_transforms` = number of elements in `qtest` plus 1.
            The first element is the transform of the global coordinate to the robot. Default is None
        """

        super().__init__(links, name = name)
        self.link3D_names = link3D_names

        if qtest is None: # default qtest
            qtest = [0 for _ in range(self.n)]
        if qtest_transforms is None: # default transforms
            qtest_transforms = [np.eye(4) for _ in range(self.n + 1)]

        self._link3D_dir = link3d_dir
        self._qtest = qtest
        self._qtest_transforms = qtest_transforms
        self._apply_3dmodel()

    # -----------------------------------------------------------------------------------#
    def _apply_3dmodel(self):
        """
        Collect the corresponding 3D model for each link.\n
        Then compute the relation between the DH transforms for each link and the pose of its corresponding 3D object
        """
        # current_path = os.path.abspath(os.path.dirname(__file__))
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

        # Get relation matrix between the pose of the DH Link and the pose of the corresponding 3D object
        self._relation_matrices = [np.linalg.inv(link_transforms[i]) @ self._qtest_transforms[i]
                                   for i in range(len(link_transforms))]

    # -----------------------------------------------------------------------------------#
    def _update_3dmodel(self)->None:
        """
        Update the robot's 3D model based on the relation matrices
        """
        link_transforms = self._get_transforms(self.q)
        for i, link in enumerate(self.links_3d):
            link.T = link_transforms[i] @ self._relation_matrices[i]

    # -----------------------------------------------------------------------------------#
    def _get_transforms(self,q)->List[np.ndarray]:
        """
        Get the transform list representing each link
        """
        transforms = [self.base.A]
        L = self.links
        for i in range(self.n):
            transforms.append(transforms[i] @ L[i].A(q[i]).A)
        return transforms

    # -----------------------------------------------------------------------------------#
    def add_to_env(self, env:swift.Swift)->None:
        """
        Add the robot to an input Swift environment
        """
        if not isinstance(env, swift.Swift):
            raise TypeError('Environment must be Swift!')
        self._update_3dmodel()
        for link in self.links_3d:
            env.add(link)

    # -----------------------------------------------------------------------------------#
    def __setattr__(self, name, value):
        """
        Overload `=` operator so the object can update its 3D model whenever a new joint state is assigned
        """
        if name == 'q' and hasattr(self, 'q') or name == 'base' and hasattr(self, 'base'):
            self._update_3dmodel()  # Update the 3D model before setting the attribute
        super().__setattr__(name, value)  # Call the base class method to set the attribute

class UTSMeshRobot(DHRobot3D):
    """Base class for UTS MATLAB-style DH robots with link-local mesh assets."""

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


class MyRobot(UTSMeshRobot):
    def __init__(self, base=None):
        links = self._create_DH()
        self.size_factor = 1.0
        super().__init__(
            links=links,
            mesh_stem="Jbot",
            mesh_dir=os.path.abspath(os.path.dirname(__file__)),
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
        deg = pi/180
        return [
            
            rtb.RevoluteDH(d=0.13156, a=0, alpha=pi/2, qlim=[-pi*160/180, pi*160/180], offset = -pi/2),
            rtb.RevoluteDH(d=0, a=0.1104, alpha=0, qlim=[-pi*85/180, pi*90/180], offset=pi/2),
            rtb.RevoluteDH(d=0, a=0.055, alpha=-pi/2 , qlim=[-pi*180/180, pi*45/180]),
            rtb.DHLink(d=0, a=0, alpha=pi/2, offset = pi/2, qlim = [0,0]),
            rtb.RevoluteDH(d=0.05, a=0, alpha=-pi/2, qlim=[-pi*160/180, pi*160/180], offset= -pi/2),
            rtb.RevoluteDH(d=0, a=0, alpha=pi/2, qlim=[-pi*100/180, pi*100/180]),
            rtb.RevoluteDH(d=0.0486, a=0, alpha=0, qlim=None,offset = pi),

            
    
        # rtb.RevoluteDH(
        #     d=0.13156,
        #     a=0,
        #     alpha=pi / 2,
        #     qlim=[-160 * deg, 160 * deg],
        # ),
        # rtb.RevoluteDH(
        #     d=0,
        #     a=-0.1104,
        #     alpha=0,
        #     offset=-pi / 2,
        #     qlim=[-85 * deg, 90 * deg],
        # ),
        # rtb.RevoluteDH(
        #     d=0,
        #     a=-0.096,
        #     alpha=pi / 2,
        #     qlim=[-180 * deg, 45 * deg],
        # ),
        # rtb.RevoluteDH(
        #     d=0.06462,
        #     a=0,
        #     alpha=-pi / 2,
        #     offset=-pi / 2,
        #     qlim=[-160 * deg, 160 * deg],
        # ),
        # rtb.RevoluteDH(
        #     d=0.07318,
        #     a=0,
        #     alpha=0,
        #     offset=pi / 2,
        #     qlim=[-100 * deg, 100 * deg],
        # ),
        # rtb.RevoluteDH(
        #     d=0.0486,
        #     a=0,
        #     alpha=0,
        #     qlim=[-175 * deg, 175 * deg],
        # ),
    

        ]


