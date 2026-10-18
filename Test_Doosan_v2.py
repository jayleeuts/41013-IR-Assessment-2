import time
import swift
import numpy as np
from spatialmath import SE3

from Doosan_Robot import Doosan_Robot
from Pool_Parts import Part_Mesh, PART_NAMES


def main():
    env = swift.Swift()
    env.launch(realtime=True)

    robot = Doosan_Robot()
    robot.base = SE3(0, 0, 0)
    robot.add_to_env(env)

    print("Pool parts found:", PART_NAMES)

    # parts, loaded by name (file name without .dae)
    Ball = Part_Mesh("Cue_Ball", pose=SE3(0.85, 0, 0.0286))      # ball origin is its centre
    Shaft = Part_Mesh("Cuestick_Shaft", pose=SE3(0.9, 0.5, 0))
    Handle = Part_Mesh("Cuestick_Handle", pose=SE3(0.9, 0.5, 0))  # same pose: assembled as exported
    env.add(Ball)
    env.add(Shaft)
    env.add(Handle)

    env.set_camera_pose([2.0, 2.0, 1.5], [0.0, 0.0, 0.5])
    env.step()
    env.hold()


if __name__ == "__main__":
    main()
