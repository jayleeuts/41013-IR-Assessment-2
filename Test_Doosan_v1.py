import time
import swift
import numpy as np
from math import pi
from spatialmath import SE3

from Doosan_Robot import Doosan_Robot


def main():
    # 1. Launch Swift Simulator
    env = swift.Swift()
    env.launch(realtime=True)

    # 2. Instantiate and add your robot
    robot = Doosan_Robot()
    robot.base = SE3(0, 0, 0)
    robot.add_to_env(env)

    # 3. Position camera
    env.set_camera_pose([2.0, 2.0, 1.5], [0.0, 0.0, 0.5])
    env.step()

    print("Doosan Robot loaded into Swift successfully!")

    # 4. Animate joints slightly so you can see it move
    t = 0
    while True:
        q = [0.5 * np.sin(t), 0.3 * np.cos(t), 0.3 * np.sin(t), 0, 0, 0]
        robot.q = q
        env.step(0.02)
        t += 0.02
        time.sleep(0.02)


if __name__ == "__main__":
    main()