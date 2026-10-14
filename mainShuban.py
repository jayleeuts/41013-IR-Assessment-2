import swift
from meca500_robot import Meca500_Robot

def main():
    # 1. Initialize Swift Environment
    env = swift.Swift()
    env.launch(realtime=True)

    # 2. Instantiate Robot
    robot = Meca500_Robot()

    # 3. Add Robot to Environment
    robot.add_to_env(env)

    print("Robot successfully loaded into Swift!")

    # 4. Hold environment open
    while True:
        env.step(0.05)

if __name__ == "__main__":
    main()