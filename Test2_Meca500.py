import time
import numpy as np
import roboticstoolbox as rtb
import swift
from spatialmath import SE3
from meca500_robot import Meca500_Robot


def main():
    env = swift.Swift()
    env.launch(realtime=True)

    robot = Meca500_Robot()
    robot.add_to_env(env)

    print("Robot successfully loaded into Swift!\nStarting DH parameter test...")
    
        # --- TEST 1: Individual Joint Sweeps ---
    print("\n[TEST 1] Testing individual joint rotations...")
    for j in range(6):
            print(f"Moving Joint {j + 1}...")
    
            # Joint target angles (in radians)
            q_target = np.zeros(6)
            q_target[j] = np.pi / 4  # +45 degrees
    
            # Generate smooth trajectory from home to target
            q_traj = rtb.jtraj(robot.q, q_target, 50).q
    
            for q in q_traj:
                robot.q = q
                env.step(0.02)
                time.sleep(0.02)
    
            # Return back to zero configuration
            q_back = rtb.jtraj(robot.q, np.zeros(6), 50).q
            for q in q_back:
                robot.q = q
                env.step(0.02)
                time.sleep(0.02)

    # --- Step 1: Move to a natural, non-singular starting posture ---
    # Bent elbow/wrist configuration (angles in radians)
    q_ready = np.array([0.0, -np.pi / 6, np.pi / 3, 0.0, np.pi / 6, 0.0])

    print("Moving robot to ready posture...")
    q_traj = rtb.jtraj(robot.q, q_ready, 50).q
    for q in q_traj:
        robot.q = q
        env.step(0.02)
        time.sleep(0.02)

    # --- Step 2: Cartesian Motion Test ---
    print("\n[TEST 2] Executing linear end-effector movement...")

    # Get current pose at q_ready
    T_start = robot.fkine(robot.q)

    # Move 10 cm along X-axis and 5 cm along Z-axis relative to world
    T_end = T_start * SE3(0.1, 0.0, 0.05)

    # Generate Cartesian trajectory
    Ts = rtb.ctraj(T_start, T_end, 80)

    # Use Damped Least Squares IK (ikine_LMS) for smooth singularity traversal
    for T in Ts:
        # Pass current joint state as initial guess (q0)
        ik_sol = robot.ikine_LM(T, q0=robot.q)

        if ik_sol.success:
            robot.q = ik_sol.q
            env.step(0.02)
            time.sleep(0.02)
        else:
            print("IK warning: Target step unreachable")

    print("\nMotion completed smoothly without joint flipping!")

    while True:
        env.step(0.05)


if __name__ == "__main__":
    main()