# ------------------------------------UR3 TEACH ROCK AND SHOOT (REAL ROBOT)-----------------------------------
# The real UR3 plays the teach-mode shot: it moves to a bench-safe shooting pose (upper arm straight up,
# gripper pointing forward 8 deg down, the same cue tilt as the simulation), rocks back and forth along the
# gripper's own axis, then shoots. Built on the lab's new_ur_control_script.py (same ROS topics). Every pose
# is checked against the table before anything is sent, and each move waits for y.

import time
import roslibpy
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3


# ------------------------------------SETTINGS----------------------------------------------------------------
# Which robot (check the label on its base), the gripper length (measure it), how close to the table it may
# go, the shooting pose (joints 2-6; the base stays facing where it is now) and the rock / shot timing.

robot_model = "UR3e"
gripper_length = 0.15
table_clearance = 0.10
shot_shape_deg = [-90.0, 110.0, -12.0, 90.0, 0.0]
n_rocks, rock_amp, rock_time = 3, 0.04, 2.0
back_swing, shoot, shoot_time, return_time = 0.08, 0.04, 1.2, 2.5
point_dt, max_tool_speed, move_time = 0.1, 0.25, 12.0

DH_PARAMS = {
    "UR3":  dict(d=[0.1519, 0, 0, 0.11235, 0.08535, 0.0819], a=[0, -0.24365, -0.21325, 0, 0, 0]),
    "UR3e": dict(d=[0.15185, 0, 0, 0.13105, 0.08535, 0.0921], a=[0, -0.24355, -0.2132, 0, 0, 0]),
}
ALPHA = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

ur3 = rtb.DHRobot([rtb.RevoluteDH(d=d, a=a, alpha=al, qlim=[-2 * np.pi, 2 * np.pi])
                   for d, a, al in zip(DH_PARAMS[robot_model]["d"], DH_PARAMS[robot_model]["a"], ALPHA)],
                  name=robot_model)
ur3.tool = SE3.Tz(gripper_length)


# ------------------------------------TALKING TO THE ROBOT----------------------------------------------------
# Read the joint angles from /ur/joint_states (the topic lists the elbow first, so joints 1 and 3 are
# swapped back) and send a whole joint trajectory to the scaled position controller, then wait for it.

def read_current_joints(client):
    latest = {}
    listener = roslibpy.Topic(client, '/ur/joint_states', 'sensor_msgs/JointState')
    listener.subscribe(lambda message: latest.update(q=list(message['position'])))
    start_time = time.time()
    while 'q' not in latest and time.time() - start_time < 5.0:
        time.sleep(0.05)
    listener.unsubscribe()
    if 'q' not in latest:
        raise RuntimeError("No joint state received from /ur/joint_states")
    q = latest['q']
    q[0], q[2] = q[2], q[0]
    print(f"[ROS] Current joint positions: {np.round(q, 3)}")
    return q


def send_trajectory(client, joint_list, time_list):
    points = [{'positions': [float(x) for x in q],
               'time_from_start': {'secs': int(t), 'nsecs': int((t - int(t)) * 1e9)}}
              for q, t in zip(joint_list, time_list)]
    topic = roslibpy.Topic(client, '/ur/scaled_pos_joint_traj_controller/command', 'trajectory_msgs/JointTrajectory')
    topic.advertise()
    topic.publish(roslibpy.Message({'joint_names': JOINT_NAMES, 'points': points}))
    print("[ROS] Trajectory published.")
    time.sleep(time_list[-1] + 1.0)
    topic.unadvertise()


# ------------------------------------TABLE CHECK-------------------------------------------------------------
# The robot stands on the bench, so the bench is z = 0: every joint after the shoulder and the fingertips
# must stay table_clearance above it for the whole move, or nothing is sent.

def check_table(joint_list, what):
    low = min(min([T.t[2] for T in ur3.fkine_all(q)[2:]] + [ur3.fkine(q).t[2]]) for q in joint_list)
    print(f"[CHECK] {what}: lowest point {low * 100:.0f} cm above the table")
    if low < table_clearance:
        raise RuntimeError(f"{what} comes within {low * 100:.0f} cm of the table - nothing sent")


# ------------------------------------THE SHOT----------------------------------------------------------------
# The shooting pose keeps the current base angle. The stroke is a list of distances along the gripper axis
# (rocks, back-swing, shot, hold, return, each with a smooth start and stop), turned into joint angles by IK
# point by point; it is refused if IK jumps to another arm shape or the tool would go faster than
# max_tool_speed.

def shooting_pose(base_angle):
    q = np.r_[base_angle, np.radians(shot_shape_deg)]
    tip = ur3.fkine(q)
    print(f"[PLAN] fingertips {tip.t[2] * 100:.0f} cm above the table, {np.linalg.norm(tip.t[:2]) * 100:.0f} cm out, "
          f"tilted {np.degrees(np.arcsin(-tip.R[2, 2])):.0f} deg down")
    return q


def rock_and_shoot_plan(start_joints):
    s_list = [0.0]

    def add(s_from, s_to, duration):
        n = max(2, int(round(duration / point_dt)))
        for k in range(1, n + 1):
            tau = k / n
            s_list.append(s_from + (s_to - s_from) * (10 * tau**3 - 15 * tau**4 + 6 * tau**5))

    for _ in range(n_rocks):
        add(0.0, -rock_amp, rock_time / 2)
        add(-rock_amp, 0.0, rock_time / 2)
    add(0.0, -back_swing, 1.5)
    add(-back_swing, shoot, shoot_time)
    add(shoot, shoot, 1.0)
    add(shoot, 0.0, return_time)
    time_list = [k * point_dt for k in range(len(s_list))]

    q = np.array(start_joints, dtype=float)
    T_start = ur3.fkine(q)
    joint_list = [q.copy()]
    for s in s_list[1:]:
        sol = ur3.ikine_LM(SE3(*(T_start.R[:, 2] * s)) * T_start, q0=q)
        if not sol.success:
            raise RuntimeError(f"IK failed {s:+.3f} m along the gripper")
        q_new = q + (sol.q - q + np.pi) % (2 * np.pi) - np.pi
        if np.max(np.abs(q_new - q)) > 0.15:
            raise RuntimeError("IK jumped to a different arm configuration")
        q = q_new
        joint_list.append(q.copy())

    top_speed = np.max(np.abs(np.diff(s_list))) / point_dt
    print(f"[PLAN] {n_rocks} rocks, {time_list[-1]:.1f} s, top tool speed {top_speed:.2f} m/s")
    if top_speed > max_tool_speed:
        raise RuntimeError("Shot is faster than max_tool_speed - make shoot_time bigger")
    check_table(joint_list, "Rock and shoot")
    return joint_list, time_list


# ------------------------------------RUN---------------------------------------------------------------------
# Move to the shooting pose (slowly, after a y), then plan the stroke from where the arm really ended up and
# shoot (after another y). Ctrl+C or any failed check stops it.

if __name__ == '__main__':
    client = roslibpy.Ros(host='192.168.27.1', port=9090)
    client.run()
    try:
        now = np.array(read_current_joints(client))
        start = shooting_pose(now[0])
        if input("Move to the shooting pose? (y): ").strip().lower() == 'y':
            now = np.array(read_current_joints(client))
            check_table([now + (start - now) * k for k in np.linspace(0, 1, 50)], "Move to the shooting pose")
            send_trajectory(client, [now, start], [0.0, move_time])
            joint_list, time_list = rock_and_shoot_plan(read_current_joints(client))
            if input("Rock and shoot? (y): ").strip().lower() == 'y':
                send_trajectory(client, joint_list, time_list)
    except KeyboardInterrupt:
        print("\n[APP] Interrupted by user.")
    except RuntimeError as err:
        print(f"[APP] Stopped: {err}")
    finally:
        print("[ROS] Finish")
        client.terminate()
