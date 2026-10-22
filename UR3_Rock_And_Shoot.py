#!/usr/bin/env python3
"""
UR3_Rock_And_Shoot - real UR3: rock the tool back and forth twice, then "shoot".

Same ROS connection as new_ur_control_script.py, but instead of jumping to one
fixed pose it works out a real motion FROM WHEREVER THE ARM IS NOW:

  1. Read the current joint angles from /ur/joint_states.
  2. Forward kinematics (UR3 model) -> where the tool is right now.
  3. Plan the tool sliding in a STRAIGHT LINE, horizontally, along the direction
     pointing away from the robot base ("forward"):
         rock back + forward, rock back + forward, back-swing, SHOOT, settle back.
  4. Inverse kinematics at every point (seeded with the previous answer) -> joints.
  5. Safety checks, print the plan, ask you to type y, then send the whole
     trajectory to /ur/scaled_pos_joint_traj_controller/command in one go.

Distances and speeds are deliberately SMALL and SLOW for a first test.
Keep your hand on the e-stop / teach pendant. 
"""
import time
import numpy as np
import roslibpy
import roboticstoolbox as rtb
from spatialmath import SE3


# ------------------------------------SETTINGS (keep small for the first test)-----------------------------

ROS_HOST = '192.168.27.1'            # ROS bridge IP (same as new_ur_control_script.py)
ROS_PORT = 9090

ROCK_AMP = 0.03                      # m, how far back each practice rock goes
ROCK_TIME = 2.0                      # s, one full rock (back and forward again)
N_ROCKS = 2
BACK_SWING = 0.05                    # m, final back-swing before the shot
BACK_TIME = 1.5                      # s
SHOOT = 0.05                         # m, how far PAST the start point the shot goes
SHOOT_TIME = 1.0                     # s, back-swing -> end of shot (smaller = faster)
HOLD_TIME = 1.0                      # s, pause at the end of the shot
RETURN_TIME = 2.0                    # s, slide back to the start point

POINT_DT = 0.1                       # s between trajectory points sent to the controller
MAX_TOOL_SPEED = 0.25                # m/s - refuse to send if the plan is faster than this
MAX_JOINT_STEP = 0.15                # rad between two points - refuse if bigger (IK flip)

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']


# ------------------------------------READ THE CURRENT JOINTS-----------------------------------------------------

def Read_Current_Joints(client, timeout=5.0):
    """Current joint angles, in JOINT_NAMES order."""
    latest = {}

    def callback(message):
        latest['pos'] = list(message['position'])

    listener = roslibpy.Topic(client, '/ur/joint_states', 'sensor_msgs/JointState')
    listener.subscribe(callback)
    start = time.time()
    while 'pos' not in latest and time.time() - start < timeout:
        time.sleep(0.05)
    listener.unsubscribe()
    if 'pos' not in latest:
        raise RuntimeError("No joint state received from /ur/joint_states")

    q = latest['pos']
    q[0], q[2] = q[2], q[0]          # same swap as new_ur_control_script.py (topic lists elbow first)
    return np.array(q, dtype=float)


# ------------------------------------PLAN THE MOTION-----------------------------------------------------

def Stroke_Profile():
    """(time, s) points. s = metres FORWARD of the start point (negative = pulled back)."""
    times, s_vals, t = [0.0], [0.0], 0.0

    def add_segment(s_from, s_to, duration):
        nonlocal t
        n = max(1, int(round(duration / POINT_DT)))
        for k in range(1, n + 1):
            tau = k / n
            blend = 10 * tau**3 - 15 * tau**4 + 6 * tau**5          # smooth start + stop
            times.append(t + duration * tau)
            s_vals.append(s_from + (s_to - s_from) * blend)
        t += duration

    for _ in range(N_ROCKS):
        add_segment(0.0, -ROCK_AMP, ROCK_TIME / 2)                   # rock back
        add_segment(-ROCK_AMP, 0.0, ROCK_TIME / 2)                   # rock forward
    add_segment(0.0, -BACK_SWING, BACK_TIME)                         # back-swing
    add_segment(-BACK_SWING, SHOOT, SHOOT_TIME)                      # SHOOT
    add_segment(SHOOT, SHOOT, HOLD_TIME)                             # hold
    add_segment(SHOOT, 0.0, RETURN_TIME)                             # back to start
    return np.array(times), np.array(s_vals)


def Plan_Joint_Trajectory(q_now):
    """Straight-line horizontal stroke away from the base, solved with IK."""
    ur3 = rtb.models.DH.UR3()
    T_start = ur3.fkine(q_now)

    forward = np.array([T_start.t[0], T_start.t[1], 0.0])            # away from the base, horizontal
    if np.linalg.norm(forward) < 0.05:
        raise RuntimeError("Tool is right above the base - move it out a bit first.")
    forward /= np.linalg.norm(forward)

    times, s_vals = Stroke_Profile()
    q_list, q = [q_now.copy()], q_now.copy()
    for s in s_vals[1:]:
        T_goal = SE3(*(forward * s)) * T_start                       # same orientation, shifted
        sol = ur3.ikine_LM(T_goal, q0=q)
        if not sol.success:
            raise RuntimeError(f"IK failed {s:+.3f} m from the start - try a different start pose.")
        if np.max(np.abs(sol.q - q)) > MAX_JOINT_STEP:
            raise RuntimeError("IK jumped to a different arm configuration - aborting.")
        q = sol.q
        q_list.append(q.copy())

    return times, s_vals, np.array(q_list), T_start, forward


def Check_And_Print(times, s_vals, q_list, T_start, forward):
    speeds = np.abs(np.diff(s_vals)) / np.diff(times)
    joint_change = np.degrees(np.max(np.abs(q_list - q_list[0]), axis=0))
    print(f"\n[PLAN] tool now at {np.round(T_start.t, 3)} m (UR3 base frame)")
    print(f"[PLAN] stroke direction (away from base): {np.round(forward, 3)}")
    print(f"[PLAN] {N_ROCKS} rocks of {ROCK_AMP*100:.0f} cm, back-swing {BACK_SWING*100:.0f} cm, "
          f"shot to {SHOOT*100:.0f} cm past start")
    print(f"[PLAN] {len(times)} points over {times[-1]:.1f} s, top tool speed {speeds.max():.2f} m/s")
    print(f"[PLAN] biggest joint change from start (deg): {np.round(joint_change, 1)}")
    if speeds.max() > MAX_TOOL_SPEED:
        raise RuntimeError(f"Plan is faster than MAX_TOOL_SPEED ({MAX_TOOL_SPEED} m/s) - slow it down.")


# ------------------------------------SEND IT-----------------------------------------------------

def Send_Trajectory(client, times, q_list):
    points = []
    for t, q in zip(times, q_list):
        points.append({
            'positions': [float(x) for x in q],
            'time_from_start': {'secs': int(t), 'nsecs': int((t - int(t)) * 1e9)},
        })
    topic = roslibpy.Topic(client, '/ur/scaled_pos_joint_traj_controller/command',
                           'trajectory_msgs/JointTrajectory')
    topic.advertise()
    time.sleep(0.5)                                                  # let the advertise settle
    topic.publish(roslibpy.Message({'joint_names': JOINT_NAMES, 'points': points}))
    print("[ROS] Trajectory published - robot moving.")
    time.sleep(times[-1] + 1.0)
    topic.unadvertise()


# ------------------------------------MAIN-----------------------------------------------------

if __name__ == '__main__':
    client = roslibpy.Ros(host=ROS_HOST, port=ROS_PORT)
    client.run()
    try:
        print("[ROS] Connected." if client.is_connected else "[ROS] NOT connected - check the IP.")
        q_now = Read_Current_Joints(client)
        print(f"[ROS] Current joints (rad): {np.round(q_now, 3)}")

        times, s_vals, q_list, T_start, forward = Plan_Joint_Trajectory(q_now)
        Check_And_Print(times, s_vals, q_list, T_start, forward)

        if input("\nType y and Enter to MOVE the robot (anything else cancels): ").strip().lower() == 'y':
            Send_Trajectory(client, times, q_list)
            print("[APP] Done.")
        else:
            print("[APP] Cancelled - nothing sent.")
    except KeyboardInterrupt:
        print("\n[APP] Interrupted by user.")
    except RuntimeError as err:
        print(f"[APP] Stopped: {err}")
    finally:
        client.terminate()
