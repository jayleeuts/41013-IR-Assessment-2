import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import swift as sw
from spatialmath.base import *
from spatialmath import SE3
import spatialgeometry as geometry
from roboticstoolbox import DHLink, DHRobot, models, jtraj, trapezoidal
from ir_support import CylindricalDHRobotPlot
from ir_support_extra_robots.robots import Turtlebot3Waffle
from ir_support.robots import UR3e
from ir_support_extra_parts.parts import part_names, part_mesh
import os
from Doosan_Robot import Doosan_Robot
from ir_support_extra_robots.robots import Turtlebot3Waffle
from Pool_Parts import Part_Path, Part_Mesh
from Pool_Objects import Cue_Ball, Cue_Stick
from Pool_Motion import (Holder_Tool, Solve_IK, Cue_Pose, Stroke_Profile,
                         Punch_Stroke_Profile, Ball_Physics, Balls_Collide)
from Pool_Scene import (CHANNEL_Y, CHANNEL_X, TABLE_Z, FELT_Z, PLAY_AREA,
                        GROOVE_HEIGHT, Turtle_Pose, Robot_Base_On, Drive_Path)


# ------------------------------------CONSTANTS-----------------------------------------------------

DT = 0.02

HANDLE_PIN = 0.033                    # handle's joint pin = how far the shaft screws down
N_TURNS = 6                           # turns to screw the shaft on
GRIP_FROM_BUTT = 0.55                 # where the handle sits in the holder (from the butt)

# carry pose: holder in front of the robot, cue straight up (robot-base frame)
CARRY_GRIP_POINT = (0.40, 0.0, 0.65)
CARRY_AIM_DEG = 180.0

START_TURTLE = (0.0, CHANNEL_Y, 180.0)          # (x, y, yaw) on the channel

# straight shot: turtlebot at the end of the table, cue along the table
STRAIGHT_TURTLE = (-CHANNEL_X, 0.0, None)       # None = keep the yaw it arrives with
BALL_1_SPOT = (0.00, 0.20)
STRAIGHT_AIM_DEG, STRAIGHT_ELEV_DEG, STRAIGHT_ROLL_DEG = 0.0, 8.0, 180.0
STRAIGHT_STRIKE_TIME = 0.65           # s, back-swing -> follow-through (shorter = harder hit)

# masse: turtlebot at the side of the table, steep cue
MASSE_TURTLE = (-0.30, -CHANNEL_Y, None)
BALL_2_SPOT = (0.30, -0.15)
MASSE_AIM_DEG, MASSE_ELEV_DEG, MASSE_ROLL_DEG = 45.0, 45.0, -90.0
SIDE_HIT = 0.6                        # off-centre hit, fraction of the ball radius
CURVE_LEFT = True
MASSE_TIP_SPEED = 1.4                 # m/s at contact

ADDRESS_GAP = 0.02                    # tip -> ball surface before the stroke
APPROACH = 0.15                       # cue slides in along its own axis from this far back
CUE_TO_BALL = 1.4                     # ball speed / tip speed at impact
JOINT_SPEED = 2.0                     # rad/s - top joint speed for joint-space moves


from jbot import MyRobot
from Pool_Parts import Part_Path, Part_Mesh
from Pool_Objects import Cue_Ball, Cue_Stick
from Pool_Motion import (Holder_Tool, Solve_IK, Cue_Pose, Stroke_Profile,
                         Punch_Stroke_Profile, Ball_Physics, Balls_Collide)
from Pool_Scene import (CHANNEL_Y, CHANNEL_X, TABLE_Z, FELT_Z, PLAY_AREA,
                        GROOVE_HEIGHT, Turtle_Pose, Robot_Base_On, Drive_Path)


# Useful variables
from math import pi

class Lab2AssessmentWorkspace:
    def __init__(self):
        self.jayleeRobot = MyRobot()
        self.turtle = Turtlebot3Waffle()
        
    
        
        
          # Event to end teach mode when 'enter' pressed
        
    
    
    
    def wait_for_enter(self):
                
        try:
        # print("Press Enter to stop.\n")
            input()
        except EOFError:
            pass
            self.stop_event.set()

    def set_joint(self,j, value):
        q = self.jayleeRobot.q.copy()
        q[j] = np.deg2rad(float(value))
        self.jayleeRobot.q = q
    
    
    def start(self):
        env.set_camera_pose([3.0, -2.6, 2.6], [0.0, 0.0, 0.8])
        self.turtle.base = SE3(0, -0.925, 0.8) * SE3.Rz(np.radians(0))
        self.turtle.add_to_env(env)
        self.jayleeRobot.base = self.turtle.base * transl(0,0.1,0)



        self.jayleeRobot.add_to_env(env)

        env.add(
            sw.Slider(
            lambda value: self.jayleeRobot.set_size(value),
            min=0.5,
            max=2.0,
            step=0.05,
            value=1.0,
            desc="Robot size",
        )
)


        

       
        for i in range(self.jayleeRobot.n):
            env.add(
                sw.Slider(lambda value, j = i: self.set_joint(j,value),
                       min=-180,
                        max=180,
                           step = 1,
                        value=np.rad2deg(self.jayleeRobot.q[i]),
                        desc=f"Joint {i+1}",)
                )

        for i in range(10000):            
            env.step(0.05)
    
        


if __name__ == "__main__":
    task = Lab2AssessmentWorkspace()
    env = sw.Swift()
    env.launch(realtime=True)

    env.set_camera_pose([3.0, -2.6, 2.6], [0.0, 0.0, 0.8])

    track = Part_Mesh("Robot_Track", pose=SE3())
    table = Part_Mesh("Pool_Table", pose=SE3(0, 0, TABLE_Z))
    env.add(track)
    env.add(table)

    turtle = Turtlebot3Waffle()
    turtle_xyyaw = START_TURTLE
    turtle.base = Turtle_Pose(*turtle_xyyaw)
    turtle.add_to_env(env)

    robot = Doosan_Robot()
    robot.base = Robot_Base_On(turtle.base)
    robot.add_to_env(env)

    holder = Part_Mesh("Cue_Holder")
    env.add(holder)

    cue = Cue_Stick(str(Part_Path("Cuestick_Shaft")), str(Part_Path("Cuestick_Handle")),
                    handle_insert=HANDLE_PIN)
    cue.Add_To_Env(env)
    GRIP_FROM_TIP = cue.length - GRIP_FROM_BUTT
    TOOL = Holder_Tool(GROOVE_HEIGHT, GRIP_FROM_TIP)

    balls, physics, spots = [], [], [BALL_1_SPOT, BALL_2_SPOT]
    for spot in spots:
        b = Cue_Ball(str(Part_Path("Cue_Ball")))
        p = Ball_Physics([spot[0], spot[1], FELT_Z + b.radius], b.radius, bounds=PLAY_AREA)
        b.Place(p.pos, p.R)
        b.Add_To_Env(env)
        balls.append(b)
        physics.append(p)
    BALL_Z = FELT_Z + balls[0].radius

    state = {"attached": False, "shaft_world": None}
    task.start()
       