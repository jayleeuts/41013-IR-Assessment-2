import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import swift as sw
from spatialmath.base import *
from spatialmath import SE3
from spatialgeometry import Sphere, Arrow, Mesh
from roboticstoolbox import DHLink, DHRobot, models, jtraj, trapezoidal
from ir_support import CylindricalDHRobotPlot
from ir_support_extra_robots.robots import Turtlebot3Waffle
from ir_support.robots import UR3e
from ir_support_extra_parts.parts import part_names, part_mesh
import os


# Useful variables
from math import pi

class Lab2AssessmentWorkspace:
    def __init__(self):
        self.JbotLink1 = DHLink(d=0.13156, a=0, alpha=pi/2, qlim=[-pi*160/180, pi*160/180])
        self.JbotLink2 = DHLink(d=0, a=0.1104, alpha=pi, qlim=[-pi*85/180, pi*90/180], offset=pi/2)
        self.JbotLink3 = DHLink(d=0, a=0.096, alpha=pi/2, qlim=[-pi*180/180, pi*45/180])
        self.JbotLink4 = DHLink(d=0, a=0, alpha=-pi/2, qlim=[-pi*160/180, pi*160/180], offset= pi/2)
        self.JbotLink5 = DHLink(d=0.107318, a=0, alpha=0, qlim=[-pi*100/180, pi*100/180])
        self.JbotLink6 = DHLink(d=0.0486, a=0, alpha=pi/2, qlim=None,offset = pi)
        self.Jbot = DHRobot([self.JbotLink1, self.JbotLink2, self.JbotLink3, self.JbotLink4, self.JbotLink5, self.JbotLink6], name="Jbot")
        cyl_viz = CylindricalDHRobotPlot(self.Jbot, cylinder_radius=0.05, color="#3478f6")
        self.Jbot = cyl_viz.create_cylinders()

        self.env = sw.Swift()
        self.env.launch(realtime=True)
        self.stop_event = threading.Event()  # Event to end teach mode when 'enter' pressed
        
    def set_joint(self,j, value):
        q = self.Jbot.q.copy()
        q[j] = np.deg2rad(float(value))
        self.Jbot.q = q
    
    def wait_for_enter(self):
                
        try:
        # print("Press Enter to stop.\n")
            input()
        except EOFError:
            pass
            self.stop_event.set()
    def start(self):
        self.env.add(self.Jbot)
        for i in range(self.Jbot.n):
            self.env.add(
                sw.Slider(lambda value, j = i: self.set_joint(j,value),
                        min=-180,
                        max=180,
                        step = 1,
                        value=np.rad2deg(self.Jbot.q[i]),
                        desc=f"Joint {i+1}",)
                )
        for i in range(1000):
            self.env.step(0.05)


if __name__ == "__main__":
    task = Lab2AssessmentWorkspace()
    task.start()
       
