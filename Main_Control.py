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
