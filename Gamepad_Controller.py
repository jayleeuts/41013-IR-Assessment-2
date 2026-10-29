import time

import numpy as np
import pygame


class GamepadController:
    """
    Direct joint-space gamepad controller for the UR3.

    This behaves like an excavator controller:
    each input controls one joint velocity directly.
    No Jacobian, inverse kinematics or damped least squares is used.

    Controller layout for this cheap controller:

        Left stick / left D-pad:
            AXIS 0: left/right
            AXIS 1: up/down

        Right D-pad / right joystick:
            BUTTON 0: up
            BUTTON 1: left
            BUTTON 2: down
            BUTTON 3: right

        Triggers:
            BUTTON 4: top-left
            BUTTON 5: top-right
            BUTTON 6: bottom-left
            BUTTON 7: bottom-right

        Central buttons:
            BUTTON 8: Select
            BUTTON 9: Start

    Joint control allocation:

        J1: left D-pad/joystick left-right
        J2: left D-pad/joystick up-down
        J3: left triggers
        J4: right D-pad/joystick up-down
        J5: right D-pad/joystick left-right
        J6: right triggers

        Select: reset UR3
        Start: reserved for shooting / power control later
    """

    def __init__(
        self,
        joystick_index=0,
        deadzone=0.15,
        joint_speed=np.radians(45.0),
        smoothing=0.25,
    ):
        self.joystick_index = joystick_index
        self.deadzone = deadzone

        # Maximum joint speed, rad/s.
        self.joint_speed = joint_speed

        # Joystick smoothing: 0 = none, closer to 1 = smoother/slower.
        self.smoothing = smoothing

        # ---- Input mappings for this controller ----

        # Left stick and left D-pad share these axes.
        self.axis_map = {
            "x": (0, 1.0),     # left/right
            "y": (1, 1.0),    # up/down; -1 makes up positive
        }

        # Right D-pad / right joystick buttons.
        self.right_map = {
            "up": 0,
            "left": 1,
            "down": 2,
            "right": 3,
        }

        # Trigger buttons.
        self.trigger_map = {
            "top_left": 4,
            "top_right": 5,
            "bottom_left": 6,
            "bottom_right": 7,
        }

        # Central buttons.
        self.button_map = {
            "select": 8,
            "start": 9,
        }

        self._filtered_dq = np.zeros(6)

        self._last_button_time = 0.0
        self.button_cooldown = 0.30

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected.")

        self.joy = pygame.joystick.Joystick(self.joystick_index)
        self.joy.init()

        self.name = self.joy.get_name()
        self.num_buttons = self.joy.get_numbuttons()
        self.num_axes = self.joy.get_numaxes()

        print(f"Joystick: {self.name}")
        print(f" - {self.num_buttons} buttons")
        print(f" - {self.num_axes} axes")

    def update(self):
        """Process pygame events. Call once per controller frame."""
        pygame.event.pump()

    def _axis(self, key):
        """Read one of the left-side analog axes, with deadzone."""
        mapping = self.axis_map.get(key)

        if mapping is None:
            return 0.0

        axis, sign = mapping

        if axis >= self.num_axes:
            return 0.0

        value = self.joy.get_axis(axis)

        if abs(value) < self.deadzone:
            return 0.0

        return sign * value

    def _button(self, key):
        """Return True if a central button is held."""
        index = self.button_map.get(key)

        if index is None or index >= self.num_buttons:
            return False

        return bool(self.joy.get_button(index))

    def _right_button(self, key):
        """Return True if a right-side directional button is held."""
        index = self.right_map.get(key)

        if index is None or index >= self.num_buttons:
            return False

        return bool(self.joy.get_button(index))

    def _trigger(self, key):
        """Return True if a trigger button is held."""
        index = self.trigger_map.get(key)

        if index is None or index >= self.num_buttons:
            return False

        return bool(self.joy.get_button(index))

    def read_joint_velocities(self):
        """
        Return direct joint velocities for all six UR3 joints.

        Returns:
            np.array([dq1, dq2, dq3, dq4, dq5, dq6]) in rad/s.
        """

        dq = np.zeros(6)

        # ---- J1: left D-pad/joystick left-right ----
        dq[0] = self._axis("x") * self.joint_speed

        # ---- J2: left D-pad/joystick up-down ----
        dq[1] = self._axis("y") * self.joint_speed

        # ---- J3: left triggers ----
        if self._trigger("top_left"):
            dq[2] += self.joint_speed

        if self._trigger("bottom_left"):
            dq[2] -= self.joint_speed

        # ---- J4: right D-pad/joystick up-down ----
        if self._right_button("up"):
            dq[3] += self.joint_speed

        if self._right_button("down"):
            dq[3] -= self.joint_speed

        # ---- J5: right D-pad/joystick left-right ----
        if self._right_button("left"):
            dq[4] += self.joint_speed

        if self._right_button("right"):
            dq[4] -= self.joint_speed

        # ---- J6: right triggers ----
        if self._trigger("top_right"):
            dq[5] += self.joint_speed

        if self._trigger("bottom_right"):
            dq[5] -= self.joint_speed

        # Smooth all joint commands for less jerky motion.
        self._filtered_dq = (
            self.smoothing * self._filtered_dq
            + (1.0 - self.smoothing) * dq
        )

        return self._filtered_dq

    def handle_buttons(self, reset_callback=None, shoot_callback=None):
        """
        Handle central buttons.

        reset_callback: called when Select is pressed.
        shoot_callback: reserved for Start; add shooting later.
        """

        now = time.time()

        if now - self._last_button_time < self.button_cooldown:
            return

        if self._button("select") and reset_callback is not None:
            reset_callback()
            self._last_button_time = now

        elif self._button("start") and shoot_callback is not None:
            shoot_callback()
            self._last_button_time = now

    def close(self):
        pygame.quit()