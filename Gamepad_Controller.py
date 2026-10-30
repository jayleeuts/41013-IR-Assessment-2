# ------------------------------------GAMEPAD CONTROLLER------------------------------------------------------
# The gamepad reader for Controller mode. Each stick, button pair or trigger pair drives one UR3 joint
# directly (like an excavator: no IK), smoothed so the motion isn't jerky. Select resets the UR3 and Start
# shoots.

import time

import numpy as np
import pygame


# ------------------------------------BUTTON LAYOUT-----------------------------------------------------------
# J1 / J2: left stick (axes 0 / 1). J3: left triggers, J4: right pad up/down, J5: right pad left/right,
# J6: right triggers - each as (button that turns it +, button that turns it -). Select = 8, Start = 9.

JOINT_BUTTONS = {2: (4, 6), 3: (0, 2), 4: (1, 3), 5: (5, 7)}
SELECT, START = 8, 9


class GamepadController:
    def __init__(self, joystick_index=0, deadzone=0.15, joint_speed=np.radians(45.0), smoothing=0.25):
        self.deadzone, self.joint_speed, self.smoothing = deadzone, joint_speed, smoothing
        self._filtered_dq = np.zeros(6)
        self._last_button_time = 0.0
        self.button_cooldown = 0.30
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected.")
        self.joy = pygame.joystick.Joystick(joystick_index)
        self.joy.init()
        self.num_buttons, self.num_axes = self.joy.get_numbuttons(), self.joy.get_numaxes()
        print(f"Joystick: {self.joy.get_name()} ({self.num_buttons} buttons, {self.num_axes} axes)")

    def update(self):
        pygame.event.pump()

    def _axis(self, axis):
        value = self.joy.get_axis(axis) if axis < self.num_axes else 0.0
        return 0.0 if abs(value) < self.deadzone else value

    def _button(self, index):
        return index < self.num_buttons and bool(self.joy.get_button(index))

    def read_joint_velocities(self):
        dq = np.zeros(6)
        dq[0], dq[1] = self._axis(0) * self.joint_speed, self._axis(1) * self.joint_speed
        for joint, (plus, minus) in JOINT_BUTTONS.items():
            dq[joint] = self.joint_speed * (self._button(plus) - self._button(minus))
        self._filtered_dq = self.smoothing * self._filtered_dq + (1.0 - self.smoothing) * dq
        return self._filtered_dq

    def handle_buttons(self, reset_callback=None, shoot_callback=None):
        now = time.time()
        if now - self._last_button_time < self.button_cooldown:
            return
        if self._button(SELECT) and reset_callback is not None:
            reset_callback()
            self._last_button_time = now
        elif self._button(START) and shoot_callback is not None:
            shoot_callback()
            self._last_button_time = now
