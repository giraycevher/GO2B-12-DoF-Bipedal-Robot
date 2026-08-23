import math
from dataclasses import dataclass

import numpy as np

from config.parameters import AnkleFeedbackConfig, DCMConfig, RobotConfig

GRAVITY = 9.81


@dataclass
class RecoveryOutput:
    d_lat: float = 0.0
    d_fwd: float = 0.0
    err_lat: float = 0.0
    err_fwd: float = 0.0

    @property
    def as_tuple(self):
        return self.d_lat, self.d_fwd


class DCMRecovery:

    def __init__(self, cfg: DCMConfig, robot: RobotConfig):
        self.cfg = cfg
        self.mid_x = robot.mid_x
        self.hip_y = robot.hip_y

    def to_walk_frame(self, p, v):
        lat = float(p[0]) - self.mid_x
        fwd = self.hip_y - float(p[1])
        v_lat = float(v[0])
        v_fwd = -float(v[1])
        return lat, fwd, v_lat, v_fwd

    def compute(self, com_pos, com_vel, plan_lat, plan_fwd,
                swing_phase: float = 0.0) -> RecoveryOutput:
        if not self.cfg.enabled or self.cfg.gain <= 0.0:
            return RecoveryOutput()

        z = max(0.05, float(com_pos[2]))
        omega = math.sqrt(GRAVITY / z)

        lat, fwd, v_lat, v_fwd = self.to_walk_frame(com_pos, com_vel)

        err_lat = (lat + v_lat / omega) - (plan_lat[0] + plan_lat[1] / omega)
        err_fwd = (fwd + v_fwd / omega) - (plan_fwd[0] + plan_fwd[1] / omega)

        k = self.cfg.gain
        d_lat = float(np.clip(k * err_lat, -self.cfg.limit_lat, self.cfg.limit_lat))
        d_fwd = float(np.clip(k * err_fwd, -self.cfg.limit_fwd, self.cfg.limit_fwd))

        f0 = self.cfg.fade_start
        if swing_phase >= f0:
            fade = max(0.0, (1.0 - swing_phase) / (1.0 - f0))
            d_lat *= fade
            d_fwd *= fade

        return RecoveryOutput(d_lat, d_fwd, err_lat, err_fwd)


class AnkleStabilizer:

    def __init__(self, cfg: AnkleFeedbackConfig):
        self.cfg = cfg

    def compute(self, base_R, ang_vel):
        if not self.cfg.enabled:
            return 0.0, 0.0
        pitch = math.asin(float(np.clip(base_R[1, 2], -1.0, 1.0)))
        roll = math.asin(float(np.clip(base_R[0, 2], -1.0, 1.0)))
        c = self.cfg.clamp
        d_pitch = self.cfg.sign * float(np.clip(
            self.cfg.kp * pitch + self.cfg.kd * ang_vel[0], -c, c))
        d_roll = self.cfg.sign * float(np.clip(
            self.cfg.kp * roll + self.cfg.kd * ang_vel[1], -c, c))
        return d_pitch, d_roll
