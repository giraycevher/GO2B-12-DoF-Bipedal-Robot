"""Kajita analitik bacak IK'si + ileri kinematik. Sadece numpy."""

import math
from typing import Optional, Tuple

import numpy as np

from config.parameters import RobotConfig


def rot_roll(q: float) -> np.ndarray:
    c, s = math.cos(q), math.sin(q)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_pitch(q: float) -> np.ndarray:
    c, s = math.cos(q), math.sin(q)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(q: float) -> np.ndarray:
    c, s = math.cos(q), math.sin(q)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def ik_leg_kajita(body_p, body_R, D, a_leg, b_leg, foot_p, foot_R):
    """Kajita, Introduction to Humanoid Robotics -- analitik 6-DOF bacak IK'si."""
    hip_offset = body_R @ np.array([0.0, D, 0.0])
    r = foot_R.T @ (body_p + hip_offset - foot_p)
    c_len = np.linalg.norm(r)

    c5 = np.clip((c_len ** 2 - a_leg ** 2 - b_leg ** 2) / (2.0 * a_leg * b_leg), -1.0, 1.0)
    q5 = math.acos(c5)

    q6a = math.asin(np.clip((a_leg / c_len) * math.sin(np.pi - q5), -1.0, 1.0))
    q7 = math.atan2(r[1], r[2])
    if q7 > math.pi / 2:
        q7 -= math.pi
    elif q7 < -math.pi / 2:
        q7 += math.pi
    q6 = -math.atan2(r[0], np.sign(r[2]) * math.sqrt(r[1] ** 2 + r[2] ** 2)) - q6a

    m = body_R.T @ foot_R @ rot_roll(-q7) @ rot_pitch(-q6 - q5)
    q2 = math.atan2(-m[0, 1], m[1, 1])
    cz, sz = math.cos(q2), math.sin(q2)
    q3 = math.atan2(m[2, 1], -m[0, 1] * sz + m[1, 1] * cz)
    q4 = math.atan2(-m[2, 0], m[2, 2])

    return np.array([q2, q3, q4, q5, q6, q7])


class LegKinematics:
    """Bacak zinciri: ileri kinematik, ters kinematik, opsiyonel DLS cila.

    Konumlar GOVDE (torso) cercevesinde, model eksenlerinde:
    X = yanal, Y = ileri yonun tersi, Z = yukari.
    IK hedefi ayak bilegi EKSEN KESISIMIDIR, ayak govdesinin orijini degil.
    """

    # (yerel pos, yerel quat) -- her eklem kendi cercevesinde +Z etrafinda doner
    CHAIN = (
        ((-0.0804341, 0.0679133, -0.000988717), (1.0, 0.0, 0.0, 0.0)),
        ((0.0, 0.0240365, -0.0492), (0.5, 0.5, 0.5, -0.5)),
        ((1.35932e-05, 0.0181133, 0.0245116), (0.707107, 0.707107, 0.0, 0.0)),
        ((0.0864, 0.0, 0.03655), (0.0, 0.0, 1.0, 0.0)),
        ((-0.0884, 0.0, 0.0), (0.0, 1.0, 3.0135e-08, 0.0)),
        ((-1.35932e-05, 0.0245116, -0.0181133), (0.707107, -0.707107, 0.0, 0.0)),
    )
    HIP_X_LEFT = -0.0804341
    HIP_X_RIGHT = +0.0695659

    def __init__(self, cfg: Optional[RobotConfig] = None):
        self.cfg = cfg or RobotConfig()
        self._local_p = [np.array(p) for p, _ in self.CHAIN]
        self._local_r = [quat_to_mat(q) for _, q in self.CHAIN]

    # ---------------------------------------------------------------- FK
    def forward(self, q, left: bool = True):
        p = np.zeros(3)
        R = np.eye(3)
        origins, axes = [], []
        for i in range(6):
            lp = self._local_p[i]
            if i == 0 and not left:
                lp = np.array([self.HIP_X_RIGHT, lp[1], lp[2]])
            p = p + R @ lp
            R = R @ self._local_r[i]
            origins.append(p.copy())
            axes.append(R[:, 2].copy())
            R = R @ rot_z(q[i])
        return p, R, origins, axes

    def ankle_center(self, q, left: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        p, R, _, _ = self.forward(q, left)
        return p + R @ self.cfg.ankle_center_in_foot, R

    # ---------------------------------------------------------------- IK
    def solve(self, foot_p, left: bool = True, foot_R=None,
              clamp: bool = True, refine: bool = False) -> np.ndarray:
        c = self.cfg
        foot_p = np.asarray(foot_p, dtype=float)
        if foot_R is None:
            foot_R = c.r_foot_level

        chain_R = np.asarray(foot_R, dtype=float) @ c.r_foot_level.T
        q = ik_leg_kajita(c.m2i @ c.body_ref_offset, np.eye(3),
                          c.d_left if left else c.d_right,
                          c.a_leg, c.b_leg,
                          c.m2i @ foot_p, c.m2i @ chain_R @ c.i2m)
        q = c.sign_map * q

        if refine:
            q = self._refine(q, foot_p, np.asarray(foot_R, float), left)
        if clamp:
            q = np.clip(q, c.q_min, c.q_max)
        return q

    def _refine(self, q, target_p, target_R, left, iters: int = 8, tol: float = 1e-5):
        q = np.array(q, dtype=float)
        lam = 1e-4
        for _ in range(iters):
            p, R, origins, axes = self.forward(q, left)
            ac = p + R @ self.cfg.ankle_center_in_foot
            err = np.zeros(6)
            err[:3] = target_p - ac
            re = target_R @ R.T
            w = np.array([re[2, 1] - re[1, 2], re[0, 2] - re[2, 0], re[1, 0] - re[0, 1]])
            s = np.linalg.norm(w)
            if s > 1e-9:
                err[3:] = w / s * math.atan2(s, np.trace(re) - 1.0)
            if np.linalg.norm(err) < tol:
                break
            J = np.zeros((6, 6))
            for i in range(6):
                J[:3, i] = np.cross(axes[i], ac - origins[i])
                J[3:, i] = axes[i]
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)
            m = np.max(np.abs(dq))
            if m > 0.3:
                dq *= 0.3 / m
            q = q + dq
        return q

    # ------------------------------------------------------------ yardim
    def reach_ratio(self, leg_drop: float) -> float:
        return leg_drop / self.cfg.max_reach
