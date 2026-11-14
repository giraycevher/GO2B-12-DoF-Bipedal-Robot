"""Kajita analytical IK must agree with forward kinematics on the mesh model."""
import math

import numpy as np
import pytest

from config.parameters import RobotConfig
from kinematics.leg_ik import LegKinematics
from planners.terrain_planner import tilt_rotation

CFG = RobotConfig()
KIN = LegKinematics(CFG)


@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("drop", [0.10, 0.13, 0.155])
@pytest.mark.parametrize("fwd", [-0.04, 0.0, 0.04])
def test_ik_fk_round_trip_level_foot(left, drop, fwd):
    hc = CFG.hip_center_left if left else CFG.hip_center_right
    target = hc + np.array([0.0, -fwd, -drop])
    q = KIN.solve(target, left=left)
    got, R = KIN.ankle_center(q, left)
    assert np.linalg.norm(got - target) < 1e-3          # < 1 mm (CAD tolerance ~0.4 mm)
    # sole level: foot orientation equals the level orientation
    ang = math.acos(np.clip((np.trace(R @ CFG.r_foot_level.T) - 1) / 2, -1, 1))
    assert ang < 1e-3


@pytest.mark.parametrize("a,b", [(0.05, 0.0), (0.0, -0.05), (0.08, 0.06)])
def test_ik_reaches_tilted_sole(a, b):
    R_t, _ = tilt_rotation(a, b)
    target = CFG.hip_center_left + np.array([0.0, 0.0, -0.13])
    q = KIN.solve(target, left=True, foot_R=R_t @ CFG.r_foot_level)
    got, R = KIN.ankle_center(q, True)
    assert np.linalg.norm(got - target) < 1e-3
    err = R @ (R_t @ CFG.r_foot_level).T
    assert math.acos(np.clip((np.trace(err) - 1) / 2, -1, 1)) < 2e-3


def test_tilt_rotation_maps_ez_to_normal():
    R, n = tilt_rotation(0.0333, 0.0)
    assert np.allclose(R @ np.array([0, 0, 1.0]), n, atol=1e-12)
    assert np.allclose(n, [-0.0333 / math.sqrt(1 + 0.0333 ** 2), 0, 1 / math.sqrt(1 + 0.0333 ** 2)])
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)


def test_legs_are_symmetric():
    assert abs(CFG.d_left + CFG.d_right) < 1e-9
    assert abs(CFG.max_reach - 0.174814) < 1e-6


@pytest.mark.parametrize("left", [True, False])
def test_ankle_joint_sign_convention(left):
    """+ankle_pitch lifts the toe, +ankle_roll lowers the +X edge (used by the
    ankle balance loop in planners/terrain_planner.py)."""
    hc = CFG.hip_center_left if left else CFG.hip_center_right
    q = KIN.solve(hc + np.array([0.0, 0.0, -0.13]), left=left)

    def edge_heights(q):
        p, R, _, _ = KIN.forward(q, left)
        a = p + R @ CFG.ankle_center_in_foot
        toe = a + R @ (CFG.r_foot_level.T @ np.array([0.0, -0.07, 0.0]))
        side = a + R @ (CFG.r_foot_level.T @ np.array([0.04, 0.0, 0.0]))
        return toe[2] - a[2], side[2] - a[2]

    toe0, side0 = edge_heights(q)
    dq = np.zeros(6)
    dq[4] = 0.05
    assert edge_heights(q + dq)[0] > toe0 + 1e-3
    dq = np.zeros(6)
    dq[5] = 0.05
    assert edge_heights(q + dq)[1] < side0 - 1e-3
