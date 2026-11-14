"""Kinematic dry run of the terrain-aware gait (no physics)."""
import math

import pytest

from config.parameters import TerrainGaitConfig
from planners.terrain_planner import dry_run


@pytest.mark.parametrize("rough", [0.0, 1.0])
def test_dry_run_feet_follow_the_terrain(rough):
    g = TerrainGaitConfig(n_steps=8)
    r = dry_run(rough, gait=g)
    assert r["ik_err"] < 1e-3                 # IK reaches every commanded ankle point
    assert r["penetration"] < 2e-3            # soles never sink > 2 mm into the terrain
    assert r["at_limit"] == 0                 # no joint hits its limit
    assert r["dcm_corr"] == 0.0               # perfect tracking -> no DCM correction
    assert math.degrees(r["joint_jump"]) < 5  # no joint jumps > 5 deg per 10 ms
    assert r["distance"] > 0.15               # it actually walked the plan
