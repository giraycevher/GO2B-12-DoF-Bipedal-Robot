"""Terrain generator: shape, flat start zone, interpolation, MuJoCo export."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from config.parameters import PROJECT_ROOT, RobotConfig
from sim.terrain import Terrain

HIP_Y = RobotConfig().hip_y


def test_flat_when_rough_is_zero():
    t = Terrain(rough=0.0, start_y=HIP_Y)
    assert np.all(t.h == 0.0)


def test_start_zone_is_flat_and_rough_ahead():
    t = Terrain(rough=1.0, start_y=HIP_Y)
    xs = np.linspace(-0.2, 0.2, 41)
    assert np.max(np.abs(t.height(xs, np.full_like(xs, HIP_Y)))) < 1e-9
    ahead = t.height(xs, np.full_like(xs, HIP_Y - 0.6))
    assert np.std(ahead) > 1e-3                         # millimetre-scale roughness


def test_same_seed_same_terrain():
    a = Terrain(rough=1.0, seed=5, start_y=HIP_Y)
    b = Terrain(rough=1.0, seed=5, start_y=HIP_Y)
    c = Terrain(rough=1.0, seed=6, start_y=HIP_Y)
    assert np.array_equal(a.h, b.h)
    assert not np.array_equal(a.h, c.h)


def test_bilinear_is_exact_on_grid_nodes_and_linear_between():
    t = Terrain(rough=1.0, start_y=HIP_Y)
    r, c = 120, 80
    x0, y0 = t.x_min + c * t.res, t.y_min + r * t.res
    assert abs(t.height(x0, y0) - t.h[r, c]) < 1e-12
    mid = t.height(x0 + 0.5 * t.res, y0)
    assert abs(mid - 0.5 * (t.h[r, c] + t.h[r, c + 1])) < 1e-12


def test_hfield_data_layout_matches_height():
    t = Terrain(rough=1.0, start_y=HIP_Y)
    d = t.hfield_data()
    r, c = 200, 50
    assert abs(d[r * t.ncol + c] * t.elev + t.h_min
               - t.height(t.x_min + c * t.res, t.y_min + r * t.res)) < 1e-6


def test_write_xml_replaces_floor_and_drops_ball(tmp_path):
    src = PROJECT_ROOT / "robot_walk.xml"
    dst = tmp_path / "terrain.xml"
    Terrain(rough=0.6, start_y=HIP_Y).write_xml(str(src), str(dst))
    root = ET.parse(dst).getroot()
    floor = [g for g in root.iter("geom") if g.get("name") == "floor"][0]
    assert floor.get("type") == "hfield"
    assert any(h.get("name") == "soil" for h in root.iter("hfield"))
    assert not any(b.get("name") == "push_ball" for b in root.iter("body"))


def test_mujoco_heightfield_matches_height_map():
    mujoco = pytest.importorskip("mujoco")
    t = Terrain(rough=1.0, start_y=HIP_Y)
    dst = PROJECT_ROOT / "robot_terrain_test.xml"     # next to robot_walk.xml (meshdir)
    try:
        t.write_xml(str(PROJECT_ROOT / "robot_walk.xml"), str(dst))
        model = mujoco.MjModel.from_xml_path(str(dst))
        t.apply_to_model(model)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        worst = t.verify(model, data)
        assert worst is not None and worst < 0.002
    finally:
        dst.unlink(missing_ok=True)
