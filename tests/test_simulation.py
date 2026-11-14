"""Short headless runs of both entry points, launched from a foreign folder."""
import subprocess
import sys

import pytest

from config.parameters import PROJECT_ROOT

pytest.importorskip("mujoco")


def _run(script, args, cwd):
    return subprocess.run([sys.executable, str(PROJECT_ROOT / script), "--headless",
                           "--no-plot", *args], cwd=cwd, capture_output=True, text=True,
                          timeout=600)


def test_model_loads():
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "robot_walk.xml"))
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    assert m.nu == 12
    assert 2.5 < m.body_subtreemass[tid] < 3.5


def test_main_runs_from_another_folder(tmp_path):
    out = _run("main.py", ["--time", "4"], tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "WALKED / STAYED UP" in out.stdout


def test_main_terrain_runs_from_another_folder(tmp_path):
    out = _run("main_terrain.py", ["--rough", "0.6", "--steps", "4", "--time", "5"], tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "height map = heightfield" in out.stdout
    assert "WALKED / STAYED UP" in out.stdout
