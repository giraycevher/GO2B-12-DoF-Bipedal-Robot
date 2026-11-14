"""The scripts must find robot_walk.xml and the packages from any working folder."""
import os
import subprocess
import sys
from pathlib import Path

from config.parameters import PROJECT_ROOT, RobotConfig, resolve_path


def test_default_model_path_is_absolute_and_exists():
    p = Path(RobotConfig().xml_path)
    assert p.is_absolute()
    assert p.exists()


def test_resolve_path_from_another_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Path(resolve_path("robot_walk.xml")) == PROJECT_ROOT / "robot_walk.xml"


def test_resolve_path_prefers_existing_cwd_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "robot_walk.xml").write_text("<mujoco/>")
    assert Path(resolve_path("robot_walk.xml")) == tmp_path / "robot_walk.xml"


def test_planner_module_runs_as_a_script_from_anywhere(tmp_path):
    """`python planners/terrain_planner.py` must import config/, kinematics/ ..."""
    code = ("import runpy, sys; sys.argv=['x']; "
            f"m = runpy.run_path(r'{PROJECT_ROOT / 'planners' / 'terrain_planner.py'}', run_name='lib'); "
            "print('ok' if 'TerrainStepPlanner' in m else 'missing')")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                         capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ""})
    assert out.stdout.strip() == "ok", out.stderr
