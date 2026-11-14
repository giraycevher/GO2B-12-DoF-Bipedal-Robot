from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

#: Repository root (the folder that holds main.py and robot_walk.xml).
#: Every default path is resolved against it, so the scripts work no matter
#: which folder they are launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path) -> str:
    """Return `path` as an absolute string.

    Relative paths are tried against the current working directory first and
    then against PROJECT_ROOT, so both `python main.py` (run inside the repo)
    and `python path/to/repo/main.py` (run from anywhere) find robot_walk.xml.
    """
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return str(p.resolve())
    return str((PROJECT_ROOT / p).resolve())


@dataclass
class RobotConfig:
    xml_path: str = str(PROJECT_ROOT / "robot_walk.xml")

    torso_body: str = "torso"
    foot_bodies: Tuple[str, str] = ("left_foot", "right_foot")
    ball_joint: str = "push_ball_free"
    ball_body: str = "push_ball"

    joint_order: Tuple[str, ...] = (
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
        "left_knee", "left_ankle_pitch", "left_ankle_roll",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
        "right_knee", "right_ankle_pitch", "right_ankle_roll",
    )

    hip_center_left: np.ndarray = field(
        default_factory=lambda: np.array([-0.080434, 0.067676, -0.050196]))
    mirror_x: float = -0.0054341

    a_leg: float = 0.086407
    b_leg: float = 0.088407

    ankle_center_in_foot: np.ndarray = field(
        default_factory=lambda: np.array([0.000007, 0.0, -0.024512]))

    r_foot_level: np.ndarray = field(
        default_factory=lambda: np.array([[0.0, 1.0, 0.0],
                                          [0.0, 0.0, 1.0],
                                          [1.0, 0.0, 0.0]]))

    sign_map: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, -1.0, 1.0, -1.0, -1.0]))

    q_min: np.ndarray = field(
        default_factory=lambda: np.array([-0.60, -0.80, -1.80, -0.05, -1.50, -0.80]))
    q_max: np.ndarray = field(
        default_factory=lambda: np.array([+0.60, +0.80, +1.80, +2.60, +1.50, +0.80]))

    joint_limits: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "hip_yaw": (-0.60, 0.60),
        "hip_roll": (-0.80, 0.80),
        "hip_pitch": (-1.80, 1.80),
        "knee": (-0.05, 2.60),
        "ankle_pitch": (-1.50, 1.50),
        "ankle_roll": (-0.80, 0.80),
    })

    actuator_gains: Dict[str, Tuple[float, float, float]] = field(default_factory=lambda: {
        "hip_yaw": (80.0, 4.0, 8.0),
        "hip_roll": (150.0, 6.0, 12.0),
        "hip_pitch": (200.0, 8.0, 15.0),
        "knee": (200.0, 8.0, 15.0),
        "ankle_pitch": (150.0, 6.0, 12.0),
        "ankle_roll": (100.0, 5.0, 10.0),
    })

    material_density: float = 1240.0
    min_robot_mass: float = 0.05

    def __post_init__(self):
        self.hip_center_right = np.array([
            2.0 * self.mirror_x - self.hip_center_left[0],
            self.hip_center_left[1],
            self.hip_center_left[2],
        ])
        self.body_ref_offset = np.array([self.mirror_x,
                                         self.hip_center_left[1],
                                         self.hip_center_left[2]])
        self.max_reach = self.a_leg + self.b_leg

        self.m2i = np.array([[0.0, -1.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0]])
        self.i2m = self.m2i.T

        body_ik_y = float((self.m2i @ self.body_ref_offset)[1])
        self.d_left = float((self.m2i @ self.hip_center_left)[1]) - body_ik_y
        self.d_right = float((self.m2i @ self.hip_center_right)[1]) - body_ik_y

    @property
    def mid_x(self) -> float:
        return self.mirror_x

    @property
    def hip_y(self) -> float:
        return float(self.body_ref_offset[1])


@dataclass
class GaitConfig:
    z_hip: float = 0.195
    squat_start_z: float = 0.215
    squat_time: float = 1.2
    settle_time: float = 0.4

    half_stance: float = 0.075

    dt_mpc: float = 0.01
    n_preview: int = 160
    step_time: int = 45
    initial_delay: int = 60
    step_len: float = 0.04
    step_height: float = 0.022
    n_steps: int = 20

    @property
    def lat_a(self) -> float:
        return -self.half_stance

    @property
    def lat_b(self) -> float:
        return +self.half_stance

    @property
    def stop_tick(self) -> int:
        return self.initial_delay + (self.n_steps - 1) * self.step_time

    @property
    def first_step_time(self) -> float:
        return self.settle_time + self.squat_time + self.initial_delay * self.dt_mpc


@dataclass
class DCMConfig:
    enabled: bool = True
    gain: float = 1.0
    limit_lat: float = 0.045
    limit_fwd: float = 0.060
    fade_start: float = 0.80


@dataclass
class PushConfig:
    time: float = 6.0
    duration: float = 0.1
    force: float = 0.0
    axis: str = "lat"
    viz: str = "arrow"
    arrow_hold: float = 0.6


@dataclass
class AnkleFeedbackConfig:
    enabled: bool = False
    kp: float = 0.60
    kd: float = 0.06
    clamp: float = 0.30
    sign: float = 1.0


@dataclass
class TerrainConfig:
    """Procedural forest-floor heightfield (see sim/terrain.py)."""
    rough: float = 0.6          # 0 = flat, 0.6 = default, 1.0 = hard, 1.5 = very hard
    seed: int = 3
    size_x: float = 1.2         # lateral width [m]
    behind: float = 0.40        # length behind the start position [m]
    ahead: float = 1.80         # length ahead of the start position [m]
    resolution: float = 0.005   # grid spacing [m]
    flat_ahead: float = 0.10    # flat zone in front of the start [m]
    blend_len: float = 0.14     # flat -> rough transition length [m]
    friction: float = 1.0       # soil friction (overrides the 1.2 of the feet)
    solref: float = 0.008       # contact time constant [s]; larger = softer soil
    check_tolerance: float = 0.002   # max allowed map/hfield mismatch [m]


@dataclass
class TerrainGaitConfig:
    """Terrain-aware gait (see planners/terrain_planner.py).

    Values are the ones validated in the rough-terrain experiments; they differ
    from GaitConfig on purpose (lower hip, shorter and slower steps)."""
    z_hip: float = 0.185        # hip-center height above the local ground [m]
    squat_extra: float = 0.020  # start this much higher, then squat down [m]
    squat_time: float = 1.2
    settle_time: float = 0.4
    half_stance: float = 0.075
    dt_mpc: float = 0.01
    n_preview: int = 160
    step_time: int = 90         # control ticks per step (90 x 0.01 s = 0.9 s)
    initial_delay: int = 60
    step_len: float = 0.03
    step_height: float = 0.028  # net clearance above the terrain [m]
    n_steps: int = 24
    # terrain adaptation
    use_map: bool = True        # False = blind: no height map, contact only
    use_slope: bool = True      # tilt the sole to the local terrain plane
    max_tilt: float = 0.14      # sole tilt limit [rad] (~8 deg)
    use_touchdown: bool = True  # freeze the descent on early contact
    touch_frac: float = 0.60    # accept early contact after this swing fraction
    touch_force_ratio: float = 0.25  # contact threshold as a fraction of m*g
    td_window: float = 0.012    # accept contact only this close to the expected ground [m]
    reach_frac: float = 0.95    # max leg length / max reach
    ground_tau: float = 0.30    # body-height ground reference time constant [s]
    # DCM foot placement (off by default: it did not help in the experiments)
    k_dcm: float = 0.0
    dcm_lim_lat: float = 0.045
    dcm_lim_fwd: float = 0.060
    dcm_start: float = 0.25
    dcm_freeze: float = 0.80
    dcm_hold: bool = True
    dcm_deadband: float = 0.008
    zmp_follow: bool = True     # move the ZMP reference with the foot
    # ankle (CoP) balance loop
    ankle_balance: bool = False
    ankle_kp: float = 0.0
    ankle_kd: float = 0.0
    ankle_clamp: float = 0.10


@dataclass
class SimConfig:
    max_time: float = 14.0
    view_dt: float = 0.02
    speed: float = 1.0
    headless: bool = False

    record: str = ""
    fps: int = 30
    width: int = 1280
    height: int = 720

    plot_path: str = "walk_trajectory.png"
    report_period: float = 1.0
    fall_height: float = 0.08
    max_ik_failures: int = 3
    refine_ik: bool = False
    plot_enabled: bool = True

    robot: RobotConfig = field(default_factory=RobotConfig)
    gait: GaitConfig = field(default_factory=GaitConfig)
    dcm: DCMConfig = field(default_factory=DCMConfig)
    push: PushConfig = field(default_factory=PushConfig)
    ankle: AnkleFeedbackConfig = field(default_factory=AnkleFeedbackConfig)
