"""
planners/terrain_planner.py
===========================
Terrain-aware gait for uneven ground. It keeps the same ZMP preview control
(controllers/preview_controller.py) and Kajita IK (kinematics/leg_ik.py) as
main.py and adds a terrain-adaptation layer on top:

  1. Terrain-aware swing     each foot keeps its own ground height; the swing
                             foot clears the highest point on its path and
                             lands at the terrain height of its target.
  2. Slope-adapted foot      a least-squares plane is fitted to the terrain
                             under the footprint; the sole is commanded
                             parallel to it.
  3. Early touchdown         if the swing foot touches down early, the descent
                             is frozen and the real ground height is measured
                             from the joint angles (forward kinematics).
  4. Reach guard             the hip is lowered if a leg would exceed
                             reach_frac * max reach (no knee lock-out).
  5. Plan re-anchoring       the ZMP reference and the remaining footsteps
                             follow where the foot actually landed.
  6. Optional DCM placement  foot placement from the capture-point error,
                             measured against the TRUE CoM of the commanded
                             posture (off by default, see README).
  7. Ankle balance loop      CoP-style correction of the stance-ankle angles
                             from the DCM error (on by default, see README).

This module does not import MuJoCo. Run it directly for a kinematic dry run
(IK -> FK foot/terrain consistency check, no physics):

    python planners/terrain_planner.py

Derivations and numerical examples for every step are in
docs/rough_terrain_methods.pdf.

Frames: "walking frame" = (lateral, forward, up). Model axes: X = lateral,
Y = backward (the robot faces model -Y), Z = up.
"""

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

if __package__ in (None, ""):                     # allow `python planners/terrain_planner.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.parameters import RobotConfig, TerrainGaitConfig      # noqa: E402
from controllers.preview_controller import ZMPPreviewController   # noqa: E402
from kinematics.leg_ik import LegKinematics                        # noqa: E402

REF_LEN = 20000          # length of the ZMP reference buffers (control ticks)


@dataclass
class Footprint:
    """Sole bounds relative to the ankle-center projection [m], walking frame.

    Measured from the collision meshes by MujocoRobot.foot_sole_bounds()."""
    fwd_min: float = -0.030     # heel
    fwd_max: float = +0.078     # toe
    lat_min: float = -0.046
    lat_max: float = +0.046

    def samples(self, n_fwd: int = 5, n_lat: int = 4):
        f = np.linspace(self.fwd_min, self.fwd_max, n_fwd)
        lat = np.linspace(self.lat_min, self.lat_max, n_lat)
        L, F = np.meshgrid(lat, f)
        return L.ravel(), F.ravel()

    @property
    def width(self) -> float:
        return self.lat_max - self.lat_min


@dataclass
class Foot:
    pos: np.ndarray                      # [lateral, forward] of the sole reference point
    g: float = 0.0                       # ground (sole) height under it, world z
    tilt: np.ndarray = field(default_factory=lambda: np.zeros(2))   # (a, b) = dz/dX, dz/dY
    lift: float = 0.0


def tilt_rotation(a: float, b: float):
    """Rotation that takes e_z to the normal of z = aX + bY (Rodrigues), and the normal."""
    n = np.array([-a, -b, 1.0])
    n /= np.linalg.norm(n)
    k = np.array([-n[1], n[0], 0.0])             # e_z x n
    s = np.linalg.norm(k)
    if s < 1e-12:
        return np.eye(3), n
    k /= s
    ang = math.atan2(s, n[2])
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
    return R, n


class TerrainStepPlanner:
    """Terrain-aware walking controller (preview control + footstep + IK targets).

    Usage (see main_terrain.py):
        planner = TerrainStepPlanner(gait_cfg, robot_cfg, terrain, ankle_h, fp_a, fp_b)
        every dt_mpc:     planner.tick(com, vel, touch_a, touch_b, ank_dz, tilt)
        every sim step:   q_a, q_b, torso_w = planner.joint_targets(z_now)
    """

    def __init__(self, gait: TerrainGaitConfig, robot: Optional[RobotConfig] = None,
                 terrain=None, ankle_h: float = 0.0567,
                 fp_a: Optional[Footprint] = None, fp_b: Optional[Footprint] = None,
                 com_model: Optional[Callable] = None, touch_force: float = 7.3,
                 refine_ik: bool = False):
        """com_model: (torso_w, q_a, q_b) -> whole-body CoM (world). If given, the
        DCM error is measured against the true CoM of the COMMANDED posture
        instead of the LIPM state (the legs are heavy, the LIPM CoM is not the
        real CoM)."""
        self.p = p = gait
        self.rcfg = robot or RobotConfig()
        self.ik = LegKinematics(self.rcfg)
        self.terrain = terrain
        self.ankle_h = float(ankle_h)
        self.fp: Dict[str, Footprint] = {"A": fp_a or Footprint(), "B": fp_b or Footprint()}
        self.com_model = com_model
        self.touch_force = float(touch_force)
        self.refine_ik = refine_ik
        self.mid_x = self.rcfg.mid_x
        self.hip_y = self.rcfg.hip_y

        self.lat_a, self.lat_b = -p.half_stance, +p.half_stance
        # footsteps: [forward, lateral]
        self.footsteps = np.array(
            [[i * p.step_len, self.lat_a if i % 2 == 0 else self.lat_b]
             for i in range(p.n_steps)])
        self.stop_tick = p.initial_delay + (p.n_steps - 1) * p.step_time
        self._build_zmp_reference()

        self.preview_lat = ZMPPreviewController(p.dt_mpc, p.z_hip, p.n_preview)
        self.preview_fwd = ZMPPreviewController(p.dt_mpc, p.z_hip, p.n_preview)
        self.step_index = 0

        self.feet = {"A": Foot(np.array([self.lat_a, 0.0])),
                     "B": Foot(np.array([self.lat_b, 0.0]))}
        for name in ("A", "B"):
            g, a, b = self._support(name, self.feet[name].pos)
            self.feet[name].g = g
            self.feet[name].tilt = np.array([a, b])
        self.start: Optional[Foot] = None       # copy of the swing foot at liftoff
        self.swing: Optional[str] = None
        self.td_lock = False
        self.g_ref = 0.5 * (self.feet["A"].g + self.feet["B"].g)
        self.reach_drop = 0.0

        self.omega = math.sqrt(9.81 / p.z_hip)
        self.corr = np.zeros(2)                 # applied DCM correction [lat, fwd]
        self.raw = np.zeros(2)                  # raw DCM error (bias removed)
        self.bias = np.zeros(2)
        self._bias_sum = np.zeros(2)
        self._bias_n = 0
        self.plan_com = None
        self.plan_vel = None
        self.early_td_count = 0
        self.ankle_corr = np.zeros(2)           # [pitch, roll] ankle correction [rad]

    # ---------------------------------------------------------------- helpers
    def to_model(self, lat, fwd, up):
        """Walking frame (lateral, forward, up) -> world / model axes."""
        return np.array([self.mid_x + lat, self.hip_y - fwd, up])

    def _build_zmp_reference(self):
        p = self.p
        self.p_ref_fwd = np.zeros(REF_LEN)
        self.p_ref_lat = np.zeros(REF_LEN)
        for i in range(p.n_steps):
            s = p.initial_delay + i * p.step_time
            e = min(s + p.step_time, REF_LEN)
            if s < REF_LEN:
                self.p_ref_fwd[s:e] = self.footsteps[i, 0]
                self.p_ref_lat[s:e] = self.footsteps[i, 1]
        last = p.initial_delay + p.n_steps * p.step_time
        if last < REF_LEN:
            self.p_ref_fwd[last:] = self.footsteps[-1, 0]
            self.p_ref_lat[last:] = self.footsteps[-1, 1]
        self.p_ref_fwd[:p.initial_delay] = 0.0
        self.p_ref_lat[:p.initial_delay] = 0.0
        # after the last step: ZMP between the feet (double support)
        self.p_ref_fwd[self.stop_tick:] = self.footsteps[-1, 0]
        self.p_ref_lat[self.stop_tick:] = 0.0

    def _nominal_target(self, k: int):
        """Nominal (forward, lateral, is_A) of the foot that swings at tick k."""
        p = self.p
        t = min((k - p.initial_delay) // p.step_time + 1, p.n_steps - 1)
        return float(self.footsteps[t, 0]), float(self.footsteps[t, 1]), (t % 2 == 0)

    @property
    def plan_lat(self) -> float:
        return self.preview_lat.position

    @property
    def plan_fwd(self) -> float:
        return self.preview_fwd.position

    @property
    def planned_distance(self) -> float:
        return float(self.footsteps[-1, 0])

    # ---------------------------------------------------------------- terrain
    def _support(self, name, pos, fallback_g=0.0):
        """Sole plane (g, a, b) of a foot standing at pos = [lateral, forward]."""
        p = self.p
        if self.terrain is None or not p.use_map:
            return float(fallback_g), 0.0, 0.0
        dl, df = self.fp[name].samples()
        dX, dY = dl, -df                                    # model axes
        X0, Y0 = self.mid_x + pos[0], self.hip_y - pos[1]
        h = self.terrain.height(X0 + dX, Y0 + dY)
        a = b = 0.0
        if p.use_slope:
            A = np.column_stack([dX, dY, np.ones_like(dX)])
            sol, *_ = np.linalg.lstsq(A, h, rcond=None)
            lim = math.tan(p.max_tilt)
            a = float(np.clip(sol[0], -lim, lim))
            b = float(np.clip(sol[1], -lim, lim))
        g = float(np.max(h - a * dX - b * dY))              # rest on the highest points
        return g, a, b

    def _path_max(self, name, p0, p1):
        """Highest terrain point the footprint passes over on the way p0 -> p1."""
        if self.terrain is None or not self.p.use_map:
            return -np.inf
        dl, df = self.fp[name].samples(3, 3)
        s = np.linspace(0.0, 1.0, 7)[:, None]
        lat = p0[0] + (p1[0] - p0[0]) * s + dl[None, :]
        fwd = p0[1] + (p1[1] - p0[1]) * s + df[None, :]
        return float(np.max(self.terrain.height(self.mid_x + lat, self.hip_y - fwd)))

    # ---------------------------------------------------------------- 100 Hz
    def tick(self, com=None, vel=None, touch_a: float = 0.0, touch_b: float = 0.0,
             ank_dz: Optional[float] = None, torso_tilt=None):
        """One control step (dt_mpc).

        com, vel   : MEASURED CoM position / velocity (world, model axes) or None
        touch_a/b  : foot contact forces [N]
        ank_dz     : MEASURED (ankle A z) - (ankle B z) [m], from joint angles +
                     torso orientation; used to set the real ground height on
                     early touchdown
        torso_tilt : (pitch, roll, pitch_rate, roll_rate) of the torso [rad, rad/s]
                     for the optional ankle balance loop
        """
        p = self.p
        si = self.step_index
        idx = np.minimum(si + np.arange(p.n_preview), REF_LEN - 1)
        cur = min(si, REF_LEN - 1)
        self.preview_lat.update(self.p_ref_lat[cur], self.p_ref_lat[idx])
        self.preview_fwd.update(self.p_ref_fwd[cur], self.p_ref_fwd[idx])
        st_l, st_f = self.preview_lat.state, self.preview_fwd.state

        # ---------------- DCM error (always measured, applied only if k_dcm > 0)
        have_meas = com is not None and vel is not None
        if have_meas:
            w = self.omega
            xi_meas = np.array([(float(com[0]) - self.mid_x) + float(vel[0]) / w,
                                (self.hip_y - float(com[1])) - float(vel[1]) / w])
            if self.plan_com is not None and self.plan_vel is not None:
                pc, pv = self.plan_com, self.plan_vel
                xi_plan = np.array([(pc[0] - self.mid_x) + pv[0] / w,
                                    (self.hip_y - pc[1]) - pv[1] / w])
            else:
                xi_plan = np.array([st_l[0] + st_l[1] / w, st_f[0] + st_f[1] / w])
            err = xi_meas - xi_plan
            # constant offset while standing: calibrate during the first 0.1 s
            if 1 <= si < 11:
                self._bias_sum += err
                self._bias_n += 1
                self.bias = self._bias_sum / self._bias_n
            self.raw = err - self.bias
        else:
            self.raw = np.zeros(2)
        apply_dcm = have_meas and p.k_dcm > 0.0

        # ---------------- optional ankle balance (CoP-style)
        self.ankle_corr = np.zeros(2)
        if p.ankle_balance and torso_tilt is not None and si >= 11:
            pitch, roll, dpitch, droll = torso_tilt
            c = p.ankle_clamp
            # DCM error ahead of the plan (raw > 0) also asks for the CoP to move
            # that way: same sign as a forward / +X torso lean.
            self.ankle_corr = np.array([
                float(np.clip(p.ankle_kp * pitch + p.ankle_kd * dpitch
                              + p.ankle_k_dcm * self.raw[1], -c, c)),
                float(np.clip(p.ankle_kp * roll + p.ankle_kd * droll
                              + p.ankle_k_dcm * self.raw[0], -c, c))])

        A, B = self.feet["A"], self.feet["B"]
        if si >= self.stop_tick:
            A.lift = B.lift = 0.0
            self.swing = None
        elif si >= p.initial_delay:
            self._swing_step(si, apply_dcm, touch_a, touch_b, ank_dz)

        self.step_index += 1

        # ---------------- CoM of the commanded posture (compared with the next measurement)
        if self.com_model is not None:
            q_a, q_b, torso_w = self.joint_targets(p.z_hip)
            c = np.asarray(self.com_model(torso_w, q_a, q_b), dtype=float)
            if self.plan_com is not None:
                self.plan_vel = (c - self.plan_com) / p.dt_mpc
            self.plan_com = c

    def _swing_step(self, si, apply_dcm, touch_a, touch_b, ank_dz):
        p = self.p
        k = si - p.initial_delay
        ph_i = k % p.step_time
        tx, ty, is_a = self._nominal_target(si)
        sw = "A" if is_a else "B"
        st = "B" if is_a else "A"

        if ph_i == 0:
            self.swing = sw
            self.td_lock = False
            f = self.feet[sw]
            self.start = Foot(f.pos.copy(), f.g, f.tilt.copy(), 0.0)
            if p.dcm_hold:
                self.corr = np.zeros(2)

        u = (ph_i + 1) / float(p.step_time)             # (0, 1]; exactly 1 on the last tick

        # ---- DCM foot-placement correction
        if apply_dcm:
            if u < p.dcm_start:
                pass                                    # not reliable yet (servo lag at transition)
            elif u < p.dcm_freeze:
                e = np.sign(self.raw) * np.maximum(0.0, np.abs(self.raw) - p.dcm_deadband)
                want = np.array([
                    np.clip(p.k_dcm * e[0], -p.dcm_lim_lat, p.dcm_lim_lat),
                    np.clip(p.k_dcm * e[1], -p.dcm_lim_fwd, p.dcm_lim_fwd)])
                if p.dcm_hold:
                    self.corr = self.corr + 0.3 * (want - self.corr)    # light low-pass
                else:
                    self.corr = want
            elif not p.dcm_hold:                        # legacy: fade to zero before landing
                self.corr = self.corr * max(0.0, (1.0 - u) / (1.0 - p.dcm_freeze)) \
                    if u < 1.0 else np.zeros(2)
        else:
            self.corr = np.zeros(2)
        tgt = np.array([ty + self.corr[0], tx + self.corr[1]])

        # ---- keep the feet apart: the swing foot may not cross towards the other one
        min_gap = 0.5 * self.fp["A"].width + 0.008
        if sw == "A":
            tgt[0] = min(tgt[0], self.feet["B"].pos[0] - 2 * min_gap)
        else:
            tgt[0] = max(tgt[0], self.feet["A"].pos[0] + 2 * min_gap)

        # ---- the ZMP reference follows the foot
        if p.zmp_follow and u <= p.dcm_freeze + 1e-9:
            a0 = p.initial_delay + (k // p.step_time + 1) * p.step_time
            a1 = min(a0 + p.step_time, self.stop_tick)
            if a0 < a1:
                self.p_ref_lat[a0:a1] = tgt[0]
                self.p_ref_fwd[a0:a1] = tgt[1]
            if a0 + p.step_time >= self.stop_tick:          # last step: between the feet
                self.p_ref_lat[self.stop_tick:] = 0.5 * (tgt[0] + self.feet[st].pos[0])
                self.p_ref_fwd[self.stop_tick:] = 0.5 * (tgt[1] + self.feet[st].pos[1])

        # ---- step finished: re-anchor the rest of the plan to where the foot landed
        if p.zmp_follow and ph_i == p.step_time - 1:
            shift = tgt - np.array([ty, tx])                # [lateral, forward]
            t_next = k // p.step_time + 1
            if np.any(shift != 0.0) and t_next + 1 < p.n_steps:
                self.footsteps[t_next + 1:, 0] += shift[1]
                self.footsteps[t_next + 1:, 1] += shift[0]
                a1_ = p.initial_delay + (t_next + 1) * p.step_time
                self.p_ref_fwd[a1_:] += shift[1]
                self.p_ref_lat[a1_:] += shift[0]

        f = self.feet[sw]
        touch = touch_a if sw == "A" else touch_b
        g_exp = self._support(sw, tgt, fallback_g=self.feet[st].g)[0]
        window = p.td_window if (self.terrain is not None and p.use_map) else np.inf

        # ---- early touchdown: freeze the descent.
        # (Window: a contact while the sole is far ABOVE the expected ground is
        #  not terrain but the robot tipping onto that foot -- ignore it.)
        if (p.use_touchdown and not self.td_lock and u > p.touch_frac
                and u < 1.0 and touch > self.touch_force
                and (f.g + f.lift - g_exp) <= window):
            self.td_lock = True
            self.early_td_count += 1
            # The COMMANDED height cannot be used: the real foot lags the
            # command by 6-10 mm (measured). Use the stance foot's ground plus
            # the measured ankle height difference instead.
            if ank_dz is not None:
                dz = ank_dz if sw == "A" else -ank_dz
                f.g = min(f.g + f.lift, self.feet[st].g + dz)
            else:
                f.g = f.g + f.lift
            f.lift = 0.0

        if not self.td_lock:
            # horizontal: lift first, then move, then descend vertically
            uu = min(1.0, max(0.0, (u - 0.12) / 0.76))
            s = 0.5 * (1.0 - math.cos(math.pi * uu))
            g_t, a_t, b_t = self._support(sw, tgt, fallback_g=self.feet[st].g)
            s0 = self.start
            f.pos = s0.pos + (tgt - s0.pos) * s
            f.g = s0.g + (g_t - s0.g) * s
            f.tilt = s0.tilt + (np.array([a_t, b_t]) - s0.tilt) * s
            # clearance: pass over the highest point on the path
            extra = max(0.0, self._path_max(sw, s0.pos, tgt) - max(s0.g, g_t))
            f.lift = (p.step_height + extra) * math.sin(math.pi * u)
            if u >= 1.0:
                f.lift = 0.0
            g_swing_ref = g_t
        else:
            g_swing_ref = f.g
        self.feet[st].lift = 0.0

        # smoothed ground reference for the body height (first-order filter)
        g_target = 0.5 * (self.feet[st].g + g_swing_ref)
        self.g_ref += (p.dt_mpc / p.ground_tau) * (g_target - self.g_ref)

    # ---------------------------------------------------------------- every sim step
    def ankle_world(self, name):
        """World position of the ankle center and the sole rotation (R_tilt)."""
        f = self.feet[name]
        R_t, n = tilt_rotation(f.tilt[0], f.tilt[1])
        S = self.to_model(f.pos[0], f.pos[1], f.g + f.lift)
        return S + n * self.ankle_h, R_t

    def joint_targets(self, z_now: float):
        """z_now: hip-center height ABOVE the ground reference.
        Returns (q_a, q_b, torso_w): 6 joint angles per leg and the torso-origin target."""
        p, r = self.p, self.rcfg
        hip_ref_w = self.to_model(self.plan_lat, self.plan_fwd, z_now + self.g_ref)
        torso_w = hip_ref_w - r.body_ref_offset
        pA, RA = self.ankle_world("A")
        pB, RB = self.ankle_world("B")

        # reach guard: lower the hip until both legs are <= reach_frac * max reach
        l_max = p.reach_frac * r.max_reach
        drop = 0.0
        for pa, hc in ((pA, r.hip_center_left), (pB, r.hip_center_right)):
            d = (torso_w + hc) - pa
            dh2 = d[0] ** 2 + d[1] ** 2
            if dh2 + d[2] ** 2 > l_max ** 2:
                v_new = math.sqrt(max(l_max ** 2 - dh2, 1e-6))
                drop = max(drop, d[2] - v_new)
        self.reach_drop = drop
        torso_w = torso_w - np.array([0.0, 0.0, drop])

        q_a = self.ik.solve(pA - torso_w, left=True, foot_R=RA @ r.r_foot_level,
                            refine=self.refine_ik)
        q_b = self.ik.solve(pB - torso_w, left=False, foot_R=RB @ r.r_foot_level,
                            refine=self.refine_ik)
        return q_a, q_b, torso_w

    def apply_ankle_correction(self, q_a, q_b):
        """Add the ankle balance correction to the stance foot (both feet in
        double support). Returns new arrays; joint 4 = ankle pitch, 5 = ankle roll.

        Signs (checked with forward kinematics, tests/test_kinematics.py):
        +ankle_pitch lifts the toe relative to the shank, +ankle_roll lowers
        the model +X edge of the sole. A torso leaning forward (pitch > 0) is
        pushed back by pressing the toes down (CoP forward): ankle_pitch -= .
        A torso leaning towards +X (roll > 0) is pushed back by pressing the
        +X edge down (CoP towards +X): ankle_roll += ."""
        if not np.any(self.ankle_corr):
            return q_a, q_b
        q_a, q_b = q_a.copy(), q_b.copy()
        dp, dr = self.ankle_corr
        for name, q in (("A", q_a), ("B", q_b)):
            if self.swing == name and self.feet[name].lift > 0.0:
                continue
            q[4] -= dp
            q[5] += dr
        return q_a, q_b


# ===========================================================================
#  KINEMATIC DRY RUN  (no physics: logic + IK + terrain consistency)
# ===========================================================================
def dry_run(rough: float, seed: int = 3, gait: Optional[TerrainGaitConfig] = None,
            ankle_h: float = 0.0567):
    """Walk the whole plan kinematically and report foot/terrain consistency.

    The CoM "measurement" equals the plan (perfect tracking), so the DCM
    correction must stay zero. Returns a dict of worst-case numbers."""
    from sim.terrain import Terrain
    rcfg = RobotConfig()
    gait = gait or TerrainGaitConfig()
    terr = Terrain(rough=rough, seed=seed, start_y=rcfg.hip_y)
    pl = TerrainStepPlanner(gait, rcfg, terr, ankle_h)
    dl, df = Footprint().samples(6, 5)
    out = dict(ik_err=0.0, penetration=0.0, joint_jump=0.0, reach_drop=0.0,
               at_limit=0, knee_min=9.0, knee_max=0.0)
    q_prev = None
    for _ in range(pl.stop_tick + 100):
        com = pl.to_model(pl.plan_lat, pl.plan_fwd, gait.z_hip)
        vel = np.array([pl.preview_lat.velocity, -pl.preview_fwd.velocity, 0.0])
        pl.tick(com, vel, 0.0, 0.0)
        q_a, q_b, torso_w = pl.joint_targets(gait.z_hip)
        out["reach_drop"] = max(out["reach_drop"], pl.reach_drop)
        for name, q, left in (("A", q_a, True), ("B", q_b, False)):
            if np.any(q <= rcfg.q_min + 1e-9) or np.any(q >= rcfg.q_max - 1e-9):
                out["at_limit"] += 1
            out["knee_min"] = min(out["knee_min"], q[3])
            out["knee_max"] = max(out["knee_max"], q[3])
            pf, R, _, _ = pl.ik.forward(q, left)
            ank = torso_w + pf + R @ rcfg.ankle_center_in_foot
            want, _ = pl.ankle_world(name)
            out["ik_err"] = max(out["ik_err"], float(np.linalg.norm(ank - want)))
            pts = np.array([ank + R @ (rcfg.r_foot_level.T @ np.array([x, -f, -ankle_h]))
                            for x, f in zip(dl, df)])
            gap = pts[:, 2] - terr.height(pts[:, 0], pts[:, 1])
            out["penetration"] = max(out["penetration"], float(-gap.min()))
        q = np.concatenate([q_a, q_b])
        if q_prev is not None:
            out["joint_jump"] = max(out["joint_jump"], float(np.max(np.abs(q - q_prev))))
        q_prev = q
    out["dcm_corr"] = float(np.abs(pl.corr).max())
    out["distance"] = float(pl.feet["A"].pos[1])
    return out


if __name__ == "__main__":
    for rough in (0.0, 0.6, 1.0):
        r = dry_run(rough)
        print(f"rough={rough:3.1f}: IK error max {r['ik_err'] * 1000:.2f} mm | "
              f"sole penetration max {r['penetration'] * 1000:.2f} mm | "
              f"joint jump max {math.degrees(r['joint_jump']):.2f} deg/tick | "
              f"knee {math.degrees(r['knee_min']):.0f}..{math.degrees(r['knee_max']):.0f} deg | "
              f"reach guard max {r['reach_drop'] * 1000:.1f} mm | at limit {r['at_limit']} | "
              f"DCM corr {r['dcm_corr'] * 1000:.2f} mm")
