import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from config.parameters import GaitConfig

REF_LEN = 20000


@dataclass
class FootTargets:
    a_lat: float
    a_fwd: float
    a_lift: float
    b_lat: float
    b_fwd: float
    b_lift: float
    swing: Optional[str] = None
    phase: float = 0.0
    step_index: int = 0


class StepPlanner:

    def __init__(self, gait: GaitConfig):
        self.gait = gait
        self.footsteps = np.array([
            [i * gait.step_len, gait.lat_a if i % 2 == 0 else gait.lat_b]
            for i in range(gait.n_steps)
        ])
        self._build_zmp_reference()

        self.foot_a = np.array([gait.lat_a, 0.0])
        self.foot_b = np.array([gait.lat_b, 0.0])
        self.lift_a = 0.0
        self.lift_b = 0.0
        self._start_a = self.foot_a.copy()
        self._start_b = self.foot_b.copy()
        self._swing = None

    def _build_zmp_reference(self):
        g = self.gait
        self.p_ref_fwd = np.zeros(REF_LEN)
        self.p_ref_lat = np.zeros(REF_LEN)

        for i in range(g.n_steps):
            s = g.initial_delay + i * g.step_time
            e = min(s + g.step_time, REF_LEN)
            if s < REF_LEN:
                self.p_ref_fwd[s:e] = self.footsteps[i, 0]
                self.p_ref_lat[s:e] = self.footsteps[i, 1]

        self.p_ref_fwd[:g.initial_delay] = 0.0
        self.p_ref_lat[:g.initial_delay] = 0.0

        stop = g.stop_tick
        if stop < REF_LEN:
            self.p_ref_fwd[stop:] = self.footsteps[-1, 0]
            self.p_ref_lat[stop:] = 0.0

    def zmp_ref(self, k: int) -> Tuple[float, float]:
        i = min(k, REF_LEN - 1)
        return float(self.p_ref_lat[i]), float(self.p_ref_fwd[i])

    def preview_window(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.minimum(k + np.arange(self.gait.n_preview), REF_LEN - 1)
        return self.p_ref_lat[idx], self.p_ref_fwd[idx]

    def nominal_swing_target(self, k: int) -> Tuple[float, float, bool]:
        g = self.gait
        if k < g.initial_delay:
            return 0.0, 0.0, False
        step = (k - g.initial_delay) // g.step_time + 1
        t = min(step, g.n_steps - 1)
        return float(self.footsteps[t, 0]), float(self.footsteps[t, 1]), (t % 2 == 0)

    def swing_phase(self, k: int) -> float:
        g = self.gait
        if k < g.initial_delay or k >= g.stop_tick:
            return 0.0
        return ((k - g.initial_delay) % g.step_time) / float(g.step_time)

    def update(self, k: int, correction: Tuple[float, float] = (0.0, 0.0)) -> FootTargets:
        g = self.gait

        if k >= g.stop_tick:
            self.lift_a = self.lift_b = 0.0
            self._swing = None
            return self._targets(k, 0.0)

        if k < g.initial_delay:
            return self._targets(k, 0.0)

        phase_i = (k - g.initial_delay) % g.step_time
        tgt_fwd, tgt_lat, is_a = self.nominal_swing_target(k)
        tgt_lat += correction[0]
        tgt_fwd += correction[1]

        if phase_i == 0:
            self._start_a = self.foot_a.copy()
            self._start_b = self.foot_b.copy()
            self._swing = "A" if is_a else "B"

        phi = (phase_i / float(g.step_time)) * math.pi
        lerp = (1.0 - math.cos(phi)) / 2.0
        target = np.array([tgt_lat, tgt_fwd])

        if self._swing == "A":
            self.foot_a = self._start_a + (target - self._start_a) * lerp
            self.lift_a = g.step_height * math.sin(phi)
            self.lift_b = 0.0
        elif self._swing == "B":
            self.foot_b = self._start_b + (target - self._start_b) * lerp
            self.lift_b = g.step_height * math.sin(phi)
            self.lift_a = 0.0

        return self._targets(k, phase_i / float(g.step_time))

    def _targets(self, k: int, phase: float) -> FootTargets:
        return FootTargets(
            a_lat=float(self.foot_a[0]), a_fwd=float(self.foot_a[1]), a_lift=self.lift_a,
            b_lat=float(self.foot_b[0]), b_fwd=float(self.foot_b[1]), b_lift=self.lift_b,
            swing=self._swing, phase=phase, step_index=k,
        )

    def current_targets(self, k: int) -> FootTargets:
        return self._targets(k, self.swing_phase(k))
