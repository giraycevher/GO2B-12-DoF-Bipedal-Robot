"""Telemetri toplama, konsol raporu ve grafik."""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class TelemetryLogger:
    report_period: float = 1.0
    data: Dict[str, List[float]] = field(default_factory=dict)
    _next_report: float = 0.0

    def record(self, **fields):
        for key, value in fields.items():
            self.data.setdefault(key, []).append(float(value))

    def get(self, key: str) -> np.ndarray:
        return np.asarray(self.data.get(key, []), dtype=float)

    @property
    def empty(self) -> bool:
        return not self.data.get("t")

    def due(self, t: float) -> bool:
        if t < self._next_report:
            return False
        self._next_report += self.report_period
        return True

    def status_line(self, **fields) -> str:
        parts = []
        for k, v in fields.items():
            parts.append(f"{k}={v}" if isinstance(v, str) else f"{k}={v}")
        return "  " + "  ".join(parts)

    # ------------------------------------------------------------ ozet
    def summary(self, exit_reason: str, first_step_time: float,
                ik_failures: int, sim_limit: float) -> str:
        if self.empty:
            return f"\n--- OZET ---\ncikis nedeni : {exit_reason}\nveri yok"

        t = self.get("t")
        cf = self.get("com_fwd")
        cl = self.get("com_lat")
        bz = self.get("base_z")

        ok = bz.min() > 0.12 and t[-1] > sim_limit * 0.95
        lines = [
            "\n--- OZET ---",
            f"cikis nedeni : {exit_reason}",
            f"sim suresi   : {t[-1]:.2f} s",
            f"ilk adim     : t={first_step_time:.2f}s",
            f"IK hatasi    : {ik_failures}",
            f"kat edilen   : {cf[-1] - cf[0]:+.3f} m (ileri)",
            f"yanal kayma  : {cl[-1] - cl[0]:+.3f} m",
            f"govde z      : bas {bz[0]:.3f}  son {bz[-1]:.3f}  min {bz.min():.3f}",
            f"SONUC        : {'AYAKTA KALDI / YURUDU' if ok else 'DUSTU veya ERKEN BITTI'}",
        ]
        return "\n".join(lines)

    # ---------------------------------------------------------- grafik
    def plot(self, path: str, push_time: float = None, push_force: float = 0.0) -> bool:
        if self.empty:
            return False
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return False

        t = self.get("t")
        fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

        ax[0].plot(t, self.get("plan_fwd"), "--", label="planlanan CoM")
        ax[0].plot(t, self.get("com_fwd"), lw=2, label="olculen CoM")
        ax[0].set_title("Ileri")
        ax[0].set_ylabel("m")

        ax[1].plot(t, self.get("plan_lat"), "--", label="planlanan CoM")
        ax[1].plot(t, self.get("com_lat"), lw=2, label="olculen CoM")
        ax[1].set_title("Yanal")
        ax[1].set_ylabel("m")

        ax[2].plot(t, self.get("base_z"), "k", lw=2, label="govde z")
        ax[2].set_title("Govde yuksekligi")
        ax[2].set_ylabel("m")

        ax[3].plot(t, self.get("dcm_lat") * 1000, label="yanal duzeltme")
        ax[3].plot(t, self.get("dcm_fwd") * 1000, label="ileri duzeltme")
        ax[3].set_title("DCM ayak yerlesim duzeltmesi")
        ax[3].set_ylabel("mm")
        ax[3].set_xlabel("t [s]")

        for a in ax:
            a.grid(True)
            a.legend(loc="upper left", fontsize=9)
            if push_time is not None and push_force:
                a.axvline(push_time, color="r", ls=":", lw=1.5)
        if push_time is not None and push_force:
            ax[0].text(push_time, ax[0].get_ylim()[1], f" itme {push_force:g}N",
                       color="r", va="top", fontsize=9)

        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close(fig)
        return True
