"""ZMP preview control: the ZMP must track a step reference."""
import numpy as np

from controllers.preview_controller import ZMPPreviewController


def run_step(ref_value=0.05, n=400, preview=160):
    c = ZMPPreviewController(0.01, 0.185, preview)
    ref = np.zeros(n + preview)
    ref[150:] = ref_value
    zmp = []
    for k in range(n):
        c.update(ref[k], ref[k:k + preview])
        zmp.append(c.zmp)
    return np.array(zmp), c


def test_zmp_tracks_a_step():
    zmp, c = run_step()
    assert abs(zmp[-1] - 0.05) < 1e-3
    assert abs(c.position - 0.05) < 1e-3          # CoM settles above the new ZMP


def test_preview_starts_moving_before_the_step():
    zmp, c = run_step()
    # with preview, the CoM starts to move before the reference changes (k = 150)
    c2 = ZMPPreviewController(0.01, 0.185, 160)
    ref = np.zeros(560)
    ref[150:] = 0.05
    pos = []
    for k in range(150):
        c2.update(ref[k], ref[k:k + 160])
        pos.append(c2.position)
    assert pos[-1] > 1e-3


def test_omega():
    c = ZMPPreviewController(0.01, 0.185, 10)
    assert abs(c.omega - 7.282) < 1e-3
