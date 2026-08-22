#!/usr/bin/env python3
"""12-DOF bipedal yuruyus -- orkestrasyon.

    python main.py
    python main.py --headless --push 15
    python main.py --record yuruyus.mp4 --headless
"""

import argparse
import sys
import time

try:
    import ctypes
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

import numpy as np

from config.parameters import SimConfig
from controllers.preview_controller import ZMPPreviewController
from controllers.push_recovery import AnkleStabilizer, DCMRecovery
from kinematics.leg_ik import LegKinematics
from planners.step_planner import StepPlanner
from sim.mujoco_interface import MujocoRobot, VideoRecorder, draw_push
from utils.data_logger import TelemetryLogger


def parse_args(cfg: SimConfig) -> SimConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=cfg.robot.xml_path)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--time", type=float, default=cfg.max_time)
    ap.add_argument("--speed", type=float, default=cfg.speed)
    ap.add_argument("--zhip", type=float, default=cfg.gait.z_hip)
    ap.add_argument("--step-len", type=float, default=cfg.gait.step_len)
    ap.add_argument("--push", type=float, default=cfg.push.force)
    ap.add_argument("--push-axis", default=cfg.push.axis, choices=["lat", "fwd"])
    ap.add_argument("--push-viz", default=cfg.push.viz, choices=["arrow", "ball", "none"])
    ap.add_argument("--no-dcm", action="store_true")
    ap.add_argument("--k-dcm", type=float, default=cfg.dcm.gain)
    ap.add_argument("--ankle-fb", action="store_true")
    ap.add_argument("--ankle-fb-sign", type=float, default=cfg.ankle.sign)
    ap.add_argument("--refine-ik", action="store_true")
    ap.add_argument("--record", default=cfg.record)
    ap.add_argument("--fps", type=int, default=cfg.fps)
    ap.add_argument("--width", type=int, default=cfg.width)
    ap.add_argument("--height", type=int, default=cfg.height)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    cfg.robot.xml_path = a.xml
    cfg.headless, cfg.max_time, cfg.speed = a.headless, a.time, a.speed
    cfg.record, cfg.fps, cfg.width, cfg.height = a.record, a.fps, a.width, a.height
    cfg.gait.z_hip, cfg.gait.step_len = a.zhip, a.step_len
    cfg.push.force, cfg.push.axis, cfg.push.viz = a.push, a.push_axis, a.push_viz
    cfg.dcm.enabled = not a.no_dcm
    cfg.dcm.gain = a.k_dcm
    cfg.ankle.enabled = a.ankle_fb
    cfg.ankle.sign = a.ankle_fb_sign
    cfg._refine_ik = a.refine_ik
    cfg._no_plot = a.no_plot
    return cfg


def main():
    cfg = parse_args(SimConfig())
    g, r, p = cfg.gait, cfg.robot, cfg.push

    robot = MujocoRobot(r)
    ik = LegKinematics(r)
    planner = StepPlanner(g)
    prev_lat = ZMPPreviewController(g.dt_mpc, g.z_hip, g.n_preview)
    prev_fwd = ZMPPreviewController(g.dt_mpc, g.z_hip, g.n_preview)
    dcm = DCMRecovery(cfg.dcm, r)
    ankle = AnkleStabilizer(cfg.ankle)
    log = TelemetryLogger(cfg.report_period)

    dt = robot.timestep
    mpc_every = max(1, int(round(g.dt_mpc / dt)))

    print(f"model        : {r.xml_path}")
    print(f"nq={robot.model.nq} nv={robot.model.nv} nu={robot.model.nu} timestep={dt}")
    print(f"robot kutlesi: {robot.robot_mass:.4f} kg")

    ankle_h = robot.measure_ankle_height()
    drop = g.z_hip - ankle_h
    print(f"ayak bilegi merkezi yerden (olculdu): {ankle_h * 1000:.1f} mm")
    print(f"bacak dususu : {drop:.4f} m = erisimin %{drop / r.max_reach * 100:.0f}'i "
          f"(max {r.max_reach:.4f} m)")
    if drop > r.max_reach * 0.95:
        sys.exit(f"HATA: --zhip {g.z_hip} fazla yuksek. "
                 f"En fazla {ankle_h + r.max_reach * 0.92:.3f} kullan.")

    def to_model(lat, fwd, up):
        return np.array([r.mid_x + lat, r.hip_y - fwd, up])

    def solve_legs(z_hip_now, targets, lift_a=None, lift_b=None):
        hip_ref = to_model(0.0, 0.0, z_hip_now)
        torso_w = hip_ref - r.body_ref_offset
        fa = to_model(targets[0], targets[1],
                      ankle_h + (targets[2] if lift_a is None else lift_a)) - torso_w
        fb = to_model(targets[3], targets[4],
                      ankle_h + (targets[5] if lift_b is None else lift_b)) - torso_w
        return (ik.solve(fa, left=True, refine=cfg._refine_ik),
                ik.solve(fb, left=False, refine=cfg._refine_ik),
                torso_w)

    q_a, q_b, torso_w = solve_legs(g.squat_start_z,
                                   (g.lat_a, 0.0, 0.0, g.lat_b, 0.0, 0.0))
    robot.set_pose(torso_w, q_a, q_b)
    robot.settle_on_ground()
    robot.send(q_a, q_b)
    print(f"baslangic govde yuksekligi: {robot.base_height:.4f} m")

    viewer = None
    if not cfg.headless:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(robot.model, robot.data)
        viewer.__enter__()

    video = None
    if cfg.record:
        video = VideoRecorder(robot.model, cfg.record, cfg.fps, cfg.width, cfg.height)
        print(f"kayit        : {cfg.record}  {cfg.width}x{cfg.height} @ {cfg.fps} fps")

    k = 0
    walking = False
    ball_fired = False
    ik_failures = 0
    recovery = dcm.compute(np.zeros(3), np.zeros(3), (0, 0), (0, 0))
    exit_reason = "sure doldu"
    next_sync = next_frame = 0.0
    frame_dt = 1.0 / max(1, cfg.fps)
    t_wall = time.perf_counter()

    try:
        while robot.time < cfg.max_time:
            if viewer is not None and not viewer.is_running():
                exit_reason = "goruntuleyici penceresi kapatildi"
                break
            t = robot.time

            # ---------------------------------------------------------- itme
            robot.clear_push()
            pushing = bool(p.force) and p.time < t < p.time + p.duration
            if p.force and p.viz == "ball":
                if not ball_fired and t >= p.time:
                    v = robot.launch_projectile(p.axis, p.force, p.duration)
                    ball_fired = True
                    if v is not None:
                        print(f"t={t:.2f}s  TOP FIRLATILDI: "
                              f"m={robot.ball_mass * 1000:.0f} g, v={v:.2f} m/s")
            elif pushing:
                robot.apply_push(p.force, p.axis)

            # -------------------------------------------------------- cokme
            if t < g.settle_time:
                z_now = g.squat_start_z
            elif t < g.settle_time + g.squat_time:
                u = (t - g.settle_time) / g.squat_time
                z_now = g.squat_start_z - (g.squat_start_z - g.z_hip) * u
            else:
                z_now = g.z_hip
                if not walking:
                    walking = True
                    print(f"t={t:.2f}s  cokme bitti, ZMP planlayici basladi "
                          f"(ilk adim {g.initial_delay * g.dt_mpc:.1f}s sonra)")

            # ------------------------------------------- MPC + planlama
            if walking and (round(t / dt) % mpc_every == 0):
                ref_lat, ref_fwd = planner.zmp_ref(k)
                win_lat, win_fwd = planner.preview_window(k)
                prev_lat.update(ref_lat, win_lat)
                prev_fwd.update(ref_fwd, win_fwd)

                com_p, com_v = robot.com_state()
                recovery = dcm.compute(
                    com_p, com_v,
                    (prev_lat.position, prev_lat.velocity),
                    (prev_fwd.position, prev_fwd.velocity),
                    planner.swing_phase(k))

                targets = planner.update(k, recovery.as_tuple)
                k += 1
            else:
                targets = planner.current_targets(k)

            # ------------------------------------------------------------ IK
            hip_ref = to_model(prev_lat.position, prev_fwd.position, z_now)
            torso_w = hip_ref - r.body_ref_offset
            fa = to_model(targets.a_lat, targets.a_fwd, ankle_h + targets.a_lift) - torso_w
            fb = to_model(targets.b_lat, targets.b_fwd, ankle_h + targets.b_lift) - torso_w
            try:
                q_a = ik.solve(fa, left=True, refine=cfg._refine_ik)
                q_b = ik.solve(fb, left=False, refine=cfg._refine_ik)
                if not (np.all(np.isfinite(q_a)) and np.all(np.isfinite(q_b))):
                    raise FloatingPointError("IK NaN")
            except Exception as e:
                ik_failures += 1
                print(f"t={t:.3f}s IK HATASI ({ik_failures}): {e}")
                if ik_failures >= cfg.max_ik_failures:
                    exit_reason = "ust uste IK hatasi"
                    break

            cmd_a, cmd_b = q_a.copy(), q_b.copy()
            if walking:
                d_pitch, d_roll = ankle.compute(robot.base_rotation,
                                                robot.base_angular_velocity)
                cmd_a[4] += d_pitch
                cmd_b[4] += d_pitch
                cmd_a[5] -= d_roll
                cmd_b[5] -= d_roll
            robot.send(np.clip(cmd_a, r.q_min, r.q_max),
                       np.clip(cmd_b, r.q_min, r.q_max))

            robot.step()

            if not robot.is_finite():
                exit_reason = "NaN (simulasyon patladi)"
                print(f"t={t:.3f}s SIMULASYON PATLADI (NaN).")
                break

            # ------------------------------------------------------ telemetri
            com_p, _ = robot.com_state()
            log.record(t=t,
                       com_fwd=r.hip_y - com_p[1], com_lat=com_p[0] - r.mid_x,
                       plan_fwd=prev_fwd.position, plan_lat=prev_lat.position,
                       base_z=robot.base_height,
                       dcm_lat=recovery.d_lat, dcm_fwd=recovery.d_fwd)

            if log.due(t):
                cf = log.get("com_fwd")
                wall = time.perf_counter() - t_wall
                print(f"  t={t:4.1f}s  ileri={cf[-1] - cf[0]:+.3f}m"
                      f"  kalca z={robot.base_height + r.body_ref_offset[2]:.4f}"
                      f" (sarkma {(z_now - robot.base_height - r.body_ref_offset[2]) * 1000:+5.1f}mm)"
                      f"  salinim={targets.swing or '-'}"
                      f"  adim={max(0, k - g.initial_delay) // g.step_time}"
                      f"  temas={robot.n_contacts}"
                      f"  hiz={t / wall if wall > 1e-6 else 0:.2f}x"
                      f"  DCM=({recovery.err_lat * 1000:+5.1f},{recovery.err_fwd * 1000:+5.1f})mm")

            if robot.base_height < cfg.fall_height:
                exit_reason = "robot dustu"
                print(f"t={t:.2f}s  govde z={robot.base_height:.3f} -- robot dustu.")
                break

            # ------------------------------------------------------ gorsel
            alpha = 0.0
            if p.force and p.viz == "arrow":
                if pushing:
                    alpha = 1.0
                elif p.time <= t < p.time + p.duration + p.arrow_hold:
                    alpha = 1.0 - (t - p.time - p.duration) / p.arrow_hold

            if video is not None and robot.time >= next_frame:
                next_frame += frame_dt
                video.capture(robot.data)
                if p.viz == "arrow" and alpha > 0:
                    draw_push(video.scene, robot, p.axis, p.force, alpha, reset=False)
                video.write()

            if viewer is not None and robot.time >= next_sync:
                next_sync += cfg.view_dt
                if p.viz == "arrow":
                    draw_push(viewer.user_scn, robot, p.axis, p.force, alpha, reset=True)
                viewer.sync()
                if cfg.speed > 0 and video is None:
                    target_wall = t_wall + robot.time / cfg.speed
                    now = time.perf_counter()
                    if target_wall > now:
                        time.sleep(target_wall - now)
    finally:
        if viewer is not None:
            viewer.__exit__(None, None, None)
        if video is not None:
            video.close()
            print(f"video yazildi: {cfg.record}")

    print(log.summary(exit_reason, g.first_step_time, ik_failures, cfg.max_time))
    if not cfg._no_plot and log.plot(cfg.plot_path, p.time, p.force):
        print(f"grafik: {cfg.plot_path}")


if __name__ == "__main__":
    main()
