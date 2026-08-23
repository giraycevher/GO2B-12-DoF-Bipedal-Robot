#!/usr/bin/env python3
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
    ap = argparse.ArgumentParser(description="12-DOF biped ZMP preview walking")
    ap.add_argument("--xml", default=cfg.robot.xml_path)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--time", type=float, default=cfg.max_time)
    ap.add_argument("--speed", type=float, default=cfg.speed,
                    help="1.0 = real time, 0 = as fast as possible")
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
    cfg.refine_ik, cfg.plot_enabled = a.refine_ik, not a.no_plot
    cfg.gait.z_hip, cfg.gait.step_len = a.zhip, a.step_len
    cfg.push.force, cfg.push.axis, cfg.push.viz = a.push, a.push_axis, a.push_viz
    cfg.dcm.enabled, cfg.dcm.gain = not a.no_dcm, a.k_dcm
    cfg.ankle.enabled, cfg.ankle.sign = a.ankle_fb, a.ankle_fb_sign
    return cfg


def main():
    cfg = parse_args(SimConfig())
    g, r, p = cfg.gait, cfg.robot, cfg.push

    robot = MujocoRobot(r)
    ik = LegKinematics(r)
    planner = StepPlanner(g)
    preview_lat = ZMPPreviewController(g.dt_mpc, g.z_hip, g.n_preview)
    preview_fwd = ZMPPreviewController(g.dt_mpc, g.z_hip, g.n_preview)
    dcm = DCMRecovery(cfg.dcm, r)
    ankle = AnkleStabilizer(cfg.ankle)
    log = TelemetryLogger(cfg.report_period)

    dt = robot.timestep
    mpc_every = max(1, int(round(g.dt_mpc / dt)))

    print(f"model        : {r.xml_path}")
    print(f"nq={robot.model.nq} nv={robot.model.nv} nu={robot.model.nu} timestep={dt}")
    print(f"robot mass   : {robot.robot_mass:.4f} kg")

    ankle_h = robot.measure_ankle_height()
    drop = g.z_hip - ankle_h
    print(f"ankle height : {ankle_h * 1000:.1f} mm (measured)")
    print(f"leg drop     : {drop:.4f} m = {drop / r.max_reach * 100:.0f}% of reach "
          f"(max {r.max_reach:.4f} m)")
    if drop > r.max_reach * 0.95:
        sys.exit(f"ERROR: --zhip {g.z_hip} too high. "
                 f"Use at most {ankle_h + r.max_reach * 0.92:.3f}.")

    def to_model(lat, fwd, up):
        return np.array([r.mid_x + lat, r.hip_y - fwd, up])

    def torso_origin(z_hip_now, lat=0.0, fwd=0.0):
        return to_model(lat, fwd, z_hip_now) - r.body_ref_offset

    origin = torso_origin(g.squat_start_z)
    q_a = ik.solve(to_model(g.lat_a, 0.0, ankle_h) - origin, True, refine=cfg.refine_ik)
    q_b = ik.solve(to_model(g.lat_b, 0.0, ankle_h) - origin, False, refine=cfg.refine_ik)
    robot.set_pose(origin, q_a, q_b)
    robot.settle_on_ground()
    robot.send(q_a, q_b)
    print(f"initial base z: {robot.base_height:.4f} m")

    viewer = None
    if not cfg.headless:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(robot.model, robot.data)
        viewer.__enter__()

    video = None
    if cfg.record:
        video = VideoRecorder(robot.model, cfg.record, cfg.fps, cfg.width, cfg.height)
        print(f"recording    : {cfg.record}  {cfg.width}x{cfg.height} @ {cfg.fps} fps")

    k = 0
    walking = False
    ball_fired = False
    ik_failures = 0
    recovery = dcm.compute(np.zeros(3), np.zeros(3), (0, 0), (0, 0))
    targets = planner.current_targets(0)
    exit_reason = "time limit reached"
    next_sync = next_frame = 0.0
    frame_dt = 1.0 / max(1, cfg.fps)
    t_wall = time.perf_counter()

    try:
        while robot.time < cfg.max_time:
            if viewer is not None and not viewer.is_running():
                exit_reason = "viewer window closed"
                break
            t = robot.time

            robot.clear_push()
            pushing = bool(p.force) and p.time < t < p.time + p.duration
            if p.force and p.viz == "ball":
                if not ball_fired and t >= p.time:
                    v = robot.launch_projectile(p.axis, p.force, p.duration)
                    ball_fired = True
                    if v is not None:
                        print(f"t={t:.2f}s  projectile launched: "
                              f"m={robot.ball_mass * 1000:.0f} g, v={v:.2f} m/s")
            elif pushing:
                robot.apply_push(p.force, p.axis)

            if t < g.settle_time:
                z_now = g.squat_start_z
            elif t < g.settle_time + g.squat_time:
                u = (t - g.settle_time) / g.squat_time
                z_now = g.squat_start_z - (g.squat_start_z - g.z_hip) * u
            else:
                z_now = g.z_hip
                if not walking:
                    walking = True
                    print(f"t={t:.2f}s  squat done, ZMP planner started "
                          f"(first step in {g.initial_delay * g.dt_mpc:.1f}s)")

            if walking and (round(t / dt) % mpc_every == 0):
                ref_lat, ref_fwd = planner.zmp_ref(k)
                win_lat, win_fwd = planner.preview_window(k)
                preview_lat.update(ref_lat, win_lat)
                preview_fwd.update(ref_fwd, win_fwd)

                com_p, com_v = robot.com_state()
                recovery = dcm.compute(
                    com_p, com_v,
                    (preview_lat.position, preview_lat.velocity),
                    (preview_fwd.position, preview_fwd.velocity),
                    planner.swing_phase(k))

                targets = planner.update(k, recovery.as_tuple)
                k += 1
            else:
                targets = planner.current_targets(k)

            origin = torso_origin(z_now, preview_lat.position, preview_fwd.position)
            foot_a = to_model(targets.a_lat, targets.a_fwd, ankle_h + targets.a_lift) - origin
            foot_b = to_model(targets.b_lat, targets.b_fwd, ankle_h + targets.b_lift) - origin
            try:
                q_a = ik.solve(foot_a, True, refine=cfg.refine_ik)
                q_b = ik.solve(foot_b, False, refine=cfg.refine_ik)
                if not (np.all(np.isfinite(q_a)) and np.all(np.isfinite(q_b))):
                    raise FloatingPointError("IK returned NaN")
            except Exception as e:
                ik_failures += 1
                print(f"t={t:.3f}s IK FAILURE ({ik_failures}): {e}")
                if ik_failures >= cfg.max_ik_failures:
                    exit_reason = "repeated IK failures"
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
                exit_reason = "NaN (simulation diverged)"
                print(f"t={t:.3f}s SIMULATION DIVERGED (NaN).")
                break

            com_p, _ = robot.com_state()
            log.record(t=t,
                       com_fwd=r.hip_y - com_p[1], com_lat=com_p[0] - r.mid_x,
                       plan_fwd=preview_fwd.position, plan_lat=preview_lat.position,
                       base_z=robot.base_height,
                       dcm_lat=recovery.d_lat, dcm_fwd=recovery.d_fwd)

            if log.due(t):
                cf = log.get("com_fwd")
                hip_z = robot.base_height + r.body_ref_offset[2]
                wall = time.perf_counter() - t_wall
                print(f"  t={t:4.1f}s  fwd={cf[-1] - cf[0]:+.3f}m  hip_z={hip_z:.4f}"
                      f" (sag {(z_now - hip_z) * 1000:+5.1f}mm)"
                      f"  swing={targets.swing or '-'}"
                      f"  step={max(0, k - g.initial_delay) // g.step_time}"
                      f"  contacts={robot.n_contacts}"
                      f"  rtf={t / wall if wall > 1e-6 else 0:.2f}x"
                      f"  dcm=({recovery.err_lat * 1000:+5.1f},"
                      f"{recovery.err_fwd * 1000:+5.1f})mm")

            if robot.base_height < cfg.fall_height:
                exit_reason = "robot fell"
                print(f"t={t:.2f}s  base z={robot.base_height:.3f} -- robot fell.")
                break

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
            print(f"video written: {cfg.record}")

    print(log.summary(exit_reason, g.first_step_time, ik_failures, cfg.max_time))
    if cfg.plot_enabled and log.plot(cfg.plot_path, p.time, p.force):
        print(f"plot: {cfg.plot_path}")


if __name__ == "__main__":
    main()
