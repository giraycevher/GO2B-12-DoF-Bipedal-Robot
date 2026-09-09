#!/usr/bin/env python3
"""
main.py
=======
Main orchestration loop for the GO2B 12-DoF biped: ZMP preview-control
(LIPM + DARE) plus a Kajita analytical IK walking controller, with DCM
(capture-point) push recovery and an optional torso-tilt -> ankle
stabilizer.

This module only wires the pieces together; every algorithm lives in
its own package, per the architecture described in README.md:

    config/parameters.py               all constants (gains, limits, sim params)
    kinematics/leg_ik.py               analytical IK/FK (LegKinematics)
    controllers/preview_controller.py  LIPM + DARE preview control (ZMPPreviewController)
    controllers/push_recovery.py       DCM capture-point + ankle stabilizer
    planners/step_planner.py           footstep / swing-leg trajectory (StepPlanner)
    utils/data_logger.py               telemetry, summary, plotting
    sim/mujoco_interface.py            the ONLY module that touches MuJoCo
    main.py                            this file

Usage
-----
    python main.py                # visual sim, free camera
    python main.py --headless     # no window, log + plot

Push testing
------------
    python main.py --push 15                     # red arrow (default)
    python main.py --push 15 --push-viz ball      # real ball impact
    python main.py --push 15 --push-viz none      # no visual
    python main.py --push 15 --push-axis fwd      # push forward
    python main.py --push 15 --no-dcm             # disable push-recovery

Recording
---------
    python main.py --record walk.mp4
    python main.py --record y.mp4 --headless --width 1920 --height 1080 --fps 60

Requires: pip install imageio imageio-ffmpeg  (or: opencv-python)
"""

import argparse
import csv
import json
import platform
import sys
import time

# Raise Windows timer resolution to 1 ms (default ~15.6 ms breaks real-time pacing)
try:
    import ctypes
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

import numpy as np

try:
    import mujoco
except ImportError:
    sys.exit("mujoco not installed:  pip install mujoco")

from config.parameters import SimConfig
from controllers.preview_controller import ZMPPreviewController
from controllers.push_recovery import AnkleStabilizer, DCMRecovery, RecoveryOutput
from kinematics.leg_ik import LegKinematics
from planners.step_planner import StepPlanner
from sim.mujoco_interface import MujocoRobot, VideoRecorder, draw_push
from utils.data_logger import TelemetryLogger


# ===========================================================================
#  FRAME TRANSFORM
# ===========================================================================
def to_model(rcfg, lat, fwd, up):
    """Gait frame (lateral, forward, height) -> world/model axes.

    Robot faces model -Y; lateral is model +X.
    """
    return np.array([rcfg.mid_x + lat, rcfg.hip_y - fwd, up])


# ===========================================================================
#  CLI
# ===========================================================================
def build_arg_parser(sim: SimConfig) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--xml", default=sim.robot.xml_path)
    ap.add_argument("--zhip", type=float, default=sim.gait.z_hip,
                    help="hip-center height above ground [m]")
    ap.add_argument("--step-len", type=float, default=sim.gait.step_len)
    ap.add_argument("--push", type=float, default=sim.push.force,
                    help="push force [N] (applied at t=6.0s for 0.1 s)")
    ap.add_argument("--push-axis", default=sim.push.axis, choices=["lat", "fwd"],
                    help="push direction: lat = lateral, fwd = forward")
    ap.add_argument("--push-viz", default=sim.push.viz,
                    choices=["arrow", "ball", "none"],
                    help="how to visualize the push: arrow = red arrow "
                         "(force via xfrc_applied), ball = a real ball is "
                         "launched at the robot, none = no visual")
    ap.add_argument("--no-dcm", action="store_true",
                    help="disable DCM foot-placement feedback")
    ap.add_argument("--k-dcm", type=float, default=sim.dcm.gain,
                    help="DCM feedback gain (0 = off, 1.0 = full)")
    ap.add_argument("--time", type=float, default=sim.max_time)
    ap.add_argument("--record", metavar="FILE.mp4", default=None,
                    help="record the walk as mp4 (offscreen render)")
    ap.add_argument("--fps", type=int, default=sim.fps, help="video frame rate")
    ap.add_argument("--width", type=int, default=sim.width)
    ap.add_argument("--height", type=int, default=sim.height)
    ap.add_argument("--speed", type=float, default=sim.speed,
                    help="playback speed: 1.0 = real time, 0.5 = slow motion, "
                         "0 = no throttling (as fast as possible)")
    ap.add_argument("--ankle-fb", action="store_true",
                    help="enable torso-tilt -> ankle feedback")
    ap.add_argument("--ankle-fb-sign", type=float, default=sim.ankle.sign,
                    help="feedback sign; try -1 if it makes things worse")
    ap.add_argument("--refine-ik", action="store_true",
                    help="refine the Kajita solution to ~0.01 mm via damped least squares")
    ap.add_argument("--no-plot", action="store_true")
    return ap


def apply_cli_overrides(sim: SimConfig, args) -> None:
    sim.robot.xml_path = args.xml
    sim.gait.z_hip = args.zhip
    sim.gait.step_len = args.step_len
    sim.push.force = args.push
    sim.push.axis = args.push_axis
    sim.push.viz = args.push_viz
    sim.dcm.enabled = not args.no_dcm
    sim.dcm.gain = args.k_dcm
    sim.max_time = args.time
    sim.record = args.record or ""
    sim.fps = args.fps
    sim.width = args.width
    sim.height = args.height
    sim.speed = args.speed
    sim.ankle.enabled = args.ankle_fb
    sim.ankle.sign = args.ankle_fb_sign
    sim.refine_ik = args.refine_ik
    sim.plot_enabled = not args.no_plot


# ===========================================================================
#  MAIN PROGRAM
# ===========================================================================
def main():
    sim = SimConfig()
    args = build_arg_parser(sim).parse_args()
    apply_cli_overrides(sim, args)

    rcfg, gait = sim.robot, sim.gait

    robot = MujocoRobot(rcfg)
    ik = LegKinematics(rcfg)
    planner = StepPlanner(gait)
    preview_lat = ZMPPreviewController(gait.dt_mpc, gait.z_hip, gait.n_preview)
    preview_fwd = ZMPPreviewController(gait.dt_mpc, gait.z_hip, gait.n_preview)
    dcm = DCMRecovery(sim.dcm, rcfg)
    ankle_stab = AnkleStabilizer(sim.ankle)
    telemetry = TelemetryLogger(report_period=sim.report_period)

    dt = robot.timestep
    preview_every = max(1, int(round(gait.dt_mpc / dt)))

    print(f"model        : {rcfg.xml_path}")
    print(f"nq={robot.model.nq} nv={robot.model.nv} nu={robot.model.nu} timestep={dt}")
    print(f"robot mass   : {robot.robot_mass:.4f} kg")

    # --- measurements ---
    ankle_h = robot.measure_ankle_height()
    print(f"ankle center height (measured): {ankle_h * 1000:.1f} mm")
    drop = gait.z_hip - ankle_h
    print(f"leg drop     : {drop:.4f} m = {drop / rcfg.max_reach * 100:.0f}% of max reach "
          f"({rcfg.max_reach:.4f} m)")
    if drop > rcfg.max_reach * 0.95:
        sys.exit(f"ERROR: --zhip {gait.z_hip} is too high, the leg would lock out. "
                 f"Use at most {ankle_h + rcfg.max_reach * 0.92:.3f}.")
    if drop < rcfg.max_reach * 0.45:
        print("WARNING: squatting very deep, the knee may exceed 120 degrees.")

    def stance_q(z_hip_now, lat_a=gait.lat_a, lat_b=gait.lat_b,
                 fwd_a=0.0, fwd_b=0.0, lift_a=0.0, lift_b=0.0):
        """Joint angles for both legs with the hip reference at z_hip_now."""
        hip_ref_w = to_model(rcfg, 0.0, 0.0, z_hip_now)
        torso_w = hip_ref_w - rcfg.body_ref_offset
        fA = to_model(rcfg, lat_a, fwd_a, ankle_h + lift_a) - torso_w
        fB = to_model(rcfg, lat_b, fwd_b, ankle_h + lift_b) - torso_w
        return (ik.solve(fA, left=True, refine=sim.refine_ik),
                ik.solve(fB, left=False, refine=sim.refine_ik), torso_w)

    qA0, qB0, torso_w0 = stance_q(gait.squat_start_z)
    robot.set_pose(torso_w0, qA0, qB0)
    robot.settle_on_ground()
    print(f"initial torso height: {robot.base_height:.4f} m")
    robot.send(qA0, qB0)

    # --- state ---
    qA, qB = qA0.copy(), qB0.copy()   # fallback commands if the very first IK call fails
    ik_fail = 0
    step_index = 0
    walking = False
    targets = planner.current_targets(step_index)
    dcm_out = RecoveryOutput()

    # Uses MuJoCo's default free camera (no fixed camera forced).
    viewer = None
    if not args.headless:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(robot.model, robot.data)
        viewer.__enter__()

    # --- recording ---
    recorder = None
    next_frame = 0.0
    frame_dt = 1.0 / max(1, sim.fps)
    if sim.record:
        recorder = VideoRecorder(robot.model, sim.record, sim.fps, sim.width, sim.height)
        print(f"recording    : {sim.record}  {sim.width}x{sim.height} @ {sim.fps} fps")

    trace_rows = []  # raw trace: time, 12 joint commands, base/CoM state, contacts
    ball_fired = False
    t_wall = time.perf_counter()
    exit_reason = "time limit reached"
    next_sync = 0.0

    try:
        while robot.time < sim.max_time:
            if viewer is not None and not viewer.is_running():
                exit_reason = "viewer window closed"
                break
            t = robot.time

            robot.clear_push()
            pushing = bool(sim.push.force) and sim.push.time < t < sim.push.time + sim.push.duration

            if sim.push.force and sim.push.viz == "ball":
                # Real projectile instead of a force: speed is chosen so the
                # ball delivers the same impulse (v = F*dt / m_ball).
                if not ball_fired and t >= sim.push.time:
                    v = robot.launch_projectile(sim.push.axis, sim.push.force, sim.push.duration)
                    if v is not None:
                        ball_fired = True
                        print(f"t={t:.2f}s  BALL LAUNCHED: m={robot.ball_mass * 1000:.0f} g, "
                              f"v={v:.2f} m/s, impulse={abs(sim.push.force) * sim.push.duration:.2f} Ns")
            elif pushing:
                robot.apply_push(sim.push.force, sim.push.axis)

            if t < gait.settle_time:
                z_now = gait.squat_start_z
            elif t < gait.settle_time + gait.squat_time:
                u = (t - gait.settle_time) / gait.squat_time
                z_now = gait.squat_start_z - (gait.squat_start_z - gait.z_hip) * u
            else:
                z_now = gait.z_hip
                if not walking:
                    walking = True
                    print(f"t={t:.2f}s  squat finished, ZMP planner started "
                          f"(first step in {gait.initial_delay * gait.dt_mpc:.1f}s)")

            # ------------------------------------------- PREVIEW CONTROL
            if walking and (round(t / dt) % preview_every == 0):
                lat_now, fwd_now = planner.zmp_ref(step_index)
                lat_win, fwd_win = planner.preview_window(step_index)
                preview_lat.update(lat_now, lat_win)
                preview_fwd.update(fwd_now, fwd_win)

                swing_phase = planner.swing_phase(step_index)
                if sim.dcm.enabled and sim.dcm.gain > 0.0:
                    com_pos, com_vel = robot.com_state()
                    dcm_out = dcm.compute(
                        com_pos, com_vel,
                        (preview_lat.position, preview_lat.velocity),
                        (preview_fwd.position, preview_fwd.velocity),
                        swing_phase)
                else:
                    dcm_out = RecoveryOutput()

                targets = planner.update(step_index, correction=dcm_out.as_tuple)
                step_index += 1

            # -------------------------------------------------------- IK
            hip_ref_w = to_model(rcfg, preview_lat.position, preview_fwd.position, z_now)
            torso_w = hip_ref_w - rcfg.body_ref_offset
            fA_b = to_model(rcfg, targets.a_lat, targets.a_fwd, ankle_h + targets.a_lift) - torso_w
            fB_b = to_model(rcfg, targets.b_lat, targets.b_fwd, ankle_h + targets.b_lift) - torso_w
            try:
                qA = ik.solve(fA_b, left=True, refine=sim.refine_ik)
                qB = ik.solve(fB_b, left=False, refine=sim.refine_ik)
                if not (np.all(np.isfinite(qA)) and np.all(np.isfinite(qB))):
                    raise FloatingPointError(f"IK NaN: A={qA} B={qB}")
            except Exception as e:
                ik_fail += 1
                print(f"t={t:.3f}s IK ERROR ({ik_fail}): {e}")
                if ik_fail >= sim.max_ik_failures:
                    exit_reason = "repeated IK failures"
                    print("Repeated IK failures -- stopping.")
                    break

            cmdA, cmdB = qA.copy(), qB.copy()
            if sim.ankle.enabled and walking:
                d_pitch, d_roll = ankle_stab.compute(robot.base_rotation, robot.base_angular_velocity)
                cmdA[4] += d_pitch; cmdB[4] += d_pitch
                cmdA[5] -= d_roll; cmdB[5] -= d_roll

            cmdA = np.clip(cmdA, rcfg.q_min, rcfg.q_max)
            cmdB = np.clip(cmdB, rcfg.q_min, rcfg.q_max)
            robot.send(cmdA, cmdB)
            robot.step()

            if not robot.is_finite():
                exit_reason = "NaN (simulation diverged)"
                print(f"t={t:.3f}s SIMULATION DIVERGED (NaN).")
                break

            com = robot.data.sensor('com_pos').data
            telemetry.record(
                t=t,
                com_fwd=rcfg.hip_y - float(com[1]),
                com_lat=float(com[0]) - rcfg.mid_x,
                plan_fwd=preview_fwd.position,
                plan_lat=preview_lat.position,
                base_z=robot.base_height,
                dcm_lat=dcm_out.d_lat,
                dcm_fwd=dcm_out.d_fwd,
            )

            # --- raw trace row: time, 12 joint commands, base/CoM state, contacts ---
            com_vel = robot.data.sensor('com_vel').data
            base_quat = robot.data.sensor('torso_quat').data
            touch_l = float(robot.data.sensor('touch_left').data[0])
            touch_r = float(robot.data.sensor('touch_right').data[0])
            cmd12 = robot.data.ctrl[robot.ctrl_idx].copy()   # 12 values, in rcfg.joint_order
            act12 = robot.data.qpos[robot.qadr].copy()       # actual joint angles
            row = {"time_s": t, "push_active": int(pushing)}
            for name, val in zip(rcfg.joint_order, cmd12):
                row[f"command_{name}_rad"] = float(val)
            for name, val in zip(rcfg.joint_order, act12):
                row[f"actual_{name}_rad"] = float(val)
            row["base_x_m"] = float(robot.data.qpos[0])
            row["base_y_m"] = float(robot.data.qpos[1])
            row["base_z_m"] = float(robot.data.qpos[2])
            row["base_quat_w"] = float(base_quat[0])
            row["base_quat_x"] = float(base_quat[1])
            row["base_quat_y"] = float(base_quat[2])
            row["base_quat_z"] = float(base_quat[3])
            row["com_x_m"] = float(com[0])
            row["com_y_m"] = float(com[1])
            row["com_z_m"] = float(com[2])
            row["com_vx_m_s"] = float(com_vel[0])
            row["com_vy_m_s"] = float(com_vel[1])
            row["com_vz_m_s"] = float(com_vel[2])
            row["left_contact"] = touch_l
            row["right_contact"] = touch_r
            trace_rows.append(row)

            # --- one diagnostic line per second ---
            if telemetry.due(t):
                hip_meas = robot.base_height + rcfg.body_ref_offset[2]
                sag = (z_now - hip_meas) * 1000.0
                cf_hist = telemetry.get("com_fwd")
                cf = (cf_hist[-1] - cf_hist[0]) if cf_hist.size else 0.0
                wall = time.perf_counter() - t_wall
                rtf = (t / wall) if wall > 1e-6 else 0.0
                print(f"  t={t:4.1f}s  fwd={cf:+.3f}m  hip z={hip_meas:.4f}"
                      f" (sag {sag:+5.1f}mm)  swing={targets.swing or '-'}"
                      f"  step={max(0, step_index - gait.initial_delay) // gait.step_time}"
                      f"  contacts={robot.n_contacts}  speed={rtf:.2f}x"
                      f"  DCM err=({dcm_out.err_lat * 1000:+5.1f},{dcm_out.err_fwd * 1000:+5.1f})mm"
                      f" corr=({dcm_out.d_lat * 1000:+5.1f},{dcm_out.d_fwd * 1000:+5.1f})mm")

            if robot.base_height < sim.fall_height:
                exit_reason = "robot fell (torso z below fall_height)"
                print(f"t={t:.2f}s  torso z={robot.base_height:.3f} -- robot fell.")
                break

            # --- video frame ---
            if recorder is not None and robot.time >= next_frame:
                next_frame += frame_dt
                recorder.capture(robot.data)
                if sim.push.viz == "arrow" and arrow_alpha > 0:
                    draw_push(recorder.scene, robot, sim.push.axis, sim.push.force,
                              arrow_alpha, reset=False)
                recorder.write()

            # viewer.sync() is expensive (~40 meshes per body), so it's
            # throttled to view_dt instead of every 1ms step.
            # Push arrow: full opacity during the push, fades over 0.6 s after.
            arrow_alpha = 0.0
            if sim.push.force and sim.push.viz == "arrow":
                if pushing:
                    arrow_alpha = 1.0
                elif sim.push.time <= t < sim.push.time + sim.push.duration + sim.push.arrow_hold:
                    arrow_alpha = 1.0 - (t - sim.push.time - sim.push.duration) / sim.push.arrow_hold

            if viewer is not None and robot.time >= next_sync:
                next_sync += sim.view_dt
                if sim.push.viz == "arrow":
                    # call even at alpha=0, to clear the previous arrow
                    draw_push(viewer.user_scn, robot, sim.push.axis, sim.push.force,
                              arrow_alpha, reset=True)
                viewer.sync()
                # Real-time throttling: sleep to an absolute target time via
                # perf_counter (avoids drift from Windows' ~16ms time.time()).
                if sim.speed > 0 and recorder is None:
                    target = t_wall + robot.time / sim.speed
                    now = time.perf_counter()
                    if target > now:
                        time.sleep(target - now)
    finally:
        if viewer is not None:
            viewer.__exit__(None, None, None)
        if recorder is not None:
            recorder.close()
            print(f"video written: {sim.record}")

    print(telemetry.summary(exit_reason, gait.first_step_time, ik_fail, sim.max_time))

    # ----------------------------------------------------------------------
    #  Raw trace + provenance (CSV + JSON): control/planner/IK/model
    #  untouched, logging only.
    # ----------------------------------------------------------------------
    if trace_rows:
        try:
            import scipy as _scipy_ver
            scipy_version = _scipy_ver.__version__
        except Exception:
            scipy_version = "not installed"
        base_name = f"main_walk_trace_push{sim.push.force:g}n"
        csv_path = base_name + ".csv"
        json_path = base_name + ".metadata.json"
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0]))
            writer.writeheader()
            writer.writerows(trace_rows)

        t_hist = telemetry.get("t")
        bz_hist = telemetry.get("base_z")
        ok_final = t_hist.size > 0 and bz_hist.min() > 0.12 and t_hist[-1] > sim.max_time * 0.95
        metadata = {
            "source_file": "main.py (modular: config/controllers/kinematics/planners/"
                            "sim/utils packages; control/planner/IK/model logic unchanged)",
            "invocation": " ".join(sys.argv),
            "xml_path": rcfg.xml_path,
            "push_force_n": sim.push.force,
            "push_axis": sim.push.axis,
            "push_time_s": sim.push.time,
            "push_duration_s": sim.push.duration,
            "push_impulse_n_s": round(abs(sim.push.force) * sim.push.duration, 4),
            "dcm_feedback_enabled": sim.dcm.enabled,
            "k_dcm": sim.dcm.gain if sim.dcm.enabled else 0.0,
            "robot_mass_kg": robot.robot_mass,
            "source_timestep_s": dt,
            "rows": len(trace_rows),
            "exit_reason": exit_reason,
            "end_time_s": float(t_hist[-1]) if t_hist.size else 0.0,
            "end_base_height_m": float(bz_hist[-1]) if bz_hist.size else None,
            "min_base_height_m": float(bz_hist.min()) if bz_hist.size else None,
            "ik_failures": ik_fail,
            "result": "STOOD / WALKED" if ok_final else "FELL or ENDED EARLY",
            "environment": {
                "os": platform.platform(),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "scipy": scipy_version,
                "mujoco": mujoco.__version__,
            },
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        print(f"raw trace    : {csv_path}")
        print(f"provenance   : {json_path}")

    if sim.plot_enabled:
        if telemetry.plot(sim.plot_path, push_time=sim.push.time, push_force=sim.push.force):
            print(f"plot: {sim.plot_path}")


if __name__ == "__main__":
    main()
