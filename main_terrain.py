#!/usr/bin/env python3
"""
main_terrain.py
===============
Walk the GO2B biped over procedurally generated uneven ground (forest-floor
style: soil undulations, roots, stones, pits).

Same pipeline as main.py -- ZMP preview control (controllers/) + Kajita IK
(kinematics/) -- plus the terrain-adaptation layer in
planners/terrain_planner.py and the heightfield in sim/terrain.py.

Usage (works from any folder)
-----------------------------
    python main_terrain.py                        # rough = 0.6, viewer window
    python main_terrain.py --rough 1.0            # harder terrain
    python main_terrain.py --rough 1.0 --seed 7   # a different terrain
    python main_terrain.py --headless             # no window, summary + plot
    python main_terrain.py --rough 1.0 --record walk.mp4

Experiments
-----------
    --blind          no height map: adapt through contact sensing only
    --no-slope       keep the soles level instead of matching the terrain
    --no-touchdown   disable early-touchdown detection
    --dcm            enable DCM foot placement (off by default, see README)
    --no-ankle       disable the ankle balance loop (on by default, see README)
    --push F         push of F newtons for 0.1 s at t = 6 s (--push-axis lat|fwd)
    --dump d.npz     100 Hz log of every controller signal

    python tools/run_terrain_tests.py      # batch of experiments -> results table

Note: on a real robot the terrain height would come from a depth camera or
LiDAR with elevation mapping; here it is read from the simulator (`--blind`
shows what happens without it).
"""

import argparse
import math
import os
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

from config.parameters import (PROJECT_ROOT, RobotConfig, TerrainConfig,
                               TerrainGaitConfig, resolve_path)
from planners.terrain_planner import Footprint, TerrainStepPlanner
from sim.mujoco_interface import MujocoRobot, VideoRecorder, draw_push
from sim.terrain import Terrain

PUSH_TIME, PUSH_DUR, ARROW_HOLD = 6.0, 0.1, 0.6
DUMP_COLS = ("t swing u touchA touchB com_l com_f vel_l vel_f com_z plan_l plan_vl plan_f plan_vf "
             "raw_l raw_f corr_l corr_f A_lat A_fwd A_g A_lift B_lat B_fwd B_g B_lift "
             "A_ank_l A_ank_f A_ank_z B_ank_l B_ank_f B_ank_z roll pitch g_ref reach td").split()


def build_arg_parser() -> argparse.ArgumentParser:
    G, T = TerrainGaitConfig(), TerrainConfig()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--xml", default=str(PROJECT_ROOT / "robot_walk.xml"),
                    help="robot model with a flat 'floor' (replaced by the terrain)")
    # terrain
    ap.add_argument("--rough", type=float, default=T.rough,
                    help="roughness: 0 = flat, 0.6 = default, 1.0 = hard, 1.5 = very hard")
    ap.add_argument("--seed", type=int, default=T.seed, help="terrain random seed")
    ap.add_argument("--soil-friction", type=float, default=T.friction)
    ap.add_argument("--soil-solref", type=float, default=T.solref,
                    help="soil contact time constant [s]; larger = softer")
    ap.add_argument("--skip-terrain-check", action="store_true")
    # gait
    ap.add_argument("--zhip", type=float, default=G.z_hip)
    ap.add_argument("--step-len", type=float, default=G.step_len)
    ap.add_argument("--step-height", type=float, default=G.step_height)
    ap.add_argument("--step-time", type=float, default=G.step_time * G.dt_mpc,
                    help="step duration [s]")
    ap.add_argument("--steps", type=int, default=G.n_steps)
    ap.add_argument("--time", type=float, default=None, help="sim time [s] (default: automatic)")
    # adaptation switches
    ap.add_argument("--blind", action="store_true", help="do not use the height map")
    ap.add_argument("--no-slope", action="store_true")
    ap.add_argument("--no-touchdown", action="store_true")
    ap.add_argument("--dcm", action="store_true", help="enable DCM foot placement")
    ap.add_argument("--k-dcm", type=float, default=1.0, help="DCM gain when --dcm is on")
    ap.add_argument("--dcm-legacy", action="store_true",
                    help="old DCM behaviour (correction fades to zero before landing)")
    ap.add_argument("--dcm-lipm-ref", action="store_true",
                    help="measure the DCM error against the LIPM state (old, biased)")
    ap.add_argument("--no-zmp-follow", action="store_true",
                    help="do not move the ZMP reference / plan with the foot (old behaviour)")
    ap.add_argument("--no-ankle", action="store_true", help="disable the ankle balance loop")
    ap.add_argument("--ankle-kp", type=float, default=None)
    ap.add_argument("--ankle-kd", type=float, default=None)
    ap.add_argument("--ankle-kdcm", type=float, default=None,
                    help="ankle correction per metre of DCM error [rad/m]")
    ap.add_argument("--refine-ik", action="store_true")
    # push
    ap.add_argument("--push", type=float, default=0.0,
                    help=f"push force [N] at t={PUSH_TIME}s for {PUSH_DUR}s")
    ap.add_argument("--push-axis", default="lat", choices=["lat", "fwd"])
    ap.add_argument("--push-viz", default="arrow", choices=["arrow", "none"])
    # output
    ap.add_argument("--record", metavar="FILE.mp4", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="1.0 = real time, 0 = as fast as possible")
    ap.add_argument("--plot", default="walk_terrain.png", help="plot file ('' = no plot)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--dump", metavar="FILE.npz", default=None,
                    help="100 Hz diagnostic log (numpy npz)")
    return ap


def gait_from_args(args, robot_mass):
    G = TerrainGaitConfig()
    G.z_hip = args.zhip
    G.step_len = args.step_len
    G.step_height = args.step_height
    G.step_time = max(20, int(round(args.step_time / G.dt_mpc)))
    G.n_steps = args.steps
    G.use_map = not args.blind
    G.use_slope = not (args.no_slope or args.blind)
    G.use_touchdown = not args.no_touchdown
    G.k_dcm = args.k_dcm if args.dcm else 0.0
    G.dcm_hold = not args.dcm_legacy
    G.zmp_follow = not args.no_zmp_follow
    G.ankle_balance = not args.no_ankle
    if G.ankle_balance:
        if args.ankle_kp is not None:
            G.ankle_kp = args.ankle_kp
        if args.ankle_kd is not None:
            G.ankle_kd = args.ankle_kd
        if args.ankle_kdcm is not None:
            G.ankle_k_dcm = args.ankle_kdcm
    touch_force = G.touch_force_ratio * robot_mass * 9.81
    return G, touch_force


def main():
    args = build_arg_parser().parse_args()
    rcfg = RobotConfig()

    # ------------------------------------------------------------ terrain + model
    terr = Terrain.from_config(
        TerrainConfig(rough=args.rough, seed=args.seed), start_y=rcfg.hip_y)
    st = terr.stats()
    src = resolve_path(args.xml)
    dst = os.path.join(os.path.dirname(src), "robot_terrain.xml")   # same folder: meshdir is relative
    terr.write_xml(src, dst, friction=args.soil_friction, solref=args.soil_solref)
    rcfg.xml_path = dst
    robot = MujocoRobot(rcfg)
    terr.apply_to_model(robot.model)             # BEFORE opening a viewer / renderer
    dt = robot.timestep
    print(f"model        : {dst}  (from {src})")
    print(f"terrain      : rough={args.rough} seed={args.seed}  {st['nrow']}x{st['ncol']} grid  "
          f"height [{st['min_mm']:+.1f}, {st['max_mm']:+.1f}] mm  "
          f"slope mean {st['slope_mean_deg']:.1f} / p99 {st['slope_p99_deg']:.1f} deg")
    print(f"robot mass   : {robot.robot_mass:.4f} kg")
    if args.soil_solref < 2 * dt:
        sys.exit(f"ERROR: --soil-solref must be at least {2 * dt} (2 x timestep).")

    # ------------------------------------------------------------ measurements
    ankle_h = robot.measure_ankle_height()
    fps_ = []
    for foot, joint in (("left_foot", "left_ankle_roll"), ("right_foot", "right_ankle_roll")):
        b = robot.foot_sole_bounds(foot, joint)
        fp = Footprint(*b) if b else Footprint()
        if not (0.03 < fp.fwd_max - fp.fwd_min < 0.25 and 0.02 < fp.width < 0.20):
            print(f"WARNING: implausible sole measurement for {foot}; using defaults.")
            fp = Footprint()
        fps_.append(fp)
    fp_a, fp_b = fps_
    print(f"ankle height : {ankle_h * 1000:.1f} mm   sole: fwd [{fp_a.fwd_min * 1000:+.0f}, "
          f"{fp_a.fwd_max * 1000:+.0f}] mm, lat [{fp_a.lat_min * 1000:+.0f}, {fp_a.lat_max * 1000:+.0f}] mm")

    gait, touch_force = gait_from_args(args, robot.robot_mass)
    drop = gait.z_hip - ankle_h
    print(f"leg drop     : {drop:.4f} m = {drop / rcfg.max_reach * 100:.0f}% of max reach")
    if drop > rcfg.max_reach * 0.90:
        sys.exit(f"ERROR: --zhip too high for rough ground; use at most "
                 f"{ankle_h + rcfg.max_reach * 0.85:.3f}.")
    preview_every = max(1, int(round(gait.dt_mpc / dt)))

    planner = TerrainStepPlanner(
        gait, rcfg, terr, ankle_h, fp_a, fp_b,
        com_model=None if args.dcm_lipm_ref else robot.make_com_model(),
        touch_force=touch_force, refine_ik=args.refine_ik)
    walk_start = gait.settle_time + gait.squat_time
    t_end = args.time if args.time is not None else \
        walk_start + (planner.stop_tick + 250) * gait.dt_mpc

    # ------------------------------------------------------------ initial pose
    z_start = gait.z_hip + gait.squat_extra
    qA0, qB0, torso_w0 = planner.joint_targets(z_start)
    robot.set_pose(torso_w0, qA0, qB0)
    robot.settle_on_ground()                      # start zone is at z = 0
    robot.send(qA0, qB0)

    if not args.skip_terrain_check and args.rough > 0:
        w = terr.verify(robot.model, robot.data)
        if w is None:
            print("WARNING: terrain check rays missed the ground (check skipped).")
        elif w > TerrainConfig().check_tolerance:
            sys.exit(f"ERROR: the controller's height map and the MuJoCo heightfield differ "
                     f"by {w * 1000:.1f} mm. Run with --blind as a workaround and open an issue.")
        else:
            print(f"terrain check: height map = heightfield (max diff {w * 1000:.2f} mm)  OK")

    # ------------------------------------------------------------ viewer / recording
    viewer = None
    if not args.headless:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(robot.model, robot.data)
        viewer.__enter__()
    recorder, rec_cam = None, None
    next_frame, frame_dt = 0.0, 1.0 / max(1, args.fps)
    if args.record:
        recorder = VideoRecorder(robot.model, args.record, args.fps, args.width, args.height)
        rec_cam = mujoco.MjvCamera()              # follows the torso, recording only
        rec_cam.distance, rec_cam.azimuth, rec_cam.elevation = 0.75, 130.0, -18.0
        print(f"recording    : {args.record}  {args.width}x{args.height} @ {args.fps} fps")

    # ------------------------------------------------------------ main loop
    jid_a = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, "left_ankle_roll")
    jid_b = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, "right_ankle_roll")
    L = {k: [] for k in ("t", "cf", "cl", "pf", "pl", "bz", "dl", "df", "gA", "gB", "gref")}
    D = []
    tilt_max = [0.0, 0.0]
    walking = False
    arrow_alpha = 0.0
    ik_fail = 0
    qA, qB = qA0.copy(), qB0.copy()
    exit_reason = "time limit reached"
    next_report = next_sync = 0.0
    cf0 = None
    t_wall = time.perf_counter()
    try:
        while robot.time < t_end:
            if viewer is not None and not viewer.is_running():
                exit_reason = "viewer window closed"
                break
            t = robot.time

            robot.clear_push()
            pushing = bool(args.push) and PUSH_TIME < t < PUSH_TIME + PUSH_DUR
            if pushing:
                robot.apply_push(args.push, args.push_axis)

            if t < gait.settle_time:
                z_now = z_start
            elif t < walk_start:
                z_now = z_start - gait.squat_extra * (t - gait.settle_time) / gait.squat_time
            else:
                z_now = gait.z_hip
                if not walking:
                    walking = True
                    print(f"t={t:.2f}s  squat finished, terrain planner started")

            # ---------------------------------------- 100 Hz: plan + feedback
            if walking and (round(t / dt) % preview_every == 0):
                c_m, v_m = robot.com_state()
                ta, tb = robot.touch_forces()
                ank_dz = robot.ankle_height_difference(planner.ik)
                planner.tick(c_m, v_m, ta, tb, ank_dz, robot.torso_tilt())
                if args.dump:
                    k_ = planner.step_index - 1 - gait.initial_delay
                    pitch, roll, _, _ = robot.torso_tilt()
                    fa, fb = planner.feet["A"], planner.feet["B"]
                    xa, xb = robot.data.xanchor[jid_a], robot.data.xanchor[jid_b]
                    D.append([t, {"A": 1, "B": 2}.get(planner.swing, 0),
                              ((k_ % gait.step_time) + 1) / gait.step_time if k_ >= 0 else 0.0,
                              ta, tb, c_m[0] - rcfg.mid_x, rcfg.hip_y - c_m[1], v_m[0], -v_m[1], c_m[2],
                              planner.preview_lat.position, planner.preview_lat.velocity,
                              planner.preview_fwd.position, planner.preview_fwd.velocity,
                              planner.raw[0], planner.raw[1], planner.corr[0], planner.corr[1],
                              fa.pos[0], fa.pos[1], fa.g, fa.lift, fb.pos[0], fb.pos[1], fb.g, fb.lift,
                              xa[0] - rcfg.mid_x, rcfg.hip_y - xa[1], xa[2],
                              xb[0] - rcfg.mid_x, rcfg.hip_y - xb[1], xb[2],
                              roll, pitch, planner.g_ref, planner.reach_drop,
                              float(planner.early_td_count)])

            # ---------------------------------------- IK
            try:
                qA_n, qB_n, _ = planner.joint_targets(z_now)
                if not (np.all(np.isfinite(qA_n)) and np.all(np.isfinite(qB_n))):
                    raise FloatingPointError(f"IK NaN: A={qA_n} B={qB_n}")
                qA, qB = planner.apply_ankle_correction(qA_n, qB_n)
            except Exception as e:
                ik_fail += 1
                print(f"t={t:.3f}s IK ERROR ({ik_fail}): {e}")
                if ik_fail >= 3:
                    exit_reason = "repeated IK failures"
                    break
            robot.send(np.clip(qA, rcfg.q_min, rcfg.q_max), np.clip(qB, rcfg.q_min, rcfg.q_max))
            robot.step()

            if not robot.is_finite():
                exit_reason = "NaN (simulation diverged)"
                break

            com = robot.data.sensor("com_pos").data
            if cf0 is None:
                cf0 = rcfg.hip_y - float(com[1])
            L["t"].append(t)
            L["cf"].append(rcfg.hip_y - float(com[1]))
            L["cl"].append(float(com[0]) - rcfg.mid_x)
            L["pf"].append(planner.plan_fwd)
            L["pl"].append(planner.plan_lat)
            L["bz"].append(robot.base_height)
            L["dl"].append(planner.corr[0])
            L["df"].append(planner.corr[1])
            L["gA"].append(planner.feet["A"].g + planner.feet["A"].lift)
            L["gB"].append(planner.feet["B"].g + planner.feet["B"].lift)
            L["gref"].append(planner.g_ref)
            if walking:
                pitch, roll, _, _ = robot.torso_tilt()
                tilt_max[0] = max(tilt_max[0], abs(roll))
                tilt_max[1] = max(tilt_max[1], abs(pitch))

            if t >= next_report:
                next_report += 1.0
                hip_meas = robot.base_height + rcfg.body_ref_offset[2]
                sag = (z_now + planner.g_ref - planner.reach_drop - hip_meas) * 1000.0
                wall = time.perf_counter() - t_wall
                print(f"  t={t:4.1f}s  fwd={L['cf'][-1] - cf0:+.3f}m  sag={sag:+5.1f}mm"
                      f"  swing={planner.swing or '-'}"
                      f"  step={max(0, planner.step_index - gait.initial_delay) // gait.step_time}"
                      f"  ground A/B={planner.feet['A'].g * 1000:+5.1f}/{planner.feet['B'].g * 1000:+5.1f}mm"
                      f"  early td={planner.early_td_count}"
                      f"  DCM err=({planner.raw[0] * 1000:+5.1f},{planner.raw[1] * 1000:+5.1f})mm"
                      f"  contacts={robot.n_contacts}  speed={t / max(wall, 1e-6):.2f}x")

            if robot.base_height < 0.08 + planner.g_ref:
                exit_reason = "robot fell"
                print(f"t={t:.2f}s  torso z={robot.base_height:.3f} -- robot fell.")
                break

            arrow_alpha = 0.0
            if args.push and args.push_viz == "arrow":
                if pushing:
                    arrow_alpha = 1.0
                elif PUSH_TIME <= t < PUSH_TIME + PUSH_DUR + ARROW_HOLD:
                    arrow_alpha = max(0.0, 1.0 - (t - PUSH_TIME - PUSH_DUR) / ARROW_HOLD)

            if recorder is not None and robot.time >= next_frame:
                next_frame += frame_dt
                rec_cam.lookat[:] = [robot.data.qpos[0], robot.data.qpos[1], 0.10]
                recorder.capture(robot.data, rec_cam)
                if arrow_alpha > 0:
                    draw_push(recorder.scene, robot, args.push_axis, args.push,
                              arrow_alpha, reset=False)
                recorder.write()

            if viewer is not None and robot.time >= next_sync:
                next_sync += 0.02
                if args.push_viz == "arrow":
                    draw_push(viewer.user_scn, robot, args.push_axis, args.push,
                              arrow_alpha, reset=True)
                viewer.sync()
                if args.speed > 0 and recorder is None:
                    target = t_wall + robot.time / args.speed
                    now = time.perf_counter()
                    if target > now:
                        time.sleep(target - now)
    finally:
        if viewer is not None:
            viewer.__exit__(None, None, None)
        if recorder is not None:
            recorder.close()
            print(f"video written: {args.record}")

    if args.dump and D:
        np.savez(args.dump, data=np.array(D), cols=np.array(DUMP_COLS))
        print(f"diagnostic log: {args.dump}")

    # ------------------------------------------------------------ summary
    print("\n--- SUMMARY ---")
    print(f"exit reason  : {exit_reason}")
    print(f"sim time     : {L['t'][-1] if L['t'] else 0:.2f} s / {t_end:.2f} s")
    print(f"IK failures  : {ik_fail}     early touchdowns: {planner.early_td_count}")
    if L["t"]:
        print(f"distance     : {L['cf'][-1] - L['cf'][0]:+.3f} m  (plan {planner.planned_distance:.3f} m)")
        print(f"lateral drift: {L['cl'][-1] - L['cl'][0]:+.3f} m")
        print(f"torso tilt   : {math.degrees(tilt_max[0]):.1f} / {math.degrees(tilt_max[1]):.1f} deg "
              f"(max roll / pitch)")
        ok = exit_reason == "time limit reached"
        print(f"RESULT       : {'WALKED / STAYED UP' if ok else 'FELL OR ENDED EARLY'}")

    if args.plot and not args.no_plot and L["t"]:
        plot_run(L, args, args.plot)


def plot_run(L, args, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    tt = L["t"]
    fig, ax = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
    ax[0].plot(tt, L["pf"], "--", label="plan")
    ax[0].plot(tt, np.array(L["cf"]) - L["cf"][0], lw=2, label="torso (measured)")
    ax[0].set_title("Forward")
    ax[0].set_ylabel("m")
    ax[1].plot(tt, L["pl"], "--", label="plan")
    ax[1].plot(tt, np.array(L["cl"]) - L["cl"][0], lw=2, label="torso (measured)")
    ax[1].set_title("Lateral")
    ax[1].set_ylabel("m")
    ax[2].plot(tt, L["bz"], "k", lw=2, label="torso z")
    ax[2].set_title("Torso height")
    ax[2].set_ylabel("m")
    ax[3].plot(tt, np.array(L["gA"]) * 1000, label="foot A sole z")
    ax[3].plot(tt, np.array(L["gB"]) * 1000, label="foot B sole z")
    ax[3].plot(tt, np.array(L["gref"]) * 1000, "k--", label="ground reference")
    ax[3].set_title("Commanded sole height (terrain + lift)")
    ax[3].set_ylabel("mm")
    ax[4].plot(tt, np.array(L["dl"]) * 1000, label="lateral")
    ax[4].plot(tt, np.array(L["df"]) * 1000, label="forward")
    ax[4].set_title("DCM foot-placement correction")
    ax[4].set_ylabel("mm")
    ax[4].set_xlabel("t [s]")
    for a in ax:
        a.grid(True)
        a.legend(loc="upper left", fontsize=8)
        if args.push:
            a.axvline(PUSH_TIME, color="r", ls=":", lw=1.5)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"plot: {path}")


if __name__ == "__main__":
    main()
