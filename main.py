#!/usr/bin/env python3
"""
main_walk.py
============
ZMP preview-control + Kajita analytical IK walking controller for
robot_walk.xml (STL/mesh biped model).

    python main_walk.py                # visual sim, free camera
    python main_walk.py --headless     # no window, log + plot

PUSH TESTING
------------
    python main_walk.py --push 15                     # red arrow (default)
    python main_walk.py --push 15 --push-viz ball      # real ball impact
    python main_walk.py --push 15 --push-viz none      # no visual
    python main_walk.py --push 15 --push-axis fwd      # push forward
    python main_walk.py --push 15 --no-dcm             # disable push-recovery

RECORDING
---------
    python main_walk.py --record walk.mp4
    python main_walk.py --record y.mp4 --headless --width 1920 --height 1080 --fps 60

Requires: pip install imageio imageio-ffmpeg  (or: opencv-python)

FIXES VS. THE OLDER main.py
----------------------------
 A. Preview controller now advances exactly on DT_MPC, not every 3 sim
    steps (that ran the LIPM state 3.3x too fast).
 B. Actuator/joint errors are logged and abort after 3 failures instead
    of being silently swallowed.
 C. Z_HIP is the measured hip-center height above ground, not an assumed
    value that could exceed the leg's actual reach.
 D. Push force is applied to the 'torso' body specifically (previously
    mj_name2id returned -1 and it landed on the wrong body).
 E. Planner output (forward, lateral) maps to model axes as:
        model_X = MID_X + lateral,  model_Y = HIP_Y - forward
"""

import argparse
import csv
import json
import math
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

from robot_utils import DynamicStepPlanner
from preview_control import init_preview_control
from kinematic_mesh import (IK_leg_mesh, ankle_center_fk, R_FOOT_LEVEL,
                            BODY_REF_OFFSET, MAX_REACH, MIRROR_X,
                            HIP_CENTER_L, HIP_CENTER_R, Q_MIN, Q_MAX)

# ===========================================================================
#  PARAMETERS
# ===========================================================================
XML_PATH = "model/robot_walk.xml"

# --- stance ----------------------------------------------------------------
# Z_HIP = hip-center height above ground (also the LIPM z_c). Max leg
# reach is 0.174814 m; don't exceed ~85% of it or the knee locks out.
Z_HIP = 0.195          # ~83% of max reach
SQUAT_START_Z = 0.215  # squats down to Z_HIP at start
SQUAT_TIME = 1.2       # squat duration [s]
SETTLE_TIME = 0.4      # settle in place before squatting

MID_X = MIRROR_X                    # body symmetry plane (model X)
HIP_Y = float(BODY_REF_OFFSET[1])   # model Y of the hip centers
HALF_STANCE = 0.075   # lateral foot offset from centerline (feet under hips)

# Leg A: model X < 0 (left_* joints, IK left=True); Leg B: model X > 0 (right_*)
LAT_A = -HALF_STANCE
LAT_B = +HALF_STANCE

# --- gait --------------------------------------------------------------
DT_MPC = 0.01
N_PREVIEW = 160        # 1.6 s preview horizon
STEP_TIME = 90         # 0.45 s per step
INITIAL_DELAY = 60     # 0.6 s standing balance before the first step
STEP_LEN = 0.04
STEP_HEIGHT = 0.022
N_STEPS = 20
MAX_SIM_TIME = 14.0

PUSH_TIME, PUSH_DUR, PUSH_FORCE = 6.0, 0.1, 25.0
PUSH_AXIS = "lat"      # "lat" = lateral (model +X), "fwd" = forward (model -Y)

# --- DCM (capture point) foot-placement feedback ---------------------------
# Measured CoM is used to compute a DCM error, added to the swing foot's
# landing target every control step (re-planned even mid-swing).
USE_DCM_FEEDBACK = True
K_DCM = 1.0            # 1.0 = fully compensate the DCM error
DCM_LIM_LAT = 0.045    # lateral correction limit [m]
DCM_LIM_FWD = 0.060    # forward correction limit [m]
DCM_FADE = 0.80        # fade correction to zero in the last 20% of swing

VIEW_DT = 0.02         # viewer refresh interval [s] (50 Hz)

# --- torso-tilt -> ankle feedback (off by default, unverified sign) --------
USE_ANKLE_STABILIZER = False
K_ANK_P, K_ANK_D, ANK_CLAMP = 0.60, 0.06, 0.30

JOINT_ORDER = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
               "left_ankle_pitch", "left_ankle_roll",
               "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee",
               "right_ankle_pitch", "right_ankle_roll"]


# ===========================================================================
#  VIDEO RECORDING
# ===========================================================================
def open_video_writer(path, fps):
    """Uses imageio (ffmpeg) if available, otherwise falls back to OpenCV."""
    try:
        import imageio.v2 as imageio
        w = imageio.get_writer(path, fps=fps, macro_block_size=None,
                               codec="libx264", quality=8)
        return ("imageio", w)
    except Exception as e1:
        try:
            import cv2
            return ("cv2", (cv2, None, path, fps))
        except Exception as e2:
            sys.exit(
                "A video library is required:\n"
                "    pip install imageio imageio-ffmpeg\n"
                "(alternative: pip install opencv-python)\n"
                f"imageio error : {e1}\n"
                f"opencv error  : {e2}")


def write_frame(kind, w, frame):
    if kind == "imageio":
        w.append_data(frame)
    else:
        cv2, vw, path, fps = w
        if vw is None:
            h, wd = frame.shape[:2]
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (wd, h))
            w = (cv2, vw, path, fps)
        vw.write(frame[:, :, ::-1])          # RGB -> BGR
    return w


def close_video(kind, w):
    if kind == "imageio":
        w.close()
    else:
        cv2, vw, path, fps = w
        if vw is not None:
            vw.release()


# ===========================================================================
#  FRAME TRANSFORM
# ===========================================================================
def to_model(lat, fwd, up):
    """Gait frame (lateral, forward, height) -> world/model axes.

    Robot faces model -Y; lateral is model +X.
    """
    return np.array([MID_X + lat, HIP_Y - fwd, up])


def quat2mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


class TightStepPlanner(DynamicStepPlanner):
    """DynamicStepPlanner subclass with the built-in push-recovery term
    disabled: swing_target() returns only the nominal footstep, since the
    real feedback (measured-CoM DCM correction) is applied in main().
    """

    def swing_target(self, step_index, state_x, state_y):
        if step_index < self.initial_delay:
            return 0.0, 0.0, False
        k = step_index - self.initial_delay
        t = min(k // self.step_time + 1, self.total_steps - 1)
        return (float(self.footsteps[t, 0]), float(self.footsteps[t, 1]),
                (t % 2 == 0))


# ===========================================================================
#  PUSH VISUALIZATION
# ===========================================================================
def add_arrow(scn, p_from, p_to, width, rgba):
    """Adds an arrow as a user geom to the scene.

    Works for both the passive viewer's user_scn and the offscreen
    Renderer's scene. Falls back to mjv_makeConnector if mjv_connector
    (MuJoCo 3.x) is unavailable.
    """
    if scn is None or scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW,
                        np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, dtype=np.float32))
    a = np.asarray(p_from, dtype=float)
    b = np.asarray(p_to, dtype=float)
    try:
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, a, b)
    except AttributeError:
        mujoco.mjv_makeConnector(g, mujoco.mjtGeom.mjGEOM_ARROW, width,
                                 a[0], a[1], a[2], b[0], b[1], b[2])
    scn.ngeom += 1


def push_arrow_geometry(data, torso_id, axis, force):
    """Start/end points of the push arrow (world, model axes)."""
    p = data.xpos[torso_id].copy()
    d = np.array([1.0, 0.0, 0.0]) if axis == "lat" else np.array([0.0, -1.0, 0.0])
    d = d * (1.0 if force >= 0 else -1.0)
    L = 0.10 + 0.010 * min(abs(force), 40.0)     # length scales with force
    return p - d * L, p + d * 0.05, d


def draw_push(scn, data, torso_id, axis, force, alpha, reset=True):
    """Draws the push arrow and a small sphere at the impact point.

    reset=True  : viewer.user_scn only holds user geoms, must be cleared
                  every frame (otherwise the arrow stays on screen).
    reset=False : renderer.scene already has the robot's geoms from
                  update_scene(); clearing ngeom would erase the robot.
    """
    if scn is None:
        return
    if reset:
        scn.ngeom = 0
    if alpha <= 0.0:
        return
    a, b, d = push_arrow_geometry(data, torso_id, axis, force)
    add_arrow(scn, a, b, 0.012, [1.0, 0.15, 0.1, float(alpha)])
    if scn.ngeom < scn.maxgeom:
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.018, 0.0, 0.0]), b, np.eye(3).flatten(),
                            np.array([1.0, 0.6, 0.1, float(alpha)], dtype=np.float32))
        scn.ngeom += 1


# ===========================================================================
#  MEASURED CoM STATE  (the real feedback signal)
# ===========================================================================
def measured_com_state(model, data, torso_id):
    """Robot's actual center-of-mass position and velocity (world, model axes).

    Uses the torso subtree (torso + both legs) rather than subtree_com[0],
    which would include the push ball and break the DCM feedback.
    """
    mujoco.mj_subtreeVel(model, data)
    return data.subtree_com[torso_id].copy(), data.subtree_linvel[torso_id].copy()


def dcm_correction(c, v, plan_lat, plan_fwd, k, lim_lat, lim_fwd):
    """Foot-placement correction from the DCM (capture point) error.

        xi = c + c_dot / omega,      omega = sqrt(g / z_com)

    xi is "where to place the foot to stop right now." The difference
    between measured and planned xi is exactly how far to move the foot --
    it captures pushes, friction mismatch, and modeling error alike.

    plan_lat / plan_fwd: (position, velocity) pairs from the preview controller.
    Returns: (d_lat, d_fwd) in meters, gait frame.
    """
    z = max(0.05, float(c[2]))
    omega = math.sqrt(9.81 / z)

    # model axes -> gait frame
    c_lat = float(c[0]) - MID_X
    c_fwd = HIP_Y - float(c[1])
    v_lat = float(v[0])
    v_fwd = -float(v[1])

    xi_lat_meas = c_lat + v_lat / omega
    xi_fwd_meas = c_fwd + v_fwd / omega
    xi_lat_plan = plan_lat[0] + plan_lat[1] / omega
    xi_fwd_plan = plan_fwd[0] + plan_fwd[1] / omega

    d_lat = float(np.clip(k * (xi_lat_meas - xi_lat_plan), -lim_lat, lim_lat))
    d_fwd = float(np.clip(k * (xi_fwd_meas - xi_fwd_plan), -lim_fwd, lim_fwd))
    return d_lat, d_fwd, (xi_lat_meas - xi_lat_plan), (xi_fwd_meas - xi_fwd_plan)


# ===========================================================================
#  GROUND / STANCE MEASUREMENT
# ===========================================================================
def lowest_point_of_feet(model, data):
    """World-frame z of the lowest point among the foot collision meshes.

    Measured from the STL vertices rather than assumed from sole thickness.
    """
    zmin = np.inf
    for gi in range(model.ngeom):
        bid = model.geom_bodyid[gi]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname not in ("left_foot", "right_foot"):
            continue
        if model.geom_contype[gi] == 0 and model.geom_conaffinity[gi] == 0:
            continue
        gp = data.geom_xpos[gi]
        gR = data.geom_xmat[gi].reshape(3, 3)
        if model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gi]
            a = model.mesh_vertadr[mid]
            n = model.mesh_vertnum[mid]
            v = model.mesh_vert[a:a + n]
            zmin = min(zmin, float(np.min((v @ gR.T)[:, 2] + gp[2])))
        else:
            zmin = min(zmin, float(gp[2] - model.geom_rbound[gi]))
    return zmin


def measure_ankle_height(model, data, qadr):
    """Height of the ankle center above ground when the sole is flat on it.

    The ankle-roll axis is horizontal and passes through the center, so
    the joint anchor's z is the center's z.
    """
    data.qpos[:] = 0.0
    data.qpos[2] = 0.30
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    zmin = lowest_point_of_feet(model, data)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_ankle_roll")
    ank_z = float(data.xanchor[jid][2])
    return ank_z - zmin


# ===========================================================================
#  MAIN PROGRAM
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--xml", default=XML_PATH)
    ap.add_argument("--zhip", type=float, default=Z_HIP,
                    help="hip-center height above ground [m]")
    ap.add_argument("--step-len", type=float, default=STEP_LEN)
    ap.add_argument("--push", type=float, default=PUSH_FORCE,
                    help="push force [N] (applied at t=6.0s for 0.1 s)")
    ap.add_argument("--push-axis", default=PUSH_AXIS, choices=["lat", "fwd"],
                    help="push direction: lat = lateral, fwd = forward")
    ap.add_argument("--push-viz", default="arrow",
                    choices=["arrow", "ball", "none"],
                    help="how to visualize the push: arrow = red arrow "
                         "(force via xfrc_applied), ball = a real ball is "
                         "launched at the robot, none = no visual")
    ap.add_argument("--no-dcm", action="store_true",
                    help="disable DCM foot-placement feedback (old behavior)")
    ap.add_argument("--k-dcm", type=float, default=K_DCM,
                    help="DCM feedback gain (0 = off, 1.0 = full)")
    ap.add_argument("--time", type=float, default=MAX_SIM_TIME)
    ap.add_argument("--record", metavar="FILE.mp4", default=None,
                    help="record the walk as mp4 (offscreen render)")
    ap.add_argument("--fps", type=int, default=30, help="video frame rate")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed: 1.0 = real time, 0.5 = slow motion, "
                         "0 = no throttling (as fast as possible)")
    ap.add_argument("--ankle-fb", action="store_true",
                    help="enable torso-tilt -> ankle feedback")
    ap.add_argument("--ankle-fb-sign", type=float, default=1.0,
                    help="feedback sign; try -1 if it makes things worse")
    ap.add_argument("--refine-ik", action="store_true",
                    help="refine the Kajita solution to 0.01 mm via DLS (rarely needed)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    z_hip = args.zhip

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    preview_every = max(1, int(round(DT_MPC / dt)))

    print(f"model        : {args.xml}")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} timestep={dt}")
    if model.nu != 12:
        sys.exit(f"ERROR: model has {model.nu} actuators, expected 12. Run fix_robot_xml.py.")

    qadr, ctrl_idx = [], []
    for n in JOINT_ORDER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if jid < 0 or aid < 0:
            sys.exit(f"ERROR: joint/actuator '{n}' not found.")
        qadr.append(model.jnt_qposadr[jid])
        ctrl_idx.append(aid)
    qadr = np.array(qadr); ctrl_idx = np.array(ctrl_idx)

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    if torso_id < 0:
        sys.exit("ERROR: body 'torso' not found.")
    # --- push ball (optional) ---
    ball_qadr = ball_dofadr = -1
    ball_mass = 0.0
    bjid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "push_ball_free")
    if bjid >= 0:
        ball_qadr = int(model.jnt_qposadr[bjid])
        ball_dofadr = int(model.jnt_dofadr[bjid])
        bbid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "push_ball")
        ball_mass = float(model.body_mass[bbid])

    # Robot mass = torso subtree (excludes the push ball etc.)
    robot_mass = float(model.body_subtreemass[torso_id])
    print(f"robot mass   : {robot_mass:.4f} kg")
    if robot_mass < 0.05:
        sys.exit("ERROR: robot mass is ~0. Placeholder inertials are still "
                 "in place, run fix_robot_xml.py.")

    # --- measurements ---
    ankle_h = measure_ankle_height(model, data, qadr)
    print(f"ankle center height (measured): {ankle_h*1000:.1f} mm")
    drop = z_hip - ankle_h
    print(f"leg drop     : {drop:.4f} m = {drop/MAX_REACH*100:.0f}% of max reach "
          f"({MAX_REACH:.4f} m)")
    if drop > MAX_REACH * 0.95:
        sys.exit(f"ERROR: --zhip {z_hip} is too high, the leg would lock out. "
                 f"Use at most {ankle_h + MAX_REACH*0.92:.3f}.")
    if drop < MAX_REACH * 0.45:
        print("WARNING: squatting very deep, the knee may exceed 120 degrees.")

    def stance_q(z_hip_now, latA=LAT_A, latB=LAT_B, fwdA=0.0, fwdB=0.0,
                 liftA=0.0, liftB=0.0):
        """Joint angles for both legs with the hip reference at z_hip_now."""
        hip_ref_w = to_model(0.0, 0.0, z_hip_now)          # hip midpoint (world)
        torso_w = hip_ref_w - BODY_REF_OFFSET               # torso origin (world)
        fA = to_model(latA, fwdA, ankle_h + liftA) - torso_w
        fB = to_model(latB, fwdB, ankle_h + liftB) - torso_w
        return (IK_leg_mesh(fA, left=True,  refine=args.refine_ik),
                IK_leg_mesh(fB, left=False, refine=args.refine_ik), torso_w)

    qA0, qB0, torso_w0 = stance_q(SQUAT_START_Z)
    data.qpos[:] = 0.0
    data.qpos[0:3] = torso_w0
    data.qpos[3] = 1.0
    data.qpos[qadr[0]:qadr[0] + 6] = qA0
    data.qpos[qadr[6]:qadr[6] + 6] = qB0
    mujoco.mj_forward(model, data)
    # settle the feet exactly onto the ground
    data.qpos[2] -= lowest_point_of_feet(model, data)
    mujoco.mj_forward(model, data)
    print(f"initial torso height: {data.qpos[2]:.4f} m")
    data.ctrl[ctrl_idx[:6]] = qA0
    data.ctrl[ctrl_idx[6:]] = qB0

    # --- planner ---
    footstep = np.array([[i * args.step_len, LAT_A if i % 2 == 0 else LAT_B, 0.0]
                         for i in range(N_STEPS)])
    Ks, Kx, F_prev, sysA, sysB, sysC = init_preview_control(DT_MPC, z_hip, N_PREVIEW)
    planner = TightStepPlanner(footstep, STEP_TIME, INITIAL_DELAY)
    planner.p_ref_y[:INITIAL_DELAY] = 0.0
    planner.p_ref_x[:INITIAL_DELAY] = 0.0
    # After the last step, force the ZMP reference back to double-support
    # (midpoint) instead of leaving it on the last single foot forever.
    STOP_TICK = INITIAL_DELAY + (N_STEPS - 1) * STEP_TIME
    planner.p_ref_x[STOP_TICK:] = footstep[-1, 0]
    planner.p_ref_y[STOP_TICK:] = 0.0

    log_t, log_cf, log_cl, log_pf, log_pl, log_bz = [], [], [], [], [], []
    log_dcm_l, log_dcm_f = [], []
    trace_rows = []   # raw trace: time, 12 joint commands, base/CoM state, contacts
    dcm_l = dcm_f = 0.0
    raw_dcm_l = raw_dcm_f = 0.0
    k_dcm = 0.0 if args.no_dcm else args.k_dcm
    ball_fired = False
    state_f = np.zeros(3)      # forward
    state_l = np.zeros(3)      # lateral
    sum_e_f = sum_e_l = 0.0

    foot_A = np.array([LAT_A, 0.0]); foot_B = np.array([LAT_B, 0.0])
    lift_A = lift_B = 0.0
    start_A = start_B = None
    tgt_A = foot_A.copy(); tgt_B = foot_B.copy()
    swing = None
    step_index = 0
    walking = False
    qA, qB = qA0.copy(), qB0.copy()
    ik_fail = 0

    # Uses MuJoCo's default free camera (no fixed camera forced).
    viewer = None
    if not args.headless:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(model, data)
        viewer.__enter__()

    # --- recording ---
    renderer = vid = None
    vid_kind = None
    next_frame = 0.0
    frame_dt = 1.0 / max(1, args.fps)
    if args.record:
        try:
            renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        except Exception as e:
            sys.exit(f"Renderer failed to open: {e}\n"
                     f"Resolution may exceed the XML's offwidth/offheight "
                     f"(currently 1920x1080).")
        vid_kind, vid = open_video_writer(args.record, args.fps)
        print(f"recording    : {args.record}  {args.width}x{args.height} @ {args.fps} fps")

    t_wall = time.perf_counter()
    exit_reason = "time limit reached"
    next_report = 0.0
    next_sync = 0.0
    try:
        while data.time < args.time:
            if viewer is not None and not viewer.is_running():
                exit_reason = "viewer window closed"
                break
            t = data.time

            data.xfrc_applied[torso_id, :] = 0.0
            pushing = bool(args.push) and PUSH_TIME < t < PUSH_TIME + PUSH_DUR

            if args.push and args.push_viz == "ball":
                # Real projectile instead of a force: speed is chosen so the
                # ball delivers the same impulse (v = F*dt / m_ball).
                if ball_qadr >= 0 and not ball_fired and t >= PUSH_TIME:
                    _, tip, d = push_arrow_geometry(data, torso_id,
                                                    args.push_axis, args.push)
                    v = abs(args.push) * PUSH_DUR / max(1e-6, ball_mass)
                    start = data.xpos[torso_id] - d * 0.60
                    data.qpos[ball_qadr:ball_qadr + 3] = start
                    data.qpos[ball_qadr + 3:ball_qadr + 7] = [1, 0, 0, 0]
                    data.qvel[ball_dofadr:ball_dofadr + 3] = d * v
                    data.qvel[ball_dofadr + 3:ball_dofadr + 6] = 0.0
                    ball_fired = True
                    print(f"t={t:.2f}s  BALL LAUNCHED: m={ball_mass*1000:.0f} g, "
                          f"v={v:.2f} m/s, impulse={abs(args.push)*PUSH_DUR:.2f} Ns")
            elif pushing:
                if args.push_axis == "lat":
                    data.xfrc_applied[torso_id, 0] = args.push      # model +X
                else:
                    data.xfrc_applied[torso_id, 1] = -args.push     # model -Y = forward

            if t < SETTLE_TIME:
                z_now = SQUAT_START_Z
            elif t < SETTLE_TIME + SQUAT_TIME:
                u = (t - SETTLE_TIME) / SQUAT_TIME
                z_now = SQUAT_START_Z - (SQUAT_START_Z - z_hip) * u
            else:
                z_now = z_hip
                if not walking:
                    walking = True
                    print(f"t={t:.2f}s  squat finished, ZMP planner started "
                          f"(first step in {INITIAL_DELAY*DT_MPC:.1f}s)")

            # ------------------------------------------- PREVIEW CONTROL
            if walking and (round(t / dt) % preview_every == 0):
                idx = np.minimum(step_index + np.arange(N_PREVIEW), 19999)
                p_l = float((sysC @ state_l).item())
                sum_e_l += p_l - planner.p_ref_y[min(step_index, 19999)]
                state_l = sysA @ state_l + sysB * (
                    -Ks * sum_e_l - float(np.dot(Kx, state_l))
                    - float(np.dot(F_prev, planner.p_ref_y[idx])))
                p_f = float((sysC @ state_f).item())
                sum_e_f += p_f - planner.p_ref_x[min(step_index, 19999)]
                state_f = sysA @ state_f + sysB * (
                    -Ks * sum_e_f - float(np.dot(Kx, state_f))
                    - float(np.dot(F_prev, planner.p_ref_x[idx])))

                if step_index >= STOP_TICK:
                    lift_A = lift_B = 0.0      # steps finished, both feet down
                    swing = None
                elif step_index >= INITIAL_DELAY:
                    k = step_index - INITIAL_DELAY
                    ph_i = k % STEP_TIME
                    tx, ty, is_A = planner.swing_target(step_index, state_f, state_l)

                    # ---------------- DCM foot-placement feedback ------
                    # Recomputed every control step, so the landing target
                    # updates even mid-swing.
                    if k_dcm > 0.0:
                        c_m, v_m = measured_com_state(model, data, torso_id)
                        dl, df, raw_dcm_l, raw_dcm_f = dcm_correction(
                            c_m, v_m,
                            (state_l[0], state_l[1]), (state_f[0], state_f[1]),
                            k_dcm, DCM_LIM_LAT, DCM_LIM_FWD)
                        # fade the correction to zero approaching touchdown
                        frac = ph_i / float(STEP_TIME)
                        fade = 1.0 if frac < DCM_FADE else \
                            max(0.0, (1.0 - frac) / (1.0 - DCM_FADE))
                        dcm_l, dcm_f = dl * fade, df * fade
                    else:
                        dcm_l = dcm_f = 0.0
                    ty += dcm_l
                    tx += dcm_f
                    # -------------------------------------------------------

                    if ph_i == 0:
                        start_A, start_B = foot_A.copy(), foot_B.copy()
                        swing = 'A' if is_A else 'B'
                    if is_A:
                        tgt_A = np.array([ty, tx])
                    else:
                        tgt_B = np.array([ty, tx])
                    ph = (ph_i / float(STEP_TIME)) * math.pi
                    s = (1 - math.cos(ph)) / 2.0
                    if swing == 'A' and start_A is not None:
                        foot_A = start_A + (tgt_A - start_A) * s
                        lift_A, lift_B = STEP_HEIGHT * math.sin(ph), 0.0
                    elif swing == 'B' and start_B is not None:
                        foot_B = start_B + (tgt_B - start_B) * s
                        lift_B, lift_A = STEP_HEIGHT * math.sin(ph), 0.0
                step_index += 1

            # -------------------------------------------------------- IK
            hip_ref_w = to_model(state_l[0], state_f[0], z_now)
            torso_w = hip_ref_w - BODY_REF_OFFSET
            fA_b = to_model(foot_A[0], foot_A[1], ankle_h + lift_A) - torso_w
            fB_b = to_model(foot_B[0], foot_B[1], ankle_h + lift_B) - torso_w
            try:
                qA = IK_leg_mesh(fA_b, left=True,  refine=args.refine_ik)
                qB = IK_leg_mesh(fB_b, left=False, refine=args.refine_ik)
                if not (np.all(np.isfinite(qA)) and np.all(np.isfinite(qB))):
                    raise FloatingPointError(f"IK NaN: A={qA} B={qB}")
            except Exception as e:
                ik_fail += 1
                print(f"t={t:.3f}s IK ERROR ({ik_fail}): {e}")
                if ik_fail >= 3:
                    exit_reason = "repeated IK failures"
                    print("Repeated IK failures -- stopping.")
                    break

            cmdA, cmdB = qA.copy(), qB.copy()
            if (USE_ANKLE_STABILIZER or args.ankle_fb) and walking:
                sgn = args.ankle_fb_sign
                R_b = quat2mat(data.qpos[3:7])
                # forward = model -Y, lateral = model +X
                pitch = math.asin(np.clip(R_b[1, 2], -1, 1))
                roll = math.asin(np.clip(R_b[0, 2], -1, 1))
                d_p = sgn * np.clip(K_ANK_P * pitch + K_ANK_D * data.qvel[3], -ANK_CLAMP, ANK_CLAMP)
                d_r = sgn * np.clip(K_ANK_P * roll + K_ANK_D * data.qvel[4], -ANK_CLAMP, ANK_CLAMP)
                cmdA[4] += d_p; cmdB[4] += d_p
                cmdA[5] -= d_r; cmdB[5] -= d_r

            data.ctrl[ctrl_idx[:6]] = np.clip(cmdA, Q_MIN, Q_MAX)
            data.ctrl[ctrl_idx[6:]] = np.clip(cmdB, Q_MIN, Q_MAX)

            mujoco.mj_step(model, data)

            if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
                exit_reason = "NaN (simulation diverged)"
                print(f"t={t:.3f}s SIMULATION DIVERGED (NaN).")
                break

            log_t.append(t)
            com = data.sensor('com_pos').data
            log_cf.append(HIP_Y - float(com[1]))     # forward
            log_cl.append(float(com[0]) - MID_X)     # lateral
            log_pf.append(state_f[0]); log_pl.append(state_l[0])
            log_bz.append(float(data.qpos[2]))
            log_dcm_l.append(dcm_l); log_dcm_f.append(dcm_f)

            # --- raw trace row: time, 12 joint commands, base/CoM state, contacts ---
            com_vel = data.sensor('com_vel').data
            base_quat = data.sensor('torso_quat').data
            touch_l = float(data.sensor('touch_left').data[0])
            touch_r = float(data.sensor('touch_right').data[0])
            cmd12 = data.ctrl[ctrl_idx].copy()      # 12 values, in JOINT_ORDER
            act12 = data.qpos[qadr].copy()          # actual joint angles
            row = {"time_s": t, "push_active": int(pushing)}
            for name, val in zip(JOINT_ORDER, cmd12):
                row[f"command_{name}_rad"] = float(val)
            for name, val in zip(JOINT_ORDER, act12):
                row[f"actual_{name}_rad"] = float(val)
            row["base_x_m"] = float(data.qpos[0])
            row["base_y_m"] = float(data.qpos[1])
            row["base_z_m"] = float(data.qpos[2])
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
            if t >= next_report:
                next_report += 1.0
                hip_meas = data.qpos[2] + BODY_REF_OFFSET[2]
                sag = (z_now - hip_meas) * 1000.0
                cf = (HIP_Y - float(data.sensor('com_pos').data[1])) - log_cf[0] if log_cf else 0.0
                nc = data.ncon if hasattr(data, "ncon") else -1
                wall = time.perf_counter() - t_wall
                rtf = (t / wall) if wall > 1e-6 else 0.0
                print(f"  t={t:4.1f}s  fwd={cf:+.3f}m  hip z={hip_meas:.4f}"
                      f" (sag {sag:+5.1f}mm)  swing={swing or '-'}"
                      f"  step={max(0,step_index-INITIAL_DELAY)//STEP_TIME}"
                      f"  contacts={nc}  speed={rtf:.2f}x"
                      f"  DCM err=({raw_dcm_l*1000:+5.1f},{raw_dcm_f*1000:+5.1f})mm"
                      f" corr=({dcm_l*1000:+5.1f},{dcm_f*1000:+5.1f})mm")

            if data.qpos[2] < 0.08:
                exit_reason = "robot fell (torso z < 0.08)"
                print(f"t={t:.2f}s  torso z={data.qpos[2]:.3f} -- robot fell.")
                break

            # --- video frame ---
            if renderer is not None and data.time >= next_frame:
                next_frame += frame_dt
                renderer.update_scene(data)
                if args.push_viz == "arrow" and arrow_alpha > 0:
                    draw_push(renderer.scene, data, torso_id,
                              args.push_axis, args.push, arrow_alpha, reset=False)
                vid = write_frame(vid_kind, vid, renderer.render())

            # viewer.sync() is expensive (~40 meshes per body), so it's
            # throttled to 50 Hz (VIEW_DT) instead of every 1ms step.
            # Push arrow: full opacity during the push, fades over 0.6 s after.
            arrow_alpha = 0.0
            if args.push and args.push_viz == "arrow":
                if pushing:
                    arrow_alpha = 1.0
                elif PUSH_TIME <= t < PUSH_TIME + PUSH_DUR + 0.6:
                    arrow_alpha = 1.0 - (t - PUSH_TIME - PUSH_DUR) / 0.6

            if viewer is not None and data.time >= next_sync:
                next_sync += VIEW_DT
                if args.push_viz == "arrow":
                    # call even at alpha=0, to clear the previous arrow
                    draw_push(viewer.user_scn, data, torso_id,
                              args.push_axis, args.push, arrow_alpha, reset=True)
                viewer.sync()
                # Real-time throttling: sleep to an absolute target time via
                # perf_counter (avoids drift from Windows' ~16ms time.time()).
                if args.speed > 0 and renderer is None:
                    target = t_wall + data.time / args.speed
                    now = time.perf_counter()
                    if target > now:
                        time.sleep(target - now)
    finally:
        if viewer is not None:
            viewer.__exit__(None, None, None)
        if renderer is not None:
            close_video(vid_kind, vid)
            renderer.close()
            print(f"video written: {args.record}")

    print("\n--- SUMMARY ---")
    print(f"exit reason  : {exit_reason}")
    print(f"sim duration : {log_t[-1] if log_t else 0:.2f} s")
    print(f"first step   : t={SETTLE_TIME+SQUAT_TIME+INITIAL_DELAY*DT_MPC:.2f}s"
          f"  (settle {SETTLE_TIME} + squat {SQUAT_TIME} + wait "
          f"{INITIAL_DELAY*DT_MPC:.1f})")
    print(f"IK failures  : {ik_fail}")
    if log_t:
        print(f"distance     : {log_cf[-1]-log_cf[0]:+.3f} m (forward)")
        print(f"lateral drift: {log_cl[-1]-log_cl[0]:+.3f} m")
        print(f"torso z      : start {log_bz[0]:.3f}  end {log_bz[-1]:.3f}  min {min(log_bz):.3f}")
        ok = min(log_bz) > 0.12 and log_t[-1] > args.time * 0.95
        print(f"RESULT       : {'STOOD / WALKED' if ok else 'FELL or ENDED EARLY'}")

    # ----------------------------------------------------------------------
    #  Raw trace + provenance (CSV + JSON): control/planner/IK/model
    #  untouched, logging only.
    # ----------------------------------------------------------------------
    if trace_rows:
        import numpy as _np_ver
        try:
            import scipy as _scipy_ver
            scipy_version = _scipy_ver.__version__
        except Exception:
            scipy_version = "not installed"
        base_name = f"main_walk_trace_push{args.push:g}n"
        csv_path = base_name + ".csv"
        json_path = base_name + ".metadata.json"
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0]))
            writer.writeheader()
            writer.writerows(trace_rows)
        ok_final = bool(log_t) and min(log_bz) > 0.12 and log_t[-1] > args.time * 0.95
        metadata = {
            "source_file": "main_walk_trace.py (logging-only copy of main_walk.py; "
                            "control/planner/IK/model unchanged)",
            "invocation": " ".join(sys.argv),
            "xml_path": args.xml,
            "push_force_n": args.push,
            "push_axis": args.push_axis,
            "push_time_s": PUSH_TIME,
            "push_duration_s": PUSH_DUR,
            "push_impulse_n_s": round(abs(args.push) * PUSH_DUR, 4),
            "dcm_feedback_enabled": not args.no_dcm,
            "k_dcm": 0.0 if args.no_dcm else args.k_dcm,
            "robot_mass_kg": robot_mass,
            "source_timestep_s": dt,
            "rows": len(trace_rows),
            "exit_reason": exit_reason,
            "end_time_s": log_t[-1] if log_t else 0.0,
            "end_base_height_m": log_bz[-1] if log_bz else None,
            "min_base_height_m": min(log_bz) if log_bz else None,
            "ik_failures": ik_fail,
            "result": "STOOD / WALKED" if ok_final else "FELL or ENDED EARLY",
            "environment": {
                "os": platform.platform(),
                "python": sys.version.split()[0],
                "numpy": _np_ver.__version__,
                "scipy": scipy_version,
                "mujoco": mujoco.__version__,
            },
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        print(f"raw trace    : {csv_path}")
        print(f"provenance   : {json_path}")

    if not args.no_plot and log_t:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
            ax[0].plot(log_t, log_pf, "--", label="Ref CoM")
            ax[0].plot(log_t, log_cf, lw=2, label="Actual CoM")
            ax[0].set_title("Sagital"); ax[0].set_ylabel("m"); ax[0].legend(); ax[0].grid(True)
            ax[1].plot(log_t, log_pl, "--", label="Ref CoM")
            ax[1].plot(log_t, log_cl, lw=2, label="Actual CoM")
            ax[1].set_title("Lateral"); ax[1].set_ylabel("m"); ax[1].legend(); ax[1].grid(True)
            ax[2].plot(log_t, log_bz, "k", lw=2)
            ax[2].set_title("Torso Height"); ax[2].set_ylabel("m"); ax[2].grid(True)
            ax[3].plot(log_t, np.array(log_dcm_l) * 1000, label="Lateral Correction")
            ax[3].plot(log_t, np.array(log_dcm_f) * 1000, label="Sagital Correction")
            ax[3].set_title("LQR reflex")
            ax[3].set_ylabel("mm"); ax[3].set_xlabel("t [s]")
            ax[3].legend(); ax[3].grid(True)
            if args.push:
                for a in ax:
                    a.axvline(PUSH_TIME, color="r", ls=":", lw=1.5)
                ax[0].text(PUSH_TIME, ax[0].get_ylim()[1], f" Push {args.push:g}N",
                           color="r", va="top", fontsize=9)
            plt.tight_layout(); plt.savefig("walk_trajectory.png", dpi=140)
            print("plot: walk_trajectory.png")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
