#!/usr/bin/env python3
"""
main_walk_terrain.py
====================
robot_walk.xml robotunu ENGEBELI (orman topragi benzeri) zeminde yurutur.
Klasik kontrol: ZMP preview control + Kajita IK (senin kinematic.py'n,
degismedi) + araziye uyum katmani (rough_gait.py).

Gerekli dosyalar (hepsi ayni klasorde):
    robot_walk.xml, assets/ (STL'ler), kinematic.py, kinematic_mesh.py,
    robot_utils.py, preview_control.py, main_walk.py,
    terrain.py, rough_gait.py, main_walk_terrain.py   <- son ucu yeni

ONERILEN SIRA  (her adim bir oncekini dogrular)
-----------------------------------------------
    python terrain.py                              # araziyi ciz (mujoco gerekmez)
    python rough_gait.py                           # kinematik kuru-kosu (mujoco gerekmez)
    python main_walk_terrain.py --rough 0          # DUZ zemin: eski yuruyusle ayni mi?
    python main_walk_terrain.py --rough 0.3
    python main_walk_terrain.py --rough 0.6        # varsayilan
    python main_walk_terrain.py --rough 1.0        # zor
    python main_walk_terrain.py --rough 0.6 --push 15      # arazide itme

KARSILASTIRMA DENEYLERI
-----------------------
    --blind          yukseklik haritasi yok; sadece dokunma sensoruyle uyum
    --no-slope       ayak egime uymaz (hep yatay)
    --no-touchdown   erken temas algilama kapali
    --dcm            DCM ayak yerlesim geri beslemesini ac (varsayilan KAPALI)
    --dcm --dcm-legacy   main_walk.py'deki eski DCM davranisi (inis aninda sifira soner)
    --seed N         baska bir arazi
    --dump d.npz     100 Hz teshis kaydi
    python run_tests.py      hepsini penceresiz kosturup tablo basar

OLCULEN SONUCLAR  (gercek mesh'ler, robot 2.99 kg, MuJoCo 4.0.1, 24 adim, plan 0.69 m)
-------------------------------------------------------------------------------------
    rough 0 / 0.3 / 0.6       : yurudu (0.68-0.70 m)
    rough 1.0, 6 farkli arazi : 6/6 yurudu (0.63-0.69 m)
    rough 1.5, 3 farkli arazi : 2/3 yurudu, ayaklar kayiyor (0.57-0.60 m)
    rough 2.0                 : dustu
    --blind (rough 1.0)       : yurudu ama 0.58 m; erken temas da kapatilirsa dustu
    --no-slope (rough 1.0)    : yurudu ama 0.56 m (egime uyum mesafeyi artiriyor)
    itme (0.1 s), duz zemin   : yanal +20 N / -11 N, ileri +-10 N dayandi;
                                yanal -14 N ve ileri 14 N dustu
    DCM geri beslemesi        : acikken de kapaliyken de AYNI itmelere dayandi,
                                arazide mesafeyi dusurdu -> varsayilan kapali.
    Neden ise yaramiyor: 1/omega = sqrt(z/g) = sqrt(0.185/9.81) = 0.137 s.
    Adim suresi 0.45 s = 3.3 zaman sabiti; DCM hatasi bir adimda e^3.3 = 27
    kat buyur. Ayak yere inene kadar is isten gecmis olur. Bu boyda bir
    robotta itmeye karsi ayak BILEGI (CoP) stratejisi ya da cok daha kisa
    adim gerekir. (--step-time 0.3 denendi: durusa gecerken dustu.)

GERCEK ROBOT ICIN NOT: burada arazi yuksekligi simulasyondan "bedava"
okunuyor. Gercekte bu bilgi bir derinlik kamerasi / LiDAR + yukseklik
haritalama (elevation mapping) ile gelir. --blind modu, o sensor yokken ne
olacagini gosterir.
"""

import argparse
import math
import os
import sys
import time

import numpy as np

try:
    import mujoco
except ImportError:
    sys.exit("mujoco kurulu degil:  pip install mujoco")

# main_walk.py'deki yardimcilar aynen kullaniliyor (kopyalamamak icin import)
from main_walk import (open_video_writer, write_frame, close_video, draw_push,
                       measured_com_state, lowest_point_of_feet,
                       measure_ankle_height, JOINT_ORDER, VIEW_DT)
from kinematic_mesh import BODY_REF_OFFSET, MAX_REACH, Q_MIN, Q_MAX, ankle_center_fk
from terrain import Terrain
from rough_gait import RoughGait, GaitParams, Footprint, MID_X, HIP_Y

XML_PATH = "robot_walk.xml"
SQUAT_EXTRA = 0.020        # baslangicta z_hip + bu kadar yuksekten coker
SQUAT_TIME = 1.2
SETTLE_TIME = 0.4
PUSH_TIME, PUSH_DUR = 6.0, 0.1


def measure_footprints(model, data):
    """Taban sinirlarini carpisma mesh'lerinden OLCER (q=0 durusunda).

    measure_ankle_height() hemen once cagrilmis olmali (qpos=0 durusu)."""
    out = {}
    mujoco.mj_forward(model, data)
    for side, key in (("left", "A"), ("right", "B")):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_ankle_roll")
        anc = data.xanchor[jid]
        pts = []
        for gi in range(model.ngeom):
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gi])
            if bname != f"{side}_foot":
                continue
            if model.geom_contype[gi] == 0 and model.geom_conaffinity[gi] == 0:
                continue
            if model.geom_type[gi] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = model.geom_dataid[gi]
            a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            gR = data.geom_xmat[gi].reshape(3, 3)
            pts.append(model.mesh_vert[a:a + n] @ gR.T + data.geom_xpos[gi])
        if not pts:
            out[key] = Footprint()
            continue
        P = np.vstack(pts)
        S = P[P[:, 2] < P[:, 2].min() + 0.003]          # tabana 3 mm'den yakin koseler
        lat = S[:, 0] - anc[0]
        fwd = -(S[:, 1] - anc[1])
        fp = Footprint(float(fwd.min()), float(fwd.max()), float(lat.min()), float(lat.max()))
        length, width = fp.fwd_max - fp.fwd_min, fp.lat_max - fp.lat_min
        if not (0.03 < length < 0.25 and 0.02 < width < 0.20):
            print(f"UYARI: {side} taban olcumu mantiksiz ({length*1000:.0f}x{width*1000:.0f} mm); "
                  f"varsayilan kullaniliyor.")
            fp = Footprint()
        out[key] = fp
    return out["A"], out["B"]


def check_terrain(model, data, terr):
    """Python'daki yukseklik haritasi ile MuJoCo'daki hfield AYNI mi? (isin atarak)"""
    worst = 0.0
    gid = np.zeros(1, dtype=np.int32)
    rng = np.random.default_rng(0)
    for _ in range(12):
        x = rng.uniform(-0.3, 0.3)
        y = HIP_Y - rng.uniform(0.45, 1.2)               # robottan uzak, engebeli bolge
        d = mujoco.mj_ray(model, data, np.array([x, y, 1.0]), np.array([0.0, 0.0, -1.0]),
                          None, 1, -1, gid)
        if d < 0:
            return None
        worst = max(worst, abs((1.0 - d) - float(terr.height(x, y))))
    return worst


def main():
    P0 = GaitParams()
    ap = argparse.ArgumentParser(description="Engebeli zeminde yuruyus")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--xml", default=XML_PATH, help="kaynak (duz zeminli) robot XML'i")
    # --- arazi
    ap.add_argument("--rough", type=float, default=0.6,
                    help="puruzluluk: 0 = duz, 0.6 = varsayilan, 1.0 = zor, 1.5 = cok zor")
    ap.add_argument("--seed", type=int, default=3, help="arazi tohumu")
    ap.add_argument("--soil-friction", type=float, default=1.0)
    ap.add_argument("--soil-solref", type=float, default=0.008,
                    help="zemin temas zaman sabiti [s]; buyuk = yumusak toprak")
    ap.add_argument("--skip-terrain-check", action="store_true")
    # --- yuruyus
    ap.add_argument("--zhip", type=float, default=P0.z_hip)
    ap.add_argument("--step-len", type=float, default=P0.step_len)
    ap.add_argument("--step-height", type=float, default=P0.step_height)
    ap.add_argument("--step-time", type=float, default=P0.step_time * P0.dt_mpc,
                    help="adim suresi [s]")
    ap.add_argument("--steps", type=int, default=P0.n_steps)
    ap.add_argument("--time", type=float, default=None, help="sim suresi [s] (bos = otomatik)")
    # --- uyum katmani anahtarlari
    ap.add_argument("--blind", action="store_true", help="yukseklik haritasini kullanma")
    ap.add_argument("--no-slope", action="store_true")
    ap.add_argument("--no-touchdown", action="store_true")
    ap.add_argument("--dcm", action="store_true",
                    help="DCM ayak yerlesim geri beslemesini AC (varsayilan kapali: "
                         "testlerde fayda saglamadi, bkz. dosya basi)")
    ap.add_argument("--no-dcm", action="store_true", help=argparse.SUPPRESS)  # eski ad, etkisiz
    ap.add_argument("--dcm-legacy", action="store_true")
    ap.add_argument("--k-dcm", type=float, default=1.0, help="--dcm acikken kazanc")
    ap.add_argument("--dcm-deadband", type=float, default=P0.dcm_deadband)
    ap.add_argument("--dcm-lipm-ref", action="store_true",
                    help="DCM hatasini LIPM durumuna gore olc (eski, hatali kiyas)")
    ap.add_argument("--no-zmp-follow", action="store_true",
                    help="ayak kaydirilinca ZMP referansini guncelleme (eski davranis)")
    ap.add_argument("--refine-ik", action="store_true")
    # --- itme
    ap.add_argument("--push", type=float, default=0.0, help=f"itme [N], t={PUSH_TIME}s'de {PUSH_DUR}s")
    ap.add_argument("--push-axis", default="lat", choices=["lat", "fwd"])
    ap.add_argument("--push-viz", default="arrow", choices=["arrow", "none"])
    # --- goruntu / kayit
    ap.add_argument("--record", metavar="DOSYA.mp4", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--dump", metavar="DOSYA.npz", default=None,
                    help="100 Hz teshis kaydi (numpy npz)")
    args = ap.parse_args()

    # ------------------------------------------------------------ arazi + model
    terr = Terrain(rough=args.rough, seed=args.seed, start_y=HIP_Y)
    st = terr.stats()
    src = os.path.abspath(args.xml)
    dst = os.path.join(os.path.dirname(src), "robot_terrain.xml")
    terr.write_xml(src, dst, friction=args.soil_friction, solref=args.soil_solref)
    model = mujoco.MjModel.from_xml_path(dst)
    terr.apply_to_model(model)          # viewer/renderer acilmadan ONCE
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    print(f"model        : {dst}  (kaynak: {args.xml})")
    print(f"arazi        : rough={args.rough} seed={args.seed}  {st['nrow']}x{st['ncol']} izgara  "
          f"yukseklik [{st['min_mm']:+.1f}, {st['max_mm']:+.1f}] mm  "
          f"egim ort {st['slope_mean_deg']:.1f} / %99 {st['slope_p99_deg']:.1f} derece")
    if args.soil_solref < 2 * dt:
        sys.exit(f"HATA: --soil-solref en az {2*dt} olmali (2 x timestep).")
    if model.nu != 12:
        sys.exit(f"HATA: {model.nu} aktuator var, 12 olmali.")

    qadr, ctrl_idx = [], []
    for n in JOINT_ORDER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if jid < 0 or aid < 0:
            sys.exit(f"HATA: '{n}' eklemi/aktuatoru yok.")
        qadr.append(model.jnt_qposadr[jid]); ctrl_idx.append(aid)
    qadr = np.array(qadr); ctrl_idx = np.array(ctrl_idx)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    if torso_id < 0:
        sys.exit("HATA: 'torso' govdesi yok.")
    robot_mass = float(model.body_subtreemass[torso_id])
    print(f"robot kutlesi: {robot_mass:.4f} kg")

    # ------------------------------------------------------------ olcumler
    ankle_h = measure_ankle_height(model, data, qadr)
    fp_A, fp_B = measure_footprints(model, data)
    print(f"ayak bilegi merkezi yerden: {ankle_h*1000:.1f} mm")
    for nm, fp in (("A", fp_A), ("B", fp_B)):
        print(f"taban {nm}      : ileri [{fp.fwd_min*1000:+.0f}, {fp.fwd_max*1000:+.0f}] mm  "
              f"yanal [{fp.lat_min*1000:+.0f}, {fp.lat_max*1000:+.0f}] mm")

    step_ticks = max(20, int(round(args.step_time / P0.dt_mpc)))
    p = GaitParams(
        z_hip=args.zhip, step_len=args.step_len, step_height=args.step_height,
        step_time=step_ticks, n_steps=args.steps,
        use_map=not args.blind, use_slope=not (args.no_slope or args.blind),
        use_touchdown=not args.no_touchdown,
        touch_force=0.25 * robot_mass * 9.81,
        k_dcm=args.k_dcm if (args.dcm and not args.no_dcm) else 0.0,
        dcm_hold=not args.dcm_legacy, refine_ik=args.refine_ik,
        dcm_deadband=args.dcm_deadband, zmp_follow=not args.no_zmp_follow)
    drop = p.z_hip - ankle_h
    print(f"bacak dususu : {drop:.4f} m = erisimin %{drop/MAX_REACH*100:.0f}'i")
    if drop > MAX_REACH * 0.90:
        sys.exit(f"HATA: --zhip fazla yuksek. Arazide en fazla {ankle_h + MAX_REACH*0.85:.3f} kullan.")
    preview_every = max(1, int(round(p.dt_mpc / dt)))
    # Komut edilen durusun GERCEK kutle merkezi (ayri bir MjData uzerinde, fizik yok)
    d_plan = mujoco.MjData(model)

    def com_model(torso_w, qA_, qB_):
        d_plan.qpos[:] = 0.0
        d_plan.qpos[0:3] = torso_w
        d_plan.qpos[3] = 1.0
        d_plan.qpos[qadr[:6]] = qA_
        d_plan.qpos[qadr[6:]] = qB_
        mujoco.mj_kinematics(model, d_plan)
        mujoco.mj_comPos(model, d_plan)
        return d_plan.subtree_com[torso_id].copy()

    gait = RoughGait(p, terr, ankle_h, fp_A, fp_B,
                     com_model=None if args.dcm_lipm_ref else com_model)
    walk_start = SETTLE_TIME + SQUAT_TIME
    t_end = args.time if args.time is not None else \
        walk_start + (gait.stop_tick + 250) * p.dt_mpc

    # ------------------------------------------------------------ baslangic durusu
    z_start = p.z_hip + SQUAT_EXTRA
    qA0, qB0, torso_w0 = gait.joint_targets(z_start)
    data.qpos[:] = 0.0
    data.qpos[0:3] = torso_w0
    data.qpos[3] = 1.0
    data.qpos[qadr[:6]] = qA0
    data.qpos[qadr[6:]] = qB0
    mujoco.mj_forward(model, data)
    data.qpos[2] -= lowest_point_of_feet(model, data)       # baslangic bolgesi z=0
    mujoco.mj_forward(model, data)
    data.ctrl[ctrl_idx[:6]] = qA0
    data.ctrl[ctrl_idx[6:]] = qB0

    if not args.skip_terrain_check and args.rough > 0:
        w = check_terrain(model, data, terr)
        if w is None:
            print("UYARI: arazi kontrol isini zemini bulamadi (kontrol atlandi).")
        elif w > 0.002:
            sys.exit(f"HATA: Python yukseklik haritasi ile MuJoCo hfield uyusmuyor "
                     f"(fark {w*1000:.1f} mm). Kontrolcu yanlis yukseklik gorur.\n"
                     f"Bu ciktiyi bana gonder; gecici olarak --blind ile calistirabilirsin "
                     f"(ya da --skip-terrain-check).")
        else:
            print(f"arazi kontrol: harita = hfield (max fark {w*1000:.2f} mm)  OK")

    # ------------------------------------------------------------ goruntu
    viewer = None
    if not args.headless:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(model, data)
        viewer.__enter__()
    renderer = vid = vid_kind = None
    next_frame = 0.0
    frame_dt = 1.0 / max(1, args.fps)
    if args.record:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        vid_kind, vid = open_video_writer(args.record, args.fps)
        rec_cam = mujoco.MjvCamera()          # sadece KAYIT icin; pencere serbest kamerada
        rec_cam.distance, rec_cam.azimuth, rec_cam.elevation = 0.75, 130.0, -18.0

    D = []
    tilt_max = [0.0, 0.0]
    DUMP_COLS = ("t swing u touchA touchB com_l com_f vel_l vel_f com_z plan_l plan_vl plan_f plan_vf "
                 "raw_l raw_f corr_l corr_f A_lat A_fwd A_g A_lift B_lat B_fwd B_g B_lift "
                 "A_ank_l A_ank_f A_ank_z B_ank_l B_ank_f B_ank_z roll pitch g_ref reach td").split()
    jidA = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_ankle_roll")
    jidB = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_ankle_roll")

    def R_roll(d):      # govdenin yanal (model X) yatisi [rad]; + = model +X'e dogru
        return math.asin(float(np.clip(d.xmat[torso_id].reshape(3, 3)[0, 2], -1, 1)))

    def R_pitch(d):     # govdenin ileri (model -Y) yatisi [rad]; + = one dogru
        return -math.asin(float(np.clip(d.xmat[torso_id].reshape(3, 3)[1, 2], -1, 1)))

    L = {k: [] for k in ("t", "cf", "cl", "pf", "pl", "bz", "dl", "df", "gA", "gB", "gref", "td")}
    walking = False
    arrow_alpha = 0.0
    ik_fail = 0
    qA, qB = qA0.copy(), qB0.copy()
    exit_reason = "sure doldu"
    next_report = next_sync = 0.0
    cf0 = None
    t_wall = time.perf_counter()
    try:
        while data.time < t_end:
            if viewer is not None and not viewer.is_running():
                exit_reason = "pencere kapatildi"
                break
            t = data.time

            data.xfrc_applied[torso_id, :] = 0.0
            pushing = bool(args.push) and PUSH_TIME < t < PUSH_TIME + PUSH_DUR
            if pushing:
                if args.push_axis == "lat":
                    data.xfrc_applied[torso_id, 0] = args.push
                else:
                    data.xfrc_applied[torso_id, 1] = -args.push

            if t < SETTLE_TIME:
                z_now = z_start
            elif t < walk_start:
                z_now = z_start - SQUAT_EXTRA * (t - SETTLE_TIME) / SQUAT_TIME
            else:
                z_now = p.z_hip
                if not walking:
                    walking = True
                    print(f"t={t:.2f}s  cokme bitti, yuruyus planlayici basladi")

            # ---------------------------------------- 100 Hz: plan + geri besleme
            if walking and (round(t / dt) % preview_every == 0):
                c_m, v_m = measured_com_state(model, data, torso_id)
                tl = float(data.sensor('touch_left').data[0])
                tr = float(data.sensor('touch_right').data[0])
                # iki bilek arasindaki yukseklik farki: enkoder + IMU'dan (FK)
                R_w = data.xmat[torso_id].reshape(3, 3)
                pA_b, _ = ankle_center_fk(data.qpos[qadr[:6]], True)
                pB_b, _ = ankle_center_fk(data.qpos[qadr[6:]], False)
                ank_dz = float((R_w @ (pA_b - pB_b))[2])
                gait.tick(c_m, v_m, tl, tr, ank_dz)
                if args.dump:
                    k_ = gait.step_index - 1 - p.initial_delay
                    D.append([t, {"A": 1, "B": 2}.get(gait.swing, 0),
                              ((k_ % p.step_time) + 1) / p.step_time if k_ >= 0 else 0.0,
                              tl, tr, c_m[0] - MID_X, HIP_Y - c_m[1], v_m[0], -v_m[1], c_m[2],
                              gait.state_l[0], gait.state_l[1], gait.state_f[0], gait.state_f[1],
                              gait.raw[0], gait.raw[1], gait.corr[0], gait.corr[1],
                              gait.feet["A"].pos[0], gait.feet["A"].pos[1],
                              gait.feet["A"].g, gait.feet["A"].lift,
                              gait.feet["B"].pos[0], gait.feet["B"].pos[1],
                              gait.feet["B"].g, gait.feet["B"].lift,
                              data.xanchor[jidA][0] - MID_X, HIP_Y - data.xanchor[jidA][1], data.xanchor[jidA][2],
                              data.xanchor[jidB][0] - MID_X, HIP_Y - data.xanchor[jidB][1], data.xanchor[jidB][2],
                              R_roll(data), R_pitch(data), gait.g_ref, gait.reach_drop,
                              float(gait.td_lock)])

            # ---------------------------------------- IK
            try:
                qA_n, qB_n, _ = gait.joint_targets(z_now)
                if not (np.all(np.isfinite(qA_n)) and np.all(np.isfinite(qB_n))):
                    raise FloatingPointError(f"IK NaN: A={qA_n} B={qB_n}")
                qA, qB = qA_n, qB_n
            except Exception as e:
                ik_fail += 1
                print(f"t={t:.3f}s IK HATASI ({ik_fail}): {e}")
                if ik_fail >= 3:
                    exit_reason = "ust uste IK hatasi"
                    break
            data.ctrl[ctrl_idx[:6]] = np.clip(qA, Q_MIN, Q_MAX)
            data.ctrl[ctrl_idx[6:]] = np.clip(qB, Q_MIN, Q_MAX)

            mujoco.mj_step(model, data)

            if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
                exit_reason = "NaN (simulasyon patladi)"
                break

            com = data.sensor('com_pos').data
            if cf0 is None:
                cf0 = HIP_Y - float(com[1])
            L["t"].append(t)
            L["cf"].append(HIP_Y - float(com[1])); L["cl"].append(float(com[0]) - MID_X)
            L["pf"].append(gait.state_f[0]); L["pl"].append(gait.state_l[0])
            L["bz"].append(float(data.qpos[2]))
            L["dl"].append(gait.corr[0]); L["df"].append(gait.corr[1])
            L["gA"].append(gait.feet["A"].g + gait.feet["A"].lift)
            L["gB"].append(gait.feet["B"].g + gait.feet["B"].lift)
            L["gref"].append(gait.g_ref); L["td"].append(gait.early_td_count)
            if walking:
                tilt_max[0] = max(tilt_max[0], abs(R_roll(data)))
                tilt_max[1] = max(tilt_max[1], abs(R_pitch(data)))

            if t >= next_report:
                next_report += 1.0
                hip_meas = data.qpos[2] + BODY_REF_OFFSET[2]
                sag = (z_now + gait.g_ref - gait.reach_drop - hip_meas) * 1000.0
                wall = time.perf_counter() - t_wall
                print(f"  t={t:4.1f}s  ileri={L['cf'][-1]-cf0:+.3f}m  sarkma={sag:+5.1f}mm"
                      f"  salinim={gait.swing or '-'}"
                      f"  adim={max(0, gait.step_index - p.initial_delay)//p.step_time}"
                      f"  zemin A/B={gait.feet['A'].g*1000:+5.1f}/{gait.feet['B'].g*1000:+5.1f}mm"
                      f"  erken temas={gait.early_td_count}"
                      f"  DCM hata=({gait.raw[0]*1000:+5.1f},{gait.raw[1]*1000:+5.1f})"
                      f" duz=({gait.corr[0]*1000:+5.1f},{gait.corr[1]*1000:+5.1f})mm"
                      f"  temas={data.ncon}  hiz={t/max(wall,1e-6):.2f}x")

            if data.qpos[2] < 0.08 + gait.g_ref:
                exit_reason = "robot dustu"
                print(f"t={t:.2f}s  govde z={data.qpos[2]:.3f} -- robot dustu.")
                break

            arrow_alpha = 0.0
            if args.push and args.push_viz == "arrow":
                if pushing:
                    arrow_alpha = 1.0
                elif PUSH_TIME <= t < PUSH_TIME + PUSH_DUR + 0.6:
                    arrow_alpha = max(0.0, 1.0 - (t - PUSH_TIME - PUSH_DUR) / 0.6)

            if renderer is not None and data.time >= next_frame:
                next_frame += frame_dt
                rec_cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.10]
                renderer.update_scene(data, rec_cam)
                if arrow_alpha > 0:
                    draw_push(renderer.scene, data, torso_id, args.push_axis, args.push,
                              arrow_alpha, reset=False)
                vid = write_frame(vid_kind, vid, renderer.render())

            if viewer is not None and data.time >= next_sync:
                next_sync += VIEW_DT
                if args.push_viz == "arrow":
                    draw_push(viewer.user_scn, data, torso_id, args.push_axis, args.push,
                              arrow_alpha, reset=True)
                viewer.sync()
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
            print(f"video yazildi: {args.record}")

    if args.dump and D:
        np.savez(args.dump, data=np.array(D), cols=np.array(DUMP_COLS))
        print(f"teshis kaydi : {args.dump}")

    # ------------------------------------------------------------ ozet
    print("\n--- OZET ---")
    print(f"cikis nedeni : {exit_reason}")
    print(f"sim suresi   : {L['t'][-1] if L['t'] else 0:.2f} s / {t_end:.2f} s")
    print(f"IK hatasi    : {ik_fail}     erken temas: {gait.early_td_count}")
    if L["t"]:
        print(f"kat edilen   : {L['cf'][-1]-L['cf'][0]:+.3f} m  (plan {gait.footstep[-1,0]:.3f} m)")
        print(f"yanal kayma  : {L['cl'][-1]-L['cl'][0]:+.3f} m")
        print(f"govde egimi  : {math.degrees(tilt_max[0]):.1f} / {math.degrees(tilt_max[1]):.1f} derece")
        ok = exit_reason == "sure doldu"
        print(f"SONUC        : {'YURUDU, AYAKTA' if ok else 'DUSTU veya ERKEN BITTI'}")

    if not args.no_plot and L["t"]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            tt = L["t"]
            fig, ax = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
            ax[0].plot(tt, L["pf"], "--", label="plan"); ax[0].plot(tt, np.array(L["cf"]) - L["cf"][0], lw=2, label="govde (olculen)")
            ax[0].set_title("Ileri (sagital)"); ax[0].set_ylabel("m")
            ax[1].plot(tt, L["pl"], "--", label="plan"); ax[1].plot(tt, np.array(L["cl"]) - L["cl"][0], lw=2, label="govde (olculen)")
            ax[1].set_title("Yanal"); ax[1].set_ylabel("m")
            ax[2].plot(tt, L["bz"], "k", lw=2); ax[2].set_title("Govde yuksekligi"); ax[2].set_ylabel("m")
            ax[3].plot(tt, np.array(L["gA"]) * 1000, label="ayak A taban z")
            ax[3].plot(tt, np.array(L["gB"]) * 1000, label="ayak B taban z")
            ax[3].plot(tt, np.array(L["gref"]) * 1000, "k--", label="zemin referansi")
            ax[3].set_title("Ayak tabani yuksekligi (arazi + kaldirma)"); ax[3].set_ylabel("mm")
            ax[4].plot(tt, np.array(L["dl"]) * 1000, label="yanal")
            ax[4].plot(tt, np.array(L["df"]) * 1000, label="ileri")
            ax[4].set_title("DCM ayak yerlesim duzeltmesi"); ax[4].set_ylabel("mm")
            ax[4].set_xlabel("t [s]")
            for a in ax:
                a.grid(True); a.legend(loc="upper left", fontsize=8) if a.get_legend_handles_labels()[0] else None
                if args.push:
                    a.axvline(PUSH_TIME, color="r", ls=":", lw=1.5)
            plt.tight_layout(); plt.savefig("walk_terrain.png", dpi=130)
            print("grafik: walk_terrain.png")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
