#!/usr/bin/env python3
"""
main_walk.py
============
robot_walk.xml (STL/mesh robot) icin ZMP preview-control + Kajita ters
kinematik yuruyus kontrolcusu. main.py ile ayni mantik; bu robotun gercek
olculerine ve cercevesine gore duzeltilmis.

    python fix_robot_xml.py            # robot_final.xml -> robot_walk.xml (bir kez)
    python kinematic_mesh.py           # IK oz-testi (mujoco gerekmez)
    python main_walk.py                # gorsel simulasyon (govdeyi izleyen kamera)
    python main_walk.py --headless     # penceresiz, log + grafik

ITMEYI GORSELLESTIRME
---------------------
    python main_walk.py --push 15                      # kirmizi ok (varsayilan)
    python main_walk.py --push 15 --push-viz ball      # gercek top firlatilir
    python main_walk.py --push 15 --push-viz none      # gorsel yok
    python main_walk.py --push 15 --push-axis fwd      # ileri dogru it
    python main_walk.py --push 15 --no-dcm             # ayak yerlesimi tepki vermesin

arrow : xfrc_applied ile kuvvet uygulanir, govdeye kirmizi bir ok cizilir.
        Ok itme suresince tam parlak, sonra 0.6 s icinde soner (0.1 s'lik
        itme tek basina gozle gorulmez). Videoya da islenir.
ball  : itme yerine GERCEK bir mermi. robot_walk.xml'deki 'push_ball'
        sahnenin uzaginda park eder, t=PUSH_TIME'da robota dogru firlatilir.
        Hizi ayni itkiyi verecek sekilde secilir: v = F*dt / m_top.
        Ornek: 15 N x 0.1 s = 1.5 Ns -> 196 g top, 7.65 m/s.

KAMERA
------
Kod kameraya HIC KARISMAZ. Pencere MuJoCo'nun varsayilan serbest kamerasiyla
acilir; sol tik surukle = dondur, sag tik surukle = kaydir, tekerlek = zoom.
Robotu takip ettirmek istersen pencerede robota CIFT TIKLA (odak noktasi
oraya kilitlenir), Ctrl+sag tik ile de MuJoCo'nun kendi takip modu acilir.

KAYIT
-----
    python main_walk.py --record yuruyus.mp4
    python main_walk.py --record y.mp4 --headless --width 1920 --height 1080 --fps 60

Kayit icin:  pip install imageio imageio-ffmpeg   (alternatif: opencv-python)
--record verildiginde gercek-zamanli frenleme kapatilir; video zaten dogru
kare hizinda yazilir, sim olabildigince hizli kosar.

KINEMATIK
---------
Ters kinematik SENIN kinematic.py'ndeki Kajita IK_leg'idir; degistirilmedi.
kinematic_mesh.py sadece dogru parametreleri hazirlar:

    Body_p  = govde orijini + (-0.005434, 0.067676, -0.050196)   (kalca merkezi hizasi)
    D       = -+0.075000 m                                        (tam simetrik)
    A_leg   = 0.086407 m      B_leg = 0.088407 m
    Foot_p  = AYAK BILEGI EKSEN KESISIMI (govde orijini degil)
    Foot_R  = zincir donusu (taban yatay istendiginde birim matris)
    cerceve = Rz(+90);  robotun ONU model -Y, YANAL +X

Dogrulama: calisma uzayinda 130 hedef, konum hatasi ort 0.40 mm / max 0.41 mm,
yonelim hatasi 0.0000 derece. Kalan 0.4 mm CAD toleransidir (kalca eksenleri
0.475 mm icinde kesisiyor).

main.py'YE GORE DUZELTILEN KONTROL HATALARI
-------------------------------------------
 A. dt UYUMSUZLUGU. preview kontrolcu dt_mpc=0.01 icin tasarlanmis ama
    "sim_counter % 3 == 0" ile 3 ms'de bir ilerletiliyordu: LIPM durumu
    gercek zamanin 3.3 katı hizinda akiyor. -> tam DT_MPC'de bir.

 B. "except: pass". model_final.xml'de 11 aktuator oldugu icin
    data.ctrl[6:12] her karede ValueError atiyor ve yutuluyordu; robot
    hicbir zaman komut almiyordu. -> her hata loglanir, 3 hatada durur.

 C. z_c ULASILAMAZ. z_c=0.35 / squat 0.40 komut ediliyordu; kalca
    merkezinden ayak bilegi merkezine max erisim 0.1748 m, kalca merkezi
    de govde orijininin 0.0502 m altinda. -> Z_HIP dogrudan kalca
    merkezinin yer uzeri yuksekligidir, tabanin yeri olculerek bulunur.

 D. "torso" govdesi yoktu; mj_name2id -1 donuyor, xfrc_applied[-1] itmeyi
    sessizce son govdeye (bir ayaga) uyguluyordu.

 E. Cerceve. Planlayici ciktilari (ileri, yanal) model eksenlerine
    ceviriliyor: model_X = MID_X + yanal,  model_Y = HIP_Y - ileri.
"""

import argparse
import math
import sys
import time

# Windows'ta time.sleep() varsayilan olarak ~15.6 ms cozunurlukle uyur; bu
# gercek-zamanli akisi bozar. Zamanlayici cozunurlugunu 1 ms'e cek.
try:
    import ctypes
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

import numpy as np

try:
    import mujoco
except ImportError:
    sys.exit("mujoco kurulu degil:  pip install mujoco")

from robot_utils import DynamicStepPlanner
from preview_control import init_preview_control
from kinematic_mesh import (IK_leg_mesh, ankle_center_fk, R_FOOT_LEVEL,
                            BODY_REF_OFFSET, MAX_REACH, MIRROR_X,
                            HIP_CENTER_L, HIP_CENTER_R, Q_MIN, Q_MAX)

# ===========================================================================
#  PARAMETRELER
# ===========================================================================
XML_PATH = "robot_walk.xml"

# --- duruş -----------------------------------------------------------------
# Z_HIP = KALCA MERKEZININ yer uzeri yuksekligi (LIPM'in z_c'si de budur).
# Bacak dususu = Z_HIP - (ayak bilegi merkezi yuksekligi ~0.0497).
# Max erisim 0.174814 m; %85'in uzerine cikma, diz kilitlenir.
Z_HIP = 0.195          # -> dusus ~0.145 m = erisimin %83'u, diz ~68 derece
SQUAT_START_Z = 0.215  # baslangicta buradan Z_HIP'e coker (%94 -> %83)
SQUAT_TIME = 1.2       # cokme suresi [s]
SETTLE_TIME = 0.4      # cokmeden once yerinde otur (temas otursun)

MID_X = MIRROR_X       # govdenin simetri duzlemi (model X)
HIP_Y = float(BODY_REF_OFFSET[1])     # kalca merkezlerinin model Y'si
# Ayaklarin orta duzleme yanal uzakligi.
# 0.075 = ayaklar TAM KALCALARIN ALTINDA (kalca yari genisligi de 0.075).
# Ped yari genisligi 0.042 m; 0.075'te aralarinda 66 mm bosluk kalir.
# Ilk surumdeki 0.055 sadece 26 mm birakiyordu ve kalca roll ayaklari iceri
# yatirdigi icin ayaklar birbirine giriyordu.
HALF_STANCE = 0.075

# Bacak A: model X < 0  (ihracattaki "left_*" eklemleri, IK'da left=True)
# Bacak B: model X > 0  (ihracattaki "right_*" eklemleri, IK'da left=False)
# D_LEFT = -0.075 oldugu icin bacak A yuruyus cercevesinde -y tarafindadir.
LAT_A = -HALF_STANCE
LAT_B = +HALF_STANCE

# --- yuruyus ---------------------------------------------------------------
DT_MPC = 0.01
N_PREVIEW = 160        # 1.6 s ileri gorus
STEP_TIME = 90         # 0.45 s / adim  (0.30 s -> CoM yanal salinimi 27 mm,
                       #                 0.45 s -> 38 mm; ayak ofseti 55 mm)
INITIAL_DELAY = 60     # 0.6 s yerinde denge (ilk adim ~t=2.2 s)
STEP_LEN = 0.04
STEP_HEIGHT = 0.022
N_STEPS = 20
MAX_SIM_TIME = 14.0    # 20 adim ~9.2 s'de biter, +2 s cift destekte otursun

#itme gerçek deger
PUSH_TIME, PUSH_DUR, PUSH_FORCE = 6.0, 0.1, 25.0
PUSH_AXIS = "lat"      # "lat" = yanal (model +X), "fwd" = ileri (model -Y)

# --- DCM (capture point) ayak yerlesim geri beslemesi ----------------------
# Eski kodda push recovery YOKTU: robot_utils.swing_target'a preview
# kontrolcunun kendi durumu gecildigi icin itmeye hicbir tepki verilmiyordu.
# Burada OLCULEN CoM'dan DCM hatasi cikarilip salinim ayaginin inecegi
# noktaya ekleniyor -- yani her kontrol adiminda ayak yeniden planlaniyor.
USE_DCM_FEEDBACK = True
K_DCM = 1.0            # 1.0 = DCM hatasini tam telafi et (teorik deger)
DCM_LIM_LAT = 0.045    # yanal duzeltme siniri [m]
DCM_LIM_FWD = 0.060    # ileri duzeltme siniri [m]
DCM_FADE = 0.80        # salinimin son %20'sinde duzeltmeyi sifira indir
                       # (inis anindaki titremeyi onler)

VIEW_DT = 0.02         # goruntuleyici tazeleme araligi [s] (50 Hz)

# --- govde egimi -> ayak bilegi geri beslemesi (main.py'de yoktu) ----------
# VARSAYILAN KAPALI. Bunu bu ortamda fizikle dogrulayamadim; isareti yanlissa
# denge saglamak yerine BOZAR. Once ham haliyle dene; salinim/dusme varsa
# --ankle-fb ile ac, kotulesiyorsa --ankle-fb-sign -1 ile isaretini cevir.
USE_ANKLE_STABILIZER = False
K_ANK_P, K_ANK_D, ANK_CLAMP = 0.60, 0.06, 0.30

JOINT_ORDER = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
               "left_ankle_pitch", "left_ankle_roll",
               "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee",
               "right_ankle_pitch", "right_ankle_roll"]


# ===========================================================================
#  VIDEO KAYDI
# ===========================================================================
def open_video_writer(path, fps):
    """imageio (ffmpeg) varsa onu, yoksa OpenCV'yi kullanir."""
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
                "Video yazmak icin bir kutuphane gerekli:\n"
                "    pip install imageio imageio-ffmpeg\n"
                "(alternatif: pip install opencv-python)\n"
                f"imageio hatasi : {e1}\n"
                f"opencv hatasi  : {e2}")


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
#  CERCEVE DONUSUMU
# ===========================================================================
def to_model(lat, fwd, up):
    """Yuruyus cercevesi (yanal, ileri, yer-uzeri) -> DUNYA, model eksenleri.

    Robotun onu model -Y, yanali model +X.
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
    """robot_utils.DynamicStepPlanner, ic "push recovery" terimi KAPALI.

    NEDEN KAPALI
    ------------
    Orijinal swing_target su duzeltmeyi ekliyor:

        delta = k_p * (com - zmp_ref) + k_v * com_v

    ama kendisine gecilen `state_x`/`state_y`, preview kontrolcunun KENDI
    LIPM durumudur -- ileri-besleme olarak entegre edilen bir model. Fizikten
    (data.qpos / data.qvel) hicbir sey okumaz. Sonuc:

      * Robot disaridan itildiginde bu iki durum DEGISMEZ, dolayisiyla delta
        da degismez: AYAK YERLESIMI ITMEYE HIC TEPKI VERMEZ.
      * Ustelik (com_planned - zmp_ref) tasarim geregi salinan bir sinyaldir;
        yani delta her adima sistematik, geri-beslemesiz bir kayma ekler.
        Yanal surukleniyi buyuten sebeplerden biri buydu.

    Gercek geri besleme artik main() icinde OLCULEN CoM'dan DCM (capture
    point) ile hesaplanip ayak hedefine ekleniyor. Burasi sadece nominal
    ayak izini dondurur.
    """

    def swing_target(self, step_index, state_x, state_y):
        if step_index < self.initial_delay:
            return 0.0, 0.0, False
        k = step_index - self.initial_delay
        t = min(k // self.step_time + 1, self.total_steps - 1)
        return (float(self.footsteps[t, 0]), float(self.footsteps[t, 1]),
                (t % 2 == 0))


# ===========================================================================
#  ITME GORSELLESTIRME
# ===========================================================================
def add_arrow(scn, p_from, p_to, width, rgba):
    """Sahneye kullanici geom'u olarak bir ok ekler.

    Hem passive viewer'in user_scn'i hem de Renderer'in scene'i icin calisir.
    mjv_connector (MuJoCo 3.x) yoksa mjv_makeConnector'a duser.
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
    """Itme okunun baslangic/bitis noktalari (dunya, model eksenleri)."""
    p = data.xpos[torso_id].copy()
    d = np.array([1.0, 0.0, 0.0]) if axis == "lat" else np.array([0.0, -1.0, 0.0])
    d = d * (1.0 if force >= 0 else -1.0)
    L = 0.10 + 0.010 * min(abs(force), 40.0)     # kuvvetle orantili uzunluk
    return p - d * L, p + d * 0.05, d


def draw_push(scn, data, torso_id, axis, force, alpha, reset=True):
    """Itme okunu ve carpma noktasindaki kureyi cizer.

    reset=True  -> viewer.user_scn icin: sahne SADECE kullanici geom'lari
                   tutar, her karede sifirlanmali (yoksa ok ekranda kalir).
    reset=False -> renderer.scene icin: update_scene() robotun tum geom'larini
                   zaten doldurdu; ngeom'u sifirlamak ROBOTU SILER. Sadece
                   sonuna ekle.
    """
    if scn is None:
        return
    if reset:
        scn.ngeom = 0
    if alpha <= 0.0:
        return
    a, b, d = push_arrow_geometry(data, torso_id, axis, force)
    add_arrow(scn, a, b, 0.012, [1.0, 0.15, 0.1, float(alpha)])
    # carpma noktasinda kucuk bir kure
    if scn.ngeom < scn.maxgeom:
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.018, 0.0, 0.0]), b, np.eye(3).flatten(),
                            np.array([1.0, 0.6, 0.1, float(alpha)], dtype=np.float32))
        scn.ngeom += 1


# ===========================================================================
#  OLCULEN CoM DURUMU  (gercek geri besleme buradan gelir)
# ===========================================================================
def measured_com_state(model, data, torso_id):
    """Robotun GERCEK kutle merkezi konumu ve hizi (dunya, model eksenleri).

    DIKKAT: subtree_com[0] (world alt agaci) TUM modeli kapsar -- sahneye
    firlatilan top da dahil. Bu, DCM geri beslemesini bozar. Bu yuzden
    TORSO alt agacini kullaniyoruz: torso + iki bacak, baska hicbir sey.
    """
    mujoco.mj_subtreeVel(model, data)
    return data.subtree_com[torso_id].copy(), data.subtree_linvel[torso_id].copy()


def dcm_correction(c, v, plan_lat, plan_fwd, k, lim_lat, lim_fwd):
    """DCM (capture point) hatasindan ayak yerlesim duzeltmesi.

        xi = c + c_dot / omega,      omega = sqrt(g / z_com)

    xi, "su an ayagi buraya koyarsan durursun" noktasidir. Olculen xi ile
    planlanan xi arasindaki fark, ayagi ne kadar oteye atmamiz gerektigini
    dogrudan verir -- itme, surtunme farki, modelleme hatasi, hepsi buraya
    yansir.

    plan_lat / plan_fwd : preview kontrolcunun (konum, hiz) ikilileri
    Doner: (d_lat, d_fwd) yuruyus cercevesinde metre
    """
    z = max(0.05, float(c[2]))
    omega = math.sqrt(9.81 / z)

    # model -> yuruyus cercevesi
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
#  ZEMIN / DURUŞ OLCUMU
# ===========================================================================
def lowest_point_of_feet(model, data):
    """Ayak carpisma mesh'lerinin en alcak noktasinin dunya z'si.

    Taban kalinligini TAHMIN etmek yerine STL koselerinden olcer.
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
    """Ayak bilegi MERKEZININ, taban duz yere basarken yerden yuksekligi.

    Ayak bilegi roll ekseni yataydir; merkez o eksen uzerinde oldugu icin
    eklem anchor'inin z'si merkezin z'sidir.
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
#  ANA PROGRAM
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--xml", default=XML_PATH)
    ap.add_argument("--zhip", type=float, default=Z_HIP,
                    help="kalca merkezinin yer uzeri yuksekligi [m]")
    ap.add_argument("--step-len", type=float, default=STEP_LEN)
    ap.add_argument("--push", type=float, default=PUSH_FORCE,
                    help="itme kuvveti [N] (t=6.0s'de 0.1 s boyunca)")
    ap.add_argument("--push-axis", default=PUSH_AXIS, choices=["lat", "fwd"],
                    help="itme yonu: lat = yanal, fwd = ileri")
    ap.add_argument("--push-viz", default="arrow",
                    choices=["arrow", "ball", "none"],
                    help="itmeyi nasil gosterelim: arrow = kirmizi ok cizilir "
                         "(kuvvet xfrc_applied ile uygulanir), ball = gercek bir "
                         "top firlatilir ve robota carpar, none = gorsel yok")
    ap.add_argument("--no-dcm", action="store_true",
                    help="DCM ayak yerlesim geri beslemesini kapat (eski davranis)")
    ap.add_argument("--k-dcm", type=float, default=K_DCM,
                    help="DCM geri besleme kazanci (0 = kapali, 1.0 = tam)")
    ap.add_argument("--time", type=float, default=MAX_SIM_TIME)
    ap.add_argument("--record", metavar="DOSYA.mp4", default=None,
                    help="yuruyusu mp4 olarak kaydet (offscreen render)")
    ap.add_argument("--fps", type=int, default=30, help="video kare hizi")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="oynatma hizi. 1.0 = gercek zaman, 0.5 = yavas cekim, "
                         "0 = frenleme yok (olabildigince hizli)")
    ap.add_argument("--ankle-fb", action="store_true",
                    help="govde egimi -> ayak bilegi geri beslemesini ac")
    ap.add_argument("--ankle-fb-sign", type=float, default=1.0,
                    help="geri besleme isareti; kotulesirse -1 dene")
    ap.add_argument("--refine-ik", action="store_true",
                    help="Kajita cozumunu DLS ile 0.01 mm'ye cilala (gerekmez)")
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
        sys.exit(f"HATA: {model.nu} aktuator var, 12 olmali. fix_robot_xml.py calistir.")

    qadr, ctrl_idx = [], []
    for n in JOINT_ORDER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if jid < 0 or aid < 0:
            sys.exit(f"HATA: '{n}' eklemi/aktuatoru yok.")
        qadr.append(model.jnt_qposadr[jid])
        ctrl_idx.append(aid)
    qadr = np.array(qadr); ctrl_idx = np.array(ctrl_idx)

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    if torso_id < 0:
        sys.exit("HATA: 'torso' govdesi yok.")
    # --- itme topu (varsa) ---
    ball_qadr = ball_dofadr = -1
    ball_mass = 0.0
    bjid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "push_ball_free")
    if bjid >= 0:
        ball_qadr = int(model.jnt_qposadr[bjid])
        ball_dofadr = int(model.jnt_dofadr[bjid])
        bbid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "push_ball")
        ball_mass = float(model.body_mass[bbid])

    # Robot kutlesi = torso alt agaci (sahnedeki top vs. haric)
    robot_mass = float(model.body_subtreemass[torso_id])
    print(f"robot kutlesi: {robot_mass:.4f} kg")
    if robot_mass < 0.05:
        sys.exit("HATA: robot kutlesi ~0. Sahte inertial'lar duruyor, "
                 "fix_robot_xml.py calistir.")

    # ---------------------------------------------------------- olcumler
    ankle_h = measure_ankle_height(model, data, qadr)
    print(f"ayak bilegi merkezi yerden (olculdu): {ankle_h*1000:.1f} mm")
    drop = z_hip - ankle_h
    print(f"bacak dususu : {drop:.4f} m  = erisimin %{drop/MAX_REACH*100:.0f}'i "
          f"(max {MAX_REACH:.4f} m)")
    if drop > MAX_REACH * 0.95:
        sys.exit(f"HATA: --zhip {z_hip} fazla yuksek, bacak kilitlenir. "
                 f"En fazla {ankle_h + MAX_REACH*0.92:.3f} kullan.")
    if drop < MAX_REACH * 0.45:
        print("UYARI: cok derin cokuyorsun, diz 120 dereceyi asabilir.")

    def stance_q(z_hip_now, latA=LAT_A, latB=LAT_B, fwdA=0.0, fwdB=0.0,
                 liftA=0.0, liftB=0.0):
        """Kalca referansi z_hip_now'da iken iki bacagin eklem acilari."""
        hip_ref_w = to_model(0.0, 0.0, z_hip_now)          # kalca orta noktasi (dunya)
        torso_w = hip_ref_w - BODY_REF_OFFSET               # govde orijini (dunya)
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
    # ayaklari yere tam otur
    data.qpos[2] -= lowest_point_of_feet(model, data)
    mujoco.mj_forward(model, data)
    print(f"baslangic govde yuksekligi: {data.qpos[2]:.4f} m")
    data.ctrl[ctrl_idx[:6]] = qA0
    data.ctrl[ctrl_idx[6:]] = qB0

    # ---------------------------------------------------------- planlayici
    footstep = np.array([[i * args.step_len, LAT_A if i % 2 == 0 else LAT_B, 0.0]
                         for i in range(N_STEPS)])
    Ks, Kx, F_prev, sysA, sysB, sysC = init_preview_control(DT_MPC, z_hip, N_PREVIEW)
    planner = TightStepPlanner(footstep, STEP_TIME, INITIAL_DELAY)
    planner.p_ref_y[:INITIAL_DELAY] = 0.0
    planner.p_ref_x[:INITIAL_DELAY] = 0.0
    # BITIS: DynamicStepPlanner adimlar bitince ZMP referansini SON ayak izinde
    # donduruyor (p_ref_y = LAT_B). Yani robot sonsuza kadar TEK AYAK uzerinde
    # durmaya calisiyor ve deviriliyor -- 11.73 s'deki dusme tam olarak buydu.
    # Adimlar bitince ZMP'yi iki ayagin ORTASINA getiriyoruz (cift destek).
    STOP_TICK = INITIAL_DELAY + (N_STEPS - 1) * STEP_TIME
    planner.p_ref_x[STOP_TICK:] = footstep[-1, 0]
    planner.p_ref_y[STOP_TICK:] = 0.0

    log_t, log_cf, log_cl, log_pf, log_pl, log_bz = [], [], [], [], [], []
    log_dcm_l, log_dcm_f = [], []
    dcm_l = dcm_f = 0.0
    raw_dcm_l = raw_dcm_f = 0.0
    k_dcm = 0.0 if args.no_dcm else args.k_dcm
    ball_fired = False
    state_f = np.zeros(3)      # ileri
    state_l = np.zeros(3)      # yanal
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

    # KAMERA AYARI YOK.
    # Onceki surumde viewer.cam.type = mjCAMERA_FIXED ile XML'deki sabit
    # kamera zorlaniyordu; bu goruntuyu kilitliyor ve fareyle donduremiyordun.
    # Artik MuJoCo'nun varsayilan serbest kamerasi kullaniliyor: fareyle
    # dondur (sol tik surukle), kaydir (sag tik surukle), yakinlastir
    # (tekerlek). Pencerede bir govdeye cift tiklayip Ctrl+sag tik ile
    # MuJoCo'nun kendi takip modunu da acabilirsin.
    viewer = None
    if not args.headless:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(model, data)
        viewer.__enter__()

    # ------------------------------------------------------------ kayit
    renderer = vid = None
    vid_kind = None
    next_frame = 0.0
    frame_dt = 1.0 / max(1, args.fps)
    if args.record:
        try:
            renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        except Exception as e:
            sys.exit(f"Renderer acilamadi: {e}\n"
                     f"Cozunurluk XML'deki offwidth/offheight'i asmis olabilir "
                     f"(su an 1920x1080).")
        vid_kind, vid = open_video_writer(args.record, args.fps)
        print(f"kayit        : {args.record}  {args.width}x{args.height} @ {args.fps} fps")

    t_wall = time.perf_counter()
    exit_reason = "sure doldu"
    next_report = 0.0
    next_sync = 0.0
    arrow_alpha = 0.0    # ilk video karesinde tanimli olsun (flake8 F821 / --record + --push)
    try:
        while data.time < args.time:
            if viewer is not None and not viewer.is_running():
                exit_reason = "goruntuleyici penceresi kapatildi"
                break
            t = data.time

            data.xfrc_applied[torso_id, :] = 0.0
            pushing = bool(args.push) and PUSH_TIME < t < PUSH_TIME + PUSH_DUR

            if args.push and args.push_viz == "ball":
                # GERCEK MERMI: kuvvet yerine carpisma. Ayni itkiyi versin diye
                # hiz = F*dt / m_top secilir.
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
                    print(f"t={t:.2f}s  TOP FIRLATILDI: m={ball_mass*1000:.0f} g, "
                          f"v={v:.2f} m/s, itki={abs(args.push)*PUSH_DUR:.2f} Ns")
            elif pushing:
                if args.push_axis == "lat":
                    data.xfrc_applied[torso_id, 0] = args.push      # model +X
                else:
                    data.xfrc_applied[torso_id, 1] = -args.push     # model -Y = ileri

            if t < SETTLE_TIME:
                z_now = SQUAT_START_Z
            elif t < SETTLE_TIME + SQUAT_TIME:
                u = (t - SETTLE_TIME) / SQUAT_TIME
                z_now = SQUAT_START_Z - (SQUAT_START_Z - z_hip) * u
            else:
                z_now = z_hip
                if not walking:
                    walking = True
                    print(f"t={t:.2f}s  cokme bitti, ZMP planlayici basladi "
                          f"(ilk adim {INITIAL_DELAY*DT_MPC:.1f}s sonra)")

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
                    lift_A = lift_B = 0.0      # adimlar bitti, iki ayak da yerde
                    swing = None
                elif step_index >= INITIAL_DELAY:
                    k = step_index - INITIAL_DELAY
                    ph_i = k % STEP_TIME
                    tx, ty, is_A = planner.swing_target(step_index, state_f, state_l)

                    # ---------------- DCM AYAK YERLESIM GERI BESLEMESI ------
                    # Her kontrol adiminda yeniden hesaplanir; yani ayak
                    # HAVADAYKEN bile inecegi nokta guncellenir.
                    if k_dcm > 0.0:
                        c_m, v_m = measured_com_state(model, data, torso_id)
                        dl, df, raw_dcm_l, raw_dcm_f = dcm_correction(
                            c_m, v_m,
                            (state_l[0], state_l[1]), (state_f[0], state_f[1]),
                            k_dcm, DCM_LIM_LAT, DCM_LIM_FWD)
                        # inise dogru duzeltmeyi yumusakca sifirla
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
                print(f"t={t:.3f}s IK HATASI ({ik_fail}): {e}")
                if ik_fail >= 3:
                    exit_reason = "ust uste IK hatasi"
                    print("Ust uste IK hatasi -- duruyorum.")
                    break

            cmdA, cmdB = qA.copy(), qB.copy()
            if (USE_ANKLE_STABILIZER or args.ankle_fb) and walking:
                sgn = args.ankle_fb_sign
                R_b = quat2mat(data.qpos[3:7])
                # ileri = model -Y, yanal = model +X
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
                exit_reason = "NaN (simulasyon patladi)"
                print(f"t={t:.3f}s SIMULASYON PATLADI (NaN).")
                break

            log_t.append(t)
            com = data.sensor('com_pos').data
            log_cf.append(HIP_Y - float(com[1]))     # ileri
            log_cl.append(float(com[0]) - MID_X)     # yanal
            log_pf.append(state_f[0]); log_pl.append(state_l[0])
            log_bz.append(float(data.qpos[2]))
            log_dcm_l.append(dcm_l); log_dcm_f.append(dcm_f)

            # --- saniyede bir teshis satiri --------------------------------
            if t >= next_report:
                next_report += 1.0
                hip_meas = data.qpos[2] + BODY_REF_OFFSET[2]
                sag = (z_now - hip_meas) * 1000.0
                cf = (HIP_Y - float(data.sensor('com_pos').data[1])) - log_cf[0] if log_cf else 0.0
                nc = data.ncon if hasattr(data, "ncon") else -1
                wall = time.perf_counter() - t_wall
                rtf = (t / wall) if wall > 1e-6 else 0.0
                print(f"  t={t:4.1f}s  ileri={cf:+.3f}m  kalca z={hip_meas:.4f}"
                      f" (sarkma {sag:+5.1f}mm)  salinim={swing or '-'}"
                      f"  adim={max(0,step_index-INITIAL_DELAY)//STEP_TIME}"
                      f"  temas={nc}  hiz={rtf:.2f}x"
                      f"  DCM hata=({raw_dcm_l*1000:+5.1f},{raw_dcm_f*1000:+5.1f})mm"
                      f" duzeltme=({dcm_l*1000:+5.1f},{dcm_f*1000:+5.1f})mm")

            if data.qpos[2] < 0.08:
                exit_reason = "robot dustu (govde z < 0.08)"
                print(f"t={t:.2f}s  govde z={data.qpos[2]:.3f} -- robot dustu.")
                break

            # ------------------------------------------------ video karesi
            if renderer is not None and data.time >= next_frame:
                next_frame += frame_dt
                renderer.update_scene(data)
                if args.push_viz == "arrow" and arrow_alpha > 0:
                    draw_push(renderer.scene, data, torso_id,
                              args.push_axis, args.push, arrow_alpha, reset=False)
                vid = write_frame(vid_kind, vid, renderer.render())

            # PERFORMANS: viewer.sync() PAHALI bir cagridir (govde basina ~40
            # mesh var). Her 1 ms adimda cagirmak simulasyonu 10-20 kat
            # yavaslatir. 50 Hz yeterli.
            # itme oku: itme suresince tam, sonrasinda 0.6 s icinde soner
            arrow_alpha = 0.0
            if args.push and args.push_viz == "arrow":
                if pushing:
                    arrow_alpha = 1.0
                elif PUSH_TIME <= t < PUSH_TIME + PUSH_DUR + 0.6:
                    arrow_alpha = 1.0 - (t - PUSH_TIME - PUSH_DUR) / 0.6

            if viewer is not None and data.time >= next_sync:
                next_sync += VIEW_DT
                if args.push_viz == "arrow":
                    # alpha 0 olsa bile cagir: eski oku temizlesin
                    draw_push(viewer.user_scn, data, torso_id,
                              args.push_axis, args.push, arrow_alpha, reset=True)
                viewer.sync()
                # GERCEK ZAMANLI FRENLEME
                # Onceki surumde uyku suresi VIEW_DT ile SINIRLANIYORDU ve
                # time.time() kullaniliyordu (Windows'ta ~16 ms cozunurluk).
                # Ikisi birlikte sim'in duvar saatinden hizli akmasina yol
                # aciyordu. Simdi perf_counter ile MUTLAK hedef zamana kadar
                # uyuyoruz; birikimli kayma olmuyor.
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

    print("\n--- OZET ---")
    print(f"cikis nedeni : {exit_reason}")
    print(f"sim suresi   : {log_t[-1] if log_t else 0:.2f} s")
    print(f"ilk adim     : t={SETTLE_TIME+SQUAT_TIME+INITIAL_DELAY*DT_MPC:.2f}s"
          f"  (yerlesme {SETTLE_TIME} + cokme {SQUAT_TIME} + bekleme "
          f"{INITIAL_DELAY*DT_MPC:.1f})")
    print(f"IK hatasi    : {ik_fail}")
    if log_t:
        print(f"kat edilen   : {log_cf[-1]-log_cf[0]:+.3f} m (ileri)")
        print(f"yanal kayma  : {log_cl[-1]-log_cl[0]:+.3f} m")
        print(f"govde z      : bas {log_bz[0]:.3f}  son {log_bz[-1]:.3f}  min {min(log_bz):.3f}")
        ok = min(log_bz) > 0.12 and log_t[-1] > args.time * 0.95
        print(f"SONUC        : {'AYAKTA KALDI / YURUDU' if ok else 'DUSTU veya ERKEN BITTI'}")

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
                ax[0].text(PUSH_TIME, ax[0].get_ylim()[1], f" Push 30N",
                           color="r", va="top", fontsize=9)
            plt.tight_layout(); plt.savefig("walk_trajectory.png", dpi=140)
            print("grafik: walk_trajectory.png")
        except ImportError:
            pass


if __name__ == "__main__":
    main()