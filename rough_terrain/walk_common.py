"""
walk_common.py
==============
main_walk.py'deki yardimci fonksiyonlarin BIREBIR kopyasi (video kaydi, itme
oku, olculen CoM, taban/bilek yuksekligi olcumu). main_walk_terrain.py bunlari
buradan alir; boylece main_walk.py dosyasinin adina bagimli olmaz.
"""
import sys

import numpy as np
import mujoco

VIEW_DT = 0.02


JOINT_ORDER = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
               "left_ankle_pitch", "left_ankle_roll",
               "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee",
               "right_ankle_pitch", "right_ankle_roll"]


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


def measured_com_state(model, data, torso_id):
    """Robotun GERCEK kutle merkezi konumu ve hizi (dunya, model eksenleri).

    DIKKAT: subtree_com[0] (world alt agaci) TUM modeli kapsar -- sahneye
    firlatilan top da dahil. Bu, DCM geri beslemesini bozar. Bu yuzden
    TORSO alt agacini kullaniyoruz: torso + iki bacak, baska hicbir sey.
    """
    mujoco.mj_subtreeVel(model, data)
    return data.subtree_com[torso_id].copy(), data.subtree_linvel[torso_id].copy()


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
