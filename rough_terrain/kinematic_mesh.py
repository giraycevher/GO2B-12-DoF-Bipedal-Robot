"""
kinematic_mesh.py
=================
robot_walk.xml (STL / Onshape kaynakli mesh robot) icin, senin kendi
`kinematic.py`'ndeki Kajita IK_leg'ini DOGRU parametrelerle cagiran ince bir
adaptor.

ONEMLI DUZELTME
---------------
Bu dosyanin ilk surumunde "Kajita IK bu robotta gecerli degil, eksenler
kesismiyor" demistim. **Bu YANLISTI.** Eksen DOGRULARINI degil, govde
cerceve ORIJINLERINI karsilastirmistim. Olcum sonucu:

    kalca: hip_yaw x hip_roll   ekseni arasi en kisa mesafe = 0.0000 mm
           hip_yaw x hip_pitch                              = 0.4751 mm
           hip_roll x hip_pitch                             = 0.0136 mm
           -> KURESEL KALCA MERKEZI (govde cerc.) = (-0.080434, 0.067676, -0.050196)

    ayak bilegi: ankle_pitch x ankle_roll                    = 0.0136 mm
           -> AYAK BILEGI MERKEZI = (-0.080757, 0.067438, -0.225009)

    A_leg (kalca merkezi -> diz ekseni)       = 0.086407 m
    B_leg (diz ekseni -> ayak bilegi merkezi) = 0.088407 m
    kalca merkezi -> ayak bilegi merkezi      = 0.174814 m
    A_leg + B_leg                             = 0.174814 m
    fark                                      = 0.13 MIKRON

Yani model tam olarak Kajita'nin kitabindaki bacak: kuresel kalca, iki
uzuv, iki eksenli ayak bilegi, duz bacakta uc nokta tam kolineer. Modelin
kinematigi kusursuz.

ILK RAPORDAKI 64 mm HATA NEREDEN GELIYORDU
------------------------------------------
IK_leg'e yanlis seyler besliyordum:

  1. Body_p olarak GOVDE ORIJINI verilmis. Dogrusu, kalca merkezi
     hizasindaki referans: govde orijini + (0, 0.067676, -0.050196).
     Fark: 49 mm dikey + 68 mm ileri.
  2. Foot_p olarak ayak-bilegi GOVDESININ ORIJINI verilmis. Dogrusu iki
     ayak bilegi ekseninin KESISIM NOKTASI -- govde orijininden 24.5 mm
     geride.
  3. A_leg = 0.0938 kullanilmis. Bu, kalca ile diz govdesi arasindaki 3B
     mesafe. Dogrusu kalca merkezinden DIZ EKSENINE olan dik uzaklik:
     0.086407. (Aradaki 7 mm, brackete ait yanal kacik.)
  4. Foot_R = np.eye(3) verilmis. Ayak govdesinin yerel cercevesi govde
     cercevesiyle hizali degil; q=0'daki sabit yonelim kullanilmali.
  5. Cerceve donusu atlanmis: ihracatta ILERI = +Y, YANAL = +X.

Bunlar duzeltilince, calisma uzayinin tamaminda:

    ortalama hata 0.74 mm, medyan 0.74 mm
    (bacak tam aciliminin %97'sini asan hedeflerde 5-14 mm'e cikar --
     orada zaten diz kilitlenir, o bolgede calismamalisin)

Kalan 0.74 mm, kalca eksenlerinin 0.475 mm'lik kesismezliginden geliyor;
yani modelleme hatasi degil, CAD toleransi.

KULLANIM
--------
    from kinematic_mesh import IK_leg_mesh, BODY_REF_OFFSET, MAX_REACH

    # foot_p: AYAK BILEGI MERKEZI, GOVDE (torso) cercevesinde
    #         model eksenleri: X=yanal, Y=ileri, Z=yukari
    q = IK_leg_mesh(foot_p, left=True)      # 6 aci, dogrudan data.ctrl'e

`kinematic.py` ayni klasordeyse onun IK_leg'i kullanilir (senin kodun,
degistirilmeden). Yoksa birebir ayni kopya devreye girer.
"""

import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
#  Senin kendi IK'ni kullan
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from kinematic import IK_leg as _IK_leg_kajita
    _IK_SOURCE = "kinematic.py (senin dosyan)"
except Exception:
    def _Rroll(q):
        c, s = math.cos(q), math.sin(q)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def _Rpitch(q):
        c, s = math.cos(q), math.sin(q)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def _IK_leg_kajita(Body_p, Body_R, D, A_leg, B_leg, Foot_p, Foot_R):
        hip_offset = Body_R @ np.array([0, D, 0])
        r = Foot_R.T @ (Body_p + hip_offset - Foot_p)
        C_len = np.linalg.norm(r)
        c5 = np.clip((C_len ** 2 - A_leg ** 2 - B_leg ** 2) / (2.0 * A_leg * B_leg), -1.0, 1.0)
        q5 = math.acos(c5)
        q6a = math.asin(np.clip((A_leg / C_len) * math.sin(np.pi - q5), -1.0, 1.0))
        q7 = math.atan2(r[1], r[2])
        if q7 > math.pi / 2:
            q7 -= math.pi
        elif q7 < -math.pi / 2:
            q7 += math.pi
        q6 = -math.atan2(r[0], np.sign(r[2]) * math.sqrt(r[1] ** 2 + r[2] ** 2)) - q6a
        R_mat = Body_R.T @ Foot_R @ _Rroll(-q7) @ _Rpitch(-q6 - q5)
        q2 = math.atan2(-R_mat[0, 1], R_mat[1, 1])
        cz, sz = math.cos(q2), math.sin(q2)
        q3 = math.atan2(R_mat[2, 1], -R_mat[0, 1] * sz + R_mat[1, 1] * cz)
        q4 = math.atan2(-R_mat[2, 0], R_mat[2, 2])
        return [q2, q3, q4, q5, q6, q7]
    _IK_SOURCE = "yerlesik kopya (kinematic.py bulunamadi)"


# ===========================================================================
#  ROBOTUN OLCULEN GEOMETRISI   (hepsi robot_final.xml'den turetildi)
#  Model cercevesi: X = yanal, Y = ileri, Z = yukari
# ===========================================================================
HIP_CENTER_L = np.array([-0.080434, 0.067676, -0.050196])   # sol kuresel kalca merkezi
MIRROR_X = -0.0054341                                        # govde simetri duzlemi
HIP_CENTER_R = np.array([2 * MIRROR_X - HIP_CENTER_L[0],
                         HIP_CENTER_L[1], HIP_CENTER_L[2]])

#: Govde orijininden, iki kalca merkezinin ORTA noktasina vektor.
#: Kajita'nin Body_p'si budur (Body_p + [0,D,0] = kalca merkezi).
BODY_REF_OFFSET = np.array([MIRROR_X, HIP_CENTER_L[1], HIP_CENTER_L[2]])

A_LEG = 0.086407          # kalca merkezi -> diz ekseni
B_LEG = 0.088407          # diz ekseni    -> ayak bilegi merkezi
MAX_REACH = A_LEG + B_LEG                                    # 0.174814 m

#: Ayak bilegi MERKEZININ, ayak govdesi yerel cercevesindeki yeri.
ANKLE_CENTER_IN_FOOT = np.array([0.000007, 0.0, -0.024512])

#: q=0'da (bacak duz, taban yatay) ayak govdesinin GOVDE cercevesindeki yonelimi.
#: "Ayak yere paralel" demek foot_R = R_FOOT_LEVEL demektir, np.eye(3) DEGIL.
R_FOOT_LEVEL = np.array([[0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [1.0, 0.0, 0.0]])

#: Model cercevesi -> Kajita IK cercevesi (x ileri, y yanal, z yukari).
#: ROBOTUN ONU model -Y yonudur. Ayak pedleri (lsp1..4) ayak bileginin bir
#: yaninda 74 mm, obur yaninda 26 mm uzaniyor; uzun taraf (parmak ucu) model
#: -Y'ye bakiyor. Bu yon secimi diz dalini da belirler: -Y ileri alindiginda
#: diz insan gibi ONE bukuluyor (pozitif), +Y alinirsa geriye (hyperextension).
#: Rz(+90): ik_x = -model_y,  ik_y = +model_x,  ik_z = model_z
M2I = np.array([[0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0]])
I2M = M2I.T

#: IK cercevesinde kalcanin yanal ofseti (Kajita'nin D'si).
#: DIKKAT: D, Body_p'ye GORELI olmali -- Body_p + [0,D,0] = kalca merkezi.
_BODY_IK_Y = float((M2I @ BODY_REF_OFFSET)[1])
#: NOT: ihracattaki "left_*" govdeleri model X<0 tarafinda. Ileri yon -Y
#: oldugu icin bu taraf robotun SAGI'na denk gelir; isimlendirme ihracattan
#: geliyor, yuruyus icin onemi yok (her bacak kendi D'sini alir).
D_LEFT = float((M2I @ HIP_CENTER_L)[1]) - _BODY_IK_Y      # -0.075000
D_RIGHT = float((M2I @ HIP_CENTER_R)[1]) - _BODY_IK_Y     # +0.075000

#: Kajita'nin [q2..q7] ciktisini modelin eklem yonlerine ceviren isaret haritasi.
#: 64 kombinasyonun tamami tarandi; asagidaki tek basina hem konumu (0.41 mm)
#: hem yonelimi (0.00000 derece) tutturan VE dizi one bukendir.
SIGN_MAP = np.array([1.0, 1.0, -1.0, 1.0, -1.0, -1.0])

#: Eklem limitleri (robot_walk.xml ile AYNI olmali)
Q_MIN = np.array([-0.60, -0.80, -1.80, -0.05, -1.50, -0.80])
Q_MAX = np.array([+0.60, +0.80, +1.80, +2.60, +1.50, +0.80])

JOINT_NAMES = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"]


# ===========================================================================
#  ILERI KINEMATIK  (dogrulama ve ayak bilegi merkezi hesabi icin)
# ===========================================================================
_CHAIN = [
    ((-0.0804341,   0.0679133, -0.000988717), (1.0,      0.0,      0.0,        0.0)),
    ((0.0,          0.0240365, -0.0492),      (0.5,      0.5,      0.5,       -0.5)),
    ((1.35932e-05,  0.0181133,  0.0245116),   (0.707107, 0.707107, 0.0,        0.0)),
    ((0.0864,       0.0,        0.03655),     (0.0,      0.0,      1.0,        0.0)),
    ((-0.0884,      0.0,        0.0),         (0.0,      1.0,      3.0135e-08, 0.0)),
    ((-1.35932e-05, 0.0245116, -0.0181133),   (0.707107, -0.707107, 0.0,       0.0)),
]
HIP_X_LEFT = -0.0804341
HIP_X_RIGHT = +0.0695659


def _q2m(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


_LOCAL_R = [_q2m(q) for _, q in _CHAIN]
_LOCAL_P = [np.array(p) for p, _ in _CHAIN]


def _rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def leg_fk(q, left=True):
    """Bacak ileri kinematigi. Doner: (ayak_govdesi_konumu, yonelim, orijinler, eksenler)"""
    p = np.zeros(3)
    R = np.eye(3)
    origins, axes = [], []
    for i in range(6):
        lp = _LOCAL_P[i]
        if i == 0 and not left:
            lp = np.array([HIP_X_RIGHT, lp[1], lp[2]])
        p = p + R @ lp
        R = R @ _LOCAL_R[i]
        origins.append(p.copy())
        axes.append(R[:, 2].copy())
        R = R @ _rz(q[i])
    return p, R, origins, axes


def ankle_center_fk(q, left=True):
    """Ayak bilegi MERKEZININ govde cercevesindeki konumu + ayak yonelimi."""
    p, R, _, _ = leg_fk(q, left)
    return p + R @ ANKLE_CENTER_IN_FOOT, R


# ===========================================================================
#  TERS KINEMATIK
# ===========================================================================
def IK_leg_mesh(foot_p, left=True, foot_R=None, clamp=True, refine=False, **_):
    """Kajita IK_leg'i bu robot icin dogru parametrelerle cagirir.

    foot_p : AYAK BILEGI MERKEZI'nin istenen konumu, GOVDE (torso) cercevesinde.
             Model eksenleri: X = yanal, Y = ileri, Z = yukari.
    left   : sol bacak mi
    foot_R : ayagin istenen yonelimi (model cercevesinde 3x3).
             None -> R_FOOT_LEVEL (taban yere paralel).
    refine : True ise sonucu sayisal olarak (DLS) mm alti hassasiyete cilalar.
             Normalde gerekmez; bacak tam acilimin %97'sini asan bolgede
             calismak zorundaysan ise yarar.

    Doner: 6 elemanli model eklem acisi vektoru
           [hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll]
    """
    foot_p = np.asarray(foot_p, dtype=float)
    if foot_R is None:
        foot_R = R_FOOT_LEVEL

    # model cercevesi -> IK cercevesi.
    # Ayak govdesinin yerel cercevesi q=0'da R_FOOT_LEVEL kadar donuk oldugu
    # icin Kajita'ya ZINCIR donusunu vermeliyiz: R_des @ R_FOOT_LEVEL^T.
    # Taban yatay istendiginde bu tam olarak BIRIM matristir.
    body_ik = M2I @ BODY_REF_OFFSET
    foot_ik = M2I @ foot_p
    chain_R = np.asarray(foot_R, dtype=float) @ R_FOOT_LEVEL.T
    footR_ik = M2I @ chain_R @ I2M
    D = D_LEFT if left else D_RIGHT

    q = np.asarray(_IK_leg_kajita(body_ik, np.eye(3), D, A_LEG, B_LEG,
                                  foot_ik, footR_ik), dtype=float)
    q = SIGN_MAP * q

    if refine:
        q = _refine(q, foot_p, foot_R, left)
    if clamp:
        q = np.clip(q, Q_MIN, Q_MAX)
    return q


def _refine(q, target_p, target_R, left, iters=8, tol=1e-5):
    """Damped least squares cila. Kajita cozumu zaten <1 mm; bu sadece
    CAD toleransindan gelen kalan hatayi da siler."""
    q = np.array(q, dtype=float)
    lam = 1e-4
    for _ in range(iters):
        p, R, origins, axes = leg_fk(q, left)
        ac = p + R @ ANKLE_CENTER_IN_FOOT
        err = np.zeros(6)
        err[:3] = target_p - ac
        Re = target_R @ R.T
        w = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
        s = np.linalg.norm(w)
        if s > 1e-9:
            err[3:] = w / s * math.atan2(s, np.trace(Re) - 1.0)
        if np.linalg.norm(err) < tol:
            break
        J = np.zeros((6, 6))
        for i in range(6):
            J[:3, i] = np.cross(axes[i], ac - origins[i])
            J[3:, i] = axes[i]
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)
        m = np.max(np.abs(dq))
        if m > 0.3:
            dq *= 0.3 / m
        q = q + dq
    return q


# ===========================================================================
#  OZ-TEST
# ===========================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    print(f"IK kaynagi: {_IK_SOURCE}")
    print(f"A_leg = {A_LEG:.6f}   B_leg = {B_LEG:.6f}   max erisim = {MAX_REACH:.6f} m")
    print(f"D_left = {D_LEFT:+.6f}   D_right = {D_RIGHT:+.6f}   (IK cercevesi)")
    print(f"Body_p ofseti (model) = {BODY_REF_OFFSET}")

    # --- eksen kesisimi kontrolu (modelin Kajita uyumlu oldugunun kaniti) ---
    def line_line(p1, d1, p2, d2):
        d1 = d1 / np.linalg.norm(d1); d2 = d2 / np.linalg.norm(d2)
        n = np.cross(d1, d2); nn = np.linalg.norm(n)
        if nn < 1e-12:
            return float(np.linalg.norm(np.cross(p2 - p1, d1)))
        return float(abs(np.dot(p2 - p1, n)) / nn)

    _, _, O, A = leg_fk(np.zeros(6), True)
    print("\n--- modelin Kajita varsayimlarini sagladiginin kaniti ---")
    for i, j in ((0, 1), (0, 2), (1, 2)):
        print(f"  {JOINT_NAMES[i]:10s} x {JOINT_NAMES[j]:10s} eksen mesafesi = "
              f"{line_line(O[i], A[i], O[j], A[j])*1000:7.4f} mm")
    print(f"  {JOINT_NAMES[4]:10s} x {JOINT_NAMES[5]:10s} eksen mesafesi = "
          f"{line_line(O[4], A[4], O[5], A[5])*1000:7.4f} mm")
    hc, _ = HIP_CENTER_L, None
    ac0, _ = ankle_center_fk(np.zeros(6), True)
    print(f"  kalca merkezi -> ayak bilegi merkezi = {np.linalg.norm(hc-ac0):.6f} m")
    print(f"  A_leg + B_leg                        = {MAX_REACH:.6f} m")
    print(f"  fark (duz bacak kolineerlik testi)   = "
          f"{abs(np.linalg.norm(hc-ac0)-MAX_REACH)*1e6:.2f} mikron")

    # --- IK -> FK dogrulugu ---
    print("\n--- IK dogrulugu (calisma uzayi taramasi) ---")
    for label, refine in (("Kajita (senin kinematic.py)", False),
                          ("Kajita + DLS cila", True)):
        errs, angs = [], []
        for left in (True, False):
            hc = HIP_CENTER_L if left else HIP_CENTER_R
            for drop in (0.10, 0.12, 0.14, 0.155, 0.165):
                for fwd in (-0.06, -0.03, 0.0, 0.03, 0.06):
                    for lat in (-0.035, 0.0, 0.035):
                        T = hc + np.array([lat, fwd, -drop])
                        if np.linalg.norm(T - hc) > MAX_REACH * 0.98:
                            continue
                        q = IK_leg_mesh(T, left=left, refine=refine)
                        got, R = ankle_center_fk(q, left)
                        errs.append(np.linalg.norm(got - T))
                        angs.append(math.degrees(math.acos(np.clip(
                            (np.trace(R @ R_FOOT_LEVEL.T) - 1) / 2, -1, 1))))
        errs = np.array(errs); angs = np.array(angs)
        print(f"  {label:28s}: {len(errs)} hedef | konum ort {errs.mean()*1000:6.3f} mm "
              f"max {errs.max()*1000:6.3f} mm | yonelim max {angs.max():.4f} derece")

    print("\n--- ornek duruslar ---")
    for drop in (0.10, 0.13, 0.155, 0.17):
        q = IK_leg_mesh(HIP_CENTER_L + np.array([0, 0, -drop]), left=True)
        print(f"  bacak dususu {drop:.3f} m ({drop/MAX_REACH*100:3.0f}% erisim) -> "
              f"diz {math.degrees(q[3]):5.1f} derece   q = {np.round(q,4)}")