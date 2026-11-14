"""
rough_gait.py
=============
Engebeli zemin icin yuruyus kontrolcusu. main_walk.py'deki ZMP preview control
+ Kajita IK yapisini AYNEN korur, ustune dort sey ekler:

  1. ARAZIYE GORE AYAK YUKSEKLIGI  -- her ayagin "zemin yuksekligi" (g) ayri
     tutulur. Salinim ayaginin inecegi noktada arazi ornegi alinir, ayak
     oraya gore iner. (Duz zeminde hepsi 0 -> eski davranis.)
  2. EGIME UYUM -- ayak izinin altindaki araziye en-kucuk-kareler duzlemi
     oturtulur; ayak tabani o duzleme paralel komut edilir (ayak bilegi
     pitch/roll). Boylece ayak tumsege tek noktadan degil, yuzeyiyle basar.
  3. ERKEN TEMAS -- salinim ayagi plandan ONCE yere degerse (dokunma
     sensoru), inis orada dondurulur; servo zemini itip govdeyi devirmez.
  4. ERISIM KORUMASI -- bir bacak tam acilima yaklasirsa govde o kadar
     alcaltilir (diz kilitlenmesi / IK doygunlugu olmaz).

Bu dosya MuJoCo GEREKTIRMEZ. Fizik olmadan mantigi denemek icin:

    python rough_gait.py        # kinematik kuru-kosu: IK->FK ile taban-zemin kontrolu

MATEMATIK (kisa turetimler)
---------------------------
(a) Ayak izi duzlemi.  Ayak izinde i = 1..N ornek noktasi, ayak referans
    noktasina gore (dX_i, dY_i), arazi yuksekligi h_i. Aranan duzlem:

        z = a*dX + b*dY + c

    En kucuk kareler:  min sum (a dX_i + b dY_i + c - h_i)^2
    -> normal denklemler  A^T A [a b c]^T = A^T h,   A_i = [dX_i  dY_i  1].
    Sonra a, b egim sinirina kirpilir ve duzlem, HICBIR ornegin altinda
    kalmayacak sekilde yukari kaydirilir:

        g = max_i ( h_i - a dX_i - b dY_i )

    Yani rijit taban, arazinin EN YUKSEK noktalarina oturur (iceri girmez).
    Sayisal ornek: on uc 4 mm, topuk 0 mm, ayak boyu 100 mm ise
    egim = 4/100 = 0.04 rad (2.3 derece); ayak bilegi pitch'i bu kadar doner.

(b) Taban normali ve ayak bilegi konumu.  z = aX + bY duzleminin normali
    n = (-a, -b, 1) / sqrt(1 + a^2 + b^2).  Ayak bilegi merkezi tabandan
    ankle_h kadar NORMAL boyunca yukaridadir:

        p_bilek = S + n * ankle_h,     S = (X, Y, g + kaldirma)

    Donme matrisi: e_z'yi n'ye en kisa yoldan tasiyan Rodrigues donmesi,
    eksen k = e_z x n, aci = acos(n_z).

(c) DCM (capture point) duzeltmesi.  LIPM:  c'' = w^2 (c - p),  w = sqrt(g/z_c).
    xi = c + c'/w  tanimlanirsa  xi' = w (xi - p): xi, ZMP'den ustel olarak
    UZAKLASIR. Ayagi xi'nin ustune koyarsan (p = xi) xi' = 0 olur, robot
    durur. O yuzden olculen xi ile planlanan xi farki kadar ayak kaydirilir.
    Sayisal ornek: z_c = 0.185 m -> w = sqrt(9.81/0.185) = 7.28 1/s.
    Itme CoM'a 0.10 m/s ekstra hiz verdiyse  d_xi = 0.10/7.28 = 13.7 mm;
    ayak 13.7 mm oteye basar.

    main_walk.py'deki HATA: duzeltme salinimin son %20'sinde SIFIRA
    indiriliyordu (fade). Ayak tam inerken nominal noktaya geri donuyordu,
    yani duzeltme inis aninda ~%6'ya dusuyordu. Burada son %20'de duzeltme
    DONDURULUR (son deger tutulur). Eski davranis: dcm_hold=False.

(d) "Planlanan xi" NEYE GORE?  (gercek mesh'lerle olcum sonucu eklendi)
    LIPM tum kutleyi kalcada tek nokta sayar. Bu robotta bacaklar agir:
    plan kalcayi yanal +-71 mm sallarken GERCEK kutle merkezi +-48 mm
    sallaniyor (oran 0.68), cunku destek bacagi yerde duruyor. Olculen xi
    dogrudan LIPM xi'si ile kiyaslanirsa, hic itme yokken bile
    |hata| = 60 mm cikiyor ve ayak her adimda bosuna 20-40 mm kayiyor.
    Cozum: kiyas, komut edilen durusun (govde konumu + 12 eklem acisi)
    ileri kinematikle hesaplanan GERCEK kutle merkezine gore yapilir:

        c_plan = (1/M) * sum_i m_i * r_i(q_komut),   v_plan = (c_plan[k]-c_plan[k-1])/dt

    Boylece hata sadece servo gecikmesi + gercek bozucu (itme, arazi) icerir.

(e) Ayak kaydiysa ZMP referansi da kaymali.  Preview control CoM'u
    p_ref'e gore planlar. Ayak nominalden d kadar oteye basar ama p_ref
    eski yerde kalirsa, bir sonraki destek fazinda CoM ayagin d kadar
    berisinde kalir -> robot salinim ayagina dogru devrilir. O yuzden
    salinim sirasinda, o ayagin destek olacagi araliktaki p_ref = hedef.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from robot_utils import DynamicStepPlanner
from preview_control import init_preview_control
from kinematic_mesh import (IK_leg_mesh, R_FOOT_LEVEL, BODY_REF_OFFSET, MAX_REACH,
                            MIRROR_X, HIP_CENTER_L, HIP_CENTER_R)

MID_X = MIRROR_X
HIP_Y = float(BODY_REF_OFFSET[1])
REF_MAX = 20000          # DynamicStepPlanner'daki tampon uzunlugu


def to_model(lat, fwd, up):
    """Yuruyus cercevesi (yanal, ileri, yukseklik) -> dunya / model eksenleri.
    Robotun onu model -Y, yanali model +X."""
    return np.array([MID_X + lat, HIP_Y - fwd, up])


@dataclass
class Footprint:
    """Tabanin, ayak bilegi merkezinin izdusumune gore sinirlari [m] (yuruyus cercevesi)."""
    fwd_min: float = -0.026     # topuk
    fwd_max: float = +0.074     # parmak ucu
    lat_min: float = -0.042
    lat_max: float = +0.042

    def samples(self, n_fwd=5, n_lat=4):
        f = np.linspace(self.fwd_min, self.fwd_max, n_fwd)
        l = np.linspace(self.lat_min, self.lat_max, n_lat)
        L, F = np.meshgrid(l, f)
        return L.ravel(), F.ravel()


@dataclass
class GaitParams:
    z_hip: float = 0.185        # kalca merkezi yuksekligi (LIPM z_c). Duz zemindeki
                                # 0.195'ten alcak: arazi icin diz payi birakir.
    half_stance: float = 0.075
    dt_mpc: float = 0.01
    n_preview: int = 160
    step_time: int = 90         # kontrol adimi cinsinden (0.45 s)
    initial_delay: int = 60
    step_len: float = 0.03
    step_height: float = 0.028  # arazinin USTUNDE kalan net bosluk
    n_steps: int = 24
    # --- arazi uyumu
    use_map: bool = True        # False = "kor" mod: yukseklik haritasi kullanilmaz
    use_slope: bool = True
    max_tilt: float = 0.14      # ayak egim siniri [rad] (~8 derece)
    use_touchdown: bool = True
    touch_frac: float = 0.60    # salinimin bu oranindan sonra erken temas kabul edilir
    touch_force: float = 1.0    # esik [N]; main dosyasi 0.25*m*g olarak ayarlar
    td_window: float = 0.012    # erken temas, komut edilen taban beklenen zeminin en cok
                                # bu kadar ustundeyken kabul edilir [m] (kor modda sinirsiz)
    reach_frac: float = 0.95    # bacak boyu / max erisim ust siniri
    ground_tau: float = 0.30    # govde yuksekligi zemin referansi zaman sabiti [s]
    # --- DCM geri beslemesi
    k_dcm: float = 0.0          # 0 = kapali (varsayilan). 1.0 = tam. Olcum: acik/kapali
                                # ayni itmelere dayaniyor, arazide mesafeyi dusuruyor.
    dcm_lim_lat: float = 0.045
    dcm_lim_fwd: float = 0.060
    dcm_start: float = 0.25     # salinimin bu oranindan ONCE olcume guvenme: adim
                                # gecisinde servo gecikmesi 20-25 mm sahte hata uretir
                                # (olculdu); 0.25-0.80 arasinda sahte hata < 6 mm
    dcm_freeze: float = 0.80    # salinimin bu oranindan sonra duzeltme dondurulur
    dcm_hold: bool = True       # False = main_walk.py'deki eski "sifira sondur" davranisi
    dcm_deadband: float = 0.008 # bu kadar hataya tepki verme [m] (nominal izleme hatasi)
    zmp_follow: bool = True     # ayak kaydirilinca ZMP referansi da onunla kayar
    refine_ik: bool = False


@dataclass
class Foot:
    pos: np.ndarray                     # [yanal, ileri] taban referans noktasi
    g: float = 0.0                      # o noktadaki taban yuksekligi (dunya z)
    tilt: np.ndarray = field(default_factory=lambda: np.zeros(2))   # (a, b) = dz/dX, dz/dY
    lift: float = 0.0


class TightStepPlanner(DynamicStepPlanner):
    """Sadece nominal ayak izini dondurur (bkz. main_walk.py'deki aciklama)."""

    def swing_target(self, step_index, state_x, state_y):
        if step_index < self.initial_delay:
            return 0.0, 0.0, False
        k = step_index - self.initial_delay
        t = min(k // self.step_time + 1, self.total_steps - 1)
        return (float(self.footsteps[t, 0]), float(self.footsteps[t, 1]), (t % 2 == 0))


def tilt_rotation(a, b):
    """e_z'yi z = aX + bY duzleminin normaline tasiyan donme matrisi ve normal."""
    n = np.array([-a, -b, 1.0])
    n /= np.linalg.norm(n)
    k = np.array([-n[1], n[0], 0.0])            # e_z x n
    s = np.linalg.norm(k)
    if s < 1e-12:
        return np.eye(3), n
    k /= s
    ang = math.atan2(s, n[2])
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
    return R, n


class RoughGait:
    """Engebeli zemin yuruyus kontrolcusu.

    Kullanim (main_walk_terrain.py):
        gait = RoughGait(params, terrain, ankle_h, fpA, fpB)
        her DT_MPC'de bir :  gait.tick(com, vel, touch_A, touch_B)
        her sim adiminda  :  qA, qB, torso_w = gait.joint_targets(z_now)
    """

    def __init__(self, params, terrain, ankle_h, fp_A=None, fp_B=None, com_model=None):
        """com_model: (torso_w, qA, qB) -> tum govdenin kutle merkezi (dunya).
        Verilirse DCM hatasi, LIPM durumuna gore degil, KOMUT EDILEN durusun
        gercek kutle merkezine gore olculur (bkz. dosya basindaki (d) maddesi)."""
        self.p = p = params
        self.com_model = com_model
        self.plan_com = None
        self.plan_vel = None
        self.terrain = terrain
        self.ankle_h = float(ankle_h)
        self.fp = {"A": fp_A or Footprint(), "B": fp_B or Footprint()}

        self.lat_A, self.lat_B = -p.half_stance, +p.half_stance
        self.footstep = np.array(
            [[i * p.step_len, self.lat_A if i % 2 == 0 else self.lat_B, 0.0]
             for i in range(p.n_steps)])
        (self.Ks, self.Kx, self.F_prev,
         self.sysA, self.sysB, self.sysC) = init_preview_control(p.dt_mpc, p.z_hip, p.n_preview)
        self.planner = TightStepPlanner(self.footstep, p.step_time, p.initial_delay)
        self.planner.p_ref_x[:p.initial_delay] = 0.0
        self.planner.p_ref_y[:p.initial_delay] = 0.0
        # adimlar bitince ZMP iki ayagin ortasina (cift destek)
        self.stop_tick = p.initial_delay + (p.n_steps - 1) * p.step_time
        self.planner.p_ref_x[self.stop_tick:] = self.footstep[-1, 0]
        self.planner.p_ref_y[self.stop_tick:] = 0.0

        self.state_f = np.zeros(3)          # ileri  (konum, hiz, ivme)
        self.state_l = np.zeros(3)          # yanal
        self.sum_e_f = self.sum_e_l = 0.0
        self.step_index = 0

        self.feet = {"A": Foot(np.array([self.lat_A, 0.0])),
                     "B": Foot(np.array([self.lat_B, 0.0]))}
        for name in ("A", "B"):
            g, a, b = self._support(name, self.feet[name].pos)
            self.feet[name].g = g
            self.feet[name].tilt = np.array([a, b])
        self.start = None                   # salinim basindaki ayak kopyasi
        self.swing = None
        self.td_lock = False
        self.g_ref = 0.5 * (self.feet["A"].g + self.feet["B"].g)
        self.reach_drop = 0.0

        self.omega = math.sqrt(9.81 / p.z_hip)
        self.corr = np.zeros(2)             # uygulanan DCM duzeltmesi [yanal, ileri]
        self.raw = np.zeros(2)              # ham DCM hatasi (sapma cikarilmis)
        self.bias = np.zeros(2)
        self._bias_sum = np.zeros(2)
        self._bias_n = 0
        self.early_td_count = 0

    # ------------------------------------------------------------ arazi
    def _support(self, name, pos, fallback_g=0.0):
        """pos=[yanal, ileri] noktasina basan ayagin (g, a, b) taban duzlemi."""
        p = self.p
        if self.terrain is None or not p.use_map:
            return float(fallback_g), 0.0, 0.0
        dl, df = self.fp[name].samples()
        dX, dY = dl, -df                                   # model eksenleri
        X0, Y0 = MID_X + pos[0], HIP_Y - pos[1]
        h = self.terrain.height(X0 + dX, Y0 + dY)
        a = b = 0.0
        if p.use_slope:
            A = np.column_stack([dX, dY, np.ones_like(dX)])
            sol, *_ = np.linalg.lstsq(A, h, rcond=None)
            lim = math.tan(p.max_tilt)
            a = float(np.clip(sol[0], -lim, lim))
            b = float(np.clip(sol[1], -lim, lim))
        g = float(np.max(h - a * dX - b * dY))
        return g, a, b

    def _path_max(self, name, p0, p1):
        """p0 -> p1 dogrusu boyunca ayak izinin gorecegi en yuksek arazi noktasi."""
        if self.terrain is None or not self.p.use_map:
            return -np.inf
        dl, df = self.fp[name].samples(3, 3)
        s = np.linspace(0.0, 1.0, 7)[:, None]
        lat = p0[0] + (p1[0] - p0[0]) * s + dl[None, :]
        fwd = p0[1] + (p1[1] - p0[1]) * s + df[None, :]
        return float(np.max(self.terrain.height(MID_X + lat, HIP_Y - fwd)))

    # ------------------------------------------------------------ 100 Hz
    def tick(self, com=None, vel=None, touch_A=0.0, touch_B=0.0, ank_dz=None):
        """Bir kontrol adimi (DT_MPC). com/vel: OLCULEN CoM konum/hiz (dunya, model
        eksenleri) ya da None (geri besleme yok).
        ank_dz: OLCULEN (A bilegi z) - (B bilegi z) [m]; eklem acilari + govde
        yoneliminden ileri kinematikle bulunur. Erken temasta zeminin GERCEK
        yuksekligini yazmak icin kullanilir."""
        p = self.p
        si = self.step_index
        pl = self.planner

        # ---------------- preview control (main_walk.py ile ayni) -----------
        idx = np.minimum(si + np.arange(p.n_preview), REF_MAX - 1)
        cur = min(si, REF_MAX - 1)
        p_l = float((self.sysC @ self.state_l).item())
        self.sum_e_l += p_l - pl.p_ref_y[cur]
        self.state_l = self.sysA @ self.state_l + self.sysB * (
            -self.Ks * self.sum_e_l - float(np.dot(self.Kx, self.state_l))
            - float(np.dot(self.F_prev, pl.p_ref_y[idx])))
        p_f = float((self.sysC @ self.state_f).item())
        self.sum_e_f += p_f - pl.p_ref_x[cur]
        self.state_f = self.sysA @ self.state_f + self.sysB * (
            -self.Ks * self.sum_e_f - float(np.dot(self.Kx, self.state_f))
            - float(np.dot(self.F_prev, pl.p_ref_x[idx])))

        # ---------------- DCM hatasi ---------------------------------------
        have_meas = com is not None and vel is not None
        if have_meas:
            w = self.omega
            xi_meas = np.array([(float(com[0]) - MID_X) + float(vel[0]) / w,
                                (HIP_Y - float(com[1])) - float(vel[1]) / w])
            if self.plan_com is not None and self.plan_vel is not None:
                pc, pv = self.plan_com, self.plan_vel
                xi_plan = np.array([(pc[0] - MID_X) + pv[0] / w,
                                    (HIP_Y - pc[1]) - pv[1] / w])
            else:
                xi_plan = np.array([self.state_l[0] + self.state_l[1] / w,
                                    self.state_f[0] + self.state_f[1] / w])
            err = xi_meas - xi_plan
            # Ayakta dururken olculen sabit sapma -> ilk 0.1 s'de kalibre et.
            # (Daha gec olmaz: preview kontrolcu ilk adimdan ~1 s ONCE plani
            # oynatmaya baslar, olcum bunu gecikmeyle izler.)
            if 1 <= si < 11:
                self._bias_sum += err
                self._bias_n += 1
                self.bias = self._bias_sum / self._bias_n
            self.raw = err - self.bias
        else:
            self.raw = np.zeros(2)
        have_meas = have_meas and p.k_dcm > 0.0     # hata hep olculur, sadece k_dcm>0 ise uygulanir

        A, B = self.feet["A"], self.feet["B"]
        if si >= self.stop_tick:
            A.lift = B.lift = 0.0
            self.swing = None
        elif si >= p.initial_delay:
            k = si - p.initial_delay
            ph_i = k % p.step_time
            tx, ty, is_A = pl.swing_target(si, self.state_f, self.state_l)
            sw = "A" if is_A else "B"
            st = "B" if is_A else "A"

            if ph_i == 0:
                self.swing = sw
                self.td_lock = False
                f = self.feet[sw]
                self.start = Foot(f.pos.copy(), f.g, f.tilt.copy(), 0.0)
                if p.dcm_hold:
                    self.corr = np.zeros(2)

            u = (ph_i + 1) / float(p.step_time)            # (0, 1]; son adimda tam 1

            # ---- DCM ayak yerlesim duzeltmesi
            if have_meas:
                if u < p.dcm_start:
                    pass                                   # henuz guvenilir degil; corr = 0
                elif u < p.dcm_freeze:
                    e = np.sign(self.raw) * np.maximum(0.0, np.abs(self.raw) - p.dcm_deadband)
                    want = np.array([
                        np.clip(p.k_dcm * e[0], -p.dcm_lim_lat, p.dcm_lim_lat),
                        np.clip(p.k_dcm * e[1], -p.dcm_lim_fwd, p.dcm_lim_fwd)])
                    if p.dcm_hold:
                        self.corr = self.corr + 0.3 * (want - self.corr)   # hafif alcak-geciren
                    else:
                        self.corr = want
                elif not p.dcm_hold:                       # eski davranis: sifira sondur
                    self.corr = self.corr * max(0.0, (1.0 - u) / (1.0 - p.dcm_freeze)) \
                        if u < 1.0 else np.zeros(2)
            else:
                self.corr = np.zeros(2)
            tgt = np.array([ty + self.corr[0], tx + self.corr[1]])
            # ayaklar birbirine girmesin: salinim ayagi orta cizgiyi gecemez
            min_gap = 0.5 * (self.fp["A"].lat_max - self.fp["A"].lat_min) + 0.008
            if sw == "A":
                tgt[0] = min(tgt[0], self.feet["B"].pos[0] - 2 * min_gap)
            else:
                tgt[0] = max(tgt[0], self.feet["A"].pos[0] + 2 * min_gap)

            # ---- ZMP referansi ayagi izlesin.  Ayak nominalden farkli yere
            # basacaksa, bir sonraki destek fazinin ZMP referansi da oraya
            # tasinmali; yoksa CoM eski noktaya gore planlanir ve robot
            # destek ayaginin ustune tam gecemez.
            if p.zmp_follow and u <= p.dcm_freeze + 1e-9:
                a0 = p.initial_delay + (k // p.step_time + 1) * p.step_time
                a1 = min(a0 + p.step_time, self.stop_tick)
                if a0 < a1:
                    pl.p_ref_y[a0:a1] = tgt[0]
                    pl.p_ref_x[a0:a1] = tgt[1]
                if a0 + p.step_time >= self.stop_tick:      # son adim: cift destek ortasi
                    pl.p_ref_y[self.stop_tick:] = 0.5 * (tgt[0] + self.feet[st].pos[0])
                    pl.p_ref_x[self.stop_tick:] = 0.5 * (tgt[1] + self.feet[st].pos[1])

            # ---- adim bitti: kalan plani, ayagin GERCEKTEN bastigi yere gore kaydir.
            # Yoksa plan robottan bagimsiz ilerler; robot geride kaldikca hata
            # buyur (olculdu: 20 adimda -64 mm) ve kalca ayaklarin onune gecer.
            if p.zmp_follow and ph_i == p.step_time - 1:
                shift = tgt - np.array([ty, tx])            # [yanal, ileri]
                t_next = k // p.step_time + 1
                if np.any(shift != 0.0) and t_next + 1 < p.n_steps:
                    self.footstep[t_next + 1:, 0] += shift[1]
                    self.footstep[t_next + 1:, 1] += shift[0]
                    a1_ = p.initial_delay + (t_next + 1) * p.step_time
                    pl.p_ref_x[a1_:] += shift[1]
                    pl.p_ref_y[a1_:] += shift[0]

            f = self.feet[sw]
            touch = touch_A if sw == "A" else touch_B
            g_exp = self._support(sw, tgt, fallback_g=self.feet[st].g)[0]
            window = p.td_window if (self.terrain is not None and p.use_map) else np.inf

            # ---- erken temas: inisi dondur
            # (pencere sarti: taban beklenen zeminden cok yukaridayken gelen
            #  temas arazi degil, robotun o ayaga DEVRILMESIDIR -- kabul etme.)
            if (p.use_touchdown and not self.td_lock and u > p.touch_frac
                    and u < 1.0 and touch > p.touch_force
                    and (f.g + f.lift - g_exp) <= window):
                self.td_lock = True
                self.early_td_count += 1
                # Yeni zemin yuksekligi. KOMUT edilen yukseklik (f.g + f.lift)
                # kullanilamaz: gercek ayak komutun 6-10 mm gerisinde kalir
                # (olculdu), zemin oldugundan yuksek sanilir. Olcum varsa:
                # destek ayaginin zemini + iki bilek arasindaki olculen fark.
                if ank_dz is not None:
                    dz = ank_dz if sw == "A" else -ank_dz
                    f.g = min(f.g + f.lift, self.feet[st].g + dz)
                else:
                    f.g = f.g + f.lift
                f.lift = 0.0

            if not self.td_lock:
                # yatay hareket: once kalk, sonra git, sonra dik in
                uu = min(1.0, max(0.0, (u - 0.12) / 0.76))
                s = 0.5 * (1.0 - math.cos(math.pi * uu))
                g_t, a_t, b_t = self._support(sw, tgt, fallback_g=self.feet[st].g)
                s0 = self.start
                f.pos = s0.pos + (tgt - s0.pos) * s
                f.g = s0.g + (g_t - s0.g) * s
                f.tilt = s0.tilt + (np.array([a_t, b_t]) - s0.tilt) * s
                # bosluk: yol uzerindeki en yuksek noktanin ustunden gec
                extra = max(0.0, self._path_max(sw, s0.pos, tgt) - max(s0.g, g_t))
                f.lift = (p.step_height + extra) * math.sin(math.pi * u)
                if u >= 1.0:
                    f.lift = 0.0
                g_swing_ref = g_t
            else:
                g_swing_ref = f.g
            self.feet[st].lift = 0.0

            # govde yuksekligi icin zemin referansi (yumusatilmis)
            g_target = 0.5 * (self.feet[st].g + g_swing_ref)
            self.g_ref += (p.dt_mpc / p.ground_tau) * (g_target - self.g_ref)

        self.step_index += 1

        # ---- bu adimin KOMUT durusunun kutle merkezi (bir sonraki olcumle kiyaslanir)
        if self.com_model is not None:
            qA, qB, torso_w = self.joint_targets(p.z_hip)
            c = np.asarray(self.com_model(torso_w, qA, qB), dtype=float)
            if self.plan_com is not None:
                self.plan_vel = (c - self.plan_com) / p.dt_mpc
            self.plan_com = c

    # ------------------------------------------------------------ her sim adimi
    def ankle_world(self, name):
        """Ayak bilegi merkezinin dunya konumu ve ayak donme matrisi (R_tilt)."""
        f = self.feet[name]
        R_t, n = tilt_rotation(f.tilt[0], f.tilt[1])
        S = to_model(f.pos[0], f.pos[1], f.g + f.lift)
        return S + n * self.ankle_h, R_t

    def joint_targets(self, z_now):
        """z_now: kalca merkezinin zemin referansi USTUNDEKI yuksekligi.
        Doner: (qA, qB, torso_w) -- 6'sar eklem acisi ve govde orijini hedefi (dunya)."""
        p = self.p
        hip_ref_w = to_model(self.state_l[0], self.state_f[0], z_now + self.g_ref)
        torso_w = hip_ref_w - BODY_REF_OFFSET
        pA, RA = self.ankle_world("A")
        pB, RB = self.ankle_world("B")

        # erisim korumasi: bacak boyu <= reach_frac * MAX_REACH olacak kadar alcal
        Lmax = p.reach_frac * MAX_REACH
        drop = 0.0
        for pa, hc in ((pA, HIP_CENTER_L), (pB, HIP_CENTER_R)):
            d = (torso_w + hc) - pa
            dh2 = d[0] ** 2 + d[1] ** 2
            if dh2 + d[2] ** 2 > Lmax ** 2:
                v_new = math.sqrt(max(Lmax ** 2 - dh2, 1e-6))
                drop = max(drop, d[2] - v_new)
        self.reach_drop = drop
        torso_w = torso_w - np.array([0.0, 0.0, drop])

        qA = IK_leg_mesh(pA - torso_w, left=True, foot_R=RA @ R_FOOT_LEVEL,
                         refine=p.refine_ik)
        qB = IK_leg_mesh(pB - torso_w, left=False, foot_R=RB @ R_FOOT_LEVEL,
                         refine=p.refine_ik)
        return qA, qB, torso_w


# ===========================================================================
#  KINEMATIK KURU-KOSU  (fizik yok; mantik + IK + arazi tutarliligi testi)
# ===========================================================================
if __name__ == "__main__":
    from terrain import Terrain
    from kinematic_mesh import leg_fk, ANKLE_CENTER_IN_FOOT, Q_MIN, Q_MAX

    np.set_printoptions(precision=4, suppress=True)
    ANKLE_H = 0.0497
    for rough in (0.0, 0.6, 1.0):
        terr = Terrain(rough=rough, start_y=HIP_Y)
        p = GaitParams()
        gait = RoughGait(p, terr, ANKLE_H)
        fp = Footprint()
        dl, df = fp.samples(6, 5)
        worst_pen = 0.0; worst_gap_stance = 0.0; worst_ik = 0.0; max_dq = 0.0
        max_reach = 0.0; at_limit = 0; max_knee = 0.0; min_knee = 9.0
        q_prev = None
        n_ticks = gait.stop_tick + 100
        for _ in range(n_ticks):
            # olcum = plan (ideal izleme) -> DCM duzeltmesi 0 olmali
            com = to_model(gait.state_l[0], gait.state_f[0], p.z_hip)
            vel = np.array([gait.state_l[1], -gait.state_f[1], 0.0])
            gait.tick(com, vel, 0.0, 0.0)
            qA, qB, torso_w = gait.joint_targets(p.z_hip)
            max_reach = max(max_reach, gait.reach_drop)
            for name, q, left in (("A", qA, True), ("B", qB, False)):
                if np.any(q <= Q_MIN + 1e-9) or np.any(q >= Q_MAX - 1e-9):
                    at_limit += 1
                max_knee = max(max_knee, q[3]); min_knee = min(min_knee, q[3])
                pf, R, _, _ = leg_fk(q, left)
                ank = torso_w + pf + R @ ANKLE_CENTER_IN_FOOT
                want, _ = gait.ankle_world(name)
                worst_ik = max(worst_ik, np.linalg.norm(ank - want))
                # taban ornek noktalari (dunya): ayak-yerel = R_FOOT_LEVEL^T * d
                pts = np.array([ank + R @ (R_FOOT_LEVEL.T @ np.array([x, -f, -ANKLE_H]))
                                for x, f in zip(dl, df)])
                gap = pts[:, 2] - terr.height(pts[:, 0], pts[:, 1])
                worst_pen = min(worst_pen, gap.min())
                if gait.feet[name].lift == 0.0 and gait.swing != name:
                    worst_gap_stance = max(worst_gap_stance, gap.min())
            q = np.concatenate([qA, qB])
            if q_prev is not None:
                max_dq = max(max_dq, np.max(np.abs(q - q_prev)))
            q_prev = q
        print(f"rough={rough:3.1f}: {n_ticks} adim | IK hatasi max {worst_ik*1000:.2f} mm | "
              f"taban-zemin girisi max {-worst_pen*1000:.2f} mm | "
              f"destek ayagi en yakin nokta boslugu max {worst_gap_stance*1000:.2f} mm | "
              f"eklem sicramasi max {math.degrees(max_dq):.2f} derece/adim | "
              f"diz {math.degrees(min_knee):.0f}..{math.degrees(max_knee):.0f} derece | "
              f"erisim korumasi max {max_reach*1000:.1f} mm | limitte {at_limit} | "
              f"DCM duz. {np.abs(gait.corr).max()*1000:.2f} mm | "
              f"ileri {gait.feet['A'].pos[1]:.3f} m")
