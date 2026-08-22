#!/usr/bin/env python3
"""
fix_robot_xml.py
================
Onshape'ten ihrac edilmis  robot_final.xml  dosyasini alir ve MuJoCo'da
gercekten calisan  robot_walk.xml  dosyasini uretir.

    python fix_robot_xml.py                       # robot_final.xml -> robot_walk.xml
    python fix_robot_xml.py --in robot_final.xml --out robot_walk.xml --density 1240

STL'lere DOKUNMAZ; sadece XML'i duzeltir. Onshape'ten yeniden ihrac edersen
bu scripti tekrar calistirman yeterli.

DUZELTILEN HATALAR
------------------
 1. SAG BACAKTA KALCA YAW EKLEMI YOK.
    robot_final.xml'de 11 aktuator var (12 olmali). Sag bacak dogrudan
    hip-roll ile basliyor. Ayrica sag bacak sol bacaktan 49.2 mm YUKARIDAN
    basliyor -- yani iki bacak farkli uzunlukta, robot ayakta duramaz.
    -> Sag bacagin ustune, solun tam aynasi konumda, eksik yaw govdesi eklenir.
       (robot_final_fixed.xml'de bu elle eklenmis ama quat'i "0.5 0.5 0.5 -0.5"
        verilmis; o quat ekseni Z'den Y'ye ceviriyor, yani eklenen eklem yaw
        degil IKINCI BIR ROLL oluyor. Dogrusu quat="1 0 0 0".)

 2. TUM KUTLELER 1e-09 kg.
    Onshape ihracati yogunluk bulamayinca butun <inertial> etiketlerine
    mass=1e-09, inertia=1e-09 yazmis. Toplam robot kutlesi ~1e-8 kg.
    Boyle bir modelde en ufak kuvvet sonsuz ivme uretir; mj_step ilk
    adimlarda NaN verir.
    -> Sahte <inertial> etiketleri silinir; geom'lara density verilir ve
       kutle/atalet tensoru MuJoCo tarafindan GERCEK STL hacimlerinden
       hesaplanir.

 3. HER MESH IKI KEZ EKLENMIS (class="visual" + class="collision").
    Bu hem kutleyi ikiye katlar hem de govde basina ~40 adet convex hull
    carpisma cismi yaratir. Diz/kalca komsu olmayan govdeler arasindaki
    self-collision devasa temas kuvvetleri uretir -> robot patlar.
    -> Butun class="collision" kopyalari silinir. Carpisma sadece AYAK
       TABANLARINDA (lsp1..lsp4 + foot plakasi) birakilir.

 4. <option> YOK.
    Varsayilan timestep=0.002 + Euler integrator, kp=300'luk position
    aktuatorlerle kararsiz.
    -> timestep=0.001, integrator=implicitfast eklenir.

 5. AKTUATOR AYARLARI.
    kp=300 / dampratio=1, ctrlrange yok. Gercek kutle ~1.5 kg olan bir robotta
    kp=300 asiri sert; ayrica ctrl sinirsiz oldugu icin IK bir kez patlarsa
    komut da patlar.
    -> kp/kv eklem bazinda, ctrlrange eklem limitine esit, forcerange eklenir.

 6. DIZ EKLEM LIMITI TERS.
    "Revolute 4" range="-0.1 2.5" dogru ama ayni eksende olan hip_pitch ve
    ankle_pitch'in isaret duzeni farkli. Eklem limitleri IK'yi limite
    dayiyordu (hip_yaw +-0.5, ankle_roll +-0.6 cok dar).
    -> Limitler kinematic_mesh.py'deki Q_MIN/Q_MAX ile ayni yapilir.

 7. main.py "torso" isimli govdeyi ariyor, boyle bir govde YOK.
    mj_name2id -1 donuyor, data.xfrc_applied[-1] sessizce SON govdeye
    (bir ayak) kuvvet uyguluyor.
    -> Govdeye name="torso" verilir (eski isim gerekiyorsa site olarak durur)
       ve CoM sensoru bir site'a baglanir.

 8. BASLANGIC YUKSEKLIGI.
    pos="0 0 0.35" ama duz bacakta taban govde orijininin sadece 0.275 m
    altinda. Robot 7.5 cm yukseklikten yere carpiyor.
    -> Ayaklar yere degecek sekilde ayarlanir.

 9. class ismi "C:\\Users\\ozmof\\..." (Windows yolu).
    -> "biped" olarak yeniden adlandirilir.

10. Zemin/isik iki ayri <worldbody> blogunda; calisir ama tek blokta
    toplanir. Ayrica zemine surtunme ve solref/solimp verilir.
"""

import argparse
import copy
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.parameters import RobotConfig

_CFG = RobotConfig()

JOINT_LIMITS = _CFG.joint_limits
# Servo basina kazanc.
# DIKKAT: ilk surumdeki degerler (kp=35) ~1.5 kg'lik bir robot varsayiyordu.
# density=1240 ile gercek kutle 2.99 kg cikti ve robot her karede 24 mm
# sarkiyordu (govde 0.265 -> 0.241 m). Cokuk duruşta dizin tasimasi gereken
# moment ~ m*g*L = 3.0*9.81*0.09 = 2.6 Nm; kp=35 ile bunu tutmak 0.075 rad
# aci hatasi demek -- her eklemde birikince govde asagi kaciyor ve IK'nin
# planladigi yukseklikle gercek yukseklik ayrisiyor.
# Asagidaki degerler 3 kg icin sarkmayi ~2 mm'ye indirir.
GAINS = _CFG.actuator_gains

# robot_final.xml'deki eklem isimleri -> anlamli isim
LEFT_JOINTS = {
    "Revolute 1":     "left_hip_yaw",
    "Revolute 2":     "left_hip_roll",
    "Revolute 3":     "left_hip_pitch",
    "Revolute 4":     "left_knee",
    "Revolute 3 (1)": "left_ankle_pitch",
    "Revolute 2 (1)": "left_ankle_roll",
}
RIGHT_JOINTS = {
    "Revolute 2 (2)":     "right_hip_roll",
    "Revolute 3 (2)":     "right_hip_pitch",
    "Revolute 4 (1)":     "right_knee",
    "Revolute 3 (1) (1)": "right_ankle_pitch",
    "Revolute 2 (1) (1)": "right_ankle_roll",
}
KIND = {  # anlamli isim -> eklem tipi
    "hip_yaw": "hip_yaw", "hip_roll": "hip_roll", "hip_pitch": "hip_pitch",
    "knee": "knee", "ankle_pitch": "ankle_pitch", "ankle_roll": "ankle_roll",
}

# Sol bacagin kalca-yaw govdesi (robot_final.xml'den)
LEFT_HIP_YAW_POS = (-0.0804341, 0.0679133, -0.000988717)
# Aynalanmis sag konum (simetri duzlemi x = -0.0054341)
RIGHT_HIP_YAW_POS = (0.0695659, 0.0679133, -0.000988717)
# Sag hip-roll govdesinin, yeni yaw govdesine gore yerel konumu
# (= sol bacaktakiyle birebir ayni: 0, 0.0240365, -0.0492)
RIGHT_ROLL_LOCAL_POS = (0.0, 0.0240365, -0.0492)

LEFT_HIP_YAW_BODY = "left_m2_5_waser_1_1__govde2"
RIGHT_ROOT_BODY = "right_ankle_motor___roll_1__govde1_2"
FOOT_BODIES = ["left_foot_1__govde1", "left_foot_1__govde1_2"]
FOOT_RENAME = {"left_foot_1__govde1": "left_foot", "left_foot_1__govde1_2": "right_foot"}
BASE_BODY = "base_link_1__govde1"
# Ayak tabanini olusturan mesh'ler -- carpisma bunlarla yapilir
SOLE_MESHES = ("lsp1_1__govde1", "lsp2_1__govde1", "lsp3_1__govde1",
               "lsp4_1__govde1", "left_foot_1__govde1")

CLASS_NAME = "biped"


def jkind(nice_name):
    return nice_name.replace("left_", "").replace("right_", "")


def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = i + "  "
        for e in elem:
            indent(e, level + 1)
        if not (e.tail or "").strip():
            e.tail = i
    if level and not (elem.tail or "").strip():
        elem.tail = i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="robot_final.xml")
    ap.add_argument("--out", dest="dst", default=_CFG.xml_path)
    ap.add_argument("--density", type=float, default=_CFG.material_density,
                    help="Baski malzemesi yogunlugu kg/m^3 (PLA 1240, PETG 1270, "
                         "ABS 1040, Alu 2700). Motorlar plastikten agir oldugu icin "
                         "istersen 1600-1800 dene.")
    ap.add_argument("--show-markers", action="store_true",
                    help="teshis isaretcilerini (imu_site yesil kure, ayak temas "
                         "kutulari) gorunur yap. Varsayilan: gorunmez.")
    ap.add_argument("--base-z", type=float, default=None,
                    help="Govde baslangic yuksekligi. Verilmezse otomatik.")
    args = ap.parse_args()

    try:
        tree = ET.parse(args.src)
    except FileNotFoundError:
        sys.exit(f"HATA: {args.src} bulunamadi. --in ile yolu ver.")
    root = tree.getroot()

    parent = {c: p for p in root.iter() for c in p}
    bodies = {b.get("name"): b for b in root.iter("body") if b.get("name")}
    changes = []

    # ---------------------------------------------------------------- 0. compiler
    comp = root.find("compiler")
    if comp is None:
        comp = ET.Element("compiler")
        root.insert(0, comp)
    comp.set("angle", "radian")
    comp.set("meshdir", comp.get("meshdir", "assets"))
    comp.set("autolimits", "true")
    comp.set("balanceinertia", "true")     # bozuk mesh atalet tensorlerini duzeltir
    comp.set("inertiafromgeom", "auto")    # <inertial> yoksa geom'dan hesapla

    # ---------------------------------------------------------------- 1. <option>
    if root.find("option") is None:
        opt = ET.Element("option")
        opt.set("timestep", "0.001")
        opt.set("integrator", "implicitfast")
        opt.set("gravity", "0 0 -9.81")
        opt.set("cone", "elliptic")
        opt.set("impratio", "10")
        root.insert(1, opt)
        changes.append("<option timestep=0.001 integrator=implicitfast> eklendi")

    # ---------------------------------------------------------------- 2. default sinifi
    for d in root.iter("default"):
        if d.get("class", "").startswith("C:") or "\\" in (d.get("class") or ""):
            d.set("class", CLASS_NAME)
            changes.append("Windows yolu olan default class adi -> 'biped'")
    for e in root.iter():
        if e.get("childclass", "").startswith("C:") or "\\" in (e.get("childclass") or ""):
            e.set("childclass", CLASS_NAME)
        if e.tag == "position" and ("\\" in (e.get("class") or "")):
            e.set("class", CLASS_NAME)

    # default/joint + visual geom density
    for d in root.iter("default"):
        if d.get("class") == CLASS_NAME:
            j = d.find("joint")
            if j is None:
                j = ET.SubElement(d, "joint")
            j.set("armature", "0.01")
            j.set("damping", "0.6")
            j.set("frictionloss", "0.05")
            j.set("limited", "true")
            for sub in d.iter("default"):
                if sub.get("class") == "visual":
                    g = sub.find("geom")
                    if g is None:
                        g = ET.SubElement(sub, "geom")
                    g.set("type", "mesh")
                    g.set("contype", "0")
                    g.set("conaffinity", "0")
                    g.set("group", "2")
                    g.set("density", str(args.density))
                if sub.get("class") == "collision":
                    g = sub.find("geom")
                    if g is None:
                        g = ET.SubElement(sub, "geom")
                    g.set("type", "mesh")
                    g.set("group", "3")
                    g.set("contype", "1")
                    g.set("conaffinity", "1")
                    g.set("condim", "3")
                    g.set("friction", "1.2 0.05 0.001")
                    g.set("mass", "0")         # kutleyi visual kopyasi tasiyor
                    g.set("solref", "0.005 1")
                    g.set("rgba", "0.9 0.3 0.2 0.35")

    # ---------------------------------------------------------------- 3. sahte inertial'lari sil
    n_inert = 0
    for b in list(root.iter("body")):
        for ine in b.findall("inertial"):
            b.remove(ine)
            n_inert += 1
    if n_inert:
        changes.append(f"{n_inert} adet sahte <inertial> (mass=1e-09) silindi -> "
                       f"kutle/atalet STL'lerden density={args.density} ile hesaplanacak")

    # ---------------------------------------------------------------- 4. collision kopyalarini sil
    n_col = 0
    for b in list(root.iter("body")):
        keep_soles = b.get("name") in FOOT_BODIES
        for g in b.findall("geom"):
            if g.get("class") == "collision":
                if keep_soles and g.get("mesh") in SOLE_MESHES:
                    continue                     # ayak tabani -> kalsin
                b.remove(g)
                n_col += 1
    changes.append(f"{n_col} adet gereksiz collision mesh silindi "
                   f"(carpisma sadece ayak tabanlarinda)")

    # ---------------------------------------------------------------- 5. eksik sag kalca yaw
    rroot = bodies.get(RIGHT_ROOT_BODY)
    if rroot is None:
        # robot_final_fixed.xml verilmisse elle eklenmis (yanlis) govde vardir
        rroot = bodies.get("right_hip_yaw_link__added")
        if rroot is not None:
            sys.exit("HATA: girdi olarak robot_final_fixed.xml verilmis gorunuyor.\n"
                     "       Lutfen HAM ihracat dosyasi robot_final.xml'i ver (--in robot_final.xml).")
        sys.exit(f"HATA: '{RIGHT_ROOT_BODY}' govdesi bulunamadi; XML beklenen "
                 f"Onshape ihracati degil.")

    base = bodies[BASE_BODY]
    idx = list(base).index(rroot)
    yaw_body = ET.Element("body")
    yaw_body.set("name", "right_hip_yaw_link")
    yaw_body.set("pos", "%g %g %g" % RIGHT_HIP_YAW_POS)
    yaw_body.set("quat", "1 0 0 0")                    # <-- KRITIK: identity
    yj = ET.SubElement(yaw_body, "joint")
    yj.set("name", "right_hip_yaw")
    yj.set("type", "hinge")
    yj.set("axis", "0 0 1")

    # Bu govdenin kendi mesh'i yok -> kutlesi 0 olur ve MuJoCo
    #   "mass and inertia of moving bodies must be larger than mjMINVAL"
    # hatasi verir. Sol bacaktaki karsiligi (kalca-yaw govdesi) bir waser + bir
    # bracket tasiyor; ihracat sag bacagi AYNALAMAYIP OTELEDIGI icin ayni
    # geom'lari birebir kopyalayabiliriz. Boylece hem kutle dogru olur hem de
    # sag bacakta eksik gorunen parcalar yerine gelir.
    left_yaw = bodies.get(LEFT_HIP_YAW_BODY)
    n_copied = 0
    if left_yaw is not None:
        for g in left_yaw.findall("geom"):
            yaw_body.append(copy.deepcopy(g))
            n_copied += 1
    if n_copied == 0:
        ine = ET.SubElement(yaw_body, "inertial")
        ine.set("pos", "0 0 0")
        ine.set("mass", "0.02")
        ine.set("diaginertia", "1e-05 1e-05 1e-05")
    changes.append(f"right_hip_yaw_link govdesine sol bacaktaki {n_copied} geom "
                   f"kopyalandi (kutlesi 0 kalmasin diye)")

    base.remove(rroot)
    rroot.set("pos", "%g %g %g" % RIGHT_ROLL_LOCAL_POS)  # quat degismez
    yaw_body.append(rroot)
    base.insert(idx, yaw_body)
    changes.append("Eksik SAG KALCA YAW eklemi eklendi (quat=identity) ve sag bacak "
                   "sol bacakla ayni yuksekligie hizalandi (49.2 mm fark giderildi)")

    # ---------------------------------------------------------------- 6. eklemleri isimlendir + limit
    rename = dict(LEFT_JOINTS)
    rename.update(RIGHT_JOINTS)
    order = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
             "left_ankle_pitch", "left_ankle_roll",
             "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee",
             "right_ankle_pitch", "right_ankle_roll"]
    found = {}
    # DIKKAT: <default> icindeki <joint> sablonunun ismi yoktur, onu atla.
    for j in root.find("worldbody").iter("joint"):
        old = j.get("name")
        if old is None:
            continue
        new = rename.get(old, old)
        j.set("name", new)
        k = jkind(new)
        if k in JOINT_LIMITS:
            lo, hi = JOINT_LIMITS[k]
            j.set("range", f"{lo} {hi}")
            j.set("limited", "true")
        found[new] = j
    missing = [n for n in order if n not in found]
    if missing:
        sys.exit(f"HATA: su eklemler bulunamadi: {missing}")
    changes.append("12 eklem anlamli isimlerle yeniden adlandirildi ve limitleri "
                   "IK ile uyumlu hale getirildi")

    # ---------------------------------------------------------------- 7. gövde isimleri
    base.set("name", "torso")
    for old, new in FOOT_RENAME.items():
        if old in bodies:
            bodies[old].set("name", new)
    changes.append("base_link -> 'torso', ayaklar -> 'left_foot'/'right_foot' "
                   "(main.py'nin aradigi isimler)")

    # ---------------------------------------------------------------- 8. ayak siteleri
    for fb_old, fb_new in FOOT_RENAME.items():
        fb = bodies.get(fb_old)
        if fb is None:
            continue
        s = ET.SubElement(fb, "site")
        s.set("name", fb_new + "_site")
        # Ayagin YEREL cercevesi: +X yukari, +Y yanal, +Z ileri.
        # Taban pedleri (lsp1..4) yerel x = -0.0497'de, y = +-0.042,
        # z = -0.074 .. +0.026 araliginda. Touch sensoru bu bolgeyi kapsasin.
        s.set("type", "box")
        s.set("pos", "-0.045 0 -0.024")
        s.set("size", "0.020 0.050 0.060")
        s.set("rgba", "1 0 0 %s" % ("0.25" if args.show_markers else "0"))
    # imu_site: govde CERCEVE ORIJINI (gercek kutle merkezi DEGIL).
    # com_pos / com_vel / torso_quat sensorleri buraya bagli; sadece
    # loglama/grafik icin. DCM geri beslemesi gercek CoM'u subtree_com'dan
    # okur, bu site'i kullanmaz.
    # Onshape ihracatinda base_link orijini govde blogunun geometrik
    # merkezinde degil, bu yuzden isaretci blogun onunde/altinda gorunur.
    ET.SubElement(base, "site").set("name", "imu_site")
    for s in base.findall("site"):
        if s.get("name") == "imu_site":
            s.set("pos", "0 0 0")
            s.set("size", "0.01")
            s.set("rgba", "0 1 0 %s" % ("0.5" if args.show_markers else "0"))

    # ---------------------------------------------------------------- 9. aktuatorler
    for a in root.findall("actuator"):
        root.remove(a)
    act = ET.SubElement(root, "actuator")
    for name in order:
        k = jkind(name)
        lo, hi = JOINT_LIMITS[k]
        kp, kv, fr = GAINS[k]
        p = ET.SubElement(act, "position")
        p.set("name", name)
        p.set("joint", name)
        p.set("kp", str(kp))
        p.set("kv", str(kv))
        p.set("ctrllimited", "true")
        p.set("ctrlrange", f"{lo} {hi}")
        p.set("forcelimited", "true")
        p.set("forcerange", f"{-fr} {fr}")
    changes.append("Aktuator blogu yeniden yazildi: 11 -> 12 aktuator, sabit sirada "
                   "(sol 6 + sag 6), kp/kv/ctrlrange/forcerange ile")

    # ---------------------------------------------------------------- 10. sensorler
    for s in root.findall("sensor"):
        root.remove(s)
    sen = ET.SubElement(root, "sensor")
    fp = ET.SubElement(sen, "framepos"); fp.set("name", "com_pos")
    fp.set("objtype", "site"); fp.set("objname", "imu_site")
    fq = ET.SubElement(sen, "framequat"); fq.set("name", "torso_quat")
    fq.set("objtype", "site"); fq.set("objname", "imu_site")
    fv = ET.SubElement(sen, "framelinvel"); fv.set("name", "com_vel")
    fv.set("objtype", "site"); fv.set("objname", "imu_site")
    for side in ("left", "right"):
        t = ET.SubElement(sen, "touch"); t.set("name", f"touch_{side}")
        t.set("site", f"{side}_foot_site")

    # ---------------------------------------------------------------- 11. contact excludes
    for c in root.findall("contact"):
        root.remove(c)
    con = ET.SubElement(root, "contact")
    for other in ("left_foot", "right_foot"):
        e = ET.SubElement(con, "exclude")
        e.set("body1", "torso"); e.set("body2", other)
    # NOT: left_foot <-> right_foot temasi BILEREK disarida birakilmadi.
    # Iki ayak birbirine carpabilmeli; gercek robotta da carpiyorlar.
    # Carpmasinlar diye duruş genisligi (main_walk.HALF_STANCE) 0.075 m,
    # yani ayaklar tam kalcalarin altinda. Ped yari genisligi 0.042 m
    # oldugu icin aralarinda 66 mm bosluk kalir.

    # ------------------------------------------- 11b. offscreen framebuffer
    # mujoco.Renderer ile video kaydi icin. Varsayilan 640x480'dir; daha
    # buyuk istenirse "offscreen framebuffer is smaller than..." hatasi verir.
    vis = root.find("visual")
    if vis is None:
        vis = ET.SubElement(root, "visual")
    g = vis.find("global")
    if g is None:
        g = ET.SubElement(vis, "global")
    g.set("offwidth", "1920")
    g.set("offheight", "1080")
    q = vis.find("quality")
    if q is None:
        q = ET.SubElement(vis, "quality")
    q.set("shadowsize", "4096")
    q.set("offsamples", "8")
    changes.append("offscreen framebuffer 1920x1080 yapildi (video kaydi icin)")

    # ---------------------------------------------------------------- 12. zemin/isik tek blokta
    wbs = root.findall("worldbody")
    main_wb = None
    for wb in wbs:
        if wb.find("body") is not None:
            main_wb = wb
            break
    if main_wb is None:
        main_wb = wbs[0]
    for wb in wbs:
        if wb is main_wb:
            continue
        for child in list(wb):
            wb.remove(child)
            main_wb.insert(0, child)
        root.remove(wb)
    for g in main_wb.findall("geom"):
        if g.get("type") == "plane":
            g.set("name", "floor")
            g.set("size", "0 0 0.05")
            g.set("friction", "1.2 0.05 0.001")
            g.set("condim", "3")

    # ---------------------------------------------------------------- 13. baslangic yuksekligi
    #  q=0'da ayak bilegi govde orijininin 0.22502 m altinda,
    #  taban ayak bileginin 0.0497 m altinda -> taban = 0.2747 m altta.
    #  Yaris baslangicinda bacaklar hafif bukuk olsun diye 0.24 kullaniyoruz;
    #  main_walk.py ilk 2 saniyede zaten cokup ayaga kalkiyor.
    base_z = args.base_z if args.base_z is not None else 0.2800
    base.set("pos", f"0 0 {base_z}")
    base.set("quat", "1 0 0 0")
    changes.append(f"Govde baslangic yuksekligi 0.35 -> {base_z} "
                   f"(duz bacakta ayak bilegi govde orijininin 0.225 m, taban ~0.275 m "
                   f"altinda). main_walk.py zaten basta ayaklari yere tam oturtuyor.")

    # ------------------------------------------------ 14b. itme topu (mermi)
    # main_walk.py --push-viz ball ile kullanilir: itmeyi xfrc_applied yerine
    # GERCEK bir carpisma olarak uygular. Normalde sahnenin uzaginda park eder
    # ve hicbir seye dokunmaz. Robotun alt agacinda DEGIL, worldbody'de --
    # yoksa robotun kutlesine/CoM'una karisir.
    ball = ET.SubElement(main_wb_placeholder := root.find("worldbody"), "body")
    ball.set("name", "push_ball")
    ball.set("pos", "8 8 0.03")
    ET.SubElement(ball, "freejoint").set("name", "push_ball_free")
    bg = ET.SubElement(ball, "geom")
    bg.set("name", "push_ball_geom")
    bg.set("type", "sphere")
    bg.set("size", "0.025")
    bg.set("density", "3000")
    bg.set("rgba", "0.95 0.25 0.15 1")
    bg.set("condim", "3")
    bg.set("contype", "1")
    bg.set("conaffinity", "1")
    changes.append("'push_ball' mermisi eklendi (sahnenin uzaginda park eder; "
                   "--push-viz ball ile firlatilir)")

    # ------------------------------------------- 15. kutlesiz govde kalmasin
    # MuJoCo, ekleme sahip her govdenin kutlesi > mjMINVAL olmasini ister.
    # Kutleyi visual (density'li) mesh geom'lari tasiyor; hic mesh'i olmayan
    # bir govde varsa ona kucuk ama gecerli bir inertial veriyoruz.
    n_fix = 0
    for b in root.find("worldbody").iter("body"):
        has_joint = (b.find("joint") is not None) or (b.find("freejoint") is not None)
        if not has_joint:
            continue
        bearing = [g for g in b.findall("geom")
                   if g.get("class") != "collision" and g.get("mass") != "0"]
        if bearing or b.find("inertial") is not None:
            continue
        ine = ET.SubElement(b, "inertial")
        ine.set("pos", "0 0 0")
        ine.set("mass", "0.02")
        ine.set("diaginertia", "1e-05 1e-05 1e-05")
        n_fix += 1
        print(f"  [uyari] '{b.get('name')}' govdesinin kutle tasiyan geom'u yok, "
              f"varsayilan inertial eklendi (mass=0.02 kg).")
    if n_fix:
        changes.append(f"{n_fix} kutlesiz hareketli govdeye varsayilan inertial eklendi "
                       f"(mjMINVAL hatasini onler)")

    # ---------------------------------------------------------------- yaz
    indent(root)
    tree.write(args.dst, encoding="utf-8", xml_declaration=True)

    print(f"\n  {args.src}  ->  {args.dst}\n")
    for i, c in enumerate(changes, 1):
        print(f"  {i:2d}. {c}")
    print(f"\n  Aktuator sirasi (data.ctrl indeksi):")
    for i, n in enumerate(order):
        print(f"       ctrl[{i:2d}] = {n}")
    print()


if __name__ == "__main__":
    main()
