#!/usr/bin/env python3
import argparse
import copy
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.parameters import RobotConfig

CFG = RobotConfig()
CLASS_NAME = "biped"

LEFT_JOINT_RENAME = {
    "Revolute 1": "left_hip_yaw",
    "Revolute 2": "left_hip_roll",
    "Revolute 3": "left_hip_pitch",
    "Revolute 4": "left_knee",
    "Revolute 3 (1)": "left_ankle_pitch",
    "Revolute 2 (1)": "left_ankle_roll",
}
RIGHT_JOINT_RENAME = {
    "Revolute 2 (2)": "right_hip_roll",
    "Revolute 3 (2)": "right_hip_pitch",
    "Revolute 4 (1)": "right_knee",
    "Revolute 3 (1) (1)": "right_ankle_pitch",
    "Revolute 2 (1) (1)": "right_ankle_roll",
}

RIGHT_HIP_YAW_POS = (0.0695659, 0.0679133, -0.000988717)
RIGHT_ROLL_LOCAL_POS = (0.0, 0.0240365, -0.0492)
LEFT_HIP_YAW_BODY = "left_m2_5_waser_1_1__govde2"
RIGHT_ROOT_BODY = "right_ankle_motor___roll_1__govde1_2"
BASE_BODY = "base_link_1__govde1"
FOOT_RENAME = {"left_foot_1__govde1": "left_foot",
               "left_foot_1__govde1_2": "right_foot"}
SOLE_MESHES = ("lsp1_1__govde1", "lsp2_1__govde1", "lsp3_1__govde1",
               "lsp4_1__govde1", "left_foot_1__govde1")


def joint_kind(name):
    return name.replace("left_", "").replace("right_", "")


def indent(elem, level=0):
    pad = "\n" + level * "  "
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "  "
        for child in elem:
            indent(child, level + 1)
        if not (child.tail or "").strip():
            child.tail = pad
    if level and not (elem.tail or "").strip():
        elem.tail = pad


def main():
    ap = argparse.ArgumentParser(
        description="Convert the raw Onshape export into a simulation-ready model")
    ap.add_argument("--in", dest="src", default="robot_final.xml")
    ap.add_argument("--out", dest="dst", default=CFG.xml_path)
    ap.add_argument("--density", type=float, default=CFG.material_density,
                    help="print material density kg/m^3 (PLA 1240, PETG 1270, ABS 1040)")
    ap.add_argument("--base-z", type=float, default=0.28)
    ap.add_argument("--show-markers", action="store_true")
    args = ap.parse_args()

    try:
        tree = ET.parse(args.src)
    except FileNotFoundError:
        sys.exit(f"ERROR: {args.src} not found.")
    root = tree.getroot()
    bodies = {b.get("name"): b for b in root.iter("body") if b.get("name")}
    changes = []

    comp = root.find("compiler")
    if comp is None:
        comp = ET.Element("compiler")
        root.insert(0, comp)
    comp.set("angle", "radian")
    comp.set("meshdir", comp.get("meshdir", "assets"))
    comp.set("autolimits", "true")
    comp.set("balanceinertia", "true")
    comp.set("inertiafromgeom", "auto")

    if root.find("option") is None:
        opt = ET.Element("option")
        opt.set("timestep", "0.001")
        opt.set("integrator", "implicitfast")
        opt.set("gravity", "0 0 -9.81")
        opt.set("cone", "elliptic")
        opt.set("impratio", "10")
        root.insert(1, opt)
        changes.append("added <option timestep=0.001 integrator=implicitfast>")

    for d in root.iter("default"):
        if "\\" in (d.get("class") or ""):
            d.set("class", CLASS_NAME)
            changes.append("renamed Windows-path default class to 'biped'")
    for e in root.iter():
        if "\\" in (e.get("childclass") or ""):
            e.set("childclass", CLASS_NAME)
        if e.tag == "position" and "\\" in (e.get("class") or ""):
            e.set("class", CLASS_NAME)

    for d in root.iter("default"):
        if d.get("class") != CLASS_NAME:
            continue
        j = d.find("joint") or ET.SubElement(d, "joint")
        j.set("armature", "0.01")
        j.set("damping", "0.6")
        j.set("frictionloss", "0.05")
        j.set("limited", "true")
        for sub in d.iter("default"):
            if sub.get("class") == "visual":
                gg = sub.find("geom") or ET.SubElement(sub, "geom")
                gg.set("type", "mesh")
                gg.set("contype", "0")
                gg.set("conaffinity", "0")
                gg.set("group", "2")
                gg.set("density", str(args.density))
            if sub.get("class") == "collision":
                gg = sub.find("geom") or ET.SubElement(sub, "geom")
                gg.set("type", "mesh")
                gg.set("group", "3")
                gg.set("contype", "1")
                gg.set("conaffinity", "1")
                gg.set("condim", "3")
                gg.set("friction", "1.2 0.05 0.001")
                gg.set("mass", "0")
                gg.set("solref", "0.005 1")
                gg.set("rgba", "0.9 0.3 0.2 0.35")

    n_inertial = 0
    for b in list(root.iter("body")):
        for ine in b.findall("inertial"):
            b.remove(ine)
            n_inertial += 1
    if n_inertial:
        changes.append(f"removed {n_inertial} placeholder <inertial> (mass=1e-09); "
                       f"mass now computed from STL volume at density={args.density}")

    n_col = 0
    for b in list(root.iter("body")):
        keep = b.get("name") in FOOT_RENAME
        for gg in b.findall("geom"):
            if gg.get("class") == "collision":
                if keep and gg.get("mesh") in SOLE_MESHES:
                    continue
                b.remove(gg)
                n_col += 1
    changes.append(f"removed {n_col} duplicate collision meshes "
                   f"(collision kept on foot soles only)")

    rroot = bodies.get(RIGHT_ROOT_BODY)
    if rroot is None:
        if bodies.get("right_hip_yaw_link__added") is not None:
            sys.exit("ERROR: input looks like robot_final_fixed.xml. "
                     "Pass the raw export robot_final.xml instead.")
        sys.exit(f"ERROR: body '{RIGHT_ROOT_BODY}' not found.")

    base = bodies[BASE_BODY]
    idx = list(base).index(rroot)
    yaw_body = ET.Element("body")
    yaw_body.set("name", "right_hip_yaw_link")
    yaw_body.set("pos", "%g %g %g" % RIGHT_HIP_YAW_POS)
    yaw_body.set("quat", "1 0 0 0")
    yj = ET.SubElement(yaw_body, "joint")
    yj.set("name", "right_hip_yaw")
    yj.set("type", "hinge")
    yj.set("axis", "0 0 1")

    left_yaw = bodies.get(LEFT_HIP_YAW_BODY)
    n_copied = 0
    if left_yaw is not None:
        for gg in left_yaw.findall("geom"):
            yaw_body.append(copy.deepcopy(gg))
            n_copied += 1
    if n_copied == 0:
        ine = ET.SubElement(yaw_body, "inertial")
        ine.set("pos", "0 0 0")
        ine.set("mass", "0.02")
        ine.set("diaginertia", "1e-05 1e-05 1e-05")
    changes.append(f"copied {n_copied} geoms into right_hip_yaw_link so it has mass")

    base.remove(rroot)
    rroot.set("pos", "%g %g %g" % RIGHT_ROLL_LOCAL_POS)
    yaw_body.append(rroot)
    base.insert(idx, yaw_body)
    changes.append("added missing right hip yaw joint (quat=identity) and aligned "
                   "the right leg with the left (49.2 mm offset removed)")

    rename = dict(LEFT_JOINT_RENAME)
    rename.update(RIGHT_JOINT_RENAME)
    found = {}
    for j in root.find("worldbody").iter("joint"):
        old = j.get("name")
        if old is None:
            continue
        new = rename.get(old, old)
        j.set("name", new)
        kind = joint_kind(new)
        if kind in CFG.joint_limits:
            lo, hi = CFG.joint_limits[kind]
            j.set("range", f"{lo} {hi}")
            j.set("limited", "true")
        found[new] = j
    missing = [n for n in CFG.joint_order if n not in found]
    if missing:
        sys.exit(f"ERROR: joints not found: {missing}")
    changes.append("renamed 12 joints and aligned limits with config.parameters")

    base.set("name", CFG.torso_body)
    for old, new in FOOT_RENAME.items():
        if old in bodies:
            bodies[old].set("name", new)
    changes.append("base_link -> 'torso', feet -> 'left_foot' / 'right_foot'")

    marker_alpha = "0.25" if args.show_markers else "0"
    for old, new in FOOT_RENAME.items():
        fb = bodies.get(old)
        if fb is None:
            continue
        s = ET.SubElement(fb, "site")
        s.set("name", new + "_site")
        s.set("type", "box")
        s.set("pos", "-0.045 0 -0.024")
        s.set("size", "0.020 0.050 0.060")
        s.set("rgba", f"1 0 0 {marker_alpha}")

    ET.SubElement(base, "site").set("name", "imu_site")
    for s in base.findall("site"):
        if s.get("name") == "imu_site":
            s.set("pos", "0 0 0")
            s.set("size", "0.01")
            s.set("rgba", "0 1 0 %s" % ("0.5" if args.show_markers else "0"))

    for a in root.findall("actuator"):
        root.remove(a)
    act = ET.SubElement(root, "actuator")
    for name in CFG.joint_order:
        kind = joint_kind(name)
        lo, hi = CFG.joint_limits[kind]
        kp, kv, fr = CFG.actuator_gains[kind]
        pos = ET.SubElement(act, "position")
        pos.set("name", name)
        pos.set("joint", name)
        pos.set("kp", str(kp))
        pos.set("kv", str(kv))
        pos.set("ctrllimited", "true")
        pos.set("ctrlrange", f"{lo} {hi}")
        pos.set("forcelimited", "true")
        pos.set("forcerange", f"{-fr} {fr}")
    changes.append("rewrote actuator block: 11 -> 12 actuators with kp/kv/ranges")

    for s in root.findall("sensor"):
        root.remove(s)
    sen = ET.SubElement(root, "sensor")
    for tag, name in (("framepos", "com_pos"), ("framequat", "torso_quat"),
                      ("framelinvel", "com_vel")):
        e = ET.SubElement(sen, tag)
        e.set("name", name)
        e.set("objtype", "site")
        e.set("objname", "imu_site")
    for side in ("left", "right"):
        e = ET.SubElement(sen, "touch")
        e.set("name", f"touch_{side}")
        e.set("site", f"{side}_foot_site")

    for c in root.findall("contact"):
        root.remove(c)
    con = ET.SubElement(root, "contact")
    for other in CFG.foot_bodies:
        e = ET.SubElement(con, "exclude")
        e.set("body1", CFG.torso_body)
        e.set("body2", other)

    vis = root.find("visual") or ET.SubElement(root, "visual")
    gl = vis.find("global") or ET.SubElement(vis, "global")
    gl.set("offwidth", "1920")
    gl.set("offheight", "1080")
    ql = vis.find("quality") or ET.SubElement(vis, "quality")
    ql.set("shadowsize", "4096")
    ql.set("offsamples", "8")
    changes.append("offscreen framebuffer set to 1920x1080 for video recording")

    worldbodies = root.findall("worldbody")
    main_wb = next((w for w in worldbodies if w.find("body") is not None), worldbodies[0])
    for wb in worldbodies:
        if wb is main_wb:
            continue
        for child in list(wb):
            wb.remove(child)
            main_wb.insert(0, child)
        root.remove(wb)
    for gg in main_wb.findall("geom"):
        if gg.get("type") == "plane":
            gg.set("name", "floor")
            gg.set("size", "0 0 0.05")
            gg.set("friction", "1.2 0.05 0.001")
            gg.set("condim", "3")

    ball = ET.SubElement(main_wb, "body")
    ball.set("name", CFG.ball_body)
    ball.set("pos", "8 8 0.03")
    ET.SubElement(ball, "freejoint").set("name", CFG.ball_joint)
    bg = ET.SubElement(ball, "geom")
    bg.set("name", "push_ball_geom")
    bg.set("type", "sphere")
    bg.set("size", "0.025")
    bg.set("density", "3000")
    bg.set("rgba", "0.95 0.25 0.15 1")
    bg.set("condim", "3")
    bg.set("contype", "1")
    bg.set("conaffinity", "1")
    changes.append("added 'push_ball' projectile parked away from the scene")

    base.set("pos", f"0 0 {args.base_z}")
    base.set("quat", "1 0 0 0")
    changes.append(f"initial base height set to {args.base_z}")

    n_fix = 0
    for b in root.find("worldbody").iter("body"):
        if b.find("joint") is None and b.find("freejoint") is None:
            continue
        bearing = [gg for gg in b.findall("geom")
                   if gg.get("class") != "collision" and gg.get("mass") != "0"]
        if bearing or b.find("inertial") is not None:
            continue
        ine = ET.SubElement(b, "inertial")
        ine.set("pos", "0 0 0")
        ine.set("mass", "0.02")
        ine.set("diaginertia", "1e-05 1e-05 1e-05")
        n_fix += 1
        print(f"  [warning] body '{b.get('name')}' had no mass-bearing geom, "
              f"default inertial added (mass=0.02 kg).")
    if n_fix:
        changes.append(f"added default inertial to {n_fix} massless moving bodies")

    indent(root)
    tree.write(args.dst, encoding="utf-8", xml_declaration=True)

    print(f"\n  {args.src}  ->  {args.dst}\n")
    for i, c in enumerate(changes, 1):
        print(f"  {i:2d}. {c}")
    print("\n  actuator order (data.ctrl index):")
    for i, n in enumerate(CFG.joint_order):
        print(f"       ctrl[{i:2d}] = {n}")
    print()


if __name__ == "__main__":
    main()
