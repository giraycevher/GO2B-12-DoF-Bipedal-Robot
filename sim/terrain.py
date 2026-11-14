"""
sim/terrain.py
==============
Procedural forest-floor terrain for MuJoCo (heightfield) plus a height query
that the controller uses. The generator itself needs only numpy + scipy;
`verify()` is the only part that calls MuJoCo.

Why the terrain looks like this
-------------------------------
The leg is 17.5 cm long. Scaled to a human leg (~90 cm), a 1 cm bump on this
robot is a ~5 cm root or stone for a person, so "forest floor" means bumps of a
few millimetres here:

    undulation : broad soil waves         (std ~4 mm, wavelength ~15-25 cm)
    fine       : gravel / leaf texture    (std ~0.8 mm)
    roots      : thin ridges across the path (height ~3.5-5.5 mm, width ~2 cm)
    stones     : small bumps and pits     (+-3..6 mm, 30 % are pits)

Everything is multiplied by `rough` (0 = flat, 1 = default-hard, 1.5 = very hard).

Coordinates
-----------
World = model axes of robot_walk.xml: X = lateral, Y = backward/forward,
Z = up. The robot walks towards model -Y. The terrain is FLAT (z = 0) where the
robot starts and blends smoothly into the rough region ahead, so the squat and
the first step happen on flat ground.
"""

import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import gaussian_filter


class Terrain:
    def __init__(self, rough=1.0, seed=3, start_y=0.0677,
                 size_x=1.2, y_behind=0.40, y_ahead=1.80, res=0.005,
                 flat_ahead=0.10, blend_len=0.14):
        """
        rough      : roughness multiplier (0 = flat)
        seed       : random seed (same seed = same terrain)
        start_y    : world Y of the ankles at the start (HIP_Y)
        size_x     : lateral width of the terrain [m]
        y_behind   : length behind the robot [m]
        y_ahead    : length in front of the robot (model -Y) [m]
        res        : grid resolution [m]
        flat_ahead : flat distance in front of the start [m]
        blend_len  : flat -> rough transition length [m]
        """
        self.rough = float(rough)
        self.seed = int(seed)
        self.res = float(res)
        self.start_y = float(start_y)

        self.x_min = -size_x / 2.0
        self.x_max = +size_x / 2.0
        self.y_max = start_y + y_behind          # back edge
        self.y_min = start_y - y_ahead           # front edge (the robot walks this way)

        self.ncol = int(round((self.x_max - self.x_min) / res)) + 1   # X boyunca
        self.nrow = int(round((self.y_max - self.y_min) / res)) + 1   # Y boyunca
        xs = self.x_min + np.arange(self.ncol) * res
        ys = self.y_min + np.arange(self.nrow) * res
        X, Y = np.meshgrid(xs, ys)               # shape (nrow, ncol): [row = Y, col = X]

        rng = np.random.default_rng(self.seed)
        h = np.zeros((self.nrow, self.ncol))

        if self.rough > 0.0:
            # 1) broad soil undulation
            u = gaussian_filter(rng.standard_normal(h.shape), sigma=0.060 / res, mode="reflect")
            h += 0.0040 * u / (u.std() + 1e-12)
            # 2) fine texture
            f = gaussian_filter(rng.standard_normal(h.shape), sigma=0.012 / res, mode="reflect")
            h += 0.0008 * f / (f.std() + 1e-12)
            # 3) roots: line segments across the path, Gaussian-profile ridges
            n_roots = 7
            for _ in range(n_roots):
                cx = rng.uniform(self.x_min * 0.6, self.x_max * 0.6)
                cy = rng.uniform(self.y_min + 0.1, start_y - flat_ahead - blend_len)
                ang = rng.uniform(-0.6, 0.6)                 # close to the X axis = crosses the path
                half = rng.uniform(0.15, 0.35)
                height = rng.uniform(0.0035, 0.0055)
                sig = rng.uniform(0.007, 0.011)
                dx, dy = math.cos(ang), math.sin(ang)
                t = np.clip((X - cx) * dx + (Y - cy) * dy, -half, half)
                d2 = (X - cx - t * dx) ** 2 + (Y - cy - t * dy) ** 2
                h += height * np.exp(-d2 / (2.0 * sig ** 2))
            # 4) stones (+) and pits (-)
            n_stones = 45
            for _ in range(n_stones):
                cx = rng.uniform(self.x_min, self.x_max)
                cy = rng.uniform(self.y_min, start_y)
                height = rng.uniform(0.003, 0.006) * (1.0 if rng.random() < 0.7 else -1.0)
                sig = rng.uniform(0.010, 0.018)
                h += height * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sig ** 2))

            h *= self.rough

            # flat start zone: fwd = start_y - Y
            fwd = start_y - Y
            s = np.clip((fwd - flat_ahead) / blend_len, 0.0, 1.0)
            mask = s * s * (3.0 - 2.0 * s)                   # smoothstep
            # remove the mean so that the flat zone sits at z = 0
            h = (h - float(np.mean(h))) * mask

        self.h = h
        self.h_min = float(h.min())
        self.h_max = float(h.max())
        self.elev = max(self.h_max - self.h_min, 1e-4)

    @classmethod
    def from_config(cls, cfg, start_y):
        """Build from a config.parameters.TerrainConfig."""
        return cls(rough=cfg.rough, seed=cfg.seed, start_y=start_y,
                   size_x=cfg.size_x, y_behind=cfg.behind, y_ahead=cfg.ahead,
                   res=cfg.resolution, flat_ahead=cfg.flat_ahead,
                   blend_len=cfg.blend_len)

    def verify(self, model, data, n_rays=12, seed=0):
        """Ray-cast the MuJoCo heightfield and compare with height().

        Returns the worst absolute mismatch [m], or None if no ray hit the
        ground. A mismatch means the controller would see a different terrain
        than the simulator."""
        import mujoco
        worst = 0.0
        gid = np.zeros(1, dtype=np.int32)
        rng = np.random.default_rng(seed)
        for _ in range(n_rays):
            x = rng.uniform(-0.3, 0.3)
            y = self.start_y - rng.uniform(0.45, 1.2)
            d = mujoco.mj_ray(model, data, np.array([x, y, 1.0]), np.array([0.0, 0.0, -1.0]),
                              None, 1, -1, gid)
            if d < 0:
                return None
            worst = max(worst, abs((1.0 - d) - float(self.height(x, y))))
        return worst

    # ------------------------------------------------------------------ query
    def height(self, X, Y):
        """Ground height [m] at world point (X, Y); arrays are accepted.
        Outside the grid the edge value is used. Bilinear interpolation."""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        fx = np.clip((X - self.x_min) / self.res, 0.0, self.ncol - 1.000001)
        fy = np.clip((Y - self.y_min) / self.res, 0.0, self.nrow - 1.000001)
        c0 = np.floor(fx).astype(int)
        r0 = np.floor(fy).astype(int)
        tx = fx - c0
        ty = fy - r0
        h = self.h
        v = (h[r0, c0] * (1 - tx) * (1 - ty) + h[r0, c0 + 1] * tx * (1 - ty)
             + h[r0 + 1, c0] * (1 - tx) * ty + h[r0 + 1, c0 + 1] * tx * ty)
        return v

    # ------------------------------------------------------------- MuJoCo side
    def hfield_data(self):
        """Normalized (0..1) data for model.hfield_data, row-major.
        Row 0 = smallest Y, column 0 = smallest X (MuJoCo's layout)."""
        return ((self.h - self.h_min) / self.elev).astype(np.float32).ravel()

    def write_xml(self, src_xml, dst_xml, friction=1.0, solref=0.008, soft_priority=True):
        """Replace the flat 'floor' plane of src_xml by a heightfield, write dst_xml.

        dst_xml must be in the SAME folder as src_xml (meshdir='assets' is relative).
        friction : soil friction. The foot geoms use 1.2 and MuJoCo would take
                   the maximum; with soft_priority=True the floor has priority
                   and THIS value is used.
        solref   : contact time constant [s]; larger = softer ground
                   (must be at least 2 x timestep).
        """
        tree = ET.parse(src_xml)
        root = tree.getroot()
        floor = None
        for g in root.iter("geom"):
            if g.get("name") == "floor":
                floor = g
                break
        if floor is None:
            raise RuntimeError("XML'de name='floor' olan geom yok.")

        # The parked push ball (main.py --push-viz ball) would fall forever
        # outside the finite heightfield; the terrain scenes do not use it.
        for wb in root.iter("worldbody"):
            for body in list(wb):
                if body.tag == "body" and body.get("name") == "push_ball":
                    wb.remove(body)

        asset = ET.SubElement(root, "asset")
        cx = 0.5 * (self.x_min + self.x_max)
        cy = 0.5 * (self.y_min + self.y_max)
        rx = 0.5 * (self.x_max - self.x_min)
        ry = 0.5 * (self.y_max - self.y_min)
        ET.SubElement(asset, "hfield", {
            "name": "soil", "nrow": str(self.nrow), "ncol": str(self.ncol),
            "size": f"{rx:.6f} {ry:.6f} {self.elev:.6f} 0.05"})

        for k in ("size", "material"):
            if k in floor.attrib:
                del floor.attrib[k]
        floor.set("type", "hfield")
        floor.set("hfield", "soil")
        floor.set("pos", f"{cx:.6f} {cy:.6f} {self.h_min:.6f}")
        floor.set("rgba", "0.36 0.26 0.17 1")
        floor.set("friction", f"{friction:g} 0.05 0.001")
        floor.set("condim", "3")
        floor.set("solref", f"{solref:g} 1")
        if soft_priority:
            floor.set("priority", "1")
        tree.write(dst_xml, encoding="utf-8", xml_declaration=True)
        return dst_xml

    def apply_to_model(self, model):
        """Write the height data into a loaded MjModel (BEFORE opening a viewer/renderer)."""
        if model.nhfield < 1:
            raise RuntimeError("Model has no hfield; load the XML produced by write_xml().")
        n = int(model.hfield_nrow[0]) * int(model.hfield_ncol[0])
        d = self.hfield_data()
        if n != d.size:
            raise RuntimeError(f"hfield size mismatch: model {n}, terrain {d.size}")
        adr = int(model.hfield_adr[0])
        model.hfield_data[adr:adr + n] = d

    def stats(self):
        rough_part = self.h[np.abs(self.h) > 1e-9]
        gy, gx = np.gradient(self.h, self.res)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        return {
            "nrow": self.nrow, "ncol": self.ncol,
            "min_mm": self.h_min * 1000, "max_mm": self.h_max * 1000,
            "std_mm": float(rough_part.std() * 1000) if rough_part.size else 0.0,
            "slope_mean_deg": float(slope[np.abs(self.h) > 1e-9].mean()) if rough_part.size else 0.0,
            "slope_p99_deg": float(np.percentile(slope, 99)),
        }
