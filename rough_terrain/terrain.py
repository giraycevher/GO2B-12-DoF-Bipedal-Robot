"""
terrain.py
==========
Orman zemini benzeri engebeli arazi ureteci (MuJoCo heightfield) + yukseklik
sorgusu. MuJoCo GEREKTIRMEZ (sadece numpy + scipy); bu yuzden tek basina
test edilebilir:

    python terrain.py            # istatistikleri yazar, terrain_preview.png cizer

ARAZI NEDEN BU SEKILDE
----------------------
Robotun bacagi 17.5 cm. Insan olcegine (bacak ~90 cm) oranlarsak 1 cm'lik bir
tumsek, insan icin ~5 cm'lik bir kok/tas demek. Yani "orman zemini" bu robot
icin MILIMETRE mertebesinde puruz demektir:

    dalgalanma : genis, yumusak toprak tumsekleri   (std ~4 mm, dalga boyu ~15-25 cm)
    ince puruz : cakil / yaprak dokusu               (std ~0.8 mm)
    kokler     : yolu capraz kesen ince sirtlar      (yukseklik ~5 mm, genislik ~2 cm)
    taslar     : kucuk tumsekler ve cukurlar         (+-3..6 mm)

Hepsi `rough` carpani ile olceklenir (0 = duz zemin, 1 = varsayilan, 1.5 = zor).

KOORDINATLAR
------------
Dunya = robot_walk.xml'in model eksenleri: X = yanal, Y = ileri/geri, Z = yukari.
Robot model -Y yonune yurur. Arazi, robotun durdugu yerde DUZDUR (z = 0) ve
ileriye dogru yumusakca engebelesir; boylece cokme + ilk adim duz zeminde olur.
"""

import math
import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import gaussian_filter


class Terrain:
    def __init__(self, rough=1.0, seed=3, start_y=0.0677,
                 size_x=1.2, y_behind=0.40, y_ahead=1.80, res=0.005,
                 flat_ahead=0.10, blend_len=0.14):
        """
        rough      : puruzluluk carpani (0 = duz)
        seed       : rastgele tohum (ayni tohum = ayni arazi)
        start_y    : robotun ayak bileklerinin baslangictaki dunya Y'si (HIP_Y)
        size_x     : arazinin yanal genisligi [m]
        y_behind   : robotun arkasinda kalan uzunluk [m]
        y_ahead    : robotun onunde (model -Y) kalan uzunluk [m]
        res        : izgara cozunurlugu [m]
        flat_ahead : baslangictan itibaren ileriye dogru tam duz kalan mesafe [m]
        blend_len  : duzden engebeliye gecis uzunlugu [m]
        """
        self.rough = float(rough)
        self.seed = int(seed)
        self.res = float(res)
        self.start_y = float(start_y)

        self.x_min = -size_x / 2.0
        self.x_max = +size_x / 2.0
        self.y_max = start_y + y_behind          # arka kenar
        self.y_min = start_y - y_ahead           # on kenar (robot buraya dogru yurur)

        self.ncol = int(round((self.x_max - self.x_min) / res)) + 1   # X boyunca
        self.nrow = int(round((self.y_max - self.y_min) / res)) + 1   # Y boyunca
        xs = self.x_min + np.arange(self.ncol) * res
        ys = self.y_min + np.arange(self.nrow) * res
        X, Y = np.meshgrid(xs, ys)               # sekil (nrow, ncol): [satir=Y, sutun=X]

        rng = np.random.default_rng(self.seed)
        h = np.zeros((self.nrow, self.ncol))

        if self.rough > 0.0:
            # 1) genis toprak dalgalanmasi
            u = gaussian_filter(rng.standard_normal(h.shape), sigma=0.060 / res, mode="reflect")
            h += 0.0040 * u / (u.std() + 1e-12)
            # 2) ince puruz
            f = gaussian_filter(rng.standard_normal(h.shape), sigma=0.012 / res, mode="reflect")
            h += 0.0008 * f / (f.std() + 1e-12)
            # 3) kokler: yolu capraz kesen dogru parcalari, Gauss kesitli sirtlar
            n_roots = 7
            for _ in range(n_roots):
                cx = rng.uniform(self.x_min * 0.6, self.x_max * 0.6)
                cy = rng.uniform(self.y_min + 0.1, start_y - flat_ahead - blend_len)
                ang = rng.uniform(-0.6, 0.6)                 # ~X eksenine yakin = yolu keser
                half = rng.uniform(0.15, 0.35)
                height = rng.uniform(0.0035, 0.0055)
                sig = rng.uniform(0.007, 0.011)
                dx, dy = math.cos(ang), math.sin(ang)
                t = np.clip((X - cx) * dx + (Y - cy) * dy, -half, half)
                d2 = (X - cx - t * dx) ** 2 + (Y - cy - t * dy) ** 2
                h += height * np.exp(-d2 / (2.0 * sig ** 2))
            # 4) taslar (+) ve cukurlar (-)
            n_stones = 45
            for _ in range(n_stones):
                cx = rng.uniform(self.x_min, self.x_max)
                cy = rng.uniform(self.y_min, start_y)
                height = rng.uniform(0.003, 0.006) * (1.0 if rng.random() < 0.7 else -1.0)
                sig = rng.uniform(0.010, 0.018)
                h += height * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sig ** 2))

            h *= self.rough

            # baslangic bolgesi duz: fwd = start_y - Y
            fwd = start_y - Y
            s = np.clip((fwd - flat_ahead) / blend_len, 0.0, 1.0)
            mask = s * s * (3.0 - 2.0 * s)                   # smoothstep
            # duz bolgenin seviyesi 0 olsun diye, gecis bolgesindeki ortalamayi cikar
            h = (h - float(np.mean(h))) * mask

        self.h = h
        self.h_min = float(h.min())
        self.h_max = float(h.max())
        self.elev = max(self.h_max - self.h_min, 1e-4)

    # ------------------------------------------------------------------ sorgu
    def height(self, X, Y):
        """(X, Y) dunya noktasindaki zemin yuksekligi [m]. Dizi de kabul eder.
        Izgara disinda kenar degeri kullanilir. Cift-dogrusal enterpolasyon."""
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

    # ------------------------------------------------------------- MuJoCo tarafi
    def hfield_data(self):
        """MuJoCo model.hfield_data icin normalize (0..1) veri, satir-oncelikli.
        Satir 0 = en kucuk Y, sutun 0 = en kucuk X (MuJoCo'nun duzeni)."""
        return ((self.h - self.h_min) / self.elev).astype(np.float32).ravel()

    def write_xml(self, src_xml, dst_xml, friction=1.0, solref=0.008, soft_priority=True):
        """src_xml'deki duz 'floor' duzlemini heightfield ile degistirip dst_xml'e yazar.

        dst_xml, src_xml ile AYNI klasorde olmali (meshdir='assets' goreli).
        friction : toprak surtunmesi. Ayak geom'lari 1.2; MuJoCo ikisinin
                   maksimumunu alir. soft_priority=True ise zemin oncelikli
                   olur ve BU deger gecerli olur (toprak daha kaygan/yumusak).
        solref   : temas zaman sabiti [s]; buyuk = daha yumusak zemin.
                   (timestep'in en az 2 kati olmali.)
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
        """Yuklenmis MjModel'e yukseklik verisini yazar (viewer/renderer ACILMADAN once)."""
        if model.nhfield < 1:
            raise RuntimeError("Modelde hfield yok; write_xml ile uretilen XML'i yukle.")
        n = int(model.hfield_nrow[0]) * int(model.hfield_ncol[0])
        d = self.hfield_data()
        if n != d.size:
            raise RuntimeError(f"hfield boyutu uyusmuyor: model {n}, arazi {d.size}")
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


if __name__ == "__main__":
    for r in (0.5, 1.0, 1.5):
        t = Terrain(rough=r)
        s = t.stats()
        print(f"rough={r:3.1f}: izgara {s['nrow']}x{s['ncol']}  yukseklik "
              f"[{s['min_mm']:+5.1f}, {s['max_mm']:+5.1f}] mm  std {s['std_mm']:.2f} mm  "
              f"egim ort {s['slope_mean_deg']:.1f} / %99 {s['slope_p99_deg']:.1f} derece")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = Terrain(rough=1.0)
        fig, ax = plt.subplots(figsize=(6, 9))
        im = ax.imshow(t.h * 1000, origin="lower", cmap="terrain",
                       extent=[t.x_min, t.x_max, t.y_min, t.y_max])
        ax.plot([0], [t.start_y], "r^", ms=10)
        ax.annotate("robot (model -Y'ye yurur)", (0, t.start_y), (0.1, t.start_y + 0.15), color="r")
        ax.set_xlabel("model X (yanal) [m]"); ax.set_ylabel("model Y [m]")
        plt.colorbar(im, label="yukseklik [mm]")
        plt.tight_layout(); plt.savefig("terrain_preview.png", dpi=110)
        print("grafik: terrain_preview.png")
    except ImportError:
        pass
