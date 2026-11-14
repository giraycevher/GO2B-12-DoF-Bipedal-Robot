#!/usr/bin/env python3
"""run_tests.py -- main_walk_terrain.py'yi bir dizi ayarla penceresiz kosturur, tablo basar.

    python run_tests.py                 # varsayilan deney seti
    python run_tests.py --jobs 4        # paralel is sayisi
"""
import argparse, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

DEFAULT = [
    "--rough 0", "--rough 0.3", "--rough 0.6", "--rough 1.0", "--rough 1.5",
    "--rough 0.6 --seed 7", "--rough 1.0 --seed 7", "--rough 1.0 --seed 11",
    "--rough 1.0 --blind", "--rough 1.0 --no-slope", "--rough 1.0 --dcm", "--rough 1.5", "--rough 1.5 --seed 7",
    "--rough 0.6 --push 8", "--rough 0.6 --push -8", "--rough 0.6 --push 6 --push-axis fwd",
]

def run(argstr):
    cmd = [sys.executable, "main_walk_terrain.py", "--headless", "--no-plot"] + argstr.split()
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    g = lambda pat: (re.search(pat, out) or [None, "?"])[1]
    return (argstr, g(r"cikis nedeni : (.*)"), g(r"sim suresi   : ([\d.]+) s"),
            g(r"kat edilen   : ([+\-\d.]+) m"), g(r"plan ([\d.]+) m"),
            g(r"yanal kayma  : ([+\-\d.]+) m"), g(r"erken temas: (\d+)"),
            g(r"govde egimi  : (.*)"), g(r"IK hatasi    : (\d+)"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--common", default="", help="her deneye eklenecek ortak argumanlar")
    ap.add_argument("cases", nargs="*", help="deney arguman dizeleri (tirnak icinde)")
    a = ap.parse_args()
    cases = [(c + " " + a.common).strip() for c in (a.cases or DEFAULT)]
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(run, cases))
    print(f"{'deney':42s} {'sonuc':12s} {'sure':>6s} {'ileri/plan':>13s} {'yanal':>7s} {'erkT':>4s}  govde egimi maks (roll/pitch)")
    for r in res:
        ok = "YURUDU" if r[1] == "sure doldu" else "DUSTU" if "dustu" in r[1] else r[1][:12]
        print(f"{r[0]:42s} {ok:12s} {r[2]:>6s} {r[3]:>6s}/{r[4]:<6s} {r[5]:>7s} {r[6]:>4s}  {r[7]}")
