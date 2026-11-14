#!/usr/bin/env python3
"""
tools/run_terrain_tests.py
==========================
Run main_terrain.py headless for a set of experiments and print a results table.

    python tools/run_terrain_tests.py                       # default experiment set
    python tools/run_terrain_tests.py --jobs 4              # parallel runs
    python tools/run_terrain_tests.py "--rough 1.0 --seed 7" "--rough 1.5"
    python tools/run_terrain_tests.py --common "--ankle"    # add flags to every case
"""
import argparse
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAIN = Path(__file__).resolve().parent.parent / "main_terrain.py"

DEFAULT = [
    "--rough 0", "--rough 0.3", "--rough 0.6", "--rough 1.0",
    "--rough 1.0 --seed 1", "--rough 1.0 --seed 2", "--rough 1.0 --seed 5",
    "--rough 1.0 --seed 7", "--rough 1.0 --seed 11",
    "--rough 1.5", "--rough 1.5 --seed 7", "--rough 1.5 --seed 11", "--rough 2.0",
    "--rough 1.0 --blind", "--rough 1.0 --no-slope",
    "--rough 0.6 --push 8", "--rough 0.6 --push -8", "--rough 0.6 --push 6 --push-axis fwd",
]


def run(argstr: str):
    cmd = [sys.executable, str(MAIN), "--headless", "--no-plot"] + argstr.split()
    out = subprocess.run(cmd, capture_output=True, text=True).stdout

    def g(pat):
        m = re.search(pat, out)
        return m.group(1) if m else "?"
    return dict(case=argstr, exit=g(r"exit reason  : (.*)"),
                dist=g(r"distance     : ([+\-\d.]+) m"), plan=g(r"plan ([\d.]+) m"),
                lat=g(r"lateral drift: ([+\-\d.]+) m"), td=g(r"early touchdowns: (\d+)"),
                tilt=g(r"torso tilt   : ([\d.]+ / [\d.]+) deg"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--common", default="", help="arguments added to every case")
    ap.add_argument("cases", nargs="*", help="argument strings, one per experiment (quoted)")
    a = ap.parse_args()
    cases = [(c + " " + a.common).strip() for c in (a.cases or DEFAULT)]
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(run, cases))
    print(f"{'experiment':44s} {'result':8s} {'distance/plan':>15s} {'lateral':>8s} "
          f"{'early td':>8s}  max roll / pitch [deg]")
    n_ok = 0
    for r in res:
        ok = r["exit"] == "time limit reached"
        n_ok += ok
        res_txt = "WALKED" if ok else ("FELL" if "fell" in r["exit"] else r["exit"][:8])
        print(f"{r['case']:44s} {res_txt:8s} {r['dist']:>7s}/{r['plan']:<7s} {r['lat']:>8s} "
              f"{r['td']:>8s}  {r['tilt']}")
    print(f"\n{n_ok}/{len(res)} walked")


if __name__ == "__main__":
    main()
