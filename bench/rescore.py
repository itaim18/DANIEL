"""Re-score HITL arms with the car's REAL footprint: an oriented 0.557 x 0.294 m rectangle (spec
length; 0.249 m track + 0.045 m tyres) centred on the root (mid-wheelbase), vs the 0.22 m circle the
sim's contact latch uses. Contact = any point of the rectangle (2 cm grid) inside a prop collider
(cylinder / oriented box / legs, the current manifest: valid for arms from p12 on).
    python bench/rescore.py ARM [ARM...]   (data: the hitl/ part of the Hugging Face dataset, runs.tar.zst unpacked,
                                            at $DANIEL_DATA/hitl, default ./data/hitl)"""
import glob, json, math, os, sys
from pathlib import Path
import numpy as np
DATA = Path(os.environ.get("DANIEL_DATA", Path(__file__).resolve().parents[1] / "data")) / "hitl"
man = json.loads((DATA / "props_manifest.json").read_text())
if os.environ.get("PERSON_LEGS"):
    for k, v in json.loads(Path(__file__).with_name("person_cells.json").read_text()).items():
        man[k] = dict(man[k], legs=v, leg_r=0.04)
S = {s["name"]: s for s in json.loads((DATA / "scenarios.json").read_text())["scenarios"]}
RUNS = DATA / "runs"
HL, HW = 0.557 / 2, 0.294 / 2
gx, gy = np.meshgrid(np.arange(-HL, HL + 1e-6, 0.02), np.arange(-HW, HW + 1e-6, 0.02))
CAR = np.stack([gx.ravel(), gy.ravel()], 1)


def inside(p, m, P):
    c, s = math.cos(p.get("yaw", 0.0)), math.sin(p.get("yaw", 0.0))
    dx, dy = P[:, 0] - p["x"], P[:, 1] - p["y"]
    lx, ly = c * dx + s * dy, -s * dx + c * dy                 # prop frame
    if "box" in m:
        return (np.abs(lx) <= m["box"][0] / 2) & (np.abs(ly) <= m["box"][1] / 2)
    if "legs" in m:
        L = np.array(m["legs"])
        return (np.hypot(lx[:, None] - L[None, :, 0], ly[:, None] - L[None, :, 1]) <= m["leg_r"]).any(1)
    return np.hypot(dx, dy) <= m["radius"]


def episode(run, name):
    sc = S[name]
    L = [json.loads(l) for l in open(RUNS / run / f"{name}.jsonl")]
    n, spawn = 0, os.environ.get("SPAWN_OK")
    for i, r in enumerate(L):
        c, s = math.cos(r["yaw"]), math.sin(r["yaw"])
        P = np.stack([r["x"] + c * CAR[:, 0] - s * CAR[:, 1], r["y"] + s * CAR[:, 0] + c * CAR[:, 1]], 1)
        hit = any(inside(p, man[p["type"]], P).any() for p in sc["props"])
        # SPAWN_OK: a contact run already present at t <= 2 (before any command acts) is the
        # settle shove of a seeded spawn (l22rts4 02: thrown 0.5 m / 62 deg into a cone), not scored
        if spawn and hit and (i <= 2 or spawn == "on"):
            spawn = "on"
            continue
        spawn = spawn and "off"
        n += hit
    return n


for arm in sys.argv[1:]:
    files = sorted(glob.glob(str(RUNS / f"summary_{arm}_*.json")) + glob.glob(str(RUNS / f"summary_{arm}s[0-9]_*.json")))
    cc, cr, reach = [], [], []
    for f in files:
        summ = json.loads(Path(f).read_text()); run = Path(f).stem[len("summary_"):]
        a = b = rch = 0
        for name, e in summ["results"].items():
            ok = e["status"] == "success"
            rect = episode(run, name)
            rch += ok; a += ok and e["collisions"] == 0; b += ok and rect == 0
        cc.append(a); cr.append(b); reach.append(rch)
    m = lambda v: f"{np.mean(v):.1f} ± {np.std(v, ddof=1):.1f}" if len(v) > 1 else str(v)
    print(f"{arm:8s} n={len(files)}  reached {m(reach)}  clean (0.22 m circle) {m(cc)} {cc}  clean (car rectangle) {m(cr)} {cr}")
