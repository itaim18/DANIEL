"""Per-seed HITL results for the thesis, scored on the car's real 0.557 x 0.294 m footprint
(rescore.py), for every arm the paper and thesis report -> bench/hitl_arms.json."""
import glob, json, math
from pathlib import Path
import numpy as np
SCR = Path(__file__).resolve().parent                    # rescore.py lives beside this file
exec((SCR / "rescore.py").read_text().split("for arm in sys.argv")[0])
ARMS = [  # tag, group, label
    ("o13", "ceiling", "Oracle: exact props, the Navigator's planner (lockstep)"),
    ("l21x", "ceiling", "Real-time stack, EXACT obstacles (no perception)"),
    ("p12rt", "gt", "Student v1 + global planner (Pi)"),
    ("p15rt", "gt", "Student v3 + global planner (Pi)"),
    ("l22rt", "gt", "v3 + PLAN_STEER 0.85 + straight-back retreat"),
    ("l26rt", "gt", "+ floor-from-horizon, pinned-retreat wedge, blocked fix"),
    ("l30rt", "gt", "+ escape-radius planning"),
    ("l31rt", "gt", "+ PLAN_STEER 0.75 + floating overhangs (the deployed planner, v3)"),
    ("l32rt", "gt", "+ base skirt on small obstacles (rejected)"),
    ("l35rt", "gt", "+ look first (neutral on own pose; not deployed)"),
    ("l36rt", "gt", "+ terrain-escape pass (neutral)"),
    ("l37rt", "gt", "+ faster follower (neutral)"),
    ("l38rt", "gt", "Student v5 (floor-weighted loss)  [deployed]"),
    ("l39rt", "gt", "+ cone colour contacts (rejected)"),
    ("p12full", "own", "Own pose: ground-plane VO only"),
    ("p16imu", "own", "Own pose: v3 + sim encoder + gyro"),
    ("l22imu", "own", "Own pose: l22 planner + encoder + gyro"),
    ("l31imu", "own", "Own pose: deployed planner + encoder + gyro"),
    ("l34imu", "own", "Own pose: + encoder-gated slip guard (deployed planner, v3)"),
    ("l35imu", "own", "Own pose: + look first"),
    ("l37imu", "own", "Own pose: + faster follower"),
    ("l38imu", "own", "Own pose: student v5  [deployed]"),
    ("p38imu", "own", "Own pose: student v5 on the REAL Pi + Hailo  [deployed]"),
    ("l39imu", "own", "Own pose: + cone colour"),
]
out = []
for tag, grp, lab in ARMS:
    files = sorted(glob.glob(str(RUNS / f"summary_{tag}_*.json")) + glob.glob(str(RUNS / f"summary_{tag}s[0-9]_*.json")))
    if not files:
        continue
    seeds = []
    for f in files:
        summ = json.loads(Path(f).read_text()); run = Path(f).stem[len("summary_"):]
        per = {}
        for name, e in summ["results"].items():
            ok = e["status"] == "success"; rect = episode(run, name)
            per[name] = {"status": e["status"], "rect_contacts": rect, "clean": bool(ok and rect == 0)}
        seeds.append({"run": run, "clean": sum(v["clean"] for v in per.values()),
                      "reached": sum(v["status"] == "success" for v in per.values()), "per": per})
    c = np.array([s["clean"] for s in seeds]); r = np.array([s["reached"] for s in seeds])
    out.append({"tag": tag, "group": grp, "label": lab, "n": len(seeds), "clean_mean": float(c.mean()),
                "clean_sd": float(c.std(ddof=1)) if len(c) > 1 else 0.0, "reached_mean": float(r.mean()),
                "reached_sd": float(r.std(ddof=1)) if len(r) > 1 else 0.0, "seeds": seeds})
    print(f"{tag:8s} n={len(seeds):2d} clean {c.mean():.1f} ± {c.std(ddof=1) if len(c) > 1 else 0:.1f}  reached {r.mean():.1f}")
Path(__file__).with_name("hitl_arms.json").write_text(json.dumps(out, indent=0))
