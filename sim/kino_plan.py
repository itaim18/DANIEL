"""Kinematic feasibility oracle for the HITL suite: hybrid A* (board/pi5/pathplan.py, the
same planner the Pi runs) with the Traxxas's real Ackermann limits (0.86 m min turn radius,
forward + reverse), its footprint, the yard's drivable map and the scenario's exact props.

hitl_scene.solvable() checks a POINT robot on a grid (props inflated by radius + car), so it
certifies a path the real car may be unable to drive. A scenario this oracle cannot solve is
infeasible for the car, and no planner can score it.

    ~/isaacsim/python.sh kino_plan.py                    # every scenario in scenarios.json
    ~/isaacsim/python.sh kino_plan.py --export T.npz     # terrain distance field for the Pi
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "wheeled_lab/env"))
sys.path.insert(0, str(REPO / "board/pi5"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pathplan  # noqa: E402


def terrain(d, res=0.05):
    """Distance (m) to the nearest non-drivable cell of the drivable map, at res (the Pi has
    no scipy: it loads this from --export)."""
    from scipy import ndimage
    ux, uy = d["ux"], d["uy"]
    scale = int(round((ux[1] - ux[0]) / res))
    free = np.kron(d["mask"].astype(np.uint8), np.ones((scale, scale), np.uint8)).astype(bool)
    dist = ndimage.distance_transform_edt(free) * res
    return {"dist": dist.astype(np.float32), "x0": float(ux[0] - (ux[1] - ux[0]) / 2),
            "y0": float(uy[0] - (uy[1] - uy[0]) / 2), "res": res}


def prop_circles(props, man):
    """(N, 3) contact circles (x, y, r): one per chair leg; a row covering a boxy prop's
    oriented footprint, corners included; else the prop's radius."""
    out = []
    for p in props:
        m = man[p["type"]]
        if "box" in m:
            w, d = m["box"]
            lng, sht = max(w, d), min(w, d)
            n = int(math.ceil(2 * lng / sht))                   # strips <= half the short side
            r = math.hypot(lng / (2 * n), sht / 2)              # each circle covers its strip's corners
            a = p.get("yaw", 0.0) + (0.0 if w >= d else math.pi / 2)
            for k in range(n):
                t = -lng / 2 + (k + 0.5) * lng / n
                out.append((p["x"] + t * math.cos(a), p["y"] + t * math.sin(a), r))
        elif "legs" in m:
            c, s = math.cos(p.get("yaw", 0.0)), math.sin(p.get("yaw", 0.0))
            out += [(p["x"] + c * lx - s * ly, p["y"] + s * lx + c * ly, m["leg_r"]) for lx, ly in m["legs"]]
        else:
            out.append((p["x"], p["y"], m["radius"]))
    return np.array(out, np.float64).reshape(-1, 3)


def rear(root_xy, yaw):
    return (root_xy[0] - pathplan.ROOT_AHEAD * math.cos(yaw), root_xy[1] - pathplan.ROOT_AHEAD * math.sin(yaw), yaw)


if __name__ == "__main__":
    import drivable_map
    import hitl_scene as hs
    d = drivable_map.build()
    T = terrain(d)
    if "--export" in sys.argv:
        dst = sys.argv[sys.argv.index("--export") + 1]
        np.savez_compressed(dst, **T)
        print(f"terrain {T['dist'].shape} @ {T['res']} m -> {dst}")
        sys.exit(0)
    man = hs.manifest()
    suite = json.loads((REPO / "experiments/2026-09-29_hitl/scenarios.json").read_text())
    for sc in suite["scenarios"]:
        t0 = time.time()
        path = pathplan.hybrid_astar(T, prop_circles(sc["props"], man), rear(sc["start"], suite["yaw0"]),
                                     sc["goal"], goal_tol=0.40, margin=0.02)
        dt = time.time() - t0
        if path:
            p = np.array(path)
            L = float(np.hypot(*np.diff(p[:, :2], axis=0).T).sum())
            rev = int((np.diff(p[1:, 3]) != 0).sum())
            print(f"{sc['name']:22s} FEASIBLE  path {L:.1f} m, {rev} direction changes  ({dt:.1f} s)")
        else:
            print(f"{sc['name']:22s} NO PATH   ({dt:.1f} s)")
