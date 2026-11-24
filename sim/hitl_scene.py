"""HITL scene: a pool of realistic, MOVABLE obstacle props + frozen avoidance scenarios.

Props are wrappers from wheeled_lab/env/make_hitl_props.py (kinematic rigid body +
invisible cylinder collider), spawned ONCE as a pool and teleported per scenario, so a
whole suite runs in one Isaac boot. Deliberately independent of rl/obstacles.py.

Scenarios live in yard-local metres (the drivable_map frame). Each one has a start
(the car keeps its DEFAULT heading -- writing the root orientation flips this asset,
see mdp.reset_yard_v1), a goal along a straight drivable corridor, and props placed
relative to that corridor. Every scenario is checked SOLVABLE (grid BFS around
inflated props on the drivable mask) before it is frozen.
"""
from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PROPS_DIR = REPO / "wheeled_lab/env/props"
POOL = {"office_chair": 3, "chair": 2, "traffic_cone": 6, "cardboard_box": 3, "barrel": 2,
        "trash_can": 2, "wet_floor_sign": 2, "school_chair": 2, "crate": 2, "person": 1,
        "person_b": 1}
CAR_R = 0.22                 # car footprint circle for the collision latch (0.557 x 0.249 m)
PARK = (150.0, 150.0, -30.0)


def manifest():
    return json.loads((PROPS_DIR / "manifest.json").read_text())


def prop_clearance(p, m, xy):
    """Gap (m) between the car's footprint circle and a placed prop's COLLIDER: its legs
    for legged props (manifest "legs", rotated by the prop's yaw), its oriented box for boxy
    ones (manifest "box": w, d), else its cylinder. Negative = overlap."""
    if "box" in m:
        c, s = math.cos(p.get("yaw", 0.0)), math.sin(p.get("yaw", 0.0))
        dx, dy = xy[0] - p["x"], xy[1] - p["y"]
        qx, qy = abs(c * dx + s * dy) - m["box"][0] / 2, abs(-s * dx + c * dy) - m["box"][1] / 2
        return math.hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0) - CAR_R
    if "legs" in m:
        c, s = math.cos(p.get("yaw", 0.0)), math.sin(p.get("yaw", 0.0))
        return min(math.hypot(p["x"] + c * lx - s * ly - xy[0], p["y"] + s * lx + c * ly - xy[1])
                   for lx, ly in m["legs"]) - m["leg_r"] - CAR_R
    return math.hypot(p["x"] - xy[0], p["y"] - xy[1]) - m["radius"] - CAR_R


def pool_slots():
    return [(name, k) for name, n in POOL.items() for k in range(n)]


def add_prop_pool(scene_cfg):
    """Attach every pool slot to the scene cfg as a kinematic RigidObject."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    for i, (name, k) in enumerate(pool_slots()):
        setattr(scene_cfg, f"prop_{name}_{k}", RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/HitlProps/{name}_{k}",
            spawn=sim_utils.UsdFileCfg(usd_path=str(PROPS_DIR / f"{name}.usda")),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(PARK[0] + 2 * i, PARK[1], PARK[2]))))


def rest_pose(p, R, H_at, tilt=True):
    """(z, quaternion xyzw) of a prop RESTING on the terrain: the plane fitted to the ground
    under its footprint, the prop's up axis along that plane's normal, its yaw about it.
    Placed upright at the centre height, a barrel on the yard's 12-26 % slopes floated up to
    ~15 cm on its downhill side, and the camera ranges an obstacle by where its bottom meets
    the floor: a floating bottom reads too FAR (a plan went 7 cm into a trash can, p1nav 01).
    Upright at the LOWEST ground instead sank an office chair's caster spokes out of view
    (p2nav 00). People stand upright (tilt=False) on the plane's height at their centre."""
    a_ = np.linspace(0.0, 2 * math.pi, 12, endpoint=False)
    pts = [(p["x"] + rr * math.cos(t), p["y"] + rr * math.sin(t)) for rr in (0.5 * R, R) for t in a_] + [(p["x"], p["y"])]
    A = np.array([(x - p["x"], y - p["y"], 1.0) for x, y in pts])
    gx, gy, z0 = np.linalg.lstsq(A, np.array([float(H_at(x, y)) for x, y in pts]), rcond=None)[0]
    yaw = p.get("yaw", 0.0)
    qy = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    if not tilt:
        return float(z0), qy
    n = np.array([-gx, -gy, 1.0]) / math.sqrt(gx * gx + gy * gy + 1.0)      # ground normal
    ang = math.acos(float(np.clip(n[2], -1.0, 1.0)))
    ax = np.array([-n[1], n[0], 0.0])                                      # z x n
    ax = ax / (np.linalg.norm(ax) or 1.0)
    qt = (*(ax * math.sin(ang / 2)), math.cos(ang / 2))
    x1, y1, z1, w1 = qt                                                    # q = tilt * yaw (xyzw)
    x2, y2, z2, w2 = qy
    q = (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)
    return float(z0), q


def place(env, scenario, H_at):
    """Teleport pool props to the scenario's layout; park the rest. Returns placed list."""
    import torch

    origin = env.scene.env_origins[0]
    used = {name: 0 for name in POOL}
    placed, man = [], manifest()
    for p in scenario["props"]:
        name, k = p["type"], used[p["type"]]
        if k >= POOL[name]:
            raise ValueError(f"scenario needs more {name} than the pool has ({POOL[name]})")
        used[name] += 1
        # Upright at the centre's height (scene v1/v2). rest_pose(tilt=True) -- the prop's up
        # axis along the fitted ground normal -- RENDERED the barrels of 03 floating ~0.2 m
        # although physics read back exactly the written pose (2026-09-30; the heightfield and
        # the splat's ground probably disagree there). The lowest-ground variant sank an
        # office chair's spokes out of view. Known limit: on the yard's slopes a prop's
        # downhill edge floats a few cm, and the camera ranges it a little too far.
        z, q = rest_pose(p, man[name]["radius"], H_at, tilt=False)
        z = p.get("z_vis", z)            # --prop-z visual: on the rendered ground (scene v3)
        # Isaac Lab 3.0 quaternions are (x, y, z, w) (isaaclab/utils/math.py). Writing
        # (w, x, y, z) laid the office chair on its back (2026-09-29).
        pose = torch.tensor([[p["x"] + float(origin[0]), p["y"] + float(origin[1]), z + float(origin[2]),
                              *map(float, q)]], device=env.device, dtype=torch.float32)   # f8 breaks Warp
        env.scene[f"prop_{name}_{k}"].write_root_pose_to_sim(pose)
        placed.append({**p, "slot": f"{name}_{k}"})
    for i, (name, k) in enumerate(pool_slots()):
        if k >= used[name]:
            pose = torch.tensor([[PARK[0] + 2 * i, PARK[1], PARK[2], 0.0, 0.0, 0.0, 1.0]],
                                device=env.device)
            env.scene[f"prop_{name}_{k}"].write_root_pose_to_sim(pose)
    return placed


# ------------------------------------------------------------------ scenario generation
def _cell(d, x, y):
    return int(np.argmin(np.abs(d["ux"] - x))), int(np.argmin(np.abs(d["uy"] - y)))


def corridor_ok(d, sx, sy, yaw, L, min_clr=0.45, q=None, q_min=0.55):
    """Straight drivable corridor, AND well reconstructed when q (recon_quality score) is
    given. The first suite landed on the scan's west edge (score 0.07-0.12, hole_frac
    1.0): the camera saw black fog instead of ground and depth had nothing to find."""
    for s in np.linspace(0, L, int(L / 0.1) + 1):
        i, j = _cell(d, sx + s * math.cos(yaw), sy + s * math.sin(yaw))
        if not d["mask"][i, j] or d["clearance"][i, j] < min_clr:
            return False
        if q is not None and not (q[i, j] >= q_min):
            return False
    return True


def solvable(d, start, goal, props, man):
    """BFS on the 0.1 m drivable grid with props inflated by (radius + car)."""
    free = d["mask"].copy()
    gx, gy = np.meshgrid(d["ux"], d["uy"], indexing="ij")
    for p in props:
        r = man[p["type"]]["radius"] + CAR_R
        free &= (gx - p["x"]) ** 2 + (gy - p["y"]) ** 2 > r * r
    s, g = _cell(d, *start), _cell(d, *goal)
    if not free[s] or not free[g]:
        return False
    seen = np.zeros_like(free)
    q = deque([s])
    seen[s] = True
    while q:
        i, j = q.popleft()
        if (i, j) == g:
            return True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            a, b = i + di, j + dj
            if 0 <= a < free.shape[0] and 0 <= b < free.shape[1] and free[a, b] and not seen[a, b]:
                seen[a, b] = True
                q.append((a, b))
    return False


PATTERNS = {
    # name: list of (type, along_frac, lateral_m)
    "chair_center": [("office_chair", 0.45, 0.0)],
    "box_offset": [("cardboard_box", 0.40, 0.25), ("trash_can", 0.75, -0.35)],
    "cone_slalom": [("traffic_cone", 0.25, 0.35), ("traffic_cone", 0.45, -0.35),
                    ("traffic_cone", 0.65, 0.35), ("traffic_cone", 0.85, -0.35)],
    "gate": [("barrel", 0.45, 0.75), ("barrel", 0.45, -0.75)],
    "chair_cluster": [("chair", 0.35, 0.3), ("school_chair", 0.55, -0.4), ("office_chair", 0.75, 0.2)],
    "person_on_path": [("person", 0.5, 0.0)],
    "crate_wall_gap": [("crate", 0.5, 0.55), ("crate", 0.5, -0.55), ("wet_floor_sign", 0.75, 0.0)],
    "mixed_clutter": [("traffic_cone", 0.3, -0.3), ("cardboard_box", 0.5, 0.4), ("chair", 0.62, -0.45),
                      ("trash_can", 0.8, 0.1), ("wet_floor_sign", 0.4, 0.9)],
    "person_and_cones": [("person_b", 0.55, 0.3), ("traffic_cone", 0.35, -0.3), ("traffic_cone", 0.75, -0.2)],
    "barrel_center": [("barrel", 0.5, 0.05)],
}


def _layout(d, sx, sy, phi, L, pattern, rng, q=None, q_min=0.55):
    props = []
    for typ, frac, lat in PATTERNS[pattern]:
        a = frac * L
        px = sx + a * math.cos(phi) - lat * math.sin(phi)
        py = sy + a * math.sin(phi) + lat * math.cos(phi)
        ci, cj = _cell(d, px, py)
        if not d["mask"][ci, cj] or (q is not None and not (q[ci, cj] >= q_min)):
            return None
        props.append({"type": typ, "x": round(px, 3), "y": round(py, 3),
                      "yaw": round(float(rng.uniform(-math.pi, math.pi)), 3)})
    return props


def generate(d, yaw0, man, lengths=(4.5, 4.0, 3.5, 3.0), offsets_deg=(0, 30, -30, 60, -60),
             seed=0, min_sep=0.8, q=None, q_min=0.55):
    """One SOLVABLE scenario per pattern. The car keeps its default heading yaw0; the
    goal lies along a straight drivable corridor at yaw0 + offset, so wider offsets
    also demand a turn toward the goal (the yard's drivable area is only 139 m^2, and
    one fixed heading alone fits just 2-6 corridors)."""
    rng = np.random.default_rng(seed)
    ok_cells = np.argwhere(d["mask"] & (d["clearance"] >= 0.6))
    rng.shuffle(ok_cells)
    out, used_starts, todo = [], [], list(PATTERNS)
    for off in offsets_deg:
        phi = yaw0 + math.radians(off)
        for L in lengths:
            for i, j in ok_cells:
                if not todo:
                    return out
                sx, sy = float(d["ux"][i]), float(d["uy"][j])
                if any(math.hypot(sx - a, sy - b) < min_sep for a, b in used_starts):
                    continue
                if not corridor_ok(d, sx, sy, phi, L, q=q, q_min=q_min):
                    continue
                gx, gy = sx + L * math.cos(phi), sy + L * math.sin(phi)
                for pattern in list(todo):
                    props = _layout(d, sx, sy, phi, L, pattern, rng, q=q, q_min=q_min)
                    if props and solvable(d, (sx, sy), (gx, gy), props, man):
                        out.append({"name": f"{len(out):02d}_{pattern}", "pattern": pattern,
                                    "start": [sx, sy], "goal": [round(gx, 3), round(gy, 3)],
                                    "length_m": L, "yaw0": yaw0, "goal_offset_deg": off,
                                    "props": props})
                        used_starts.append((sx, sy))
                        todo.remove(pattern)
                        break
    return out


def _selftest():
    """rest_pose: on a 20 % slope a prop's up axis is the ground normal, it sits at the
    centre's height, keeps its yaw, and a person stays upright; prop_clearance of a box."""
    H = lambda x, y: 0.2 * x + 1.0                              # noqa: E731
    z, (x, y, zq, w) = rest_pose({"x": 2.0, "y": 0.0, "yaw": 0.7}, 0.3, H)
    up = np.array([2 * (x * zq + w * y), 2 * (y * zq - w * x), 1 - 2 * (x * x + y * y)])
    assert abs(z - 1.4) < 1e-6 and np.allclose(up, np.array([-0.2, 0.0, 1.0]) / math.hypot(0.2, 1.0), atol=1e-6), up
    fwd = (1 - 2 * (y * y + zq * zq), 2 * (x * y + w * zq))
    assert abs(math.atan2(fwd[1], fwd[0]) - 0.7) < 0.02, "yaw lost in the tilt"
    z2, q2 = rest_pose({"x": 2.0, "y": 0.0, "yaw": 0.7}, 0.3, H, tilt=False)
    assert abs(z2 - 1.4) < 1e-6 and q2[:2] == (0.0, 0.0), "a person must stand upright"
    box = {"box": [0.7, 0.5], "radius": 0.366}
    p = {"x": 0.0, "y": 0.0, "yaw": math.pi / 2}                # long side along world y
    assert abs(prop_clearance(p, box, (0.6, 0.0)) - 0.13) < 1e-9 and abs(prop_clearance(p, box, (0.0, 0.6)) - 0.03) < 1e-9
    print("hitl_scene selftest: OK (rest_pose on a 20 % slope, upright people, box clearance)")


if __name__ == "__main__":
    _selftest()
