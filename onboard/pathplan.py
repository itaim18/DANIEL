"""Global kinematic path planning + path following for the Traxxas, numpy only (Pi 5).

Why: the reactive arc planner (avoid.py) tops out at ~3/10 clean on the HITL suite
(v14, 5 seeds) while a hybrid-A* oracle proves all 10 scenarios drivable for this car
(0.86 m turn radius; 3 need a planned reverse). The misses are wedging in terrain the
camera cannot see (ledges, stone edging) and local minima behind clutter. So: plan a
kinematically feasible path on the yard's PRIOR terrain map (this yard is scanned) plus
the obstacles known so far, follow it, re-plan when it is blocked.

    terrain = load_terrain("yard_terrain.npz")            # laptop: kino_plan.py --export
    path = hybrid_astar(terrain, obstacles, start_rear, goal_xy)
    f = Follower(path); steer, throttle, done = f.step(pose_rear)
    python3 pathplan.py                                   # self-check

Frames: world x, y (m), heading th CCW; poses are the REAR AXLE (the root is ROOT_AHEAD
ahead of it). Obstacles are circles (x, y, r): a pose collides when the ROOT circle
(CAR_R) overlaps one -- the sim's contact metric -- or a body circle leaves drivable terrain.
"""
from __future__ import annotations

import heapq
import math
import os
import time

import numpy as np

WHEELBASE, STEER_MAX = 0.313, 0.349
ROOT_AHEAD, CAR_R = 0.156, 0.22
BODY, BODY_R = (-0.02, 0.16, 0.34), 0.16          # footprint circles ahead of the rear axle
STEP, N_TH, XY = 0.25, 72, 0.1
C_CAM = 0.36                                      # camera ahead of the rear axle (slam2d.C_REAR)
# Sensed obstacles: a hit COUNT per 10 cm cell, an obstacle from MIN_HITS on, never cleared
# (static world). Not slam2d's log-odds grid: its free-space rays erased thin chair legs --
# from sim GT depth 1062 scan hits near a chair left 3 occupied cells (n1 04_chair_cluster),
# and the planner drove plans through the chair.
# MIN_HITS 3 at 20 Hz: offline on 4 real-Hailo episodes (every 2nd frame), 1 hit kept 53-102
# phantom cells per episode (>0.3 m from any prop or terrain), 2 hits 10-42. The scan never
# reaches past 2.5 m, so HIT_RANGE does not bind.
# HIT_RES 5 cm, not 10: on 6 real-Hailo episodes core cells (within 0.1 m of a true surface)
# rose 2-4x and a 0.9 m barrel gate read 0.56 m open instead of 0.47 (the cell disc is
# half the size, and hits localise to it).
HIT_RES, MIN_HITS, HIT_RANGE = 0.05, 3, 3.5
# Only hits on obstacles at least NAV_H tall (avoid.py scan column 3) are mapped: the car
# clears ~7 cm, and splat floaters / depth speckle are a few cm. On 6 dumped episodes
# (2 sim-GT, 4 real-Hailo), 6 cm cut phantom hits (>0.3 m from any prop or terrain)
# 2376 -> 75 and kept 56 % of the prop hits. Unfiltered, the phantoms were scattered
# single cells whose 0.39 m exclusion discs walled off open ground: n2 (GT depth) 2/10.
# Within NEAR_Z every hit counts: of the short (< NAV_H) hits there, 2656 were on props and
# 2 phantom (0.6-1.0 m: 6262 / 444; > 1.5 m: 625 / 1133) -- thin legs in the Hailo's
# blurred depth read short, and up close they must still be mapped.
NAV_H, NEAR_Z = 0.06, 0.6
# ... and a cell must hold REL_MIN of the hits of the densest cell within 0.2 m. A sensed
# obstacle is a dense core on its true surface plus a sparse fringe (hit error p90 +0.16 m
# outward over 5 Hailo episodes); with an absolute count the fringe fills in over time and
# every obstacle grows by it -- a 0.9 m gate between two barrels read shut (v9 03).
REL_MIN = 0.3
# Space no ray has seen is passable but costs UNSEEN_W x more within OBS_R of the start: the
# camera sees +-34 deg, and a hard turn swings the body into space beside the car it never
# looked at (n3 GT 04: a chair leg on the inside of a left turn, first seen at 0.10 m
# clearance, contact). The planner now looks before it turns. Rays count as seen out to SEEN_Z.
UNSEEN_W, OBS_R, SEEN_Z = 3.0, 0.9, 2.5
WIDE, WIDE_LEN = 0.20, 1.6    # OBS_R 0.9, not 1.5: 4x fewer expansions (v8 03 dump: 31k -> 7k pops), still halves a blind swing
# (0.18 / 1.25 until 2026-10-01: with the accurate v3 depth the car brushed props 1-7 cm on paths that
# hugged 0.10 while a wider one was < 1.6x longer; a flat margin 0.14 instead blocked the narrow gaps)
WIDE_BUDGET = 1.0              # s; 0.5 until 2026-10-01 -- a wide search that times out keeps the hugging path
# Speed loop: the yard is sloped, and at a fixed throttle -0.35 the sim car reversed up a 12 %
# grade at 0.23 m/s but up 26 % at 0.05 m/s -- a 0.25 m reverse leg then stalled, over and
# over (n3 Pi 04, 07). Throttle is boosted (x1.1 per frame, <= BOOST_MAX) while the car,
# committed to one direction for BOOST_AFTER frames, travels under half the commanded speed.
# The hold keeps the ESC's brake + neutral phases (avoid_drive.EscGate) out of it.
V_MAX, BOOST_MAX, BOOST_AFTER = 0.72, 2.5, 12
# Dead-end memory (user, 2026-11-19: "drive into the dead end, come back, take the other branch -- and
# remember that decision"). No plan while a sensed obstacle sits within DEAD_SHUT m ahead of the bumper and
# DEAD_SIDE m of the centre line = the way ahead is shut. A no-go disc of DEAD_R is left ON the nearest shut
# point (it plugs the gaps a sparse cone wall leaves; a "dead end" that was really one cone only inflates
# it), and a forward-only car then backs out along its own trail (~0.6 m a step, up to DEAD_RETREATS steps,
# i.e. as far as its 10 m trail reaches) until a forward plan exists: turning at 1.17 m it needs ~2 m of
# runway before a fork to take the other branch.
# DEAD_SHUT 0.9 m: the reactive layer stops (and near-hit backs up) 0.4-0.9 m short of a wall. And a dead end needs
# TWO failed plans in a row from the same spot (within 0.25 m): in the 2026-11-19 lab missions (cluttered room,
# 1.2 m, one failure, r 0.3) it fired 10-14 times a mission and its discs narrowed the passages into more failures.
# The discs are obstacles like any other, and memory()/restore() carry them to the next mission.
DEAD_SHUT, DEAD_SIDE, DEAD_R, DEAD_RETREATS = 0.9, 0.35, 0.15, 16


# Obstacle footprint, rear-axle offsets: the root circle (the sim's contact metric) AND the
# front / rear body circles. With the root circle alone the bumpers were unchecked: n3 Pi 04
# put the front body 3 cm into a chair leg at +0.05 m "clearance".
FOOT = ((ROOT_AHEAD, CAR_R), (BODY[0], BODY_R), (BODY[2], BODY_R))
FOOT_OFF, FOOT_R = np.array([f[0] for f in FOOT]), np.array([f[1] for f in FOOT])


def footprint_clear(obs, x, y, th):
    """Per obstacle circle: its clearance to the car's footprint at rear-axle pose (x, y, th)."""
    obs = np.asarray(obs, np.float64).reshape(-1, 3)
    return np.min([np.hypot(obs[:, 0] - x - off * math.cos(th), obs[:, 1] - y - off * math.sin(th)) - obs[:, 2] - r
                   for off, r in FOOT], axis=0) if len(obs) else np.zeros(0)


def load_terrain(path):
    z = np.load(path)
    return {"dist": z["dist"].astype(np.float32), "x0": float(z["x0"]), "y0": float(z["y0"]),
            "res": float(z["res"])}


def open_terrain(size=40.0, res=0.1):
    """No prior map: open ground around the start (only sensed obstacles constrain the plan)."""
    n = int(size / res)
    return {"dist": np.full((n, n), 9.0, np.float32), "x0": -size / 2, "y0": -size / 2, "res": res}


def terrain_clear(T, px, py):
    """Distance (m) from points to the nearest non-drivable terrain (0 outside the map)."""
    i = ((np.asarray(px) - T["x0"]) / T["res"]).astype(int)
    j = ((np.asarray(py) - T["y0"]) / T["res"]).astype(int)
    inside = (i >= 0) & (i < T["dist"].shape[0]) & (j >= 0) & (j < T["dist"].shape[1])
    out = np.zeros(np.shape(px), np.float32)
    out[inside] = T["dist"][i[inside], j[inside]]
    return out


def _collides(T, obs, xs, ys, ths, margin, terrain=True):
    if len(obs):
        for off, r in FOOT:
            px, py = xs + off * np.cos(ths), ys + off * np.sin(ths)
            d = np.hypot(px[:, None] - obs[None, :, 0], py[:, None] - obs[None, :, 1]) - obs[None, :, 2]
            if (d < r + margin).any():
                return True
    for b in BODY if terrain else ():
        if (terrain_clear(T, xs + b * np.cos(ths), ys + b * np.sin(ths)) < BODY_R).any():
            return True
    return False


def _primitives(steers=(-1.0, -0.5, 0.0, 0.5, 1.0), n=5):
    """Motion primitives in the node's frame: (P, n, 3) samples dx, dy, dth + per-primitive
    (dir, steer). All ten are collision-checked in ONE vectorised call per expansion (one
    call per primitive spent 80 % of the planner's time in numpy call overhead)."""
    out, meta = [], []
    for direction in (1, -1):
        for sv in steers:
            kc = math.tan(sv * STEER_MAX) / WHEELBASE
            sd = np.linspace(0, STEP, n + 1)[1:] * direction
            if abs(kc) < 1e-6:
                out.append(np.c_[sd, np.zeros(n), np.zeros(n)])
            else:
                out.append(np.c_[np.sin(kc * sd) / kc, (1 - np.cos(kc * sd)) / kc, kc * sd])
            meta.append((direction, sv))
    return np.array(out), meta


# Plan at PLAN_STEER of full lock, not 1.0: the sim car turns WIDER than the bicycle model says
# (full lock: radius 0.95 m measured vs 0.86 m assumed, p15rt + p19rt, n 4476 windows), so
# full-lock plans saturated the follower and it drifted to the OUTSIDE of the turn -- round the
# very object it was curving past (cross-track p50 4.5 / p90 15 cm at |steer| = 1 vs 1.4 / 6.3
# below 0.5). 0.85 plans a 1.01 m radius and leaves the follower steering to correct.
# 0.75 (1.17 m): at 0.85 the follower still sat at full lock in 9.2 % of forward nav frames and
# 4 of 11 contact onsets followed >= 10 saturated frames (l22rt); all 10 exact-obstacle scenes
# stay plannable at 0.75.
PLAN_STEER = 0.75
PRIMS, PRIM_META = _primitives(steers=tuple(PLAN_STEER * k for k in (-1.0, -0.5, 0.0, 0.5, 1.0)))


def obstacle_field(obs, center, half, res=0.03, cap=0.45):
    """Distance (m) to the nearest obstacle circle's edge on a grid round `center`, EXACT at
    every cell centre within `cap` of a circle (cap otherwise; the footprint never needs more
    than CAR_R + margin). Per plan, collision checks become lookups: checking the footprint
    against every sensed cell per expansion made a failed search take 5.5 s on the Pi (n4:
    the sim's 10 s socket timeout). A chamfer-propagated field under-read by up to 10 cm and
    closed real gaps (01, 02 infeasible)."""
    n = int(2 * half / res) + 1
    x0, y0 = center[0] - half, center[1] - half
    D = np.full((n, n), cap, np.float32)
    k = int((cap) / res) + 2
    for ox, oy, r in obs:
        i0, i1 = max(int((ox - r - x0) / res) - k, 0), min(int((ox + r - x0) / res) + k + 1, n)
        j0, j1 = max(int((oy - r - y0) / res) - k, 0), min(int((oy + r - y0) / res) + k + 1, n)
        if i0 >= i1 or j0 >= j1:
            continue
        cx = x0 + (np.arange(i0, i1) + 0.5) * res
        cy = y0 + (np.arange(j0, j1) + 0.5) * res
        e = np.maximum(np.hypot(cx[:, None] - ox, cy[None, :] - oy) - r, 0.0)
        np.minimum(D[i0:i1, j0:j1], e, out=D[i0:i1, j0:j1])
    return {"dist": D, "x0": x0, "y0": y0, "res": res, "cap": cap}


def field_at(F, px, py):
    """obstacle_field lookup; `cap` (no obstacle near) outside the grid. Minus half a cell
    diagonal: a query anywhere in a cell reads the cell centre's value."""
    i = ((np.asarray(px) - F["x0"]) / F["res"]).astype(int)
    j = ((np.asarray(py) - F["y0"]) / F["res"]).astype(int)
    inside = (i >= 0) & (i < F["dist"].shape[0]) & (j >= 0) & (j < F["dist"].shape[1])
    out = np.full(np.shape(px), F["cap"], np.float32)
    out[inside] = F["dist"][i[inside], j[inside]] - 0.71 * F["res"]
    return out


H_RES, H_HALF = 0.1, 3.5
H_TABLE = None


def ackermann_table(goal_tol=0.35):
    """Obstacle-free cost-to-go of THIS planner's lattice (same primitives and costs): for a
    goal at (dx, dy) in the rear-axle frame of a car at heading 0, the cheapest cost to bring
    the root within goal_tol. The hybrid-A* heuristic: goals beside or behind the car need a
    reverse the holonomic heuristics cannot see, and A* flooded the lattice (a dumped v6val
    01 failure needed ~40000 expansions for a plan with one reverse). Built once by Dijkstra
    (~1 min), cached next to this file."""
    import pathlib
    cache = pathlib.Path(__file__).with_name("ackermann_h.npz")
    if cache.exists():
        return np.load(cache)["h"]
    n = int(2 * H_HALF / H_RES)
    best = np.full((n, n), np.inf)

    def key(x, y, th):
        return int(round(x / XY)), int(round(y / XY)), int(round(th / (2 * math.pi / N_TH))) % N_TH

    openq, seen = [(0.0, 0.0, 0.0, 0.0, 1)], set()
    while openq:
        g, x, y, th, d0 = heapq.heappop(openq)
        k = key(x, y, th)
        if k in seen:
            continue
        seen.add(k)
        rx, ry = x + ROOT_AHEAD * math.cos(th), y + ROOT_AHEAD * math.sin(th)
        i, j = int((rx + H_HALF) / H_RES), int((ry + H_HALF) / H_RES)
        if 0 <= i < n and 0 <= j < n:
            best[i, j] = min(best[i, j], g)
        c, sn = math.cos(th), math.sin(th)
        for pi_, (direction, sv) in enumerate(PRIM_META):
            ex, ey, eth = PRIMS[pi_, -1]
            nx, ny, nth = x + c * ex - sn * ey, y + sn * ex + c * ey, (th + eth) % (2 * math.pi)
            if abs(nx) > H_HALF + 1.0 or abs(ny) > H_HALF + 1.0 or key(nx, ny, nth) in seen:
                continue
            cost = STEP * (1.0 if direction > 0 else 2.0) + (0.6 if direction != d0 else 0.0) + 0.05 * abs(sv)
            heapq.heappush(openq, (g + cost, nx, ny, nth, direction))
    # within goal_tol of the goal is enough: a disc min-filter of the per-cell best
    r = int(goal_tol / H_RES)
    h = best.copy()
    P = np.pad(best, r, constant_values=np.inf)
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di * di + dj * dj <= r * r:
                h = np.minimum(h, P[r + di:r + di + n, r + dj:r + dj + n])
    h = np.where(np.isfinite(h), h, 0.0).astype(np.float32)
    np.savez_compressed(cache, h=h)
    return h


def goal_field(T, obs, goal, start, res=0.1, pad=3.0):
    """Holonomic distance-to-goal (m) round terrain + obstacles on a grid over the start-goal
    box: the hybrid-A* heuristic. Euclidean alone let A* flood every pocket in front of a
    clutter wall (04_chair_cluster: 5.4 s on the Pi). Obstacles are UNDER-inflated by one
    cell so no pose the car can reach reads inf. 8-neighbour relaxation, numpy only."""
    x0, y0 = min(goal[0], start[0]) - pad, min(goal[1], start[1]) - pad
    nx = int((max(goal[0], start[0]) + pad - x0) / res) + 1
    ny = int((max(goal[1], start[1]) + pad - y0) / res) + 1
    cx, cy = np.meshgrid(x0 + (np.arange(nx) + 0.5) * res, y0 + (np.arange(ny) + 0.5) * res, indexing="ij")
    occ = terrain_clear(T, cx, cy) < 0.02
    for ox, oy, r in obs:
        occ |= np.hypot(cx - ox, cy - oy) < r + CAR_R - res
    D = np.full((nx, ny), np.inf)
    gi, gj = int((goal[0] - x0) / res), int((goal[1] - y0) / res)
    D[max(gi - 1, 0):gi + 2, max(gj - 1, 0):gj + 2] = np.hypot(
        cx[max(gi - 1, 0):gi + 2, max(gj - 1, 0):gj + 2] - goal[0], cy[max(gi - 1, 0):gi + 2, max(gj - 1, 0):gj + 2] - goal[1])
    steps = [(di, dj, res * math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
    P = np.full((nx + 2, ny + 2), np.inf)
    for _ in range(4 * (nx + ny)):
        P[1:-1, 1:-1] = D
        Dn = D.copy()
        for di, dj, c in steps:
            np.minimum(Dn, P[1 + di:nx + 1 + di, 1 + dj:ny + 1 + dj] + c, out=Dn)
        Dn[occ] = np.inf
        if np.array_equal(Dn, D):
            break
        D = Dn
    return D, x0, y0, res


# Circles the car overlaps at the start are DROPPED for a plan only within ESC_R of the start
# (driving out of them); beyond it they are hard obstacles. Dropped for the WHOLE plan, a path
# could run straight through a crate the bumper touched (l26rt 07: 0/3, 149-375 contact frames);
# before p26 the follower's blocked check vetoed such plans -- as a re-plan livelock.
ESC_R, FREE_R = 0.3, 0.6


def hybrid_astar(T, obs, start, goal, goal_tol=0.35, margin=0.04, max_expand=200000, free_radius=0.0,
                 observed=None, time_budget=None, hard=None, dropped=None, forward_only=False):
    """start: rear-axle (x, y, th). goal: root (x, y). obs: (N, 3) circles. Returns a list of
    rear-axle states (x, y, th, dir, steer) from start to goal, or None; steer is that of the
    primitive that reached the state. free_radius: terrain is not
    checked within this distance of the start (a car the physics let into mapped non-drivable
    ground must be allowed to drive out of it). observed: bool grid at HIT_RES on T's origin,
    True where a ray has seen free floor (see UNSEEN_W). time_budget: give up after this many
    seconds (a failed search always runs to its limit). hard: (M, 3) circles checked with NO
    margin -- obstacles already inside the margin at the start (see Navigator). forward_only: no
    reverse primitives (the real car's ESC never reversed, 2026-10-07)."""
    hard = np.zeros((0, 3)) if hard is None else np.asarray(hard, np.float64).reshape(-1, 3)
    dropped = np.zeros((0, 3)) if dropped is None else np.asarray(dropped, np.float64).reshape(-1, 3)
    near = np.asarray(obs, np.float64).reshape(-1, 3)
    gx, gy = goal
    sx, sy, sth = start
    t_end = time.perf_counter() + time_budget if time_budget else None
    D, fx0, fy0, fres = goal_field(T, near, goal, (sx, sy))
    F = obstacle_field(near, ((sx + gx) / 2, (sy + gy) / 2), math.hypot(gx - sx, gy - sy) / 2 + 3.0) \
        if len(near) else None

    global H_TABLE
    if H_TABLE is None:
        H_TABLE = ackermann_table()
    hn = H_TABLE.shape[0]

    def h(x, y, th):
        c, sn = math.cos(th), math.sin(th)
        rx, ry = x + ROOT_AHEAD * c, y + ROOT_AHEAD * sn
        e = math.hypot(rx - gx, ry - gy)
        i, j = int((rx - fx0) / fres), int((ry - fy0) / fres)
        if 0 <= i < D.shape[0] and 0 <= j < D.shape[1] and D[i, j] < 1e9:
            e = max(e, float(D[i, j]) - fres)
        dx, dy = c * (gx - x) + sn * (gy - y), -sn * (gx - x) + c * (gy - y)    # goal in the car frame
        ti, tj = int((dx + H_HALF) / H_RES), int((dy + H_HALF) / H_RES)
        if 0 <= ti < hn and 0 <= tj < hn:
            e = max(e, float(H_TABLE[ti, tj]) - 2 * H_RES)
        return e

    def key(x, y, th):
        return int(round(x / XY)), int(round(y / XY)), int(round(th / (2 * math.pi / N_TH))) % N_TH

    s0 = (sx, sy, sth % (2 * math.pi), 1, 0.0)
    openq, came, seen, n = [(h(sx, sy, sth), 0.0, s0, None)], {}, set(), 0
    while openq and n < max_expand:
        if t_end and n % 256 == 0 and time.perf_counter() > t_end:
            return None
        _, g, s, parent = heapq.heappop(openq)
        k = key(*s[:3])
        if k in seen:
            continue
        seen.add(k)
        came[k] = (s, parent)
        n += 1
        x, y, th = s[:3]
        if math.hypot(x + ROOT_AHEAD * math.cos(th) - gx, y + ROOT_AHEAD * math.sin(th) - gy) < goal_tol:
            out, kk = [], k
            while kk is not None:
                st, kk = came[kk]
                out.append(st)
            return out[::-1]
        c, sn = math.cos(th), math.sin(th)
        xs = x + c * PRIMS[..., 0] - sn * PRIMS[..., 1]                  # (10, 5)
        ys = y + sn * PRIMS[..., 0] + c * PRIMS[..., 1]
        ths = th + PRIMS[..., 2]
        cth, sth_ = np.cos(ths), np.sin(ths)
        # every footprint point of all 10 primitives in ONE lookup per field (numpy call
        # overhead, not arithmetic, is what a Pi expansion costs)
        fx = xs[None] + FOOT_OFF[:, None, None] * cth[None]
        fy = ys[None] + FOOT_OFF[:, None, None] * sth_[None]
        bad = np.zeros(PRIMS.shape[:2], bool)
        if F is not None:
            bad |= (field_at(F, fx, fy) < FOOT_R[:, None, None] + margin).any(0)
        if len(hard):
            dh = np.hypot(fx[..., None] - hard[:, 0], fy[..., None] - hard[:, 1]) - hard[:, 2]
            bad |= (dh < FOOT_R[:, None, None, None]).any(-1).any(0)
        if len(dropped):
            dd = np.hypot(fx[..., None] - dropped[:, 0], fy[..., None] - dropped[:, 1]) - dropped[:, 2]
            bad |= (dd < FOOT_R[:, None, None, None]).any(-1).any(0) & (np.hypot(xs - sx, ys - sy) >= ESC_R)
        near_start = np.hypot(xs - sx, ys - sy) < free_radius
        bx, by = fx[1:], fy[1:]                                           # the body circles
        bad |= (terrain_clear(T, bx, by) < BODY_R).any(0) & ~near_start
        bad = bad.any(1)
        unseen = np.zeros(len(PRIM_META))
        if observed is not None:
            rx, ry = xs + ROOT_AHEAD * cth, ys + ROOT_AHEAD * sth_
            side = CAR_R + 0.10
            px_ = np.stack([rx, rx - side * sth_, rx + side * sth_])            # root + both sides
            py_ = np.stack([ry, ry + side * cth, ry - side * cth])
            oi = np.clip(((px_ - T["x0"]) / HIT_RES).astype(int), 0, observed.shape[0] - 1)
            oj = np.clip(((py_ - T["y0"]) / HIT_RES).astype(int), 0, observed.shape[1] - 1)
            close = np.hypot(xs - sx, ys - sy) < OBS_R
            unseen = ((~observed[oi, oj]).mean(0) * close).mean(1)
        for pi_, (direction, sv) in enumerate(PRIM_META):
            if bad[pi_] or (forward_only and direction < 0):
                continue
            ns = (float(xs[pi_, -1]), float(ys[pi_, -1]), float(ths[pi_, -1]) % (2 * math.pi), direction, sv)
            if key(*ns[:3]) in seen:
                continue
            cost = (STEP * (1.0 if direction > 0 else 2.0) + (0.6 if direction != s[3] else 0.0) + 0.05 * abs(sv)
                    + UNSEEN_W * STEP * unseen[pi_])
            heapq.heappush(openq, (g + cost + h(ns[0], ns[1], ns[2]), g + cost, ns, k))
    return None


def plan_cascade(T, obs, p, goal, observed, margin, max_expand, forward_only=False):
    """The Navigator's plan: standard margin, then a relaxed retry, then a wide re-plan of a
    path that hugs its margin. -> (path or None, ms). Module level, so a worker process can
    run it (async_plan)."""
    t0 = time.perf_counter()
    # ponytail: circles the car already overlaps would make every primitive collide at once;
    # they are dropped for this plan (a real contact shows up as a wedge)
    clr = footprint_clear(obs, *p)
    path = None
    # time budgets: the hardest full-knowledge plans take ~1.6 s on the Pi 5; a failed
    # search would otherwise run to max_expand (4.3 s) on each attempt
    for m, fr, tb in ((margin, 0.0, 2.0), (0.06, 0.6, 1.0)):
        # cells the car already overlaps are dropped; cells clear of it but inside the
        # margin stay, with NO margin -- the car may drive away from them, not through.
        # Until n4 they kept the margin and blocked every primitive at the start: a cell
        # sensed 5 cm nearer than planned for made every re-plan fail in 40 ms (01).
        # 0.06, not 0.03, on the retry: the follower's own tracking error is 5.6 cm at p90
        path = hybrid_astar(T, obs[clr >= m], p, goal, margin=m, max_expand=max_expand,
                            free_radius=fr, observed=observed, time_budget=tb, hard=obs[(clr > 0) & (clr < m)],
                            dropped=obs[clr <= 0], forward_only=forward_only)
        if path:
            break
    # A path that hugs its margin is re-planned WIDE, and the wide one is taken if it is at
    # most WIDE_LEN longer: hugging 0.10 passed a person 4 cm outside contact and the
    # follower's own 5 cm error (p90) made it one (v8 08). Trying WIDE FIRST took any wide
    # path however long -- a 10 m detour for a 4 m barrel scenario (v9 09). (A soft
    # proximity cost did the job at 2-4x the search.)
    if path and m == margin and len(obs) and min(float(footprint_clear(obs, *st[:3]).min()) for st in path) < WIDE:
        wide = hybrid_astar(T, obs[clr >= WIDE], p, goal, margin=WIDE, max_expand=max_expand,
                            observed=observed, time_budget=WIDE_BUDGET, hard=obs[(clr > 0) & (clr < WIDE)],
                            dropped=obs[clr <= 0], forward_only=forward_only)
        if wide and _length(wide) <= WIDE_LEN * _length(path):
            path = wide
    return path, (time.perf_counter() - t0) * 1e3


_WT = None
_POOL = None


def _worker_init(T):
    global _WT
    _WT = T


def _worker_plan(args):
    path, ms = plan_cascade(_WT, *args)
    # PATHPLAN_SLOW: a laptop standing in for the Pi (HITL local brain) waits out the Pi's planning
    # time -- the car stops while a plan is pending. Pi 5 / RTX-laptop CPU: 12.4 / 5.3 s on the
    # self-test, x2.35. Unset on the Pi.
    slow = float(os.environ.get("PATHPLAN_SLOW", "1"))
    if slow > 1:
        time.sleep((slow - 1) * ms / 1000)
        ms *= slow
    return path, ms


def _pool(T):
    """One planner process, started with SPAWN: not a thread (the GIL would stall the frame
    loop the plan is meant to free) and not fork (the Pi's server holds the Hailo device)."""
    global _POOL
    if _POOL is None or _POOL[1] is not T:
        if _POOL is not None:
            _POOL[0].terminate()
        import multiprocessing
        _POOL = (multiprocessing.get_context("spawn").Pool(1, initializer=_worker_init, initargs=(T,)), T)
    return _POOL[0]


def _length(path):
    P = np.array([st[:2] for st in path])
    return float(np.hypot(*np.diff(P, axis=0).T).sum())


def compose(a, b):
    """Pose a (+) b for (x, y, th) poses."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2])


def invert(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2])


def rear_of(root_xy, yaw):
    return (root_xy[0] - ROOT_AHEAD * math.cos(yaw), root_xy[1] - ROOT_AHEAD * math.sin(yaw), yaw)


class Follower:
    """Pure pursuit along a hybrid-A* path, one direction segment at a time. A segment of at
    most OPEN_N states (a 3-point-turn leg, <= 0.5 m) is REPLAYED instead: each primitive's
    own steer for its arc length, measured by odometry. Pure pursuit cannot track it -- its
    lookahead exceeds the leg -- and circled the end of a 0.16 m reverse leg for 0.6 m (v5val
    04). Where the plan passes within TIGHT of an obstacle it drives slower on a shorter
    lookahead: tracking error was p90 5.6 / p95 7.7 cm at speed 0.5 (4307 nav steps)."""
    OPEN_N, TIGHT = 2, 0.15

    def __init__(self, path, lookahead=0.45, speed=0.5, clear=None, open_n=OPEN_N, ignore=()):
        self.ignore = {(round(o[0], 3), round(o[1], 3)) for o in ignore}   # cells the plan dropped
        self.segs, self.clear = [[]], [[]]
        cl = np.full(len(path), 9.0) if clear is None else np.asarray(clear)
        for st, c in zip(path[1:], cl[1:]):
            if self.segs[-1] and st[3] != self.segs[-1][-1][3]:
                self.segs.append([])
                self.clear.append([])
            self.segs[-1].append(tuple(st) + (0.0,) * (5 - len(st)))
            self.clear[-1].append(c)
        keep = [k for k, sg in enumerate(self.segs) if len(sg)]
        self.segs = [np.array(self.segs[k]) for k in keep]
        self.clear = [np.array(self.clear[k]) for k in keep]
        p0 = np.array(tuple(path[0]) + (0.0,) * (5 - len(path[0])))
        self.p0 = p0[:2]
        self.starts = [p0] + [sg[-1] for sg in self.segs[:-1]]       # where each segment begins
        self.i, self.la, self.v, self.frames, self.trav, self.last = 0, lookahead, speed, 0, 0.0, None
        self.open_n, self.tight = open_n, False
        self.best_s, self.prog_frame = -1.0, 0     # furthest arc length reached on this segment

    def step(self, pose):
        """pose: rear axle (x, y, th). -> (steer, throttle, done)."""
        if self.last is not None:
            self.trav += math.hypot(pose[0] - self.last[0], pose[1] - self.last[1])
        self.last = pose
        if self.i >= len(self.segs):
            return 0.0, 0.0, True
        seg = self.segs[self.i]
        self.tight = False
        if len(seg) <= self.open_n:                      # replay the primitives open-loop
            if self.trav >= STEP * len(seg) - 0.02:
                self.i, self.frames, self.trav, self.best_s, self.prog_frame = self.i + 1, 0, 0.0, -1.0, 0
                return self.step(pose)
            self.frames += 1
            k = min(int(self.trav / STEP), len(seg) - 1)
            return float(seg[k, 4]), float(seg[0, 3] * 0.35), False
        x, y, th = pose
        d = np.hypot(seg[:, 0] - x, seg[:, 1] - y)
        j = int(np.argmin(d))
        xl, yl, thl, direction = seg[-1][:4]
        # A segment ends only once the car has PASSED its end along the end heading. "Within
        # 0.15 m of the end" (until n1) skipped a whole 0.25 m reverse leg: the car stopped
        # 0.12 m short of the forward leg's end, which put it within 0.15 m of the reverse
        # leg's end too, and the hard left that followed cut into terrain (n1 01_box_offset).
        passed = ((x - xl) * math.cos(thl) + (y - yl) * math.sin(thl)) * direction > -0.03
        if j == len(seg) - 1 and passed:                 # end of this segment: next one
            self.i, self.frames, self.trav, self.best_s, self.prog_frame = self.i + 1, 0, 0.0, -1.0, 0
            return self.step(pose)
        self.frames += 1
        # progress = the furthest projection onto the segment (from its start); +5 cm counts
        pts = np.vstack([self.starts[self.i][None, :2], seg[:, :2]])
        ab = np.diff(pts, axis=0)
        tq = np.clip(((np.array([x, y]) - pts[:-1]) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
        kq = int(np.argmin(np.hypot(*(pts[:-1] + tq[:, None] * ab - np.array([x, y])).T)))
        s_along = float(np.hypot(*ab[:kq].T).sum() + tq[kq] * math.hypot(*ab[kq]))
        if s_along > self.best_s + 0.05 or self.best_s < 0:
            self.best_s, self.prog_frame = max(s_along, self.best_s), self.frames
        self.tight = tight = self.clear[self.i][j:j + 3].min() < self.TIGHT
        la, v = (0.35, 0.3) if tight else (self.la, self.v)
        ahead = np.nonzero((np.arange(len(seg)) >= j) & (d >= la))[0]
        tx, ty = seg[ahead[0], :2] if len(ahead) else seg[-1, :2]
        dx, dy = tx - x, ty - y
        lx = math.cos(th) * dx + math.sin(th) * dy       # target in the car frame (x fwd, y left)
        ly = -math.sin(th) * dx + math.cos(th) * dy
        ld2 = max(lx * lx + ly * ly, 1e-6)
        steer = math.atan(2 * ly / ld2 * WHEELBASE) / STEER_MAX   # same arc forward or reverse
        near_end = np.hypot(seg[-1, 0] - x, seg[-1, 1] - y) < 0.35
        return float(np.clip(steer, -1, 1)), float(direction * (min(0.35, v) if near_end else v)), False

    def no_progress(self):
        """Frames since the car last got 5 cm further along this (pure-pursuit) segment."""
        if self.i >= len(self.segs) or len(self.segs[self.i]) <= self.open_n:
            return 0
        return self.frames - self.prog_frame

    def stalled(self):
        """On this segment for far longer than driving it takes (~55 frames/m at speed 0.5):
        a full-lock reverse leg that crept round a circle and never passed its end ran 180+
        frames (n3 GT 08)."""
        if self.i >= len(self.segs):
            return False
        seg = self.segs[self.i]
        return self.frames > 40 + 90 * STEP * len(seg)

    def off_path(self, pose, tol=0.25):     # 0.5 until v6val: 07 ran 0.45 m off its plan into a crate
        """Distance to the segment's POLYLINE, from its start. Against its states alone, a
        car at the start of a one-state 0.25 m reverse leg read 0.25 m "off" (v7 00: three
        re-plans in a row, then a 6 m detour). Open-loop legs are guarded by stall / wedge."""
        if self.i >= len(self.segs) or len(self.segs[self.i]) <= self.open_n:
            return False
        pts = np.vstack([self.starts[self.i][None, :2], self.segs[self.i][:, :2]])
        a, b = pts[:-1], pts[1:]
        ab = b - a
        t = np.clip(((np.asarray(pose[:2]) - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
        return float(np.hypot(*(a + t[:, None] * ab - np.asarray(pose[:2])).T).min()) > tol

    def blocked(self, T, obs, pose, horizon=3.0):
        """Does the path ahead (next `horizon` m) now hit an obstacle? Judged as the planner
        judged it: the cells it dropped (the car overlapped them) only beyond ESC_R of the
        plan's start, terrain only beyond FREE_R (the retry plans without it near the start).
        Checking both everywhere re-flagged every fresh plan on the next frame: l22rt 07, 25
        re-plans in 5 s."""
        if self.i >= len(self.segs):
            return False
        seg = self.segs[self.i]
        j = int(np.argmin(np.hypot(seg[:, 0] - pose[0], seg[:, 1] - pose[1])))
        ahead = np.vstack([seg[j:]] + self.segs[self.i + 1:])[: int(horizon / STEP) + 1]
        r0 = np.hypot(ahead[:, 0] - self.p0[0], ahead[:, 1] - self.p0[1])
        if len(obs):
            esc, far = r0 < ESC_R, r0 >= ESC_R
            keep = np.array([(round(o[0], 3), round(o[1], 3)) not in self.ignore for o in obs], bool)
            if esc.any() and keep.any() and _collides(T, obs[keep], *ahead[esc, :3].T, 0.0, terrain=False):
                return True
            if far.any() and _collides(T, obs, *ahead[far, :3].T, 0.0, terrain=False):
                return True
        tf = r0 >= FREE_R
        return bool(tf.any() and _collides(T, np.zeros((0, 3)), *ahead[tf, :3].T, 0.0))


class Navigator:
    """Onboard global navigation: the yard's PRIOR terrain map + obstacles SENSED this episode
    (hit counts from avoid.plan's metric scan) -> hybrid A* -> pure pursuit. Re-plans when
    the path ahead is blocked by a new obstacle, the car leaves the path, the car is wedged
    (it then remembers a bump where its bumper is), or the path ends short of the goal.

    Frames: `start` is the rear-axle yard pose at reset = the origin of the SLAM frame
    (slam2d's convention); step() takes the rear-axle pose in that frame."""

    # max_expand: the hardest oracle plan (04, every prop known) popped 9374 nodes; a FAILED
    # search always spends the whole budget (~1.5 s on the Pi at 15000), hence the backoff.
    def __init__(self, T, start, goal, margin=0.10, max_expand=15000, backoff=10, async_plan=False,
                 speed_loop=True, extrude=0.0, base=0.0, forward_only=False):
        self.T, self.start, self.goal = T, tuple(start), tuple(goal)
        # extrude: an object-DEPTH prior. The camera maps only an obstacle's near surface; each hit
        # is also marked up to `extrude` m further along its ray (behind the surface, away from the
        # car, so it does not narrow a gap seen from the front). With the accurate v3 student the car
        # brushed the unmapped back/side of a trash can / person while curving round it (p15rt/p16rt:
        # 639 of 810 contact steps in successful runs were path following, 1-7 cm deep).
        self.extrude = extrude
        # base: map a LOW hit (< NAV_H, beyond NEAR_Z) when a tall hit of the same frame is within
        # `base` m -- a cone's base plate, a person's feet, an office chair's spokes. On p15rt's
        # v3 frames, low hits within 0.25 m of a tall one were 79 % on props (8 % phantom); the
        # isolated ones 21 % (41 % phantom), which is why NAV_H drops them.
        self.base = base
        # async_plan: plan in a worker process and STOP meanwhile. Synchronous, a 0.5-1.5 s
        # re-plan blocked the Pi's loop: in real time the car drove on blind for that long,
        # and ground VO then had to match frames a second apart (a 0.5 m jump got through its
        # speed gate) -- full-realism HITL p5full: 144-193 contact steps, SLAM ATE 0.9 m.
        self.async_plan, self._pending = async_plan, None
        self.forward_only = forward_only       # plans without reverse legs, and never a retreat
        # speed_loop needs a TRUE speed. Fed by ground VO it was positive feedback: VO read half
        # the car's 0.36 m/s, the loop boosted throttle to 1.0, and at 0.8 m/s VO failed outright
        # (full-realism HITL p6full 02, SLAM error 1.1 m). Off on an own-pose car; stalls on a
        # slope are still caught by the stall and wedge re-plans.
        self.speed_loop = speed_loop
        if async_plan:
            _pool(T)                                # start the worker now, not at the first plan
        self.margin, self.max_expand, self.backoff, self.hold = margin, max_expand, backoff, 0
        self.boost, self.dir_frames, self.last_dir = 1.0, 0, 0
        self.trail, self.retreating, self.retreats = [], False, 0   # rear-axle poses every 5 cm + travel dir
        self.straight_back = False                  # a pinned retreat switched to straight back
        self.on_plan = None        # called just before a (re)plan: the real car coasts meanwhile
        self.debug_dir = None      # set: every failed plan is saved there (npz)
        self.t0 = int(time.time())                  # episode id in the dump names (plan counts restart)
        self.fol, self.hist, self.bumps, self.plans, self.plan_ms, self.path = None, [], [], 0, 0.0, None
        self.dead_ends = []                         # (x, y, r) no-go discs where the way ahead was shut
        self.backing_out = False                    # a dead end was found and no forward plan exists yet
        self.fail_at = None                         # where the last plan failed (a dead end needs two in a row)
        self.live = np.zeros((0, 2))                # THIS frame's mapped hits (yard xy): what dead_end() judges
        self.hits = np.zeros((int(T["dist"].shape[0] * T["res"] / HIT_RES) + 1,
                              int(T["dist"].shape[1] * T["res"] / HIT_RES) + 1), np.int16)
        self.observed = np.zeros(self.hits.shape, bool)

    def memory(self):
        """Map memory: what this mission sensed (hits, seen floor, bumps, dead ends), in T's frame."""
        return {"hits": self.hits, "observed": self.observed, "x0": self.T["x0"], "y0": self.T["y0"],
                "bumps": np.array(self.bumps, float).reshape(-1, 3), "dead": np.array(self.dead_ends, float).reshape(-1, 3)}

    def restore(self, z):
        """A previous mission's memory(). The frames must agree: with no prior terrain, T's frame is the
        rear axle at goal-set time, so the car must start on the same mark, facing the same way."""
        if z["hits"].shape != self.hits.shape or (float(z["x0"]), float(z["y0"])) != (self.T["x0"], self.T["y0"]):
            raise ValueError("map memory was saved on a different grid")
        self.hits[:], self.observed[:] = z["hits"], z["observed"]
        self.bumps = [tuple(map(float, b)) for b in z["bumps"]]
        self.dead_ends = [tuple(map(float, d)) for d in z["dead"]]

    def save(self, path):
        np.savez_compressed(path, **self.memory())

    def load(self, path):
        self.restore(np.load(path))

    def rear_blocked(self, pose_frame):
        """The Brain found a reverse blocked (pose and image still): remember a bump behind the car, stop backing out."""
        p = compose(self.start, pose_frame)
        reach = BODY[0] - BODY_R - 0.09
        self.bumps.append((p[0] + reach * math.cos(p[2]), p[1] + reach * math.sin(p[2]), 0.08))
        self.retreats, self.backing_out = max(3, DEAD_RETREATS), False
        if self.retreating:
            self.fol, self.retreating, self.straight_back = None, False, False

    def back_mode(self):
        return "nav_deadend" if self.forward_only else "nav_retreat"

    def dead_end(self, p):
        """No plan from p: if the way ahead is shut NOW, remember a no-go disc where it is shut -> True.
        Judged on this frame's hits (self.live), not the map: on 2026-11-23 the map -- old phantom cells,
        people's legs, its own earlier discs -- called the way shut with 2.2 m free ahead (~9 of 16 dead ends)."""
        if not len(self.live):
            return False
        c, s = math.cos(p[2]), math.sin(p[2])
        bx, by = p[0] + (BODY[2] + BODY_R) * c, p[1] + (BODY[2] + BODY_R) * s          # front bumper
        d = self.live - [bx, by]
        ahead, side = d @ [c, s], d @ [-s, c]
        k = np.flatnonzero((ahead > -0.05) & (ahead < DEAD_SHUT) & (np.abs(side) < DEAD_SIDE))
        if not len(k):
            return False
        x, y = self.live[k[np.argmin(ahead[k])]]
        self.backing_out = True                     # keep backing out while there is no forward plan
        if not any(math.hypot(x - q[0], y - q[1]) < DEAD_R for q in self.dead_ends):
            self.dead_ends.append((x, y, DEAD_R))
        return True

    def mark(self, wx, wy):
        """One frame's obstacle points (yard xy) -> +1 per cell (at most once per frame)."""
        i, j = ((np.asarray(wx) - self.T["x0"]) / HIT_RES).astype(int), ((np.asarray(wy) - self.T["y0"]) / HIT_RES).astype(int)
        ok = (i >= 0) & (i < self.hits.shape[0]) & (j >= 0) & (j < self.hits.shape[1])
        idx = np.unique(i[ok] * self.hits.shape[1] + j[ok])
        self.hits.reshape(-1)[idx] += 1
        return idx

    def sense(self, pose, scan):
        """avoid.plan's scan (camera frame X right, Z fwd, hit) at the yard rear-axle pose."""
        sc = np.asarray(scan, np.float64)
        tall = (sc[:, 3] >= NAV_H) | (sc[:, 1] < NEAR_Z) if sc.shape[1] > 3 else True
        near = (sc[:, 2] > 0) & (np.hypot(sc[:, 0], sc[:, 1]) < HIT_RANGE)
        if self.base > 0 and sc.shape[1] > 3 and (near & tall).any() and (near & ~tall).any():
            th_, lo = sc[near & tall, :2], np.nonzero(near & ~tall)[0]
            dmin = np.hypot(sc[lo, None, 0] - th_[None, :, 0], sc[lo, None, 1] - th_[None, :, 1]).min(1)
            tall = tall.copy()
            tall[lo[dmin < self.base]] = True
        h = sc[near & tall]
        x, y, th = pose
        c, s = math.cos(th), math.sin(th)
        cx, cy = x + C_CAM * c, y + C_CAM * s
        self.live = np.c_[cx + h[:, 1] * c + h[:, 0] * s, cy + h[:, 1] * s - h[:, 0] * c]
        if self.extrude > 0 and len(h):
            r = np.hypot(h[:, 0], h[:, 1])[:, None]
            k = 1.0 + np.arange(HIT_RES, self.extrude + 1e-6, HIT_RES)[None, :] / r       # (n, m) scale
            hx, hz = np.concatenate([h[:, :1], h[:, :1] * k], 1).ravel(), np.concatenate([h[:, 1:2], h[:, 1:2] * k], 1).ravel()
            self.mark(cx + hz * c + hx * s, cy + hz * s - hx * c)
        else:
            self.mark(cx + h[:, 1] * c + h[:, 0] * s, cy + h[:, 1] * s - h[:, 0] * c)
        # free floor along every ray, short of its hit (or out to SEEN_Z): seen space. (Also
        # CLEARING hit cells that near rays see through was tried: it erased a chair, 27 -> 2
        # cells -- the Hailo's thin legs are not hit every frame.)
        rng = np.hypot(sc[:, 0], sc[:, 1])
        free_to = np.where(sc[:, 2] > 0, rng - 0.15, rng)
        dist = np.linspace(0.0, 1.0, int(SEEN_Z / (HIT_RES / 2)))[None, :] * rng[:, None]
        t = dist / np.maximum(rng[:, None], 1e-6)
        X, Z = t * sc[:, :1], t * sc[:, 1:2]
        i = ((cx + Z * c + X * s - self.T["x0"]) / HIT_RES).astype(int)
        j = ((cy + Z * s - X * c - self.T["y0"]) / HIT_RES).astype(int)
        ok = (i >= 0) & (i < self.hits.shape[0]) & (j >= 0) & (j < self.hits.shape[1]) & (dist <= free_to[:, None])
        seen = ok & (dist <= SEEN_Z)
        self.observed[i[seen], j[seen]] = True

    def obstacles(self, pose, radius=6.0):
        """Cells hit MIN_HITS times, and REL_MIN of their neighbourhood's densest cell, near
        the car -> yard circles (+ remembered bumps)."""
        cand = np.argwhere(self.hits >= MIN_HITS)
        if len(cand):
            h, (ni, nj) = self.hits, self.hits.shape
            ci, cj = cand[:, 0], cand[:, 1]
            local = np.zeros(len(cand), np.int32)
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    local = np.maximum(local, h[np.clip(ci + di, 0, ni - 1), np.clip(cj + dj, 0, nj - 1)])
            cand = cand[h[ci, cj] >= REL_MIN * local]
        occ = cand
        xy = (occ + 0.5) * HIT_RES + [self.T["x0"], self.T["y0"]]
        xy = xy[np.hypot(xy[:, 0] - pose[0], xy[:, 1] - pose[1]) < radius]
        circ = np.c_[xy, np.full(len(xy), 0.7 * HIT_RES)]
        return np.vstack([circ, np.array(self.bumps).reshape(-1, 3), np.array(self.dead_ends).reshape(-1, 3)])

    def step(self, pose_frame, scan=None, last_throttle=0.0, dt=0.05):
        """-> (steer, throttle, mode), or None when there is no plan (caller falls back)."""
        r = self._step(pose_frame, scan, last_throttle)
        if r is None:
            self.boost, self.dir_frames = 1.0, 0
            return None
        d = int(np.sign(r[1]))
        self.dir_frames = self.dir_frames + 1 if d == self.last_dir and d else 0
        self.last_dir = d
        # no boost where the plan passes an obstacle closely: slow THERE means pushing on it,
        # not a hill (v6val 07 boosted 0.30 -> 0.75 into a crate)
        if self.speed_loop and len(self.hist) >= 6 and self.dir_frames >= BOOST_AFTER and not (self.fol and self.fol.tight):
            h = np.array(self.hist[-6:])
            v = float(np.hypot(*np.diff(h[:, :2], axis=0).T).sum()) / (5 * dt)
            slow = v < 0.5 * abs(last_throttle) * V_MAX
            self.boost = min(self.boost * 1.1, BOOST_MAX) if slow else max(self.boost * 0.97, 1.0)
        elif self.dir_frames == 0:
            self.boost = 1.0
        return r[0], float(np.clip(r[1] * self.boost, -1.0, 1.0)), r[2]

    def _step(self, pose_frame, scan, last_throttle):
        p = compose(self.start, pose_frame)
        if scan is not None:
            self.sense(p, scan)
        obs = self.obstacles(p)
        if not self.retreating and (not self.trail or math.hypot(p[0] - self.trail[-1][0], p[1] - self.trail[-1][1]) > 0.05):
            self.trail = (self.trail + [(p[0], p[1], p[2], 1 if last_throttle >= 0 else -1)])[-200:]   # 10 m: deep pockets
        # wedged = throttle commanded for 1.5 s but under 4 cm TRAVELLED. Until n3 it was net
        # displacement over 40 frames, which a legitimate 3-point turn (forward, reverse to the
        # same spot) also has: every such turn dropped a false bump ahead, forcing another
        # reverse (n3 GT 01, re-plans every ~40 frames, stuck).
        self.hist = (self.hist + [(p[0], p[1], abs(last_throttle))])[-30:]
        h = np.array(self.hist)
        wedged = (len(h) == 30 and float(np.hypot(*np.diff(h[:, :2], axis=0).T).sum()) < 0.04
                  and float(h[:, 2].mean()) > 0.2)
        # ... or no PROGRESS along the path for 30 frames under throttle. Travel alone missed a
        # car pinned on a cone: it pivoted and its VO pose jittered past the 4 cm, and it pushed
        # at full lock for 199 steps (full-realism p7full 02).
        # A straight-back that is pinned too is wedged: something unseen is behind or beside
        # the car (l22rt 08: a cone base 3 cm off the flank, 240-280 frames at full reverse,
        # retreat -> straight back -> stall -> retreat again, until the 35 s timeout).
        wedged = wedged or (self.fol is not None and (not self.retreating or self.straight_back)
                            and self.fol.no_progress() > 30 and len(h) >= 10 and float(h[-10:, 2].mean()) > 0.2)
        if wedged and self.retreating:  # backing out is what failed: no more retreats, replan
            self.retreats = max(3, DEAD_RETREATS)   # under either cap (a dead end allows more)
        if wedged:                     # an obstacle nobody saw: remember it at the bumper
            # at the bumper, just clear of the footprint: a bump the car already overlaps would
            # be dropped by the plan-start filter below
            reach = BODY[2] + BODY_R + 0.09 if last_throttle >= 0 else BODY[0] - BODY_R - 0.09
            self.bumps.append((p[0] + reach * math.cos(p[2]), p[1] + reach * math.sin(p[2]), 0.08))
            self.hist = []
        if self._pending is not None:                  # async: stopped until the plan is in
            job, obs0, p0, why0 = self._pending
            if not job.ready():
                return 0.0, 0.0, "nav_planning"
            self._pending = None
            path, self.plan_ms = job.get()
            return self._adopt(path, obs0, p, why0)
        if self.fol is None and self.hold > 0:        # a plan just failed: the fallback drives
            self.hold -= 1
            return None
        if self.retreating and self.fol is not None and self.fol.no_progress() > 30 and not self.straight_back:
            # A retreat that makes no progress is pinned: following the curved trail backwards
            # swings the nose into what it is backing away from (p21rt 08: 500+ steps at full
            # reverse lock against a cone, jitter defeating the 4 cm travel wedge). Back STRAIGHT.
            c_, s_ = math.cos(p[2]), math.sin(p[2])
            self.fol = Follower([(p[0], p[1], p[2], 1)] + [(p[0] - d * c_, p[1] - d * s_, p[2], -1)
                                                           for d in (0.1, 0.2, 0.3, 0.4)], open_n=0)
            self.straight_back = True
        steer, thr, done = self.fol.step(p) if self.fol else (0.0, 0.0, True)
        # a retreat trusts its trail (just driven): sensed cells on it -- the obstacle the car
        # is backing away from -- must not abort it; only its end, a stall or a wedge do
        live = not self.retreating
        why = ("none" if self.fol is None else "end" if done else "wedged" if wedged else
               "off_path" if live and self.fol.off_path(p) else "stalled" if self.fol.stalled() else
               "blocked" if live and self.fol.blocked(self.T, obs, p) else "")
        if why:
            if self.on_plan:
                self.on_plan()
            job = (obs, p, self.goal, self.observed, self.margin, self.max_expand, self.forward_only)
            if self.async_plan:
                self._pending = (_pool(self.T).apply_async(_worker_plan, (job,)), obs, p, why)
                self.fol = None
                return 0.0, 0.0, "nav_planning"
            path, self.plan_ms = plan_cascade(self.T, *job)
            return self._adopt(path, obs, p, why)
        return steer, thr, self.back_mode() if self.retreating else "nav" if thr >= 0 else "nav_rev"

    def _adopt(self, path, obs, p, why):
        """A finished plan (or its failure) takes over: follower, dump, retreat or hold."""
        self.plans += 1
        cl = [float(footprint_clear(obs, *st[:3]).min()) if len(obs) else 9.0 for st in path] if path else None
        drop = obs[footprint_clear(obs, *path[0][:3]) <= 0] if path and len(obs) else ()
        self.fol, self.path = (Follower(path, clear=cl, ignore=drop), path) if path else (None, None)
        if not path and self.debug_dir:             # the failing problem, for offline probing
            np.savez(f"{self.debug_dir}/fail_{self.t0}_{self.plans:04d}.npz", obs=obs, pose=np.array(p),
                     goal=np.array(self.goal), observed=self.observed, hits=self.hits)
        again = not path and self.fail_at is not None and math.hypot(p[0] - self.fail_at[0], p[1] - self.fail_at[1]) < 0.25
        self.fail_at = None if path else p
        dead = again and self.dead_end(p)           # remembered: the next plan avoids this pocket
        # A forward-only car (the real one) never retreats (user, 2026-11-23). Its trail retreats were 16.1 of the
        # 16.7 m it reversed that day, ~9 of 16 with nothing close, chained up to 4 m with a -216 deg spin. With no
        # plan the reactive layer drives; avoid.plan's near-hit / stuck backup is its only reverse. The dead-end
        # disc above is still remembered.
        cap = DEAD_RETREATS if self.backing_out else 3
        # retreat legs: trail points BEHIND the car, driven in reverse. Negating the trail's own direction
        # replayed a leg the car had just REVERSED as forward -- back into the plants it backed away from.
        back = [(x, y, th, -1) for x, y, th, _ in self.trail[::-1][1:13]
                if (x - p[0]) * math.cos(p[2]) + (y - p[1]) * math.sin(p[2]) < 0]
        if not path and not self.forward_only and self.retreats < cap and len(self.trail) >= 6 and len(back) >= 2:
            # No plan from here (boxed in by what the camera found up close, n3 Pi 08):
            # back out along the car's own trail -- driven moments ago, so free -- ~0.6 m,
            # then plan again. The trail is frozen while retreating and each retreat uses
            # it up, so a second one continues further back (v6val: a retreat appended to
            # the trail, and the next "retreat" drove forward into the crate again).
            self.trail = self.trail[:-12] or self.trail[:1]
            self.fol = Follower([(p[0], p[1], p[2], 1)] + back, open_n=0)
            self.path, self.retreating, self.retreats = None, True, self.retreats + 1
            self.straight_back = False
            steer, thr, _ = self.fol.step(p)
            return steer, thr, self.back_mode()
        self.retreating = False
        if not path:
            self.hold = self.backoff
            return None
        self.retreats, self.backing_out = 0, False
        steer, thr, _ = self.fol.step(p)
        return steer, thr, "nav_plan:" + why


def _check_field():
    """obstacle_field never reads more clearance than the truth, and at most 3 cm less."""
    rng = np.random.default_rng(0)
    obs = np.c_[rng.uniform(-3, 3, (80, 2)), rng.uniform(0.04, 0.4, 80)]
    F = obstacle_field(obs, (0.0, 0.0), 4.0)
    q = rng.uniform(-3.5, 3.5, (4000, 2))
    true = np.minimum((np.hypot(q[:, None, 0] - obs[:, 0], q[:, None, 1] - obs[:, 1]) - obs[:, 2]).min(1), F["cap"])
    got = field_at(F, q[:, 0], q[:, 1])
    over, under = float((got - np.maximum(true, 0)).max()), float((np.maximum(true, 0) - got).max())
    assert over <= 1e-6, f"field reads {over:.3f} m MORE clearance than the truth"
    assert under < 0.045, f"field reads {under:.3f} m less clearance than the truth"
    return over, under


def _rollout(path, pose, steps=900):
    """Follow a path on the kinematic bicycle at 20 Hz; returns the final root xy."""
    f, pose = Follower(path), np.array(pose, float)
    for _ in range(steps):
        steer, thr, done = f.step(pose)
        if done:
            break
        v = thr * 0.72 * 0.05
        pose = pose + np.array([v * math.cos(pose[2]), v * math.sin(pose[2]),
                                v * math.tan(steer * STEER_MAX) / WHEELBASE])
    return pose[:2] + ROOT_AHEAD * np.array([math.cos(pose[2]), math.sin(pose[2])])


def _selftest_base():
    T = open_terrain()
    sc = np.array([[0.0, 1.5, 1.0, 0.30],                  # tall (a cone body) at 1.5 m
                   [0.15, 1.5, 1.0, 0.02],                 # low, 0.15 m beside it: its base -> mapped
                   [1.2, 1.5, 1.0, 0.02]])                 # low, 1.2 m away: a floater -> dropped
    for b, want in ((0.25, 2), (0.0, 1)):
        n = Navigator(T, (0.0, 0.0, 0.0), (5.0, 0.0), base=b)
        n.sense((0.0, 0.0, 0.0), sc)
        assert int((n.hits > 0).sum()) == want, (b, int((n.hits > 0).sum()))
    print("base selftest: OK (a low hit beside a tall one is mapped, an isolated one is not)")


def _selftest_blocked_consistent():
    T = open_terrain()
    path = [(STEP * k, 0.0, 0.0, 1) for k in range(16)]
    here = np.array([[0.2, 0.05, 0.035]])                     # a cell the car overlaps at the start
    assert Follower(path).blocked(T, here, (0.0, 0.0, 0.0)), "an overlapped cell must block without ignore"
    assert not Follower(path, ignore=here).blocked(T, here, (0.0, 0.0, 0.0)), "a dropped cell re-blocked the plan"
    ahead = np.array([[2.0, 0.0, 0.035]])
    assert Follower(path, ignore=here).blocked(T, np.vstack([here, ahead]), (0.0, 0.0, 0.0)), "a new cell ahead must block"
    deep = np.array([[0.9, 0.0, 0.035]])                      # dropped, but ON the path past the escape
    assert Follower(path, ignore=deep).blocked(T, deep, (0.0, 0.0, 0.0)), "a dropped cell beyond ESC_R must block"
    # the planner must not drive THROUGH the face of a crate the bumper touches, its back unseen
    # (every face cell is inside the footprint, so all are dropped; p26 planned straight on)
    face = np.array([(0.45, y, 0.035) for y in np.arange(-0.15, 0.151, 0.05)])
    pth, _ = plan_cascade(T, face, (0.0, 0.0, 0.0), (3.0, 0.0), None, 0.10, 15000)
    assert pth, "no plan away from a touched face"
    far_ = [st for st in pth if math.hypot(st[0], st[1]) >= ESC_R]
    assert all(footprint_clear(face, *st[:3]).min() > 0 for st in far_), "plan drove through the touched face"
    print("blocked-consistency selftest: OK (dropped cells free only within ESC_R; plans go round a touched wall)")


def _selftest_pinned_retreat():
    nr = Navigator(open_terrain(), (0.0, 0.0, 0.0), (5.0, 0.0))
    for k in range(40):                                   # drove a curve: 2 m with a left turn
        nr.step((0.05 * k, 0.002 * k * k, 0.004 * k), None, 0.5)
    ring = np.array([(2.0 + 0.6 * math.cos(a), 3.2 + 0.6 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 60)])
    pose = (1.95, 3.04, 0.156)
    ring = np.array([(pose[0] + 0.6 * math.cos(a), pose[1] + 0.6 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 60)])
    for _ in range(MIN_HITS):
        nr.mark(ring[:, 0], ring[:, 1])
    r = nr.step(pose, None, 0.5)
    assert nr.retreating, f"expected a retreat: {r}"
    rj = np.random.default_rng(3)
    for _ in range(40):                                   # pinned: jitters +-2 cm, goes nowhere
        j = rj.uniform(-0.02, 0.02, 2)
        r = nr.step((pose[0] + j[0], pose[1] + j[1], pose[2]), None, -0.5)
    seg = nr.fol.segs[0]
    assert nr.straight_back and np.allclose(seg[:, 2], pose[2]) and abs(r[0]) < 0.35, f"pinned retreat not straightened: {r}"
    for _ in range(40):                                   # the straight-back is pinned too
        j = rj.uniform(-0.02, 0.02, 2)
        r = nr.step((pose[0] + j[0], pose[1] + j[1], pose[2]), None, -0.5)
    assert not nr.retreating and len(nr.bumps), f"pinned straight-back never gave up: {r}"
    print("pinned-retreat selftest: OK (no progress for 30 frames -> straight back, steer 0; pinned again -> wedged, replan)")


def _selftest_extrude():
    T = open_terrain()
    n = Navigator(T, (0.0, 0.0, 0.0), (5.0, 0.0), extrude=0.15)
    n.sense((0.0, 0.0, 0.0), np.array([[0.0, 1.0, 1.0, 0.3]]))      # X 0, Z 1 m, hit, 0.3 m tall
    xs = sorted({round((i + 0.5) * HIT_RES + T["x0"], 3) for i, j in np.argwhere(n.hits > 0)})
    want = 1.0 + C_CAM
    assert min(xs) >= want - HIT_RES and max(xs) <= want + 0.15 + HIT_RES and max(xs) - min(xs) >= 0.10, xs
    n0 = Navigator(T, (0.0, 0.0, 0.0), (5.0, 0.0))
    n0.sense((0.0, 0.0, 0.0), np.array([[0.0, 1.0, 1.0, 0.3]]))
    assert int((n0.hits > 0).sum()) == 1, "extrude off must mark one cell"
    print(f"extrude selftest: OK (hit at {want:.2f} m marks {min(xs):.2f}..{max(xs):.2f} m; off: 1 cell)")


def _selftest():
    """(1) A wall of obstacles between start and goal in open ground: the path goes round it
    without contact and the follower reaches the goal. (2) Goal BEHIND the car in a 1.2 m
    corridor (a U-turn needs ~2 m): the only way is a reverse, and the follower must drive it."""
    res = 0.05
    T = {"dist": np.full((400, 400), 9.0, np.float32), "x0": -10.0, "y0": -10.0, "res": res}
    wall = np.array([[2.0, y, 0.2] for y in np.linspace(-1.0, 1.0, 9)])
    goal = (4.0, 0.0)
    path = hybrid_astar(T, wall, (0.0, 0.0, 0.0), goal)
    assert path is not None, "no path round a 2 m wall in open ground"
    p = np.array(path)
    rx, ry = p[:, 0] + ROOT_AHEAD * np.cos(p[:, 2]), p[:, 1] + ROOT_AHEAD * np.sin(p[:, 2])
    clear = (np.hypot(rx[:, None] - wall[None, :, 0], ry[:, None] - wall[None, :, 1]) - wall[None, :, 2]).min()
    assert clear >= CAR_R, f"path touches the wall (clearance {clear:.2f})"
    err = float(np.hypot(*(_rollout(path, (0, 0, 0)) - goal)))
    assert err < 0.45, f"follower ended {err:.2f} m from the goal"

    yy = T["y0"] + (np.arange(400) + 0.5) * res
    T2 = dict(T, dist=np.tile(np.clip(0.6 - np.abs(yy), 0, None), (400, 1)).astype(np.float32))
    goal2 = (-2.5, 0.25)                                  # behind AND off-axis: reverse must steer
    path2 = hybrid_astar(T2, np.zeros((0, 3)), (0.0, 0.0, 0.0), goal2)
    assert path2 is not None, "no path to a goal behind the car in a corridor"
    assert min(s[3] for s in path2) < 0, "corridor path does not reverse"
    err2 = float(np.hypot(*(_rollout(path2, (0, 0, 0)) - goal2)))
    assert err2 < 0.45, f"reverse follower ended {err2:.2f} m from the goal"
    assert hybrid_astar(T2, np.zeros((0, 3)), (0.0, 0.0, 0.0), goal2, max_expand=20000, forward_only=True) is None, \
        "forward-only found a way to a goal behind the car in a 1.2 m corridor (needs a reverse)"
    loop = hybrid_astar(T, np.zeros((0, 3)), (0.0, 0.0, 0.0), (-1.5, 0.0), forward_only=True)
    assert loop is not None and min(s[3] for s in loop) > 0, "forward-only: no forward loop to a goal behind"

    # (3) Navigator: the grid frame is rotated/shifted from the yard; a wall is SENSED across
    # the straight path after 10 frames -> blocked -> re-plan round it, no contact, goal reached.
    start = (1.0, 2.0, 0.3)
    goal3 = compose(start, (4.0 + ROOT_AHEAD, 0.0, 0.0))[:2]
    nav, pg = Navigator(T, start, goal3), np.zeros(3)
    wall_g = np.array([[2.2, y] for y in np.linspace(-0.8, 0.8, 17)])       # grid frame
    wall_y = np.array([compose(start, (x, y, 0.0))[:2] for x, y in wall_g])
    min_clr, err3 = 9.0, 9.0
    for k in range(900):
        if k == 10:
            for _ in range(MIN_HITS):
                nav.mark(wall_y[:, 0], wall_y[:, 1])
        r = nav.step(pg)
        assert r is not None, f"navigator found no plan at frame {k}"
        steer, thr, _ = r
        v = thr * 0.72 * 0.05
        pg = pg + np.array([v * math.cos(pg[2]), v * math.sin(pg[2]), v * math.tan(steer * STEER_MAX) / WHEELBASE])
        py = compose(start, pg)
        root = np.array([py[0] + ROOT_AHEAD * math.cos(py[2]), py[1] + ROOT_AHEAD * math.sin(py[2])])
        if k > 10:
            min_clr = min(min_clr, float(np.hypot(*(wall_y - root).T).min()))
        err3 = float(np.hypot(*(root - goal3)))
        if err3 < 0.4:
            break
    assert nav.plans >= 2, "the sensed wall never triggered a re-plan"
    assert not nav.bumps, "a clean drive round the wall produced a false wedge"
    assert min_clr > CAR_R, f"navigator drove within {min_clr:.2f} m of the sensed wall"
    assert err3 < 0.4, f"navigator ended {err3:.2f} m from the goal"
    # (3b) the same with async_plan: STOPPED (throttle 0) whenever a plan is pending, then
    # round the wall to the goal; the plans ran in the worker process
    na, pa, stopped, err_a = Navigator(T, start, goal3, async_plan=True), np.zeros(3), 0, 9.0
    for k in range(3000):
        if k == 10:
            for _ in range(MIN_HITS):
                na.mark(wall_y[:, 0], wall_y[:, 1])
        r = na.step(pa)
        assert r is not None, "async navigator found no plan"
        if r[2] == "nav_planning":
            assert r[1] == 0.0, "moving while a plan is pending"
            stopped += 1
            time.sleep(0.002)
            continue
        v = r[1] * 0.72 * 0.05
        pa = pa + np.array([v * math.cos(pa[2]), v * math.sin(pa[2]), v * math.tan(r[0] * STEER_MAX) / WHEELBASE])
        py_ = compose(start, pa)
        err_a = float(np.hypot(py_[0] + ROOT_AHEAD * math.cos(py_[2]) - goal3[0], py_[1] + ROOT_AHEAD * math.sin(py_[2]) - goal3[1]))
        if err_a < 0.4:
            break
    assert stopped > 0 and na.plans >= 2 and err_a < 0.4, f"async: {na.plans} plans, {stopped} stopped frames, ends {err_a:.2f} m off"

    # (4) forward -> short reverse -> forward: stopping 0.12 m short of the forward leg's end
    # must NOT skip the 0.25 m reverse leg
    fr = [(0.0, 0.0, 0.0, 1)] + [(0.25 * k, 0.0, 0.0, 1) for k in (1, 2, 3, 4)] + [(0.75, 0.0, 0.0, -1)] \
        + [(0.75 + 0.25 * k, 0.0, 0.0, 1) for k in (1, 2, 3)]
    f5 = Follower(fr)
    xs5, pose5 = [], np.array([0.88, 0.0, 0.0])
    for _ in range(200):
        st5, thr5, done5 = f5.step(pose5)
        if done5:
            break
        v = thr5 * 0.72 * 0.05
        pose5 = pose5 + np.array([v * math.cos(pose5[2]), v * math.sin(pose5[2]), v * math.tan(st5 * STEER_MAX) / WHEELBASE])
        xs5.append(pose5[0])
    assert min(xs5) < 0.80, f"the reverse leg was skipped (car never went below x={min(xs5):.2f})"
    # a 2-primitive full-lock reverse leg is replayed open-loop and lands on its planned pose
    kc = math.tan(STEER_MAX) / WHEELBASE
    legs = [(0.0, 0.0, 0.0, 1, 0.0)]
    for k in (1, 2):
        th_k = -kc * STEP * k
        legs.append(((math.sin(th_k)) / kc, -(math.cos(th_k) - 1.0) / kc, th_k, -1, 1.0))
    fo, pose_o = Follower(legs), np.array([0.0, 0.0, 0.0])
    for _ in range(200):
        st_o, thr_o, done_o = fo.step(tuple(pose_o))
        if done_o:
            break
        v = thr_o * 0.72 * 0.05
        pose_o = pose_o + np.array([v * math.cos(pose_o[2]), v * math.sin(pose_o[2]), v * math.tan(st_o * STEER_MAX) / WHEELBASE])
    e_xy = math.hypot(pose_o[0] - legs[-1][0], pose_o[1] - legs[-1][1])
    assert e_xy < 0.05 and abs(pose_o[2] - legs[-1][2]) < 0.1, f"open-loop reverse leg ends {e_xy:.3f} m off"
    fp = Follower([(0, 0, 0, 1, 0.0), (-0.25, 0, 0, -1, 0.0)] + [(0.25 * k, 0, 0, 1, 0.0) for k in range(4)])
    assert not fp.off_path((0.0, 0.0, 0.0)), "a car at the start of a one-state reverse leg is ON its path"
    fp.i = 1
    assert not fp.off_path((0.37, 0.1, 0.0)) and fp.off_path((0.37, 0.3, 0.0)), "off-path is to the polyline"
    f5b = Follower(fr)                                    # a car that never moves stalls
    for _ in range(40 + int(90 * STEP * 4) + 2):
        f5b.step((0.0, 0.0, 0.0))
    assert f5b.stalled(), "a car stuck on a segment is never declared stalled"

    # (5) a car sitting inside mapped non-drivable ground still gets a plan out of it
    T6 = dict(T, dist=T["dist"].copy())
    T6["dist"][195:205, 195:205] = 0.0                    # a 0.5 m patch round the origin
    assert hybrid_astar(T6, np.zeros((0, 3)), (0.0, 0.0, 0.0), (3.0, 0.0)) is None
    assert Navigator(T6, (0.0, 0.0, 0.0), (3.0, 0.0)).step((0.0, 0.0, 0.0)) is not None, \
        "no plan out of mapped non-drivable ground"

    # (6) sense(): a scan hit 2 m ahead and 0.5 m right of the camera lands there in the yard,
    # and only after MIN_HITS frames
    ns = Navigator(open_terrain(), (0.0, 0.0, 0.0), (5.0, 0.0))
    pose6, ray = (1.0, 2.0, 0.3), np.array([[0.5, 2.0, 1.0, 0.3], [0.0, 3.9, 1.0, 0.3], [-0.5, 1.0, 0.0, 0.0],
                                            [0.2, 1.5, 1.0, 0.03]])      # the last: a 3 cm floater
    near6 = np.array([[0.0, 0.4, 1.0, 0.02]])                               # short but 0.4 m away: counts
    want = (1.0 + (C_CAM + 2.0) * math.cos(0.3) + 0.5 * math.sin(0.3), 2.0 + (C_CAM + 2.0) * math.sin(0.3) - 0.5 * math.cos(0.3))
    for _ in range(MIN_HITS - 1):
        ns.sense(pose6, ray)
    assert not len(ns.obstacles(pose6)), "MIN_HITS - 1 frames of hits are already an obstacle"
    ns.sense(pose6, ray)
    o6 = ns.obstacles(pose6)
    assert len(o6) == 1, f"{len(o6)} obstacles (3.9 m is past HIT_RANGE, a miss is no hit, 3 cm is below NAV_H)"
    assert math.hypot(o6[0, 0] - want[0], o6[0, 1] - want[1]) < HIT_RES, (o6[0], want)
    for _ in range(MIN_HITS):
        ns.sense(pose6, near6)
    assert len(ns.obstacles(pose6)) == 2, "a short hit within NEAR_Z must be mapped"
    ray_pts = [compose(pose6, (C_CAM + z, -0.5 * z / 2.0, 0.0))[:2] for z in (1.0, 2.2)]   # along the 2 m hit ray
    oi = [(int((x - ns.T["x0"]) / HIT_RES), int((y - ns.T["y0"]) / HIT_RES)) for x, y in ray_pts]
    assert ns.observed[oi[0]] and not ns.observed[oi[1]], "free floor is seen before a hit, not behind it"

    # (6b) pinned: the car jitters in place (pose noise +-2 cm, travel > 4 cm per 1.5 s) while
    # the follower commands throttle along a long straight path -> wedged by no-progress
    npn = Navigator(open_terrain(), (0.0, 0.0, 0.0), (5.0, 0.0))
    rngj = np.random.default_rng(1)
    for k in range(80):
        jit = rngj.uniform(-0.02, 0.02, 2)
        npn.step((0.5 + jit[0], jit[1], 0.0), None, 0.5)
    assert npn.bumps, "a car pinned in place under throttle was never taken for wedged"

    # (7) with only a strip straight ahead observed, a goal ahead-left is reached by driving
    # INTO the seen strip first, not by swinging the body at once into unseen space
    To = open_terrain()
    obsd = np.zeros((int(To["dist"].shape[0] * To["res"] / HIT_RES) + 1,) * 2, bool)
    xi = ((np.arange(obsd.shape[0]) + 0.5) * HIT_RES + To["x0"])[:, None]
    yj = ((np.arange(obsd.shape[1]) + 0.5) * HIT_RES + To["y0"])[None, :]
    obsd[(xi > -0.5) & (xi < 2.5) & (np.abs(yj) < 0.6)] = True
    swing = []
    for grid_ in (None, obsd):
        pth = np.array(hybrid_astar(To, np.zeros((0, 3)), (0.0, 0.0, 0.0), (1.2, 1.6), observed=grid_))
        early = pth[np.hypot(pth[:, 0], pth[:, 1]) < 0.8]
        swing.append(float(np.abs(early[:, 1] + ROOT_AHEAD * np.sin(early[:, 2])).max()))
    # <= since PLAN_STEER 0.85: the wider-turning primitives no longer swing early even unguided
    assert swing[1] <= swing[0] + 1e-9 and swing[1] < 0.45, f"early lateral swing without/with observed: {swing}"

    # (7b) with room on both sides, the Navigator passes a post WIDE (the 0.18 attempt)
    npost = Navigator(open_terrain(), (0.0, 0.0, 0.0), (4.0, 0.0))
    for _ in range(MIN_HITS):
        npost.mark(np.array([2.0]), np.array([0.25]))
    npost.step((0.0, 0.0, 0.0))
    pw = np.array(npost.path)
    prx, pry = pw[:, 0] + ROOT_AHEAD * np.cos(pw[:, 2]), pw[:, 1] + ROOT_AHEAD * np.sin(pw[:, 2])
    po = npost.obstacles((0.0, 0.0, 0.0))[0]                               # the post as mapped
    pclr = float((np.hypot(prx - po[0], pry - po[1]) - po[2] - CAR_R).min())
    assert pclr > 0.15, f"passes the post at {pclr:.2f} m, hugging the 0.10 margin"

    # (7c) ... but never a detour longer than WIDE_LEN x the tight way: a wall with a gap that
    # fits the 0.10 margin and not the wide one; round its end is ~1.3x (taken since 2026-10-01,
    # WIDE_LEN 1.6; through the gap while it was 1.25)
    ng = Navigator(open_terrain(), (0.0, 0.0, 0.0), (4.0, 0.0))
    ys = np.r_[np.arange(0.415, 1.5, 0.1), -np.arange(0.415, 1.5, 0.1)]
    for _ in range(MIN_HITS):
        ng.mark(np.full(len(ys), 2.0), ys)
    ng.step((0.0, 0.0, 0.0))
    pg_ = np.array(ng.path)
    at_wall = pg_[np.argmin(np.abs(pg_[:, 0] + ROOT_AHEAD * np.cos(pg_[:, 2]) - 2.0))]
    assert _length(ng.path) <= WIDE_LEN * 4.2, f"took a long detour: y at the wall {at_wall[1]:.2f}, {_length(ng.path):.1f} m"

    # (8) the speed loop: a car crawling at 10 % of the commanded speed gets more throttle, and
    # none during the first BOOST_AFTER frames of a direction (the ESC brake/neutral phases)
    nv = Navigator(open_terrain(), (0.0, 0.0, 0.0), (5.0, 0.0))
    x, thr, got = 0.0, 0.0, []
    for k in range(60):
        r8 = nv.step((x, 0.0, 0.0), None, thr)
        thr = r8[1]
        got.append(thr)
        x += 0.1 * abs(thr) * V_MAX * 0.05                  # a steep hill: 10 % of the speed
    assert abs(got[BOOST_AFTER - 2]) <= 0.5 + 1e-9, f"boosted before the direction hold: {got[:BOOST_AFTER]}"
    assert abs(got[-1]) > 0.9, f"no boost on a hill: throttle {got[-1]:.2f}"
    nv2, x, thr = Navigator(open_terrain(), (0.0, 0.0, 0.0), (5.0, 0.0), speed_loop=False), 0.0, 0.0
    for _ in range(60):                                   # an own-pose car: never boosted
        thr = nv2.step((x, 0.0, 0.0), None, thr)[1]
        x += 0.1 * abs(thr) * V_MAX * 0.05
    assert abs(thr) <= 0.5 + 1e-9, f"boosted with speed_loop off: {thr:.2f}"

    # (9) boxed in -> no plan -> retreat: back out along the trail it just drove
    nr = Navigator(open_terrain(), (0.0, 0.0, 0.0), (5.0, 0.0))
    for k in range(40):                                   # drove 2 m straight ahead
        nr.step((0.05 * k, 0.0, 0.0), None, 0.5)
    ring = np.array([(2.0 + 0.6 * math.cos(a), 0.6 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 60)])
    for _ in range(MIN_HITS):
        nr.mark(ring[:, 0], ring[:, 1])                   # a ring closes round the car: no plan
    r9 = nr.step((1.95, 0.0, 0.0), None, 0.5)
    assert r9 is not None and r9[2] == "nav_retreat" and r9[1] < 0, f"no retreat when boxed in: {r9}"
    # follow the retreat(s) back out of the ring (the ring on the trail must not abort them),
    # then a real plan round the ring takes over
    pose9, thr9 = np.array([1.95, 0.0, 0.0]), r9[1]
    for _ in range(80):
        r9 = nr.step(tuple(pose9), None, thr9)
        if r9 is None:
            break
        thr9 = r9[1]
        v = thr9 * 0.72 * 0.05
        pose9 = pose9 + np.array([v * math.cos(pose9[2]), v * math.sin(pose9[2]), v * math.tan(r9[0] * STEER_MAX) / WHEELBASE])
    assert pose9[0] < 1.4 and not nr.retreating and nr.fol is not None, \
        f"retreats must back the car out of the ring (x={pose9[0]:.2f}) and then find a plan round it"

    # (10) an obstacle inside the margin but clear of the car blocks every primitive unless it
    # is kept with NO margin: the car must still get a plan that drives away from it
    nh = Navigator(open_terrain(), (0.0, 0.0, 0.0), (4.0, 0.0))
    near_cell = np.array([[BODY[2] + BODY_R + 0.12, 0.0]])      # its edge 5 cm ahead of the bumper
    for _ in range(MIN_HITS):
        nh.mark(near_cell[:, 0], near_cell[:, 1])
    r10 = nh.step((0.0, 0.0, 0.0))
    assert r10 is not None and r10[2].startswith("nav_plan"), f"no plan with a cell 5 cm off the bumper: {r10}"

    # (11) a wedged car remembers a bump at its bumper that the next plan still sees
    for thr in (0.5, -0.5):
        nb = Navigator(open_terrain(), (0.0, 0.0, 0.0), (5.0, 0.0))
        for _ in range(41):
            nb.step((0.0, 0.0, 0.0), None, thr)
        assert footprint_clear(nb.bumps, 0.0, 0.0, 0.0)[0] > 0, "bump dropped by the plan-start filter"
    # ... but the TRAVEL of a 3-point turn (forward, then reverse to the same spot) is not a
    # wedge. Follower off: progress along a plan is tested in (6b), and a real 3-point turn
    # follows planned reverse segments, each of which restarts its progress
    nt = Navigator(open_terrain(), (0.0, 0.0, 0.0), (5.0, 0.0))
    nt.step((0.0, 0.0, 0.0))
    nt.fol, nt.hold = None, 10 ** 9
    for k in range(60):
        x = 0.1 * math.sin(k / 10.0 * math.pi)            # 0.1 m forward and back, 2 s period
        nt.step((x, 0.0, 0.0), None, 0.5 if math.cos(k / 10.0 * math.pi) > 0 else -0.5)
    assert not nt.bumps, "a 3-point turn was taken for a wedge"
    fo, fu = _check_field()
    print(f"pathplan selftest: OK (obstacle field over {fo:.3f} / under {fu:.3f} m; "
          f"wall: clearance {clear:.2f} m, ends {err:.2f} m off; "
          f"corridor reverse: ends {err2:.2f} m off; navigator: {nav.plans} plans, "
          f"wall clearance {min_clr:.2f} m, ends {err3:.2f} m off)")


def _selftest_memory():
    """The dead-end demo in miniature. A divider splits the way to the goal into two 1.4 m lanes;
    the LEFT one is closed. Mission 1 ends up in the left pocket facing its closing wall: no
    forward plan -> a dead-end disc. Its memory is saved; mission 2 (a fresh Navigator, loaded)
    plans from the start and must take the RIGHT lane."""
    import os
    import tempfile
    T = open_terrain(size=20.0, res=0.1)
    goal = (5.5, 0.0)
    pts = [(x, 0.0) for x in np.arange(1.5, 4.01, 0.05)]                       # divider
    pts += [(x, y) for x in np.arange(1.5, 4.01, 0.05) for y in (1.4, -1.4)]   # outer walls
    closed = [(3.55, y) for y in np.arange(0.0, 1.41, 0.05)]                   # the left lane's end
    n1 = Navigator(T, (0.0, 0.0, 0.0), goal, forward_only=True)
    for _ in range(MIN_HITS):
        n1.mark(*np.array(pts + closed).T)
    p = (2.95, 0.7, 0.0)                                   # deep in the left pocket, facing its end
    assert n1.step(p, None, 0.0) is None, "planned a forward way out of a 1.4 m pocket"
    for _ in range(n1.backoff + 1):                        # the hold runs out, it plans again from the same spot
        n1.step(p, None, 0.0)
    assert not n1.dead_ends, "a dead end from the map alone (nothing seen this frame)"
    n1.live = np.array(closed)                             # now the camera sees the closing wall
    for _ in range(n1.backoff + 1):                        # a third failure, the wall in view: a dead end
        n1.step(p, None, 0.0)
    assert len(n1.dead_ends) == 1, n1.dead_ends
    dx, dy, _ = n1.dead_ends[0]
    assert abs(dx - 3.55) < 0.06 and abs(dy - 0.7) < DEAD_SIDE, n1.dead_ends     # on the closing wall, ahead
    n1.backing_out = False
    q = (0.0, 0.0, math.pi / 2)                            # facing open ground (the fork is 1 m to the right)
    n1.live = np.array(pts + closed)
    assert not n1.dead_end(q) and not n1.backing_out, "open ground called a dead end"
    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    n1.save(path)
    n2 = Navigator(T, (0.0, 0.0, 0.0), goal, forward_only=True)
    n2.load(path)
    os.unlink(path)
    assert n2.dead_ends == n1.dead_ends and (n2.hits == n1.hits).all() and (n2.observed == n1.observed).all()
    route, _ = plan_cascade(T, n2.obstacles((0.0, 0.0, 0.0)), (0.0, 0.0, 0.0), goal, n2.observed,
                            n2.margin, 40000, forward_only=True)
    assert route is not None, "no route with the memory loaded"
    mid = np.array([q[:2] for q in route if 2.0 < q[0] < 3.5])
    assert len(mid) and (mid[:, 1] < 0).all(), "mission 2 drove into the remembered dead end"
    print(f"memory selftest: OK (dead end at {dx:.2f},{dy:.2f}; after load the route takes the right lane, "
          f"y {mid[:, 1].min():.2f}..{mid[:, 1].max():.2f} m between the lanes' ends)")


def _scan_sim(pose, cones, reach=2.5, n=64, fov=math.radians(68), cone_r=0.06):
    """A camera-like per-column scan of round obstacles (X right, Z forward, hit, height), reach-limited."""
    x, y, th = pose
    cx, cy = x + C_CAM * math.cos(th), y + C_CAM * math.sin(th)
    a = np.linspace(fov / 2, -fov / 2, n)                          # left -> right
    d = np.stack([np.cos(th + a), np.sin(th + a)], 1)              # (n, 2) ray directions
    rel = cones - [cx, cy]                                         # (m, 2)
    proj = d @ rel.T                                               # (n, m)
    perp2 = (rel ** 2).sum(1)[None, :] - proj ** 2
    t = np.where((proj > 0) & (perp2 < cone_r ** 2), proj - np.sqrt(np.maximum(cone_r ** 2 - perp2, 0)), np.inf).min(1)
    hit = t < reach
    r = np.where(hit, t, reach)
    return np.c_[-r * np.sin(a), r * np.cos(a), hit.astype(float), np.where(hit, 0.3, 0.0)]


def _drive_sim(nav, cones, goal, steps=3000, dt=0.05):
    """Closed loop (20 Hz; the real car runs ~10 Hz -> dt=0.1): the Navigator drives a kinematic car through
    `cones`; when it has no plan a reactive stand-in creeps toward the goal while 0.6 m ahead is clear and,
    like avoid.plan's stuck backup, backs up straight for BACKUP_S after STUCK_S with no way forward. Reverse
    passes only as "fb_backup" (the real car's gate). -> (reached, seconds, contact frames, trail, modes)."""
    STUCK_S, BACKUP_S = 1.0, 1.5                                   # avoid.py's
    pose, thr, trail, modes, contacts, stuck, backup = np.zeros(3), 0.0, [], [], 0, 0.0, 0.0
    for k in range(steps):
        sc = _scan_sim(tuple(pose), cones)
        r = nav.step(tuple(pose), sc, thr, dt)
        if r is None:
            ahead = sc[(np.abs(sc[:, 0]) < 0.25), 1].min()
            gb = math.atan2(goal[1] - pose[1], goal[0] - pose[0]) - pose[2]
            steer, thr, mode = float(np.clip(2.0 * math.remainder(gb, 2 * math.pi), -1, 1)), (0.35 if ahead > 0.6 else 0.0), "fb"
            stuck = stuck + dt if thr == 0.0 else 0.0
            if stuck >= STUCK_S:
                stuck, backup = 0.0, BACKUP_S
            if backup > 0:
                backup -= dt
                steer, thr, mode = 0.0, -0.6, "fb_backup"
        else:
            (steer, thr, mode), stuck, backup = r, 0.0, 0.0
        if thr < 0 and mode != "fb_backup":
            thr = 0.0
        modes.append(mode)
        v = thr * V_MAX * dt
        pose = pose + [v * math.cos(pose[2]), v * math.sin(pose[2]), v * math.tan(steer * STEER_MAX) / WHEELBASE]
        body = np.array([[pose[0] + o * math.cos(pose[2]), pose[1] + o * math.sin(pose[2])] for o in BODY])
        contacts += int((np.hypot(*(body[:, None, :] - cones[None]).transpose(2, 0, 1)) < BODY_R + 0.06).any())
        trail.append(pose.copy())
        root = pose[:2] + ROOT_AHEAD * np.array([math.cos(pose[2]), math.sin(pose[2])])
        if math.hypot(root[0] - goal[0], root[1] - goal[1]) < 0.45:
            return True, (k + 1) * dt, contacts, np.array(trail), modes
    return False, steps * dt, contacts, np.array(trail), modes


def _selftest_deadend_demo(dt=0.05):
    """The user's demo, closed loop. A cone fork 2.5 m ahead; the left lane (shorter to the goal) is a
    3 m pocket whose end the 2.5 m scan cannot see from the fork. Mission 1 goes left and finds the dead
    end; the Navigator itself never reverses (user, 2026-11-23) -- only the reactive stuck backup does;
    mission 2, with mission 1's memory, goes right at once."""
    import os
    import tempfile
    xs = np.arange(2.5, 7.01, 0.2)
    cones = np.array([(x, y) for x in xs for y in (0.0, 1.4, -1.4)] + [(5.5, y) for y in np.arange(0.2, 1.31, 0.2)])
    goal = (8.5, 0.6)
    n1 = Navigator(open_terrain(size=30.0), (0.0, 0.0, 0.0), goal, forward_only=True)
    ok1, t1, c1, tr1, m1 = _drive_sim(n1, cones, goal, dt=dt, steps=int(150 / dt))
    went_left = bool(((tr1[:, 0] > 3.5) & (tr1[:, 1] > 0.3)).any())
    backed = sum(m == "fb_backup" for m in m1)
    assert went_left and n1.dead_ends, (ok1, t1, went_left, n1.dead_ends)
    assert not any(m in ("nav_deadend", "nav_retreat") for m in m1), "a forward-only Navigator reversed"
    assert c1 == 0, f"mission 1 touched a cone on {c1} frames"
    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    n1.save(path)
    n2 = Navigator(open_terrain(size=30.0), (0.0, 0.0, 0.0), goal, forward_only=True)
    n2.load(path)
    os.unlink(path)
    ok2, t2, c2, tr2, m2 = _drive_sim(n2, cones, goal, dt=dt, steps=int(150 / dt))
    assert ok2 and c2 == 0 and not any(m == "fb_backup" for m in m2), (ok2, t2, c2)
    assert not ((tr2[:, 0] > 3.0) & (tr2[:, 0] < 7.0) & (tr2[:, 1] > 0.0)).any(), "mission 2 entered the left lane"
    print(f"dead-end demo selftest @ {1 / dt:.0f} Hz: OK (mission 1: left, dead end, {backed} stuck-backup frames, "
          f"{'goal in %.0f s' % t1 if ok1 else 'NOT out of the pocket in %.0f s' % t1}, 0 contacts; "
          f"mission 2 with memory: right lane at once, goal in {t2:.0f} s, 0 contacts)")


if __name__ == "__main__":
    _selftest_deadend_demo()
    _selftest_deadend_demo(dt=0.1)
    _selftest_memory()
    _selftest()
    _selftest_extrude()
    _selftest_base()
    _selftest_pinned_retreat()
    _selftest_blocked_consistent()
