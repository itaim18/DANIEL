"""Onboard obstacle avoidance from monocular relative disparity (DAv2 on the Hailo-8L).

    plan(disp, ctx) -> dict(steer, throttle, ...)      # pure numpy, runs on the Pi
    python3 avoid.py                                    # self-check

THE TRICK: ground-plane metric calibration, every frame.
DAv2 outputs affine-invariant inverse depth, D = a/Z + b, with a, b unknown and
drifting. For a level camera at height h, a floor point at image row v has
1/Z = (v - cy) / (fy h), LINEAR in v. So in the rows the floor occupies, the per-row
median of D is a line D_f(v) = alpha (v - cy) + beta. Fitting it recovers the affine
map, and metric depth follows for EVERY pixel:
    Z = fy h alpha / (D - beta)
No scale prior and no second sensor; only the camera height and intrinsics. Points
then go to the car frame (lateral X, forward Z, height above ground Y). Anything
standing 5-50 cm above the floor inside the car's swept envelope is an obstacle.

PLANNER v1: score Ackermann arcs. For each candidate steer, sweep the car's
footprint along the arc it would drive for HORIZON metres. The cost is:
  - collision: first obstacle point inside the inflated footprint
  - progress: arc-end bearing vs the goal bearing
  - smoothness: change from the last steer
Throttle scales with the chosen arc's free length. When every arc is blocked it backs
up while steering the other way (recovery), for a few steps.

ctx keys: fx, fy, cx, cy (pixels, in the disparity map's frame), cam_h (m),
goal_bearing (rad, + = left), goal_dist (m), last_steer, state (dict, persisted by
the caller).
"""
from __future__ import annotations

import math

import numpy as np

WHEELBASE = 0.313          # mdp.AckermannPwmActionCfg
STEER_MAX = 0.349          # rad at steer = +-1 (STEER_LIMIT_RAD)
HALF_W = 0.16              # car half-width incl. wheels (track 0.285 + tyre)
MARGIN = 0.10              # extra clearance kept from obstacles
H_MIN, H_MAX = 0.05, 0.50  # obstacle = this band above the floor (m)
HORIZON = 2.2              # arc look-ahead (m)
Z_MAX = 4.0                # ignore depth beyond this (m): DAv2 far field is unreliable
N_ARCS = 21


def fit_floor(disp, cy, v0_frac=0.62, prev=None):
    """Robust per-row line fit of the floor in the bottom image band.

    Returns (alpha, beta, ok). Per-row MEDIANS over the central 70 % of columns, so
    an obstacle covering some columns does not move the row statistic. Trimmed least
    squares: refit after dropping rows whose residual exceeds 2x the median. When too
    few rows are consistent (the whole bottom is an obstacle) keep the previous fit.
    """
    H, W = disp.shape
    v0 = max(int(H * v0_frac), int(cy) + 8)
    rows = np.arange(v0, H - 2)
    c0, c1 = int(W * 0.15), int(W * 0.85)
    med = np.median(disp[rows, c0:c1], axis=1)
    x = rows - cy
    keep = np.ones(len(rows), bool)
    alpha = beta = None
    for _ in range(3):
        if keep.sum() < 6:
            break
        A = np.stack([x[keep], np.ones(keep.sum())], 1)
        (alpha, beta), *_ = np.linalg.lstsq(A, med[keep], rcond=None)
        r = np.abs(med - (alpha * x + beta))
        keep = r <= max(2.0 * np.median(r[keep]), 1e-6)
    ok = alpha is not None and alpha > 1e-6
    if ok:
        # Plausibility: a floor spans the disparity range from the near floor (bottom row)
        # down to the far field at the horizon, where it meets beta. A box face filling the
        # bottom band fits a FLAT line (alpha ~ 4e-5 vs 0.025 expected) that was accepted,
        # stored and reused for the rest of the episode: 0 obstacles, straight into the box.
        far = np.percentile(disp[:max(int(cy) - 4, 1)], 5)
        span = float(np.median(disp[H - 3])) - far
        ok = (alpha * (H - 1 - cy) >= max(0.5 * span, 1e-3 * abs(far) + 1e-6)
              and beta <= far + 0.25 * max(span, 0.0))
    if not ok:
        if prev is not None:
            return prev[0], prev[1], False
        return 1.0, 0.0, False
    return float(alpha), float(beta), True


def obstacle_points(disp, ctx, alpha, beta, stride=2):
    """Metric car-frame obstacle points (X lateral, Z forward) from calibrated disparity."""
    H, W = disp.shape
    fx, fy, cx, cy, h = ctx["fx"], ctx["fy"], ctx["cx"], ctx["cy"], ctx["cam_h"]
    d = disp[::stride, ::stride]
    v = np.arange(0, H, stride, dtype=np.float32)[:, None]
    u = np.arange(0, W, stride, dtype=np.float32)[None, :]
    den = d - beta
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = fy * h * alpha / den
    valid = (den > 1e-6) & (Z > 0.15) & (Z < Z_MAX)
    X = (u - cx) * Z / fx
    Y = h - (v - cy) * Z / fy                       # height above the floor plane
    obs = valid & (Y > H_MIN) & (Y < H_MAX)
    return np.stack([X[obs], Z[obs]], 1), (Z, Y, valid, obs)


V_MAX = 0.72               # m/s at throttle 1 (12 rad/s x 0.06 m; sim: 0.37 m/s at 0.5)
DEADLOCK_FRAMES = 20       # boxed-in stop frames before memory is dropped (v14)
LOOP_MIN_FREE = 1.0        # forward_only: a full-lock loop needs this much collision-checked arc (m)
STUCK_S = 1.0              # near_hit_reverse: no forward way out for this long -> a short straight backup
BACKUP_S = 1.5             # near_hit_reverse: length of every backup, s (incl. the ESC gate's 0.45 s brake + gap).
                           # ponytail: time, not distance (no wheel odometry); calibrate on the floor (~0.3 m wanted)
MEM_S = 4.0                # s an obstacle point is remembered once it is beside or behind the car
RUN_K = 5                  # an obstacle needs this many consecutive flagged rows (px)
RANGE_MODE = "run"         # "run": median calibrated depth of the run; "contact": floor row


def column_scan(disp, ctx, alpha, beta, stride=2):
    """v2 perception, stixel-style: per image column, walk UP from the bottom and stop at
    the first run of RUN_K rows that stand H_MIN above the fitted floor. One (X, Z) point
    per column. Isolated floor speckle (splat noise, grass, DAv2 texture) never makes a
    run, and anything behind the first obstacle is occluded anyway.
    Returns (pts[N, 2] as X right / Z forward, debug tuple like obstacle_points)."""
    H, W = disp.shape
    fx, fy, cx, cy, h = ctx["fx"], ctx["fy"], ctx["cx"], ctx["cy"], ctx["cam_h"]
    d = disp[:, ::stride]
    d = np.concatenate([d[:1], (d[:-2] + d[1:-1] + d[2:]) / 3.0, d[-1:]])   # 3-row smooth
    v_top = int(cy + fy * h / Z_MAX) + 1           # rows above: floor is beyond Z_MAX
    rows = np.arange(v_top, H, dtype=np.float32)
    Zf = fy * h / (rows - cy)                       # floor range of each row
    den = d[v_top:] - beta
    with np.errstate(divide="ignore", invalid="ignore"):
        Zp = np.where(den > 1e-6, fy * h * alpha / den, np.inf)
    obs = Zp < Zf[:, None] * (1.0 - H_MIN / h)      # pixel stands >= H_MIN above the floor
    ob = obs[::-1].astype(np.int32)                 # bottom row first
    cs = np.cumsum(ob, 0)
    win = cs[RUN_K - 1:] - np.vstack([np.zeros((1, ob.shape[1]), np.int32), cs[:-RUN_K]])
    full = win >= RUN_K
    has = full.any(0)
    first = full.argmax(0)                          # bottom-first index of the run start
    vb = (H - 1 - first).astype(np.int64)           # image row of the obstacle's lowest pixel
    cols = np.nonzero(has)[0]
    if RANGE_MODE == "contact":
        Z = Zf[vb[cols] - v_top] * (1.0 - H_MIN / h)
    else:
        Z = np.array([np.median(Zp[max(vb[c] - v_top - RUN_K + 1, 0):vb[c] - v_top + 1, c])
                      for c in cols])
    u = cols * stride
    keep = np.isfinite(Z) & (Z > 0.15) & (Z < Z_MAX)
    pts = np.stack([(u[keep] - cx) * Z[keep] / fx, Z[keep]], 1)
    mask = np.zeros((H, W // stride + (W % stride > 0)), bool)[:, :d.shape[1]]
    for c, v in zip(cols[keep], vb[cols][keep]):
        mask[max(v - RUN_K + 1, 0):v + 1, c] = True
    return pts, (None, None, None, mask[::stride])


RANGE_K = 0.5              # range = floor range of the run's first row x this (bench sweep)
FLOOR_JUMP = 1.3           # max frame-to-frame rise of the floor profile (v9)
FLOOR_Q = 25               # per-row percentile that models the floor (bench sweep)
FLOOR_MODE = "rows"        # "rows": non-parametric per-row floor; "line": global affine fit


def floor_profile(disp, ctx, st):
    """v3 floor model, NON-parametric: the FLOOR_Q-th percentile of each row's disparity
    over the central 80 % of columns, made monotone (the floor only gets nearer going down
    the image) and smoothed in time. DAv2's floor response is NOT a line in image rows
    (it bends near the car), and the v2 line fit left a curved phantom wall across
    every Hailo frame (2026-09-29). A low percentile ignores obstacles covering < 65 % of
    a row: they are nearer than the floor, so they sit above that percentile.
    Returns (f, b): f[v] floor disparity per row, b the far-field disparity (the
    affine offset at infinity), from the region above the horizon."""
    H, W = disp.shape
    c0, c1 = int(W * 0.1), int(W * 0.9)
    f = np.percentile(disp[:, c0:c1], FLOOR_Q, axis=1)
    # Monotone non-decreasing downward, from the HORIZON: above it there is no floor. Run from
    # row 0, a chair seat over the camera (0.42 m, near) was carried down into every floor row,
    # so the floor read nearer than the chair's own tubes and the scan never flagged them, with
    # GT depth too (HITL 04: tube at 0.68 m scanned at 1.16 m; from the horizon 0.76 m).
    v0 = int(ctx["cy"]) + 1
    f[v0:] = np.maximum.accumulate(f[v0:])
    b = float(np.percentile(disp[:max(int(ctx["cy"]) - 4, 1)], 3))
    fb = f - b
    prev = st.get("fprof")
    if prev is not None and prev.shape == fb.shape:
        # v9: the floor cannot come NEARER than FLOOR_JUMP x last frame's (above the far
        # field). A box face filling the bottom of the view IS the 25th percentile of those
        # rows, so it became "floor" and vanished: 0 obstacles, 2.2 m free, 0.45 m from a
        # box (v7 GT, 01_box_offset). DAv2's own frame-to-frame scale jitter passes.
        fb = np.minimum(fb, FLOOR_JUMP * np.maximum(prev, 1e-6))
        fb = 0.5 * fb + 0.5 * prev
    st["fprof"] = fb
    return b + fb, b


VERT_SLOPE, VERT_RUN, VERT_DV = 0.3, 16, 4   # v9 vertical-face detector (bench sweep)
HT_BAND = 0.12             # obstacle height: rows within this fraction of the contact disparity
# v11 floating overhangs (offline on the 50 l22rt episodes, student vs the same rule on GT depth:
# 82-91 % of pulled-in columns confirmed in the chair scenes 00/04/06; the "floating" test cut
# 07's crate-top pulls 3994 -> 371 columns)
OH_LO, OH_HI, OH_Z, OH_N, OH_REL, OH_OPEN, OH_GAP = 0.25, 0.60, 1.5, 4, 0.8, 1.25, 6


def overhang(disp, ctx, f, b, stride=2):
    """Per scan column: the range (m) of a FLOATING overhang -- a surface OH_LO-OH_HI above the
    floor within OH_Z, with open space under it (the column between it and the horizon is
    mostly farther): a seat, a table top -- else inf. Above the horizon there is no floor
    contact to range from, so the floor profile is the ruler: K = Zf (f - b), Z = K / (d - b)
    (median error -2..+8 %, p10 -9..-19 % on obstacle pixels, student, l22rt)."""
    H, W = disp.shape
    fy, cy, h = ctx["fy"], ctx["cy"], ctx["cam_h"]
    v_top = int(cy + fy * h / Z_MAX) + 1
    rows = np.arange(v_top + 4, H - 4)
    va = int(cy) - 2
    if len(rows) < 8 or va < OH_GAP + OH_N:
        return np.full(len(range(0, W, stride)), np.inf)
    K = float(np.median(fy * h / (rows - cy) * (f[rows] - b)))
    d = disp[:va, ::stride]
    Z = K / np.maximum(d - b, 1e-6)
    vv = np.arange(va, dtype=np.float32)[:, None]
    hgt = h + (cy - vv) * Z / fy
    m = (hgt > OH_LO) & (hgt < OH_HI) & (Z > 0.15) & (Z < OH_Z)
    has = m.sum(0) >= OH_N
    vlo = np.where(has, va - 1 - np.argmax(m[::-1], 0), 0)            # lowest overhang row
    # range = its NEAREST part (3rd-nearest pixel): a surface above the camera reaches the
    # horizon at its FAR edge, so the lowest row's range was the back of a seat (1.09 vs 0.70 m)
    zo = np.sort(np.where(m, Z, np.inf), 0)[min(2, va - 1)]
    below = (vv > vlo[None, :]) & (Z < OH_OPEN * zo[None, :])          # near stuff under it
    floating = has & (va - 1 - vlo >= OH_GAP) & (below.sum(0) < 0.5 * np.maximum(va - 1 - vlo, 1))
    return np.where(floating, zo, np.inf)


def _first_run(obs, k, H):
    """Per column: is there a run of k flagged rows, and the image row of the LOWEST run's
    bottom pixel (obs covers rows H-len(obs)..H-1)."""
    ob = obs[::-1].astype(np.int32)                 # bottom row first
    cs = np.cumsum(ob, 0)
    win = cs[k - 1:] - np.vstack([np.zeros((1, ob.shape[1]), np.int32), cs[:-k]])
    full = win >= k
    return full.any(0), (H - 1 - full.argmax(0)).astype(np.int64)


def column_scan_rows(disp, ctx, f, b, stride=2):
    """column_scan against the per-row floor profile. The ratio (D-b)/(f-b) is
    Z_floor/Z of the pixel whatever DAv2's scale; >= 1/(1-H_MIN/h) means the pixel
    stands >= H_MIN above the floor. Range comes from ground-contact GEOMETRY of the
    run's lowest row, so DAv2's scale never enters it."""
    H, W = disp.shape
    fx, fy, cx, cy, h = ctx["fx"], ctx["fy"], ctx["cx"], ctx["cy"], ctx["cam_h"]
    d = disp[:, ::stride]
    d = np.concatenate([d[:1], (d[:-2] + d[1:-1] + d[2:]) / 3.0, d[-1:]])
    v_top = int(cy + fy * h / Z_MAX) + 1
    rows = np.arange(v_top, H, dtype=np.float32)
    Zf = fy * h / (rows - cy)
    den = np.maximum(f[v_top:] - b, 1e-6)[:, None]
    obs = (d[v_top:] - b) / den > 1.0 / (1.0 - H_MIN / h)
    has, vb = _first_run(obs, RUN_K, H)
    vb_r = vb.copy()
    rk = ctx.get("range_k", RANGE_K)              # 0 = the geometric correction (the real car, Brain)
    Zr = np.where(has, Zf[np.clip(vb - v_top, 0, len(Zf) - 1)] * (rk or (1.0 - H_MIN / h)), np.inf)
    # v9 second detector, SCALE-FREE: the floor's disparity rises down each column,
    # (D-b) ~ (v-cy), so slope = dD/dv * (v-cy)/(D-b) is ~1 on floor and ~0 on a vertical
    # face. Independent of the floor profile, so a box face filling the whole bottom of the
    # view (which the percentile floor absorbs) still shows: measured slope p50 0.01-0.13
    # on a box at 0.45 m vs 0.77-0.94 on clean floor (GT and float DAv2). Range = the floor
    # range of the face's lowest row: the face starts AT its contact row.
    k = VERT_DV
    ds = sum(d[:, j:d.shape[1] - 2 + j] for j in range(3)) / 3.0           # 3-column smooth
    ds = np.pad(ds, ((0, 0), (1, 1)), mode="edge")
    g = np.zeros_like(ds)
    g[k:-k] = (ds[2 * k:] - ds[:-2 * k]) / (2 * k)
    vv = np.arange(H, dtype=np.float32)[:, None]
    slope = g / np.maximum(ds - b, 1e-6) * (vv - cy)
    vert = (slope < VERT_SLOPE)[v_top:]
    vert[-k:] = vert[-k - 1]                                                # no gradient at the edge rows
    hv, vbv = _first_run(vert, VERT_RUN, H)
    Zv = np.where(hv, Zf[np.clip(vbv - v_top, 0, len(Zf) - 1)], np.inf)
    Z_all = np.minimum(Zr, Zv)
    cols = np.nonzero(np.isfinite(Z_all))[0]
    Z = Z_all[cols]
    vb = np.where(Zv <= Zr, vbv, vb)
    u = cols * stride
    keep = (Z > 0.15) & (Z < Z_MAX)
    pts = np.stack([(u[keep] - cx) * Z[keep] / fx, Z[keep]], 1)
    mask = np.zeros(d.shape, bool)
    for c, v in zip(cols[keep], vb[cols][keep]):
        mask[max(v - RUN_K + 1, 0):v + 1, c] = True
    # v10: every column as a 2D ray (X, Z, hit): the obstacle's range, or floor visible out to
    # Z_MAX. A 68 deg, 4 m "laser scan" for slam2d.py; no planning decision reads it.
    # Ranges are the EXACT contact geometry (factor 1 - H_MIN/h for the ratio detector),
    # not the planner's deliberately short RANGE_K: a map built from shortened ranges is
    # a shrunken map (synthetic box at 1.5 m read 1.0 m).
    z_ratio = np.where(has, Zf[np.clip(vb_r - v_top, 0, len(Zf) - 1)] * (1.0 - H_MIN / h), np.inf)
    z_geo = np.minimum(z_ratio, Zv)
    zr = np.where(np.isfinite(z_geo) & (z_geo > 0.15) & (z_geo < Z_MAX), z_geo, Z_MAX)
    uu = np.arange(len(Z_all)) * stride
    # 4th column: the obstacle's TOP height above the floor (m). A vertical surface keeps
    # its contact disparity up the column (above the horizon too, where the floor-ratio
    # detectors cannot look); the floor behind a low object drifts off within a row or two.
    # A splat floater or a leaf is a few cm, a leg / cone / box face is tall. (The flagged
    # run alone cannot say: the scan stops at the Z_MAX horizon row, which caps a height at
    # range Z to h (1 - Z / Z_MAX), 7 cm at 2 m.)
    hit = zr < Z_MAX
    ci = np.arange(d.shape[1])
    vbh = np.clip(np.where(Zv <= z_ratio, vbv, vb_r) - 2, 0, H - 1)       # inside the run, not its mixed edge
    dc = np.maximum(d[vbh, ci], 1e-6)
    same = np.abs(d - dc) < HT_BAND * dc
    up = np.zeros(same.shape, np.int32)
    up[0] = same[0]
    for r in range(1, H):
        up[r] = (up[r - 1] + 1) * same[r]
    top = vbh + 2 - up[vbh, ci] + 1
    top_h = np.where(hit, h - (top - cy) * zr / fy, 0.0)
    # v11: FLOATING OVERHANGS are obstacles at their own range. The car fits under a 0.42 m
    # seat but not between the chair's thin tubes, which the student ranges ~0.4 m long (HITL
    # 04: contact in 5/5 l22rt seeds; with sim GT depth the scan misses them too).
    zo = overhang(disp, ctx, f, b, stride)[:len(zr)]
    oh = np.isfinite(zo) & (zo < OH_REL * zr)
    zr, hit, top_h = np.where(oh, zo, zr), hit | oh, np.where(oh, np.maximum(top_h, OH_LO), top_h)
    if oh.any():
        pts = np.vstack([pts, np.stack([(uu[oh] - cx) * zo[oh] / fx, zo[oh]], 1)])
    rays = np.stack([(uu - cx) * zr / fx, zr, hit.astype(np.float64), top_h], 1)
    return pts, (rays, None, None, mask[::stride])


C_REAR = 0.36              # camera ahead of the rear axle: 0.20 root->camera + 0.156 half wheelbase


def arc_poses(steer, length, n=24):
    """Camera (nose) positions + headings in the camera frame (x right, z forward) while
    the car drives `length` m. An Ackermann car turns about its REAR AXLE, so the rear
    axle follows the circle and the camera leads it by C_REAR. Until v5 the camera itself
    rode the circle, which put the modelled body OUTSIDE the turn; the real rear cuts
    INSIDE it (off-tracking) and clipped props on the way round."""
    s = np.linspace(0.0, length, n)
    delta = -steer * STEER_MAX                      # steer + = left; x axis points right
    k = math.tan(delta) / WHEELBASE
    if abs(k) < 1e-6:
        return np.zeros(n), s, np.zeros(n)
    th = k * s
    rx, rz = (1 - np.cos(th)) / k, np.sin(th) / k - C_REAR
    return rx + C_REAR * np.sin(th), rz + C_REAR * np.cos(th), th


CAM_FWD = 0.20             # camera ahead of the car root (the goal is given from the root)


def goal_xz(ctx):
    """Goal in the camera frame (X right, Z forward) from the root-relative bearing/dist."""
    b, d = ctx.get("goal_bearing", 0.0), ctx.get("goal_dist", 9.0)
    return -d * math.sin(b), d * math.cos(b) - CAM_FWD


BODY = ((0.18, 0.0), (0.36, 0.0), (0.46, 0.0))   # (m behind the camera, -): mid, rear axle, rear end
BODY_RAD = HALF_W + 0.04              # tight: the body only needs to clear, the nose keeps MARGIN


def _sweep_hits(pts, x, z, th, rad):
    """Per arc sample: does the car collide there? The NOSE (camera) keeps rad; body
    circles 0.18 / 0.36 / 0.46 m behind it along the heading keep BODY_RAD. v5 checked the
    nose only, and the inner rear cut the corner round props (off-tracking): 16 contact
    steps on 01_box_offset, all while driving forward (2026-09-29).
    Body circles are skipped at the first sample: what is beside the body NOW cannot be
    avoided by picking an arc, only by stopping, which the nose check already does."""
    dx = pts[None, :, 0] - x[:, None]
    dz = pts[None, :, 1] - z[:, None]
    hit = ((dx * dx + dz * dz) < rad * rad).any(1)
    hx, hz = np.sin(th), np.cos(th)                  # heading (x right, z forward)
    for back, _ in BODY:
        bx = pts[None, :, 0] - (x - back * hx)[:, None]
        bz = pts[None, :, 1] - (z - back * hz)[:, None]
        b = ((bx * bx + bz * bz) < BODY_RAD * BODY_RAD).any(1)
        b[0] = False
        hit |= b
    return hit


GRID_R, GRID_C = 4.5, 0.15  # local map: half-size and cell (m), centred on the camera
GEO = True                  # v7: score arcs by GEODESIC progress on the local map


def _shift(a, di, dj, fill):
    n0, n1 = a.shape
    out = np.full_like(a, fill)
    out[max(di, 0):n0 + min(di, 0), max(dj, 0):n1 + min(dj, 0)] = \
        a[max(-di, 0):n0 + min(-di, 0), max(-dj, 0):n1 + min(-dj, 0)]
    return out


def geodesic_field(pts, gx, gz):
    """Distance-to-goal (m) around obstacles on a local grid (rows = z, cols = x).

    Obstacle points (current view + memory) are inflated by the nose radius; unknown
    space counts as free. 8-neighbour relaxation until it stops changing: numpy only, a
    few ms. With it, a wall between the car and the goal makes the way ROUND the wall
    the progress direction; with straight-line progress (v5) no arc made progress
    behind the crate wall, and 3-point turns dithered in place (v5 GT, 07)."""
    n = int(round(2 * GRID_R / GRID_C))
    occ = np.zeros((n, n), bool)
    if len(pts):
        ij = np.floor((pts[:, ::-1] + GRID_R) / GRID_C).astype(int)
        ij = ij[((ij >= 0) & (ij < n)).all(1)]
        occ[ij[:, 0], ij[:, 1]] = True
        r = int(math.ceil((HALF_W + MARGIN) / GRID_C))
        base = occ.copy()
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if di * di + dj * dj <= r * r and (di or dj):
                    occ |= _shift(base, di, dj, False)
    c = int(GRID_R / GRID_C)                       # the camera's cell
    occ[c - 3:c + 1, c - 1:c + 2] = False          # the car's own footprint is where it IS
    g = np.clip(np.floor((np.array([gz, gx]) + GRID_R) / GRID_C).astype(int), 0, n - 1)
    D = np.full((n, n), np.inf)
    if occ[g[0], g[1]]:                            # goal inside an inflated obstacle: nearest free
        free = np.argwhere(~occ)
        g = free[np.argmin(((free - g) ** 2).sum(1))]
        gz, gx = (g + 0.5) * GRID_C - GRID_R
    for di in (-1, 0, 1):                          # seed with EXACT distances: no cell bias
        for dj in (-1, 0, 1):
            a, b = g[0] + di, g[1] + dj
            if 0 <= a < n and 0 <= b < n and not occ[a, b]:
                D[a, b] = math.hypot((a + 0.5) * GRID_C - GRID_R - gz, (b + 0.5) * GRID_C - GRID_R - gx)
    steps = [(di, dj, GRID_C * math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
    P = np.full((n + 2, n + 2), np.inf)            # inf border: P[1-di:, 1-dj:] == _shift(D, di, dj, inf), no allocs
    for _ in range(4 * n):
        P[1:-1, 1:-1] = D
        Dn = D
        for di, dj, cost in steps:
            Dn = np.minimum(Dn, P[1 - di:1 - di + n, 1 - dj:1 - dj + n] + cost)
        Dn[occ] = np.inf
        if np.array_equal(Dn, D):
            break
        D = Dn
    return D


def _geo_at(D, x, z):
    """Geodesic distance at points, sub-cell: min over the 3x3 cells around each point of
    (cell value + exact distance to that cell's centre). A plain cell lookup was 0.18 m off
    and faked 0.15 m of 'progress' for any arc that crossed into a neighbour cell."""
    n = D.shape[0]
    i0 = np.floor((z + GRID_R) / GRID_C).astype(int)
    j0 = np.floor((x + GRID_R) / GRID_C).astype(int)
    best = np.full(np.shape(x), np.inf)
    I = [np.clip(i0 + d, 0, n - 1) for d in (-1, 0, 1)]   # 6 clips, not 18 (Pi: 5 ms/frame of np.clip overhead)
    J = [np.clip(j0 + d, 0, n - 1) for d in (-1, 0, 1)]
    for i in I:
        for j in J:
            off = np.hypot((i + 0.5) * GRID_C - GRID_R - z, (j + 0.5) * GRID_C - GRID_R - x)
            best = np.minimum(best, D[i, j] + off)
    return best


def score_arcs(pts, ctx):
    """Free length and score for each candidate steer.

    v5 scores PROGRESS: how much nearer the goal each arc's end point gets, driving only
    as far as it is free (and no farther than the goal). v1-v4 scored the arc's end
    HEADING against the goal bearing, weighted below free length, so the car preferred
    long free arcs, overshot, and parked the goal inside its 0.86 m turning circle
    (v4 GT, 01_box_offset: 0.91 m away at +64 deg, stuck for good)."""
    steers = np.linspace(-1.0, 1.0, N_ARCS)
    last = ctx.get("last_steer", 0.0)
    gx, gz = goal_xz(ctx)
    gd = math.hypot(gx, gz)
    # v11: the SLAM map's obstacles (ctx["map_pts"], camera frame) join the GEODESIC field
    # only -- routing remembers every wall seen this episode, while collision checks stay on
    # what the camera sees now + short memory, so a misaligned map can neither block a real
    # gap nor hide a real obstacle.
    mp = ctx.get("map_pts")
    if mp is not None and len(mp):
        # only where the camera cannot look NOW (beside / behind / beyond reach): in view the
        # fresh scan is the truth. A map blob ahead -- one chair seen from many poses with
        # ~0.2 m pose error, inflated -- walled off the goal and the car reversed 150 of 271
        # steps with 2.2 m free ahead (v11 HITL, 00_chair_center).
        tl = ctx["cx"] / ctx["fx"]
        X, Z = mp[:, 0], mp[:, 1]
        mp = mp[(Z < 0) | (X < -tl * Z) | (X > tl * Z) | (np.hypot(X, Z) > Z_MAX)]
    D = geodesic_field(np.vstack([pts, mp]) if mp is not None and len(mp) else pts, gx, gz) if GEO else None
    d0 = float(_geo_at(D, np.array([0.0]), np.array([0.0]))[0]) if GEO else np.inf
    rad = HALF_W + MARGIN
    s_along = np.linspace(0, HORIZON, 24)
    free, scores, prog = np.zeros(N_ARCS), np.zeros(N_ARCS), np.zeros(N_ARCS)
    for i, s in enumerate(steers):
        x, z, th = arc_poses(s, HORIZON)
        fl = HORIZON
        if len(pts):
            hit = _sweep_hits(pts, x, z, th, rad)
            if hit.any():
                fl = float(s_along[np.argmax(hit)])
        free[i] = fl
        k = s_along <= fl + 1e-6                     # the drivable part of the arc
        if np.isfinite(d0):                          # geodesic: round the obstacles
            dg = _geo_at(D, x[k], z[k])
            prog[i] = d0 - float(dg.min()) if k.any() and np.isfinite(dg).any() else 0.0
        else:                                        # no known way (or v5 mode): straight line
            dend = np.hypot(x[k] - gx, z[k] - gz).min() if k.any() else gd
            prog[i] = gd - dend
        scores[i] = 2.0 * prog[i] / HORIZON + 0.5 * min(fl, HORIZON) / HORIZON - 0.3 * abs(s - last)
    return steers, free, scores, prog


def reverse_clear(pts, steer, length=0.5):
    """Can the car back up `length` m at this steer without its body touching a KNOWN
    point (memory: the camera cannot see behind)? The rear swings sideways when reversing
    with lock, so the whole body is swept along the reverse arc. v6 only looked straight
    back, and backed into a cone it had just passed for 202 steps (GT, 08)."""
    if not len(pts):
        return True
    x, z, th = arc_poses(steer, -length, n=12)
    return not _sweep_hits(pts, x, z, th, HALF_W + 0.05).any()


def dead_reckon(mem, steer, throttle, dt):
    """Move remembered car-frame points (X right, Z fwd, age) by one command's motion.
    Command-based odometry: the Traxxas has no encoders and the Pi no IMU."""
    ds = throttle * V_MAX * dt
    dth = ds * math.tan(steer * STEER_MAX) / WHEELBASE          # + = left (CCW)
    X = mem[:, 0] + ds * math.sin(dth / 2)                       # car slid left by -X
    Z = mem[:, 1] - ds * math.cos(dth / 2)
    c, s_ = math.cos(dth), math.sin(dth)
    return np.stack([X * c + Z * s_, -X * s_ + Z * c, mem[:, 2] + dt], 1)


def in_view(pts, ctx, shape):
    """Car-frame points the camera sees now: inside the horizontal FOV, beyond the image's
    bottom row on the floor, and within Z_MAX."""
    H, W = shape
    tl, tr = ctx["cx"] / ctx["fx"], (W - ctx["cx"]) / ctx["fx"]
    z_min = ctx["fy"] * ctx["cam_h"] / max(H - 1 - ctx["cy"], 1.0)
    X, Z = pts[:, 0], pts[:, 1]
    return (Z >= z_min) & (Z < Z_MAX) & (X > -tl * Z) & (X < tr * Z)


def in_body(pts):
    """Points inside the car's own footprint NOW: nothing can be there, so a remembered
    point that dead-reckoning slid into the body is drift (wheel slip against a mound
    makes command odometry lie), not an obstacle. Left in, it blocked every arc 1.5 m
    from the goal on open floor (v8 GT, 05_person_on_path)."""
    inside = np.zeros(len(pts), bool)
    for back in (0.0,) + tuple(b for b, _ in BODY):
        inside |= np.hypot(pts[:, 0], pts[:, 1] + back) < HALF_W
    return inside


def beside_or_behind(pts, ctx, shape):
    """Memory keeps only what the camera CANNOT re-check: points outside the horizontal
    FOV (beside the car while it turns past them) or behind it. Under the nose is not
    memory: v3 remembered bottom-row detections ranged at ~0.18 m, and they blocked
    every arc for 2 s, so the car reversed 375 of 421 steps away from its goal
    (2026-09-29)."""
    H, W = shape
    tl, tr = ctx["cx"] / ctx["fx"], (W - ctx["cx"]) / ctx["fx"]
    X, Z = pts[:, 0], pts[:, 1]
    return (Z < 0) | (X < -tl * Z) | (X > tr * Z)


def plan(disp, ctx):
    """One control step: disparity -> {steer, throttle, ...}. ctx['state'] persists;
    ctx['dt'] is the control period (s)."""
    st = ctx.setdefault("state", {})
    alpha, beta, ok = fit_floor(disp, ctx["cy"], prev=st.get("floor"))
    if ok:                                           # only a PLAUSIBLE fit becomes the prior
        st["floor"] = (alpha, beta)
    if FLOOR_MODE == "rows":
        f, b = floor_profile(disp, ctx, st)
        pts, dbg = column_scan_rows(disp, ctx, f, b)
    else:
        pts, dbg = column_scan(disp, ctx, alpha, beta)
    if ctx.get("veto") is not None:                  # stereo_veto: the D435's metric stereo checks the near hits
        pts, dbg, ctx["vetoed"] = ctx["veto"](pts, dbg)
    # v2 memory: obstacles that LEFT the view (beside or under the nose while turning past
    # them) are dead-reckoned for MEM_S; inside the view the current frame is trusted.
    mem = st.get("mem", np.zeros((0, 3)))
    if len(mem):
        mem = dead_reckon(mem, *st.get("last_cmd", (0.0, 0.0)), ctx.get("dt", 0.05))
        mem = mem[(mem[:, 2] < MEM_S) & beside_or_behind(mem, ctx, disp.shape)
                  & (np.hypot(mem[:, 0], mem[:, 1]) < HORIZON + 1.0) & ~in_body(mem)]
    st["mem"] = np.vstack([mem, np.c_[pts, np.zeros(len(pts))]]) if len(pts) else mem
    allp = st["mem"][:, :2]
    steers, free, scores, prog = score_arcs(allp, ctx)
    base = {"floor_ok": ok, "n_obs": int(len(pts)), "n_mem": int(len(mem)), "steers": steers,
            "free_arr": free, "dbg": dbg, "alpha": alpha, "beta": beta, "scan": dbg[0]}

    def out(steer, throttle, mode, fl):
        st["last_cmd"] = (float(steer), float(throttle))
        return {"steer": float(steer), "throttle": float(throttle), "mode": mode, "free": float(fl), **base}

    i = int(np.argmax(scores))
    fl = free[i]
    bearing = ctx.get("goal_bearing", 0.0)
    back = st.get("backup", 0)
    if back > 0:                                     # committed reverse (3-point turn)
        done = st.get("backup_done", 0)
        turned = abs(bearing) < 0.5 and prog[i] > 0.2 and done >= 8
        if not reverse_clear(allp, st["backup_steer"], 0.3) or turned:
            st["backup"] = 0
        else:
            st["backup"], st["backup_done"] = back - 1, done + 1
            return out(st["backup_steer"], -0.6, "backup", free.max())
    st["blocked"] = st.get("blocked", 0) + 1 if fl < 0.35 else 0
    # No forward arc gets nearer: the goal sits behind an obstacle or inside the turning
    # circle. Reverse with the wheels turned AWAY from where the nose must go: backing up
    # with steer -1 swings the nose LEFT. Side = the goal's side, or the freer side when
    # the goal is straight ahead.
    stalled = prog[i] < 0.05 and ctx.get("goal_dist", 9.0) > 0.3
    if st["blocked"] >= 2 or stalled:
        side = np.sign(bearing) if abs(bearing) > 0.3 else \
            np.sign(free[steers > 0].sum() - free[steers < 0].sum()) or 1.0
        near_hit = st["blocked"] >= 2                # < 0.35 m free on two frames: about to hit
        if not (ctx.get("forward_only") and ctx.get("near_hit_reverse")):
            st["blocked"] = 0                        # near-hit mode keeps counting: a goal behind (stalled)
                                                     # fires every frame and would reset it before it reaches 2
        # near_hit_reverse (the real car since 2026-11-17): forward-only EXCEPT this near-hit backup --
        # the 3-point turn for a goal behind (stalled) stays a forward loop.
        if ctx.get("forward_only") and not (near_hit and ctx.get("near_hit_reverse")):
            # No reverse (the real car's XL-5HV never reversed from the PCA until 2026-11-17). A
            # goal behind / inside the turning circle: a FORWARD loop at full lock toward it, at
            # the slowest drive throttle, ONLY while LOOP_MIN_FREE of that arc is clear of
            # everything seen or remembered (car half-width + MARGIN) -- re-checked every frame.
            # Otherwise stop and look again (deadlock below). Never a blind move.
            j = int(np.argmin(np.abs(steers - side)))
            if free[j] >= LOOP_MIN_FREE:
                return out(steers[j], 0.35, "loop", free[j])
        else:
            for rs in (-float(side), 0.0, float(side)):  # preferred swing, straight, other lock
                if reverse_clear(allp, rs):
                    n = max(4, round(BACKUP_S / max(ctx.get("dt", 0.05), 1e-3))) if ctx.get("near_hit_reverse") else 24
                    st["backup"], st["backup_done"], st["backup_steer"], st["blocked"] = n, 0, rs, 0
                    return out(rs, -0.6, "backup", fl)
        # boxed in front AND behind: stay put -- but not forever. The scene does not change
        # while the car does not move, so the same decision repeats: 198 stop steps in a row
        # on HITL v10stu1 08. After 1 s, forget the remembered points (cones already passed,
        # dead-reckoning drift) and decide again on fresh perception only.
        if ctx.get("forward_only") and ctx.get("near_hit_reverse"):
            # a dead end beyond the near-hit distance (a table leg at 0.48 m, no room to loop: the first stand run
            # waited forever): after STUCK_S without a forward way out, back up STRAIGHT for BACKUP_S if the swept
            # body behind is clear of everything seen or remembered, then the forward planners try again
            st["stuck_s"] = st.get("stuck_s", 0.0) + ctx.get("dt", 0.05)
            if st["stuck_s"] >= STUCK_S and reverse_clear(allp, 0.0):
                n = max(4, round(BACKUP_S / max(ctx.get("dt", 0.05), 1e-3)))
                st["backup"], st["backup_done"], st["backup_steer"], st["blocked"], st["stuck_s"] = n, 0, 0.0, 0, 0.0
                return out(0.0, -0.6, "backup", fl)
        st["deadlock"] = st.get("deadlock", 0) + 1
        if st["deadlock"] >= DEADLOCK_FRAMES:
            st["mem"], st["deadlock"] = np.zeros((0, 3)), 0
        return out(steers[i], 0.0, "stop", fl)
    st["deadlock"], st["stuck_s"] = 0, 0.0
    if fl < 0.35:                                    # first blocked frame: stop and look again
        return out(steers[i], 0.0, "stop", fl)
    throttle = float(np.clip(0.35 + 0.65 * (fl - 0.35) / (HORIZON - 0.35), 0.35, 1.0))
    if ctx.get("goal_dist", 9.0) < 1.2:              # slow near the goal: turn, don't overshoot
        throttle = min(throttle, 0.45)
    return out(steers[i], throttle, "drive", fl)


def render(disp, out, ctx, size=(392, 224)):
    """Debug view (BGR uint8): disparity colormap, obstacle pixels in cyan, the chosen
    arc projected into the image, plus a bird's-eye inset of free length per arc."""
    import cv2
    lo, hi = np.percentile(disp, [1, 99])
    img = cv2.applyColorMap((np.clip((disp - lo) / (hi - lo + 1e-8), 0, 1) * 255)
                            .astype(np.uint8), cv2.COLORMAP_INFERNO)
    Z, Y, valid, obs = out["dbg"]
    m = cv2.resize(obs.astype(np.uint8), (img.shape[1], img.shape[0]),
                   interpolation=cv2.INTER_NEAREST).astype(bool)
    img[m] = (0.4 * img[m] + 0.6 * np.array([255, 255, 0])).astype(np.uint8)
    # chosen arc on the floor, projected with the same pinhole
    x, z, _ = arc_poses(out["steer"], min(out["free"], HORIZON), n=30)
    fx, fy, cx, cy, h = ctx["fx"], ctx["fy"], ctx["cx"], ctx["cy"], ctx["cam_h"]
    for xi, zi in zip(x, z):
        if zi > 0.2:
            u = int(cx + fx * xi / zi)
            v = int(cy + fy * h / zi)
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 3, (0, 255, 0) if out["mode"] == "drive" else (0, 0, 255), -1)
    # free-length bars per arc (left of the car on the left)
    fa = out["free_arr"]
    bw = img.shape[1] // len(fa)
    for i, f in enumerate(fa[::-1]):
        hbar = int(40 * f / HORIZON)
        cv2.rectangle(img, (i * bw, img.shape[0] - hbar), ((i + 1) * bw - 1, img.shape[0]),
                      (80, 200, 80) if f >= HORIZON - 1e-6 else (60, 60, 220), -1)
    cv2.putText(img, f"{out['mode']} s{out['steer']:+.2f} t{out['throttle']:+.2f} "
                f"free {out['free']:.2f}m obs {out['n_obs']}", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return cv2.resize(img, size) if img.shape[1::-1] != size else img


def _selftest():
    """Synthetic scene: flat floor + a box ahead-right. Checks the floor calibration
    recovers metric depth, the box is found at the right range, and the planner steers
    left, with DAv2-like unknown affine (a, b) applied to the inverse depth."""
    H, W = 224, 392
    fx = fy = 291.1
    cx, cy, h = 196.0, 112.0, 0.20
    v = np.arange(H, dtype=np.float64)[:, None] * np.ones((1, W))
    u = np.ones((H, 1)) * np.arange(W, dtype=np.float64)[None, :]
    with np.errstate(divide="ignore"):
        Zf = np.where(v > cy, fy * h / (v - cy), 60.0)          # floor, far "sky" above horizon
    Z = Zf.copy()
    # box: 0.4 m wide, 0.4 m tall, front face at Z=1.5 m, lateral centre +0.25 m (right)
    Zb, xb, hb = 1.5, 0.25, 0.40
    X = (u - cx) * Zb / fx
    top_v = cy - (hb - h) * fy / Zb                               # box top row
    bot_v = cy + h * fy / Zb                                      # floor contact row
    box = (np.abs(X - xb) < 0.2) & (v >= top_v) & (v <= bot_v)
    Z[box] = Zb
    a, b = 3.7, 0.9                                               # unknown DAv2 affine
    disp = (a / Z + b).astype(np.float32)
    ctx = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "cam_h": h, "goal_bearing": 0.0,
           "goal_dist": 5.0, "last_steer": 0.0}
    alpha, beta, ok = fit_floor(disp, cy)
    assert ok, "floor fit failed on a clean floor"
    Zhat = fy * h * alpha / (disp[box] - beta)
    err = np.median(np.abs(Zhat - Zb)) / Zb
    assert err < 0.10, f"box range off by {err:.1%}"
    out = plan(disp, ctx)
    assert out["mode"] == "drive", out["mode"]
    assert out["steer"] > 0.1, f"box is to the RIGHT, must steer left (+), got {out['steer']:+.2f}"
    # scan column 3 = obstacle top height: the 0.40 m box reads tall (the rows above v_top
    # are followed by disparity), a 3 cm floater on the floor at the same range reads short
    rays = np.asarray(out["scan"])
    tb = rays[(rays[:, 2] > 0) & (np.abs(rays[:, 0] - xb) < 0.1), 3]
    assert len(tb) and np.median(tb) > 0.25, f"box top height {np.median(tb) if len(tb) else None}"
    Zl = Zf.copy()
    low = (np.abs(X - xb) < 0.2) & (v >= cy + (h - 0.03) * fy / Zb) & (v <= bot_v)
    Zl[low] = Zb
    rl = np.asarray(plan((a / Zl + b).astype(np.float32), dict(ctx, state={}))["scan"])
    tl = rl[(rl[:, 2] > 0) & (np.abs(rl[:, 0] - xb) < 0.1), 3]
    assert not len(tl) or np.median(tl) < 0.06, f"3 cm floater reads {np.median(tl):.3f} m tall"
    # a seat over the camera (its underside, background below it) must not become the floor:
    # the box face is caught anyway (vertical detector), so check the floor model itself
    Zs = Z.copy()
    Zs[(v < cy / 2) & ~box] = 0.4
    fs, _ = floor_profile((a / Zs + b).astype(np.float32), ctx, {})
    fl = a / Zf[:, 0] + b
    assert np.all(fs[int(cy) + 2:] < 1.05 * fl[int(cy) + 2:]), "a seat above the horizon raised the floor"
    with np.errstate(divide="ignore", invalid="ignore"):          # a 0.42 m seat 0.7-1.1 m ahead
        Zp = np.where(v < cy, fy * (0.42 - h) / np.maximum(cy - v, 1e-6), np.inf)
        seat = (Zp >= 0.7) & (Zp <= 1.1) & (np.abs((u - cx) * Zp / fx) < 0.25)
    Zo = Zf.copy()
    Zo[seat] = Zp[seat]
    ro = np.asarray(plan((a / Zo + b).astype(np.float32), dict(ctx, state={}))["scan"])
    under = ro[np.abs(ro[:, 0]) < 0.15]
    assert len(under) and (under[:, 2] > 0).all() and np.median(under[:, 1]) < 0.85, \
        f"a seat 0.7 m ahead is not an obstacle: {under[:, 1:3] if len(under) else None}"
    zb_ = rays[(rays[:, 2] > 0) & (np.abs(rays[:, 0] - xb) < 0.1), 1]
    assert abs(np.median(zb_) - Zb) < 0.1, f"overhang moved the box: {np.median(zb_):.2f}"
    blocked = plan(np.full_like(disp, disp.max()), dict(ctx))     # wall everywhere
    assert blocked["mode"] in ("stop", "backup", "drive")
    floor = (a / Zf + b).astype(np.float32)                        # empty scene
    t3 = plan(floor, dict(ctx, state={}, goal_bearing=math.radians(100), goal_dist=0.7))
    assert t3["mode"] == "backup" and t3["steer"] < 0, \
        f"goal inside the turning circle on the left: reverse steering right, got {t3['mode']} {t3['steer']:+.2f}"
    t3f = plan(floor, dict(ctx, state={}, goal_bearing=math.radians(100), goal_dist=0.7, forward_only=True))
    assert t3f["mode"] == "loop" and t3f["throttle"] > 0 and t3f["steer"] > 0, \
        f"forward-only, goal inside the turning circle on the left: loop left, got {t3f['mode']} {t3f['steer']:+.2f}"
    bf_ctx = dict(ctx, state={}, forward_only=True)
    bf = [plan(np.full_like(disp, disp.max()), bf_ctx) for _ in range(3)]
    assert all(o["throttle"] >= 0 for o in bf), f"forward-only, walled in: must not reverse, got {[o['mode'] for o in bf]}"
    nh_ctx = dict(ctx, state={}, forward_only=True, near_hit_reverse=True)
    nh = [plan(np.full_like(disp, disp.max()), nh_ctx) for _ in range(3)]     # a wall at the bumper, 3 frames
    assert nh[0]["throttle"] >= 0 and any(o["mode"] == "backup" and o["throttle"] < 0 for o in nh[1:]), \
        f"near-hit reverse: a wall at the bumper must back up from the 2nd blocked frame, got {[o['mode'] for o in nh]}"
    dead = (a / np.minimum(Zf, 0.55) + b).astype(np.float32)        # a wall 0.55 m ahead: free ~0.38 m, no loop room
    dc = dict(ctx, state={}, goal_bearing=0.0, goal_dist=2.0, forward_only=True, near_hit_reverse=True, dt=0.1)
    dz = [plan(dead, dc) for _ in range(30)]
    first = next((k for k, o in enumerate(dz) if o["throttle"] < 0), None)
    assert first is not None and 9 <= first <= 11 and dz[first]["mode"] == "backup" and dz[first]["steer"] == 0.0, \
        f"dead end beyond near-hit: a straight backup after STUCK_S, got {[o['mode'] for o in dz]}"
    nb = sum(1 for o in dz[first:first + 20] if o["throttle"] < 0)
    assert 10 <= nb <= 16, f"a backup lasts about BACKUP_S (15 frames at dt 0.1), got {nb}"
    dz0 = [plan(dead, dict(dc, state={}, near_hit_reverse=False)) for _ in range(30)]
    assert all(o["throttle"] >= 0 for o in dz0), "strict forward-only must still never reverse at a dead end"
    t3n = plan(floor, dict(ctx, state={}, goal_bearing=math.radians(100), goal_dist=0.7, forward_only=True,
                           near_hit_reverse=True))
    assert t3n["mode"] == "loop" and t3n["throttle"] > 0, \
        f"near-hit reverse, goal behind but nothing near: forward loop, never a 3-point reverse; got {t3n['mode']}"
    far = plan(floor, dict(ctx, state={}, goal_bearing=math.radians(20), goal_dist=3.0))
    assert far["mode"] == "drive" and far["steer"] > 0, f"open floor, goal ahead-left: {far}"
    wall = np.stack([np.linspace(-0.9, 0.7, 33), np.full(33, 1.3)], 1)  # wall across, gap-free
    st_, fr_, sc_, pr_ = score_arcs(wall, dict(ctx, goal_bearing=0.0, goal_dist=3.5))
    ib = int(np.argmax(sc_))
    assert abs(st_[ib]) >= 0.3 and pr_[ib] > 0.1, f"wall ahead: must head round it, got steer {st_[ib]:+.2f}"
    import time
    t0 = time.perf_counter(); geodesic_field(wall, 0.0, 3.3); gms = (time.perf_counter() - t0) * 1e3
    # v9: a box face filling the bottom of the view must NOT become the floor
    st9 = {"fprof": None}
    c9 = dict(ctx, state={}, goal_bearing=0.0, goal_dist=3.0)
    plan(floor, c9)                                               # a clean floor first
    Zw = Zf.copy()
    top = cy - (0.5 - h) * fy / 0.45                               # 0.5 m tall box face, 0.45 m ahead:
    Zw[(v >= top) & (np.abs(u - cx) < 0.45 * W)] = 0.45            # its base is below the image bottom
    wall_disp = (a / Zw + b).astype(np.float32)
    for _ in range(8):                                             # the approach: several frames
        o9 = plan(wall_disp, c9)
    assert o9["free"] < 0.5, f"box filling the view must block, got free {o9['free']:.2f} m, {o9['n_obs']} pts"
    # v14: boxed in front AND behind -> stop, but drop memory after DEADLOCK_FRAMES
    cd = dict(ctx, state={}, goal_bearing=0.0, goal_dist=3.0)
    for _ in range(3):
        plan(wall_disp, cd)
    behind = np.array([[x, z, 0.0] for x in np.linspace(-0.6, 0.6, 13) for z in (-0.5, -0.7)])
    modes = []
    for _ in range(25):                                           # DEADLOCK_FRAMES = 20 must fire within 25
        cd["state"]["mem"] = behind if len(cd["state"].get("mem", [])) else cd["state"].get("mem", behind)
        modes.append(plan(wall_disp, cd)["mode"])
        if not len(cd["state"]["mem"]):
            break
    assert "stop" in modes and not len(cd["state"]["mem"]), f"deadlock never broken: {modes[-5:]}"
    cone_behind_left = np.array([[-0.25, -0.75]])
    assert not reverse_clear(cone_behind_left, 1.0), "reversing with left lock swings the rear into it"
    assert reverse_clear(cone_behind_left, -1.0), "right lock swings the rear away from it"
    r = dead_reckon(np.array([[0.0, 1.0, 0.0]]), 1.0, 1.0, 0.5)          # hard left, 0.36 m
    assert r[0, 0] > 0.2 and r[0, 1] < 1.0, f"a left turn must move a point ahead to the right: {r}"
    print(f"avoid selftest: OK (box range err {err:.1%}, steer {out['steer']:+.2f}, "
          f"free {out['free']:.2f} m, {out['n_obs']} obstacle points; wall -> steer {st_[ib]:+.2f}, "
          f"geodesic field {gms:.1f} ms)")


if __name__ == "__main__":
    _selftest()
