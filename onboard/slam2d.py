"""2D SLAM from the planner's per-column scan: log-odds occupancy grid + correlative scan
matching, seeded by command dead-reckoning. Pure numpy, sized for the Pi 5.

The planner already turns every depth frame into a METRIC 2D scan: per image column, the
ground-contact range of the first obstacle, or "floor visible up to Z_MAX" (avoid.py
column_scan_rows, range from ground-plane geometry, so DAv2's unknown scale never enters).
That is a laser scan with a 68 deg fan and 4 m reach, so the classic 2D-lidar recipe
applies:
  predict  bicycle model on the rear axle from the COMMANDED steer/throttle (no encoders)
  match    correlative search (Olson 2009) around the prediction: score every candidate
           pose by the likelihood field of the map at the scan's obstacle endpoints
  update   ray-cast each column into the log-odds grid: free cells along the ray,
           occupied at the endpoint
The pose is the rear-axle pose in a world frame whose origin is the start pose (x forward,
y left, theta CCW).

    python3 slam2d.py       # self-check: a car driving past a wall; SLAM beats dead-reckoning
"""
from __future__ import annotations

import math

import numpy as np

WHEELBASE, STEER_MAX, V_MAX = 0.313, 0.349, 0.72   # avoid.py
C_REAR = 0.36                                       # camera ahead of the rear axle (avoid.py)
L_OCC, L_FREE, L_MIN, L_MAX = 0.85, -0.4, -2.0, 3.5
SEARCH_XY, SEARCH_TH = 0.15, math.radians(6.0)
# Motion prior on the correlative score. Without it the search slides BACK along a wall:
# the far end the car just revealed is not in the map yet, so a backward shift always
# overlaps more known map (map-edge bias; the self-test pinned the pose at x=0.1 m
# while the car drove 2 m).
PRIOR_W, PRIOR_XY, PRIOR_TH = 0.15, 0.08, math.radians(4.0)
# With a ground-VO prediction (good to ~1 cm/frame) the prior is much tighter: matching
# then only corrects slow drift. With the command-odometry prior, matching made VO worse
# (HITL v10slamvo 00: VO-only ATE 0.23 m, VO + match 0.36 m).
PRIOR_XY_VO, PRIOR_TH_VO = 0.02, math.radians(1.0)
R_MATCH = 3.0        # m: only nearer hits are matched (monocular range error grows as Z^2, and the
                     # newly revealed end of a wall always appears near the 4 m reach)


class Slam2D:
    def __init__(self, res=0.05, size_m=40.0):
        self.res = res
        self.n = int(size_m / res)
        self.L = np.zeros((self.n, self.n), np.float32)      # log-odds, index [ix, iy]
        self.o = self.n // 2                                   # world origin cell
        self.pose = np.zeros(3)                                # x, y, theta (rear axle)
        self.frames = 0

    # ---------------------------------------------------------------- geometry
    def cam(self, pose=None):
        x, y, th = self.pose if pose is None else pose
        return x + C_REAR * math.cos(th), y + C_REAR * math.sin(th)

    def to_world(self, pts, pose):
        """Camera-frame points (X right, Z forward) -> world xy for a rear-axle pose."""
        x, y, th = pose
        cx, cy = x + C_REAR * math.cos(th), y + C_REAR * math.sin(th)
        c, s = math.cos(th), math.sin(th)
        return np.stack([cx + pts[:, 1] * c + pts[:, 0] * s, cy + pts[:, 1] * s - pts[:, 0] * c], 1)

    def cell(self, xy):
        return np.floor(xy / self.res).astype(np.int64) + self.o

    # ---------------------------------------------------------------- steps
    def predict(self, steer, throttle, dt):
        v = throttle * V_MAX
        x, y, th = self.pose
        w = v * math.tan(steer * STEER_MAX) / WHEELBASE             # steer + = left = CCW
        self.pose = np.array([x + v * dt * math.cos(th + w * dt / 2),
                              y + v * dt * math.sin(th + w * dt / 2), th + w * dt])

    def likelihood(self, i0, j0, m):
        """Occupied cells in an m x m window at (i0, j0), blurred into a smooth field
        (Olson's lookup table), [0, 1]. Windowed: the full 800x800 grid per frame is
        too slow for the Pi."""
        f = (self.L[i0:i0 + m, j0:j0 + m] > 0.5).astype(np.float32)
        for _ in range(2):                                     # ~2-cell smooth, separable
            f = np.maximum(f, 0.6 * (np.roll(f, 1, 0) + np.roll(f, -1, 0)) / 2)
            f = np.maximum(f, 0.6 * (np.roll(f, 1, 1) + np.roll(f, -1, 1)) / 2)
        return f

    def match(self, hits, tight=False):
        """Correlative search around the predicted pose; returns (pose, score gain)."""
        pxy, pth = (PRIOR_XY_VO, PRIOR_TH_VO) if tight else (PRIOR_XY, PRIOR_TH)
        if len(hits) < 12 or self.frames < 3:
            return self.pose, 0.0
        m = int(10.0 / self.res)                               # 10 x 10 m window round the camera
        ci, cj = self.cell(np.array([self.cam()]))[0]
        i0, j0 = int(np.clip(ci - m // 2, 0, self.n - m)), int(np.clip(cj - m // 2, 0, self.n - m))
        field = self.likelihood(i0, j0, m)
        k = int(round(SEARCH_XY / self.res))
        off = np.stack(np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1), indexing="ij"), -1).reshape(-1, 2)
        prior = PRIOR_W * (off ** 2).sum(1) * self.res ** 2 / pxy ** 2
        best, best_s, s0 = self.pose, -np.inf, None
        for t in np.linspace(-SEARCH_TH, SEARCH_TH, 9):
            p = self.pose + np.array([0.0, 0.0, t])
            c = self.cell(self.to_world(hits, p)) - [i0, j0]            # N x 2, window cells
            cc = c[None] + off[:, None]                                   # K x N x 2, all shifts
            ok = (cc >= 0).all(2) & (cc < m).all(2)
            v = np.where(ok, field[np.clip(cc[..., 0], 0, m - 1), np.clip(cc[..., 1], 0, m - 1)], 0.0)
            sc = v.sum(1) / len(hits) - prior - PRIOR_W * (t / pth) ** 2
            if abs(t) < 1e-9:
                s0 = float(sc[len(off) // 2])
            b = int(np.argmax(sc))
            if sc[b] > best_s + 1e-9:
                best_s, best = float(sc[b]), p + np.array([off[b, 0] * self.res, off[b, 1] * self.res, 0.0])
        return (best, best_s - s0) if best_s > 0.05 else (self.pose, 0.0)

    def integrate(self, rays, pose):
        """rays: (N, 3) camera-frame X, Z of each column's ray end, and hit (1) / free (0).
        Vectorised: every ray sampled at once, each cell updated at most once per frame
        (a per-ray Python loop cost 19 ms/frame on the Pi 5)."""
        if not len(rays):
            return
        cam = np.array(self.cam(pose))
        ends = self.to_world(rays[:, :2], pose)
        n = int(np.hypot(*(ends - cam).T).max() / self.res) + 2
        ts = np.linspace(0.0, 1.0, n)
        pts = cam + ts[None, :, None] * (ends - cam)[:, None, :]           # R x n x 2
        cells = self.cell(pts.reshape(-1, 2)).reshape(len(rays), n, 2)
        rng = np.hypot(*(ends - cam).T)
        last = (ts[None, :] * rng[:, None]) > (rng[:, None] - self.res)     # samples in the end cell
        hit = rays[:, 2] > 0
        free = ~(last & hit[:, None])
        ok = (cells >= 0).all(2) & (cells < self.n).all(2)
        flat = cells[..., 0] * self.n + cells[..., 1]
        Lf = self.L.reshape(-1)
        Lf[np.unique(flat[free & ok])] += L_FREE
        endc = flat[np.arange(len(rays)), -1][hit & ok[:, -1]]
        Lf[np.unique(endc)] += L_OCC
        np.clip(self.L, L_MIN, L_MAX, out=self.L)

    def predict_vo(self, d):
        """Move by a GroundVO camera motion (dX right, dZ fwd, dth left) since the last frame."""
        x, y, th = self.pose
        c, s_ = math.cos(th), math.sin(th)
        camx, camy = x + C_REAR * c + d[1] * c + d[0] * s_, y + C_REAR * s_ + d[1] * s_ - d[0] * c
        th2 = th + d[2]
        self.pose = np.array([camx - C_REAR * math.cos(th2), camy - C_REAR * math.sin(th2), th2])

    def step(self, rays, steer, throttle, dt, vo=None, match=True, odo=None):
        """One frame: predict (ground VO if it tracked, else the LAST command), match,
        integrate. Returns the pose.
        odo = (ds, dth) WHEEL odometry since the last frame: rear-axle encoder travel (m) and a
        gyro's heading change (rad), or dth None for Ackermann heading from the steer. It
        replaces VO and the match: offline on the 50 p12full episodes, encoder + gyro is 0.08 m
        median ATE vs 0.42 for ground VO (encoder + Ackermann 0.25)."""
        if odo is not None:
            ds, dth = odo
            if dth is None:
                dth = ds * math.tan(steer * STEER_MAX) / WHEELBASE
            x, y, th = self.pose
            self.pose = np.array([x + ds * math.cos(th + dth / 2), y + ds * math.sin(th + dth / 2), th + dth])
            self.frames += 1
            return self.pose.copy()
        if vo is not None:
            self.predict_vo(vo)
        else:
            self.predict(steer, throttle, dt)
        if not match:
            self.frames += 1
            return self.pose.copy()
        hits = rays[(rays[:, 2] > 0) & (np.hypot(rays[:, 0], rays[:, 1]) < R_MATCH), :2]
        self.pose, _ = self.match(hits, tight=vo is not None)
        self.integrate(rays, self.pose)
        self.frames += 1
        return self.pose.copy()


class GroundVO:
    """Ground-plane visual odometry: track floor texture between consecutive frames (LK
    optical flow), lift every track onto the ground plane with the camera's KNOWN height
    and intrinsics (Z = fy h / (v - cy), metric, no scale ambiguity), and fit the 2D rigid
    motion with RANSAC (obstacle points, which are not on the floor, are the outliers).
    Command dead-reckoning cannot see wheel slip or a car wedged on a curb: in HITL it was
    off by 1-4 m where the car reversed or scrubbed (v10stu1slam 01/03)."""

    def __init__(self, min_tracks=25, max_speed=None, max_turn=1.5):
        """max_speed/max_turn: the plausibility gate (m/s, rad/s). None = 1.5 x V_MAX, the SIMULATED car's top
        speed -- right in HITL, wrong on the real Traxxas: on the 2026-10-16 park drive it rejected 23 % of frames,
        exactly the fast ones (real speed p90 2.9 m/s), and the command fallback (also V_MAX) then measured the
        drive 2.3x too short. The real car passes a PHYSICAL bound (5 m/s, 3 rad/s), not one fitted to a reference."""
        self.prev = None
        self.min_tracks = min_tracks
        self.max_speed, self.max_turn = max_speed, max_turn

    def step(self, gray, intr, cam_h, dt=None):
        """-> (dX_right, dZ_fwd, dtheta_left) of the CAMERA since the last frame, or None.
        None also when the fit is not trustworthy: fewer than half the tracks agree (the
        nose against a barrel puts most "floor" tracks on the barrel), or the motion is
        faster than the car can go in dt."""
        import cv2
        prev, self.prev = self.prev, gray
        self.flow = None
        if prev is None or prev.shape != gray.shape:
            return None
        H = gray.shape[0]
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        # floor within 1.5 m, 400 tracks, forward-backward LK check (< 0.7 px round trip):
        # offline on the 50 p8full episodes, median ATE 0.329 -> 0.253 m vs 3 m / 200 / none
        v0 = int(cy + fy * cam_h / 1.5)
        mask = np.zeros_like(prev)
        mask[v0:H - 2] = 255
        p0 = cv2.goodFeaturesToTrack(prev, 400, 0.01, 4, mask=mask)
        if p0 is None or len(p0) < self.min_tracks:
            return None
        p1, ok, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, winSize=(21, 21), maxLevel=3)
        pb, okb, _ = cv2.calcOpticalFlowPyrLK(gray, prev, p1, None, winSize=(21, 21), maxLevel=3)
        ok = (ok[:, 0] == 1) & (okb[:, 0] == 1) & (np.hypot(*(pb - p0)[:, 0].T) < 0.7)
        a, b = p0[ok, 0], p1[ok, 0]
        self.flow = b - a                                   # floor-track flow, px: for still()
        good = (a[:, 1] > cy + 2) & (b[:, 1] > cy + 2)
        a, b = a[good], b[good]
        if len(a) < self.min_tracks:
            return None

        def ground(uv):
            Z = fy * cam_h / (uv[:, 1] - cy)
            return np.stack([(uv[:, 0] - cx) * Z / fx, Z], 1)

        A, B = ground(a), ground(b)                       # same floor point, frame k and k+1
        # inlier threshold follows the ground-projection error of a 0.5 px track error,
        # which grows as Z^2 / (fy h): 2 cm near, ~11 cm at 3 m
        thr = 0.02 + 0.5 * A[:, 1] ** 2 / (fy * cam_h)
        rng = np.random.default_rng(0)
        best, best_in = None, None
        for _ in range(60):                                 # RANSAC on 2-point rigid fits
            i = rng.choice(len(A), 2, replace=False)
            R, t = _rigid(A[i], B[i])
            err = np.hypot(*(A @ R.T + t - B).T)
            inl = err < thr
            if best_in is None or inl.sum() > best_in.sum():
                best, best_in = (R, t), inl
        if best_in.sum() < self.min_tracks * 0.6:
            return None
        R, t = _rigid(A[best_in], B[best_in])
        # B = R A + t maps floor points into the NEW camera frame; the camera moved by the
        # inverse: rotation -angle(R), translation -R^T t (x right, z forward)
        ang = math.atan2(R[1, 0], R[0, 0])                  # angle in the (X, Z) plane
        mv = -R.T @ t
        # R rotates floor points by ang in the (X right, Z fwd) plane; a camera turning LEFT
        # makes the world swing RIGHT, i.e. ang = -left turn (checked in _selftest_vo)
        vmax = 1.5 * V_MAX if self.max_speed is None else self.max_speed
        if dt is not None and (np.hypot(*mv) > vmax * dt + 0.02 or abs(ang) > self.max_turn * dt + 0.02):
            return None                                     # faster than the car can move / turn
        return float(mv[0]), float(mv[1]), float(-ang)


STILL_PX = 2.0
# ...and only when the wheels claim real travel: a slow legit reverse (~1.2 cm/frame) moves the
# floor ~1 px, under STILL_PX, and was zeroed as slip -- reverse odometry read 7 % short, some
# episodes 10 % (l31imu, 50 episodes). Slip under contact is the boosted wheel spinning, ~3.6 cm.
DS_SLIP = 0.02


def still(flow, fx, dth, ds=None):
    """Is the car standing still although its wheels turn (slip: wedged, pushing, spinning)?
    flow = GroundVO.flow (floor tracks, px, since the last frame); dth = the heading change
    (gyro, rad). A camera turning LEFT by dth moves the scene RIGHT by ~fx dth px, so that is
    taken out; what is left is translation. A stalled car's camera still wobbles 0.3-2.4 px
    while the wheels spin (p14imus1 03). Over p13imu + p14imu's dumps (03/04/07/09), still
    pairs (< 4 mm / 0.1 s, n 2288) vs moving (> 12 mm, n 5218): 1.0 px catches 66.6 % / calls
    0.15 % of moving still; 2.0 px 83.8 % / 1.21 %; 3.0 px 91.3 % / 4.64 %. A missed still
    frame adds ~2.5 cm of phantom travel; a missed moving one loses ~1 % of the distance.
    The encoder alone ran 1-5 m off there."""
    if flow is None or len(flow) < 20 or (ds is not None and abs(ds) < DS_SLIP):
        return False
    return float(np.median(np.hypot(flow[:, 0] - fx * dth, flow[:, 1]))) < STILL_PX


def _rigid(A, B):
    """Least-squares 2D rotation + translation with B ~ R A + t (Kabsch)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    return R, cb - R @ ca


def map_points(slam, radius=5.0, min_logodds=1.5):
    """Confidently occupied map cells (net >= 2 hits) within `radius` of the camera, in the
    camera frame (X right, Z fwd)."""
    x, y, th = slam.pose
    cx, cy = slam.cam()
    r = int(radius / slam.res)
    ci, cj = slam.cell(np.array([[cx, cy]]))[0]
    i0, j0 = max(ci - r, 0), max(cj - r, 0)
    occ = np.argwhere(slam.L[i0:ci + r, j0:cj + r] > min_logodds) + [i0, j0]
    if not len(occ):
        return np.zeros((0, 2))
    w = (occ - slam.o + 0.5) * slam.res - [cx, cy]
    c, s_ = math.cos(th), math.sin(th)
    return np.stack([w[:, 0] * s_ - w[:, 1] * c, w[:, 0] * c + w[:, 1] * s_], 1)   # X right, Z fwd


def rays_from_scan(scan, z_max):
    """avoid.plan's out['scan'] (per-column X, Z, hit) -> integrate() rays."""
    return np.asarray(scan, np.float64)


def _render_floor(tex, pose, intr, cam_h, shape):
    """Synthetic camera image of a textured flat floor. pose = camera (x fwd, y left, yaw)."""
    import cv2
    H, W = shape
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    below = v > cy + 1
    Z = np.where(below, fy * cam_h / np.maximum(v - cy, 1e-3), 0)
    X = (u - cx) * Z / fx
    x, y, th = pose
    wx = x + Z * math.cos(th) + X * math.sin(th)          # X right = (sin th, -cos th)
    wy = y + Z * math.sin(th) - X * math.cos(th)
    res = 0.01
    mx = (wx / res + tex.shape[1] / 2).astype(np.float32)
    my = (wy / res + tex.shape[0] / 2).astype(np.float32)
    img = cv2.remap(tex, mx, my, cv2.INTER_LINEAR)
    img[~below] = 128
    return img


def _selftest_vo():
    import cv2
    rng = np.random.default_rng(1)
    tex = cv2.GaussianBlur((rng.random((2000, 2000)) * 255).astype(np.uint8), (0, 0), 2.0)
    intr = {"fx": 291.1, "fy": 291.1, "cx": 196.0, "cy": 112.0}
    vo = GroundVO()
    a = _render_floor(tex, (0.0, 0.0, 0.0), intr, 0.14, (224, 392))
    th = math.radians(3.0)                                  # 0.10 m forward, then 3 deg LEFT
    b = _render_floor(tex, (0.10, 0.0, th), intr, 0.14, (224, 392))
    vo.step(a, intr, 0.14)
    dx, dz, dth = vo.step(b, intr, 0.14)
    assert abs(dz - 0.10) < 0.01 and abs(dx) < 0.01 and abs(dth - th) < math.radians(0.4), \
        f"VO got dX {dx:+.3f} dZ {dz:+.3f} dth {math.degrees(dth):+.2f} deg"
    print(f"ground VO selftest: OK (dZ {dz:.3f} m of 0.100, dX {dx:+.3f}, dth {math.degrees(dth):+.2f} of +3.00 deg)")


def _selftest_still():
    rng = np.random.default_rng(0)
    rot = np.c_[np.full(100, 291.1 * 0.01), np.zeros(100)] + rng.normal(0, 0.2, (100, 2))   # turning 0.01 rad
    assert still(rot, 291.1, 0.01) and not still(rot + [0.0, 3.0], 291.1, 0.01) and not still(rot, 291.1, 0.0)
    assert not still(None, 291.1, 0.0) and not still(rot[:5], 291.1, 0.01)
    assert still(rot, 291.1, 0.01, ds=0.036) and not still(rot, 291.1, 0.01, ds=0.012), "slow travel must be trusted"
    print("still selftest: OK (pure rotation = still; +3 px of translation = moving; slow wheels trusted)")


def _selftest_odo():
    r = WHEELBASE / math.tan(0.5 * STEER_MAX)                       # Ackermann radius at steer 0.5
    for gyro in (True, False):
        s = Slam2D()
        for _ in range(100):                                        # a quarter circle, 100 frames
            ds = (math.pi / 2) * r / 100
            s.step(None, 0.5, 0.5, 0.05, odo=(ds, ds / r if gyro else None))
        assert np.allclose(s.pose, [r, r, math.pi / 2], atol=1e-3), s.pose
    print(f"wheel odometry selftest: OK (quarter circle r {r:.3f} m from encoder + gyro and encoder + Ackermann)")


def _selftest():
    """A car drives straight past a wall with a 6 % odometry speed error (slip): scan
    matching must hold the along-track position that dead-reckoning loses."""
    rng = np.random.default_rng(0)
    wall = np.stack([np.linspace(0.5, 6.0, 120), np.full(120, 1.2)], 1)       # world, left
    posts = np.array([[2.0, -0.9], [3.5, -1.0], [5.0, -0.8]])                  # world, right
    world = np.vstack([wall, posts + rng.normal(0, 0.01, posts.shape)])
    slam, true, dr = Slam2D(), np.zeros(3), np.zeros(3)
    fov = math.radians(34)
    for k in range(60):
        # true motion 0.036 m/step; the command claims 6 % less (wheel slip the other way)
        true = true + np.array([0.036, 0.0, 0.0])
        cx, cy = true[0] + C_REAR, true[1]
        rel = world - [cx, cy]
        Z, X = rel[:, 0], -rel[:, 1]
        rays = []
        for b in np.linspace(-fov, fov, 64):          # one ray per image column, first hit wins
            ang = np.arctan2(X, Z)
            on = (np.abs(ang - b) < 0.02) & (Z > 0.2)
            r = np.hypot(X[on], Z[on]).min() if on.any() else 9.0
            if r < 4.0:
                r += rng.normal(0, 0.02)
                rays.append([r * math.sin(b), r * math.cos(b), 1.0])
            else:
                rays.append([4.0 * math.tan(b), 4.0, 0.0])
        rays = np.array(rays)
        slam.step(rays, 0.0, 0.036 / 0.05 / V_MAX * 0.94, 0.05)
        dr = dr + np.array([0.036 * 0.94, 0, 0])
    mp = map_points(slam)
    back = slam.to_world(mp, slam.pose)
    assert len(mp) and np.abs(back[:, 1] - 1.2).min() < 0.05, "map_points must invert to_world"
    e_slam, e_dr = abs(slam.pose[0] - true[0]), abs(dr[0] - true[0])
    assert e_slam < 0.5 * e_dr, f"SLAM {e_slam:.3f} m vs dead-reckoning {e_dr:.3f} m"
    print(f"slam2d selftest: OK (along-track error SLAM {e_slam:.3f} m vs dead-reckoning {e_dr:.3f} m "
          f"after {true[0]:.2f} m)")


if __name__ == "__main__":
    _selftest()
    _selftest_vo()
    _selftest_odo()
    _selftest_still()
