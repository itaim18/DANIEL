"""The D435's own metric stereo as a CHECK on the network's near hits (user, 2026-11-19: "sometimes you think stuff is
close to you when it is not"). The network stays the sensor; the stereo only vetoes a near hit it clearly contradicts.

Measured in the lab that evening (experiments/2026-11-19_lab_missions/range_vs_stereo.py, 2138 frames): the network's
nearest corridor hit was 0.3-0.6 m on 957 frames, the stereo's on 697; on 106 of the network's (11 %) the stereo saw
nothing within 1.2x + 0.1 m. Those false near hits stop the car and trigger near-hit backups and dead ends.

    bins = stereo_bins(Z, calib, normal, height)       # per frame: nearest obstacle + how far the stereo SEES, per degree
    pts, dbg, n = veto(pts, dbg, *bins)                  # inside avoid.plan, on the column scan (ctx["veto"])
    python3 stereo_veto.py                               # self-test

A network hit at range r (VETO_MIN <= r <= VETO_MAX, bearing b) is vetoed when the stereo has floor or obstacle points
at bearing b out to at least VETO_K r + VETO_ADD (it SEES that far) and its nearest obstacle there is beyond that:
the hit moves to the stereo's obstacle, or becomes free floor. Under VETO_MIN the D435 measures nothing (~0.28 m), so
the network's word stands; where the stereo has no data (glare, holes, night IR saturation) nothing is vetoed.
"""
import collections
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

VETO_MIN, VETO_MAX, VETO_K, VETO_ADD = 0.3, 1.0, 1.2, 0.1
OBST_LO, OBST_HI, FLOOR_TOL = 0.06, 0.5, 0.04    # m above the floor plane: avoid's NAV_H .. above the car's roof; floor
BIN0, NBIN = -45, 91                             # 1-degree bearing bins, -45..+45 (the colour camera sees +-34.7)
Z_MAX = 4.0                                      # avoid.Z_MAX: a no-hit ray's range


def calib(path=Path(__file__).with_name("d435_calib.json")):
    c = json.loads(Path(path).read_text())
    d = c["intrinsics"]["depth"]
    e = c["extrinsics"]["depth_to_color"]
    return {"fx": d["fx"], "fy": d["fy"], "ppx": d["ppx"], "ppy": d["ppy"],
            "R": np.array(e["rotation_colmajor"], float).reshape(3, 3, order="F"), "t": np.array(e["translation_m"], float)}


def stereo_bins(Z, cal, normal, height, sub=4):
    """Z: (480, 848) metres (0 / NaN = no match). normal/height: the floor fit (cam_pose, depth-camera frame, up-normal).
    -> (near, seen): per bearing degree in the COLOUR camera frame, the nearest obstacle range (inf = none) and the
    farthest range at which the stereo has floor or obstacle points (0 = no data)."""
    z = Z[::sub, ::sub]
    v, u = np.mgrid[0:Z.shape[0]:sub, 0:Z.shape[1]:sub]
    ok = np.isfinite(z) & (z > 0.25) & (z < Z_MAX)
    zz = z[ok]
    P = np.stack([(u[ok] - cal["ppx"]) * zz / cal["fx"], (v[ok] - cal["ppy"]) * zz / cal["fy"], zz], 1)
    hgt = P @ np.asarray(normal, float) + height                   # height above the floor (up-normal, y down)
    Pc = P @ cal["R"].T + cal["t"]                                  # -> colour camera frame (the network's)
    r = np.hypot(Pc[:, 0], Pc[:, 2])
    k = np.round(np.degrees(np.arctan2(Pc[:, 0], Pc[:, 2]))).astype(int) - BIN0
    inb = (k >= 0) & (k < NBIN)
    obst = inb & (hgt >= OBST_LO) & (hgt <= OBST_HI)
    seen_m = inb & (obst | (np.abs(hgt) < FLOOR_TOL))
    near, seen = np.full(NBIN, np.inf), np.zeros(NBIN)
    np.minimum.at(near, k[obst], r[obst])
    np.maximum.at(seen, k[seen_m], r[seen_m])
    return near, seen


def _bin(x, z):
    return np.clip(np.round(np.degrees(np.arctan2(x, z))).astype(int) - BIN0, 0, NBIN - 1)


def veto(pts, dbg, near, seen):
    """avoid's column scan (pts: obstacle points (n, 2) X right / Z fwd; dbg[0]: rays (cols, 4) X, Z, hit, top_h)
    -> the same with the contradicted near hits moved to the stereo's obstacle or freed; and how many rays moved."""
    rays = np.array(dbg[0], float, copy=True)
    r = np.hypot(rays[:, 0], rays[:, 1])
    k = _bin(rays[:, 0], rays[:, 1])
    lim = VETO_K * r + VETO_ADD
    v = (rays[:, 2] > 0) & (r >= VETO_MIN) & (r <= VETO_MAX) & (seen[k] >= lim) & (near[k] > lim)
    if not v.any():
        return pts, dbg, 0
    b = np.arctan2(rays[v, 0], rays[v, 1])
    rn = near[k[v]]
    far = ~(np.isfinite(rn) & (rn < Z_MAX))
    rn = np.where(far, 0.0, rn)
    rays[v, 0] = np.where(far, Z_MAX * np.tan(b), rn * np.sin(b))   # a free ray, as avoid writes one: Z = Z_MAX
    rays[v, 1] = np.where(far, Z_MAX, rn * np.cos(b))
    rays[v, 2] = (~far).astype(float)
    if len(pts):
        pr = np.hypot(pts[:, 0], pts[:, 1])
        pk = _bin(pts[:, 0], pts[:, 1])
        plim = VETO_K * pr + VETO_ADD
        pts = pts[~((pr >= VETO_MIN) & (pr <= VETO_MAX) & (seen[pk] >= plim) & (near[pk] > plim))]
    moved = rays[v][~far, :2]
    if len(moved):
        pts = np.vstack([np.asarray(pts, float).reshape(-1, 2), moved])
    return pts, (rays,) + tuple(dbg[1:]), int(v.sum())


class StereoReader:
    """The newest D435 depth frames, metres. Live through v4l2-ctl (as sensor_rec.py; no librealsense on the Pi) --
    or, while REC ALL owns the depth node, the frames REC ALL itself writes (handover / take_back)."""

    def __init__(self, dev="/dev/video0", fps=15):
        self.dev, self.fps, self.proc, self.rec_dir = dev, fps, None, None
        self.frames, self.t = collections.deque(maxlen=15), 0.0
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def latest(self, max_age=0.3):
        with self.lock:
            return self.frames[-1] if self.frames and time.monotonic() - self.t < max_age else None

    def stack(self):
        """The last frames (for the floor fit): (n, 480, 848) metres, NaN = no match."""
        with self.lock:
            if len(self.frames) < 3:
                raise ValueError("no stereo depth yet")
            return np.stack(list(self.frames))

    def handover(self, rec_dir):
        """REC ALL is about to open the depth node: release it, then follow REC ALL's own frames."""
        self.rec_dir = Path(rec_dir)
        p = self.proc
        if p is not None:
            p.terminate()
        for _ in range(100):
            if self.proc is None:
                return
            time.sleep(0.05)

    def take_back(self):
        self.rec_dir = None

    def _push(self, Z):
        Z = Z.astype(np.float32) / 1000.0
        Z[Z == 0] = np.nan
        with self.lock:
            self.frames.append(Z)
            self.t = time.monotonic()

    def _run(self):
        while True:
            if self.rec_dir is not None:
                self._tail(self.rec_dir)
            else:
                try:
                    self._live()
                except Exception:          # unplugged / busy: never die, try again
                    pass
                time.sleep(0.5)

    def _live(self):
        import sensor_rec as SR
        subprocess.run(["v4l2-ctl", "-d", self.dev, f"--set-fmt-video=width={SR.W},height={SR.H},pixelformat=Z16 ",
                        f"--set-parm={self.fps}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        self.proc = subprocess.Popen(["v4l2-ctl", "-d", self.dev, "--stream-mmap", "--stream-to=-"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        try:
            for buf in SR.frames(self.proc.stdout):
                self._push(np.frombuffer(buf, np.uint16).reshape(SR.H, SR.W))
                if self.rec_dir is not None:
                    break
        finally:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

    def _tail(self, d, max_age=0.3):
        """REC ALL's newest depth frame from RAM (sensor_rec.LIVE_DEPTH), pushed once, only while fresh. Its PNGs
        lag on the SD card past any fixed k-N: reading those left the veto off for whole missions (2026-11-23)."""
        import sensor_rec as SR
        try:
            st = os.stat(SR.LIVE_DEPTH)
            if time.time() - st.st_mtime < max_age and st.st_mtime_ns != getattr(self, "_mt", None):
                self._mt = st.st_mtime_ns
                self._push(np.load(SR.LIVE_DEPTH))
        except (OSError, ValueError):
            pass
        time.sleep(0.5 / self.fps)


def _selftest():
    """A level camera 0.17 m over a flat floor, a box face 1.5 m ahead on the left. A false network hit at 0.5 m
    straight ahead is freed; a hit at 0.5 m on the box's bearing moves out to the box; under VETO_MIN, past VETO_MAX
    and where the stereo has no data nothing is vetoed."""
    cal = {"fx": 424.8, "fy": 424.8, "ppx": 424.0, "ppy": 240.0, "R": np.eye(3), "t": np.zeros(3)}
    h, H, W = 0.17, 480, 848
    v, u = np.mgrid[0:H, 0:W].astype(float)
    with np.errstate(divide="ignore"):
        Z = np.where(v > cal["ppy"] + 1, cal["fy"] * h / (v - cal["ppy"]), np.nan)          # floor
    X = (u - cal["ppx"]) * Z / cal["fx"]
    box = (X > -0.6) & (X < -0.2) & (Z > 1.5)                                               # a face at 1.5 m, left
    Zb = np.where((u - cal["ppx"]) / cal["fx"] * 1.5 > -0.6, 1.5, np.nan)
    hb = h - (v - cal["ppy"]) * 1.5 / cal["fy"]
    face = ((u - cal["ppx"]) * 1.5 / cal["fx"] > -0.6) & ((u - cal["ppx"]) * 1.5 / cal["fx"] < -0.2) & (hb > 0) & (hb < 0.4)
    Z = np.where(face, 1.5, np.where(box, np.nan, Z))
    Z[:, :40] = np.nan                                                                      # a hole at the image edge
    near, seen = stereo_bins(Z, cal, normal=(0.0, -1.0, 0.0), height=h)
    bb = int(round(np.degrees(np.arctan2(-0.4, 1.5)))) - BIN0
    assert abs(near[bb] - np.hypot(0.4, 1.5)) < 0.05, near[bb]
    assert not np.isfinite(near[-BIN0]) and seen[-BIN0] > 2.0, (near[-BIN0], seen[-BIN0])
    b_box = np.arctan2(-0.4, 1.5)
    rays = np.array([[0.0, 0.5, 1, 0.2],                                  # false: nothing ahead
                     [0.5 * np.sin(b_box), 0.5 * np.cos(b_box), 1, 0.2],  # the box is there, but at 1.55 m
                     [1.4 * np.sin(b_box), 1.4 * np.cos(b_box), 1, 0.2],  # beyond VETO_MAX: untouched
                     [0.0, 0.2, 1, 0.2],                                  # under VETO_MIN: the stereo cannot see it
                     [0.0, Z_MAX, 0, 0.0]])
    pts = rays[rays[:, 2] > 0, :2]
    p2, d2, n = veto(pts, (rays, None), near, seen)
    r2 = d2[0]
    assert n == 2 and r2[0, 2] == 0 and r2[0, 1] == Z_MAX, r2[0]
    assert r2[1, 2] == 1 and abs(np.hypot(*r2[1, :2]) - np.hypot(0.4, 1.5)) < 0.05, r2[1]
    assert (r2[2:] == rays[2:]).all()
    assert not ((np.abs(p2[:, 0]) < 1e-9) & (np.abs(p2[:, 1] - 0.5) < 1e-9)).any() and len(p2) == 3, p2
    blind = (near.copy(), np.zeros(NBIN))
    assert veto(pts, (rays, None), *blind)[2] == 0, "vetoed with no stereo data"
    import tempfile                                   # REC ALL hand-over: fresh frame pushed once, stale one never
    import sensor_rec as SR
    r = StereoReader.__new__(StereoReader)            # no thread, no camera
    r.fps, r.frames, r.t, r.lock = 15, collections.deque(maxlen=15), 0.0, threading.Lock()
    with tempfile.TemporaryDirectory() as td:
        SR.LIVE_DEPTH = f"{td}/d.npy"
        np.save(SR.LIVE_DEPTH, np.full((4, 4), 1500, np.uint16))
        r._tail(None)
        r._tail(None)
        assert len(r.frames) == 1 and abs(np.nanmax(r.latest()) - 1.5) < 1e-6, "fresh frame not pushed exactly once"
        os.utime(SR.LIVE_DEPTH, (time.time() - 1, time.time() - 1))
        r.frames.clear()
        r._tail(None)
        assert not r.frames, "a stale REC ALL frame was pushed"
    print(f"stereo_veto selftest: OK (a false hit at 0.5 m freed, a hit short of a 1.55 m box moved to it, "
          f"none under {VETO_MIN} m / past {VETO_MAX} m / without stereo data; REC ALL frames: fresh once, stale never)")


if __name__ == "__main__":
    _selftest()
