"""Camera height / pitch / roll over the floor, measured from the D435's OWN metric depth
(the stereo Z16 node, /dev/video0) -- the numbers ground-plane VO and avoid's ranging need.

    python3 cam_pose.py              # on the Pi, car on its wheels on flat floor
    python3 cam_pose.py --selftest

RANSAC plane on the lower 45 % of the depth image, then a least-squares refit on its inliers.
No librealsense here to read the factory intrinsics, so FX is the D435's typical depth focal
length at 640x480: a +/-4 % error moves the height +/-4 % and the pitch ~0.15 deg. The colour
camera shares the mount, so its pitch is the same.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

FX, W, H = 385.0, 640, 480


def capture(dev="/dev/video0", n=15, raw="/tmp/cam_pose_depth.raw"):
    """-> (H, W) metres, the per-pixel median of n frames (0 = no stereo match -> NaN)."""
    subprocess.run(["v4l2-ctl", "-d", dev, f"--set-fmt-video=width={W},height={H},pixelformat=Z16 ",
                    "--stream-mmap", f"--stream-count={n}", f"--stream-to={raw}"],
                   check=True, capture_output=True, timeout=20)
    d = np.fromfile(raw, np.uint16).reshape(-1, H, W)[3:].astype(np.float32)   # skip auto-exposure settle
    d[d == 0] = np.nan
    return np.nanmedian(d, 0) / 1000.0


def fit_floor(Z, fx=FX, seed=0, cx=None, cy=None):
    h, w = Z.shape
    cx, cy = (w / 2 if cx is None else cx), (h / 2 if cy is None else cy)   # factory principal point when known
    v, u = np.mgrid[0:h, 0:w]
    m = np.isfinite(Z) & (Z > 0.15) & (Z < 4.0) & (v > h * 0.55)
    z = Z[m]
    P = np.stack([(u[m] - cx) * z / fx, (v[m] - cy) * z / fx, z], 1)   # x right, y down, z fwd
    if len(P) < 500:
        raise ValueError("too little valid depth in the lower image -- is the floor in view?")
    rng, best = np.random.default_rng(seed), None
    for _ in range(400):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        nrm = np.cross(b - a, c - a)
        if np.linalg.norm(nrm) < 1e-9:
            continue
        nrm /= np.linalg.norm(nrm)
        inl = np.abs((P - a) @ nrm) < 0.015
        if best is None or inl.sum() > best.sum():
            best = inl
    Q = P[best]
    c0 = Q.mean(0)
    nrm = np.linalg.svd(Q - c0, full_matrices=False)[2][-1]
    if nrm[1] > 0:
        nrm = -nrm                                       # the floor normal points UP (camera y is down)
    return {"height_m": round(float(abs(nrm @ c0)), 3),
            "pitch_down_deg": round(float(np.degrees(np.arcsin(-nrm[2]))), 2),
            "roll_deg": round(float(np.degrees(np.arctan2(nrm[0], -nrm[1]))), 2),
            "floor_frac": round(float(best.mean()), 2),
            "rms_mm": round(float(np.sqrt(np.mean(((Q - c0) @ nrm) ** 2)) * 1e3), 1),
            "normal": [float(x) for x in nrm]}                # floor up-normal in this camera's frame


def capture_rec(rec_dir, n=15):
    """While REC ALL holds the depth node (v4l2 capture() then fails: 2026-11-19, every mission started during a
    recording drove on the typed fallback pose): the newest n depth PNGs it wrote -> (480, 848) metres."""
    import cv2
    fs = sorted((Path(rec_dir) / "depth").glob("*.png"))[-n - 2:-2]     # the very newest may be half-written
    if len(fs) < 3:
        raise ValueError("REC ALL has written no depth yet")
    d = np.stack([cv2.imread(str(f), cv2.IMREAD_UNCHANGED) for f in fs]).astype(np.float32)
    d[d == 0] = np.nan
    return np.nanmedian(d, 0) / 1000.0


def measure(rec_dir=None, frames=None):
    """The camera pose over the floor. From REC ALL's own depth frames when rec_dir (a recording) is given, from
    frames ((n, 480, 848) metres, NaN = no match: the cockpit's stereo reader owns the node) when given."""
    if rec_dir is None and frames is None:
        return fit_floor(capture())
    c = json.loads(Path(__file__).with_name("d435_calib.json").read_text())["intrinsics"]["depth"]
    Z = np.nanmedian(frames, 0) if frames is not None else capture_rec(rec_dir)
    return fit_floor(Z, fx=c["fx"], cx=c["ppx"], cy=c["ppy"])


def _selftest():
    for h_true, pitch in ((0.159, 3.3), (0.289, 0.0), (0.12, 8.0)):
        t = np.radians(pitch)
        nrm = np.array([0.0, -np.cos(t), -np.sin(t)])     # floor's up-normal, camera pitched down by t
        v, u = np.mgrid[0:H, 0:W]
        rays = np.stack([(u - W / 2) / FX, (v - H / 2) / FX, np.ones((H, W))], -1)
        den = rays @ nrm
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.where(den < -1e-6, -h_true / den, np.nan)           # ray meets the floor at depth Z
        Z += np.random.default_rng(1).normal(0, 0.002, Z.shape)        # 2 mm stereo noise
        r = fit_floor(Z)
        assert abs(r["height_m"] - h_true) < 0.004 and abs(r["pitch_down_deg"] - pitch) < 0.15, (h_true, pitch, r)
    print("cam_pose selftest: OK (height within 4 mm, pitch within 0.15 deg on synthetic floors)")


if __name__ == "__main__":
    _selftest() if sys.argv[1:] == ["--selftest"] else print(measure())
