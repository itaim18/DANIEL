"""Laptop side of the HITL link to board/pi5/hitl_pi.py (lockstep request/response).

    PiBrain(host).step(rgb_uint8, hdr) -> (reply_dict, viz_bgr_or_None)
    python3 hitl_client.py --host 10.42.0.139 --video clip.mp4      # round-trip benchmark
    python3 hitl_client.py --replay runs/<run>/dump                  # f*.jpg -> f*_hailo.npy

Kept free of Isaac imports so it can be benchmarked on its own.
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import time

import cv2
import numpy as np


def _recv_exact(s, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Pi closed the connection")
        buf += chunk
    return bytes(buf)


class PiBrain:
    def __init__(self, host="10.42.0.139", port=7000, jpeg_q=90):
        # 60 s: a Pi re-plan is bounded at ~3.5 s, but a slow reply must never kill a lockstep suite
        self.s = socket.create_connection((host, port), timeout=60.0)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.q = jpeg_q

    def _send(self, payload: bytes):
        self.s.sendall(struct.pack(">I", len(payload)) + payload)

    def _recv(self):
        return _recv_exact(self.s, struct.unpack(">I", _recv_exact(self.s, 4))[0])

    def step(self, rgb: np.ndarray, hdr: dict):
        """rgb: HxWx3 uint8 RGB. Returns (reply, viz BGR or None, round-trip ms)."""
        t0 = time.perf_counter()
        jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, self.q])[1].tobytes()
        self._send(json.dumps(hdr).encode())
        self._send(jpg)
        reply = json.loads(self._recv())
        v = self._recv()
        if "disp_shape" in reply:                   # hdr["disp"]: raw Hailo disparity
            viz = np.frombuffer(v, np.float16).reshape(reply["disp_shape"]).astype(np.float32)
        else:
            viz = cv2.imdecode(np.frombuffer(v, np.uint8), cv2.IMREAD_COLOR) if v else None
        reply["rtt_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
        reply["jpeg_kb"] = round(len(jpg) / 1024, 1)
        return reply, viz

    def close(self):
        self.s.close()


def sim_intrinsics(w=504, h=280, hfov_deg=69.4, cam_h=0.20):
    """The yard sim camera (arena_env_cfg CameraCfg: 504x280, focal 15.13 mm)."""
    fx = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
    return {"fx": float(fx), "fy": float(fx), "cx": w / 2, "cy": h / 2, "cam_h": cam_h}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="10.42.0.139")
    ap.add_argument("--video")
    ap.add_argument("--replay", help="dump dir from hitl_sim --dump-every: save the REAL Hailo "
                                     "disparity of every POV jpg next to it, for offline avoid dev")
    ap.add_argument("--suffix", default="hailo", help="replay output: f*_<suffix>.npy")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", help="write the Pi's debug views as an mp4")
    a = ap.parse_args()
    brain = PiBrain(a.host)
    if a.replay:
        from pathlib import Path
        n = 0
        for meta in sorted(Path(a.replay).rglob("meta.json")):
            hdr = dict(json.loads(meta.read_text()), viz=False, disp=True)
            for f in sorted(meta.parent.glob("f[0-9][0-9][0-9][0-9].jpg")):
                rgb = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
                np.save(f.with_name(f"{f.stem}_{a.suffix}.npy"), brain.step(rgb, hdr)[1].astype(np.float16))
                n += 1
        print(f"replayed {n} frames through the Hailo")
        raise SystemExit(0)
    cap = cv2.VideoCapture(a.video)
    hdr = dict(sim_intrinsics(), goal_bearing=0.0, goal_dist=5.0, viz=True, reset=True)
    rtt, hailo, plan, w = [], [], [], None
    for i in range(a.n):
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(cv2.resize(bgr, (504, 280)), cv2.COLOR_BGR2RGB)
        r, viz, = brain.step(rgb, hdr)[:2]
        hdr["reset"] = False
        rtt.append(r["rtt_ms"]); hailo.append(r["ms"]["hailo"]); plan.append(r["ms"]["plan"])
        if a.out and viz is not None:
            if w is None:
                w = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 15, viz.shape[1::-1])
            w.write(viz)
    if w:
        w.release()
    p = lambda v, q: round(float(np.percentile(v, q)), 1)  # noqa: E731
    print(f"frames {len(rtt)}  rtt p50 {p(rtt, 50)} p99 {p(rtt, 99)} ms | hailo p50 {p(hailo, 50)} "
          f"| plan p50 {p(plan, 50)} p99 {p(plan, 99)} ms | last {r['mode']} s{r['steer']:+.2f} "
          f"t{r['throttle']:+.2f} free {r['free']:.2f}")
