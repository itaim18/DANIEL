"""One run's record -- run.json (config), commands.jsonl (one line per brain step), debug.mp4
(camera | planner view), summary.json (outcome) -- and the laptop side of cockpit recording.

avoid_drive.py --out writes it on the Pi. Cockpit missions are recorded on the LAPTOP: the Pi
streams each step over GET /record and this script, left running on the laptop, writes the
files into the repo:

    python3 onboard/run_record.py                       # -> experiments/<date>_cockpit/run_<hhmmss>/

Stream format (both ends live here): per message, '>II' (json length, jpeg length), the JSON
({"type": "start"|"step"|"end"|"hb", ...}), then the JPEG (may be empty).
"""
import argparse
import json
import shutil
import struct
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np


def debug_frame(rgb, disp, o, ctx):
    """-> (camera | avoid.render as one BGR frame, the planner view alone)."""
    import avoid                                    # Pi-side only: the laptop never renders
    viz = avoid.render(disp, o, ctx)
    cam = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (viz.shape[1], viz.shape[0]))
    return np.hstack([cam, viz]), viz


class Recorder:
    def __init__(self, out, meta, fps=10):
        self.out, self.fps = Path(out), fps
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "run.json").write_text(json.dumps(meta, indent=1))
        self.log, self.vw, self.n = open(self.out / "commands.jsonl", "w"), None, 0

    def write(self, rec, frame):
        """A rec with "t" (s since start) is placed in REAL time: the previous frame is held
        until this one is due, so the video plays at the speed the car drove; a gap (deadman
        up) holds at most 1 s. Without "t" (avoid_drive): one video frame per step."""
        self.log.write(json.dumps(rec) + "\n")
        if frame is None:
            return
        if self.vw is None:
            self.vw = cv2.VideoWriter(str(self.out / "debug.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                      self.fps, frame.shape[1::-1])
            self.t0 = rec.get("t")
        elif self.t0 is not None:
            gap = round((rec["t"] - self.t0) * self.fps) - self.n   # slots before this frame is due
            fill = min(max(gap, 0), self.fps)
            self.t0 += max(gap - fill, 0) / self.fps                # a capped gap is cut, not owed
            for _ in range(fill):
                self.vw.write(self.last)
            self.n += fill
        self.vw.write(frame)
        self.n, self.last = self.n + 1, frame

    def close(self, summary=None):
        self.log.close()
        if self.vw:
            self.vw.release()
        if summary is not None:
            (self.out / "summary.json").write_text(json.dumps(summary, indent=1))


def pack(msg, jpg=b""):
    j = json.dumps(msg).encode()
    return struct.pack(">II", len(j), len(jpg)) + j + jpg


def _read(r, n):
    b = b""
    while len(b) < n:
        c = r.read(n - len(b))
        if not c:
            raise ConnectionError("stream closed")
        b += c
    return b


def consume(r, root, min_free_gb, runs):
    """Write the runs on one /record stream until it raises (link lost). Steps route by run id."""
    while True:
        jl, il = struct.unpack(">II", _read(r, 8))
        msg = json.loads(_read(r, jl))
        jpg = _read(r, il) if il else b""
        run = msg.get("run")
        if msg["type"] == "start":
            free_gb = shutil.disk_usage(root).free / 1e9
            if free_gb < min_free_gb:
                print(f"[rec] {run} NOT recorded: {free_gb:.1f} GB free < {min_free_gb}", flush=True)
                continue
            day, hms = run.split("_", 1)
            runs[run] = Recorder(Path(root) / f"{day}_cockpit" / f"run_{hms}", msg["meta"],
                                 fps=msg["meta"].get("video_fps", 10))
            print(f"[rec] {run} started -> {runs[run].out}", flush=True)
        elif msg["type"] == "step" and run in runs:
            runs[run].write(msg["rec"], cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR) if jpg else None)
        elif msg["type"] == "end" and run in runs:
            runs.pop(run).close(msg["summary"])
            print(f"[rec] {run} ended: {msg['summary']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pi", default="10.42.0.139", help="the Pi (cable: 10.42.0.139; WiFi: its hotspot or Tailscale address)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "recordings"))
    ap.add_argument("--min-free-gb", type=float, default=5.0,   # was 10 (user, 2026-11-23)
                    help="skip new runs below this much free disk (the Pi's SD card must never fill)")
    a = ap.parse_args()
    Path(a.root).mkdir(parents=True, exist_ok=True)   # disk_usage needs it; the Pi's copy may lack it
    runs = {}                                         # a mission and a manual session can overlap
    while True:
        try:
            r = urllib.request.urlopen(f"http://{a.pi}:{a.port}/record", timeout=10)  # the Pi beats every 2 s
            print(f"[rec] connected to {a.pi}; recording cockpit runs into {a.root}", flush=True)
            consume(r, a.root, a.min_free_gb, runs)
        except (OSError, ValueError) as e:            # Pi rebooting, cockpit restarting, WiFi gone
            for run, rec in runs.items():
                rec.close({"status": f"recorder: link lost ({type(e).__name__})"})
                print(f"[rec] link lost mid-run; partial run kept at {rec.out}", flush=True)
            runs.clear()
            time.sleep(2)


def _selftest():
    import tempfile
    d = tempfile.mkdtemp()
    r = Recorder(d, {}, fps=20)
    img = np.zeros((32, 32, 3), np.uint8)
    for t in [0.0, 0.05, 0.10, 0.30, 5.30, 5.35]:        # 20 Hz, a 0.2 s lag, a 5 s pause
        r.write({"t": t}, img)
    r.close()
    n = int(cv2.VideoCapture(f"{d}/debug.mp4").get(cv2.CAP_PROP_FRAME_COUNT))
    # 3 on time; .30 is 3 slots late (hold 3); the 5 s pause holds 20 (1 s cap); then 2 on time
    assert r.n == n == 3 + 3 + 1 + 20 + 1 + 1, (r.n, n)
    r = Recorder(d, {}, fps=10)
    for _ in range(3):
        r.write({"k": 0}, img)                             # no "t": one frame per step
    r.close()
    assert r.n == 3
    # a manual session (ARM ESC) opened inside a paused mission: each step lands in its own run
    import io
    jpg = cv2.imencode(".jpg", img)[1].tobytes()
    A, B = "2026-10-16_120000", "2026-10-16_120005_manual"
    msgs = [({"type": "start", "run": A, "meta": {}}, b""), ({"type": "step", "run": A, "rec": {"t": 0.0}}, jpg),
            ({"type": "start", "run": B, "meta": {}}, b""), ({"type": "step", "run": B, "rec": {"t": 0.0}}, jpg),
            ({"type": "step", "run": A, "rec": {"t": 0.1}}, jpg), ({"type": "end", "run": B, "summary": {}}, b""),
            ({"type": "hb"}, b""), ({"type": "step", "run": A, "rec": {"t": 0.2}}, jpg)]
    runs = {}
    try:
        consume(io.BytesIO(b"".join(pack(m, j) for m, j in msgs)), d, 0, runs)
    except ConnectionError:                                # end of stream = link lost
        pass
    lines = {k: len(open(f"{d}/2026-10-16_cockpit/run_{k}/commands.jsonl").readlines()) for k in ("120005_manual",)}
    assert lines == {"120005_manual": 1} and list(runs) == [A] and runs[A].n >= 3, (lines, list(runs))
    try:
        consume(io.BytesIO(pack(*msgs[0])), d, 1e9, {})   # below the free-disk floor: nothing written
    except ConnectionError:
        pass
    print("run_record selftest: OK (real-time placement, gaps capped at 1 s, no-t = 1 frame/step, "
          "overlapping runs routed by id, free-disk floor)")


if __name__ == "__main__":
    import sys
    _selftest() if sys.argv[1:] == ["--selftest"] else main()
