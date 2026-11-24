"""REC ALL's D435 depth + IR capture. cockpit.py spawns it with the session dir and SIGTERMs it on stop:

    ~/daniel-venv/bin/python sensor_rec.py SESSION_DIR      # writes depth/ ir_left/ ir_right/ + CSVs
    python3 sensor_rec.py --selftest

The cockpit holds the colour node (video4) and records RGB + Hailo itself; this process owns the stereo
module: video0 Z16 depth and video2 Y8I (left + right IR interleaved, left = first byte). Read through
v4l2-ctl as cam_pose.capture does (no librealsense on the Pi). Every frame keeps its KERNEL timestamp
(CLOCK_MONOTONIC, start of exposure), the same clock as the cockpit's time.monotonic(), so streams pair
by nearest time. Measured 2026-10-16 on this Pi: depth + IR both 29.98 fps at 848x480 alongside the
cockpit; PNG level 1 = 21 ms / 266 KB depth, 14.5 ms / 225 KB per IR image; SD 73-80 MB/s.
Needs 90-rec-all-writeback.conf on the Pi: default dirty-page limits stall the SD for seconds.
"""
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

W, H, FPS = 848, 480, 30     # Intel's recommended D435 depth mode, ~16:9 like the cockpit's 640x360 RGB
# Each depth frame also goes to RAM for the live stereo veto (stereo_veto.StereoReader._tail): reading REC ALL's
# PNGs starved it -- the SD write pool lags past k-3, so the veto ran on 0 % of frames in 3 of 5 recorded missions
# (2026-11-23). ~0.8 MB, replaced atomically every frame.
LIVE_DEPTH = "/dev/shm/daniel_depth.npy"
MAX_PENDING = 450            # PNG jobs in flight (~5 s, ~350 MB): rides out an SD stall; past it a frame is DROPPED, counted
STREAMS = {"depth": ("/dev/video0", "Z16 "), "ir": ("/dev/video2", "Y8I ")}   # both 2 bytes / pixel
FRAME_BYTES = W * H * 2
TS = re.compile(rb"seq:\s*(\d+).*?ts:\s*([\d.]+)")


def parse_ts(line):
    """A v4l2-ctl --verbose buffer line -> (seq, t_monotonic) or None."""
    m = TS.search(line) if b"dqbuf" in line else None
    return (int(m[1]), float(m[2])) if m else None


def frames(f, n=FRAME_BYTES):
    """Exact n-byte frames from a pipe (a read may return less), until EOF. A leading v4l2-ctl
    "Format Video Capture:" block is skipped: on 2026-10-16 it rode the frame pipe and shifted every
    frame by 365 B (Z16) / 380 B (Y8I) -- experiments/2026-10-19_realworld/recover_stream.py."""
    b = bytearray()
    while len(b) < 4096:
        c = f.read(4096 - len(b))
        if not c:
            break
        b += c
    if b.startswith(b"Format Video Capture:"):
        del b[:b.index(b"\n", b.find(b"\tFlags")) + 1]
    while True:
        while len(b) < n:
            c = f.read(n - len(b))
            if not c:
                return
            b += c
        yield bytes(b[:n])
        b = bytearray(b[n:])


def split_y8i(buf):
    """Y8I -> (left, right) uint8 (H, W): the two imagers interleaved byte by byte."""
    a = np.frombuffer(buf, np.uint8).reshape(H, W, 2)
    return a[..., 0].copy(), a[..., 1].copy()


def capture(name, out, pool, stats, stop):
    import cv2
    dev, fourcc = STREAMS[name]
    # format in its OWN call: v4l2-ctl prints the format block to stdout, which must carry frames only
    subprocess.run(["v4l2-ctl", "-d", dev, f"--set-fmt-video=width={W},height={H},pixelformat={fourcc}",
                    f"--set-parm={FPS}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    p = subprocess.Popen(["v4l2-ctl", "-d", dev, "--stream-mmap", "--stream-to=-", "--verbose"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    ts = queue.Queue()
    threading.Thread(target=lambda: [ts.put(t) for t in map(parse_ts, p.stderr) if t], daemon=True).start()
    dirs = ["depth"] if name == "depth" else ["ir_left", "ir_right"]
    for d in dirs:
        (out / d).mkdir(exist_ok=True)
    st = stats[name] = {"frames": 0, "drops": 0, "seq_gaps": 0, "pending": 0}
    lock = threading.Lock()

    def save(path, img):
        cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        with lock:
            st["pending"] -= 1

    last_seq = None
    with open(out / f"{name}.csv", "w", buffering=1) as csv:
        # t_mono = the v4l2 buffer stamp; t_arrival = host monotonic time the frame left the pipe. On
        # 2026-10-16 the D435's depth/IR stamps ticked a perfect 33.4 ms while the CONTENT ran ~8 % ahead
        # (10 s by 135 s, with stall steps): the stamps are synthesised from the camera clock and hide
        # stalls. Pair with RGB by t_arrival (ms of pipe jitter, no drift), never by t_mono alone.
        csv.write("k,seq,t_mono,t_arrival\n")
        for k, buf in enumerate(frames(p.stdout)):
            t_arr = time.monotonic()
            if stop.is_set():
                break
            try:
                seq, t = ts.get(timeout=2.0)
            except queue.Empty:
                seq, t = -1, float("nan")
            if last_seq is not None and seq > last_seq + 1:
                st["seq_gaps"] += seq - last_seq - 1          # the camera/driver skipped frames
            last_seq = seq
            imgs = ([np.frombuffer(buf, np.uint16).reshape(H, W)] if name == "depth" else split_y8i(buf))
            if name == "depth":                               # before the drop check: the veto needs it even then
                with open(LIVE_DEPTH + ".tmp", "wb") as f:
                    np.save(f, imgs[0])
                os.replace(LIVE_DEPTH + ".tmp", LIVE_DEPTH)
            with lock:
                if st["pending"] + len(imgs) > MAX_PENDING:   # the SD card / CPU fell behind
                    st["drops"] += 1
                    continue
                st["pending"] += len(imgs)
            for d, img in zip(dirs, imgs):
                pool.submit(save, out / d / f"{k:06d}.png", img)
            csv.write(f"{k},{seq},{t:.6f},{t_arr:.6f}\n")
            st["frames"] += 1
    p.terminate()
    p.wait(timeout=5)


def main(out):
    out = Path(out)
    stop, stats, t0 = threading.Event(), {}, time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    parent = os.getppid()
    pool = ThreadPoolExecutor(max_workers=3)                 # cv2 releases the GIL while encoding
    threads = [threading.Thread(target=capture, args=(n, out, pool, stats, stop), daemon=True) for n in STREAMS]
    for th in threads:
        th.start()
    while not stop.is_set():
        if os.getppid() != parent:                           # the cockpit died: never record unattended
            stop.set()
        stop.wait(0.5)
    for th in threads:
        th.join(timeout=5)
    pool.shutdown(wait=True)                                 # every accepted frame reaches the disk
    secs = time.monotonic() - t0
    for s in stats.values():
        s.pop("pending", None)
        s["fps"] = round(s["frames"] / secs, 2)
    (out / "sensor_summary.json").write_text(json.dumps({"seconds": round(secs, 1), "streams": stats,
                                                         "res": [W, H], "fps_set": FPS}, indent=1))
    os.sync()


def _selftest():
    import io
    line = b"cap dqbuf: 1 seq:    149 bytesused: 814080 ts: 1659.621743 delta: 33.309 ms fps: 29.98 field: None"
    assert parse_ts(line) == (149, 1659.621743) and parse_ts(b"VIDIOC_S_FMT: ok") is None
    left, right = np.full((H, W), 7, np.uint8), np.full((H, W), 9, np.uint8)
    y8i = np.stack([left, right], -1).tobytes()
    class Trickle(io.BytesIO):                               # a pipe hands out partial reads
        def read(self, n=-1):
            return super().read(min(n, 100_000))
    got = list(frames(Trickle(y8i * 3 + y8i[:1000])))        # 3 whole frames + a torn tail
    assert len(got) == 3 and all(g == y8i for g in got)
    hdr = b"Format Video Capture:\n\tWidth/Height      : 848/480\n\tFlags             : \n"
    got = list(frames(Trickle(hdr + y8i * 2)))                # the 2026-10-16 bug: header in the pipe
    assert len(got) == 2 and all(g == y8i for g in got)
    L, R = split_y8i(got[0])
    assert (L == 7).all() and (R == 9).all()
    print("sensor_rec selftest: OK (ts parse, exact frames from partial reads, torn tail dropped, "
          "leading v4l2-ctl header skipped, Y8I split)")


if __name__ == "__main__":
    _selftest() if sys.argv[1:] == ["--selftest"] else main(sys.argv[1])
