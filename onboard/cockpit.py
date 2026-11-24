"""Runs on the Pi 5: the cockpit -- live D435 colour + Hailo depth in a web page you drive the
car from (keyboard on a laptop, hold-buttons on a phone), plus AUTONOMY: give a goal in metres
(or click the map) and the car drives there on the best real-car brain we have.

    # car/: PWM + all three failsafes
    I2C_BUS=1 ./run_board_server.sh --no-video
    # onboard/:
    ~/daniel-venv/bin/python cockpit.py --failsafes-confirmed [--hef ~/models/<x>.hef | --no-depth]
    # then open http://<pi>:8090

The camera is read through V4L2 (/dev/video4 = the D435's colour node), so librealsense is
not needed. 640x360 is the mode whose FOV (69.4 deg) the sim is calibrated to (board_server.py).
Driving goes THROUGH board_server.py on localhost:6001 as an ordinary client, so its 0.4 s
deadman, outputs-off-on-exit and the systemd/watchdog layers all stay in charge.
Manual: drive_tk.Drive (steer map left = 1150 us, ESC reverse gate). Autonomy: avoid_drive.Brain,
the SAME object avoid_drive.py runs (ground VO + slam2d + pathplan.Navigator hybrid A* / pure
pursuit, reactive avoid.plan as fallback), through avoid_drive's EscGate + to_pwm.
Autonomy follows the F1TENTH convention: the car moves only while the operator HOLDS the deadman
(G / the hold button); any manual key disengages it. A browser that goes quiet for KEYS_TTL_S
(tab hidden, laptop asleep, WiFi gone) is an instant stop. No auth: anyone who can reach :8090
can drive, same as :6001.
"""
import argparse
import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "car"))
from drive_tk import TICK_S, Drive  # noqa: E402
from run_record import debug_frame, pack  # noqa: E402
import stereo_veto  # noqa: E402

STEREO, STEREO_CAL = None, None      # the D435 depth reader (main) and its factory calibration: the near-hit veto

KEYS = {"fwd", "rev", "left", "right", "stop", "run", "arm"}
# ARM ESC: since its 2026-10-07 reprofile the XL-5 does not arm on our steady neutral after a boot
# or a PWM cut; this forward sweep (bench_pwm --forward-only) armed it. Operator-held ONLY -- run
# unattended at boot it would drive the car off by itself. The real fix is an ESC calibration.
ARM_STEPS, ARM_HOLD_S = (1500, 1519, 1538, 1556, 1575, 1594, 1612, 1631, 1650), 3.0
KEYS_TTL_S = 0.3
STREAM_FPS = 15                      # per viewer per stream; the brain still sees every frame
D435_HFOV = 69.4                     # 640x360 colour, measured (board_server.py / preflight FOV row)
S = {"bgr": None, "rgb": b"", "rgb_seq": 0, "depth": b"", "depth_seq": 0, "cam_fps": 0.0,
     "depth_ms": None, "depth_fps": 0.0, "depth_err": None, "ops": {}, "cap": 0.25, "viewers": {"rgb": 0, "depth": 0},
     "cmd": (1500, 1500, "neutral"), "board": False,
     "mission": None, "mission_req": None, "auto_out": None, "driving": False, "run_held": False, "arm_t": 0.0, "nav": {"mission": False},
     "rec_q": [], "rec_drops": 0, "manual": None, "frame": None, "recall": None, "recall_last": None}
DRY = False                          # --dry-run, or a failsafe not verified: only neutral reaches the PWM
DRY_ASKED, FAILSAFE_AUTO = False, False
MAP_FILE = Path.home() / "DANIEL/maps/memory.npz"   # map memory between missions (REMEMBER MAP)
MISSION_MAX_S = 60.0                 # a goal not reached after this much HELD driving ends as a timeout (user, 2026-11-17:
                                     # "targets are usually reachable and should be done under a minute at most")
REVERSE = "near_hit"                 # --reverse: near_hit (back up only when < 0.35 m free) | none | full (planned reverse)
cv = threading.Condition()


def publish(name, jpg):
    with cv:
        S[name] = jpg
        S[name + "_seq"] += 1
        cv.notify_all()


def emit(msg, jpg=b""):
    """To every laptop recorder on GET /record (run_record.py). A slow one loses steps, never
    stalls the brain."""
    for q in list(S["rec_q"]):
        try:
            q.put_nowait(pack(msg, jpg))
        except queue.Full:
            S["rec_drops"] += 1


def pose_now():
    """The camera pose over the floor from the D435's stereo: REC ALL's own frames while it records, else the stereo
    reader's (each owns video0 in turn; cam_pose's direct capture would find it busy)."""
    import cam_pose
    if S["recall"]:
        return cam_pose.measure(rec_dir=S["recall"]["dir"])
    return cam_pose.measure(frames=STEREO.stack()) if STEREO else cam_pose.measure()


def end_mission(m, status):
    m["status"] = status
    S["auto_out"] = None
    b = m["brain"]
    if b is not None and m.get("keep"):           # REMEMBER MAP: the next mission starts from this one's map
        try:
            MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
            b.save_memory(MAP_FILE)
            m["memory"] = "saved"
        except Exception as e:
            m["memory"] = f"save failed: {type(e).__name__}"
    x, y, th = b.slam.pose if b else (0.0, 0.0, 0.0)
    emit({"type": "end", "run": m["run"], "summary": {
        "status": status, "frames": m["frames"], "seconds": round(time.time() - m["t0"], 1),
        "final_pose_rear": [round(float(x), 3), round(float(y), 3), round(float(th), 3)],
        "dist_to_goal": b and b.ctx.get("goal_dist") and round(float(b.ctx["goal_dist"]), 2),
        "plans": b and b.nav.plans, "dead_ends": b and len(b.nav.dead_ends), "memory": m.get("memory"),
        "recorder_drops": S["rec_drops"]}})


def failsafe_faults():
    """The three runaway layers, checked ON the board (preflight.sh checks the same from the laptop):
    L1 hardware watchdog armed, L2 boot-time PWM clear ran OK, L3/L3b board_server running as the
    systemd unit whose ExecStopPost cuts the PWM. [] = all live."""
    def sh(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    bad = []
    if sh("systemctl", "show", "-p", "RuntimeWatchdogUSec", "--value") != "10s":
        bad.append("L1 watchdog")
    if [sh("systemctl", "show", "pwm-safe-boot.service", "-p", k, "--value")
            for k in ("LoadState", "UnitFileState", "Result")] != ["loaded", "enabled", "success"]:
        bad.append("L2 pwm-safe-boot")
    if (sh("systemctl", "--user", "is-active", "board-server") != "active"
            or "i2cset" not in sh("systemctl", "--user", "show", "board-server", "-p", "ExecStopPost")):
        bad.append("L3b board-server unit")
    return bad


def failsafe_watch():
    """--failsafes-auto (the boot service): no human is there to assert the failsafes, so verify
    them, and stay DRY while any is missing. Re-checked every 10 s: at boot the user units can
    start before pwm-safe-boot has run, and a layer that goes away later must stop the car."""
    global DRY
    while True:
        S["failsafe"] = failsafe_faults()
        DRY = DRY_ASKED or bool(S["failsafe"])
        time.sleep(10)


def jpeg(img, q=80):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def camera(dev):
    cap, n, t, fseq = None, 0, time.time(), 0
    while True:
        if cap is None:
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, bgr = cap.read()
        if not ok:                                # unplugged / stalled: reopen, never die
            cap.release()
            cap = None
            S["cam_fps"] = 0.0
            time.sleep(0.5)
            continue
        fseq += 1                                 # POS_MSEC = the V4L2 buffer's kernel (monotonic) stamp
        S["bgr"], S["frame"] = bgr, (bgr, fseq, cap.get(cv2.CAP_PROP_POS_MSEC) / 1e3, time.monotonic())
        rec_rgb(S["frame"])
        publish("rgb", jpeg(bgr))
        n += 1
        if time.time() - t >= 1.0:
            S["cam_fps"], n, t = n / (time.time() - t), 0, time.time()


def depth(hef):
    try:
        _depth(hef)
    except Exception as e:                        # the cockpit must outlive a broken NPU
        traceback.print_exc()
        S["depth_err"] = type(e).__name__
        if S["mission"] and S["mission"]["status"] == "active":
            end_mission(S["mission"], f"error: depth thread {type(e).__name__}")
        img = np.zeros((360, 640, 3), np.uint8)
        for y, line in ((160, "depth unavailable"), (205, S["depth_err"])):
            cv2.putText(img, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (90, 90, 255), 2, cv2.LINE_AA)
        publish("depth", jpeg(img))


MANUAL_IDLE_S = 3.0


def manual_rec(rgb, depth_bgr):
    """Teleop and ARM ESC are recorded like missions: a session opens on the first non-neutral
    PWM and closes after MANUAL_IDLE_S at neutral, so an idle cockpit writes nothing."""
    su, tu, label = S["cmd"]
    now, s = time.time(), S["manual"]
    if (su, tu) != (1500, 1500):
        if s is None:
            run = time.strftime("%Y-%m-%d_%H%M%S") + "_manual"
            s = S["manual"] = {"run": run, "t0": now, "k": 0}
            emit({"type": "start", "run": run, "meta": {
                "run": run, "kind": "manual: teleop / ARM ESC", "throttle_cap": S["cap"], "dry_run": DRY,
                "camera": "D435 640x360 V4L2", "video_fps": 20, "t": now}})
        s["last"] = now
    if s is None:
        return
    if now - s["last"] > MANUAL_IDLE_S:
        emit({"type": "end", "run": s["run"], "summary": {
            "status": "manual: idle", "frames": s["k"], "seconds": round(now - s["t0"], 1),
            "recorder_drops": S["rec_drops"]}})
        S["manual"] = None
        return
    frame = np.hstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.resize(depth_bgr, rgb.shape[1::-1])])
    emit({"type": "step", "run": s["run"], "rec": {"k": s["k"], "t": round(now - s["t0"], 3), "pwm": [su, tu],
                                                   "cmd": label, "cap": S["cap"], "dry": DRY}},
         jpeg(frame) if S["rec_q"] else b"")
    s["k"] += 1


# ============================================================== REC ALL
# Every D435 stream + the Hailo output, lossless, in experiments/<date>_sensors/rec_<hhmmss>/ ON THE PI:
# this process writes RGB (it owns video4), the Hailo disparity and the commands; sensor_rec.py writes
# depth + IR. ~33 MB/s is far past the hotspot, so onboard/sync_sensors.sh copies it to the laptop.
EXP = Path(__file__).resolve().parents[1] / "recordings"
# below it: prune verified sessions, else refuse/stop. Was 10: on the replacement 29.7 GB card (2026-11-23) that
# left ~7 GB = 3.5 min of REC ALL. 5 GB (user, 2026-11-23) covers the OS + the ~2 s auto-stop check (~70 MB at 33 MB/s)
MIN_FREE_GB = float(os.environ.get("MIN_FREE_GB", 5))
VERIFIED = "VERIFIED_ON_LAPTOP"     # sync_sensors.sh writes it after a checksum-identical laptop copy
SRC_LOG = deque(maxlen=3)           # (rgb, frame seq, kernel t) of the frames handed to the Hailo
REC_LOCK, STOP_LOCK = threading.Lock(), threading.Lock()


def free_gb():
    return shutil.disk_usage(EXP).free / 1e9


def prune():
    """Free space by deleting the OLDEST sessions the laptop holds a verified copy of. Nothing else, ever."""
    for d in sorted(EXP.glob("*_sensors/rec_*")):
        if free_gb() >= MIN_FREE_GB:
            break
        if closed(d) and (d / VERIFIED).exists():
            shutil.rmtree(d)
            print(f"[recall] pruned {d} (verified on the laptop)", flush=True)
    return free_gb() >= MIN_FREE_GB


def recall_start():
    """-> None, or why it refused."""
    EXP.mkdir(exist_ok=True)
    with STOP_LOCK:
        if S["recall"]:
            return "already recording"
        if free_gb() < MIN_FREE_GB and not prune():
            return f"only {free_gb():.1f} GB free on the Pi (needs {MIN_FREE_GB:.0f}), nothing laptop-verified to prune"
        d = EXP / f"{time.strftime('%Y-%m-%d')}_sensors" / f"rec_{time.strftime('%H%M%S')}"
        (d / "rgb").mkdir(parents=True)
        calib = Path(__file__).with_name("d435_calib.json")
        if calib.exists():
            shutil.copy(calib, d)
        ctl = {dev: subprocess.run(["v4l2-ctl", "-d", dev, "-l"], capture_output=True, text=True).stdout
               for dev in ("/dev/video0", "/dev/video2", "/dev/video4")}
        (d / "meta.json").write_text(json.dumps({
            "start_wall": time.time(), "start_mono": time.monotonic(), "throttle_cap": S["cap"], "dry_run": DRY,
            "calib": "d435_calib.json" if calib.exists() else "MISSING: run onboard/d435_calib.py once (laptop)",
            "clock": "every t_* is CLOCK_MONOTONIC s; t_kernel = the V4L2 buffer stamp (start of exposure)",
            "streams": {
                "rgb": "rgb/NNNNNN.jpg q95 640x360, colour /dev/video4 YUYV (69.4 deg HFOV); rgb.csv",
                "depth": "depth/NNNNNN.png uint16 mm 848x480, Z16 /dev/video0; depth.csv",
                "ir_left/ir_right": "NNNNNN.png uint8 848x480, Y8I /dev/video2 de-interleaved; ir.csv",
                "hailo_v5": "hailo_v5.f16 raw float16 frames, shape + crop in hailo_v5.json; hailo_v5.csv",
                "commands": "commands.jsonl, one line per 20 Hz control tick"},
            "v4l2_controls": ctl}, indent=1))
        if STEREO:
            STEREO.handover(d)                        # the stereo reader lets go of video0 for sensor_rec
        r = {"dir": d, "t0": time.monotonic(), "free0": free_gb(), "q": queue.Queue(maxsize=150), "closed": False,   # 5 s: rides out an SD stall
             "n": {"rgb": 0, "hailo_v5": 0, "commands": 0}, "rgb_drops": 0, "hailo_meta": False,
             "rgb_csv": open(d / "rgb.csv", "w", buffering=1), "h_csv": open(d / "hailo_v5.csv", "w", buffering=1),
             "f16": open(d / "hailo_v5.f16", "wb"), "cmds": open(d / "commands.jsonl", "w", buffering=1),
             "proc": subprocess.Popen([sys.executable, str(Path(__file__).with_name("sensor_rec.py")), str(d)])}
        r["rgb_csv"].write("k,seq,t_kernel,t_mono\n")
        r["h_csv"].write("k,src_seq,src_t_kernel,t_mono,infer_ms\n")
        r["writer"] = threading.Thread(target=_rgb_writer, args=(r,), daemon=True)
        r["writer"].start()
        (EXP / ".recording").write_text(str(d))      # sync_sensors.sh stays off the SD card meanwhile
        S["recall"] = r
    print(f"[recall] recording -> {d}", flush=True)


def recall_stop(why="stopped by operator"):
    with STOP_LOCK:
        r, S["recall"] = S["recall"], None
        if not r:
            return
        r["proc"].terminate()                         # sensor_rec drains its PNG queue, then exits
        r["q"].put(None)
        r["writer"].join(timeout=15)
        try:
            r["proc"].wait(timeout=30)
        except subprocess.TimeoutExpired:
            r["proc"].kill()
        if STEREO:
            STEREO.take_back()                        # video0 is free again: the stereo reader streams it live
        with REC_LOCK:
            r["closed"] = True
            for f in ("rgb_csv", "h_csv", "f16", "cmds"):
                r[f].close()
        d, secs = r["dir"], time.monotonic() - r["t0"]
        try:
            sens = json.loads((d / "sensor_summary.json").read_text())["streams"]
        except (OSError, ValueError, KeyError):
            sens = {"sensor_rec": "wrote no summary"}
        streams = {k: {"frames": n, "fps": round(n / secs, 2)} for k, n in r["n"].items()}
        streams["rgb"]["drops"] = r["rgb_drops"]
        gb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
        summ = {"status": why, "seconds": round(secs, 1), "gb": round(gb, 2), "free_gb_end": round(free_gb(), 1),
                "streams": {**streams, **sens}}
        (d / "summary.json").write_text(json.dumps(summ, indent=1))      # LAST: non-empty = closed
        os.sync()    # a power cut right after stop zeroed this file and the data tails (2026-10-16)
        (EXP / ".recording").unlink(missing_ok=True)
        S["recall_last"] = {"dir": str(d), "status": why, "gb": summ["gb"], "seconds": summ["seconds"]}
    print(f"[recall] {why}: {d} {summ}", flush=True)


def closed(d):
    """A session is closed once summary.json has CONTENT: a power cut can leave it at 0 bytes."""
    s = d / "summary.json"
    return s.exists() and s.stat().st_size > 0


def recall_close_orphans():
    """A session the cockpit died in has no (or an empty) summary.json; close it so the laptop syncs it."""
    for d in EXP.glob("*_sensors/rec_*"):
        if not closed(d):
            (d / "summary.json").write_text(json.dumps({"status": "closed at cockpit start: it stopped mid-recording"}))
    (EXP / ".recording").unlink(missing_ok=True)


def recall_status():
    r, f = S["recall"], round(free_gb(), 1)
    if not r:
        return {"on": False, "free_gb": f, "last": S["recall_last"]}
    secs, gb = time.monotonic() - r["t0"], max(r["free0"] - f, 0.0)
    return {"on": True, "secs": round(secs), "gb": round(gb, 2), "mbps": round(gb * 1e3 / max(secs, 1)), "free_gb": f,
            "n": r["n"], "rgb_drops": r["rgb_drops"]}


def _rgb_writer(r):
    while (item := r["q"].get()) is not None:
        bgr, seq, tk, tm = item
        k = r["n"]["rgb"]
        cv2.imwrite(str(r["dir"] / "rgb" / f"{k:06d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        r["rgb_csv"].write(f"{k},{seq},{tk:.6f},{tm:.6f}\n")
        r["n"]["rgb"] += 1


def rec_rgb(frame):
    r = S["recall"]
    if r:
        try:
            r["q"].put_nowait(frame)
        except queue.Full:                            # the SD card fell behind: count it, never stall the camera
            r["rgb_drops"] += 1


def rec_hailo(rgb, disp, ms, net, hef):
    r = S["recall"]
    if not r:
        return
    seq, tk = next(((s, t) for img, s, t in SRC_LOG if img is rgb), (-1, float("nan")))
    with REC_LOCK:
        if r["closed"]:
            return
        if not r["hailo_meta"]:
            (r["dir"] / "hailo_v5.json").write_text(json.dumps({
                "shape": list(disp.shape), "dtype": "float16", "hef": hef,
                "crop_box_xywh_in_rgb": [int(v) for v in net.box],
                "units": "the student's raw output, relative disparity (near = large)"}))
            r["hailo_meta"] = True
        r["f16"].write(disp.astype(np.float16).tobytes())
        r["h_csv"].write(f"{r['n']['hailo_v5']},{seq},{tk:.6f},{time.monotonic():.6f},{ms:.2f}\n")
        r["n"]["hailo_v5"] += 1


def rec_cmd(held, now):
    r = S["recall"]
    if not r:
        return
    m, (su, tu, label) = S["mission"], S["cmd"]
    with REC_LOCK:
        if r["closed"]:
            return
        r["cmds"].write(json.dumps({"t_mono": round(now, 4), "pwm": [su, tu], "cmd": label, "keys": sorted(held),
                                    "cap": S["cap"], "driving": S["driving"], "dry": DRY,
                                    "mission": m and {"status": m["status"], "mode": m["mode"]}}) + "\n")
        r["n"]["commands"] += 1
    if r["n"]["commands"] % 20 == 0 and free_gb() < MIN_FREE_GB and not prune():
        # never block the 20 Hz control loop on the drain
        threading.Thread(target=recall_stop, args=(f"auto-stop: under {MIN_FREE_GB:.0f} GB free",),
                         daemon=True).start()


def newest_frames():
    seen = 0
    while True:
        with cv:
            cv.wait_for(lambda: S["rgb_seq"] != seen)
            seen, (bgr, fseq, tk, _) = S["rgb_seq"], S["frame"]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        SRC_LOG.append((rgb, fseq, tk))           # REC ALL maps each Hailo output back to its frame
        yield rgb


def _depth(hef):
    from avoid_drive import Brain, depth_stream
    from edge_depth import Net
    from hailo_platform import HailoPowerMode, HailoSchedulingAlgorithm, VDevice
    vp = VDevice.create_params()
    vp.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN   # else all-zero output (Fact 43)
    net = Net(hef, "hailo", VDevice(vp))
    net.im.set_power_mode(HailoPowerMode.ULTRA_PERFORMANCE)
    n, t0 = 0, time.time()
    # the NPU infers frame k+1 while this thread colours / plans frame k (avoid_drive.depth_stream)
    for k, (rgb, disp, ms) in enumerate(depth_stream(newest_frames(), net)):
        S["depth_ms"] = ms
        rec_hailo(rgb, disp, ms, net, hef)
        req, S["mission_req"] = S["mission_req"], None
        m = S["mission"]
        if req and req["action"] == "start":
            S["auto_out"] = None
            if m and m["status"] == "active":
                end_mission(m, "aborted: superseded by a new goal")
            run = time.strftime("%Y-%m-%d_%H%M%S")
            meta = {"run": run, "goal_root_m": list(req["goal"]), "cam_height": req["cam_h"],
                    "pitch_deg": req["pitch"], "hfov": D435_HFOV, "hef": hef, "throttle_cap": S["cap"],
                    "dry_run": DRY, "brain": f"avoid_drive.Brain(nav=True, forward_only={REVERSE != 'full'}, near_hit_reverse={REVERSE == 'near_hit'}, range_k=0)", "camera": "D435 640x360 V4L2",
                    "video_fps": 20, "t": time.time()}
            m = {"run": run, "t0": time.time(), "frames": 0, "status": "starting", "mode": "starting",
                 "trail": [], "brain": None}
            S["mission"] = m
            emit({"type": "start", "run": run, "meta": meta})
            try:
                cam_h, pitch = req["cam_h"], req["pitch"]
                try:                                  # the camera pose, measured NOW on the floor (D435 stereo)
                    import cam_pose
                    r = pose_now()
                    if r["floor_frac"] >= 0.6 and 0.05 < r["height_m"] < 0.5:
                        cam_h, pitch = r["height_m"], r["pitch_down_deg"]
                        m["plane"] = (r["normal"], r["height_m"])     # the stereo veto's floor
                    m["cam_pose"] = r
                except Exception as e:                # no floor in view: keep the typed values
                    m["cam_pose"] = {"error": type(e).__name__}
                emit({"type": "step", "run": run, "rec": {"k": -1, "t": 0.0, "cam_pose": m["cam_pose"],
                                              "used": {"cam_height": cam_h, "pitch_deg": pitch}}})
                m["brain"] = Brain(goal=req["goal"], nav=True, hfov=D435_HFOV, cam_height=cam_h,
                                   pitch_deg=pitch, forward_only=REVERSE != "full",     # reverse works since 2026-11-17 (ESC LED
                                   near_hit_reverse=REVERSE == "near_hit",      # solid red); used only at a near-hit
                                   range_k=0.0)                           # true ranges (stereo-checked)
                m["keep"], m["memory"] = req.get("keep", False), "off"
                if m["keep"]:
                    try:                              # same mark, same heading as the mission that saved it
                        m["brain"].load_memory(MAP_FILE)
                        m["memory"] = "loaded"
                    except FileNotFoundError:
                        m["memory"] = "new"
                    except Exception as e:
                        m["memory"] = f"load failed: {type(e).__name__}"
                emit({"type": "step", "run": run, "rec": {"k": -1, "t": 0.0, "memory": m["memory"]}})
                m["status"] = "active"
            except Exception as e:
                traceback.print_exc()
                end_mission(m, f"error: {type(e).__name__}")
        elif req and m and m["status"] == "active":
            end_mission(m, "aborted: " + req.get("why", "cancelled"))
        if m and m["status"] == "active" and not S["run_held"]:
            # deadman up: the brain is PAUSED, not stepped -- stepping it on a car that is not
            # allowed to move made the Navigator call itself stalled and plan a reverse retreat
            # ... and its reverse check restarts: counting the paused seconds called a reverse that had not begun
            # "blocked behind" (2026-11-23: 102435 t20, 103650 t40 froze 15 s)
            m["brain"].t_prev, m["brain"].rev, m["mode"], S["auto_out"] = None, None, "paused", None
        if m and m["status"] == "active" and S["run_held"]:
            b = m["brain"]
            try:
                zs = STEREO.latest() if STEREO and m.get("plane") else None
                o = b.step(rgb, disp, net.box, time.perf_counter(),
                           stereo=None if zs is None else stereo_veto.stereo_bins(zs, STEREO_CAL, *m["plane"]))
            except Exception as e:                # a planner bug must never leave the car driving
                traceback.print_exc()
                end_mission(m, f"error: {type(e).__name__}")
            else:
                S["auto_out"] = (float(o["steer"]), float(o["throttle"]), time.monotonic())
                if not S["driving"]:              # deadman up / ESC braking: the car did not move
                    b.ctx["state"]["last_cmd"] = (float(o["steer"]), 0.0)
                m["mode"] = o["mode"]
                x, y, th = b.slam.pose
                root = (x + 0.156 * np.cos(th), y + 0.156 * np.sin(th))
                if not m["trail"] or np.hypot(root[0] - m["trail"][-1][0], root[1] - m["trail"][-1][1]) > 0.02:
                    m["trail"] = (m["trail"] + [[round(root[0], 3), round(root[1], 3)]])[-2000:]
                if k % 4 == 0:
                    S["nav"] = nav_state(m)
                frame, view = debug_frame(rgb, disp, o, b.ctx)
                publish("depth", jpeg(cv2.resize(view, (rgb.shape[1], rgb.shape[0]))))
                su, tu, phase = S["cmd"]
                emit({"type": "step", "run": m["run"], "rec": {
                    "k": m["frames"], "t": round(time.time() - m["t0"], 3), "steer": round(float(o["steer"]), 3),
                    "throttle": round(float(o["throttle"]), 3), "mode": o["mode"], "esc": phase,
                    "pwm": [su, tu], "driving": S["driving"], "plans": b.nav.plans,
                    "pose": [round(float(v), 3) for v in b.slam.pose], "free": round(float(o["free"]), 2),
                    "goal_dist": round(float(b.ctx["goal_dist"]), 3), "ms_depth": round(ms, 1),
                    "dead": len(b.nav.dead_ends), "progress": b.progress(),
                    "stereo": zs is not None, "vetoed": o["vetoed"]}},
                     jpeg(frame) if S["rec_q"] else b"")
                m["frames"] += 1
                m["held_s"] = m.get("held_s", 0.0) + float(b.ctx.get("dt", 0.0))   # brain dt: paused time not counted
                if o["mode"] == "goal":
                    end_mission(m, "arrived")
                elif m["held_s"] > MISSION_MAX_S:
                    end_mission(m, f"timeout: {MISSION_MAX_S:.0f} s of driving without arriving")
        else:
            if m and k % 6 == 0:
                S["nav"] = nav_state(m)
            elif not m:
                S["nav"] = {"mission": False}
            lo, hi = np.percentile(disp, (2, 98))
            v = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
            img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            publish("depth", jpeg(cv2.resize(img, (rgb.shape[1], rgb.shape[0]))))
            manual_rec(rgb, img)
        n += 1
        if time.time() - t0 >= 1.0:
            S["depth_fps"], n, t0 = n / (time.time() - t0), 0, time.time()


def nav_state(m):
    """Map telemetry for the browser, world frame = the rear axle at mission start (x fwd, y left)."""
    b = m["brain"]
    if b is None:
        return {"mission": False}
    x, y, th = b.slam.pose
    sl = b.slam
    ix, iy = np.nonzero(sl.L > 1.5)                                   # confidently occupied cells
    occ = np.stack([(ix - sl.o + 0.5) * sl.res, (iy - sl.o + 0.5) * sl.res], 1)
    occ = occ[np.hypot(occ[:, 0] - x, occ[:, 1] - y) < 8.0][::max(1, len(occ) // 2000)]
    path = getattr(b.nav, "path", None)
    path = np.asarray(path, float)[:, :2] if path is not None and len(path) else np.zeros((0, 2))
    g, dist = b.goal_w, b.ctx.get("goal_dist")
    return {"mission": True, "status": m["status"], "mode": m["mode"],
            "goal": [round(float(g[0]), 2), round(float(g[1]), 2)],
            "pose": [round(float(x + 0.156 * np.cos(th)), 3), round(float(y + 0.156 * np.sin(th)), 3),
                     round(float(th), 3)],
            "dist": None if dist is None else round(float(dist), 2), "plans": b.nav.plans,
            "dead": [[round(float(v), 2) for v in d] for d in b.nav.dead_ends],   # T frame = this frame (start 0,0,0)
            "prog": b.progress(), "memory": m.get("memory"),
            "trail": m["trail"][::max(1, len(m["trail"]) // 400)], "occ": np.round(occ, 2).tolist(),
            "path": np.round(path[::max(1, len(path) // 300)], 2).tolist()}


def control(board):
    from avoid_drive import EscGate, to_pwm
    drive, esc, sock = Drive(), EscGate(), None
    while True:
        time.sleep(TICK_S)
        now = time.monotonic()
        # One slot per operator page (laptop, phone): the deadman is held if ANY fresh operator holds
        # it, ANY operator's manual key / STOP wins, and the LOWEST throttle cap applies. A single
        # shared slot let an idle open page overwrite the holder's G 20x a second (2026-10-07).
        ops = [o for o in list(S["ops"].values()) if now - o[1] < KEYS_TTL_S]
        fresh = bool(ops)
        held = set().union(*(o[0] for o in ops))    # no fresh operator = deadman up = stop (goal kept)
        S["cap"] = min((o[2] for o in ops), default=S["cap"])
        manual, m, out = held - {"run", "arm"}, S["mission"], S["auto_out"]
        active = m is not None and m["status"] == "active"
        if active and manual:                     # a real manual key disengages autonomy
            S["mission_req"] = {"action": "cancel", "why": "manual override"}
        # ARM works with a goal set too: the usual order after a boot is SET GOAL, hold G, see nothing
        # move, hold ARM -- which `not active` silently ignored (2026-10-07). The brain pauses meanwhile.
        arming = "arm" in held and not manual
        S["run_held"] = active and "run" in held and not manual and not arming
        S["arm_t"] = S["arm_t"] + TICK_S if arming else 0.0       # releasing restarts the sweep
        if arming:
            k = int(S["arm_t"] / ARM_HOLD_S)
            us = min(ARM_STEPS[k], round(1500 + 500 * S["cap"])) if k < len(ARM_STEPS) else 1500
            S["cmd"], S["driving"] = (1500, us, f"arming {us} us" if k < len(ARM_STEPS) else "armed? release"), False
        elif S["run_held"] and out and now - out[2] < 0.5:
            thr, phase = esc(out[1], now)
            su, tu = to_pwm(out[0], thr, S["cap"])
            S["cmd"], S["driving"] = (su, tu, "auto " + phase), phase in ("fwd", "rev") and not DRY
        else:
            esc(0.0, now)
            S["cmd"], S["driving"] = drive.step(manual if fresh else {"stop"}, S["cap"], now), False
        rec_cmd(held, now)
        try:
            if sock is None:
                sock = socket.create_connection((board, 6001), timeout=1)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.sendall(("1500,1500\n" if DRY else "{},{}\n".format(*S["cmd"][:2])).encode())
            S["board"] = True
        except OSError:
            if sock is not None:
                sock.close()
            sock, S["board"] = None, False
            time.sleep(0.5)


class Handler(BaseHTTPRequestHandler):
    # Keep-alive: HTTP/1.0 (the default) opened a NEW TCP connection for every /keys post, 20 per
    # second, and under video load a lost SYN costs TCP's 1 s retransmit -- the 1000 ms tail that
    # tripped the 0.3 s deadman (2026-10-07). Every reply here carries Content-Length or never ends.
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/record":
            q = queue.Queue(maxsize=300)
            S["rec_q"].append(q)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            try:
                while True:
                    try:
                        self.wfile.write(q.get(timeout=2.0))
                    except queue.Empty:
                        self.wfile.write(pack({"type": "hb"}))      # keeps the laptop's read alive
            except OSError:
                pass
            finally:
                S["rec_q"].remove(q)
            return
        if self.path == "/ops":                    # who is connected and holding what (debug)
            now = time.monotonic()
            return self._send(json.dumps({k: {"keys": sorted(o[0]), "age_s": round(now - o[1], 2), "cap": o[2]}
                                          for k, o in list(S["ops"].items())}).encode(), "application/json")
        if self.path == "/nav":
            # NaN is not JSON: one NaN anywhere made the browser drop every map update
            return self._send(json.dumps(S["nav"]).replace("NaN", "null").encode(), "application/json")
        name = {"/rgb.mjpg": "rgb", "/depth.mjpg": "depth"}.get(self.path)
        if name is None:
            return self._send(PAGE.encode(), "text/html; charset=utf-8")
        # DROP, never queue (board_server video_server, 2026-08-24): with the kernel's auto-tuned send
        # buffer, two viewers over campus WiFi + Tailscale backed up 1.3 MB and pushed every round
        # trip, control included, to 155-460 ms (2026-10-07). A small buffer makes a slow viewer skip
        # to the newest frame instead; the rate cap keeps two viewers well inside the link.
        self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024)
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seen, last = -1, 0.0
        S["viewers"][name] += 1
        try:
            while True:
                with cv:
                    cv.wait_for(lambda: S[name + "_seq"] != seen, timeout=2.0)
                    seen, jpg = S[name + "_seq"], S[name]
                if time.monotonic() - last < S["viewers"][name] / STREAM_FPS:   # budget shared by viewers
                    continue
                last = time.monotonic()
                if jpg:
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                                     % len(jpg) + jpg + b"\r\n")
        except OSError:
            pass                                  # viewer went away
        finally:
            S["viewers"][name] -= 1

    def _json(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

    def do_POST(self):
        if self.path == "/measure":                # camera height / pitch from the D435's own stereo depth
            import cam_pose
            try:
                r = pose_now()
            except Exception as e:
                r = {"error": f"{type(e).__name__}: {e}"[:200]}
            return self._send(json.dumps(r).encode(), "application/json")
        if self.path == "/record_all":
            try:
                on = bool(self._json().get("on"))
            except (ValueError, TypeError, AttributeError):
                return self._send(b'{"ok": false, "error": "bad request"}', "application/json")
            err = recall_start() if on else recall_stop()
            return self._send(json.dumps({"ok": err is None, "error": err, "recall": recall_status()}).encode(),
                              "application/json")
        if self.path == "/mission":
            try:
                msg = self._json()
                if msg.get("action") == "start":
                    S["mission_req"] = {"action": "start",
                                        "goal": (float(msg["goal"][0]), float(msg["goal"][1])),
                                        "cam_h": min(max(float(msg.get("cam_h", 0.14)), 0.03), 1.0),
                                        "pitch": min(max(float(msg.get("pitch", 0.0)), -30.0), 30.0),
                                        "keep": bool(msg.get("keep"))}
                elif msg.get("action") == "forget":
                    MAP_FILE.unlink(missing_ok=True)
                else:
                    S["mission_req"] = {"action": "cancel", "why": "cancelled"}
            except (ValueError, TypeError, KeyError, IndexError):
                return self._send(b'{"ok": false}', "application/json")
            return self._send(b'{"ok": true}', "application/json")
        if self.path != "/keys":
            return self._send(b"", "text/plain")
        now = time.monotonic()
        try:
            msg = self._json()
            op = f"{self.client_address[0]}/{msg.get('id', '')}"[:80]
            S["ops"][op] = (set(msg.get("keys", [])) & KEYS, now, min(max(float(msg.get("cap", 0.25)), 0.05), 1.0))
        except (ValueError, TypeError, AttributeError):
            S["ops"][f"{self.client_address[0]}/bad"] = ({"stop"}, now, 0.05)
        for k, o in list(S["ops"].items()):            # forget pages closed > 10 s ago
            if now - o[1] > 10:
                S["ops"].pop(k, None)
        su, tu, phase = S["cmd"]
        m = S["mission"]
        self._send(json.dumps({"steer": su, "throttle": tu, "phase": phase, "board": S["board"],
                               "fps": round(S["cam_fps"], 1),
                               "depth_ms": S["depth_ms"] and round(S["depth_ms"]),
                               "depth_fps": round(S["depth_fps"], 1), "depth": DEPTH_ON,
                               "depth_err": S["depth_err"], "driving": S["driving"], "rec": len(S["rec_q"]), "dry": DRY, "recall": recall_status(),
                               "failsafe": S.get("failsafe"),
                               "operators": sum(now - o[1] < 1.0 for o in list(S["ops"].values())),
                               "mission": m and {"status": m["status"], "mode": m["mode"]}}).encode(),
                   "application/json")


PAGE = """<!doctype html><html lang=en><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>DANIEL cockpit</title>
<style>
:root{--bg:#0b0f14;--card:#121821;--line:#223041;--ink:#e6edf3;--dim:#8b98a8;--acc:#4cc2ff;
 --go:#2ecc71;--brk:#ff5d5d;--warn:#f5b041}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.4 system-ui,-apple-system,Segoe UI,sans-serif;-webkit-user-select:none;user-select:none}
header{display:flex;align-items:center;gap:14px;padding:12px 18px;border-bottom:1px solid var(--line);flex-wrap:wrap}
h1{font-size:17px;margin:0;letter-spacing:.08em}h1 b{color:var(--acc)}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-left:auto}
.chip{padding:4px 10px;border:1px solid var(--line);border-radius:99px;color:var(--dim);font-size:13px;white-space:nowrap}
.chip i{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--dim);margin-right:6px}
.ok i{background:var(--go)}.bad i{background:var(--brk)}.ok,.bad{color:var(--ink)}
#c_mode.isauto{border-color:var(--acc);color:var(--acc)}
main{padding:16px 18px;display:grid;gap:16px;max-width:1400px;margin:auto}
.views{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:900px){.views{grid-template-columns:1fr}}
.view{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden;position:relative}
.view img{width:100%;aspect-ratio:16/9;display:block;background:#000;object-fit:cover}
.fps{position:absolute;right:12px;top:10px;padding:3px 9px;border-radius:7px;background:#000a;
 font:600 13px ui-monospace,monospace;color:var(--go)}
.tag{position:absolute;left:12px;top:10px;padding:3px 9px;border-radius:7px;background:#000a;font-size:12px;letter-spacing:.06em}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}
.panel{display:grid;grid-template-columns:1.3fr 1fr auto;gap:22px;align-items:center}
@media(max-width:900px){.panel,.auto{grid-template-columns:1fr!important}}
.g{margin:6px 0 14px}.g label{display:flex;justify-content:space-between;color:var(--dim);font-size:13px;margin-bottom:6px}
.g label b{color:var(--ink);font-variant-numeric:tabular-nums}
.track{position:relative;height:14px;background:#0a0e13;border:1px solid var(--line);border-radius:99px}
.track:after{content:"";position:absolute;left:50%;top:-4px;bottom:-4px;width:2px;background:var(--line)}
.fill{position:absolute;top:2px;bottom:2px;border-radius:99px;transition:all .06s linear}
.keys{display:flex;flex-wrap:wrap;gap:6px;color:var(--dim);font-size:13px}
kbd{border:1px solid var(--line);border-bottom-width:3px;border-radius:6px;padding:1px 7px;color:var(--ink);font:12px ui-monospace,monospace}
input[type=range]{width:100%;accent-color:var(--acc)}
.pad{display:grid;grid-template-columns:repeat(3,64px);grid-template-rows:64px 64px auto auto;gap:8px}
button{font:inherit;cursor:pointer}
.pad button{border:1px solid var(--line);background:#16202b;color:var(--ink);border-radius:12px;font-size:24px;touch-action:none}
.pad button.on{background:var(--acc);color:#000}
#stop{grid-column:1/-1;height:auto;padding:10px;font-size:15px;font-weight:700;letter-spacing:.1em;background:#3a1418;border-color:#6b2128;color:#ffb3b3}
#stop.on{background:var(--brk);color:#fff}
#arm{grid-column:1/-1;padding:10px;font-size:13px;font-weight:700;letter-spacing:.08em;background:#2a2410;border-color:#6b5a21;color:#f5d58b}
#arm.on{background:var(--warn);color:#000}
#recall{grid-column:1/-1;padding:10px;font-size:13px;font-weight:700;letter-spacing:.08em;background:#2a1014;border-color:#6b2130;color:#f5a3b0}
#recall.on{background:var(--brk);color:#fff}
.lost{outline:3px solid var(--brk)}
.auto{display:grid;grid-template-columns:minmax(280px,1fr) 1.1fr;gap:22px}
h2{margin:0 0 12px;font-size:13px;letter-spacing:.12em;color:var(--dim);font-weight:600}
#map{width:100%;aspect-ratio:1;background:#0a0e13;border:1px solid var(--line);border-radius:12px;cursor:crosshair;display:block}
.row{display:flex;gap:10px;align-items:end;flex-wrap:wrap;margin-bottom:12px}
.f{display:flex;flex-direction:column;gap:4px;font-size:12px;color:var(--dim)}
.f input{width:88px;padding:8px 10px;border-radius:9px;border:1px solid var(--line);background:#0a0e13;color:var(--ink);font:15px ui-monospace,monospace}
.pre{border:1px solid var(--line);background:#16202b;color:var(--ink);border-radius:9px;padding:8px 12px}
.btn{border:0;border-radius:11px;padding:11px 18px;font-weight:700;letter-spacing:.06em}
#go{background:var(--acc);color:#001018}#cancel{background:#26313d;color:var(--ink)}
#hold{width:100%;margin:10px 0 12px;padding:22px;border-radius:14px;border:2px solid var(--line);background:#16202b;
 color:var(--dim);font-size:18px;font-weight:800;letter-spacing:.1em;touch-action:none}
#hold.armed{border-color:var(--acc);color:var(--acc)}#hold.on{background:var(--go);border-color:var(--go);color:#002010}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.stat{background:#0a0e13;border:1px solid var(--line);border-radius:10px;padding:8px 10px}
.stat span{display:block;font-size:11px;color:var(--dim);letter-spacing:.08em}.stat b{font:600 15px ui-monospace,monospace}
#mstat{font-weight:700;letter-spacing:.06em}.hint{font-size:12px;color:var(--dim);margin-top:10px}
</style>
<header><h1><b>DANIEL</b> COCKPIT</h1>
<div class=chips><span class=chip id=c_mode>MANUAL</span><span class=chip id=c_link><i></i>link</span>
<span class=chip id=c_board><i></i>board</span><span class=chip id=c_cam><i></i>camera</span>
<span class=chip id=c_depth><i></i>depth</span><span class=chip id=c_rec><i></i>rec</span><span class=chip id=c_recall><i></i>rec all</span><span class=chip id=c_fs><i></i>failsafes</span><span class=chip id=c_ops>1 operator</span><span class=chip id=c_esc>ESC neutral</span></div></header>
<main>
<div class=views>
 <div class=view><span class=tag>REALSENSE D435 &middot; RGB</span><span class=fps id=f_rgb>-- fps</span><img src="/rgb.mjpg" alt="camera"></div>
 <div class=view id=dview><span class=tag id=dtag>HAILO-8L &middot; DEPTH (near = bright)</span><span class=fps id=f_depth>-- fps</span><img src="/depth.mjpg" alt="depth"></div>
</div>
<div class="card auto">
 <div><h2>MAP &middot; SLAM &middot; PLAN</h2><canvas id=map width=560 height=560></canvas>
  <div class=keys style="margin-top:8px"><span style="color:#4cc2ff">&#9632; car / trail</span>
  <span style="color:#2ecc71">&#9632; planned path</span><span style="color:#ff5d5d">&#9632; obstacles</span>
  <span style="color:#f5b041">&#9632; goal</span><span style="color:#ff2d55">&#10005; dead end</span><span>grid 1 m &middot; click to set a goal</span></div></div>
 <div><h2>AUTONOMY &middot; GO TO GOAL</h2>
  <div class=row><label class=f>FORWARD m<input id=gx type=number step=0.5 value=3></label>
   <label class=f>LEFT m<input id=gy type=number step=0.5 value=0></label>
   <button class=pre data-d=2>2 m</button><button class=pre data-d=5>5 m</button><button class=pre data-d=10>10 m</button></div>
  <div class=row><label class=f>CAMERA HEIGHT m<input id=ch type=number step=0.005 value=0.18></label>
   <label class=f>PITCH DOWN &deg;<input id=cp type=number step=0.1 value=2.05></label>
   <button class=pre id=meas title="fit the floor in the D435's stereo depth: car on its wheels, flat floor in view">MEASURE</button>
   <button class="btn" id=go>SET GOAL</button><button class="btn" id=cancel>CANCEL</button></div>
  <div class=row><label class=f title="load the last remembered map at SET GOAL and save this mission's map at its end: obstacles and dead ends carry over. Start on the SAME mark, facing the SAME way."><input id=keep type=checkbox> REMEMBER MAP</label>
   <button class=pre id=forget title="delete the remembered map">FORGET MAP</button></div>
  <div class=hint id=mres>SET GOAL measures the camera pose itself (D435 stereo floor fit); these values are the fallback.</div>
  <div id=mstat>MANUAL &middot; no goal</div>
  <button id=hold>HOLD TO DRIVE AUTONOMOUSLY &nbsp;<kbd>G</kbd></button>
  <div class=stats><div class=stat><span>TO GOAL</span><b id=s_dist>--</b></div>
   <div class=stat><span>PLANNER</span><b id=s_mode>--</b></div>
   <div class=stat><span>POSE x,y,&theta;</span><b id=s_pose>--</b></div>
   <div class=stat><span>RE-PLANS</span><b id=s_plans>--</b></div>
   <div class=stat><span>PROGRESS m/s &middot; NEAREST</span><b id=s_prog>--</b></div>
   <div class=stat><span>MAP MEMORY &middot; DEAD ENDS</span><b id=s_mem>--</b></div></div>
  <div class=hint>The car moves only while you hold G (or the button). Release = stop.
   Any manual key cancels the goal. Odometry is camera-only: measure the camera height.
   It reverses only briefly: at a near-hit (under 0.35 m free) or after 1 s with no way forward (ESC LED solid
   red = armed). The camera faces forward: hold G only while the path behind is clear. Gives up after 60 s.</div></div>
</div>
<div class="card panel" id=panel>
 <div>
  <div class=g><label>STEERING <b id=s_us>1500 us</b></label><div class=track><div class=fill id=s_fill></div></div></div>
  <div class=g><label>THROTTLE <b id=t_us>1500 us</b></label><div class=track><div class=fill id=t_fill></div></div></div>
  <div class=keys><span><kbd>W</kbd>/<kbd>&uarr;</kbd> forward</span><span><kbd>S</kbd>/<kbd>&darr;</kbd> brake</span>
  <span><kbd>A</kbd>/<kbd>&larr;</kbd> left</span><span><kbd>D</kbd>/<kbd>&rarr;</kbd> right</span><span><kbd>Space</kbd> stop</span></div>
 </div>
 <div><div class=g><label>THROTTLE CAP (manual + auto) <b id=cap_v>0.25</b></label>
  <input type=range id=cap min=0.05 max=1 step=0.05 value=0.25></div>
  <div class=keys>Switching tab or window stops the car. So does losing the link.</div></div>
 <div class=pad>
  <span></span><button data-k=fwd>&#9650;</button><span></span>
  <button data-k=left>&#9664;</button><button data-k=rev>&#9660;</button><button data-k=right>&#9654;</button>
  <button id=stop>STOP</button>
  <button id=arm title="after a boot: hold until the wheels turn, then release. It drives FORWARD.">ARM ESC &middot; hold</button>
  <button id=recall title="tap to start / stop: every D435 stream (RGB, depth, IR L+R) + the Hailo output, lossless, on the Pi; the laptop syncs it">&#9679; REC ALL</button>
 </div>
</div></main>
<script>
const K={KeyW:'fwd',ArrowUp:'fwd',KeyS:'rev',ArrowDown:'rev',KeyA:'left',ArrowLeft:'left',
 KeyD:'right',ArrowRight:'right',Space:'stop',KeyG:'run'};
const held=new Set(),touch=new Set();let latched=false;const CID=Math.random().toString(36).slice(2,10);const $=id=>document.getElementById(id);
const typing=()=>document.activeElement&&document.activeElement.tagName==='INPUT'&&document.activeElement.type==='number';
addEventListener('keydown',e=>{if(K[e.code]&&!typing()){held.add(K[e.code]);e.preventDefault()}});
addEventListener('keyup',e=>{if(K[e.code])held.delete(K[e.code])});
const clear=()=>{held.clear();touch.clear();document.querySelectorAll('.pad .on,#hold.on').forEach(b=>b!==$('stop')&&b.classList.remove('on'))};
addEventListener('blur',clear);document.addEventListener('visibilitychange',()=>document.hidden&&clear());
function holdable(b,k){b.addEventListener('pointerdown',e=>{b.setPointerCapture(e.pointerId);touch.add(k);b.classList.add('on')});
 ['pointerup','pointercancel','lostpointercapture'].forEach(n=>b.addEventListener(n,()=>{touch.delete(k);b.classList.remove('on')}))}
document.querySelectorAll('.pad button[data-k]').forEach(b=>holdable(b,b.dataset.k));holdable($('hold'),'run');holdable($('arm'),'arm');
$('stop').onclick=()=>{latched=!latched;$('stop').classList.toggle('on',latched);$('stop').textContent=latched?'STOPPED - tap to release':'STOP'};
$('cap').oninput=()=>$('cap_v').textContent=(+$('cap').value).toFixed(2);
function chip(id,ok,txt){const c=$(id);c.className='chip '+(ok?'ok':'bad');c.lastChild.textContent=txt}
function bar(el,us,lo,hi,col){const f=(us-1500)/(us<1500?1500-lo:hi-1500)/2;
 el.style.left=(f<0?50+f*100:50)+'%';el.style.width=Math.abs(f*100)+'%';el.style.background=col}
let MS=null;
let RA=false,raErr='',raErrT=0;const mmss=x=>Math.floor(x/60)+':'+String(x%60).padStart(2,'0');
$('recall').onclick=async()=>{try{const r=await (await fetch('/record_all',{method:'POST',body:JSON.stringify({on:!RA})})).json();
 if(!r.ok){raErr='REC ALL refused: '+r.error;raErrT=Date.now()+8000}}catch(e){raErr='REC ALL: no reply';raErrT=Date.now()+8000}};
async function tick(){
 const keys=latched?['stop']:[...new Set([...held,...touch])];
 try{const r=await fetch('/keys',{method:'POST',cache:'no-store',
   body:JSON.stringify({id:CID,keys,cap:+$('cap').value})});const s=await r.json();MS=s.mission;
  chip('c_link',1,'link');chip('c_board',s.board,s.board?'board':'board LOST');
  chip('c_cam',s.fps>5,'camera '+s.fps+' fps');
  chip('c_depth',s.depth&&s.depth_ms!=null&&!s.depth_err,!s.depth?'depth off':s.depth_err?'depth: '+s.depth_err:
   s.depth_ms!=null?'depth '+s.depth_ms+' ms':'depth starting');
  chip('c_rec',s.rec>0,s.rec>0?'REC (laptop)':'not recording');
  const ra=s.recall;RA=!!(ra&&ra.on);$('recall').classList.toggle('on',RA);$('recall').textContent=RA?'\u25a0 STOP REC ALL':'\u25cf REC ALL';
  chip('c_recall',RA||Date.now()>raErrT,Date.now()<raErrT?raErr:RA?`REC ALL ${mmss(ra.secs)} \u00b7 ${ra.gb.toFixed(1)} GB \u00b7 ${ra.mbps} MB/s \u00b7 Pi free ${ra.free_gb} GB`:
   `rec all off \u00b7 Pi free ${ra.free_gb} GB`+(ra.last?` \u00b7 last ${ra.last.gb} GB, ${ra.last.status}`:''));
  if(RA)$('c_recall').style.color='var(--brk)';else $('c_recall').style.color='';
  $('c_ops').textContent=s.operators+(s.operators==1?' operator':' operators');
  const fs=s.failsafe;chip('c_fs',!fs||!fs.length,fs==null?'failsafes: human-confirmed':fs.length?'NOT LIVE: '+fs.join(', '):'failsafes verified');
  if(s.dry)$('c_esc').style.color='var(--warn)';
  $('f_rgb').textContent=s.fps.toFixed(1)+' fps';
  $('f_depth').textContent=s.depth_err?'--':s.depth_fps.toFixed(1)+' fps \\u00b7 '+(s.depth_ms??'--')+' ms';
  $('c_esc').textContent=(s.dry?'DRY RUN \\u00b7 ':'')+'ESC '+s.phase;$('s_us').textContent=s.steer+' us';$('t_us').textContent=s.throttle+' us';
  bar($('s_fill'),s.steer,1150,1850,'var(--acc)');   // left = 1150 us = bar to the left
  bar($('t_fill'),s.throttle,1000,2000,s.throttle>=1500?'var(--go)':'var(--brk)');
  const act=MS&&MS.status==='active';$('c_mode').textContent=act?(s.driving?'AUTO \\u00b7 DRIVING':'AUTO \\u00b7 HOLD G'):'MANUAL';
  $('c_mode').className='chip'+(act?' isauto':'');$('hold').classList.toggle('armed',act);
  $('dtag').textContent=act&&MS.mode!=='paused'?'PLANNER \\u00b7 depth, obstacles, chosen arc':'HAILO-8L \\u00b7 DEPTH (near = bright)';
  $('mstat').textContent=!MS?'MANUAL \\u00b7 no goal':act?(s.driving?'DRIVING TO GOAL':MS.mode==='paused'?'GOAL SET \\u00b7 hold G to drive':'PLANNING\\u2026 keep holding G'):MS.status.toUpperCase();
  $('mstat').style.color=!MS?'var(--dim)':act?'var(--acc)':MS.status==='arrived'?'var(--go)':'var(--warn)';
  $('panel').classList.toggle('lost',!s.board)}
 catch(e){chip('c_link',0,'link LOST - car stopping');$('panel').classList.add('lost')}
 setTimeout(tick,50)}
// ---- map: world frame = rear axle at goal-set time, x forward (up), y left (left) ----
const cvs=$('map'),g=cvs.getContext('2d');let NAV={mission:false},V=null,preview=false;
const running=()=>NAV.mission&&NAV.status==='active';
function previewGoal(){if(running())return;preview=true;NAV={mission:false};draw()}
const pending=()=>[+$('gx').value+0.156,+$('gy').value];
function view(){let p=[[0,0],[0.156,0]];if(NAV.mission){p.push(NAV.goal,NAV.pose);p=p.concat(NAV.trail)}else p.push(pending());
 let xs=p.map(q=>q[0]),ys=p.map(q=>q[1]);let x0=Math.min(...xs)-1.5,x1=Math.max(...xs)+1.5,y0=Math.min(...ys)-1.5,y1=Math.max(...ys)+1.5;
 const span=Math.max(6,x1-x0,y1-y0);return{sc:cvs.width/span,ox:(x0+x1)/2,oy:(y0+y1)/2}}
const P=(x,y)=>[cvs.width/2-(y-V.oy)*V.sc,cvs.height/2-(x-V.ox)*V.sc];
function line(pts,col,w){if(pts.length<2)return;g.strokeStyle=col;g.lineWidth=w;g.beginPath();
 pts.forEach((q,i)=>{const[a,b]=P(q[0],q[1]);i?g.lineTo(a,b):g.moveTo(a,b)});g.stroke()}
function car(x,y,th){const[a,b]=P(x,y),s=Math.max(7,0.3*V.sc);g.save();g.translate(a,b);g.rotate(-th);
 g.fillStyle='#4cc2ff';g.beginPath();g.moveTo(0,-s);g.lineTo(s*0.6,s*0.7);g.lineTo(-s*0.6,s*0.7);g.closePath();g.fill();g.restore()}
function flag(x,y){const[a,b]=P(x,y);g.strokeStyle='#f5b041';g.lineWidth=2;g.beginPath();g.arc(a,b,10,0,7);g.stroke();
 g.beginPath();g.moveTo(a-15,b);g.lineTo(a+15,b);g.moveTo(a,b-15);g.lineTo(a,b+15);g.stroke()}
function draw(){V=view();const W=cvs.width;g.clearRect(0,0,W,W);g.strokeStyle='#17212c';g.lineWidth=1;
 const span=W/V.sc;for(let k=Math.floor(V.ox-span);k<=V.ox+span;k++){const[,b]=P(k,0);g.beginPath();g.moveTo(0,b);g.lineTo(W,b);g.stroke()}
 for(let k=Math.floor(V.oy-span);k<=V.oy+span;k++){const[a]=P(0,k);g.beginPath();g.moveTo(a,0);g.lineTo(a,W);g.stroke()}
 if(NAV.mission){g.fillStyle='#ff5d5d';const r=Math.max(2,0.05*V.sc);NAV.occ.forEach(q=>{const[a,b]=P(q[0],q[1]);g.fillRect(a-r/2,b-r/2,r,r)});
  (NAV.dead||[]).forEach(d=>{const[a,b]=P(d[0],d[1]),r=6;g.strokeStyle='#ff2d55';g.lineWidth=2;
   g.beginPath();g.arc(a,b,r,0,7);g.moveTo(a-r*0.7,b-r*0.7);g.lineTo(a+r*0.7,b+r*0.7);g.moveTo(a+r*0.7,b-r*0.7);g.lineTo(a-r*0.7,b+r*0.7);g.stroke()});
  line(NAV.path,'#2ecc71',3);line(NAV.trail,'#4cc2ff',2);flag(NAV.goal[0],NAV.goal[1]);car(NAV.pose[0],NAV.pose[1],NAV.pose[2])}
 else{flag(...pending());car(0.156,0,0)}}
cvs.addEventListener('click',e=>{if(running())return;const r=cvs.getBoundingClientRect();
 const sx=(e.clientX-r.left)*cvs.width/r.width,sy=(e.clientY-r.top)*cvs.height/r.height;
 $('gx').value=(V.ox-(sy-cvs.height/2)/V.sc-0.156).toFixed(1);$('gy').value=(V.oy-(sx-cvs.width/2)/V.sc).toFixed(1);previewGoal()});
['gx','gy'].forEach(i=>$(i).oninput=previewGoal);
document.querySelectorAll('.pre').forEach(b=>b.onclick=()=>{if(running())return;$('gx').value=b.dataset.d;$('gy').value=0;previewGoal()});
const post=(o)=>fetch('/mission',{method:'POST',body:JSON.stringify(o)});
$('go').onclick=()=>(preview=false,post({action:'start',goal:[+$('gx').value,+$('gy').value],cam_h:+$('ch').value,pitch:+$('cp').value,keep:$('keep').checked}));
$('forget').onclick=()=>{if(!running())post({action:'forget'})};
$('cancel').onclick=()=>post({action:'cancel'});
$('meas').onclick=async()=>{$('mres').textContent='measuring the floor with the D435 stereo depth\u2026';
 try{const r=await (await fetch('/measure',{method:'POST'})).json();
  if(r.error){$('mres').textContent='measure failed: '+r.error;return}
  $('ch').value=r.height_m;$('cp').value=r.pitch_down_deg;
  $('mres').textContent='measured: height '+r.height_m+' m, pitch '+r.pitch_down_deg+'\u00b0 down, roll '+r.roll_deg+
   '\u00b0 | floor '+Math.round(r.floor_frac*100)+'% of points, rms '+r.rms_mm+' mm'+(r.floor_frac<0.5?' \u2014 LOW: is the floor in view?':'')}
 catch(e){$('mres').textContent='measure failed: link'}};
async function navtick(){try{const r=await fetch('/nav',{cache:'no-store'});const n=await r.json();
 if(!n.mission)NAV={mission:false};else if(n.status==='active'||!preview){NAV=n;preview=false}draw();
 if(n.mission){$('s_dist').textContent=n.dist!=null?n.dist.toFixed(2)+' m':'--';$('s_mode').textContent=n.mode;
  $('s_pose').textContent=n.pose[0].toFixed(2)+', '+n.pose[1].toFixed(2)+', '+(n.pose[2]*57.3).toFixed(0)+'\\u00b0';$('s_plans').textContent=n.plans;
  $('s_prog').textContent=n.prog?(n.prog.rate>=0?'+':'')+n.prog.rate.toFixed(2)+' \u00b7 '+n.prog.best.toFixed(2)+' m':'--';
  $('s_mem').textContent=(n.memory||'off')+' \u00b7 '+(n.dead||[]).length}}
 catch(e){}setTimeout(navtick,250)}
if(!""" + "{depth}" + """){$('dview').style.display='none';$('go').disabled=true}
draw();tick();navtick();
</script></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hef", default=str(Path.home() / "models/dav2student_v5_288x512.hef"))
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--cam", default="/dev/video4", help="the D435 colour node (v4l2-ctl --list-formats: YUYV)")
    ap.add_argument("--board", default="127.0.0.1", help="where board_server.py listens")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--failsafes-confirmed", action="store_true", help="a human asserts, in session, the failsafes are live")
    ap.add_argument("--failsafes-auto", action="store_true",
                    help="boot service: verify L1/L2/L3b on the board instead, and stay DRY while any is missing")
    ap.add_argument("--dry-run", action="store_true",
                    help="everything runs and records, but the PWM only ever gets neutral: hand-carry tests")
    ap.add_argument("--reverse", choices=["near_hit", "none", "full"], default="near_hit",
                    help="near_hit: forward-only plans, back up only when < 0.35 m is free on two frames (default; "
                         "the first stand run with full reverse reversed on 53 %% of frames with 2.2 m free); none: "
                         "never reverse (2026-10-07..11-17); full: planned reverse legs / retreats / 3-point turns (HITL)")
    a = ap.parse_args()
    if not (a.failsafes_confirmed or a.failsafes_auto or a.dry_run):
        raise SystemExit("REFUSED: the cockpit drives the car. Run preflight.sh; when every failsafe "
                         "row is PASS, re-run with --failsafes-confirmed (or --failsafes-auto).")
    DEPTH_ON, DRY, DRY_ASKED, FAILSAFE_AUTO = not a.no_depth, a.dry_run, a.dry_run, a.failsafes_auto
    REVERSE = a.reverse
    if FAILSAFE_AUTO:
        DRY = True                                # until the first check passes
        threading.Thread(target=failsafe_watch, daemon=True).start()
    PAGE = PAGE.replace("{depth}", "true" if DEPTH_ON else "false")
    recall_close_orphans()
    # systemd stop / restart mid-recording: close the session properly (the drain takes a few s)
    signal.signal(signal.SIGTERM, lambda *_: (recall_stop("cockpit stopped"), sys.exit(0)))
    threading.Thread(target=camera, args=(a.cam,), daemon=True).start()
    if DEPTH_ON:
        threading.Thread(target=depth, args=(a.hef,), daemon=True).start()
        STEREO, STEREO_CAL = stereo_veto.StereoReader(), stereo_veto.calib()   # owns video0 (REC ALL borrows it)
    threading.Thread(target=control, args=(a.board,), daemon=True).start()
    print(f"cockpit on http://0.0.0.0:{a.port}  camera {a.cam}  depth {a.hef if DEPTH_ON else 'off'}",
          flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()
