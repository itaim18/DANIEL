"""The real car brain on the Pi 5: camera -> depth on the Hailo-8L -> avoid.plan -> PWM.

The same avoid.py + Net that the HITL suite drives in Isaac Sim (experiments/2026-09-29_hitl);
only the frame source and the command sink differ.

    # dry run, no car, no board: a video or a live Pi camera, commands logged + debug video
    python3 avoid_drive.py --hef ~/models/<hef> --video clip.mp4 --out run/
    python3 avoid_drive.py --hef ~/models/<hef> --camera --out run/
    python3 avoid_drive.py --hef ~/models/<hef> --camera --goal 5,0 --nav --out run/   # global planning
    python3 avoid_drive.py --selftest                                               # ESC gate check

    # ARMED: sends steer/throttle to board_server.py (control port 6001, one "steer_us,throttle_us"
    # line per frame; the board's operator deadman + watchdog still gate every line)
    python3 avoid_drive.py --hef ~/models/<hef> --camera --host <board> \
        --arm --failsafes-confirmed --max-throttle 0.25

SAFETY: writes NOTHING unless both --arm and --failsafes-confirmed are given. The second
flag is a HUMAN assertion, made in session, that the three runaway failsafes are live
(CLAUDE.md; memory traxxas-runaway-failsafes). It is logged into the run record. On any
exception, Ctrl-C or a Hailo stall the car gets neutral. Throttle is capped by --max-throttle
BEFORE the PWM map. Without --goal the planner's goal is "straight ahead, far": it cruises and avoids.
With --goal X,Y it localises itself (ground-plane visual odometry: floor features tracked
between frames and lifted to metric ground points with the known camera height; plus the
slam2d occupancy map for routing) and drives to that point, then holds.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import avoid  # noqa: E402
import pathplan  # noqa: E402
import slam2d  # noqa: E402
import stereo_veto  # noqa: E402
from edge_depth import Net  # noqa: E402
from hitl_pi import crop_intrinsics  # noqa: E402
from run_record import Recorder, debug_frame  # noqa: E402

CONTROL_PORT = 6001
# car/drive_policy.py:84-89 (mdp.AckermannPwmActionCfg). A CALIBRATION,
# not a constant: a servo whose full lock is not at 1150/1850 under- or over-steers by the
# ratio (bench_pwm_calibration.md). STEER_US is (steer -1, steer +1) = (right, left) lock:
# measured on the real car 2026-10-07, 1850 us turns RIGHT, so avoid.py's "+ = left" must
# reach 1150. The sim's (1150, 1850) has +1 -> 1850 = sim-left: mirrored on this car.
STEER_US, THROTTLE_US, NEUTRAL = (1850, 1150), (1000, 2000), 1500
PROG_S = 3.0          # s: the window of Brain.progress()'s closing speed (cockpit telemetry, mission logs)
# Blocked BEHIND (the camera faces forward, so nothing sees it): reverse PWM has been out REV_CHECK_S (after the ESC
# gate's brake + gap) and BOTH the SLAM pose moved < REV_STILL_M AND the camera image changed < REV_STILL_IMG grey
# levels (40x22 thumbnail) -> stop reversing until the car has driven REV_CLEAR_M forward. Both, not either: in the
# 2026-11-19 lab missions free reverses often showed a big image change with ~1 cm of VO motion (VO lagging), while
# the stuck ones were still in both (2 cm / 0.4 levels; 4 cm / 0.5) and then repeated their 1.5 s backups.
REV_CHECK_S, REV_STILL_M, REV_STILL_IMG, REV_CLEAR_M = 0.5, 0.05, 3.0, 0.2


class EscGate:
    """The XL-5HV in its factory Trail mode: "a single push forward on the throttle applies
    brake, and a second push activates reverse" (traxxas.com/news/trx-4-trail-crawl-modes) --
    from a STANDSTILL too, not only after forward (measured 2026-10-07: a single reverse push
    from rest never reversed). So every entry into reverse is: reverse PWM for BRAKE_S (push
    1), neutral for GAP_S, then reverse (push 2).
    2026-11-17: the real car REVERSES through this gate (manual, from the cockpit; ESC LED solid red = armed).
    CALIBRATION, unverified on the car: time the brake-to-stop on the first drive."""
    BRAKE_S, GAP_S = 0.30, 0.15

    def __init__(self):
        self.phase, self.t = "neutral", 0.0

    def __call__(self, throttle, now):
        """-> (throttle to send, phase). phase brake/gap: the car is NOT driving backwards yet."""
        if throttle > 0.02:
            self.phase = "fwd"
        elif throttle < -0.02:
            if self.phase in ("fwd", "neutral"):    # EVERY reverse entry, standstill too
                self.phase, self.t = "brake", now
            if self.phase == "brake" and now - self.t >= self.BRAKE_S:
                self.phase, self.t = "gap", now
            if self.phase == "gap":
                if now - self.t < self.GAP_S:
                    return 0.0, "gap"
                self.phase = "rev"
        elif self.phase == "rev":
            self.phase = "neutral"
        return throttle, self.phase


def to_pwm(steer, throttle, max_throttle):
    s = float(np.clip(steer, -1, 1))
    t = float(np.clip(throttle, -max_throttle, max_throttle))
    return (int(round(sum(STEER_US) / 2 + s * (STEER_US[1] - STEER_US[0]) / 2)),
            int(round(sum(THROTTLE_US) / 2 + t * (THROTTLE_US[1] - THROTTLE_US[0]) / 2)))


class Board:
    """board_server.py control client; a no-op unless armed."""

    def __init__(self, host, armed):
        self.s = socket.create_connection((host, CONTROL_PORT), timeout=5.0) if armed else None
        if self.s:
            self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def send(self, su, tu):
        if self.s:
            self.s.sendall(f"{su},{tu}\n".encode())

    def neutral(self):
        for _ in range(10):
            try:
                self.send(NEUTRAL, NEUTRAL)
            except OSError:
                return
            time.sleep(0.05)


def frames(a):
    """Yield RGB frames: Pi camera (picamera2) or a video file, at the POV aspect."""
    if a.video:
        cap = cv2.VideoCapture(a.video)
        while True:
            ok, bgr = cap.read()
            if not ok:
                return
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(main={"size": (a.cam_w, a.cam_h), "format": "RGB888"}))
    cam.start()
    try:
        while True:
            # picamera2 "RGB888" is BGR in memory order (drive_policy.PiCam: same conversion)
            yield cv2.cvtColor(cam.capture_array(), cv2.COLOR_BGR2RGB)
    finally:
        cam.stop()


def depth_stream(src, net):
    """-> (rgb, disparity, depth ms) with the NPU one frame AHEAD of the CPU: the Hailo
    infers frame k+1 while the planner, VO and SLAM work on frame k, so a cycle costs
    max(NPU, CPU) instead of their sum. One inference in flight at a time: Net writes
    into one output buffer, so each result is copied before the next launch."""
    it = iter(src)
    rgb = next(it, None)
    if rgb is None:
        return
    t = time.perf_counter()
    w = net.launch(net.pre(rgb))
    for nxt in it:
        disp = net.post(w()).copy()
        ms = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        w = net.launch(net.pre(nxt))                  # the NPU starts on frame k+1 ...
        yield rgb, disp, ms                           # ... while the CPU plans frame k
        rgb = nxt
    yield rgb, net.post(w()).copy(), (time.perf_counter() - t) * 1e3


class Brain:
    """One frame of the real-car brain: disparity [+ ground VO + slam2d + Navigator] -> avoid.plan's
    output dict (steer/throttle in action units, mode, scan, ...). main() and the Pi cockpit
    (cockpit.py) both drive through it, so the best configuration has ONE implementation. The ESC
    gate and the PWM map stay with the caller: they belong to the actuator, not the planner."""

    def __init__(self, goal=None, nav=False, terrain=None, start=None, goal_bearing=0.0, goal_tol=0.4,
                 hfov=66.0, cam_height=0.14, pitch_deg=0.0, debug_dir=None, forward_only=False, range_k=None,
                 near_hit_reverse=False):
        if nav and goal is None:
            raise ValueError("nav needs a goal")
        if terrain and not start:
            raise ValueError("terrain needs start (where the car sits in the map)")
        self.hfov, self.cam_height, self.pitch_deg = hfov, cam_height, pitch_deg
        self.goal_bearing, self.goal_tol = np.radians(goal_bearing), goal_tol
        # forward_only: plans have no reverse legs, no retreats, no backups; and no negative throttle
        # ever leaves step(). The real car could not reverse from 2026-10-07 until 2026-11-17 (it does
        # now, ESC LED solid red); kept for replays of the October deployed brain and as an escape hatch.
        # near_hit_reverse (with forward_only; the real car since 2026-11-17): the ONE reverse allowed is the
        # reactive backup when < 0.35 m is free on two frames (avoid.plan "backup") -- no planned reverse legs,
        # retreats or 3-point turns (the first stand run reversed on 53 % of frames with 2.2 m free ahead).
        self.forward_only, self.near_hit_reverse = forward_only, near_hit_reverse
        self.ctx, self.t_prev = {"forward_only": forward_only, "near_hit_reverse": near_hit_reverse}, None
        if range_k is not None:
            # avoid.RANGE_K = 0.5 halves every obstacle range (a HITL sweep). On the real car the
            # geometric correction (range_k=0) put a box at 0.73 m vs the D435's stereo 0.72 m
            # (2026-10-07); 0.5 put it at 0.36-0.42 m and the car refused to move.
            self.ctx["range_k"] = range_k
        self.slam = self.gvo = self.goal_w = self.nav = None
        if goal is not None:
            self.slam, self.gvo = slam2d.Slam2D(), slam2d.GroundVO()
            self.goal_w = np.asarray(goal, float) + [0.156, 0.0]          # root -> rear axle frame
        if nav:
            if terrain:
                sx, sy, syaw = (float(v) for v in start.split(","))
                T, st = pathplan.load_terrain(terrain), pathplan.rear_of((sx, sy), np.radians(syaw))
            else:
                T, st = pathplan.open_terrain(), (0.0, 0.0, 0.0)
            self.nav = pathplan.Navigator(T, st, pathplan.compose(st, (self.goal_w[0], self.goal_w[1], 0.0))[:2],
                                          async_plan=True,   # plan in a worker process; the car stops meanwhile
                                          speed_loop=False,  # its speed is VO's: boosting on it ran away in HITL
                                          forward_only=forward_only)
            if debug_dir:
                self.nav.debug_dir = debug_dir                           # failed plans, for offline probing
                Path(debug_dir).mkdir(parents=True, exist_ok=True)
        self.prog, self.best = [], None                 # (t, goal distance) over the last PROG_S; nearest yet
        self.rev = None             # the reverse in progress: {"t0", "p0", "img0"} (p0/img0 once the ESC really reverses)
        self.rear_block = None      # (x, y) where reversing was found blocked; no reverse until REV_CLEAR_M away

    def save_memory(self, path):
        """Map memory for the next mission: the planner's map (+ dead ends) and SLAM's grid."""
        np.savez_compressed(path, slam_L=self.slam.L, **{"nav_" + k: v for k, v in self.nav.memory().items()})

    def load_memory(self, path):
        """The previous mission's map. Same frame only if the car starts on the same mark, facing the
        same way (the frame is the rear axle at goal-set time)."""
        z = np.load(path)
        if z["slam_L"].shape != self.slam.L.shape:
            raise ValueError("map memory was saved on a different SLAM grid")
        self.nav.restore({k[4:]: z[k] for k in z.files if k.startswith("nav_")})
        self.slam.L[:] = z["slam_L"]

    def progress(self):
        """Goal distance now, nearest yet, and the closing speed over the last PROG_S (m/s, + = nearer)."""
        if not self.prog:
            return None
        (t0, d0), (t1, d1) = self.prog[0], self.prog[-1]
        return {"dist": round(d1, 2), "best": round(self.best, 2), "rate": round((d0 - d1) / max(t1 - t0, 1e-3), 2)}

    def step(self, rgb, disp, box, now, stereo=None):
        """stereo: stereo_veto.stereo_bins() of a fresh D435 depth frame, or None: the network's near hits the
        stereo clearly contradicts are dropped (o["vetoed"] = rays moved)."""
        ctx, slam = self.ctx, self.slam
        H, W = rgb.shape[:2]
        fx = (W / 2) / np.tan(np.radians(self.hfov) / 2)
        hdr = {"fx": fx, "fy": fx, "cx": W / 2, "cy": H / 2 - fx * np.tan(np.radians(self.pitch_deg)),
               "cam_h": self.cam_height}
        if not np.isfinite(disp).all() or float(disp.std()) < 1e-6:
            raise RuntimeError("depth output is constant/NaN: the NPU is not really running; "
                               "refusing to drive on it")
        ctx.update(crop_intrinsics(hdr, box, disp.shape))
        dt = (now - self.t_prev) if self.t_prev else 0.05
        ctx.update(goal_bearing=self.goal_bearing, goal_dist=10.0, dt=dt)
        self.t_prev = now
        if slam is not None:                        # goal from the Pi's own pose
            x, y, th = slam.pose
            g = self.goal_w - [x + 0.156 * np.cos(th), y + 0.156 * np.sin(th)]
            ctx["goal_dist"] = float(np.hypot(*g))
            ctx["goal_bearing"] = float((np.arctan2(g[1], g[0]) - th + np.pi) % (2 * np.pi) - np.pi)
            self.prog = [q for q in self.prog if now - q[0] <= PROG_S] + [(now, ctx["goal_dist"])]
            self.best = ctx["goal_dist"] if self.best is None else min(self.best, ctx["goal_dist"])
            ctx["map_pts"] = slam2d.map_points(slam)
        cmd = ctx.get("state", {}).get("last_cmd", (0.0, 0.0))
        ctx["veto"] = None if stereo is None else (lambda p_, d_: stereo_veto.veto(p_, d_, *stereo))
        o = avoid.plan(disp, ctx)
        o["vetoed"] = ctx.pop("vetoed", 0)
        if slam is not None:
            vo = self.gvo.step(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), hdr, self.cam_height, dt)
            slam.step(o["scan"], cmd[0], cmd[1], dt, vo=vo)
            if self.nav is not None:
                r = self.nav.step(tuple(slam.pose), o["scan"], cmd[1], dt)
                if r is None:                            # no path: the reactive planner drives
                    o = dict(o, mode="fb_" + o["mode"])
                else:
                    o = dict(o, steer=r[0], throttle=r[1], mode=r[2])
                    ctx["state"]["stuck_s"] = 0.0           # the global plan drives: the reactive stuck timer restarts
                    ctx["state"]["last_cmd"] = (float(r[0]), float(r[1]))
            if ctx["goal_dist"] < self.goal_tol:           # arrived: stop and hold
                o = dict(o, steer=0.0, throttle=0.0, mode="goal")
        if self.forward_only and o["throttle"] < 0 and not reverse_allowed(o["mode"], self.near_hit_reverse):
            o = dict(o, throttle=0.0, mode=o["mode"] + "|no_rev")
            ctx.setdefault("state", {})["last_cmd"] = (float(o["steer"]), 0.0)
        o = self.rear_guard(o, rgb, now)
        ctx["last_steer"] = o["steer"]
        return o

    def rear_guard(self, o, rgb, now):
        """Stop a reverse that does not move (blocked behind), and allow none until the car drove REV_CLEAR_M away."""
        pose = None if self.slam is None else self.slam.pose[:2].copy()
        if self.rear_block is not None and (np.hypot(*(pose - self.rear_block)) > REV_CLEAR_M if pose is not None
                                            else o["throttle"] > 0):      # no pose (no goal): the first forward command
            self.rear_block = None
        if o["throttle"] >= 0:
            self.rev = None
            return o
        if self.rear_block is not None:
            return self._no_reverse(o)
        thumb = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (40, 22), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.rev is None:
            self.rev = {"t0": now, "p0": None, "img0": None}
        el = now - self.rev["t0"] - (EscGate.BRAKE_S + EscGate.GAP_S)      # time the ESC has really been reversing
        if el >= 0 and self.rev["img0"] is None:
            self.rev["p0"], self.rev["img0"] = pose, thumb
        if el >= REV_CHECK_S:
            moved = 0.0 if pose is None or self.rev["p0"] is None else float(np.hypot(*(pose - self.rev["p0"])))
            if moved < REV_STILL_M and float(np.abs(thumb - self.rev["img0"]).mean()) < REV_STILL_IMG:
                self.rear_block = pose if pose is not None else np.zeros(2)
                self.rev = None
                if self.nav is not None:
                    self.nav.rear_blocked(tuple(self.slam.pose))
                st = self.ctx.setdefault("state", {})
                st["backup"], st["stuck_s"] = 0, 0.0                       # end avoid.plan's time-based backup now
                return self._no_reverse(o)
            self.rev = {"t0": now - (EscGate.BRAKE_S + EscGate.GAP_S), "p0": pose, "img0": thumb}   # check the next 0.5 s
        return o

    def _no_reverse(self, o):
        self.ctx.setdefault("state", {})["last_cmd"] = (float(o["steer"]), 0.0)
        return dict(o, throttle=0.0, mode=o["mode"] + "|rear_blocked")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hef", required=True, help="HEF on the Pi, or an ONNX with --backend onnx (laptop dry run)")
    ap.add_argument("--backend", choices=["hailo", "onnx"], default="hailo")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--camera", action="store_true")
    ap.add_argument("--cam-w", type=int, default=1280)
    ap.add_argument("--cam-h", type=int, default=720)
    ap.add_argument("--hfov", type=float, default=66.0, help="camera horizontal FOV (deg): CALIBRATE")
    ap.add_argument("--cam-height", type=float, default=0.14, help="lens height above ground (m): MEASURE")
    ap.add_argument("--pitch-deg", type=float, default=0.0,
                    help="lens pitched DOWN by this (deg): MEASURE. Ranging and VO put the horizon at "
                         "cy = H/2 - fy tan(pitch); 3 deg off moves it ~50 px at 720p and doubles or "
                         "halves every range. Car on flat floor, a floor mark D m straight ahead at image "
                         "row v (of H): pitch = atan(h/D) - atan((v - H/2)/fy), fy = (W/2)/tan(hfov/2)")
    ap.add_argument("--host", default="127.0.0.1", help="board_server.py (PWM bridge); only used with --arm")
    ap.add_argument("--arm", action="store_true")
    ap.add_argument("--failsafes-confirmed", action="store_true")
    ap.add_argument("--max-throttle", type=float, default=0.25)
    ap.add_argument("--goal-bearing", type=float, default=0.0, help="deg, + = left, when no --goal")
    ap.add_argument("--goal", help="'X,Y' metres from the start pose (x forward, y left): the Pi "
                                   "localises itself (ground-plane VO + slam2d map) and drives there")
    ap.add_argument("--goal-tol", type=float, default=0.4)
    ap.add_argument("--forward-only", action="store_true",
                    help="no reverse: plans without reverse legs, no retreats/backups (the real car before 2026-11-17)")
    ap.add_argument("--near-hit-reverse", action="store_true",
                    help="with --forward-only: still back up when < 0.35 m is free on two frames (the real car since 2026-11-17)")
    ap.add_argument("--nav", action="store_true",
                    help="with --goal: GLOBAL planning (pathplan.Navigator: hybrid A* on the sensed "
                         "occupancy map [+ --terrain], pure pursuit, re-plan when blocked) instead of "
                         "the reactive planner, which stays the fallback when no path exists")
    ap.add_argument("--terrain", help="nav: prior drivable-terrain field (kino_plan.py --export); "
                                      "needs --start. Without it: open ground, sensed obstacles only")
    ap.add_argument("--start", help="nav + --terrain: 'X,Y,YAW_DEG' of the car (root) in the terrain "
                                    "map's frame, as placed: MEASURE it")
    ap.add_argument("--out", help="dir: commands.jsonl + debug.mp4")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    armed = a.arm and a.failsafes_confirmed
    if a.arm and not a.failsafes_confirmed:
        raise SystemExit("--arm needs --failsafes-confirmed (a human assertion, in session)")

    vdev = None
    if a.backend == "hailo":
        from hailo_platform import HailoPowerMode, HailoSchedulingAlgorithm, VDevice
        # The scheduler MUST be on: with VDevice() defaults the async InferModel call on a
        # never-activated network returns at once with the output buffer untouched -- all
        # zeros in 1 ms, no exception. The first Pi dry run drove 300 frames on that
        # (2026-09-30). hitl_pi.py always set ROUND_ROBIN, which is why HITL was fine.
        vp = VDevice.create_params()
        vp.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        vdev = VDevice(vp)
        net = Net(a.hef, "hailo", vdev)
        net.im.set_power_mode(HailoPowerMode.ULTRA_PERFORMANCE)
    else:
        net = Net(a.hef, "onnx")
    out = Path(a.out) if a.out else None
    rec_out = Recorder(out, dict(vars(a), armed=armed, t=time.time())) if out else None
    board, k = Board(a.host, armed), 0
    esc = EscGate()
    if a.nav and not a.goal:
        raise SystemExit("--nav needs --goal")
    if a.terrain and not a.start:
        raise SystemExit("--terrain needs --start (where the car sits in the map)")
    brain = Brain(goal=[float(v) for v in a.goal.split(",")] if a.goal else None, nav=a.nav,
                  terrain=a.terrain, start=a.start, goal_bearing=a.goal_bearing, goal_tol=a.goal_tol,
                  hfov=a.hfov, cam_height=a.cam_height, pitch_deg=a.pitch_deg,
                  debug_dir=str(out / "navfail") if out and a.nav else None, forward_only=a.forward_only,
                  near_hit_reverse=a.near_hit_reverse)
    ctx, slam, nav = brain.ctx, brain.slam, brain.nav
    if nav is not None:
        # a re-plan takes up to ~1.5 s on the Pi (3.5 s if it fails): neutral throttle meanwhile,
        # steering held, instead of the last command carrying the car on blind
        nav.on_plan = lambda: board.send(to_pwm(ctx.get("last_steer", 0.0), 0.0, a.max_throttle)[0], NEUTRAL)
    print(f"[drive] {Path(a.hef).name} {net.h}x{net.w} {'ARMED' if armed else 'dry run'} "
          f"max_throttle {a.max_throttle}", flush=True)
    try:
        for rgb, disp, ms_depth in depth_stream(frames(a), net):
            t1 = time.perf_counter()
            o = brain.step(rgb, disp, net.box, t1)
            thr, esc_phase = esc(o["throttle"], time.monotonic())
            if esc_phase in ("brake", "gap"):             # not moving backwards yet: odometry knows
                ctx["state"]["last_cmd"] = (float(o["steer"]), 0.0)
            su, tu = to_pwm(o["steer"], thr, a.max_throttle)
            board.send(su, tu)
            t2 = time.perf_counter()
            rec = {"k": k, "steer": round(o["steer"], 3), "throttle": round(o["throttle"], 3), "mode": o["mode"],
                   "esc": esc_phase, "plans": nav.plans if nav else None,
                   "pose": [round(float(v), 3) for v in slam.pose] if slam is not None else None,
                   "free": round(o["free"], 2), "pwm": [su, tu], "ms_depth": round(ms_depth, 1),
                   "ms_plan": round((t2 - t1) * 1e3, 1)}
            if rec_out:
                rec_out.write(rec, debug_frame(rgb, disp, o, ctx)[0])
            if k % 20 == 0:
                print(f"[drive] {k} {rec}", flush=True)
            k += 1
            if a.max_frames and k >= a.max_frames:
                break
    finally:
        board.neutral()
        if rec_out:
            rec_out.close()
        if vdev is not None:
            net.cim.shutdown()
            net.cim = net.im = None
            vdev.release()
        print(f"[drive] stopped after {k} frames, car at neutral", flush=True)


def reverse_allowed(mode, near_hit_reverse):
    """The forward-only brain's last guard: a negative throttle leaves it only as the reactive near-hit / stuck
    backup (avoid.plan "backup", or "fb_backup" when the global planner has no path), and only with
    near_hit_reverse. The Navigator's dead-end retreat ("nav_deadend") is out (user, 2026-11-23: it was 16.1 of
    the 16.7 m reversed that day)."""
    return near_hit_reverse and mode in ("backup", "fb_backup")


def _selftest():
    """forward -> reverse brakes BRAKE_S, sits at neutral GAP_S, then reverses; and so does
    reverse -> neutral -> reverse (the XL-5HV wants two pushes on every entry)."""
    e = EscGate()
    seq = [(0.3, 0.0), (-0.3, 0.05), (-0.3, 0.2), (-0.3, 0.36), (-0.3, 0.45), (-0.3, 0.52), (0.0, 0.6), (-0.3, 0.65)]
    got = [e(t, now) for t, now in seq]
    assert [g[1] for g in got] == ["fwd", "brake", "brake", "gap", "gap", "rev", "neutral", "brake"], got
    assert got[1][0] < 0 and got[3][0] == 0.0 and got[5][0] < 0
    assert to_pwm(1.0, 0.0, 0.25)[0] == 1150, "steer + = left must be 1150 us on the real car"
    s = EscGate()                                   # from REST: still two pushes
    assert [s(-0.3, t)[1] for t in (0.0, 0.31, 0.47)] == ["brake", "gap", "rev"]
    assert reverse_allowed("fb_backup", True) and reverse_allowed("backup", True)
    assert not any(reverse_allowed(m, True) for m in ("nav_rev", "nav_retreat", "nav_plan:blocked", "nav_plan:wedged"))
    assert not reverse_allowed("backup", False), "strict forward-only must never pass a reverse"
    assert not reverse_allowed("nav_deadend", True), "the Navigator's dead-end retreat is out (2026-11-23)"
    import tempfile                                 # map memory survives a new mission's Brain
    b = Brain(goal=(3.0, 0.0), nav=True, forward_only=True)
    for _ in range(pathplan.MIN_HITS):
        b.nav.mark([1.0, 1.05], [0.5, 0.5])
    b.nav.dead_ends.append((2.0, 0.7, pathplan.DEAD_R))
    b.slam.L[10, 10] = 2.5
    with tempfile.TemporaryDirectory() as d:
        b.save_memory(f"{d}/m.npz")
        b2 = Brain(goal=(3.0, 0.0), nav=True, forward_only=True)
        b2.load_memory(f"{d}/m.npz")
    assert (b2.nav.hits == b.nav.hits).all() and b2.nav.dead_ends == b.nav.dead_ends and b2.slam.L[10, 10] == 2.5
    b2.prog, b2.best = [(0.0, 3.0), (2.0, 2.0)], 2.0
    assert b2.progress() == {"dist": 2.0, "best": 2.0, "rate": 0.5}, b2.progress()
    rg = np.random.default_rng(0)                   # blocked behind: reverse out, nothing moves -> stop, bump behind
    still = rg.integers(0, 255, (360, 640, 3), np.uint8)
    b3 = Brain(goal=(3.0, 0.0), nav=True, forward_only=True, near_hit_reverse=True)
    back = {"steer": 0.0, "throttle": -0.6, "mode": "fb_backup"}
    outs = [b3.rear_guard(dict(back), still, 0.05 * k) for k in range(25)]
    k_stop = next(k for k, o in enumerate(outs) if o["throttle"] == 0.0)
    assert 0.9 <= 0.05 * k_stop <= 1.05 and outs[-1]["mode"].endswith("rear_blocked"), (k_stop, outs[-1])
    assert len(b3.nav.bumps) == 1 and b3.nav.bumps[0][0] < 0, b3.nav.bumps
    b3.slam.pose[:] = (0.25, 0.0, 0.0)              # drove forward 0.25 m: reversing allowed again
    b3.rear_guard({"steer": 0.0, "throttle": 0.4, "mode": "nav"}, still, 2.0)
    assert b3.rear_guard(dict(back), still, 2.05)["throttle"] < 0
    b4 = Brain(goal=(3.0, 0.0), nav=True, forward_only=True, near_hit_reverse=True)
    for k in range(40):                             # a reverse that moves (pose and image) is never stopped
        b4.slam.pose[0] = -0.004 * k
        o = b4.rear_guard(dict(back), rg.integers(0, 255, (360, 640, 3), np.uint8), 0.05 * k)
        assert o["throttle"] < 0, k
    print("avoid_drive selftest: OK (ESC gate: push 0.30 s, neutral 0.15 s, then reverse; "
          "steer +1 = 1150 us = left; map memory round-trips; progress 0.5 m/s; "
          f"a still reverse stops at {0.05 * k_stop:.2f} s with a bump behind, a moving one does not)")


if __name__ == "__main__":
    _selftest() if sys.argv[1:] == ["--selftest"] else main()
