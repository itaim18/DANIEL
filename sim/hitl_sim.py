"""HITL obstacle avoidance: Isaac Sim yard + Traxxas -> REAL Pi 5 / Hailo-8L DAv2 -> command.

    wheeled_lab/rl/run_headless.sh /tmp/hitl.log $PWD/hitl_sim.py \
        --brain pi --scenarios experiments/<date>_hitl/scenarios.json --out <dir> --tag v1

Lockstep per 50 ms sim control step: the car's POV (504x280) goes to board/pi5/hitl_pi.py
on the Pi over TCP, where DAv2 runs on the Hailo-8L and avoid.plan chooses steer and
throttle. The sim advances only when the reply arrives. --brain gt runs the SAME
avoid.plan locally on the sim's ground-truth depth: the upper bound that isolates what
DAv2-on-Hailo costs the driving.

Per scenario: jsonl log, a 1280x720 composite mp4 [chase | POV | Pi depth+decision |
map+telemetry], and PNG screenshots at start / first obstacle / end. Suite summary:
<out>/summary_<tag>.json. --gen N writes a fresh scenario suite for the car's actual
default heading and exits.
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("GT_DEPTH", "1")          # the depth AOV (the GT twin + clearance logs)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--brain", choices=["pi", "gt", "fp32", "straight", "pursuit", "oracle"], default="pi",
                help="gt: avoid on sim depth (upper bound). fp32: avoid on FLOAT DAv2-S 224x392 "
                     "run here (the HEF's own ONNX: what the Hailo gives if int8 cost nothing). "
                     "pursuit: steer at the goal, ignore obstacles -- collects approach frames "
                     "at every range for offline avoid development (with --dump-every). "
                     "oracle: hybrid A* on the drivable map + EXACT props (board/pi5/pathplan.py), "
                     "pure-pursuit followed, re-planned when off path or wedged -- is 10/10 "
                     "physically drivable with perfect knowledge?")
ap.add_argument("--host", default="10.42.0.139")
ap.add_argument("--port", type=int, default=7000, help="brain server port (a local hitl_pi.py --backend torch: 7101+)")
ap.add_argument("--scenarios", default=str(REPO / "experiments/2026-09-29_hitl/scenarios.json"))
ap.add_argument("--only", default="", help="comma-separated scenario name prefixes")
ap.add_argument("--gen", type=int, default=0, help="generate a suite for the live heading and exit")
ap.add_argument("--gen-data", type=int, default=0, metavar="N",
                help="distillation data: N POV frames at random well-scanned drivable spots with 2-6 "
                     "random props in view (never crates: scenario 07 stays a novel-object test) -> "
                     "<out>/gendata/*.jpg (+ *_depth.npy, sim GT), then exit")
ap.add_argument("--out", default=str(REPO / "experiments/2026-09-29_hitl/runs"))
ap.add_argument("--tag", default="v1")
ap.add_argument("--seed", type=int, default=0,
                help="0: the frozen start. N>0: start shifted uniformly within +-0.15 m (fixed per "
                     "seed x scenario) -- repeat episodes for mean/std (heading cannot be written)")
ap.add_argument("--max-steps", type=int, default=700, help="35 s of sim time at 20 Hz")
ap.add_argument("--no-video", action="store_true")
ap.add_argument("--slam", action="store_true",
                help="pi brain: the Pi fixes the goal in its OWN SLAM frame at reset and navigates on its "
                     "own pose (board/pi5/slam2d.py); the sim's goal bearing is sent but ignored. "
                     "Logs the SLAM and dead-reckoning poses against ground truth (ATE)")
ap.add_argument("--map-nav", action="store_true",
                help="with --slam: the Pi adds its SLAM map's obstacles to the planner's geodesic "
                     "routing field (path finding round everything seen this episode)")
ap.add_argument("--cam-h-visual", action="store_true",
                help="send the camera height above the RENDERED splat ground, measured from the depth "
                     "buffer after settling (median over floor rows), instead of root z - heightfield + "
                     "0.14. The splat sits ~1 cm below the collider: 0.144 m visual vs 0.133 m physics "
                     "(60 frames), which made ground-plane VO read ~8 % short. The real car: tape-measure it. "
                     "SUPERSEDED by --cam-h: over 50 p8full episodes it read 0.109-0.428 (07 sits on a scan hole)")
ap.add_argument("--cam-h", type=float, default=None, metavar="M",
                help="send this FIXED camera height (the tape-measured mount, as on the real car) instead of "
                     "the per-episode estimate. 0.133: offline VO scale ratio 0.999 (median, 50 p8full episodes)")
ap.add_argument("--nav", action="store_true",
                help="pi brain (gt/fp32: the same code run here): GLOBAL navigation on the Pi (pathplan.Navigator): hybrid A* on the yard's "
                     "prior terrain map + obstacles the Pi SENSES (its occupancy grid), pure pursuit, "
                     "re-plan when blocked. The sim sends the start pose + goal in yard coordinates at "
                     "reset (the real car: placed at a measured spot) and the GT pose every step, which "
                     "the Pi uses only without --slam")
ap.add_argument("--esc-gate", action="store_true",
                help="the real ESC's forward->reverse rule (avoid_drive.EscGate): brake 0.30 s + neutral "
                     "0.15 s before any reverse after forward, applied here as throttle 0")
ap.add_argument("--realtime", action="store_true",
                help="pi brain: a command takes effect round(rtt / 50 ms) control steps after its frame "
                     "(at least 1), the car keeps the old command meanwhile, and the next frame is sent "
                     "only when the Pi is free -- the real car's timing, instead of lockstep's free latency")
ap.add_argument("--pov-ss", type=int, default=1, metavar="S",
                help="render the POV camera at S x its resolution and downsample (RGB: area average; "
                     "GT depth: min-pool, the nearest surface) back to 504x280: anti-aliased frames, "
                     "thin chair legs no longer shimmer between pixels. The Pi sees the same frame size")
ap.add_argument("--goal-tol", type=float, default=0.45)
ap.add_argument("--nav-exact", action="store_true",
                help="--nav: the Navigator gets the props' EXACT circles (as the oracle) and no sensed hits -- "
                     "the real nav stack (real time, async plans, ESC gate, recovery) with perfect obstacles")
ap.add_argument("--nav-base", type=float, default=0.0, metavar="M",
                help="--nav: also map LOW hits (< NAV_H) within M m of a tall hit of the same frame: cone base "
                     "plates, feet, chair spokes (pathplan.Navigator base; 79 %% on props at 0.25 m)")
ap.add_argument("--nav-extrude", type=float, default=0.0, metavar="M",
                help="--nav: object-depth prior, each sensed hit also marked up to M m further along its ray "
                     "(pathplan.Navigator extrude): the camera maps only an obstacle's near surface")
ap.add_argument("--nav-margin", type=float, default=0.10, metavar="M",
                help="--nav: the planner's clearance margin (pathplan.Navigator margin). 0.10 was set with "
                     "the v1 student, which ranged props ~30 %% short; with the accurate v3 depth the car "
                     "brushed people at 0.10 (p15rt: 05 and 08 reached with contact in 3/3 seeds)")
ap.add_argument("--measure-ground", action="store_true",
                help="write visual_ground.json next to --scenarios and exit: the RENDERED (splat) ground "
                     "height under every prop, from a camera 1.2 m above it looking down, props parked")
ap.add_argument("--prop-z", choices=["heightfield", "visual"], default="heightfield",
                help="rest props on the collision heightfield's plane (scene v1/v2) or on the rendered "
                     "ground (visual_ground.json). Props sat a few cm INSIDE the splat's fuzzy ground: the "
                     "base showed through it and the student ranged them ~30 %% short (findings 2026-10-01)")
ap.add_argument("--odo", choices=["enc", "imu"], default=None,
                help="pi brain + --slam: give the Pi WHEEL ODOMETRY the real car does not have yet (a "
                     "hardware counterfactual). enc = rear-axle encoder (wheel joint speed x 0.06 m, slip "
                     "included), heading from Ackermann on the Pi; imu = encoder + a gyro (true yaw rate + "
                     "0.1 deg/s bias + 0.3 deg/sqrt(s) noise). Offline on 50 p12full episodes: VO 0.42 m "
                     "median ATE, enc 0.25, enc + gyro 0.08")
ap.add_argument("--stuck-s", type=float, default=10.0, metavar="S",
                help="end an episode as stuck when the best goal distance has not improved by 0.10 m in S "
                     "seconds (10: the reactive-planner era). A global plan's detour AWAY from the goal "
                     "trips 10 s while the car is still driving (p5nav/p10rt: 33 stuck episodes moved "
                     "1.1-3.7 m in their last 10 s); 35 = only the --max-steps budget ends an episode")
ap.add_argument("--dump-every", type=int, default=0,
                help="save POV jpg + GT depth (float16 npy) every N steps -> <out>/dump/<scenario>/ "
                     "for OFFLINE avoidance development against real Hailo output")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
args.enable_cameras = True
args.multi_gpu = False                                   # NuRec splat: launch-time requirement
args.kit_args = "--enable isaacsim.replicator.nurec_utils"
app = AppLauncher(args).app

import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "board/pi5"))
sys.path.insert(0, str(REPO / "wheeled_lab/env"))

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402

import avoid  # noqa: E402
import drivable_map  # noqa: E402
import hitl_scene as hs  # noqa: E402
import kino_plan  # noqa: E402
import pathplan  # noqa: E402
from edge_depth import crop_box  # noqa: E402
from avoid_drive import EscGate  # noqa: E402
from hitl_client import PiBrain  # noqa: E402

DT = 1.0 / 20.0
VW, VH = 640, 360


def build_env():
    from yard_v1_env_cfg import YardV1EnvCfg
    cfg = YardV1EnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1.0e5                      # episodes end on OUR criteria only
    for t in ("success", "stuck", "tipped"):
        if hasattr(cfg.terminations, t):
            setattr(cfg.terminations, t, None)        # no auto-reset teleports mid-scenario
    cfg.events.dome_random = None                     # fixed lighting: reproducible suites
    if args.pov_ss > 1:                               # same FOV, S x the pixels (--pov-ss)
        cfg.scene.camera.width *= args.pov_ss
        cfg.scene.camera.height *= args.pov_ss
    if not args.no_video or args.measure_ground:     # rendered every step: skipped when unused
        cfg.scene.hitl_chase = CameraCfg(
            prim_path="{ENV_REGEX_NS}/hitl_chase_cam", update_period=0.0, width=VW, height=VH,
            data_types=["rgb"] + (["distance_to_image_plane"] if args.measure_ground else []),
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 100.0)),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 2.0), convention="world"))
    hs.add_prop_pool(cfg.scene)
    return ManagerBasedRLEnv(cfg=cfg)


def car_pose(env):
    """Yard-local xy + heading. Heading via quat_apply (Euler yaw reads a constant
    +-180 on this asset -- run_suite.car_pose)."""
    r = env.scene["robot"]
    pos = (r.data.root_pos_w[0:1, :3] - env.scene.env_origins[0:1]).cpu().numpy()[0]
    fwd = quat_apply(r.data.root_quat_w[0:1], torch.tensor([[1.0, 0.0, 0.0]], device=env.device))
    return pos, float(torch.atan2(fwd[0, 1], fwd[0, 0]).item())


def pov_rgb(env):
    rgb = to_u8(env.scene["camera"].data.output["rgb"][0])
    if args.pov_ss > 1:
        rgb = cv2.resize(rgb, (rgb.shape[1] // args.pov_ss, rgb.shape[0] // args.pov_ss), interpolation=cv2.INTER_AREA)
    return rgb


def pov_depth(env):
    Z = env.scene["camera"].data.output["distance_to_image_plane"][0, ..., 0]
    if args.pov_ss > 1:
        Z = -torch.nn.functional.max_pool2d(-Z[None, None], args.pov_ss)[0, 0]
    return Z.cpu().numpy()


def to_u8(rgb):
    rgb = rgb[..., :3]
    return (rgb.clamp(0, 255) if rgb.max() > 1.5 else rgb * 255).byte().cpu().numpy()


def aim_chase(env, yaw):
    car = env.scene["robot"].data.root_pos_w[0:1, :3]
    f = torch.tensor([[math.cos(yaw), math.sin(yaw), 0.0]], device=env.device)
    eye = car - 1.8 * f + torch.tensor([[0.0, 0.0, 1.3]], device=env.device)
    env.scene["hitl_chase"].set_world_poses_from_view(eye, car + 1.2 * f)


def place_car(env, x, y, H_at):
    robot = env.scene["robot"]
    o = env.scene.env_origins[0]
    root = robot.data.default_root_state[0:1].clone()
    root[0, 0], root[0, 1] = x + float(o[0]), y + float(o[1])
    root[0, 2] = float(H_at(x, y)) + 0.02 + float(o[2])    # 2 cm clear: flush spawns sink
    root[:, 7:] = 0.0                                      # orientation NOT written (flips)
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos[0:1].clone(),
                                   robot.data.default_joint_vel[0:1].clone())
    robot.reset()


def park_cones(env):
    for name in ("cone_red", "cone_green", "cone_blue", "Cone_red", "Cone_green", "Cone_blue"):
        if name in env.scene.keys():
            o = env.scene[name]
            pose = torch.tensor([[200.0, 200.0, -30.0, 1.0, 0.0, 0.0, 0.0]], device=env.device)
            o.write_root_pose_to_sim(pose)


def gt_disparity(env, box, out_hw, intr, cam_h):
    Z = pov_depth(env)
    Z = np.where(np.isfinite(Z) & (Z > 0.05), Z, 1e3)
    # The heightfield collider is invisible and the splat's depth is patchy near the car
    # (bottom row read ~6 m, 2026-09-29), so avoid's floor fit failed. Take the min with a
    # flat ground plane under the level camera: Z = fy*h/(v-cy). Elsewhere the splat DOES
    # write depth, with speckle and floater blobs just above the ground: "GT" depth has
    # phantom obstacles of a few cm (n2, 2026-09-30; pathplan maps only taller hits).
    # ponytail: flat local ground; ray-march the heightfield if slopes start to matter.
    v = np.arange(Z.shape[0], dtype=np.float32)[:, None] - intr["cy"]
    Zf = np.where(v > 0.5, intr["fy"] * cam_h / np.maximum(v, 0.5), 1e3)
    Z = np.minimum(Z, np.broadcast_to(Zf, Z.shape))
    x0, y0, cw, ch = box
    return cv2.resize((1.0 / Z[y0:y0 + ch, x0:x0 + cw]).astype(np.float32), out_hw[::-1],
                      interpolation=cv2.INTER_AREA)


_FP32 = None


def fp32_disparity(pov, box):
    """Float DAv2-S 224x392 (the ONNX the HEF was compiled from) on the Pi's exact crop."""
    global _FP32
    import onnxruntime as ort
    if _FP32 is None:
        _FP32 = ort.InferenceSession(str(Path.home() / "thesis-depth-robotics/onnx_dav2/dav2_s_224x392_raw.onnx"),
                                     providers=["CPUExecutionProvider"])
    x0, y0, cw, ch = box
    x = cv2.resize(pov[y0:y0 + ch, x0:x0 + cw], (392, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return _FP32.run(None, {_FP32.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0].squeeze()


def minimap(d, sc, traj, placed, man, size=(VW, VH), plan_xy=None):
    """Bird's-eye of the scenario window: drivable map, props, path, goal."""
    sx, sy = sc["start"]
    gx, gy = sc["goal"]
    cxm, cym = (sx + gx) / 2, (sy + gy) / 2
    span = max(abs(gx - sx), abs(gy - sy)) + 3.0
    W, Hh = size
    scale = min(W, Hh) / span
    img = np.full((Hh, W, 3), 30, np.uint8)

    def px(x, y):
        return int(W / 2 + (x - cxm) * scale), int(Hh / 2 - (y - cym) * scale)

    ii = np.where((np.abs(d["ux"] - cxm) < span / 2 + 1))[0]
    jj = np.where((np.abs(d["uy"] - cym) < span / 2 + 1))[0]
    for i in ii[::2]:
        for j in jj[::2]:
            if d["mask"][i, j]:
                cv2.circle(img, px(d["ux"][i], d["uy"][j]), 1, (60, 90, 60), -1)
    for p in placed:
        cv2.circle(img, px(p["x"], p["y"]), max(2, int(man[p["type"]]["radius"] * scale)),
                   (40, 140, 240), -1)
        cv2.putText(img, p["type"][:6], px(p["x"] + 0.2, p["y"]), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (200, 200, 200), 1)
    cv2.circle(img, px(gx, gy), max(3, int(0.45 * scale)), (80, 220, 80), 2)
    for a, b in zip((plan_xy or [])[:-1], (plan_xy or [])[1:]):
        cv2.line(img, px(*a), px(*b), (255, 60, 255), 1)
    for a, b in zip(traj[:-1], traj[1:]):
        cv2.line(img, px(*a), px(*b), (255, 255, 255), 2)
    if traj:
        cv2.circle(img, px(*traj[-1]), max(3, int(hs.CAR_R * scale)), (0, 200, 255), -1)
    return img


def compose(chase, pov, viz, mm, lines):
    top = np.hstack([cv2.cvtColor(chase, cv2.COLOR_RGB2BGR),
                     cv2.resize(cv2.cvtColor(pov, cv2.COLOR_RGB2BGR), (VW, VH))])
    vz = cv2.resize(viz, (VW, VH)) if viz is not None else np.zeros((VH, VW, 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(mm, t, (8, 18 + 17 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1)
    return np.vstack([top, np.hstack([vz, mm])])


def open_video(path, fps=20):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
                             "bgr24", "-s", f"{2 * VW}x{2 * VH}", "-r", str(fps), "-i", "-",
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
                             "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)


ROOT_AHEAD = 0.156          # board/pi5/hitl_pi.py: sim root ahead of slam2d's rear axle


def _slam_pair(ev, r, gt_rel):
    """(ground truth, SLAM, dead-reckoning) root positions in the start frame."""
    if "slam_pose" not in r:
        return
    def root(p):
        return (p[0] + ROOT_AHEAD * math.cos(p[2]) - ROOT_AHEAD, p[1] + ROOT_AHEAD * math.sin(p[2]))
    ev.setdefault("slam_pairs", []).append((gt_rel, root(r["slam_pose"]), root(r["dr_pose"]),
                                            root(r.get("vo_pose", r["dr_pose"]))))
    ev["vo_ok_steps"] = ev.get("vo_ok_steps", 0) + int(bool(r.get("vo_ok")))


def run_scenario(env, sc, brain, d, H_at, man, out, intr, cam_h):
    sx, sy = sc["start"]
    if args.seed:
        j = np.random.default_rng([args.seed, int(sc["name"][:2])]).uniform(-0.15, 0.15, 2)
        sx, sy = sx + float(j[0]), sy + float(j[1])
    place_car(env, sx, sy, H_at)
    placed = hs.place(env, sc, H_at)
    park_cones(env)
    act = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    for _ in range(8):                                    # settle on the terrain
        env.step(act)
    # Camera height above the LOCAL ground, from the ROBOT ROOT + the camera's 0.14 m mount
    # (arena_env_cfg CameraCfg offset). NOT camera.data.pos_w: it is not refreshed unless
    # update_latest_camera_pose is set, so it kept the spawn pose and sent the Pi
    # cam_h = -0.19 m -- every metric depth flipped sign and the car backed up at once.
    rp, _ = car_pose(env)
    cam_h = float(rp[2] - H_at(rp[0], rp[1]) + 0.14)
    if args.cam_h_visual:
        Z = pov_depth(env)
        cy_, fy_ = intr["cy"], intr["fy"]
        rows = np.arange(int(cy_) + 25, Z.shape[0] - 2)
        band = Z[rows][:, Z.shape[1] // 5: -Z.shape[1] // 5]
        med = np.array([np.median(r[np.isfinite(r) & (r < 20)]) if np.isfinite(r).any() else np.nan for r in band])
        ok = np.isfinite(med)
        if ok.sum() > 20:
            cam_h = float(np.median(med[ok] * (rows[ok] - cy_) / fy_))
    if args.cam_h:
        cam_h = args.cam_h
    gx, gy = sc["goal"]
    log = open(out / f"{sc['name']}.jsonl", "w")
    ff = None if args.no_video else open_video(out / f"{sc['name']}.mp4")
    traj, status, ctx_gt = [], "timeout", {}
    box = crop_box(280, 504, 224, 392)
    ev = {"steps": 0, "collisions": 0, "min_clear": 9.0, "hailo_ms": [], "rtt_ms": [],
          "path_m": 0.0, "backup_steps": 0}
    last_xy, stuck_ref, shots, plan_xy = None, None, set(), None
    rt = {"pending": None, "due": 0, "sent": 0, "applied": {"steer": 0.0, "throttle": 0.0, "mode": "wait",
                                                             "free": 0.0}, "viz": None}
    esc = EscGate()
    robot = env.scene["robot"]
    rear_ids = robot.find_joints("wheel_r.*")[0]
    odo, yaw_prev, g_rng = [0.0, 0.0], None, np.random.default_rng([args.seed, int(sc["name"][:2]), 7])
    for t in range(args.max_steps):
        pos, yaw = car_pose(env)
        if args.odo and yaw_prev is not None:                 # since the last step (simulated sensors)
            odo[0] += float(robot.data.joint_vel[0, rear_ids].mean()) * 0.06 * DT
            odo[1] += ((yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi + math.radians(0.1) * DT
                       + math.radians(0.3) * math.sqrt(DT) * float(g_rng.standard_normal()))
        yaw_prev = yaw
        xy = (float(pos[0]), float(pos[1]))
        traj.append(xy)
        if last_xy is not None:
            ev["path_m"] += math.hypot(xy[0] - last_xy[0], xy[1] - last_xy[1])
        last_xy = xy
        dist = math.hypot(gx - xy[0], gy - xy[1])
        bearing = (math.atan2(gy - xy[1], gx - xy[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
        clear = min([hs.prop_clearance(p, man[p["type"]], xy) for p in placed] or [9.0])
        ev["min_clear"] = min(ev["min_clear"], clear)
        if clear < 0.0:
            ev["collisions"] += 1
        pov = pov_rgb(env)
        hdr = dict(intr, cam_h=cam_h, goal_bearing=bearing, goal_dist=dist, dt=DT, viz=not args.no_video,
                   reset=(t == 0), slam=args.slam, map_nav=args.map_nav, nav=args.nav,
                   pose=[xy[0], xy[1], yaw])
        if t == 0 and args.nav:
            hdr.update(start_pose=[xy[0], xy[1], yaw], goal_xy=[gx, gy], async_plan=args.realtime,
                       nav_margin=args.nav_margin, nav_extrude=args.nav_extrude, nav_base=args.nav_base)
            if args.nav_exact:
                hdr["exact_obs"] = np.asarray(kino_plan.prop_circles(placed, man), float).reshape(-1, 3).tolist()
        if args.odo:                                          # motion since the last frame the Pi got
            hdr["odo"] = [odo[0], odo[1] if args.odo == "imu" else None]
        if t == 0:
            origin = (xy, yaw)
        c0, s0_ = math.cos(-origin[1]), math.sin(-origin[1])
        gt_rel = (c0 * (xy[0] - origin[0][0]) - s0_ * (xy[1] - origin[0][1]),
                  s0_ * (xy[0] - origin[0][0]) + c0 * (xy[1] - origin[0][1]))
        if args.dump_every and t % args.dump_every == 0:
            dd = out / "dump" / sc["name"]
            dd.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dd / f"f{t:04d}.jpg"), cv2.cvtColor(pov, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            np.save(dd / f"f{t:04d}_depth.npy", pov_depth(env).astype(np.float16))
            if t == 0:
                (dd / "meta.json").write_text(json.dumps(dict(hdr, scenario=sc["name"])))
        if args.brain == "pi" and args.realtime:
            if rt["pending"] is not None and t >= rt["due"]:
                rt["applied"], rt["pending"] = rt["pending"], None
            if rt["pending"] is None:                    # the Pi is free: this frame goes now
                hdr["dt"] = max(t - rt["sent"], 1) * DT
                r, rt["viz"] = brain.step(pov, hdr)
                odo = [0.0, 0.0]
                _slam_pair(ev, r, gt_rel)
                ev["hailo_ms"].append(r["ms"]["hailo"])
                ev["rtt_ms"].append(r["rtt_ms"])
                rt["pending"], rt["sent"] = r, t
                rt["due"] = t + max(1, round(r["rtt_ms"] / (DT * 1000)))
            rep, viz = dict(rt["applied"], ms=(rt["pending"] or rt["applied"]).get("ms", {"hailo": 0}),
                            rtt_ms=(rt["pending"] or rt["applied"]).get("rtt_ms", 0)), rt["viz"]
        elif args.brain == "pi":
            rep, viz = brain.step(pov, hdr)
            odo = [0.0, 0.0]
            _slam_pair(ev, rep, gt_rel)
            ev["hailo_ms"].append(rep["ms"]["hailo"])
            ev["rtt_ms"].append(rep["rtt_ms"])
        elif args.brain in ("gt", "fp32"):
            disp = gt_disparity(env, box, (224, 392), intr, cam_h) if args.brain == "gt" \
                else fp32_disparity(pov, box)
            if t == 0:
                ctx_gt = {}
            import hitl_pi
            ctx_gt.update(hitl_pi.crop_intrinsics(hdr, box, (224, 392)))
            ctx_gt.update(goal_bearing=bearing, goal_dist=dist, dt=DT)
            cmd_gt = ctx_gt.get("state", {}).get("last_cmd", (0.0, 0.0))
            o = avoid.plan(disp, ctx_gt)
            if args.nav:                                 # the Pi's nav code, on sim depth here
                if t == 0:
                    nav_gt = hitl_pi.nav_reset(hdr, d["T"])
                hitl_pi.nav_step(nav_gt, None, o, hdr, ctx_gt, cmd_gt)
            ctx_gt["last_steer"] = o["steer"]
            viz = avoid.render(disp, o, ctx_gt) if not args.no_video else None
            rep = {k: o[k] for k in ("steer", "throttle", "mode", "free", "plans", "plan_ms", "path") if k in o}
        elif args.brain == "oracle":
            rr = kino_plan.rear(xy, yaw)
            if t == 0:
                orc = {"circ": kino_plan.prop_circles(placed, man), "fol": None, "plans": 0, "hist": []}
            orc["hist"].append(xy)
            moved = math.hypot(xy[0] - orc["hist"][-40][0], xy[1] - orc["hist"][-40][1]) if len(orc["hist"]) >= 40 else 9.0
            steer, thr, done = orc["fol"].step(rr) if orc["fol"] else (0.0, 0.0, True)
            if done or orc["fol"].off_path(rr) or moved < 0.05:
                # the Navigator's own cascade (margin 0.10, a relaxed retry free round the start, hard
                # circles for cells already inside the margin): raw hybrid A* at 0.10 / 0.04 sat
                # still for 500-700 re-plans when a start jitter or a stop left the car inside a
                # margin (o12: 07 x2, 03 x1) -- the oracle's limit, not the yard's
                path, _ = pathplan.plan_cascade(d["T"], np.asarray(orc["circ"], float).reshape(-1, 3), rr,
                                                (gx, gy), None, 0.10, 15000)
                orc["plans"] += 1
                orc["hist"] = []
                orc["fol"] = pathplan.Follower(path) if path else None
                plan_xy = [(x + pathplan.ROOT_AHEAD * math.cos(a), y + pathplan.ROOT_AHEAD * math.sin(a))
                           for x, y, a in (st[:3] for st in path)] if path else None
                steer, thr, _ = orc["fol"].step(rr) if path else (0.0, 0.0, True)
            ev["plans"] = orc["plans"]
            rep, viz = {"steer": steer, "throttle": thr, "free": 0.0,
                        "mode": "noplan" if orc["fol"] is None else ("path" if thr >= 0 else "path_rev")}, None
        elif args.brain == "pursuit":
            rep, viz = {"steer": float(np.clip(bearing / 0.35, -1, 1)), "throttle": 0.5,
                        "mode": "pursuit", "free": 0.0}, None
        else:
            rep, viz = {"steer": 0.0, "throttle": 0.6, "mode": "straight", "free": 0.0}, None
        if rep.get("mode") == "backup":
            ev["backup_steps"] += 1
        if rep.get("path"):
            plan_xy = rep["path"]
        if "plans" in rep:
            ev["plans"] = rep["plans"]
            if str(rep.get("mode", "")).startswith("nav_plan"):
                ev.setdefault("plan_ms", []).append(rep["plan_ms"])
        if str(rep.get("mode", "")).startswith("fb_"):
            ev["fallback_steps"] = ev.get("fallback_steps", 0) + 1
        act[0, 0], act[0, 1] = float(rep["steer"]), float(rep["throttle"])
        if args.esc_gate:
            thr_e, ph = esc(float(rep["throttle"]), t * DT)
            act[0, 1] = 0.0 if ph in ("brake", "gap") else thr_e
        log.write(json.dumps({"t": t, "x": round(xy[0], 3), "y": round(xy[1], 3), "yaw": round(yaw, 3),
                              "goal_dist": round(dist, 3), "bearing": round(bearing, 3),
                              "clear": round(clear, 3), **{k: rep[k] for k in rep if k != "ms"},
                              "ms": rep.get("ms")}) + "\n")
        if ff is not None:
            aim_chase(env, yaw)
            chase = to_u8(env.scene["hitl_chase"].data.output["rgb"][0])
            mm = minimap(d, sc, traj, placed, man, plan_xy=plan_xy)
            lines = [f"{sc['name']}  [{args.brain} {args.tag}]  t={t * DT:5.1f}s",
                     f"steer {rep['steer']:+.2f}  throttle {rep['throttle']:+.2f}  {rep.get('mode', '')}",
                     f"goal {dist:4.2f} m  bearing {math.degrees(bearing):+5.0f} deg",
                     f"clearance {clear:+.2f} m  collisions {ev['collisions']}",
                     (f"Pi: hailo {rep['ms']['hailo']:.0f} ms  rtt {rep['rtt_ms']:.0f} ms"
                      if args.brain == "pi" else {"gt": "brain: sim GT depth (upper bound)",
                                                  "fp32": "brain: float DAv2-S on the laptop"}.get(args.brain, args.brain))]
            frame = compose(chase, pov, viz, mm, lines)
            ff.stdin.write(frame.tobytes())
            key = "start" if t == 5 else ("close" if clear < 0.25 and "close" not in shots else None)
            if key and key not in shots:
                cv2.imwrite(str(out / f"{sc['name']}_{key}.png"), frame)
                shots.add(key)
        env.step(act)
        # termination on OUR criteria
        if dist < args.goal_tol:
            status = "success"
            break
        if float(env.scene["robot"].data.projected_gravity_b[0, 2]) > -0.5:
            status = "tipped"
            break
        # stuck = no PROGRESS: the best goal distance has not improved by 0.10 m in
        # --stuck-s. Until v4 it was "moved < 0.10 m in 3 s", which also killed a slow
        # 3-point turn (reverse / forward legs cancel out in displacement).
        if stuck_ref is None or dist < stuck_ref[0] - 0.10:
            stuck_ref = (dist, t)
        elif t - stuck_ref[1] > args.stuck_s / DT:
            status = "stuck"
            break
    ev.update(status=status, steps=t + 1, time_s=round((t + 1) * DT, 2), cam_h=round(cam_h, 3),
              final_goal_dist=round(dist, 3), min_clear=round(ev["min_clear"], 3),
              path_m=round(ev["path_m"], 2),
              hailo_ms_p50=round(float(np.median(ev["hailo_ms"])), 1) if ev["hailo_ms"] else None,
              rtt_ms_p50=round(float(np.median(ev["rtt_ms"])), 1) if ev["rtt_ms"] else None)
    ev.pop("hailo_ms"), ev.pop("rtt_ms")
    pm = ev.pop("plan_ms", None)
    if pm:
        ev.update(plan_ms_p50=round(float(np.median(pm)), 1), plan_ms_max=round(float(np.max(pm)), 1))
    pairs = ev.pop("slam_pairs", None)
    if pairs:
        g, sl, dr, vo = (np.array([p[i] for p in pairs]) for i in range(4))
        ev["slam_ate_m"] = round(float(np.sqrt(((sl - g) ** 2).sum(1).mean())), 3)
        ev["dr_ate_m"] = round(float(np.sqrt(((dr - g) ** 2).sum(1).mean())), 3)
        ev["vo_ate_m"] = round(float(np.sqrt(((vo - g) ** 2).sum(1).mean())), 3)
        ev["vo_ok_frac"] = round(ev.pop("vo_ok_steps", 0) / len(pairs), 3)
        ev["slam_final_err_m"] = round(float(np.hypot(*(sl[-1] - g[-1]))), 3)
        ev["dr_final_err_m"] = round(float(np.hypot(*(dr[-1] - g[-1]))), 3)
        (out / f"{sc['name']}_slam.json").write_text(json.dumps({"gt": g.tolist(), "slam": sl.tolist(),
                                                                  "dr": dr.tolist(), "vo": vo.tolist()}))
    if ff is not None:
        cv2.imwrite(str(out / f"{sc['name']}_end.png"), frame)
        ff.stdin.close()
        ff.wait()
    log.close()
    return ev


def gen_data(env, d, H_at, man):
    """Random POV frames for distilling the depth student (board/pi5/student_depth.py)."""
    import recon_quality
    q = recon_quality.maps()["score"]
    rng = np.random.default_rng(args.seed)
    cells = np.argwhere(d["mask"] & (d["clearance"] >= 0.5) & (q >= 0.65))
    types = [t for t in hs.POOL if t != "crate"]
    out = Path(args.out) / "gendata"
    out.mkdir(parents=True, exist_ok=True)
    act = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    for k in range(args.gen_data):
        i, j = cells[rng.integers(len(cells))]
        x, y = float(d["ux"][i]), float(d["uy"][j])
        place_car(env, x, y, H_at)
        _, yaw = car_pose(env)
        used, props = {t: 0 for t in hs.POOL}, []
        for _ in range(int(rng.integers(2, 7))):
            t = types[rng.integers(len(types))]
            if used[t] >= hs.POOL[t]:
                continue
            r, b = rng.uniform(0.6, 4.0), rng.uniform(-0.6, 0.6)
            px, py = x + r * math.cos(yaw + b), y + r * math.sin(yaw + b)
            if all(math.hypot(px - o["x"], py - o["y"]) > 0.8 for o in props):
                used[t] += 1
                props.append({"type": t, "x": px, "y": py, "yaw": float(rng.uniform(-math.pi, math.pi))})
        hs.place(env, {"props": props}, H_at)
        for _ in range(4):
            env.step(act)
        cv2.imwrite(str(out / f"g{args.seed:02d}_{k:05d}.jpg"),
                    cv2.cvtColor(pov_rgb(env), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        np.save(out / f"g{args.seed:02d}_{k:05d}_depth.npy", pov_depth(env).astype(np.float16))  # GT: student v4
        if k % 100 == 0:
            print(f"[gendata] {k}/{args.gen_data}", flush=True)
    return 0


def measure_ground(env, scs, H_at, man, path):
    """Rendered ground height under every prop (props parked): the chase camera 1.2 m above the
    prop's spot, looking down; the median surface height inside its footprint radius."""
    cam, o = env.scene["hitl_chase"], env.scene.env_origins[0]
    act = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    hs.place(env, {"props": []}, H_at)
    res = {}
    for s in scs:
        zs = []
        for p in s["props"]:
            zh = float(H_at(p["x"], p["y"]))
            eye = torch.tensor([[p["x"] + float(o[0]), p["y"] + float(o[1]), zh + 1.2 + float(o[2])]], device=env.device)
            cam.set_world_poses_from_view(eye, eye + torch.tensor([[0.01, 0.0, -1.0]], device=env.device))
            for _ in range(6):
                env.step(act)
            Z = cam.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
            f = float(cam.data.intrinsic_matrices[0, 0, 0])
            v, u = np.mgrid[:Z.shape[0], :Z.shape[1]]
            sel = (np.hypot(u - Z.shape[1] / 2, v - Z.shape[0] / 2) * 1.2 / f < man[p["type"]]["radius"]) & np.isfinite(Z) & (Z < 3)
            zg = float(eye[0, 2]) - float(o[2]) - Z[sel]
            zs.append(round(float(np.median(zg)), 4))
            print(f"[ground] {s['name']:20s} {p['type']:14s} heightfield {zh:+.3f}  rendered {zs[-1]:+.3f} "
                  f"(p10 {np.percentile(zg, 10):+.3f} p90 {np.percentile(zg, 90):+.3f})  diff {zs[-1] - zh:+.3f} m", flush=True)
        res[s["name"]] = zs
    path.write_text(json.dumps(res, indent=1))
    print(f"[ground] -> {path}", flush=True)
    return 0


def main():
    d = drivable_map.build()
    man = hs.manifest()

    def H_at(x, y):
        return d["H"][int(np.argmin(np.abs(d["ux"] - x))), int(np.argmin(np.abs(d["uy"] - y)))]

    if args.brain == "oracle" or args.nav:
        d["T"] = kino_plan.terrain(d)
    env = build_env()
    env.reset()
    park_cones(env)
    _, yaw0 = car_pose(env)
    cam = env.scene["camera"]
    K = cam.data.intrinsic_matrices[0:1].cpu().numpy()[0]
    cpos = (cam.data.pos_w[0:1] - env.scene.env_origins[0:1]).cpu().numpy()[0]
    rpos, _ = car_pose(env)
    cam_h = float(cpos[2] - H_at(rpos[0], rpos[1]))
    S = args.pov_ss                                   # intrinsics of the DOWNSAMPLED frame
    intr = {"fx": float(K[0, 0]) / S, "fy": float(K[1, 1]) / S,
            "cx": float(K[0, 2]) / S, "cy": float(K[1, 2]) / S}   # Isaac: cx = W/2 (edge convention)
    print(f"[hitl] heading yaw0={math.degrees(yaw0):.1f} deg  cam_h={cam_h:.3f} m  K={intr}", flush=True)

    scen_path = Path(args.scenarios)
    if args.gen_data:
        return gen_data(env, d, H_at, man)
    if args.gen:
        import recon_quality
        rq = recon_quality.maps()
        assert rq["score"].shape == d["mask"].shape
        scs = hs.generate(d, yaw0, man, q=rq["score"], q_min=0.65)[:args.gen]
        scen_path.parent.mkdir(parents=True, exist_ok=True)
        scen_path.write_text(json.dumps({"yaw0": yaw0, "cam_h": cam_h, "intr": intr,
                                         "scenarios": scs}, indent=1))
        print(f"[hitl] wrote {len(scs)} scenarios -> {scen_path}", flush=True)
        return 0
    suite = json.loads(scen_path.read_text())
    if abs(suite["yaw0"] - yaw0) > 0.05:
        print(f"[hitl] WARNING heading {yaw0:.3f} != suite {suite['yaw0']:.3f}", flush=True)
    scs = suite["scenarios"]
    if args.only:
        pre = args.only.split(",")
        scs = [s for s in scs if any(s["name"].startswith(p) for p in pre)]
    vg_path = scen_path.with_name("visual_ground.json")
    if args.measure_ground:
        return measure_ground(env, scs, H_at, man, vg_path)
    if args.prop_z == "visual":
        vg = json.loads(vg_path.read_text())
        for s in scs:
            for p, z in zip(s["props"], vg[s["name"]]):
                p["z_vis"] = z
    out = Path(args.out) / f"{args.tag}_{args.brain}"
    out.mkdir(parents=True, exist_ok=True)
    brain = PiBrain(args.host, args.port) if args.brain == "pi" else None
    results = {}
    for sc in scs:
        t0 = time.time()
        ev = run_scenario(env, sc, brain, d, H_at, man, out, intr, cam_h)
        ev["wall_s"] = round(time.time() - t0, 1)
        results[sc["name"]] = ev
        print(f"[hitl] {sc['name']:22s} {ev['status']:8s} t={ev['time_s']:5.1f}s coll={ev['collisions']} "
              f"minclr={ev['min_clear']:+.2f} path={ev['path_m']:.1f}m goal_d={ev['final_goal_dist']:.2f} "
              f"hailo={ev.get('hailo_ms_p50')} rtt={ev.get('rtt_ms_p50')} wall={ev['wall_s']}s"
              + (f" slamATE={ev['slam_ate_m']} voATE={ev['vo_ate_m']} drATE={ev['dr_ate_m']} vo_ok={ev['vo_ok_frac']}"
                 if "slam_ate_m" in ev else "")
              + (f" plans={ev['plans']} plan_ms={ev.get('plan_ms_p50')}/{ev.get('plan_ms_max')} fb={ev.get('fallback_steps', 0)}"
                 if "plans" in ev else ""), flush=True)
    n = len(results)
    ok = sum(1 for e in results.values() if e["status"] == "success" and e["collisions"] == 0)
    summ = {"tag": args.tag, "brain": args.brain, "realtime": args.realtime, "slam": args.slam,
            "map_nav": args.map_nav, "nav": args.nav, "esc_gate": args.esc_gate, "seed": args.seed, "cam_h_fixed": args.cam_h, "pov_ss": args.pov_ss, "stuck_s": args.stuck_s, "odo": args.odo, "prop_z": args.prop_z, "nav_margin": args.nav_margin, "nav_extrude": args.nav_extrude, "nav_base": args.nav_base, "nav_exact": args.nav_exact,
            "n": n, "clean_success": ok,
            "success": sum(1 for e in results.values() if e["status"] == "success"),
            "collision_eps": sum(1 for e in results.values() if e["collisions"] > 0),
            "cam_h": cam_h, "results": results}
    (Path(args.out) / f"summary_{args.tag}_{args.brain}.json").write_text(json.dumps(summ, indent=1))
    print(f"[hitl] SUITE {args.tag}/{args.brain}: clean success {ok}/{n}, success {summ['success']}/{n}, "
          f"episodes with collision {summ['collision_eps']}/{n}", flush=True)
    if brain:
        brain.close()
    env.close()
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close()
    sys.exit(code)
