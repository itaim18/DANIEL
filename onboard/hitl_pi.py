"""HITL brain on the Pi 5: sim POV frame -> DAv2 on the Hailo-8L -> avoid.plan -> command.

    python3 hitl_pi.py --hef ~/models/dav2_s_224x392_o0b8r80.hef          # :7000
    python3 hitl_pi.py --hef ... --brain straight                          # plumbing test

Lockstep, one request per sim control step, over one TCP connection:
  request   >I len + JSON header, then >I len + JPEG (the car's POV, any size)
            header: fx fy cx cy cam_h (the SIM camera, full-frame pixels), goal_bearing,
            goal_dist, speed, viz (bool), reset (bool: new episode -> clear state),
            slam (bool: the goal is fixed in the Pi's own SLAM frame at reset and every later
            goal_bearing / goal_dist is IGNORED -- the Pi navigates on its own pose),
            nav (bool: global planning, pathplan.Navigator, on the --terrain prior map + the
            obstacles sensed this episode; reset also carries start_pose + goal_xy in yard
            coordinates; each step's GT `pose` is used only without slam)
  reply     >I len + JSON {steer, throttle, mode, free, ms{...}, ...},
            then >I len + JPEG debug view (0 bytes when viz is false)
The sim does not advance until the reply arrives, so the Hailo latency is measured
and logged but can never corrupt a result.
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import avoid  # noqa: E402
import pathplan  # noqa: E402
import slam2d  # noqa: E402
from edge_depth import Net  # noqa: E402

ROOT_AHEAD = 0.156     # car root (the sim's pose) ahead of the rear axle (slam2d's pose)


def recv_exact(s, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def recv_msg(s):
    return recv_exact(s, struct.unpack(">I", recv_exact(s, 4))[0])


def send_msg(s, payload: bytes):
    s.sendall(struct.pack(">I", len(payload)) + payload)


def crop_intrinsics(hdr, box, out_hw):
    """Sim full-frame intrinsics -> the disparity map's frame (crop, then resize)."""
    x0, y0, cw, ch = box
    sx, sy = out_hw[1] / cw, out_hw[0] / ch
    return {"fx": hdr["fx"] * sx, "fy": hdr["fy"] * sy,
            "cx": (hdr["cx"] - x0) * sx, "cy": (hdr["cy"] - y0) * sy, "cam_h": hdr["cam_h"]}


def nav_reset(hdr, T):
    """New episode with hdr["nav"]. Failed plans are saved to ~/navfail/ for probing."""
    nav = pathplan.Navigator(T, pathplan.rear_of(hdr["start_pose"][:2], hdr["start_pose"][2]), hdr["goal_xy"],
                             async_plan=bool(hdr.get("async_plan")),     # real time: plan in a worker, stop meanwhile
                             speed_loop=not hdr.get("slam"),             # no speed loop on a VO speed
                             margin=hdr.get("nav_margin", 0.10), extrude=hdr.get("nav_extrude", 0.0),
                             base=hdr.get("nav_base", 0.0))
    if hdr.get("exact_obs"):          # HITL --nav-exact: the props' TRUE circles, no sensing (isolates
        nav.bumps = [tuple(map(float, c)) for c in hdr["exact_obs"]]   # perception from the nav stack)
        nav.exact = True
    nav.debug_dir = str(Path.home() / "navfail")          # not /dev/shm: logind wipes it when ssh ends
    Path(nav.debug_dir).mkdir(exist_ok=True)
    return nav


def nav_step(nav, slam, out, hdr, ctx, cmd):
    """Global planning over avoid.plan's output `out` (updated in place), on the Pi's own
    SLAM pose, or without SLAM on the GT pose the sim sends."""
    t_n = time.perf_counter()
    pg = tuple(slam.pose) if slam is not None else pathplan.compose(
        pathplan.invert(nav.start), pathplan.rear_of(hdr["pose"][:2], hdr["pose"][2]))
    r = nav.step(pg, None if getattr(nav, "exact", False) else out["scan"], cmd[1], ctx.get("dt", 0.05))
    if r is None:                                # no path: the reactive planner drives
        out["mode"] = "fb_" + out["mode"]
    else:
        out["steer"], out["throttle"], out["mode"] = r
        ctx["state"]["last_cmd"] = (float(r[0]), float(r[1]))
    out["plans"], out["plan_ms"] = nav.plans, round(nav.plan_ms, 1)
    out["nav_ms"] = round((time.perf_counter() - t_n) * 1e3, 1)
    if out["mode"].startswith("nav_plan") and nav.path:
        out["path"] = [[round(x + ROOT_AHEAD * np.cos(th), 3), round(y + ROOT_AHEAD * np.sin(th), 3)]
                       for x, y, th in (st[:3] for st in nav.path)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hef", default=str(Path.home() / "models/dav2student_v5_288x512.hef"))
    ap.add_argument("--port", type=int, default=7000)
    ap.add_argument("--backend", choices=["hailo", "torch"], default="hailo",
                    help="torch: the student on a CUDA GPU (a laptop standing in for the Pi: the HEF is "
                         "int8 = float, chip vs float Pearson 1.000; torch vs its ONNX 1.0000), --weights")
    ap.add_argument("--weights", help="torch backend: student weights (.pt, student_depth.build())")
    ap.add_argument("--pad-ms", type=float, default=0.0,
                    help="reply no sooner than this after the request arrives: emulate the Pi's round trip "
                         "(Pi + Hailo p50 84 ms, p10-p90 70-91, p21rt) so --realtime timing matches")
    ap.add_argument("--brain", choices=["avoid", "straight"], default="avoid")
    ap.add_argument("--save", metavar="DIR", help="save each disparity (float16 npy) + header")
    ap.add_argument("--terrain", default=str(Path.home() / "models/yard_terrain.npz"),
                    help="nav: the yard's drivable-terrain distance field (kino_plan.py --export)")
    a = ap.parse_args()

    if a.backend == "torch":
        net, vdev = TorchNet(a.weights), None
    else:
        net, vdev = _hailo_net(a.hef)
    T = pathplan.load_terrain(a.terrain) if Path(a.terrain).exists() else None   # nav needs it
    try:
        _serve(a, net, T)
    finally:
        if vdev is not None:
            net.cim.shutdown()
            net.cim = net.im = None
            vdev.release()


class TorchNet:
    """edge_depth.Net's interface (pre / launch / post, box, h, w) for the student on CUDA."""

    def __init__(self, weights):
        import torch
        import student_depth as sd
        from edge_depth import crop_box
        self.torch, self.crop_box = torch, crop_box
        self.m = sd.build().cuda().eval()
        self.m.load_state_dict(torch.load(weights, map_location="cuda"))
        self.h, self.w, self.box = sd.H, sd.W, None

    def pre(self, rgb):
        if self.box is None:
            self.box = self.crop_box(*rgb.shape[:2], self.h, self.w)
        x, y, cw, ch = self.box
        img = cv2.resize(rgb[y:y + ch, x:x + cw], (self.w, self.h), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray((img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])

    def launch(self, x):
        with self.torch.no_grad():
            y = self.m(self.torch.from_numpy(x).cuda())[0].float().cpu().numpy()
        return lambda: y

    def post(self, y):
        return np.asarray(y, np.float32).reshape(self.h, self.w)


def _hailo_net(hef):
    from hailo_platform import HailoPowerMode, HailoSchedulingAlgorithm, VDevice
    p = VDevice.create_params()
    p.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    vdev = VDevice(p)
    net = Net(hef, "hailo", vdev)
    net.im.set_power_mode(HailoPowerMode.ULTRA_PERFORMANCE)
    return net, vdev


def _serve(a, net, T):
    save = Path(a.save) if a.save else None
    if save:
        save.mkdir(parents=True, exist_ok=True)
    print(f"[hitl_pi] {Path(a.weights if a.backend == 'torch' else a.hef).name} ({a.backend}) in {net.h}x{net.w} brain={a.brain} :{a.port} "
          f"terrain={'yes' if T else 'NO (nav unavailable)'}", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", a.port))
    srv.listen(1)
    try:
        while True:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[hitl_pi] sim connected {addr}", flush=True)
            ctx, k, slam, dr, goal_w, vo_only, gvo = {}, 0, None, None, None, None, None
            nav = None
            try:
                while True:
                    hdr = json.loads(recv_msg(conn))
                    jpg = recv_msg(conn)
                    t0 = time.perf_counter()
                    rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR),
                                       cv2.COLOR_BGR2RGB)
                    t1 = time.perf_counter()
                    disp = net.post(net.launch(net.pre(rgb))())
                    t2 = time.perf_counter()
                    if hdr.get("reset"):
                        ctx = {}
                        slam = dr = goal_w = nav = None
                        if hdr.get("nav"):
                            nav = nav_reset(hdr, T)
                        if hdr.get("slam"):
                            slam, dr = slam2d.Slam2D(), slam2d.Slam2D()
                            vo_only, gvo = slam2d.Slam2D(), slam2d.GroundVO()
                            b, d = hdr["goal_bearing"], hdr["goal_dist"]
                            goal_w = np.array([ROOT_AHEAD + d * np.cos(b), d * np.sin(b)])
                    ctx.update(crop_intrinsics(hdr, net.box, disp.shape))
                    ctx.update({k2: hdr[k2] for k2 in ("goal_bearing", "goal_dist", "speed", "dt") if k2 in hdr})
                    if slam is not None and hdr.get("map_nav"):
                        ctx["map_pts"] = slam2d.map_points(slam)
                    if slam is not None:                 # goal from the Pi's OWN pose
                        x, y, th = slam.pose
                        root = np.array([x + ROOT_AHEAD * np.cos(th), y + ROOT_AHEAD * np.sin(th)])
                        g = goal_w - root
                        ctx["goal_dist"] = float(np.hypot(*g))
                        ctx["goal_bearing"] = float((np.arctan2(g[1], g[0]) - th + np.pi) % (2 * np.pi) - np.pi)
                    if a.brain == "straight":
                        out = {"steer": 0.0, "throttle": 0.6, "mode": "straight", "free": 0.0}
                        viz = None
                    else:
                        cmd = ctx.get("state", {}).get("last_cmd", (0.0, 0.0))  # what the car did since
                        out = avoid.plan(disp, ctx)
                        ctx["last_steer"] = out["steer"]
                        if slam is not None and out.get("scan") is not None:
                            t_s = time.perf_counter()
                            vo = gvo.step(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), hdr, hdr["cam_h"],
                                          ctx.get("dt", 0.05))
                            odo = hdr.get("odo")                 # simulated wheel encoder [+ gyro]
                            if odo is not None:
                                dth = odo[1] if odo[1] is not None else (vo[2] if vo is not None else None)
                                out["odo"] = [round(float(odo[0]), 4), None if odo[1] is None else round(float(odo[1]), 5)]
                                out["slip"] = bool(dth is not None and slam2d.still(gvo.flow, hdr["fx"], dth, ds=odo[0]))
                                if out["slip"]:
                                    odo = [0.0, odo[1]]          # wheels turn, the image does not: slip
                            slam.step(out["scan"], cmd[0], cmd[1], ctx.get("dt", 0.05), vo=vo, odo=odo)
                            vo_only.step(out["scan"], cmd[0], cmd[1], ctx.get("dt", 0.05), vo=vo, match=False)
                            dr.predict(cmd[0], cmd[1], ctx.get("dt", 0.05))
                            out["slam_pose"] = [round(float(v), 4) for v in slam.pose]
                            out["vo_pose"] = [round(float(v), 4) for v in vo_only.pose]
                            out["vo_ok"] = vo is not None
                            out["dr_pose"] = [round(float(v), 4) for v in dr.pose]
                            out["slam_ms"] = round((time.perf_counter() - t_s) * 1e3, 2)
                        if nav is not None and out.get("scan") is not None:
                            nav_step(nav, slam, out, hdr, ctx, cmd)
                        viz = avoid.render(disp, out, ctx) if hdr.get("viz") else None
                    t3 = time.perf_counter()
                    if save is not None:
                        np.save(save / f"d{k:06d}.npy", disp.astype(np.float16))
                    reply = {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                             for kk, vv in out.items()
                             if kk in ("steer", "throttle", "mode", "free", "floor_ok", "n_obs",
                                       "alpha", "beta", "slam_pose", "dr_pose", "vo_pose", "vo_ok",
                                       "slam_ms", "plans", "plan_ms", "nav_ms", "path", "odo", "slip")}
                    reply["ms"] = {"decode": round((t1 - t0) * 1e3, 2),
                                   "hailo": round((t2 - t1) * 1e3, 2),
                                   "plan": round((t3 - t2) * 1e3, 2)}
                    if a.pad_ms:                     # a laptop answering as slowly as the Pi does
                        time.sleep(max(0.0, a.pad_ms / 1000 - (time.perf_counter() - t0)))
                    if hdr.get("disp"):              # offline dev: raw Hailo disparity, not a viz
                        reply["disp_shape"] = list(disp.shape)
                        send_msg(conn, json.dumps(reply).encode())
                        send_msg(conn, disp.astype(np.float16).tobytes())
                    else:
                        send_msg(conn, json.dumps(reply).encode())
                        send_msg(conn, cv2.imencode(".jpg", viz, [cv2.IMWRITE_JPEG_QUALITY, 75])[1]
                                 .tobytes() if viz is not None else b"")
                    k += 1
            except (ConnectionError, OSError) as e:
                print(f"[hitl_pi] sim disconnected after {k} steps ({e})", flush=True)
            finally:
                conn.close()
    finally:
        srv.close()


if __name__ == "__main__":
    main()
