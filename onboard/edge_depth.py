"""Edge Depth Lab: several monocular depth nets live on one Pi 5 (Hailo-8L + CPU).

    # two nets on the Hailo-8L at once, live on a phone at http://<pi>:8080
    python3 edge_depth.py --camera --hef fast_depth.hef --hef scdepthv3.hef --serve 8080
    # the same nets' fp32 ONNX twins and our DAv2-Small on the Pi CPU, on recorded frames
    python3 edge_depth.py --frames DIR --onnx fastdepth.onnx --onnx dav2_small_280x504_sim.onnx
    # record the T3 quality set: frames + every model's output
    python3 edge_depth.py --camera --hef ... --onnx ... --dump DIR --seconds 30
    python3 edge_depth.py --selftest

Models are recognised by file name (fast_depth / scdepth / dav2). Each one sees a
CENTER CROP of the frame at its own input aspect, never a squash, and the crop box
is recorded so depth_agreement.py can compare everything on the shared region.

Hailo nets run through ONE VDevice with the HailoRT scheduler, launched async, so
two HEFs share the NPU per frame. The camera is opened ONCE: repeated IMX500
start/stop wedged its RP2040 on 2026-09-28 (memory pi5-imx500-recovery).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


# name -> (mean, std) the zoo folds into the HEF (hailo_model_zoo cfg/networks/*.yaml
# normalization_params); the ONNX twins need it applied by hand.
ZOO_NORM = {"fast_depth": (0.0, 255.0), "scdepth": (114.75, 57.375)}


def scdepth_post(x):
    """hailo-apps postprocess/cpp/depth_estimation.cpp: logits -> metres."""
    return 1.0 / (1.0 / (1.0 + np.exp(-x)) * 10.0 + 0.009)


def family(path):
    s = Path(path).stem.lower()
    for k in ("fast_depth", "fastdepth", "scdepth", "dav2", "depth_anything", "da3", "depth_pro", "student"):
        if k in s:   # Hailo zoo names DAv2 "depth_anything_v2_vits"; the student keeps DAv2's contract
            return {"fastdepth": "fast_depth", "depth_pro": "depthpro",
                    "depth_anything": "dav2", "student": "dav2"}.get(k, k)
    raise SystemExit(f"{path}: unknown model family "
                     "(want fast_depth/scdepth/dav2/da3/depth_pro/student in the name)")


def crop_box(h, w, out_h, out_w):
    """Largest centred box of the model's aspect inside an h x w frame -> (x, y, cw, ch)."""
    a = out_w / out_h
    cw, ch = (w, round(w / a)) if w / h < a else (round(h * a), h)
    return (w - cw) // 2, (h - ch) // 2, cw, ch


class Net:
    """One depth model. run(rgb) returns (H,W) float32 in the model's own units:
    'depth' (metres, far = high) or 'disp' (inverse depth, near = high)."""

    def __init__(self, path, backend, vdevice=None):
        self.path, self.backend, self.fam = str(path), backend, family(path)
        # the FILE STEM, not the family: four DAv2 ONNX in one run all named "dav2/cpu"
        # overwrote each other's outputs and dumps (caught 2026-09-28)
        self.name = f"{Path(path).stem}/{backend}"
        self.kind = "disp" if self.fam == "dav2" else "depth"
        if backend == "hailo":
            from hailo_platform import FormatType
            # Hold the VDevice: it owns the scheduler, and the configured model keeps only a WEAK
            # pointer to it. Net(h, "hailo", VDevice(p)) let Python free it at once -> HailoRT logs
            # "Frame accumulator is supported only when scheduler is enabled" and every job times
            # out, while hailortcli runs the same HEF fine (2026-10-07).
            self.vdevice = vdevice
            self.im = vdevice.create_infer_model(self.path)
            self.im.set_batch_size(1)
            self.im.output().set_format_type(FormatType.FLOAT32)
            self.h, self.w = self.im.input().shape[:2]
            self.cim = self.im.configure()
            self.out = np.empty(self.im.output().shape, np.float32)
        elif self.fam == "depthpro":
            # Apple Depth Pro (torch): metric depth; resizes to 1536x1536 internally,
            # so it gets a 1:1 crop and nothing is squashed.
            import dataclasses
            import depth_pro
            import torch
            cfg = dataclasses.replace(depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT,
                                      checkpoint_uri=self.path)
            self.model, self.tf = depth_pro.create_model_and_transforms(
                config=cfg, device=torch.device("cpu"), precision=torch.float32)
            self.model.eval()
            self.h = self.w = 1536
        else:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            self.sess = ort.InferenceSession(self.path, so, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
            self.h, self.w = self.sess.get_inputs()[0].shape[2:]
        self.box = None

    def pre(self, rgb):
        if self.box is None:
            self.box = crop_box(*rgb.shape[:2], self.h, self.w)
        x, y, cw, ch = self.box
        img = cv2.resize(rgb[y:y + ch, x:x + cw], (self.w, self.h), interpolation=cv2.INTER_AREA)
        if self.backend == "hailo":
            return np.ascontiguousarray(img)                    # uint8 NHWC; norm is in the HEF
        if self.fam == "depthpro":
            return self.tf(img)                                 # its own ToTensor + [-1,1] norm
        if self.fam in ("dav2", "da3"):
            # RGB/255 NCHW: the thesis DAv2 contract. DA3 (export_da3_onnx.py) and the
            # *_norm DAv2 exports bake ImageNet stats into the graph, so the same input.
            # NOT dav2_preproc.preprocess_numpy: it always resizes to the thesis 280x504,
            # which broke every other export size (224x392, 2026-09-28). img is already
            # at this model's own size; with IMAGENET_NORM False the two are identical.
            return np.ascontiguousarray((img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
        m, s = ZOO_NORM[self.fam]
        return np.ascontiguousarray(((img.astype(np.float32) - m) / s).transpose(2, 0, 1)[None])

    def launch(self, x):
        """Hailo: start async, return a waiter. ONNX: run now, return the result."""
        if self.backend == "hailo":
            b = self.cim.create_bindings()
            b.input().set_buffer(x)
            b.output().set_buffer(self.out)
            self.cim.wait_for_async_ready(timeout_ms=2000)
            job = self.cim.run_async([b])
            return lambda: (job.wait(2000), self.out)[1]
        if self.fam == "depthpro":
            import torch
            with torch.no_grad():
                y = self.model.infer(x, f_px=None)["depth"].cpu().numpy()
            return lambda: y
        y = self.sess.run(None, {self.inp: x})[0]
        return lambda: y

    def post(self, y):
        y = np.asarray(y, np.float32).reshape(self.h, self.w)
        return scdepth_post(y) if self.fam == "scdepth" else y


class Background(threading.Thread):
    """Live mode only: a CPU net too slow for the camera works on the NEWEST frame in
    its own thread, so the Hailo tiles never wait for it. Its tile refreshes at its
    own rate; `times` is its full pre+infer+post per update."""

    def __init__(self, net):
        super().__init__(daemon=True)
        self.net, self.rgb, self.out, self.times = net, None, None, []
        self.ev = threading.Event()
        self.start()

    def submit(self, rgb):
        self.rgb = rgb
        self.ev.set()

    def run(self):
        while True:
            self.ev.wait()
            self.ev.clear()
            t = time.perf_counter()
            n = self.net
            self.out = n.post(n.launch(n.pre(self.rgb))())
            self.times.append((time.perf_counter() - t) * 1e3)


# ------------------------------------------------------------------ frame sources
def camera_frames(seconds):
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},          # 4:3, full sensor width
        sensor={"output_size": (2028, 1520)}, buffer_count=2))  # full-FOV binned mode
    cam.start()
    try:
        end = time.time() + seconds
        while time.time() < end:
            yield cv2.cvtColor(cam.capture_array(), cv2.COLOR_BGR2RGB)  # RGB888 is BGR in memory
    finally:
        cam.stop()
        cam.close()


def dir_frames(d, repeat):
    paths = sorted(p for p in Path(d).iterdir() if p.suffix.lower() in (".jpg", ".png"))
    if not paths:
        raise SystemExit(f"no .jpg/.png in {d}")
    for _ in range(repeat):
        for p in paths:
            yield cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ system probes
def pmic_watts(text):
    """Sum V*I over the PMIC rails vcgencmd reports. The Pi's own rails only: the
    Hailo M.2 module is fed separately and is NOT in this number."""
    amps = dict(re.findall(r"(\S+)_A current\(\d+\)=([\d.]+)A", text))
    volts = dict(re.findall(r"(\S+)_V volt\(\d+\)=([\d.]+)V", text))
    return sum(float(a) * float(volts[k]) for k, a in amps.items() if k in volts)


def cpu_busy():
    f = [int(v) for v in open("/proc/stat").readline().split()[1:]]
    return sum(f) - f[3] - f[4], sum(f)          # busy, total (idle + iowait excluded)


class Sampler(threading.Thread):
    """1 Hz: temperature, throttle flags, PMIC watts, CPU%."""

    def __init__(self):
        super().__init__(daemon=True)
        self.rows, self.stop = [], threading.Event()

    def run(self):
        t0, (b0, a0) = time.time(), cpu_busy()
        while not self.stop.wait(1.0):
            b1, a1 = cpu_busy()
            v = lambda *c: subprocess.run(["vcgencmd", *c], capture_output=True, text=True).stdout
            self.rows.append({
                "t": round(time.time() - t0, 1),
                "temp_c": int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000,
                "throttled": v("get_throttled").strip().split("=")[-1],
                "pmic_w": round(pmic_watts(v("pmic_read_adc")), 3),
                "cpu_pct": round(100.0 * (b1 - b0) / max(a1 - a0, 1), 1)})
            b0, a0 = b1, a1


# ------------------------------------------------------------------ live view
class Viewer:
    """Latest composite as MJPEG at http://<pi>:<port>/ (stdlib only)."""

    def __init__(self, port):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.jpg, self.cond = b"", threading.Condition()
        viewer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path != "/stream":
                    body = (b"<!doctype html><meta name=viewport content='width=device-width'>"
                            b"<title>Edge Depth Lab</title><body style='margin:0;background:#000'>"
                            b"<img src=/stream style='width:100%'>")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    return self.wfile.write(body)
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                try:
                    while True:
                        with viewer.cond:
                            viewer.cond.wait(2.0)
                            jpg = viewer.jpg
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d"
                                         b"\r\n\r\n" % len(jpg) + jpg + b"\r\n")
                except OSError:
                    pass

        threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever,
                         daemon=True).start()
        print(f"[view] http://0.0.0.0:{port}/", flush=True)

    def push(self, bgr):
        ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with self.cond:
            self.jpg = jpg.tobytes()
            self.cond.notify_all()


def tile(img_bgr, text, h=240):
    t = cv2.resize(img_bgr, (round(img_bgr.shape[1] * h / img_bgr.shape[0]), h))
    cv2.putText(t, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(t, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return t


def colour(out, kind):
    """Near = bright for every model, per-frame 1-99 % stretch (display only)."""
    d = out if kind == "disp" else 1.0 / np.maximum(out, 1e-3)
    lo, hi = np.percentile(d, [1, 99])
    return cv2.applyColorMap((np.clip((d - lo) / (hi - lo + 1e-8), 0, 1) * 255).astype(np.uint8),
                             cv2.COLORMAP_INFERNO)


def pct(v, q):
    return round(float(np.percentile(v, q)), 2) if v else None


# ------------------------------------------------------------------ main loop
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hef", action="append", default=[])
    ap.add_argument("--onnx", action="append", default=[])
    ap.add_argument("--torch", action="append", default=[], metavar="depth_pro.pt",
                    help="torch-only models (Apple Depth Pro); run on the CPU")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", action="store_true")
    src.add_argument("--frames", metavar="DIR")
    ap.add_argument("--repeat", type=int, default=1, help="--frames: passes over the set")
    ap.add_argument("--seconds", type=float, default=30.0, help="--camera run length")
    ap.add_argument("--serve", type=int, metavar="PORT")
    ap.add_argument("--dump", metavar="DIR", help="save frames + raw outputs for T3")
    ap.add_argument("--log", metavar="JSON")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.hef or a.onnx or a.torch) or not (a.camera or a.frames):
        ap.error("need at least one --hef/--onnx/--torch and --camera or --frames")

    vdev = None
    if a.hef:
        from hailo_platform import HailoSchedulingAlgorithm, VDevice
        p = VDevice.create_params()
        p.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        vdev = VDevice(p)
    nets = ([Net(h, "hailo", vdev) for h in a.hef] + [Net(o, "cpu") for o in a.onnx]
            + [Net(t, "cpu") for t in a.torch])
    names = [n.name for n in nets]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate model names {names}: outputs would overwrite each other")
    for n in nets:
        print(f"[net] {n.name:18s} in {n.h}x{n.w}  {n.path}", flush=True)
    # live: CPU nets run beside the loop so the Hailo tiles keep camera rate;
    # replay (--frames): every net on every frame, synchronously, so --dump is complete
    bgs = {n.name: Background(n) for n in nets if n.backend == "cpu"} if a.camera else {}
    sync = [n for n in nets if n.name not in bgs]

    view = Viewer(a.serve) if a.serve else None
    dump = Path(a.dump) if a.dump else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)
    sampler = Sampler()
    sampler.start()
    frames = camera_frames(a.seconds) if a.camera else dir_frames(a.frames, a.repeat)

    rec, last_view, t_start, i = [], 0.0, None, 0
    try:
        while True:
            t0 = time.perf_counter()
            rgb = next(frames, None)
            if rgb is None:
                break
            t_start = t_start or t0
            r = {"cap_ms": (time.perf_counter() - t0) * 1e3}
            pending = {}
            for n in sync:                                      # hailo first: async launch
                tp = time.perf_counter()
                x = n.pre(rgb)
                ti = time.perf_counter()
                pending[n.name] = (n.launch(x), ti)             # hailo: async; cpu: sync
                r[n.name] = {"pre_ms": (ti - tp) * 1e3}
            for bg in bgs.values():
                bg.submit(rgb)
            outs = {}
            for n in sync:
                wait, ti = pending[n.name]
                y = wait()
                tq = time.perf_counter()
                outs[n.name] = n.post(y)
                r[n.name]["infer_ms"] = (tq - ti) * 1e3        # launch -> result, incl. NPU sharing
                r[n.name]["post_ms"] = (time.perf_counter() - tq) * 1e3
            r["step_ms"] = (time.perf_counter() - t0) * 1e3
            rec.append({k: (round(v, 3) if isinstance(v, float) else
                            {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in r.items()})
            if dump:
                cv2.imwrite(str(dump / f"f{i:05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                for n in sync:                                  # background nets: not per-frame
                    np.save(dump / f"f{i:05d}__{n.name.replace('/', '_')}.npy",
                            outs[n.name].astype(np.float16))
            if view and time.time() - last_view > 1 / 15:       # encode at <=15 Hz
                last_view = time.time()
                fps = len(rec) / max(time.perf_counter() - t_start, 1e-6)
                tiles = [tile(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), f"camera  {fps:.1f} fps")]
                tiles += [tile(colour(outs[n.name], n.kind),
                               f"{n.name} {r[n.name]['infer_ms']:.1f} ms") for n in sync]
                tiles += [tile(colour(bg.out, bg.net.kind),
                               f"{name} {bg.times[-1] / 1e3:.1f} s/frame")
                          for name, bg in bgs.items() if bg.out is not None]
                view.push(np.hstack(tiles))
            i += 1
    finally:
        sampler.stop.set()
        wall = time.perf_counter() - (t_start or time.perf_counter())
        summ = {"frames": len(rec), "wall_s": round(wall, 2),
                "fps": round(len(rec) / wall, 2) if wall > 0 else None,
                "step_ms_p50": pct([s["step_ms"] for s in rec], 50),
                "step_ms_p99": pct([s["step_ms"] for s in rec], 99)}
        for n in nets:
            bg = bgs.get(n.name)
            v = bg.times if bg else [s[n.name]["infer_ms"] for s in rec]
            summ[n.name] = {"infer_ms_p50": pct(v, 50), "infer_ms_p99": pct(v, 99),
                            "in": [n.h, n.w], "crop_box": n.box}
            if bg:   # full pre+infer+post per update, on whichever frame was newest
                summ[n.name].update(background=True, updates=len(v),
                                    updates_per_s=round(len(v) / wall, 3) if wall > 0 else None)
        rows = sampler.rows
        if rows:
            summ["pmic_w_mean"] = round(float(np.mean([s["pmic_w"] for s in rows])), 3)
            summ["temp_c_max"] = max(s["temp_c"] for s in rows)
            summ["cpu_pct_mean"] = round(float(np.mean([s["cpu_pct"] for s in rows])), 1)
            summ["throttled_seen"] = sorted({s["throttled"] for s in rows})
        print(json.dumps(summ, indent=1), flush=True)
        if dump:
            (dump / "meta.json").write_text(json.dumps(
                {n.name: {"kind": n.kind, "in": [n.h, n.w], "crop_box": n.box, "path": n.path}
                 for n in nets}, indent=1))
        if a.log:
            Path(a.log).write_text(json.dumps({"argv": sys.argv, "summary": summ,
                                               "frames": rec, "system": rows}, indent=1))
        # Explicit, ordered teardown: leaving HailoRT objects to interpreter exit
        # segfaulted / bus-errored AFTER the summary was written (2026-09-28).
        pending = None
        for n in nets:
            if n.backend == "hailo":
                n.cim.shutdown()
                n.cim = n.im = None
        if vdev is not None:
            vdev.release()


def selftest():
    y0 = scdepth_post(np.float32(0.0))
    assert abs(y0 - 1 / (0.5 * 10 + 0.009)) < 1e-6, y0
    ys = scdepth_post(np.linspace(-6, 6, 50))
    assert np.all(np.diff(ys) < 0), "scdepth depth must fall as the logit rises"
    assert crop_box(480, 640, 224, 224) == (80, 0, 480, 480)
    assert crop_box(480, 640, 280, 504) == (0, 62, 640, 356)
    assert crop_box(480, 640, 256, 320) == (20, 0, 600, 480)
    txt = (" VDD_CORE_A current(7)=2.00000000A\n VDD_CORE_V volt(15)=0.50000000V\n"
           " 3V3_SYS_A current(1)=1.00000000A\n 3V3_SYS_V volt(9)=3.00000000V\n"
           " EXT5V_V volt(24)=5.10000000V\n")
    assert abs(pmic_watts(txt) - 4.0) < 1e-9, pmic_watts(txt)
    assert family("x/scdepthv3.hef") == "scdepth" and family("fastdepth.onnx") == "fast_depth"
    assert family("da3_base_280x504_sim.onnx") == "da3" and family("depth_pro.pt") == "depthpro"
    assert family("depth_anything_v2_vits.hef") == "dav2"
    print("edge_depth selftest: OK")


if __name__ == "__main__":
    main()
