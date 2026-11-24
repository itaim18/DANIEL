"""Compile Depth-Anything-V2 for the Hailo-8L with the Hailo Dataflow Compiler (x86).

    ~/hailo-dfc-venv/bin/python onboard/compile_dav2_hailo.py \
        --onnx ~/thesis-depth-robotics/onnx_dav2/dav2_s_224x224_raw.onnx --name dav2_s_224

Stages, each saved as a .har beside the output so a failure resumes, not restarts:
  parse    ONNX -> Hailo graph
  optimize int8 quantisation, calibrated on calib_domain_v1 (the frames the Q6A's w8a8
           DAv2 was calibrated on, AI Hub job jgn7e3dqp) so the two chips compare
  eval     emulator: int8 vs the float graph on HELD-OUT domain frames -- the int8 cost,
           measured before any Pi time (disparity, scale+shift aligned; corr + AbsRel)
  compile  -> .hef for hailo8l

Model script: onboard/dav2_hailo.alls = hailo_model_zoo master's own
cfg/alls/generic/depth_anything_v2_vits.alls (published for 224x224 only: 32.4 FPS b1 /
91.4 b8 on x86; 22.2 / 60.6 measured on this Pi 5, PCIe Gen3 x1), with NORM filled in:
  --norm imagenet  the model as trained (Hailo's choice)
  --norm raw       the THESIS contract, x/255 only (dav2_preproc KNOWN DEVIATION)
The ONNX is the bare net (export_dav2_onnx.py --raw); normalisation lives in the HEF.
Calibration: 240 robot-POV frames (calib_pov_v1) + calib_domain_v1 minus the held-out.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from edge_depth import crop_box  # noqa: E402  (the crop the Pi applies)

DEPS = Path.home() / "thesis-depth-robotics"
DOMAIN, POV = DEPS / "calib_domain_v1", DEPS / "calib_pov_v1"
OUT = DEPS / "hailo_dav2"
NORMS = {"imagenet": "[123.675, 116.28, 103.53], [58.395, 57.12, 57.375]",
         "raw": "[0.0, 0.0, 0.0], [255.0, 255.0, 255.0]"}


def prep(p, h, w):
    """One image file at model size, uint8 RGB HWC, the Pi's centre crop."""
    rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    x0, y0, cw, ch = crop_box(*rgb.shape[:2], h, w)
    return cv2.resize(rgb[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_AREA)


def frames(h, w, d):
    """A dir's frames (png/jpg) at model size, uint8-range float32 NHWC."""
    return np.stack([prep(p, h, w) for p in sorted(d.iterdir())
                     if p.suffix in (".png", ".jpg")]).astype(np.float32)


def agreement(q, f):
    """int8 vs float disparity: scale+shift aligned (lstsq), corr and AbsRel."""
    q, f = q.ravel().astype(np.float64), f.ravel().astype(np.float64)
    m = f > np.percentile(f, 5)
    A = np.stack([q[m], np.ones(m.sum())], 1)
    (s, t), *_ = np.linalg.lstsq(A, f[m], rcond=None)
    a = s * q[m] + t
    return {"corr": float(np.corrcoef(q, f)[0, 1]),
            "abs_rel": float(np.mean(np.abs(a - f[m]) / np.abs(f[m])))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--hw", default="hailo8l")
    ap.add_argument("--norm", choices=NORMS, default="imagenet")
    ap.add_argument("--alls", type=Path, default=HERE / "dav2_hailo.alls",
                    help="model script (NORM is substituted); student_hailo.alls for the CNN student")
    ap.add_argument("--holdout", type=int, default=13, help="last N domain frames: eval only")
    ap.add_argument("--alloc", action="append", default=[], metavar="CMD",
                    help="allocator-only model-script line applied just before compile, e.g. "
                         "'performance_param(compiler_optimization_level=max)'. Reuses the "
                         "quantised .har, so a layout variant costs no re-optimisation")
    ap.add_argument("--suffix", default="", help="HEF name suffix for a layout variant")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="FEASIBILITY/SPEED probe: 16 calib frames, no finetune. Allocation "
                         "and FPS do not depend on quantisation quality, so a resolution sweep "
                         "costs minutes; only the winner gets the ~1 h finetune. Never quote "
                         "a --fast HEF's QUALITY.")
    ap.add_argument("--extra", action="append", default=[], metavar="CMD",
                    help="optimisation-time model-script line appended to the recipe, e.g. "
                         "'quantization_param({NAME/ew_add*}, precision_mode=a16_w16)' "
                         "(NAME = the network name). Changes the model: needs a new --tag")
    ap.add_argument("--tag", default="", help="network-name suffix for a recipe variant")
    ap.add_argument("--no-finetune", action="store_true",
                    help="full calibration set, finetune disabled: the finetuned 224x392 needed "
                         "far longer chip layout than the un-finetuned one (2026-09-28)")
    ap.add_argument("--calib-dir", action="extend", nargs="+", default=[], type=Path,
                    help="extra calibration image dirs (png/jpg), e.g. HITL sim dump scenarios. "
                         "The whole set is then shuffled (seed 0) and ALL of it calibrates: Hailo "
                         "takes the FIRST calibset_size (default 64) and finetune dataset_size "
                         "items in order, so r80's ranges saw only calib_pov_v1 p000-p063")
    ap.add_argument("--ft", help="finetune args replacing the recipe's, e.g. 'policy=enabled, "
                                 "epochs=4, dataset_size=512, batch_size=8, learning_rate=0.000001'")
    ap.add_argument("--optimize-only", action="store_true",
                    help="stop after optimize (+eval); re-run with --alloc to compile")
    a = ap.parse_args()
    a.name += a.tag
    if a.fast:
        a.name += "_fast"
        a.skip_eval = True
    elif a.no_finetune:
        a.name += "_cal"

    from hailo_sdk_client import ClientRunner, InferenceContext
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / a.name
    log = {"onnx": a.onnx, "name": a.name, "hw": a.hw, "norm": a.norm, "stages": {}}

    def stage(tag, fn):
        t = time.time()
        r = fn()
        log["stages"][tag] = round(time.time() - t, 1)
        print(f"[{tag}] {log['stages'][tag]} s", flush=True)
        return r

    import onnx
    dims = onnx.load(a.onnx, load_external_data=False).graph.input[0].type.tensor_type.shape.dim
    h, w = dims[2].dim_value, dims[3].dim_value            # (1, 3, H, W)
    log["input_hw"] = [h, w]

    recipe = a.alls.read_text().replace("NORM", NORMS[a.norm]) + \
        "".join(e.replace("NAME", a.name) + "\n" for e in a.extra)
    parsed, quant = Path(f"{base}_parsed.har"), Path(f"{base}_quant.har")
    if quant.exists():
        runner = ClientRunner(har=str(quant), hw_arch=a.hw)
    else:
        if parsed.exists():
            runner = ClientRunner(har=str(parsed), hw_arch=a.hw)
        else:
            runner = ClientRunner(hw_arch=a.hw)
            stage("parse", lambda: runner.translate_onnx_model(a.onnx, a.name))
            runner.save_har(str(parsed))
        calib = np.concatenate([frames(h, w, POV), frames(h, w, DOMAIN)[:-a.holdout]]
                               + [frames(h, w, d) for d in a.calib_dir])
        script = recipe
        if a.extra:
            log["extra"] = a.extra
        if a.calib_dir:
            log["calib_dirs"] = [str(d) for d in a.calib_dir]
            calib = calib[np.random.default_rng(0).permutation(len(calib))]
        ft = "policy=disabled" if a.fast or a.no_finetune else a.ft
        if ft:
            log["ft"] = ft
            script = "\n".join(f"post_quantization_optimization(finetune, {ft})"
                               if l.startswith("post_quantization_optimization(finetune") else l
                               for l in script.splitlines()) + "\n"
        if a.fast:
            calib = calib[::len(calib) // 16][:16]
        if a.fast or a.no_finetune or a.calib_dir:
            script += f"model_optimization_config(calibration, batch_size=8, calibset_size={len(calib)})\n"
        log["calib_n"] = len(calib)
        runner.load_model_script(script)
        stage("optimize", lambda: runner.optimize(calib))
        runner.save_har(str(quant))

    held = frames(h, w, DOMAIN)[-a.holdout:]

    def emulate():
        outs = {}
        for tag, ctx in (("float", InferenceContext.SDK_NATIVE),
                         ("int8", InferenceContext.SDK_QUANTIZED)):
            with runner.infer_context(ctx) as c:
                outs[tag] = np.asarray(runner.infer(c, held))
        rows = [agreement(q, f) for q, f in zip(outs["int8"], outs["float"])]
        return {k: round(float(np.median([r[k] for r in rows])), 4) for k in rows[0]}

    if not a.skip_eval:
        log["int8_vs_float_heldout"] = stage("eval", emulate)
        print(f"[eval] int8 vs float on {a.holdout} held-out frames: {log['int8_vs_float_heldout']}")
    if a.optimize_only:
        Path(f"{base}_optimize_log.json").write_text(json.dumps(log, indent=1))
        print(json.dumps(log, indent=1))
        return

    if a.alloc:
        # A quantised runner refuses append=True for anything but
        # compiler_optimization_level=max (client_runner.load_model_script), so reload the
        # WHOLE recipe plus the allocator lines; optimisation commands are not re-run.
        log["alloc"] = a.alloc
        runner.load_model_script(recipe + "\n".join(a.alloc) + "\n", append=False)
    hef = stage("compile", runner.compile)
    out = Path(f"{base}{a.suffix}.hef")
    out.write_bytes(hef)
    log["hef"], log["hef_mb"] = str(out), round(out.stat().st_size / 1e6, 1)
    Path(f"{base}{a.suffix}_compile_log.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
