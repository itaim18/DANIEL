"""On-chip depth benchmark (Raspberry Pi 5 + Hailo-8L): every HEF on the same frames through
edge_depth.Net (centre crop at the model's aspect, uint8 in, norm in the HEF), one VDevice with
the ROUND_ROBIN scheduler. Per HEF: out_<stem>.npz (float16 outputs in the model's own units,
the crop box, kind disp|depth) and timing: sequential batch-1 frame time = CPU pre + NPU + post.
    python3 chipbench.py FRAMES_DIR HEF...           (run on the Pi)"""
import json, sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "onboard"))
from edge_depth import Net
from hailo_platform import HailoSchedulingAlgorithm, VDevice
p = VDevice.create_params(); p.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
vdev = VDevice(p)
frames = sorted(Path(sys.argv[1]).glob("*.jpg"))
rgbs = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in frames]
res = json.loads(Path("chip_timing.json").read_text()) if Path("chip_timing.json").exists() else {}  # merge, keep earlier HEFs
for hef in sys.argv[2:]:
    net = Net(hef, "hailo", vdev)
    run = lambda rgb: net.post(net.launch(net.pre(rgb))())
    for _ in range(5):
        run(rgbs[0])
    outs, ts, tn = [], [], []
    for rgb in rgbs:
        t0 = time.perf_counter(); x = net.pre(rgb); t1 = time.perf_counter()
        y = net.launch(x)(); t2 = time.perf_counter()
        outs.append(net.post(y).astype(np.float16)); ts.append(time.perf_counter() - t0); tn.append(t2 - t1)
    out = np.stack(outs)
    assert np.isfinite(out.astype(np.float32)).all() and out.std() > 0, f"{hef}: constant/NaN output"
    stem = Path(hef).stem
    np.savez_compressed(f"out_{stem}.npz", out=out, box=np.array(net.box), kind=net.kind, names=np.array([f.stem for f in frames]))
    res[stem] = {"in_hw": [net.h, net.w], "box": list(net.box), "kind": net.kind,
                 "frame_ms_p50": round(1e3 * float(np.median(ts)), 2), "npu_ms_p50": round(1e3 * float(np.median(tn)), 2),
                 "fps_seq_b1": round(1.0 / float(np.median(ts)), 1)}
    print(stem, res[stem], flush=True)
    del net
Path("chip_timing.json").write_text(json.dumps(res, indent=1))
