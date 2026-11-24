"""CNN student distilled from Depth-Anything-V2-L, built to survive the Hailo-8L's int8.

Why: our DAv2-S 224x392 HEF matches float on real robot video (Pearson median 0.99) but
breaks on a third of the HITL yard renders. Re-calibration and QFT only move it from 48 %
to 37 % of held-out frames below 0.8. The cap is structural: hailo8l runs every
attention layer in int8 (a8_w8_a8), and at 449 tokens a diffuse attention weight
(~1/449) is below one int8 step (requant/log.md, 2026-09-30). A conv/BN/ReLU/resize/concat
net has no softmax to crush, so int8 costs almost nothing, and it runs at a higher
resolution in real time.

    PY=~/isaacsim/python.sh     # torch 2.10 cu128 (RTX 5090)
    $PY student_depth.py frames              # frame list + split -> EXP/frames.json
    $PY student_depth.py label               # DAv2-L teacher -> EXP/labels/*.npy (f16)
    $PY student_depth.py train [--epochs N]  # -> EXP/student.pt, EXP/train_log.json
    $PY student_depth.py eval                # held-out Pearson vs DAv2-L and vs DAv2-S
    $PY student_depth.py export              # -> EXP/dav2student_r34_288x512.onnx (RGB/255 NCHW in)

Every frame is standardised to the HITL POV format (504x280 RGB, the Pi's input). Held
out: sim scenarios 07/08/09 of every HITL run and the real robot_camera_10s clip.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments/2026-09-30_student_depth"
DEPS = Path.home() / "thesis-depth-robotics"
H, W = 288, 512                      # student input: /32-divisible, the POV's 1.78 aspect
TH, TW = 280, 504                    # teacher input: /14-divisible, the POV's own size
HOLDOUT_SCEN = ("07_", "08_", "09_")
REAL_VIDEOS = {  # path: (sample every N frames, held out)
    Path.home() / "vidmap/ariel-obstacle.mp4": (10, False),
    Path.home() / "Downloads/drive_frames.mp4": (10, False),
    REPO / "experiments/2026-09-28_pi5_edge_depth/video/robot_pov_10s.mp4": (3, True),
}


def pov(bgr):
    """Any frame -> 504x280 RGB, centre-cropped to 1.8:1 (what the Pi receives)."""
    h, w = bgr.shape[:2]
    a = TW / TH
    cw, ch = (w, round(w / a)) if w / h < a else (round(h * a), h)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return cv2.cvtColor(cv2.resize(bgr[y0:y0 + ch, x0:x0 + cw], (TW, TH), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ data
def cmd_frames(a):
    """Standardise every source frame to EXP/img/<id>.jpg; split by source."""
    img = EXP / "img"
    img.mkdir(parents=True, exist_ok=True)
    items = []

    def add(bgr, key, split, kind):
        p = img / f"{key}.jpg"
        if not p.exists():
            cv2.imwrite(str(p), cv2.cvtColor(pov(bgr), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        items.append({"id": key, "split": split, "kind": kind})

    for f in sorted((REPO / "experiments/2026-09-29_hitl/runs").glob("*/dump/*/f[0-9][0-9][0-9][0-9].jpg")):
        if int(f.stem[1:]) % 4:                       # dumps are every 2 steps; keep 0.2 s apart
            continue
        run, scen = f.parts[-4], f.parts[-2]
        add(cv2.imread(str(f)), f"sim_{run}_{scen}_{f.stem}",
            "val" if scen.startswith(HOLDOUT_SCEN) else "train", "sim")
    for f in sorted((DEPS / "calib_pov_v1").glob("*.png")):
        add(cv2.imread(str(f)), f"realpov_{f.stem}", "train", "real")
    for vid, (every, held) in REAL_VIDEOS.items():
        cap, k = cv2.VideoCapture(str(vid)), 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if k % every == 0:
                add(bgr, f"real_{vid.stem}_{k:05d}", "val" if held else "train", "real")
            k += 1
    # v2 data: random-layout sim renders (hitl_sim --gen-data; never crates, so scenario 07
    # stays a novel-object test) and DIVERSE real video (any footage: the teacher labels it),
    # because v1 generalised worst exactly on the unseen prop (07 crates: 13 % of frames < 0.8).
    for f in sorted((EXP / "gendata").glob("*.jpg")):
        add(cv2.imread(str(f)), f"gen_{f.stem}", "train", "sim")
    held = {v.resolve() for v in REAL_VIDEOS}
    for vid in sorted(set((Path.home() / "Downloads").glob("*.mp4")) | set((Path.home() / "vidmap").glob("*.mp4"))):
        if vid.resolve() in held or any(v.name == vid.name for v in REAL_VIDEOS):
            continue
        cap = cv2.VideoCapture(str(vid))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for k in np.linspace(0, max(n - 1, 0), 40).astype(int) if n > 0 else []:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, bgr = cap.read()
            if ok:
                add(bgr, f"realdiv_{vid.stem[:40]}_{k:06d}", "train", "real")
    (EXP / "frames.json").write_text(json.dumps(items))
    n = {(i["split"], i["kind"]): 0 for i in items}
    for i in items:
        n[(i["split"], i["kind"])] += 1
    print(f"{len(items)} frames: {n}")


def teacher(encoder="vitl"):
    import torch
    sys.path.insert(0, str(DEPS / "depth-anything-v2-src"))
    from depth_anything_v2.dpt import DepthAnythingV2
    cfg = {"vits": (64, [48, 96, 192, 384]), "vitb": (128, [96, 192, 384, 768]),
           "vitl": (256, [256, 512, 1024, 1024])}[encoder]
    m = DepthAnythingV2(encoder=encoder, features=cfg[0], out_channels=cfg[1])
    m.load_state_dict(torch.load(DEPS / f"depth-anything-v2-src/checkpoints/depth_anything_v2_{encoder}.pth",
                                 map_location="cpu"))
    return m.cuda().eval()


def cmd_label(a):
    """DAv2 teacher disparity at 280x504 for every frame (float16), batched on the GPU."""
    import torch
    items = json.loads((EXP / "frames.json").read_text())
    out = EXP / f"labels_{a.teacher}"
    out.mkdir(exist_ok=True)
    todo = [i for i in items if not (out / f"{i['id']}.npy").exists()]
    m = teacher(a.teacher)
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in range(0, len(todo), 16):
            chunk = todo[b:b + 16]
            x = np.stack([cv2.cvtColor(cv2.imread(str(EXP / "img" / f"{i['id']}.jpg")), cv2.COLOR_BGR2RGB)
                          for i in chunk])
            x = (torch.from_numpy(x).cuda().permute(0, 3, 1, 2).float() / 255 - mean) / std
            y = m(x).float().cpu().numpy()
            for i, d in zip(chunk, y):
                np.save(out / f"{i['id']}.npy", d.astype(np.float16))
    print(f"labelled {len(todo)} frames with DAv2-{a.teacher} in {time.time() - t0:.0f} s -> {out}")


# ------------------------------------------------------------------ model
def build():
    """ResNet-34 encoder + U-Net decoder: conv, BN, ReLU, bilinear x2, concat. Every op maps
    1:1 onto the Hailo dataflow compiler; no attention, no LayerNorm."""
    import torch
    import torch.nn as nn
    import torchvision

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            try:
                r = torchvision.models.resnet34(weights="IMAGENET1K_V1")
            except Exception as e:                      # offline: train from scratch (slower)
                print(f"[student] no ImageNet weights ({e}); random init", flush=True)
                r = torchvision.models.resnet34()
            self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)        # /2, 64
            self.pool = r.maxpool
            self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4   # /4../32
            self.d4 = block(512 + 256, 256)
            self.d3 = block(256 + 128, 128)
            self.d2 = block(128 + 64, 64)
            self.d1 = block(64 + 64, 48)
            self.d0 = block(48, 32)
            self.head = nn.Conv2d(32, 1, 3, padding=1)
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):                           # x: RGB in [0, 1], NCHW
            up = lambda t: nn.functional.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)  # noqa: E731
            x = (x - self.mean) / self.std
            s0 = self.stem(x)
            s1 = self.l1(self.pool(s0))
            s2 = self.l2(s1)
            s3 = self.l3(s2)
            s4 = self.l4(s3)
            y = self.d4(torch.cat([up(s4), s3], 1))
            y = self.d3(torch.cat([up(y), s2], 1))
            y = self.d2(torch.cat([up(y), s1], 1))
            y = self.d1(torch.cat([up(y), s0], 1))
            return self.head(self.d0(up(y)))[:, 0]      # relative disparity, H x W

    return Student()


def _norm(d):
    """Scale-shift normalisation per image (MiDaS): zero median, unit mean abs deviation."""
    import torch
    med = d.flatten(1).median(1).values.view(-1, 1, 1)
    s = (d - med).abs().flatten(1).mean(1).clamp_min(1e-6).view(-1, 1, 1)
    return (d - med) / s


def ssi_loss(pred, tgt, floor_w=1.0):
    """Scale-shift-invariant L1 + multi-scale gradient matching (MiDaS). floor_w > 1 weights the
    rows below the image centre (the horizon of the level sim camera): the floor and every
    object's bottom, where the scan finds contacts. v3 resolves a 3 cm base no better than a
    5 cm one (scan side deficit 4.2 cm at H_MIN 3 cm vs 1.7 cm for GT depth, l31rt)."""
    import torch
    import torch.nn.functional as F
    p, t = _norm(pred), _norm(tgt)

    def wmean(x):
        if floor_w == 1.0:
            return x.mean()
        w = torch.ones(x.shape[-2], device=x.device)
        w[x.shape[-2] // 2:] = floor_w
        return (x * w[:, None]).sum() / (w.sum() * x.shape[-1] * x.shape[0])
    loss = wmean((p - t).abs())
    for k in range(4):
        pk, tk = (p, t) if k == 0 else (F.avg_pool2d(p[:, None], 2 ** k)[:, 0], F.avg_pool2d(t[:, None], 2 ** k)[:, 0])
        r = pk - tk
        loss = loss + 0.5 * (wmean((r[:, :, 1:] - r[:, :, :-1]).abs()) + wmean((r[:, 1:] - r[:, :-1]).abs()))
    return loss


class Frames:
    def __init__(self, items, labels, train):
        self.items, self.labels, self.train = items, labels, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i = self.items[k]
        rgb = cv2.cvtColor(cv2.imread(str(EXP / "img" / f"{i['id']}.jpg")), cv2.COLOR_BGR2RGB)
        lab = np.load(self.labels / f"{i['id']}.npy").astype(np.float32)
        if self.train:
            rng = np.random
            if rng.rand() < 0.5:
                rgb, lab = rgb[:, ::-1], lab[:, ::-1]
            s = rng.uniform(0.8, 1.0)                           # random crop, same box both
            ch, cw = int(TH * s), int(TW * s)
            y0, x0 = rng.randint(0, TH - ch + 1), rng.randint(0, TW - cw + 1)
            rgb, lab = rgb[y0:y0 + ch, x0:x0 + cw], lab[y0:y0 + ch, x0:x0 + cw]
            rgb = np.clip(rgb.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20)
                          + rng.normal(0, rng.uniform(0, 4), rgb.shape), 0, 255)
        x = cv2.resize(np.ascontiguousarray(rgb).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA) / 255
        y = cv2.resize(np.ascontiguousarray(lab), (W, H), interpolation=cv2.INTER_AREA)
        return x.transpose(2, 0, 1).astype(np.float32), y.astype(np.float32)


def cmd_train(a):
    import torch
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    labels = EXP / (a.labels or f"labels_{a.teacher}")
    items = [i for i in json.loads((EXP / "frames.json").read_text()) if (labels / f"{i['id']}.npy").exists()]
    tr = [i for i in items if i["split"] == "train"]
    tr = tr + [i for i in tr if i["kind"] == "real"] * a.real_x      # real frames are scarce
    va = [i for i in items if i["split"] == "val"][::3]
    dl = torch.utils.data.DataLoader(Frames(tr, labels, True), batch_size=a.bs, shuffle=True,
                                     num_workers=10, drop_last=True, persistent_workers=True)
    net = build().cuda()
    if a.init:                                      # fine-tune (v3: from v1 on sim GT depth)
        net.load_state_dict(torch.load(EXP / a.init))
    enc = [p for n, p in net.named_parameters() if n.startswith(("stem", "l1", "l2", "l3", "l4"))]
    dec = [p for n, p in net.named_parameters() if not n.startswith(("stem", "l1", "l2", "l3", "l4"))]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr / 3}, {"params": dec, "lr": a.lr}], weight_decay=1e-4)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr / 3, a.lr], total_steps=steps, pct_start=0.05)
    log, best = [], -1.0
    for ep in range(a.epochs):
        net.train()
        t0, tot = time.time(), 0.0
        for x, y in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = net(x.cuda(non_blocking=True))
            loss = ssi_loss(p.float(), y.cuda(non_blocking=True), a.floor_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        r = evaluate(net, va, labels)
        log.append({"epoch": ep, "loss": tot / len(dl), "val_pearson": r, "s": round(time.time() - t0, 1)})
        print(f"[train] ep {ep:2d} loss {tot / len(dl):.4f} val {r}  {time.time() - t0:.0f}s", flush=True)
        if r["p50_all"] > best:
            best = r["p50_all"]
            torch.save(net.state_dict(), EXP / a.out)
        (EXP / f"train_log{'' if a.out == 'student.pt' else '_' + Path(a.out).stem}.json").write_text(
            json.dumps({"args": vars(a), "log": log}, indent=1))


def evaluate(net, items, labels, ref="teacher"):
    """Per-frame Pearson of the student vs a reference disparity, by kind."""
    import torch
    net.eval()
    r = {"sim": [], "real": []}
    with torch.no_grad():
        for b in range(0, len(items), 32):
            chunk = items[b:b + 32]
            ds = Frames(chunk, labels, False)
            xs, ys = zip(*[ds[k] for k in range(len(chunk))])
            p = net(torch.from_numpy(np.stack(xs)).cuda()).float().cpu().numpy()
            for i, pi, yi in zip(chunk, p, ys):
                r[i["kind"]].append(float(np.corrcoef(pi.ravel(), yi.ravel())[0, 1]))
    out = {}
    for k, v in r.items():
        if v:
            v = np.array(v)
            out[f"p50_{k}"] = round(float(np.median(v)), 3)
            out[f"p5_{k}"] = round(float(np.percentile(v, 5)), 3)
            out[f"lt08_{k}"] = round(float(np.mean(v < 0.8)), 3)
    allv = np.concatenate([np.array(v) for v in r.values() if v])
    out["p50_all"] = round(float(np.median(allv)), 3)
    return out


def cmd_eval(a):
    import torch
    items = [i for i in json.loads((EXP / "frames.json").read_text()) if i["split"] == "val"]
    net = build().cuda()
    net.load_state_dict(torch.load(EXP / "student.pt"))
    print("student vs DAv2-L (held-out):", evaluate(net, items, EXP / "labels_vitl"))
    if (EXP / "labels_vits").exists():
        print("student vs DAv2-S (held-out):", evaluate(net, items, EXP / "labels_vits"))


def cmd_export(a):
    import torch
    net = build()
    net.load_state_dict(torch.load(EXP / a.out, map_location="cpu"))
    net.eval()
    out = EXP / (f"dav2student_r34_{H}x{W}.onnx" if a.out == "student.pt" else f"dav2student_{Path(a.out).stem}_{H}x{W}.onnx")
    torch.onnx.export(net, torch.zeros(1, 3, H, W), str(out), input_names=["input"], output_names=["disp"],
                      opset_version=17, dynamo=False)
    print(f"exported {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["frames", "label", "train", "eval", "export"])
    ap.add_argument("--teacher", default="vitl", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--real-x", type=int, default=4, help="extra copies of each real training frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", help="label dir under EXP (default labels_<teacher>); frames without a label are skipped. "
                                     "labels_gtmix: sim frames on sim GT disparity, real frames on DAv2-L (v3)")
    ap.add_argument("--init", help="weights under EXP to fine-tune from")
    ap.add_argument("--floor-w", type=float, default=1.0, help="loss weight of the rows below the image centre (v5: 4)")
    ap.add_argument("--out", default="student.pt", help="weights file under EXP (train writes, export reads)")
    a = ap.parse_args()
    EXP.mkdir(parents=True, exist_ok=True)
    {"frames": cmd_frames, "label": cmd_label, "train": cmd_train, "eval": cmd_eval, "export": cmd_export}[a.cmd](a)
