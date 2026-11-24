"""Score every model on the thesis benchmark.
Sim (400 held-out frames, scenes 07/08/09, metric GT from the renderer): per frame, GT cropped to
the model's own box and resized (nearest) to its output; prediction -> disparity; least-squares
scale + shift to GT disparity on valid pixels (the standard protocol for relative-depth models);
AbsRel and delta1 (< 1.25) on three regions: all (0.1-20 m), near (< 3 m), and the NAVIGATION BAND
(5-40 cm above the floor within 2.5 m: what the obstacle scan reads). Real (100 held-out robot
frames, no GT): Pearson vs the DAv2-L teacher in the model's own crop.
Chip fidelity: the student HEF on the real Hailo-8L vs the same student in float.
    python bench/score.py   (data: the bench/ split of huggingface.co/datasets/itaimizlish/daniel-yard-depth,
                             at $DANIEL_DATA/bench, default ./data/bench)  ->  bench/scores.json"""
import json, os
from pathlib import Path
import math, sys
import cv2, numpy as np
H = Path(__file__).resolve().parent
R = H.parent
DATA = Path(os.environ.get("DANIEL_DATA", R / "data")) / "bench"
PRED = DATA / "predictions"
sys.path[:0] = [str(R / "onboard"), str(R / "sim")]
import pathplan as pp, kino_plan
MAN = json.loads((DATA / "props_manifest.json").read_text())
SC = {s["name"][:2]: s for s in json.loads((DATA / "scenarios.json").read_text())["scenarios"]}
FR = json.loads((DATA / "frames.json").read_text())
SBOX = (3, 0, 498, 280)
timing = json.loads((DATA / "timing.json").read_text()) if (DATA / "timing.json").exists() else {}
chip_t = json.loads((DATA / "chip_timing.json").read_text())


def load_models():
    M = {}
    for f in sorted(PRED.glob("preds_*.npy")):
        name = f.stem[len("preds_"):]
        M[name] = {"sim": np.load(f), "real": np.load(PRED / f"real_{name}.npy"), "box": SBOX, "kind": "disp", "where": "RTX 5090 (float)"}
    for f in sorted(PRED.glob("out_*.npz")):
        z = np.load(f); stem = f.stem[len("out_"):]; names = list(z["names"])
        sim_i = [names.index(f"b{k:03d}") for k in range(len(FR))]
        real_i = [i for i, n in enumerate(names) if n.startswith("r")]
        M["hailo:" + stem] = {"sim": z["out"][sim_i], "real": z["out"][real_i], "box": tuple(int(v) for v in z["box"]),
                              "kind": str(z["kind"]), "where": "Hailo-8L (int8)"}
    return M


def to_disp(p, kind):
    p = p.astype(np.float32)
    return p if kind == "disp" else 1.0 / np.maximum(p, 1e-3)


def crop_resize(a, box, hw, interp=cv2.INTER_NEAREST):
    x, y, w, h = box
    return cv2.resize(a[y:y + h, x:x + w], (hw[1], hw[0]), interpolation=interp)


def metrics(zp, z, m):
    if m.sum() < 50:
        return None
    r = np.abs(zp[m] - z[m]) / z[m]
    q = np.maximum(zp[m] / z[m], z[m] / zp[m])
    return float(r.mean()), float((q < 1.25).mean())


def main():
    M = load_models()
    gts, hts, obj = [], [], []
    for r in FR:
        z = np.load(DATA / r["gt"]).astype(np.float32)
        meta = r["meta"]
        v = np.arange(z.shape[0], dtype=np.float32)[:, None]
        u = np.arange(z.shape[1], dtype=np.float32)[None, :]
        hts.append(meta["cam_h"] - (v - meta["cy"]) * z / meta["fy"])
        gts.append(z)
        # OBSTACLE pixels: back-projected with the car's GT pose, on a prop collider (+5 cm).
        # Excludes the splat's depth speckle / floaters and terrain, which dominate raw GT near the floor.
        st = r["pose"]                                 # the car's GT pose at this frame
        rear = pp.rear_of((st["x"], st["y"]), st["yaw"]); th = rear[2]
        camx, camy = rear[0] + pp.C_CAM * math.cos(th), rear[1] + pp.C_CAM * math.sin(th)
        X = (u - meta["cx"]) * z / meta["fx"]
        wx = camx + z * math.cos(th) + X * math.sin(th); wy = camy + z * math.sin(th) - X * math.cos(th)
        circ = np.asarray(kino_plan.prop_circles(SC[r["scene"]]["props"], MAN), float).reshape(-1, 3)
        on = np.zeros(z.shape, bool)
        for cx_, cy_, rr in circ:
            on |= np.hypot(wx - cx_, wy - cy_) < rr + 0.05
        obj.append(on & np.isfinite(z))
    teacher_real = M["dav2_l"]["real"].astype(np.float32) if "dav2_l" in M else None
    out = {}
    for name, m in M.items():
        acc = {"all": [], "near": [], "obst": [], "band": []}
        for k, (z, ht, ob) in enumerate(zip(gts, hts, obj)):
            p = to_disp(m["sim"][k], m["kind"]); hw = p.shape
            zc, hc = crop_resize(z, m["box"], hw), crop_resize(ht, m["box"], hw)
            oc = crop_resize(ob.astype(np.uint8), m["box"], hw) > 0
            valid = np.isfinite(zc) & (zc > 0.1) & (zc < 20) & np.isfinite(p)
            if valid.sum() < 500:
                continue
            A = np.stack([p[valid], np.ones(valid.sum())], 1)
            s, t = np.linalg.lstsq(A, 1.0 / zc[valid], rcond=None)[0]
            zp = 1.0 / np.clip(s * p + t, 1 / 20, 1 / 0.05)
            for reg, mask in (("all", valid), ("near", valid & (zc < 3)), ("obst", valid & oc & (zc < 3)),
                              ("band", valid & oc & (zc < 3) & (hc > 0.05) & (hc < 0.40))):
                v = metrics(zp, zc, mask)
                if v:
                    acc[reg].append(v)
        row = {"where": m["where"], "box": m["box"], "kind": m["kind"], "n_sim": len(acc["all"])}
        for reg, v in acc.items():
            a = np.array(v)
            row[f"absrel_{reg}"], row[f"d1_{reg}"] = round(float(a[:, 0].mean()), 4), round(float(a[:, 1].mean()), 4)
            row[f"absrel_{reg}_med"], row[f"n_{reg}"] = round(float(np.median(a[:, 0])), 4), len(a)
        if teacher_real is not None:
            prs = []
            for k in range(len(m["real"])):
                p = to_disp(m["real"][k], m["kind"])
                x, y, w, h = m["box"]
                # the teacher map lives in the student crop (SBOX at 288x512): map the box into it
                sx, sy = 512 / SBOX[2], 288 / SBOX[3]
                x0, x1 = int(round((x - SBOX[0]) * sx)), int(round((x + w - SBOX[0]) * sx))
                y0, y1 = int(round((y - SBOX[1]) * sy)), int(round((y + h - SBOX[1]) * sy))
                t = cv2.resize(teacher_real[k][max(y0, 0):y1, max(x0, 0):x1], (p.shape[1], p.shape[0]), interpolation=cv2.INTER_AREA)
                prs.append(float(np.corrcoef(p.ravel(), t.ravel())[0, 1]))
            row["real_pearson_vs_dav2l_p50"] = round(float(np.median(prs)), 4)
            row["real_pearson_vs_dav2l_p5"] = round(float(np.percentile(prs, 5)), 4)
        stem = name.split(":")[-1]
        if name in timing:
            row.update(timing[name])
        if stem in chip_t:
            row.update({"hailo_" + k: v for k, v in chip_t[stem].items() if k in ("fps_seq_b1", "npu_ms_p50", "in_hw")})
        out[name] = row
        print(f"{name:34s} AbsRel all {row['absrel_all']:.3f} near {row['absrel_near']:.3f} obst {row['absrel_obst']:.3f} (med {row['absrel_obst_med']:.3f}) band {row['absrel_band']:.3f} | "
              f"d1 all {row['d1_all']:.3f} obst {row['d1_obst']:.3f} band {row['d1_band']:.3f} | real r50 {row.get('real_pearson_vs_dav2l_p50', float('nan')):.3f} "
              f"| {row.get('hailo_fps_seq_b1', '')} FPS", flush=True)
    # chip fidelity: student HEF vs the same student in float, sim + real
    for v in ("v3", "v5"):
        if f"student_{v}" not in M or f"hailo:dav2student_{v}_288x512" not in M:
            continue
        a, b = M[f"student_{v}"], M[f"hailo:dav2student_{v}_288x512"]
        pr = [float(np.corrcoef(x.astype(np.float32).ravel(), y.astype(np.float32).ravel())[0, 1])
              for x, y in zip(np.concatenate([a["sim"], a["real"]]), np.concatenate([b["sim"], b["real"]]))]
        out[f"_chip_fidelity_student_{v}"] = {"pearson_p50": round(float(np.median(pr)), 5), "pearson_p5": round(float(np.percentile(pr, 5)), 5), "n": len(pr)}
        print(f"chip fidelity {v} (HEF vs float):", out[f"_chip_fidelity_student_{v}"])
    (H / "scores.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
