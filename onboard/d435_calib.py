"""One-time read of the D435's FACTORY calibration -> onboard/d435_calib.json, the numbers needed to
register its depth onto the RGB the NN sees (REC ALL sessions, depth_agreement-style comparisons).

    python3 -m venv /tmp/rs && /tmp/rs/bin/pip install pyrealsense2 numpy
    /tmp/rs/bin/python onboard/d435_calib.py          # D435 on THIS machine's USB
    python3 onboard/d435_calib.py --from-text rs_enumerate_c.txt   # or: the Pi's `rs-enumerate-devices -c` output
    python3 onboard/d435_calib.py --selftest

Runs on the laptop: pyrealsense2 has no wheel for the Pi (aarch64, py3.13, 2026-10-16) and nothing on
the Pi reads it over V4L2. The streams match what the cockpit + REC ALL use: depth / IR 848x480 @30,
colour 640x360 @30. Y8I's first byte is the LEFT imager (librealsense), = REC ALL's ir_left.
"""
import json
import re
import sys
from pathlib import Path

PATH = Path(__file__).with_name("d435_calib.json")
STREAMS = {"depth": ("Depth", "848x480"), "ir_left": ("Infrared 1", "848x480"), "ir_right": ("Infrared 2", "848x480"),
           "color": ("Color", "640x360")}


def from_text(txt):
    """The same JSON from `rs-enumerate-devices -c` text (the Pi has librealsense's tools but no pyrealsense2).
    The tool prints the rotation as rows of the matrix (rotation[j*3+i]); librealsense stores it column-major."""
    out = {"device": {"source": "rs-enumerate-devices -c"}, "intrinsics": {}, "extrinsics": {}}
    for k, (name, res) in STREAMS.items():
        m = re.search(r'Intrinsic of "' + name + r'" / ' + res + r' / \{[^}]*\}(.*?)FOV', txt, re.S)
        if not m:
            raise SystemExit(f"no intrinsics for {name} {res}")
        f = lambda key: float(re.search(r"\b" + key + r":\s*([-\d.e+]+)", m.group(1)).group(1))
        w, h = map(int, res.split("x"))
        coeffs = re.search(r"Coeffs:\s*([-\d.e+\s]+)", m.group(1)).group(1).split()
        out["intrinsics"][k] = {"width": w, "height": h, "fx": f("Fx"), "fy": f("Fy"), "ppx": f("PPX"), "ppy": f("PPY"),
                                "model": re.search(r"Distortion:\s*(.+)", m.group(1)).group(1).strip(),
                                "coeffs": [float(c) for c in coeffs[:5]]}
    for k, (a_, b_) in {"depth_to_color": ("Depth", "Color"), "ir_left_to_ir_right": ("Infrared 1", "Infrared 2")}.items():
        m = re.search(r'Extrinsic from "' + a_ + r'"\s*To\s*"' + b_ + r'"\s*:\s*Rotation Matrix:(.*?)Translation Vector:([^\n]*)', txt, re.S)
        if not m:
            raise SystemExit(f"no extrinsics {a_} -> {b_}")
        R = [float(x) for x in m.group(1).split()]
        rows = [R[0:3], R[3:6], R[6:9]]
        out["extrinsics"][k] = {"rotation_colmajor": [rows[i][j] for j in range(3) for i in range(3)],
                                "translation_m": [float(x) for x in m.group(2).split()[:3]]}
    return out


def _selftest():
    t = """Intrinsic of "Depth" / 848x480 / {Z16}
  Width:      \t848
  PPX:        \t425.5
  PPY:        \t239.5
  Fx:         \t421.0
  Fy:         \t421.5
  Distortion: \tBrown Conrady
  Coeffs:     \t0  \t0  \t0  \t0  \t0
  FOV (deg):  \t90.4 x 59.3
"""
    t += t.replace('"Depth"', '"Infrared 1"').replace("Z16", "Y8") + t.replace('"Depth"', '"Infrared 2"').replace("Z16", "Y8")
    t += t.split("Intrinsic")[1].join(["Intrinsic", ""]).replace('"Depth" / 848x480', '"Color" / 640x360').replace("421.0", "460.0")
    t += """Extrinsic from "Depth"\t  To\t  "Color" :
 Rotation Matrix:
   1        0.01     0
   -0.01    1        0
   0        0        1

 Translation Vector: 0.0149  0.0001  0.0002

Extrinsic from "Infrared 1"\t  To\t  "Infrared 2" :
 Rotation Matrix:
   1  0  0
   0  1  0
   0  0  1

 Translation Vector: -0.0501  0  0
"""
    o = from_text(t)
    import numpy as np
    R = np.array(o["extrinsics"]["depth_to_color"]["rotation_colmajor"]).reshape(3, 3).T     # as bench1000.load_calib
    assert np.allclose(R, [[1, 0.01, 0], [-0.01, 1, 0], [0, 0, 1]]), R
    assert o["intrinsics"]["color"]["fx"] == 460.0 and o["intrinsics"]["depth"]["ppx"] == 425.5
    assert o["extrinsics"]["ir_left_to_ir_right"]["translation_m"][0] == -0.0501
    print("d435_calib selftest: OK")


def live():
    import pyrealsense2 as rs

    def intr(p):
        i = p.as_video_stream_profile().get_intrinsics()
        return {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy,
                "model": str(i.model), "coeffs": list(i.coeffs)}

    def extr(a, b):
        e = a.get_extrinsics_to(b)
        return {"rotation_colmajor": list(e.rotation), "translation_m": list(e.translation)}

    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 30)
    cfg.enable_stream(rs.stream.infrared, 2, 848, 480, rs.format.y8, 30)
    cfg.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
    prof = rs.pipeline().start(cfg)
    dev = prof.get_device()
    s = {n: prof.get_stream(*k) for n, k in (("depth", (rs.stream.depth,)), ("ir_left", (rs.stream.infrared, 1)),
                                              ("ir_right", (rs.stream.infrared, 2)), ("color", (rs.stream.color,)))}
    return {"device": {k: dev.get_info(getattr(rs.camera_info, k)) for k in ("name", "serial_number", "firmware_version")},
            "depth_scale_m": dev.first_depth_sensor().get_depth_scale(),
            "intrinsics": {n: intr(p) for n, p in s.items()},
            "extrinsics": {"depth_to_color": extr(s["depth"], s["color"]),
                           "ir_left_to_ir_right": extr(s["ir_left"], s["ir_right"])}}


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        _selftest()
        sys.exit()
    out = from_text(Path(sys.argv[2]).read_text()) if sys.argv[1:2] == ["--from-text"] else live()
    PATH.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("->", PATH)
