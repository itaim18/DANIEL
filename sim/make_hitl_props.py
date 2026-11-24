#!/usr/bin/env python3
"""HITL obstacle props: fetch NVIDIA Isaac 6.0 assets + local ones, wrap them as MOVABLE
kinematic rigid bodies with an invisible cylinder collider. CPU only (pxr), no Kit.

    ~/isaacsim/python.sh wheeled_lab/env/make_hitl_props.py            # fetch + wrap
    ~/isaacsim/python.sh wheeled_lab/env/make_hitl_props.py --wrap-only

Why a wrapper (and not UsdFileCfg straight at the asset):
  * NVIDIA's office/warehouse props carry no RigidBodyAPI, so Isaac Lab's
    RigidObject cannot bind them. Only a rigid body can be TELEPORTED per scenario
    (write_root_pose_to_sim), which is what lets a whole scenario suite run in ONE
    Isaac boot (40-130 s each) instead of one boot per layout.
  * Units and up-axis differ per asset (NVIDIA ships metersPerUnit 0.01, some Y-up),
    and USD does not rescale across a reference arc. The wrapper folds scale +
    rotation + translation so the prop's bbox BOTTOM-CENTRE sits at the origin. The
    sim places the origin on the terrain height.
  * Collision is a separate invisible cylinder sized to the prop's footprint. It is
    convex and cheap, and the car bumps into something the size of what it sees; the
    visual mesh's own geometry never collides (no triangle-mesh cost, no concave
    surprises).
Output: wheeled_lab/env/props/<name>.usda + manifest.json {name: {w, d, h, radius, src}}.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils

HERE = Path(__file__).resolve().parent
OUT = HERE / "props"
SRC = OUT / "src"
ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0"
OCD = Path.home() / "OCD/assets/usd"

# name -> (source, kind) ; source is an S3 key under ROOT or an absolute local path
PROPS = {
    "office_chair": ("Isaac/Environments/Office/Props/SM_ChairOffice.usd", "s3"),
    "chair": ("Isaac/Environments/Office/Props/SM_Chair.usd", "s3"),
    "traffic_cone": ("Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd", "s3"),
    "cardboard_box": ("Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd", "s3"),
    "barrel": ("Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_01.usd", "s3"),
    "trash_can": ("Isaac/Environments/Office/Props/SM_TrashCan.usd", "s3"),
    "wet_floor_sign": ("Isaac/Environments/Simple_Warehouse/Props/S_WetFloorSign.usd", "s3"),
    "school_chair": (str(OCD / "chair/SchoolChair_01.usd"), "local"),
    "crate": (str(OCD / "cargo/wooden_crate_01.usd"), "local"),
    "person": (str(OCD / "person/male_adult_construction_05_new/"
                     "male_adult_construction_05_new_posed.usda"), "local"),
    "person_b": (str(OCD / "person/F_Business_02/F_Business_02_posed.usda"), "local"),
}


ROUND = {"traffic_cone", "barrel", "trash_can", "office_chair"}


def fetch(key: str, seen: set) -> Path | None:
    """Download an S3 USD layer and, recursively, every asset it references."""
    if key in seen:
        return SRC / key
    seen.add(key)
    dst = SRC / key
    if not dst.exists() or dst.stat().st_size == 0:
        dst.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["curl", "-sS", "-f", "--retry", "3", "--max-time", "120",
                            "-o", str(dst), f"{ROOT}/{key}"], capture_output=True)
        if r.returncode != 0:
            dst.unlink(missing_ok=True)
            print(f"  miss {key}")
            return None
        print(f"  got  {key} ({dst.stat().st_size} B)")
    if dst.suffix.lower() == ".mdl":
        # MDL imports are not USD references: `import .::OmniUe4Function::*` pulls a
        # SIBLING module, and textures are string paths. Missing either makes Kit fail
        # the material ("could not find module '.::OmniUe4Base'", 2026-09-29).
        import re
        text = dst.read_text(errors="ignore")
        for mod in set(re.findall(r"\.::(\w+)", text)):
            fetch(_norm(str(Path(key).parent / f"{mod}.mdl")), seen)
        for tex in set(re.findall(r'"(\.{0,2}/?[\w./-]+\.(?:png|jpg|jpeg|exr|tga|hdr))"', text)):
            fetch(_norm(str(Path(key).parent / tex)), seen)
    if dst.suffix.lower() in (".usd", ".usda", ".usdc"):
        layers, refs, payloads = UsdUtils.ExtractExternalReferences(str(dst))
        for p in [*layers, *refs, *payloads]:
            if not p or "://" in p or p.startswith("/"):
                continue                      # absolute/remote/MDL-builtin: skip
            dep = str((Path(key).parent / p.split("<")[0]).as_posix())
            dep = str(Path(dep).resolve().relative_to(Path("/").resolve())) \
                if dep.startswith("/") else _norm(dep)
            fetch(dep, seen)
    return dst


def _norm(p: str) -> str:
    parts = []
    for s in p.split("/"):
        if s == "..":
            if parts:
                parts.pop()
        elif s and s != ".":
            parts.append(s)
    return "/".join(parts)


def bake_skinned(src: Path) -> Path:
    """Skinned character -> static mesh in its posed shape (props/src/<stem>_baked.usdc).

    Isaac characters render as a ~1 cm blob in the headless Isaac Lab app: their mesh
    points live in joint space and nothing there runs the skinning. The first HITL
    person was invisible, and the car drove into it (2026-09-29). UsdSkel.BakeSkinning
    writes the posed points; then the skel bindings are stripped so nothing re-skins
    them. Flatten() makes texture paths absolute, so the baked file can live anywhere.
    Units come from the sublayer: the *_posed.usda root authors no metersPerUnit or
    upAxis, so USD's fallbacks (0.01, Y) had shrunk the person to 1.8 cm, lying down.
    """
    from pxr import UsdSkel
    st = Usd.Stage.Open(str(src))
    flat = Usd.Stage.Open(st.Flatten())
    if not UsdSkel.BakeSkinning(flat.Traverse()):
        raise RuntimeError(f"BakeSkinning failed for {src}")
    for prim in list(flat.Traverse()):
        if not prim.IsValid():                     # under a Skeleton deactivated just now
            continue
        if prim.IsA(UsdSkel.Skeleton) or prim.IsA(UsdSkel.Animation):
            prim.SetActive(False)
            continue
        if prim.IsA(UsdSkel.Root):
            prim.SetTypeName("Xform")
        prim.RemoveAPI(UsdSkel.BindingAPI)
        for prop in prim.GetPropertyNames():
            if prop.startswith(("skel:", "primvars:skel:")):
                prim.RemoveProperty(prop)
    sub = [Usd.Stage.Open(str(src.parent / l)) for l in st.GetRootLayer().subLayerPaths]
    ref = sub[0] if sub else st
    UsdGeom.SetStageMetersPerUnit(flat, UsdGeom.GetStageMetersPerUnit(ref))
    UsdGeom.SetStageUpAxis(flat, UsdGeom.GetStageUpAxis(ref))
    dst = SRC / f"{src.stem}_baked.usdc"
    dst.parent.mkdir(parents=True, exist_ok=True)
    flat.Export(str(dst))
    return dst


def measure(path: Path):
    stage = Usd.Stage.Open(str(path))
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    up = UsdGeom.GetStageUpAxis(stage)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    rng = cache.ComputeWorldBound(root).ComputeAlignedRange()
    return mpu, up, rng, root.GetPath()


def wrap(name: str, src: Path, meta: dict, scale_extra: float = 1.0):
    mpu, up, rng, root_path = measure(src)
    lo, hi = rng.GetMin(), rng.GetMax()
    s = mpu * scale_extra
    if up == "Y":      # rotate Y-up asset to Z-up: (x, y, z) -> (x, -z, y)
        w, d, h = (hi[0] - lo[0]) * s, (hi[2] - lo[2]) * s, (hi[1] - lo[1]) * s
        cx, cyy, bot = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2, lo[1]
        rot = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90))
        pre = Gf.Matrix4d().SetTranslate(Gf.Vec3d(-cx, -bot, -cyy))
    else:
        w, d, h = (hi[0] - lo[0]) * s, (hi[1] - lo[1]) * s, (hi[2] - lo[2]) * s
        cx, cyy, bot = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]
        rot = Gf.Matrix4d(1.0)
        pre = Gf.Matrix4d().SetTranslate(Gf.Vec3d(-cx, -cyy, -bot))
    xf = pre * rot * Gf.Matrix4d().SetScale(Gf.Vec3d(s, s, s))
    radius = 0.5 * math.hypot(w, d) * 0.85     # footprint circle, slightly inside the bbox corners
    if name in ROUND:
        # Round bodies: the collider must be what the camera sees. The bbox includes
        # handles and the chair's caster spread, and the 0.85-half-diagonal circle made
        # the barrel's collider 0.39 m round a 0.30 m body (2026-09-29).
        radius = 0.5 * min(w, d)
    if name.startswith("person"):
        radius = 0.25                          # feet + legs; the arms are above the car
    # BOXY props collide as a box of their own footprint. The 0.85-half-diagonal cylinder
    # stood up to 12 cm outside a box's faces, where the camera sees open floor: the car hit
    # invisible volume, and the contact metric scored it (sensed-vs-collider range to a box
    # read +11 % long even on sim GT depth, 2026-09-30).
    boxy = name in BOXY

    out = OUT / f"{name}.usda"
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    prop = UsdGeom.Xform.Define(stage, "/Prop")
    stage.SetDefaultPrim(prop.GetPrim())
    from pxr import UsdPhysics
    rb = UsdPhysics.RigidBodyAPI.Apply(prop.GetPrim())
    rb.CreateKinematicEnabledAttr(True)
    geom = stage.DefinePrim("/Prop/Geom", "Xform")
    rel = Path(src).resolve()
    geom.GetReferences().AddReference(str(Path(_rel(rel, out.parent))), root_path)
    UsdGeom.Xformable(geom).AddTransformOp().Set(xf)
    legs = _body_cells(stage) if name in LEGGED else None
    # LEGGED props collide with their LEGS only: a solid cylinder bulged 6 cm past the
    # line between a chair's legs, where the camera sees open floor, and the car drove
    # into that invisible wall for 252 steps (v6 GT, 04_chair_cluster). The seat is above
    # the car (0.45 m vs 0.25 m), so a real Traxxas fits under it; so does this one.
    # Legs = where the surface crosses the car's BODY band, not where it touches the floor:
    # the floor-contact cells turned "chair"'s 2 cm floor tube into a 0.40 m fence (contact
    # in 04 in 9 of 10 GT-pose runs) and kept only the feet of school_chair's splayed legs.
    for k, (cx_, cy_) in enumerate(legs if legs else [(0.0, 0.0)]):
        hh = min(h, 0.40) if legs else h
        if boxy:
            col = UsdGeom.Cube.Define(stage, "/Prop/Collider")
            col.CreateSizeAttr(1.0)
            col.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, hh / 2))
            col.AddScaleOp().Set(Gf.Vec3f(w, d, hh))
        else:
            col = UsdGeom.Cylinder.Define(stage, f"/Prop/Collider{k if legs else ''}")
            col.CreateRadiusAttr(LEG_R if legs else radius)
            col.CreateHeightAttr(hh)
            col.CreateAxisAttr("Z")
            col.AddTranslateOp().Set(Gf.Vec3d(cx_, cy_, hh / 2))
        col.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    stage.GetRootLayer().Save()
    meta[name] = {"w": round(w, 3), "d": round(d, 3), "h": round(h, 3),
                  "radius": round(radius, 3), "src": str(src), "mpu": mpu, "up": up}
    if legs:
        meta[name].update(legs=legs, leg_r=LEG_R)
    if boxy:
        meta[name].update(box=[round(w, 3), round(d, 3)])      # collider footprint, prop frame
    print(f"  wrap {name:15s} {w:.2f} x {d:.2f} x {h:.2f} m  r={radius:.2f}  mpu={mpu} up={up}")


LEGGED = {"chair", "school_chair"}
BOXY = {"cardboard_box", "crate", "wet_floor_sign"}
LEG_R = 0.04


def _body_cells(stage, cell=0.06, z_lo=0.04, z_hi=0.30):
    """Centres of the 6 cm cells where the prop's SURFACE passes through the car's body band
    (z_lo..z_hi): what the car hits. A floor tube under 4 cm is climbed (not a leg); a
    seat above 30 cm is driven under. The surface is sampled ~1 point per cm^2 of each
    triangle: a straight tube has vertices only at its two ends."""
    import numpy as np
    xc = UsdGeom.XformCache()
    rng = np.random.default_rng(0)
    band = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(prim)
        pts, cnt, idx = (m.GetPointsAttr().Get(), m.GetFaceVertexCountsAttr().Get(),
                         m.GetFaceVertexIndicesAttr().Get())
        if pts is None or cnt is None or idx is None:
            continue
        M = np.array(xc.GetLocalToWorldTransform(prim)).T
        P = (np.c_[np.array(pts, float), np.ones(len(pts))] @ M.T)[:, :3]
        cnt, idx = np.array(cnt), np.array(idx)
        tri = np.concatenate([np.stack([np.full(c - 2, idx[s]), idx[s + 1:s + c - 1], idx[s + 2:s + c]], 1)
                              for s, c in zip(np.r_[0, np.cumsum(cnt)[:-1]], cnt) if c >= 3])
        A, B, C = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
        n = np.maximum(1, np.ceil(0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1) * 1e4).astype(int))
        k = np.repeat(np.arange(len(tri)), n)
        u, v = rng.random(len(k)), rng.random(len(k))
        f = u + v > 1
        u[f], v[f] = 1 - u[f], 1 - v[f]
        S = A[k] + u[:, None] * (B[k] - A[k]) + v[:, None] * (C[k] - A[k])
        band.append(S[(S[:, 2] >= z_lo) & (S[:, 2] <= z_hi), :2])
    cells = np.unique(np.floor(np.vstack(band) / cell).astype(int), axis=0)
    return [[round((i + 0.5) * cell, 3), round((j + 0.5) * cell, 3)] for i, j in cells]


def _rel(target: Path, base: Path) -> str:
    import os
    return os.path.relpath(target, base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrap-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta, seen = {}, set()
    for name, (src, kind) in PROPS.items():
        if kind == "s3":
            p = SRC / src if a.wrap_only else fetch(src, seen)
        else:
            p = Path(src)
            if name.startswith("person") and p.exists():
                p = bake_skinned(p)
        if p is None or not Path(p).exists():
            print(f"  SKIP {name}: source missing ({src})")
            continue
        try:
            wrap(name, Path(p), meta)
        except Exception as e:          # one bad asset must not sink the set
            print(f"  FAIL {name}: {e}")
    (OUT / "manifest.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(meta)} props -> {OUT}")


if __name__ == "__main__":
    main()
