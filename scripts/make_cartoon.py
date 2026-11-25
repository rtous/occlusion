"""Cartoonization figure: each person is replaced by a Mixamo character (models/*.fbx) posed with
the SAM 3D Body keypoints, rendered with Blender, and composited over the inpainted background
with the occluded parts erased (occlusion masks of the main method).

Needs the cached outputs of run_detect.py / run_compose.py (OCC_BODY=sam3d, the default) and
Blender >= 3.0 (`--blender`). SAM 3D Body is re-run on each image with the cached person boxes
and masks (the caches keep the vertices but not the keypoints); the new bodies are matched to
the cached ones by their vertices.

Writes paper_v2/figures/cartoonization.jpg (one row per image: original | de-occluded meshes |
de-occluded characters) and WORK/outputs/cartoon/<img>.png.
"""
import argparse
import glob
import json
import runpy
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from occlusion.config import IMAGES, OUTPUTS, REPO, out_dir, load_faces  # noqa: E402
from occlusion.body3d import Body  # noqa: E402
from occlusion.render import render_all  # noqa: E402
from occlusion.compose import compose, person_layers  # noqa: E402
from occlusion import occlusion as occ  # noqa: E402

V = "vlcsnap-2022-04-20-"
ap = argparse.ArgumentParser()
ap.add_argument("--images", nargs="+", default=[str(IMAGES / "oldframes" / f"{V}{t}.png") for t in
                                                ("20h01m24s874", "19h34m02s125", "19h56m40s250")] +
                [str(IMAGES / "crowd" / f"{s}.jpg") for s in ("000000397354", "000000363666")])
ap.add_argument("--models", nargs="+", default=[str(REPO / "models" / f"{m}.fbx") for m in
                                                ("elisabeth", "joe", "kate", "remy", "shae")])
ap.add_argument("--max-side", type=int, default=1280, help="render resolution (longest side)")
ap.add_argument("--erase-frac", type=float, default=0.03,
                help="object-occlusion masks are dilated by this fraction of the person height, "
                     "because the character silhouette differs from the body mesh")
ap.add_argument("--blender", default="blender")
ap.add_argument("--out", default=str(REPO / "paper_v2" / "figures" / "cartoonization.jpg"))
a = ap.parse_args()
OUT = OUTPUTS / "cartoon"
OUT.mkdir(parents=True, exist_ok=True)


def mhr70_names():
    """Keypoint names of SAM 3D Body's MHR70 format, read from its metadata module."""
    from occlusion.body3d_sam3d import SAM3D_DIR
    for f in glob.glob(f"{SAM3D_DIR}/**/mhr70*.py", recursive=True):
        info = runpy.run_path(f).get("pose_info", {}).get("keypoint_info")
        if info:
            return [info[i]["name"] for i in sorted(info)]
    raise RuntimeError(f"MHR70 keypoint names not found under {SAM3D_DIR}")


def joints_from_kp(kp, names):
    """MHR70 keypoints (70,3) -> joint dict for blender_mixamo.py."""
    ix = {n: i for i, n in enumerate(names)}
    J = lambda *ns: np.mean([kp[ix[n]] for n in ns], 0).tolist()
    out = dict(neck=J("neck"), head=J("left_ear", "right_ear"), l_eye=J("left_eye"), r_eye=J("right_eye"))
    for s, side in (("l", "left"), ("r", "right")):
        for j in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
            out[f"{s}_{j}"] = J(f"{side}_{j}")
        out[f"{s}_toe"] = J(f"{side}_big_toe", f"{side}_small_toe")
        mid = [n for n in names if n.startswith(side) and "middle" in n]     # base = closest to the wrist
        if mid:
            w = kp[ix[f"{side}_wrist"]]
            out[f"{s}_hand"] = kp[ix[min(mid, key=lambda n: np.linalg.norm(kp[ix[n]] - w))]].tolist()
    return out


def dilate(m, r):
    r = max(int(r), 1)
    return cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0


from occlusion.body3d_sam3d import Sam3DBodyEstimator  # noqa: E402
est = Sam3DBodyEstimator()
names = mhr70_names()
rows = []
for f in map(Path, a.images):
    img = cv2.imread(str(f))
    H, W = img.shape[:2]
    m = np.load(out_dir("modal") / (f.stem + ".npz"))
    modal = m["masks"].reshape(-1, H, W)
    b = np.load(out_dir("body3d") / (f.stem + ".npz"))
    faces = load_faces(b)
    bodies = [Body(v, s) for v, s in zip(b["verts"], b["scores"])]
    depths, _ = render_all(bodies, faces, b["K"], H, W)
    persons = occ.compute(modal, depths)
    bg = cv2.imread(str(out_dir("compose") / (f.stem + "_bg.png")))
    meshes = compose(bg, person_layers(bodies, faces, b["K"], H, W), depths, persons)

    # keypoints: re-run SAM 3D Body with the cached prompts, match to the cached bodies
    fresh, _ = est(img, m["boxes"], modal)
    fv = np.array([x.verts for x in fresh])
    kps = []
    for bd in bodies:
        err = np.abs(fv - bd.verts[None]).mean((1, 2))
        if err.min() > 0.01:
            print(f"warning: {f.name}: cached body reproduced with a mean error of {err.min():.3f} m")
        kps.append(fresh[int(err.argmin())].params["kp3d"])

    s = min(1.0, a.max_side / max(H, W))
    job = dict(width=round(W * s), height=round(H * s), K=(np.diag([s, s, 1]) @ b["K"]).tolist(), persons=[])
    for n, p in enumerate(persons):
        job["persons"].append(dict(model=a.models[(len(rows) + n) % len(a.models)],
                                   joints=joints_from_kp(kps[p.body_idx], names),
                                   out=str(OUT / f"{f.stem}_p{n}.png")))
    jf = OUT / f"{f.stem}.json"
    jf.write_text(json.dumps(job))
    subprocess.run([a.blender, "-b", "--python", str(REPO / "scripts" / "blender_mixamo.py"), "--", str(jf)],
                   check=True, stdout=subprocess.DEVNULL)

    # composite: per pixel the nearest character that is not occluded there
    out = bg.astype(np.float32)
    layers, alphas, dist = [], [], []
    for n, p in enumerate(persons):
        rgba = cv2.resize(cv2.imread(job["persons"][n]["out"], cv2.IMREAD_UNCHANGED), (W, H),
                          interpolation=cv2.INTER_LINEAR)
        al = rgba[..., 3].astype(np.float32) / 255
        ys = np.where(p.amodal.any(1))[0]
        al[dilate(p.occ_object, a.erase_frac * (ys[-1] - ys[0] + 1)) & ~p.modal] = 0
        al[p.occ_human & ~p.modal] = 0
        dp = depths[p.body_idx]
        d = np.where(np.isfinite(dp), dp, np.median(dp[np.isfinite(dp)]))   # outside the mesh: its median
        layers.append(rgba[..., :3].astype(np.float32)); alphas.append(al); dist.append(d)
    if persons:
        dist = np.where(np.array(alphas) > 0.5, np.array(dist), np.inf)
        front = dist.argmin(0)
        none = ~np.isfinite(dist.min(0))
        order = np.argsort([-np.median(d[np.isfinite(d)]) if np.isfinite(d).any() else 0 for d in dist])
        for n in order:                                  # far to near, nearest wins where overlapping
            al = alphas[n] * ((front == n) | none)
            out = out * (1 - al[..., None]) + layers[n] * al[..., None]
    out = out.clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT / (f.stem + ".png")), out)
    rows.append((img, meshes, out))
    print(f.name, len(persons), "characters")

G, Wt = 6, 600                                           # tile width, as in the previous figure
lines = []
for tiles in rows:
    h = round(tiles[0].shape[0] * Wt / tiles[0].shape[1])
    t = [cv2.resize(x, (Wt, h), interpolation=cv2.INTER_AREA) for x in tiles]
    sep = np.full((h, G, 3), 255, np.uint8)
    lines.append(np.hstack([t[0], sep, t[1], sep, t[2]]))
fig = np.vstack(sum([[r, np.full((G, r.shape[1], 3), 255, np.uint8)] for r in lines], [])[:-1])
cv2.imwrite(a.out, fig, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("wrote", a.out)
