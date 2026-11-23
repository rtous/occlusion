"""Cartoonization figure: cartoon characters (models/*.blend, SMPL-X rig) posed with the
Multi-HMR SMPL-X parameters, rendered with Blender, and composited over the inpainted
background with the occluded parts erased.

Needs the cached outputs of run_detect.py / run_compose.py (Multi-HMR, default OCC_BODY) and
Blender (apt-get install blender). Multi-HMR is re-run on each image to recover the pose
parameters (the caches only keep vertices); bodies are matched to the cached ones by vertices.

Writes paper/figures/cartoon.jpg (one row per image: original | de-occluded cartoon) and
WORK/outputs/cartoon/<img>.png.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from occlusion.config import IMAGES, OUTPUTS, REPO, load_faces  # noqa: E402
from occlusion.body3d import Body, BodyEstimator  # noqa: E402
from occlusion.render import render_all  # noqa: E402
from occlusion import occlusion as occ  # noqa: E402

V = "vlcsnap-2022-04-20-"
ap = argparse.ArgumentParser()
ap.add_argument("--images", nargs="+", default=[str(IMAGES / "oldframes" / f"{V}20h01m24s874.png"),
                                                 str(IMAGES / "crowd" / "000000363666.jpg")])
ap.add_argument("--models", nargs="+", default=sorted(str(p) for p in (REPO / "models").glob("*.blend")))
ap.add_argument("--max-side", type=int, default=1280, help="render resolution (longest side)")
ap.add_argument("--erase-frac", type=float, default=0.03,
                help="object-occlusion masks are dilated by this fraction of the person height, "
                     "because the cartoon silhouette differs from the SMPL-X one")
ap.add_argument("--blender", default="blender")
ap.add_argument("--out", default=str(REPO / "paper" / "figures" / "cartoon.jpg"))
a = ap.parse_args()
OUT = OUTPUTS / "cartoon"
OUT.mkdir(parents=True, exist_ok=True)
hmr = None


def dilate(m, r):
    r = max(int(r), 1)
    return cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0


rows = []
for f in map(Path, a.images):
    img = cv2.imread(str(f))
    H, W = img.shape[:2]
    modal = np.load(OUTPUTS / "modal" / (f.stem + ".npz"))["masks"].reshape(-1, H, W)
    b = np.load(OUTPUTS / "body3d" / (f.stem + ".npz"))
    bodies = [Body(v, s) for v, s in zip(b["verts"], b["scores"])]
    depths, _ = render_all(bodies, load_faces(b), b["K"], H, W)
    persons = occ.compute(modal, depths)
    bg = cv2.imread(str(OUTPUTS / "compose" / (f.stem + "_bg.png")))

    # pose parameters: re-run Multi-HMR and match to the cached bodies
    if hmr is None:
        hmr = BodyEstimator()
    fresh, _ = hmr(img)
    fv = np.array([x.verts for x in fresh])
    params = [fresh[int(np.argmin(np.abs(fv - bd.verts[None]).mean((1, 2))))].params for bd in bodies]

    s = min(1.0, a.max_side / max(H, W))
    w, h = round(W * s), round(H * s)
    K = (np.diag([s, s, 1]) @ b["K"]).tolist()
    job = dict(width=w, height=h, K=K, persons=[])
    for n, p in enumerate(persons):
        q = params[p.body_idx]
        job["persons"].append(dict(model=a.models[n % len(a.models)], rotvec=q["rotvec"].tolist(),
                                   pelvis=q["pelvis"].tolist(), out=str(OUT / f"{f.stem}_p{n}.png")))
    jf = OUT / f"{f.stem}.json"
    jf.write_text(json.dumps(job))
    subprocess.run([a.blender, "-b", "--python", str(REPO / "scripts" / "blender_cartoon.py"), "--", str(jf)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # composite: per pixel the nearest character that is not occluded there
    out = bg.astype(np.float32)
    layers, alphas, dist = [], [], []
    for n, p in enumerate(persons):
        rgba = cv2.resize(cv2.imread(job["persons"][n]["out"], cv2.IMREAD_UNCHANGED), (W, H),
                          interpolation=cv2.INTER_LINEAR)
        al = rgba[..., 3].astype(np.float32) / 255
        ys = np.where(p.amodal.any(1))[0]
        hidden = dilate(p.occ_object, a.erase_frac * (ys[-1] - ys[0] + 1)) & ~p.modal
        al[hidden] = 0
        d = np.where(np.isfinite(depths[p.body_idx]), depths[p.body_idx], params[p.body_idx]["pelvis"][2])
        layers.append(rgba[..., :3].astype(np.float32)); alphas.append(al); dist.append(d)
    if persons:
        dist = np.where(np.array(alphas) > 0.5, np.array(dist), np.inf)
        order = np.argsort(-np.array([np.median(d[np.isfinite(d)]) if np.isfinite(d).any() else 0 for d in dist]))
        front = dist.argmin(0)
        for n in order:                                  # far to near, nearest wins where overlapping
            al = alphas[n] * ((front == n) | (dist.min(0) == np.inf))
            out = out * (1 - al[..., None]) + layers[n] * al[..., None]
    out = out.clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT / (f.stem + ".png")), out)
    rows.append((img, out))
    print(f.name, len(persons), "characters")

hh = 360
row_imgs = []
for i, o in rows:
    r = [cv2.resize(x, (round(x.shape[1] * hh / x.shape[0]), hh), interpolation=cv2.INTER_AREA) for x in (i, o)]
    row_imgs.append(np.hstack([r[0], np.full((hh, 6, 3), 255, np.uint8), r[1]]))
wmax = 1600
row_imgs = [cv2.resize(r, (wmax, round(r.shape[0] * wmax / r.shape[1]))) for r in row_imgs]
fig = np.vstack(sum([[r, np.full((6, wmax, 3), 255, np.uint8)] for r in row_imgs], [])[:-1])
cv2.imwrite(a.out, fig, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("wrote", a.out)
