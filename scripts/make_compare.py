"""Side-by-side comparison of the two body models (Multi-HMR vs SAM 3D Body) from cached
outputs (run_detect.py with OCC_BODY unset and OCC_BODY=sam3d, run_compose.py for both).

Writes paper/figures/compare_occlusion.jpg
  original | Multi-HMR meshes | Multi-HMR occlusion | SAM 3D Body meshes | SAM 3D Body occlusion
and paper/figures/compare_deocclusion.jpg
  original | Multi-HMR de-occluded | SAM 3D Body de-occluded
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from occlusion.config import IMAGES, OUTPUTS, REPO, load_faces  # noqa: E402
from occlusion.body3d import Body  # noqa: E402
from occlusion.render import render_all, overlay_meshes  # noqa: E402
from occlusion import occlusion as occ  # noqa: E402
from occlusion.compose import compose, person_layers  # noqa: E402

V = "vlcsnap-2022-04-20-"
DEF_OCC = [f"oldframes/{V}20h01m24s874.png", "crowd/000000363666.jpg", "crowd/000000397354.jpg",
           f"oldframes/{V}19h56m40s250.png", "crowd/000000281759.jpg", f"oldframes/{V}20h01m01s279.png"]
DEF_DEOCC = [f"oldframes/{V}20h01m24s874.png", "crowd/000000363666.jpg", "crowd/000000397354.jpg"]
ap = argparse.ArgumentParser()
ap.add_argument("--occ", nargs="+", default=DEF_OCC, help="images (relative to WORK/images)")
ap.add_argument("--deocc", nargs="+", default=DEF_DEOCC)
ap.add_argument("--out", default=str(REPO / "paper" / "figures"))
a = ap.parse_args()
VARIANTS = {"multihmr": "", "sam3d": "_sam3d"}


def load(rel, suffix):
    f = IMAGES / rel
    img = cv2.imread(str(f))
    h, w = img.shape[:2]
    modal = np.load(OUTPUTS / f"modal{suffix}" / (f.stem + ".npz"))["masks"].reshape(-1, h, w)
    b = np.load(OUTPUTS / f"body3d{suffix}" / (f.stem + ".npz"))
    faces = load_faces(b)
    bodies = [Body(v, s) for v, s in zip(b["verts"], b["scores"])]
    depths, ids = render_all(bodies, faces, b["K"], h, w)
    persons = occ.compute(modal, depths)
    bg = cv2.imread(str(OUTPUTS / f"compose{suffix}" / (f.stem + "_bg.png")))
    layers = person_layers(bodies, faces, b["K"], h, w)
    return dict(img=img, ids=ids, persons=persons, deocc=compose(bg, layers, depths, persons))


def row(panels, height=300, gap=6):
    ps = [cv2.resize(p, (round(p.shape[1] * height / p.shape[0]), height), interpolation=cv2.INTER_AREA)
          for p in panels]
    return np.hstack(sum([[p, np.full((height, gap, 3), 255, np.uint8)] for p in ps], [])[:-1])


def stack(rows, width=2400, gap=6):
    rs = [cv2.resize(r, (width, round(r.shape[0] * width / r.shape[1])), interpolation=cv2.INTER_AREA)
          for r in rows]
    return np.vstack(sum([[r, np.full((gap, width, 3), 255, np.uint8)] for r in rs], [])[:-1])


def save(name, im):
    cv2.imwrite(str(Path(a.out) / name), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", Path(a.out) / name)


rows = []
for rel in a.occ:
    r = {k: load(rel, s) for k, s in VARIANTS.items()}
    img = r["multihmr"]["img"]
    p = [img]
    for k in VARIANTS:
        p += [overlay_meshes(img, r[k]["ids"]), occ.occlusion_overlay(img, r[k]["persons"])]
    rows.append(row(p))
save("compare_occlusion.jpg", stack(rows))

rows = []
for rel in a.deocc:
    r = {k: load(rel, s) for k, s in VARIANTS.items()}
    rows.append(row([r["multihmr"]["img"], r["multihmr"]["deocc"], r["sam3d"]["deocc"]]))
save("compare_deocclusion.jpg", stack(rows, width=1800))
