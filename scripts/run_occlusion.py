"""Milestone 4: occlusion masks from cached run_detect.py outputs.

Writes WORK/outputs/occlusion/<img>.jpg  (original | modal | amodal | occlusion: red=object, blue=human)
and <img>.npz with per-person masks.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from occlusion.config import IMAGES, OUTPUTS, REPO, list_images  # noqa: E402
from occlusion.render import render_all  # noqa: E402
from occlusion import occlusion as occ  # noqa: E402
from occlusion.body3d import Body  # noqa: E402

faces = np.load(REPO / "SMPLX_NEUTRAL.npz", allow_pickle=True)["f"].astype(np.int64)
out = OUTPUTS / "occlusion"
out.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--images", default=str(IMAGES))
a = ap.parse_args()
for f in list_images(a.images):
    fm, fb = OUTPUTS / "modal" / (f.stem + ".npz"), OUTPUTS / "body3d" / (f.stem + ".npz")
    if not (fm.exists() and fb.exists()):
        print("skip", f.name, "(run run_detect.py first)")
        continue
    img = cv2.imread(str(f))
    h, w = img.shape[:2]
    modal = np.load(fm)["masks"].reshape(-1, h, w)
    b = np.load(fb)
    bodies = [Body(v, s) for v, s in zip(b["verts"], b["scores"])]
    depths, _ = render_all(bodies, faces, b["K"], h, w)
    persons = occ.compute(modal, depths)
    panel = np.hstack([img,
                       occ.masks_panel(img.shape, persons, "modal"),
                       occ.masks_panel(img.shape, persons, "amodal"),
                       occ.occlusion_overlay(img, persons)])
    s = 1600 / panel.shape[1]
    cv2.imwrite(str(out / (f.stem + ".jpg")), cv2.resize(panel, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    np.savez_compressed(out / (f.stem + ".npz"),
                        amodal=np.array([p.amodal for p in persons]),
                        modal=np.array([p.modal for p in persons]),
                        occ_human=np.array([p.occ_human for p in persons]),
                        occ_object=np.array([p.occ_object for p in persons]))
    print(f"{f.name}: {len(modal)} masks, {len(bodies)} meshes, {len(persons)} matched, occ ratios",
          [round(p.occ_ratio, 2) for p in persons])
print("outputs in", out)
