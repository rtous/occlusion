"""Milestone 5: background inpainting (LaMa) + de-occlusion composite.

Needs cached outputs of run_modal.py, run_body3d.py, run_occlusion.py.
Writes WORK/outputs/compose/<img>.jpg:
  original | inpainted background | meshes without occlusion handling | de-occluded meshes
and <img>_bg.png (the inpainted background).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from occlusion.config import IMAGES, OUTPUTS, REPO  # noqa: E402
from occlusion.body3d import Body  # noqa: E402
from occlusion.render import render_all  # noqa: E402
from occlusion import occlusion as occ  # noqa: E402
from occlusion.inpaint import Inpainter, background_mask  # noqa: E402
from occlusion.compose import compose, person_layers  # noqa: E402

faces = np.load(REPO / "SMPLX_NEUTRAL.npz", allow_pickle=True)["f"].astype(np.int64)
out = OUTPUTS / "compose"
out.mkdir(parents=True, exist_ok=True)
inp = Inpainter()
for f in sorted(IMAGES.glob("*.jpg")):
    fm, fb = OUTPUTS / "modal" / (f.stem + ".npz"), OUTPUTS / "body3d" / (f.stem + ".npz")
    if not (fm.exists() and fb.exists()):
        print("skip", f.name)
        continue
    img = cv2.imread(str(f))
    h, w = img.shape[:2]
    modal = np.load(fm)["masks"].reshape(-1, h, w)
    b = np.load(fb)
    bodies = [Body(v, s) for v, s in zip(b["verts"], b["scores"])]
    depths, _ = render_all(bodies, faces, b["K"], h, w)
    persons = occ.compute(modal, depths)

    bg_file = out / (f.stem + "_bg.png")
    if bg_file.exists():
        bg = cv2.imread(str(bg_file))
    else:
        bg = inp(img, background_mask(modal, img.shape))
        cv2.imwrite(str(bg_file), bg)
    layers = person_layers(bodies, faces, b["K"], h, w)
    panel = np.hstack([img, bg, compose(bg, layers, depths, persons, handle_occlusion=False),
                       compose(bg, layers, depths, persons)])
    s = 1600 / panel.shape[1]
    cv2.imwrite(str(out / f.name), cv2.resize(panel, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    print(f.name, "done")
print("outputs in", out)
