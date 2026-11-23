"""Milestone 3: Multi-HMR meshes rendered over each image -> WORK/outputs/body3d."""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from occlusion.config import IMAGES, OUTPUTS, list_images  # noqa: E402
from occlusion.body3d import BodyEstimator  # noqa: E402
from occlusion.render import render_all, overlay_meshes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--images", default=str(IMAGES))
ap.add_argument("--force", action="store_true", help="recompute even if cached")
ap.add_argument("--only", default=None, help="process a single file name")
a = ap.parse_args()

out = OUTPUTS / "body3d"
out.mkdir(parents=True, exist_ok=True)
est = BodyEstimator()
files = list_images(a.images)
if a.only:
    files = [f for f in files if f.name == a.only]
for f in files:
    if not a.force and (out / (f.stem + ".npz")).exists():
        continue
    img = cv2.imread(str(f))
    h, w = img.shape[:2]
    t = time.time()
    bodies, K = est(img)
    t1 = time.time()
    depths, ids = render_all(bodies, est.faces, K, h, w)
    cv2.imwrite(str(out / (f.stem + ".jpg")), np.hstack([img, overlay_meshes(img, ids)]))
    np.savez_compressed(out / (f.stem + ".npz"), K=K,
                        verts=np.array([b.verts for b in bodies], np.float32),
                        scores=np.array([b.score for b in bodies]))
    print(f"{f.name}: {len(bodies)} bodies, hmr {t1 - t:.1f}s, render {time.time() - t1:.1f}s")
print("outputs in", out)
