"""Milestones 2+3: modal masks (YOLO11 + SAM 2) and 3D bodies (Multi-HMR), with the
cross-model selection of occlusion.select() and the SAM fallback (pipeline.detect).

Writes WORK/outputs/modal/<img>.{npz,jpg} (masks, boxes, scores) and
WORK/outputs/body3d/<img>.{npz,jpg} (K, verts, scores). Cached: use --force to recompute.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from occlusion.config import IMAGES, OUTPUTS, list_images  # noqa: E402
from occlusion.modal import overlay  # noqa: E402
from occlusion.render import overlay_meshes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--images", default=str(IMAGES))
ap.add_argument("--force", action="store_true", help="recompute even if cached")
ap.add_argument("--only", default=None, help="process a single file name")
a = ap.parse_args()

om, ob = OUTPUTS / "modal", OUTPUTS / "body3d"
om.mkdir(parents=True, exist_ok=True)
ob.mkdir(parents=True, exist_ok=True)
files = [f for f in list_images(a.images) if not a.only or f.name == a.only]
files = [f for f in files if a.force or not ((om / (f.stem + ".npz")).exists() and (ob / (f.stem + ".npz")).exists())]
if files:
    from occlusion.pipeline import Pipeline
    pipe = Pipeline()
for f in files:
    img = cv2.imread(str(f))
    t = time.time()
    r = pipe(img)
    ids = np.where(np.isfinite(r.depths).any(0), r.depths.argmin(0), -1) if len(r.depths) else \
        np.full(img.shape[:2], -1)
    boxes = [(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
             for ys, xs in (np.nonzero(m) for m in r.modal)]
    cv2.imwrite(str(om / (f.stem + ".jpg")), overlay(img, [type("P", (), {"mask": m}) for m in r.modal]))
    cv2.imwrite(str(ob / (f.stem + ".jpg")), np.hstack([img, overlay_meshes(img, ids)]))
    np.savez_compressed(om / (f.stem + ".npz"), masks=r.modal, scores=r.modal_scores,
                        boxes=np.array(boxes, np.float32).reshape(-1, 4))
    np.savez_compressed(ob / (f.stem + ".npz"), K=r.K,
                        verts=np.array([b.verts for b in r.bodies], np.float32).reshape(-1, 10475, 3),
                        scores=np.array([b.score for b in r.bodies]))
    print(f"{f.name}: {len(r.modal)} masks, {len(r.bodies)} bodies, {time.time() - t:.1f}s")
print("outputs in", om, ob)
