"""Milestone 2: modal masks on all images in WORK/images, overlays to WORK/outputs/modal."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from occlusion.config import IMAGES, OUTPUTS  # noqa: E402
from occlusion.modal import ModalSegmenter, overlay  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--images", default=str(IMAGES))
ap.add_argument("--no-sam", action="store_true")
a = ap.parse_args()

out = OUTPUTS / "modal"
out.mkdir(parents=True, exist_ok=True)
seg = ModalSegmenter(use_sam=not a.no_sam)
for f in sorted(Path(a.images).glob("*.jpg")):
    img = cv2.imread(str(f))
    persons = seg(img)
    cv2.imwrite(str(out / f.name), overlay(img, persons))
    np.savez_compressed(out / (f.stem + ".npz"),
                        boxes=np.array([p.box for p in persons]).reshape(-1, 4),
                        masks=np.array([p.mask for p in persons]))
    print(f.name, len(persons), "persons")
print("outputs in", out)
