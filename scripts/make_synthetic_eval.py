"""Synthetic occlusion benchmark.

For each image in WORK/images/single (one fully visible person, see select_coco.py):
  GT amodal = visible mask (YOLO+SAM) of that person.
  Paste an occluder over part of the person:
    - 'object': an elliptical/rectangular patch cut from another image
    - 'human' : another person's cut-out (from another single image)
  GT occlusion = GT amodal AND occluder alpha.
  Plus a 'clean' copy (no occluder, GT occlusion empty) to measure false positives.
Writes WORK/synth/<id>_<kind>.jpg and .npz (gt_amodal, gt_occ, kind).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from occlusion.config import IMAGES, WORK  # noqa: E402
from occlusion.modal import ModalSegmenter  # noqa: E402

rng = np.random.default_rng(0)
src = sorted((IMAGES / "single").glob("*.jpg"))
out = WORK / "synth"
out.mkdir(parents=True, exist_ok=True)
seg = ModalSegmenter()

# 1. visible mask of the main person in each source image
people = {}
for f in src:
    img = cv2.imread(str(f))
    ps = seg(img)
    if ps:
        p = max(ps, key=lambda p: p.mask.sum())
        people[f] = (img, p.mask)
print(len(people), "source persons")
keys = list(people)


def bbox(m):
    ys, xs = np.where(m)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def object_occluder(H, W, box, donor):
    """Random ellipse/rectangle covering 20-45% of the person's bbox, textured from donor."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    alpha = np.zeros((H, W), np.uint8)
    cx = int(rng.uniform(x0, x1)); cy = int(rng.uniform(y0 + 0.3 * bh, y1))
    ax, ay = int(rng.uniform(0.25, 0.6) * bw), int(rng.uniform(0.12, 0.3) * bh)
    if rng.random() < 0.5:
        cv2.ellipse(alpha, (cx, cy), (ax, ay), 0, 0, 360, 1, -1)
    else:
        cv2.rectangle(alpha, (cx - ax, cy - ay), (cx + ax, cy + ay), 1, -1)
    tex = cv2.resize(donor, (W, H))
    return alpha.astype(bool), tex


def human_occluder(H, W, box, donor_img, donor_mask):
    """Another person's cut-out, scaled to a similar height, shifted sideways to overlap."""
    x0, y0, x1, y1 = box
    dx0, dy0, dx1, dy1 = bbox(donor_mask)
    s = (y1 - y0) * rng.uniform(0.9, 1.15) / (dy1 - dy0)
    crop_m = donor_mask[dy0:dy1, dx0:dx1].astype(np.uint8)
    crop_i = donor_img[dy0:dy1, dx0:dx1]
    nh, nw = max(int((dy1 - dy0) * s), 2), max(int((dx1 - dx0) * s), 2)
    crop_m = cv2.resize(crop_m, (nw, nh), interpolation=cv2.INTER_NEAREST)
    crop_i = cv2.resize(crop_i, (nw, nh))
    shift = rng.uniform(0.3, 0.7) * (x1 - x0) * rng.choice([-1, 1])
    px = int((x0 + x1) / 2 + shift - nw / 2)
    py = int(y1 + rng.uniform(0.0, 0.1) * (y1 - y0) - nh)   # feet a bit lower = in front
    alpha = np.zeros((H, W), bool)
    tex = np.zeros((H, W, 3), np.uint8)
    xs, ys = max(px, 0), max(py, 0)
    xe, ye = min(px + nw, W), min(py + nh, H)
    if xe <= xs or ye <= ys:
        return alpha, tex
    alpha[ys:ye, xs:xe] = crop_m[ys - py:ye - py, xs - px:xe - px] > 0
    tex[ys:ye, xs:xe] = crop_i[ys - py:ye - py, xs - px:xe - px]
    return alpha, tex


n = 0
for i, f in enumerate(keys):
    img, m = people[f]
    H, W = m.shape
    box = bbox(m)
    # clean copy: no occlusion at all (measures false positives from mesh misfit)
    cv2.imwrite(str(out / f"{f.stem}_clean.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    np.savez_compressed(out / f"{f.stem}_clean.npz", gt_amodal=m, gt_occ=np.zeros_like(m),
                        occluder=np.zeros_like(m), kind="clean")
    for kind in ("object", "human"):
        donor = keys[(i + 1 + rng.integers(len(keys) - 1)) % len(keys)]
        if kind == "object":
            alpha, tex = object_occluder(H, W, box, people[donor][0])
        else:
            alpha, tex = human_occluder(H, W, box, *people[donor])
        gt_occ = m & alpha
        ratio = gt_occ.sum() / m.sum()
        if not 0.05 < ratio < 0.6:
            continue
        syn = img.copy()
        syn[alpha] = tex[alpha]
        name = f"{f.stem}_{kind}"
        cv2.imwrite(str(out / f"{name}.jpg"), syn, [cv2.IMWRITE_JPEG_QUALITY, 95])
        np.savez_compressed(out / f"{name}.npz", gt_amodal=m, gt_occ=gt_occ,
                            occluder=alpha, kind=kind)
        n += 1
print(n, "synthetic images in", out)
