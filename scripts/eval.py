"""Evaluate occlusion masks on the synthetic benchmark (make_synthetic_eval.py).

For each synthetic image the target person is the predicted person whose visible mask
best overlaps the GT visible region (gt_amodal & ~occluder). Metrics:
  occ IoU      : IoU(predicted occlusion mask, GT occlusion mask)
  amodal IoU   : IoU(projected mesh silhouette, GT amodal)
  label acc    : fraction of correctly-labelled pixels inside pred∩GT occlusion
                 (human for pasted people, object for pasted patches)
A missed person (no match) counts as IoU 0.
Per-image results: WORK/outputs/eval/results.csv; summary printed and saved to summary.txt.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from occlusion.config import WORK, OUTPUTS  # noqa: E402
from occlusion.body3d import Body  # noqa: E402
from occlusion.render import render_all  # noqa: E402
from occlusion import occlusion as occ  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--open-frac", type=float, default=0.01)
ap.add_argument("--dilate-frac", type=float, default=0.03)
ap.add_argument("--tag", default="", help="suffix for summary file")
ap.add_argument("--cached-only", action="store_true", help="only score images whose network outputs are cached")
a = ap.parse_args()


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


out = OUTPUTS / "eval"
cache = out / "cache"          # raw network outputs, so parameters can be re-scored quickly
cache.mkdir(parents=True, exist_ok=True)
pipe = None
faces = None
rows = []
for f in sorted((WORK / "synth").glob("*.jpg")):
    gt = np.load(f.with_suffix(".npz"))
    img = cv2.imread(str(f))
    h, w = img.shape[:2]
    cf = cache / (f.stem + ".npz")
    if not cf.exists() and a.cached_only:
        continue
    if not cf.exists():
        if pipe is None:
            from occlusion.pipeline import Pipeline
            pipe = Pipeline()
        r = pipe(img)
        np.savez_compressed(cf, modal=r.modal, K=r.K, verts=np.array([b.verts for b in r.bodies],
                            np.float32).reshape(-1, 10475, 3), scores=np.array([b.score for b in r.bodies]))
    c = np.load(cf)
    if faces is None:
        from occlusion.config import REPO
        faces = np.load(REPO / "SMPLX_NEUTRAL.npz", allow_pickle=True)["f"].astype(np.int64)
    depths, _ = render_all([Body(v, s) for v, s in zip(c["verts"], c["scores"])], faces, c["K"], h, w)
    res = type("R", (), {})()
    res.persons = occ.compute(c["modal"].reshape(-1, h, w), depths,
                             dilate_frac=a.dilate_frac, open_frac=a.open_frac)
    vis = gt["gt_amodal"] & ~gt["occluder"]
    kind = str(gt["kind"])
    row = dict(image=f.name, kind=kind, n_pred=len(res.persons), occ_iou=0.0 if gt["gt_occ"].any() else np.nan, fp_ratio=np.nan,
               amodal_iou=0.0, label_acc=np.nan, gt_ratio=gt["gt_occ"].sum() / gt["gt_amodal"].sum())
    if res.persons:
        p = max(res.persons, key=lambda p: (p.modal & vis).sum())
        if (p.modal & vis).sum() > 0:
            row["occ_iou"] = iou(p.occ, gt["gt_occ"]) if gt["gt_occ"].any() else np.nan
            row["fp_ratio"] = (p.occ & ~gt["gt_occ"]).sum() / gt["gt_amodal"].sum()
            row["amodal_iou"] = iou(p.amodal, gt["gt_amodal"])
            both = p.occ & gt["gt_occ"]
            if both.sum():
                good = p.occ_human if kind == "human" else p.occ_object
                row["label_acc"] = (good & both).sum() / both.sum()
            panel = np.hstack([img, occ.occlusion_overlay(img, [p]),
                               np.dstack([gt["gt_occ"] * 255] * 3).astype(np.uint8)])
            cv2.imwrite(str(out / f.name), cv2.resize(panel, None, fx=0.5, fy=0.5))
    rows.append(row)
    print(f"{f.name}: occ IoU {row['occ_iou']:.2f}  amodal IoU {row['amodal_iou']:.2f}  label {row['label_acc']:.2f}")

with open(out / "results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
lines = [f"open_frac={a.open_frac} dilate_frac={a.dilate_frac}"]
for kind in ("object", "human", "clean"):
    R = [r for r in rows if r["kind"] == kind]
    if not R:
        continue
    la = [r["label_acc"] for r in R if not np.isnan(r["label_acc"])]
    lines.append(f"{kind:7s} n={len(R):3d}  occ IoU {np.nanmean([r['occ_iou'] for r in R] + [np.nan]):.3f}  "
                 f"false-pos ratio {np.nanmean([r['fp_ratio'] for r in R]):.3f}  "
                 f"amodal IoU {np.mean([r['amodal_iou'] for r in R]):.3f}  "
                 f"label acc {np.mean(la) if la else float('nan'):.3f}")
print("\n".join(lines))
(out / f"summary{a.tag}.txt").write_text("\n".join(lines) + "\n")
