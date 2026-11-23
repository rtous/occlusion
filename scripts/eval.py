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
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from occlusion.config import WORK, OUTPUTS  # noqa: E402
from occlusion.pipeline import Pipeline  # noqa: E402
from occlusion import occlusion as occ  # noqa: E402


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


out = OUTPUTS / "eval"
out.mkdir(parents=True, exist_ok=True)
pipe = Pipeline()
rows = []
for f in sorted((WORK / "synth").glob("*.jpg")):
    gt = np.load(f.with_suffix(".npz"))
    img = cv2.imread(str(f))
    res = pipe(img)
    vis = gt["gt_amodal"] & ~gt["occluder"]
    kind = str(gt["kind"])
    row = dict(image=f.name, kind=kind, n_pred=len(res.persons), occ_iou=0.0,
               amodal_iou=0.0, label_acc=np.nan, gt_ratio=gt["gt_occ"].sum() / gt["gt_amodal"].sum())
    if res.persons:
        p = max(res.persons, key=lambda p: (p.modal & vis).sum())
        if (p.modal & vis).sum() > 0:
            row["occ_iou"] = iou(p.occ, gt["gt_occ"])
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
lines = []
for kind in ("object", "human", "all"):
    R = [r for r in rows if kind == "all" or r["kind"] == kind]
    if not R:
        continue
    la = [r["label_acc"] for r in R if not np.isnan(r["label_acc"])]
    lines.append(f"{kind:7s} n={len(R):3d}  occ IoU {np.mean([r['occ_iou'] for r in R]):.3f}  "
                 f"amodal IoU {np.mean([r['amodal_iou'] for r in R]):.3f}  "
                 f"label acc {np.mean(la) if la else float('nan'):.3f}")
print("\n".join(lines))
(out / "summary.txt").write_text("\n".join(lines) + "\n")
