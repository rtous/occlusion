#!/usr/bin/env bash
# Run the whole pipeline on the sample, crowd and old-movie images + the synthetic evaluation.
# Every step caches its outputs on Drive (occlusion_work/outputs) and skips what is done,
# so after a disconnect just run:  bash setup_colab.sh && bash run_all.sh
set -e
cd "$(dirname "$0")"
I=/content/drive/MyDrive/occlusion_work/images
# samples, COCO groups (paper figures) and the old paper's movie frames (internal comparison)
for D in $I $I/crowd $I/oldframes; do
  python scripts/run_detect.py --images $D       # M2+M3 -> outputs/modal, outputs/body3d
  python scripts/run_occlusion.py --images $D    # M4    -> outputs/occlusion (cheap, always recomputed)
  python scripts/run_compose.py --images $D      # M5    -> outputs/compose   (backgrounds cached)
done
python scripts/make_synthetic_eval.py || true   # M6 benchmark (exits if already generated)
python scripts/eval.py                          # M6 -> outputs/eval (network outputs cached per image)
python scripts/make_figures.py                  # paper/figures from the crowd images
