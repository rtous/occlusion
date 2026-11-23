#!/usr/bin/env bash
# Run the whole pipeline on the sample, crowd and old-movie images + the synthetic evaluation.
# Every step caches its outputs on Drive (occlusion_work/outputs) and skips what is done,
# so after a disconnect just run:  bash setup_colab.sh && bash run_all.sh
set -e
cd "$(dirname "$0")"
I=/content/drive/MyDrive/occlusion_work/images
# samples, COCO groups (paper figures) and the old paper's movie frames (internal comparison)
for B in multihmr sam3d; do export OCC_BODY=$B    # sam3d = main method, multihmr = comparison
for D in $I $I/crowd $I/oldframes; do
  python scripts/run_detect.py --images $D       # M2+M3 -> outputs/modal, outputs/body3d
  python scripts/run_occlusion.py --images $D    # M4    -> outputs/occlusion (cheap, always recomputed)
  python scripts/run_compose.py --images $D      # M5    -> outputs/compose   (backgrounds cached)
done
python scripts/make_synthetic_eval.py || true   # M6 benchmark (exits if already generated)
python scripts/eval.py                          # M6 -> outputs/eval (network outputs cached per image)
done
export OCC_BODY=sam3d
python scripts/make_figures.py                  # paper/figures from the crowd images
V=vlcsnap-2022-04-20-
python scripts/make_figures.py --images $I/oldframes --name movies --intro \
  --qual ${V}20h01m24s874,${V}19h34m02s125,${V}19h56m40s250,${V}20h01m01s279,${V}20h21m21s245,${V}20h01m56s515 \
  --pick ${V}20h01m24s874,${V}19h34m02s125,${V}19h56m40s250    # paper/figures/movies*.jpg
python scripts/make_compare.py    # paper/figures/compare_*.jpg (needs the OCC_BODY=sam3d outputs too)
