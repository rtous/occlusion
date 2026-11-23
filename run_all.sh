#!/usr/bin/env bash
# Run the whole pipeline on the sample images + the synthetic evaluation.
# Every step caches its outputs on Drive (occlusion_work/outputs) and skips what is done,
# so after a disconnect just run:  bash setup_colab.sh && bash run_all.sh
set -e
cd "$(dirname "$0")"
python scripts/run_modal.py        # M2  -> outputs/modal
python scripts/run_body3d.py       # M3  -> outputs/body3d
python scripts/run_occlusion.py    # M4  -> outputs/occlusion   (cheap, always recomputed)
python scripts/run_compose.py      # M5  -> outputs/compose     (backgrounds cached)
# old paper's movie frames, for side-by-side comparison with the old figures
O=/content/drive/MyDrive/occlusion_work/images/oldframes
python scripts/run_modal.py --images $O && python scripts/run_body3d.py --images $O
python scripts/run_occlusion.py --images $O && python scripts/run_compose.py --images $O
python scripts/make_synthetic_eval.py || true   # M6 benchmark (exits if already generated)
python scripts/eval.py             # M6  -> outputs/eval (network outputs cached per image)
