#!/usr/bin/env bash
# Set up a fresh Colab session. Run from the repo root: bash setup_colab.sh
# Assumes Google Drive is mounted at /content/drive (from the notebook:
#   from google.colab import drive; drive.mount('/content/drive')).
set -e
WORK=/content/drive/MyDrive/occlusion_work
if [ -d /content/drive/MyDrive ]; then mkdir -p "$WORK/weights"; else echo "WARNING: Drive not mounted, using ./work"; fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# (simple-lama-inpainting is NOT used: it pins numpy 1.26, which fails to build on py3.13)
pip install -q ultralytics trimesh

# Body model file: copy from Drive into the repo root if present there.
SMPL=basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
if [ ! -f "$SMPL" ] && [ -f "$WORK/$SMPL" ]; then cp "$WORK/$SMPL" .; fi
[ -f "$SMPL" ] || echo "NOTE: $SMPL not found (needed for the 3D body step)"

python scripts/download_images.py
echo "setup done"
