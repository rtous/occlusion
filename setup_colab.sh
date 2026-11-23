#!/usr/bin/env bash
# Set up a fresh Colab session. Run from the repo root: bash setup_colab.sh
# Mount Drive first (notebook cell): from google.colab import drive; drive.mount('/content/drive')
set -e
REPO=$(cd "$(dirname "$0")" && pwd)
WORK=/content/drive/MyDrive/occlusion_work
[ -d /content/drive/MyDrive ] || { echo "WARNING: Drive not mounted, using $REPO/work"; WORK=$REPO/work; }
W=$WORK/weights
mkdir -p "$W/multihmr"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# simple-lama-inpainting is NOT used: it pins numpy 1.26, which fails to build on py3.13
pip install -q ultralytics trimesh einops roma smplx pyrender scipy

# --- Body model files (licensed, gitignored): kept on Drive, copied to the repo root ---
for f in basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl SMPLX_NEUTRAL.npz; do
  if [ ! -f "$REPO/$f" ] && [ -f "$WORK/$f" ]; then echo "copying $f from Drive"; cp "$WORK/$f" "$REPO/"; fi
  if [ -f "$REPO/$f" ] && [ ! -f "$WORK/$f" ]; then echo "backing up $f to Drive"; cp "$REPO/$f" "$WORK/"; fi
  [ -f "$REPO/$f" ] || echo "NOTE: $f not found (put it in $WORK)"
done

# --- Multi-HMR (3D bodies): code in /content/third_party, weights on Drive ---
MH=/content/third_party/multi-hmr
[ -d $MH ] || git clone -q --depth 1 https://github.com/naver/multi-hmr.git $MH
[ -s $W/multihmr/multiHMR_672_L.pt ] || wget -q -O $W/multihmr/multiHMR_672_L.pt \
  https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multiHMR_672_L.pt
[ -s $W/multihmr/smpl_mean_params.npz ] || wget -q -O $W/multihmr/smpl_mean_params.npz \
  "https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4"
# Multi-HMR expects ./models/{smplx/SMPLX_NEUTRAL.npz, smpl_mean_params.npz, multiHMR/*.pt}
mkdir -p $MH/models/smplx
ln -sfn $W/multihmr $MH/models/multiHMR
ln -sf $W/multihmr/smpl_mean_params.npz $MH/models/smpl_mean_params.npz
ln -sf $REPO/SMPLX_NEUTRAL.npz $MH/models/smplx/SMPLX_NEUTRAL.npz

# --- SAM 3D Body (optional alternative body model, OCC_BODY=sam3d): code + gated checkpoint ---
# Checkpoint access: request it at https://huggingface.co/facebook/sam-3d-body-vith, then run
# `hf auth login` once in the session; the checkpoint is then cached on Drive.
S3=/content/third_party/sam-3d-body
[ -d $S3 ] || git clone -q --depth 1 https://github.com/facebookresearch/sam-3d-body.git $S3
pip install -q pytorch-lightning yacs timm roma omegaconf braceexpand huggingface_hub
if [ ! -s $W/sam-3d-body-vith/model.ckpt ]; then
  hf download facebook/sam-3d-body-vith --local-dir $W/sam-3d-body-vith >/dev/null 2>&1 \
    && echo "SAM 3D Body checkpoint cached on Drive" \
    || echo "NOTE: SAM 3D Body checkpoint not available (gated; see setup_colab.sh). Multi-HMR still works."
fi

# --- Images: samples, COCO crowd (qualitative) and single-person (synthetic eval) ---
mkdir -p "$WORK/coco"
if [ ! -f "$WORK/coco/person_keypoints_val2017.json" ]; then
  wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O /tmp/coco_ann.zip
  unzip -q -j -o /tmp/coco_ann.zip annotations/person_keypoints_val2017.json -d "$WORK/coco"; rm /tmp/coco_ann.zip
fi
python "$REPO/scripts/download_images.py"
python "$REPO/scripts/download_images.py" coco_crowd crowd
python "$REPO/scripts/download_images.py" coco_single single
bash "$REPO/scripts/fetch_old_frames.sh"   # old paper's 30 movie frames (internal comparison only)
python -c "import torch; print('GPU:', torch.cuda.is_available())"
echo "setup done"
