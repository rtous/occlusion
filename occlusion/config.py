"""Shared paths and device selection.

Work files (weights, checkpoints, outputs we don't want to lose) live on
Google Drive when it is mounted; otherwise in ./work (gitignored).
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_DRIVE = Path("/content/drive/MyDrive")
WORK = Path(os.environ.get(
    "OCC_WORK", _DRIVE / "occlusion_work" if _DRIVE.exists() else REPO / "work"))
WEIGHTS = WORK / "weights"
OUTPUTS = WORK / "outputs"
IMAGES = WORK / "images"

# Body model file (SMPL neutral), looked up in the repo root, then on Drive.
SMPL_NAME = "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"


def smpl_path():
    for p in (REPO / SMPL_NAME, WORK / SMPL_NAME, WEIGHTS / SMPL_NAME):
        if p.exists():
            return p
    return None


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


for _d in (WEIGHTS, OUTPUTS, IMAGES):
    _d.mkdir(parents=True, exist_ok=True)
