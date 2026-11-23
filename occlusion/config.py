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


def list_images(folder):
    """All .jpg/.jpeg/.png files in a folder (non-recursive), sorted."""
    from pathlib import Path
    return sorted(p for p in Path(folder).iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


# Body model: "sam3d" (SAM 3D Body, default, main method) or "multihmr" (comparison).
# Multi-HMR writes to outputs/{modal,body3d,...}, SAM 3D Body to outputs/{modal,body3d,...}_sam3d,
# so results of both can be reported.
BODY = os.environ.get("OCC_BODY", "sam3d")


def out_dir(name):
    return OUTPUTS / (name if BODY == "multihmr" else f"{name}_{BODY}")


def load_faces(npz=None):
    """Mesh faces stored in a body cache (SAM 3D Body / MHR); SMPL-X faces otherwise."""
    import numpy as np
    if npz is not None and "faces" in npz:
        return npz["faces"].astype(np.int64)
    return np.load(REPO / "SMPLX_NEUTRAL.npz", allow_pickle=True)["f"].astype(np.int64)
