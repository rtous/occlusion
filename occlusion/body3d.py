"""3D bodies (SMPL-X meshes in camera space) with Multi-HMR, single forward pass.

Requires setup_colab.sh (clones Multi-HMR into /content/third_party/multi-hmr and
links the checkpoint + SMPLX_NEUTRAL.npz into its ./models dir).
"""
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .config import device

MULTIHMR_DIR = os.environ.get("MULTIHMR_DIR", "/content/third_party/multi-hmr")
MODEL_NAME = "multiHMR_672_L"


@dataclass
class Body:
    verts: np.ndarray    # (V,3) camera coords (OpenCV: x right, y down, z forward), metres
    score: float


@contextmanager
def _in_multihmr():
    """Multi-HMR uses relative paths (./models/...) and top-level imports."""
    old = os.getcwd()
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if MULTIHMR_DIR not in sys.path:
        sys.path.insert(0, MULTIHMR_DIR)
    os.chdir(MULTIHMR_DIR)
    try:
        yield
    finally:
        os.chdir(old)


class BodyEstimator:
    def __init__(self, model_name=MODEL_NAME, fov=60.0):
        self.dev = torch.device(device())
        self.fov = fov
        with _in_multihmr():
            from model import Model
            ckpt = torch.load(f"models/multiHMR/{model_name}.pt",
                              map_location=self.dev, weights_only=False)
            kw = dict(vars(ckpt["args"]))
            kw["type"] = ckpt["args"].train_return_type
            kw["img_size"] = ckpt["args"].img_size[0]
            self.model = Model(**kw).to(self.dev).eval()
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            self.faces = np.asarray(self.model.smpl_layer["neutral_10"].bm_x.faces,
                                    dtype=np.int64)
        self.size = self.model.img_size

    def __call__(self, img_bgr, det_thresh=0.1):
        """Returns (bodies, K) with K the 3x3 intrinsics for the ORIGINAL image."""
        from utils import normalize_rgb, get_focalLength_from_fieldOfView
        h, w = img_bgr.shape[:2]
        S = self.size
        s = S / max(h, w)
        nh, nw = round(h * s), round(w * s)
        ox, oy = (S - nw) // 2, (S - nh) // 2          # centered zero padding
        sq = np.zeros((S, S, 3), np.uint8)
        sq[oy:oy + nh, ox:ox + nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        x = torch.from_numpy(normalize_rgb(sq)).unsqueeze(0).to(self.dev)

        f = get_focalLength_from_fieldOfView(fov=self.fov, img_size=S)
        K = torch.tensor([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1]],
                         dtype=torch.float32, device=self.dev).unsqueeze(0)
        with torch.no_grad(), torch.autocast(self.dev.type, enabled=self.dev.type == "cuda"):
            humans = self.model(x, is_training=False, nms_kernel_size=3,
                                det_thresh=det_thresh, K=K)
        name = "verts_smplx" if humans and "verts_smplx" in humans[0] else "v3d"
        bodies = [Body(hm[name].float().cpu().numpy(), float(hm["scores"]))
                  for hm in humans]
        # intrinsics in original-image pixels: u_orig = (u_sq - ox) / s
        Ko = np.array([[f / s, 0, (S / 2 - ox) / s],
                       [0, f / s, (S / 2 - oy) / s],
                       [0, 0, 1]], np.float64)
        return bodies, Ko
