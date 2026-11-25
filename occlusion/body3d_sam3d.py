"""3D bodies with SAM 3D Body (Meta, 2026; MHR mesh), as an alternative to Multi-HMR.

Top-down: one body per person box. We prompt it with our own boxes and visible (modal)
masks, and give all persons the same camera (60 deg FOV on the longest side, as for
Multi-HMR) so that their depths are comparable.

Setup (setup_colab.sh): code in /content/third_party/sam-3d-body; checkpoint (gated on
Hugging Face: request access to facebook/sam-3d-body-vith) cached in WEIGHTS/sam-3d-body-vith.
"""
import os
import sys

import cv2
import numpy as np
import torch

from .body3d import Body
from .config import WEIGHTS

SAM3D_DIR = os.environ.get("SAM3D_DIR", "/content/third_party/sam-3d-body")
CKPT_DIR = WEIGHTS / "sam-3d-body-vith"


def camera(h, w, fov=60.0):
    f = max(h, w) / 2 / np.tan(np.radians(fov) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


class Sam3DBodyEstimator:
    def __init__(self, ckpt_dir=CKPT_DIR, use_masks=True, fov=60.0):
        if SAM3D_DIR not in sys.path:
            sys.path.insert(0, SAM3D_DIR)
        from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
        model, cfg = load_sam_3d_body(str(ckpt_dir / "model.ckpt"), device="cuda",
                                      mhr_path=str(ckpt_dir / "assets" / "mhr_model.pt"))
        self.est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
        self.faces = np.asarray(self.est.faces, np.int64)
        self.use_masks = use_masks
        self.fov = fov

    def __call__(self, img_bgr, boxes, masks=None):
        """boxes (N,4) xyxy, masks (N,H,W) bool -> list of Body (camera coords), K."""
        h, w = img_bgr.shape[:2]
        K = camera(h, w, self.fov)
        if len(boxes) == 0:
            return [], K
        kw = dict(bboxes=np.asarray(boxes, np.float32),
                  cam_int=torch.from_numpy(K).float()[None])
        if self.use_masks and masks is not None:
            kw["masks"] = np.asarray(masks, np.uint8)
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()), torch.no_grad():   # it prints a lot
            outs = self.est.process_one_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), **kw)
        bodies = [Body((o["pred_vertices"] + o["pred_cam_t"]).astype(np.float32), 1.0,
                       params={"kp3d": (o["pred_keypoints_3d"] + o["pred_cam_t"]).astype(np.float32)})
                  for o in outs]
        return bodies, K
