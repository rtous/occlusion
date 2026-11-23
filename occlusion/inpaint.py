"""Background recovery with LaMa (TorchScript big-lama, loaded directly).

simple-lama-inpainting is not installable on py3.13 (pins numpy 1.26), so we
just load the same TorchScript checkpoint it uses.
"""
import urllib.request

import cv2
import numpy as np
import torch

from .config import WEIGHTS, device

LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"


class Inpainter:
    def __init__(self, max_side=1024):
        p = WEIGHTS / "big-lama.pt"
        if not p.exists():
            print("downloading", LAMA_URL)
            urllib.request.urlretrieve(LAMA_URL, p)
        self.dev = torch.device(device())
        self.model = torch.jit.load(str(p), map_location=self.dev).eval()
        self.max_side = max_side

    def __call__(self, img_bgr, mask):
        """mask: HxW bool, True = fill. Returns inpainted BGR image, same size."""
        h, w = img_bgr.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        sh, sw = int(round(h * s)), int(round(w * s))
        ph, pw = (8 - sh % 8) % 8, (8 - sw % 8) % 8
        im = cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        m = cv2.resize(mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
        im = np.pad(im, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        m = np.pad(m, ((0, ph), (0, pw)), mode="reflect")
        x = torch.from_numpy(im.copy()).permute(2, 0, 1)[None].float().div(255).to(self.dev)
        mk = torch.from_numpy(m)[None, None].float().to(self.dev)
        with torch.no_grad():
            y = self.model(x, mk)[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
        y = cv2.resize(y[:sh, :sw, ::-1], (w, h), interpolation=cv2.INTER_CUBIC)
        out = img_bgr.copy()
        out[mask] = y[mask]                     # keep original pixels outside the hole
        return out


def background_mask(modal_masks, img_shape, dilate_frac=0.015):
    """Union of all visible people, dilated (erases segmentation edge errors)."""
    h, w = img_shape[:2]
    m = np.zeros((h, w), np.uint8) if len(modal_masks) == 0 else \
        np.any(modal_masks, 0).astype(np.uint8)
    r = max(int(dilate_frac * max(h, w)), 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m, k).astype(bool)
