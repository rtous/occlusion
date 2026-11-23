"""End-to-end: image -> modal masks, 3D bodies, occlusion masks (+ optional composite)."""
from dataclasses import dataclass, field

import numpy as np

from .body3d import BodyEstimator
from .modal import ModalSegmenter
from .render import render_all, render_shaded
from . import occlusion as occ


@dataclass
class Result:
    modal: np.ndarray            # (M,H,W) bool
    bodies: list
    K: np.ndarray
    depths: np.ndarray           # (N,H,W)
    persons: list = field(default_factory=list)   # occlusion.PersonOcclusion


class Pipeline:
    def __init__(self, use_sam=True):
        self.seg = ModalSegmenter(use_sam=use_sam)
        self.hmr = BodyEstimator()
        self.faces = self.hmr.faces
        self._inpainter = None

    def __call__(self, img_bgr):
        h, w = img_bgr.shape[:2]
        people = self.seg(img_bgr)
        modal = np.array([p.mask for p in people]).reshape(-1, h, w)
        bodies, K = self.hmr(img_bgr)
        depths, _ = render_all(bodies, self.faces, K, h, w)
        return Result(modal, bodies, K, depths, occ.compute(modal, depths))

    def composite(self, img_bgr, res):
        from .inpaint import Inpainter, background_mask
        from .compose import compose
        if self._inpainter is None:
            self._inpainter = Inpainter()
        h, w = img_bgr.shape[:2]
        bg = self._inpainter(img_bgr, background_mask(res.modal, img_bgr.shape))
        color, ids = render_shaded(res.bodies, self.faces, res.K, h, w)
        return bg, compose(bg, color, ids, res.persons)
