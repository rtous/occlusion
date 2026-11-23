"""End-to-end: image -> modal masks, 3D bodies, occlusion masks (+ optional composite)."""
from dataclasses import dataclass, field

import numpy as np

from .body3d import BodyEstimator
from .modal import ModalSegmenter, resolve_overlaps
from .render import render_all
from . import occlusion as occ


@dataclass
class Result:
    modal: np.ndarray            # (M,H,W) bool, selected and non-overlapping
    modal_scores: np.ndarray     # (M,) YOLO confidence (body score for SAM fallback masks)
    bodies: list
    K: np.ndarray
    depths: np.ndarray           # (N,H,W)
    persons: list = field(default_factory=list)   # occlusion.PersonOcclusion


def detect(seg, hmr, faces, img_bgr):
    """Run both detectors, keep consistent detections (occlusion.select) and, for confident
    meshes without any visible mask, prompt SAM 2 with the mesh's bounding box."""
    h, w = img_bgr.shape[:2]
    people = seg(img_bgr)
    modal = np.array([p.mask for p in people], bool).reshape(-1, h, w)
    mscores = np.array([p.score for p in people], np.float32)
    bodies, K = hmr(img_bgr)
    depths, _ = render_all(bodies, faces, K, h, w)
    sils = np.isfinite(depths)
    km, kb, no_mask = occ.select(modal, mscores, sils, [b.score for b in bodies])
    modal, mscores = modal[km], mscores[km]
    extra, extra_s = [], []
    if seg.sam is not None:
        taken = modal.any(0) if len(modal) else np.zeros((h, w), bool)
        for j in no_mask:
            ys, xs = np.nonzero(sils[j])
            m = seg.segment_box(img_bgr, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)) & ~taken
            if occ.match(m[None], sils[j][None]):        # IoU >= 0.2 with the mesh
                extra.append(m); extra_s.append(bodies[j].score); taken |= m
    if extra:
        modal = np.concatenate([modal, np.array(extra)])
        mscores = np.concatenate([mscores, np.array(extra_s, np.float32)])
    if len(modal):
        modal = resolve_overlaps(modal)
        ok = modal.reshape(len(modal), -1).sum(1) > 0.0005 * h * w
        modal, mscores = modal[ok], mscores[ok]
    return modal, mscores, [bodies[j] for j in kb], K, depths[kb]


class Pipeline:
    def __init__(self, use_sam=True):
        self.seg = ModalSegmenter(use_sam=use_sam)
        self.hmr = BodyEstimator()
        self.faces = self.hmr.faces
        self._inpainter = None

    def __call__(self, img_bgr):
        modal, mscores, bodies, K, depths = detect(self.seg, self.hmr, self.faces, img_bgr)
        return Result(modal, mscores, bodies, K, depths, occ.compute(modal, depths))

    def composite(self, img_bgr, res):
        from .inpaint import Inpainter, background_mask
        from .compose import compose, person_layers
        if self._inpainter is None:
            self._inpainter = Inpainter()
        h, w = img_bgr.shape[:2]
        bg = self._inpainter(img_bgr, background_mask(res.modal, img_bgr.shape))
        layers = person_layers(res.bodies, self.faces, res.K, h, w)
        return bg, compose(bg, layers, res.depths, res.persons)
