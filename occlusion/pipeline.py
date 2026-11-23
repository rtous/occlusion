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


def detect_topdown(seg, hmr, faces, img_bgr, mhr_det=None, min_inside=0.4, single_conf=0.5):
    """For a top-down body model (SAM 3D Body): one body per person mask, prompted with
    that person's box and visible mask. The person masks come from the cross-model
    detection (detect(), with Multi-HMR as a second person detector) when mhr_det =
    (Multi-HMR estimator, its faces) is given, otherwise from the confident YOLO detections.
    Bodies with less than min_inside of their vertices projected inside the image are
    dropped (e.g. a head at the image border turned into a full body)."""
    h, w = img_bgr.shape[:2]
    if mhr_det is not None:
        modal, mscores, _, _, mdepths = detect(seg, mhr_det[0], mhr_det[1], img_bgr)
        # a person found by only one of the two detectors must be confident (e.g. blurry
        # foreground hands or spectators at the border are often weak YOLO-only detections)
        sils = np.isfinite(mdepths)
        ok = [i for i in range(len(modal))
              if mscores[i] >= single_conf or (len(sils) and occ.match(modal[i][None], sils))]
        modal, mscores = modal[ok], mscores[ok]
    else:
        people = [p for p in seg(img_bgr) if p.score >= occ.STRONG_MODAL]
        modal = np.array([p.mask for p in people], bool).reshape(-1, h, w)
        mscores = np.array([p.score for p in people], np.float32)
        if len(modal):
            modal = resolve_overlaps(modal)
            ok = modal.reshape(len(modal), -1).sum(1) > 0.0005 * h * w
            modal, mscores = modal[ok], mscores[ok]
    boxes = [(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) for ys, xs in (np.nonzero(m) for m in modal)]
    bodies, K = hmr(img_bgr, np.array(boxes, np.float32).reshape(-1, 4), modal)
    keep = []
    for b in bodies:
        uv = b.verts[:, :2] / np.maximum(b.verts[:, 2:], 1e-6) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)).mean()
        if inside >= min_inside:
            keep.append(b)
    depths, _ = render_all(keep, faces, K, h, w)
    return modal, mscores, keep, K, depths


class Pipeline:
    def __init__(self, use_sam=True):
        from .config import BODY
        self.seg = ModalSegmenter(use_sam=use_sam)
        if BODY == "sam3d":
            from functools import partial
            from .body3d_sam3d import Sam3DBodyEstimator
            mh = BodyEstimator()                    # second person detector (cross-model detection)
            self.hmr = Sam3DBodyEstimator()
            self._detect = partial(detect_topdown, mhr_det=(mh, mh.faces))
        else:
            self.hmr, self._detect = BodyEstimator(), detect
        self.faces = self.hmr.faces
        self._inpainter = None

    def __call__(self, img_bgr):
        modal, mscores, bodies, K, depths = self._detect(self.seg, self.hmr, self.faces, img_bgr)
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
