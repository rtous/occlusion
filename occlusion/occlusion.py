"""Occlusion masks: occ_i = amodal_i AND NOT modal_i, labelled human/object.

amodal_i = silhouette of person i's projected 3D mesh.
modal_i  = visible segmentation mask matched to that mesh (Hungarian on IoU).
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class PersonOcclusion:
    body_idx: int        # index into bodies / depth stack
    modal_idx: int       # index into modal masks (-1 if unmatched)
    amodal: np.ndarray   # HxW bool
    modal: np.ndarray    # HxW bool
    occ: np.ndarray      # HxW bool  (= occ_human | occ_object)
    occ_human: np.ndarray
    occ_object: np.ndarray

    @property
    def occ_ratio(self):
        return self.occ.sum() / max(self.amodal.sum(), 1)


def _kernel(r):
    r = max(int(r), 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def match(modal_masks, sils, min_iou=0.2):
    """Hungarian matching between modal masks (M,H,W) and mesh silhouettes (N,H,W)."""
    M, N = len(modal_masks), len(sils)
    if M == 0 or N == 0:
        return []
    a = modal_masks.reshape(M, -1).astype(np.float32)
    b = sils.reshape(N, -1).astype(np.float32)
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None] - inter
    iou = inter / np.maximum(union, 1)
    r, c = linear_sum_assignment(-iou)
    return [(i, j, iou[i, j]) for i, j in zip(r, c) if iou[i, j] >= min_iou]


STRONG_MODAL = 0.35   # YOLO confidence of a "confident" person detection
STRONG_BODY = 0.3     # Multi-HMR detection score of a "confident" body


def select(modal_masks, modal_scores, sils, body_scores, min_iou=0.2,
           t_modal=STRONG_MODAL, t_body=STRONG_BODY):
    """Which detections to keep. Both detectors run with a low threshold; confident
    detections are always kept, and a weak detection is kept only when it matches a
    confident detection of the OTHER model that has no confident partner
    (heavily occluded people are often missed by one of the two models).
    Returns (modal indices, body indices, body indices still without a modal mask)."""
    ms, bs = np.asarray(modal_scores), np.asarray(body_scores)
    sm, wm = np.where(ms >= t_modal)[0], np.where(ms < t_modal)[0]
    sb, wb = np.where(bs >= t_body)[0], np.where(bs < t_body)[0]
    pairs = match(modal_masks[sm], sils[sb], min_iou)
    um = np.setdiff1d(sm, [sm[i] for i, _, _ in pairs]).astype(int)
    ub = np.setdiff1d(sb, [sb[j] for _, j, _ in pairs]).astype(int)
    extra_b = [wb[j] for _, j, _ in match(modal_masks[um], sils[wb], min_iou)]
    extra_m = [wm[i] for i, j, _ in match(modal_masks[wm], sils[ub], min_iou)]
    got = {ub[j] for _, j, _ in match(modal_masks[wm], sils[ub], min_iou)}
    keep_m = sorted(map(int, list(sm) + extra_m))
    keep_b = sorted(map(int, list(sb) + extra_b))
    no_mask = [int(j) for j in ub if j not in got]
    return keep_m, keep_b, no_mask


def compute(modal_masks, depths, min_iou=0.2, dilate_frac=0.03, open_frac=0.01):
    """modal_masks: (M,H,W) bool; depths: (N,H,W) float (inf = empty).

    dilate_frac / open_frac are relative to the person's amodal bbox height:
    - modal is clipped to a dilated amodal (removes stray segmentation pixels),
    - occ is morphologically opened (removes thin slivers from mesh misfit).
    """
    sils = np.isfinite(depths)
    pairs = match(modal_masks, sils, min_iou)
    out = []
    for mi, bj, _ in pairs:
        amodal = sils[bj]
        ys = np.where(amodal.any(1))[0]
        hgt = ys[-1] - ys[0] + 1
        modal = modal_masks[mi] & cv2.dilate(amodal.astype(np.uint8),
                                             _kernel(dilate_frac * hgt)).astype(bool)
        occ = amodal & ~modal
        occ = cv2.morphologyEx(occ.astype(np.uint8), cv2.MORPH_OPEN,
                               _kernel(open_frac * hgt)).astype(bool)
        # human-occluded: another mesh is in front, or another person's visible mask is there
        others = np.delete(np.arange(len(depths)), bj)
        in_front = (depths[others] < depths[bj][None]).any(0) if len(others) else np.zeros_like(occ)
        other_modal = np.delete(modal_masks, mi, axis=0).any(0) if len(modal_masks) > 1 \
            else np.zeros_like(occ)
        occ_h = occ & (in_front | other_modal)
        out.append(PersonOcclusion(bj, mi, amodal, modal, occ, occ_h, occ & ~occ_h))
    return out


HUMAN_COLOR = (255, 120, 0)    # BGR: blue-ish
OBJECT_COLOR = (0, 0, 255)     # BGR: red


def occlusion_overlay(img_bgr, persons, alpha=0.7):
    out = (img_bgr * 0.5).astype(np.uint8)
    for p in persons:
        cnts, _ = cv2.findContours(p.amodal.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 255, 255), 1)
        for m, c in ((p.occ_human, HUMAN_COLOR), (p.occ_object, OBJECT_COLOR)):
            out[m] = (alpha * np.array(c) + (1 - alpha) * out[m]).astype(np.uint8)
    return out


def masks_panel(shape, persons, which):
    """Binary-style panel (white on black) for modal / amodal / occ, like the paper figure."""
    out = np.zeros(shape[:2] + (3,), np.uint8)
    for p in persons:
        out[getattr(p, which)] = 255
    return out
