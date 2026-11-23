"""De-occlusion composite: meshes over the inpainted background, with the parts
that were hidden in the original image (occlusion masks) erased.

Per pixel we draw the nearest mesh among the persons that are NOT occluded there,
so erasing a back person never leaves a hole when another body should show."""
import numpy as np

from .render import render_shaded, PALETTE


def person_layers(bodies, faces, K, h, w):
    """Per-person flat-shaded colour layers (N,H,W,3)."""
    return np.stack([render_shaded([b], faces, K, h, w, palette=[PALETTE[i % len(PALETTE)]])[0]
                     for i, b in enumerate(bodies)]) if bodies else np.zeros((0, h, w, 3), np.uint8)


def compose(background, layers, depths, persons, handle_occlusion=True):
    """layers: (N,H,W,3) colours; depths: (N,H,W); persons: occlusion.PersonOcclusion list.
    Unmatched meshes (no visible person) are not drawn."""
    d = np.full_like(depths, np.inf)
    for p in persons:
        vis = np.isfinite(depths[p.body_idx])
        if handle_occlusion:
            vis &= ~p.occ
        d[p.body_idx][vis] = depths[p.body_idx][vis]
    out = background.copy()
    if len(d) == 0:
        return out
    k = d.argmin(0)
    draw = np.isfinite(d.min(0))
    yy, xx = np.nonzero(draw)
    out[yy, xx] = layers[k[yy, xx], yy, xx]
    return out
