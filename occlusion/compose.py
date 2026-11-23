"""De-occlusion composite: meshes over the inpainted background, with the parts
that were hidden in the original image (occlusion masks) erased."""
import numpy as np


def compose(background, mesh_color, mesh_ids, persons):
    """persons: list of occlusion.PersonOcclusion (body_idx refers to mesh_ids values).
    Unmatched meshes (no visible person) are not drawn."""
    out = background.copy()
    for p in persons:
        draw = (mesh_ids == p.body_idx) & ~p.occ
        out[draw] = mesh_color[draw]
    return out


def naive_compose(background, mesh_color, mesh_ids, persons):
    """Same without occlusion handling (for comparison figures)."""
    out = background.copy()
    for p in persons:
        draw = mesh_ids == p.body_idx
        out[draw] = mesh_color[draw]
    return out
