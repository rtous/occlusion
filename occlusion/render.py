"""Tiny CPU mesh rasteriser (painter's algorithm with cv2.fillPoly).

Good enough for silhouettes and a per-person depth map; no OpenGL needed.
"""
import cv2
import numpy as np

SUB = 4  # sub-pixel bits for cv2.fillPoly (shift)


def project(verts, K):
    z = verts[:, 2]
    uv = verts[:, :2] / np.maximum(z[:, None], 1e-6)
    uv = uv * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, z


def render_depth(verts, faces, K, h, w):
    """Per-pixel depth of one mesh (np.inf where empty)."""
    uv, z = project(verts, K)
    tri_z = z[faces]                                   # (F,3)
    ok = (tri_z > 0.05).all(1)
    tri = uv[faces[ok]]                                # (F,3,2)
    tz = tri_z[ok].mean(1)
    inside = ((tri[..., 0].max(1) >= 0) & (tri[..., 0].min(1) < w) &
              (tri[..., 1].max(1) >= 0) & (tri[..., 1].min(1) < h))
    tri, tz = tri[inside], tz[inside]
    order = np.argsort(-tz)                            # far to near
    pts = np.round(tri[order] * (1 << SUB)).astype(np.int32)
    depth = np.full((h, w), np.inf, np.float32)
    for p, d in zip(pts, tz[order]):
        cv2.fillConvexPoly(depth, p, float(d), lineType=cv2.LINE_8, shift=SUB)
    return depth


def render_all(bodies, faces, K, h, w):
    """Returns depth stack (N,H,W) and id map (H,W; -1 = background)."""
    if not bodies:
        return np.zeros((0, h, w), np.float32), np.full((h, w), -1, np.int32)
    depths = np.stack([render_depth(b.verts, faces, K, h, w) for b in bodies])
    ids = depths.argmin(0).astype(np.int32)
    ids[np.isinf(depths.min(0))] = -1
    return depths, ids


def overlay_meshes(img_bgr, ids, alpha=0.6):
    out = img_bgr.copy()
    rng = np.random.default_rng(1)
    for i in range(ids.max() + 1):
        m = ids == i
        c = rng.integers(60, 255, 3)
        out[m] = (alpha * c + (1 - alpha) * out[m]).astype(np.uint8)
    return out


PALETTE = [(80, 160, 240), (120, 200, 120), (230, 150, 90), (200, 120, 200),
           (90, 200, 220), (220, 200, 100), (150, 150, 240), (240, 120, 140)]  # BGR


def render_shaded(bodies, faces, K, h, w, palette=PALETTE):
    """Flat-shaded colour render of all meshes together (global painter's order).
    Returns (color HxWx3 uint8, ids HxW int32 with -1 = background)."""
    color = np.zeros((h, w, 3), np.uint8)
    ids = np.full((h, w), -1, np.int32)
    tris, depth, shade, owner = [], [], [], []
    for i, b in enumerate(bodies):
        uv, z = project(b.verts, K)
        v = b.verts[faces]
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        c = v.mean(1)
        view = -c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
        lam = np.abs((n * view).sum(1))           # headlight lambert
        ok = (z[faces] > 0.05).all(1)
        tris.append(uv[faces[ok]]); depth.append(z[faces[ok]].mean(1))
        shade.append(0.35 + 0.65 * lam[ok]); owner.append(np.full(ok.sum(), i))
    if not tris:
        return color, ids
    tris, depth = np.concatenate(tris), np.concatenate(depth)
    shade, owner = np.concatenate(shade), np.concatenate(owner)
    order = np.argsort(-depth)
    pts = np.round(tris[order] * (1 << SUB)).astype(np.int32)
    for p, s, o in zip(pts, shade[order], owner[order]):
        c = [int(s * x) for x in palette[o % len(palette)]]
        cv2.fillConvexPoly(color, p, c, lineType=cv2.LINE_8, shift=SUB)
        cv2.fillConvexPoly(ids, p, int(o), lineType=cv2.LINE_8, shift=SUB)
    return color, ids
