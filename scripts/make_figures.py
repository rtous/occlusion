"""Paper figures from cached outputs (run_detect.py on the same images).

Writes paper/figures/*.jpg:
  qualitative.jpg  grid on COCO group images: original | modal | amodal | occlusion
  occlusion_ex.jpg one row of the above (intro figure)
  modal_masks.jpg  per-person modal masks over 3 images
  meshes.jpg       Multi-HMR meshes over the same 3 images
  background.jpg   original | inpainted background
  deocclusion.jpg  original | background | meshes w/o occlusion handling | de-occluded
  cartoon.jpg      original | de-occluded meshes over a stylised background
Default images were hand-picked for readability; pass --qual '' / --pick '' to rank automatically.
Only COCO (CC-licensed) images are used; the movie frames stay internal.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from occlusion.config import IMAGES, OUTPUTS, REPO, list_images  # noqa: E402
from occlusion.body3d import Body  # noqa: E402
from occlusion.render import render_all, overlay_meshes  # noqa: E402
from occlusion import occlusion as occ  # noqa: E402
from occlusion.modal import overlay  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--images", default=str(IMAGES / "crowd"))
ap.add_argument("--n-qual", type=int, default=6)
ap.add_argument("--qual", default="000000247917,000000363666,000000281759,000000203389,000000306139,000000090891", help="comma-separated stems for the qualitative grid")
ap.add_argument("--pick", default="000000247917,000000363666,000000281759", help="comma-separated stems for the single-image figures")
ap.add_argument("--out", default=str(REPO / "paper" / "figures"))
a = ap.parse_args()
FIG = Path(a.out)
FIG.mkdir(parents=True, exist_ok=True)
faces = np.load(REPO / "SMPLX_NEUTRAL.npz", allow_pickle=True)["f"].astype(np.int64)


def load(f):
    fm, fb = OUTPUTS / "modal" / (f.stem + ".npz"), OUTPUTS / "body3d" / (f.stem + ".npz")
    if not (fm.exists() and fb.exists()):
        return None
    img = cv2.imread(str(f))
    h, w = img.shape[:2]
    modal = np.load(fm)["masks"].reshape(-1, h, w)
    b = np.load(fb)
    bodies = [Body(v, s) for v, s in zip(b["verts"], b["scores"])]
    depths, ids = render_all(bodies, faces, b["K"], h, w)
    persons = occ.compute(modal, depths)
    return dict(f=f, img=img, modal=modal, bodies=bodies, K=b["K"], depths=depths, ids=ids, persons=persons)


def row(panels, height=360, gap=6):
    ps = [cv2.resize(p, (int(p.shape[1] * height / p.shape[0]), height), interpolation=cv2.INTER_AREA)
          for p in panels]
    out = []
    for p in ps:
        out += [p, np.full((height, gap, 3), 255, np.uint8)]
    return np.hstack(out[:-1])


def stack(rows, width=2400, gap=6):
    rs = [cv2.resize(r, (width, int(r.shape[0] * width / r.shape[1])), interpolation=cv2.INTER_AREA)
          for r in rows]
    out = []
    for r in rs:
        out += [r, np.full((gap, width, 3), 255, np.uint8)]
    return np.vstack(out[:-1])


def save(name, im):
    cv2.imwrite(str(FIG / name), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", FIG / name)


def occ_row(r):
    i, p = r["img"], r["persons"]
    return row([i, occ.masks_panel(i.shape, p, "modal"), occ.masks_panel(i.shape, p, "amodal"),
                occ.occlusion_overlay(i, p)])


def score(r):
    """Prefer images with both kinds of occlusion and several matched people."""
    hum = sum(p.occ_human.sum() for p in r["persons"])
    obj = sum(p.occ_object.sum() for p in r["persons"])
    tot = r["img"].shape[0] * r["img"].shape[1]
    return (min(hum, 4 * obj + 1) + 0.5 * obj) / tot * min(len(r["persons"]), 6)


files = list_images(a.images)
res = {f.stem: r for f in files if (r := load(f)) is not None}
print(len(res), "images with cached outputs in", a.images)
if not res:
    sys.exit("nothing to do: run run_detect.py --images " + a.images)
ranked = sorted(res, key=lambda k: -score(res[k]))
qual = a.qual.split(",") if a.qual else ranked[:a.n_qual]
pick = a.pick.split(",") if a.pick else ranked[:3]
print("qualitative:", qual, "\npick:", pick)

save("qualitative.jpg", stack([occ_row(res[k]) for k in qual]))
save("occlusion_ex.jpg", occ_row(res[pick[0]]))
save("modal_masks.jpg", row([overlay(res[k]["img"], [type("P", (), {"mask": m}) for m in res[k]["modal"]])
                             for k in pick]))
save("meshes.jpg", row([overlay_meshes(res[k]["img"], res[k]["ids"]) for k in pick]))

# background / de-occlusion / cartoon need the inpainted background (cached by run_compose.py)
from occlusion.inpaint import Inpainter, background_mask  # noqa: E402
from occlusion.compose import compose, person_layers  # noqa: E402
inp = None
rows_d, rows_b, rows_c = [], [], []
for k in pick[:2]:
    r = res[k]
    i = r["img"]
    h, w = i.shape[:2]
    bf = OUTPUTS / "compose" / (k + "_bg.png")
    if bf.exists():
        bg = cv2.imread(str(bf))
    else:
        inp = inp or Inpainter()
        bg = inp(i, background_mask(r["modal"], i.shape))
        bf.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(bf), bg)
    layers = person_layers(r["bodies"], faces, r["K"], h, w)
    rows_b.append(row([i, bg]))
    rows_d.append(row([i, bg, compose(bg, layers, r["depths"], r["persons"], handle_occlusion=False),
                       compose(bg, layers, r["depths"], r["persons"])]))
    toon = cv2.stylization(bg, sigma_s=60, sigma_r=0.45)
    rows_c.append(row([i, compose(toon, layers, r["depths"], r["persons"])]))
save("background.jpg", rows_b[0])
save("deocclusion.jpg", stack(rows_d))
save("cartoon.jpg", stack(rows_c, width=1600))
