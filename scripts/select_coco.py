"""Pick COCO val2017 images using person keypoint annotations.

- data/coco_crowd.txt : 30 images with 3-8 people and several occluded keypoints (qualitative figure)
- data/coco_single.txt: 60 images with one large, fully visible person (synthetic evaluation)

Needs WORK/coco/person_keypoints_val2017.json (see setup_colab.sh).
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from occlusion.config import WORK, REPO  # noqa: E402

d = json.load(open(WORK / "coco" / "person_keypoints_val2017.json"))
lic = {l["id"]: l["name"] for l in d["licenses"]}
FREE = {4, 5}  # Attribution, Attribution-ShareAlike (crowd figure, may be shown in the paper)
EVAL = {1, 2, 4, 5, 7, 8}  # + NonCommercial, no-known-copyright, US gov (no NoDerivs: we paste occluders)
imgs = {i["id"]: i for i in d["images"] if i["license"] in EVAL}
anns = defaultdict(list)
for a in d["annotations"]:
    if a["image_id"] in imgs:
        anns[a["image_id"]].append(a)

crowd, single = [], []
for iid, A in anns.items():
    im = imgs[iid]
    W, H = im["width"], im["height"]
    if any(a["iscrowd"] for a in A):
        continue
    big = [a for a in A if a["area"] > 0.01 * W * H]
    occluded_kp = sum(a["keypoints"][2::3].count(1) for a in big)
    if im["license"] in FREE and 3 <= len(big) <= 8 and len(A) <= 10 and occluded_kp >= 8:
        crowd.append((occluded_kp, iid))
    if len(A) == 1:
        a = A[0]
        x, y, w, h = a["bbox"]
        v = a["keypoints"][2::3]
        if (v.count(2) == 17 and h > 0.5 * H and x > 5 and y > 5
                and x + w < W - 5 and y + h < H - 5):
            single.append(iid)

random.seed(0)
crowd = [i for _, i in sorted(crowd, reverse=True)[:30]]
single = sorted(random.sample(single, min(60, len(single))))


def write(name, ids, desc):
    lines = [f"# {desc}", "# url  # license, flickr source"]
    for i in ids:
        im = imgs[i]
        lines.append(f"{im['coco_url']}  # {lic[im['license']]}, {im['flickr_url']}")
    (REPO / "data" / name).write_text("\n".join(lines) + "\n")
    print(name, len(ids))


write("coco_crowd.txt", crowd, "COCO val2017, 3-8 people with occlusions (qualitative)")
write("coco_single.txt", single, "COCO val2017, one fully visible person (synthetic eval)")
