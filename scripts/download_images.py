"""Download images listed in data/<list>.txt into WORK/images/<subdir>.

  python scripts/download_images.py                       # data/images.txt -> WORK/images
  python scripts/download_images.py coco_crowd crowd      # data/coco_crowd.txt -> WORK/images/crowd
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from occlusion.config import IMAGES, REPO  # noqa: E402

lst = sys.argv[1] if len(sys.argv) > 1 else "images"
dst_dir = IMAGES / sys.argv[2] if len(sys.argv) > 2 else IMAGES
dst_dir.mkdir(parents=True, exist_ok=True)
for line in (REPO / "data" / f"{lst}.txt").read_text().splitlines():
    url = line.split("#")[0].strip()
    if not url:
        continue
    dst = dst_dir / url.rsplit("/", 1)[-1]
    if not dst.exists():
        urllib.request.urlretrieve(url, dst)
print("images in", dst_dir)
