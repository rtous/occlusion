"""Download the sample images listed in data/images.txt into WORK/images."""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from occlusion.config import IMAGES, REPO  # noqa: E402

for line in (REPO / "data" / "images.txt").read_text().splitlines():
    line = line.split("#")[0].strip()
    if not line:
        continue
    dst = IMAGES / line.rsplit("/", 1)[-1]
    if not dst.exists():
        print("downloading", line)
        urllib.request.urlretrieve(line, dst)
print("images in", IMAGES)
