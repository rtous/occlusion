#!/usr/bin/env bash
# The 30 movie frames used in the old paper (github.com/snicolau/HumanBodyOcclusionMaskEstimation).
# Copyrighted film stills: for internal comparison with the old results only, do not redistribute.
set -e
DST=${1:-/content/drive/MyDrive/occlusion_work/images/oldframes}
[ -d /content/drive/MyDrive ] || DST=${1:-$(dirname "$0")/../work/images/oldframes}
if [ -n "$(ls "$DST" 2>/dev/null)" ]; then echo "already in $DST"; exit 0; fi
TMP=$(mktemp -d)
git clone -q --depth 1 https://github.com/snicolau/HumanBodyOcclusionMaskEstimation.git "$TMP/old"
mkdir -p "$DST" && cp "$TMP/old/frames/"*.png "$DST/"
rm -rf "$TMP"
echo "$(ls "$DST" | wc -l) frames in $DST"
