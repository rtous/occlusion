# occlusion

Refreshed human body occlusion mask estimation (occlusion = amodal − modal). See `PLAN.md`.

## Run (Colab)

```python
from google.colab import drive; drive.mount('/content/drive')
```
```bash
cd /content/occlusion
bash setup_colab.sh            # deps, sample images, SMPL file from Drive
python scripts/run_modal.py    # milestone 2: modal masks -> occlusion_work/outputs/modal
```

Weights, images and outputs go to `/content/drive/MyDrive/occlusion_work/` (or `./work` without Drive).
Put `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` (SMPL neutral) in the repo root or in `occlusion_work/`.

## Status

- [x] M1 environment (`setup_colab.sh`)
- [x] M2 modal masks: YOLO11m-seg + SAM 2.1-b box refinement (`occlusion/modal.py`). ~30 s/image on CPU.
- [ ] M3 3D bodies (HMR2.0 / SMPL) — needs GPU runtime + SMPL file
- [ ] M4 occlusion masks
- [ ] M5 inpainting (LaMa, loaded directly; `simple-lama-inpainting` does not install on py3.13)
- [ ] M6 synthetic eval + figures
