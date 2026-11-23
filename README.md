# occlusion

Refreshed human body occlusion mask estimation (occlusion = amodal − modal). Method: `PLAN.md`. Status and how to resume: `PROGRESS.md`.

## Pipeline

1. **Modal masks**: YOLO11-seg person instances, refined with SAM 2 box prompts (`occlusion/modal.py`).
2. **Amodal masks**: Multi-HMR SMPL-X meshes, projected with a small CPU rasteriser (`occlusion/body3d.py`, `occlusion/render.py`).
3. **Occlusion masks**: Hungarian matching of masks and meshes, `occ = amodal & ~modal`, each pixel labelled human or object occluded (`occlusion/occlusion.py`).
4. **Background + composite**: LaMa inpainting, then meshes drawn with the occluded parts erased (`occlusion/inpaint.py`, `occlusion/compose.py`).

## Run (Colab)

```python
from google.colab import drive; drive.mount('/content/drive')
```
```bash
cd /content/occlusion
bash setup_colab.sh   # once per session; everything heavy is cached on Drive
bash run_all.sh       # all steps, resumable
```

Individual steps: `scripts/run_modal.py`, `run_body3d.py`, `run_occlusion.py`, `run_compose.py`, `make_synthetic_eval.py`, `eval.py` (`--open-frac`, `--dilate-frac` to re-score cached outputs).

Body models (licensed, not committed): `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` and `SMPLX_NEUTRAL.npz` in `/content/drive/MyDrive/occlusion_work/`.
