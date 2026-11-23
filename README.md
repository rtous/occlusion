# occlusion

Refreshed human body occlusion mask estimation (occlusion = amodal − modal). Method: `PLAN.md`. Status and how to resume: `PROGRESS.md`.

## Pipeline

1. **Modal masks**: YOLO11-seg person instances, refined with SAM 2 box prompts (`occlusion/modal.py`).
2. **Amodal masks**: Multi-HMR SMPL-X meshes, projected with a small CPU rasteriser (`occlusion/body3d.py`, `occlusion/render.py`).
   Both detectors run with a low threshold and cross-check each other; confident meshes without a visible mask get one from SAM 2 (`occlusion.select`, `pipeline.detect`).
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

Individual steps: `scripts/run_detect.py`, `run_occlusion.py`, `run_compose.py`, `make_synthetic_eval.py`, `eval.py` (`--open-frac`, `--dilate-frac` to re-score cached outputs).

Body models (licensed, not committed): `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` and `SMPLX_NEUTRAL.npz` in `/content/drive/MyDrive/occlusion_work/`.

## Where we are (plain-language summary)

### The idea, in one sentence
To find the parts of a person hidden behind something, we estimate the person's **whole body** (amodal mask, from a 3D body model), estimate the **visible body** (modal mask, from segmentation), and subtract: whatever belongs to the whole body but is not visible is occluded.

### Old paper vs. this version

| Step | Old paper (`old_paper.tex`) | Now | What changed in practice |
|---|---|---|---|
| Visible body (modal) | Binary People Segmentation (U-Net). One mask for *all* people together. | YOLO11 + SAM 2. One mask **per person**. | We can now tell people apart, so we know *who* hides *whom*. |
| Whole body (amodal) | OpenPose keypoints + SMPLify-X fitting, a slow per-person optimisation. ORCNN was also tried and discarded (noisy masks). | Multi-HMR: all people's 3D bodies (SMPL-X) in one network pass. | No fitting, no OpenPose, one model; same idea of projecting the 3D body. |
| Human vs. object occlusion | Done by superimposing the front person's mesh. | Automatic: each hidden pixel is labelled *hidden by a person* (another body is in front) or *hidden by an object*. | Same idea, now automatic and per pixel. |
| Background | Stable Diffusion v1.4 inpainting. | LaMa inpainting. | Faster, no prompts, deterministic. |
| Final composite (cartoon-style) | Meshes over the background, hidden parts erased. | Same; each pixel shows the nearest body that is not hidden there. | Same output. |
| Evaluation | Visual only, 30 movie frames. | Visual (COCO images) **plus a number**: a synthetic test where we paste occluders over fully visible people, so the true occlusion mask is known. | The old paper had no quantitative result. |

### What works
- **The whole pipeline runs end to end**, from a photo to occlusion masks and the de-occluded composite, including on CPU (slowly).
- **Visible masks are good.** YOLO and SAM agree closely (about 0.9 overlap).
- **3D bodies line up with the people**, especially heads and torsos.
- **Real occlusions are found.** In a street photo, a man behind a sign pole and a pedestrian behind a bin are cut correctly. In a group photo of 13 kids, those standing behind others get their hidden parts marked as *hidden by a person*.

### What doesn't work well yet
- **False occlusions from imperfect 3D poses.** When the 3D body's arm or leg is slightly off, the part of the mesh that sticks out beyond the real person looks "hidden" even though nothing hides it. A morphological opening (tuned on the benchmark) removes most thin slivers, but this remains the main weakness: the mesh overlaps the real body with only ~0.7 IoU.
- **Tight crops** (half-body shots) give rougher poses, e.g. a wrong arm in `zidane.jpg`.
- Small objects the segmenter includes in a person (trophies, rackets) leave little gaps.
- LaMa leaves dark smudges when the hole (all people together) is very large; mostly hidden by the meshes in the composite.
- A person missed by both detectors (rare) gets no occlusion mask.

### Numbers (synthetic benchmark, 77 images)
Occlusion IoU 0.39 (object occluders) / 0.45 (person occluders); occluder type correct for ~98% of pixels; only 2% of a non-occluded body is wrongly marked as occluded. Detecting with both models at a low threshold and cross-checking them fixed most missed, heavily occluded people. Details: `PROGRESS.md`, paper Tables 1-2.

### Status
Method done; paper draft complete (`paper/main.pdf`: method, results and ablation tables, COCO and movie-frame figures, related work up to 2026).
Next: review by the authors, journal template. Optional: refine meshes to fit the visible masks.

Detailed, technical progress notes: `PROGRESS.md`.
