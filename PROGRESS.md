# Progress / handoff notes

Read this first when resuming (after `CLAUDE.md`, then `PLAN.md` for the method). Update it at the end of every session.

## Resume in a new Colab session (GPU runtime)

```python
from google.colab import drive; drive.mount('/content/drive')
```
```bash
cd /content/occlusion && git pull
bash setup_colab.sh      # deps, Multi-HMR, SAM 3D Body, Blender, weights, SMPL-X files, images (cached on Drive)
bash run_all.sh          # optional: everything is cached on Drive, so this only regenerates figures (~minutes)
bash paper/build.sh      # rebuild paper/main.pdf (installs TeX Live on a fresh VM, ~3 min)
```

- The SAM 3D Body checkpoint is already cached on Drive (`weights/sam-3d-body-vith`), so no Hugging Face login is needed.
  Only if it were lost: `! hf auth login` (device code, user `rtous` has access), then `bash setup_colab.sh`.
- `git push` needs a token from the user (push with `https://rtous:<token>@github.com/rtous/occlusion.git`; never store it in
  the repo config). If commits could not be pushed, back them up: `git bundle create /content/drive/MyDrive/occlusion_work/occlusion_repo.bundle origin/main..main`
  and restore with `git pull <bundle> main`.

## Status (2026-11-23, end of session 2): method done, paper draft complete

**Main method = SAM 3D Body** (user decision). Multi-HMR is used as a second person detector and reported as the comparison body model.

| Part | State | Notes |
|---|---|---|
| Modal masks | done | YOLO11m-seg + SAM 2.1-b box refinement (`occlusion/modal.py`), detections kept down to conf 0.1 |
| Person selection | done | cross-model detection with Multi-HMR (`occlusion.select`, `pipeline.detect`): weak detections kept only if they pair with an unmatched confident one of the other model; SAM 2 mask for confident meshes without mask; persons found by only one detector need conf >= 0.5 (`pipeline.detect_topdown`, `single_conf`) |
| 3D bodies | done | SAM 3D Body ViT-H (`occlusion/body3d_sam3d.py`), one per person, prompted with box + modal mask, shared 60deg camera; bodies with <40% vertices inside the image dropped. Multi-HMR 672_L (`occlusion/body3d.py`) for comparison |
| Occlusion masks | done | Hungarian matching, `occ = amodal & ~modal`, opening `open_frac` 0.02 (tuned), human/object label by depth (`occlusion/occlusion.py`) |
| Background + composite | done | LaMa (`occlusion/inpaint.py`), per-pixel nearest non-occluded mesh (`occlusion/compose.py`) |
| Synthetic benchmark | done | 77 images (30 clean, 26 object, 21 human), `scripts/make_synthetic_eval.py`, `scripts/eval.py` |
| Figures | done | `scripts/make_figures.py` (COCO groups + `--name movies --intro` for movie frames / Figure 1), `scripts/make_compare.py` (Multi-HMR vs SAM 3D Body), `paper/figures/diagram.tex` |
| Paper | draft complete | no `\todo` left; waiting for the user's review |
| Cartoon figure (paper_v2 Fig. 7) | code ready, **needs a Colab run** | `scripts/make_cartoon.py`: re-runs SAM 3D Body with the cached boxes/masks to get the MHR70 3D keypoints, poses the Mixamo characters `models/*.fbx` with `scripts/blender_mixamo.py` (direction-based retargeting from joint positions, tested on the Mac with SMPL-X poses, all 5 characters OK) and composites them with the occlusion masks -> `paper_v2/figures/cartoonization.jpg` (original | meshes | characters). Run on Colab: `python scripts/make_cartoon.py --blender <blender>`; then check the figure and update its caption in `paper_v2/main.tex` |

## Results (synthetic benchmark, open_frac 0.02)

| body model | variant | occ IoU | false-pos ratio | amodal IoU | label acc |
|---|---|---|---|---|---|
| **SAM 3D Body (final)** | object | 0.489 | 0.034 | 0.777 | 0.906 |
| | human | 0.532 | 0.022 | 0.767 | 0.972 |
| | clean | n/a | 0.006 | 0.817 | n/a |
| Multi-HMR | object | 0.385 | 0.059 | 0.681 | 0.977 |
| | human | 0.452 | 0.044 | 0.669 | 0.975 |
| | clean | n/a | 0.021 | 0.724 | n/a |

Ablation (SAM 3D Body): YOLO-only person detection gives human occ IoU 0.498 / amodal 0.691 (4 missed targets; final misses 2:
`170191`, `364102`). open_frac 0 / 0.01 / 0.03 / 0.05: object 0.494 / 0.502 / 0.413 / 0.251, human 0.560 / 0.569 / 0.425 / 0.166,
clean FP 0.023 / 0.013 / 0.004 / 0.000. dilate_frac has no effect.
Lower object label acc: pasted patches textured with pictures of people get detected as persons (benchmark artifact).
Timing on T4: detection + reconstruction ~10 s/img with SAM 3D Body (1.2 s with Multi-HMR), rasterise + occlusion 0.5 s, LaMa 0.9 s.
Re-score cached outputs: `python scripts/eval.py --cached-only [--open-frac X --tag _X]` (add `OCC_BODY=multihmr` for Multi-HMR).

## Where files live

Code in the repo; everything else in `/content/drive/MyDrive/occlusion_work/`:

| Path | Content |
|---|---|
| `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`, `SMPLX_NEUTRAL.npz` | licensed body models; `setup_colab.sh` copies them to the repo root (gitignored) |
| `weights/` | yolo11m-seg, sam2.1_b, big-lama.pt, `multihmr/` (Multi-HMR + smpl_mean_params), `sam-3d-body-vith/` (gated checkpoint + MHR model) |
| `coco/person_keypoints_val2017.json` | used by `scripts/select_coco.py` |
| `images/` (6 samples), `images/crowd/` (30 COCO groups), `images/single/` (30, benchmark sources) | lists in `data/*.txt` |
| `images/oldframes/` (30 PNG) | movie frames (`scripts/fetch_old_frames.sh`), second qualitative set (films listed in the paper appendix) |
| `synth/` | synthetic benchmark |
| `outputs/{modal,body3d,occlusion,compose,eval}_sam3d` | **main method** (SAM 3D Body); `eval_sam3d/cache` = raw outputs per benchmark image |
| `outputs/{modal,body3d,occlusion,compose,eval}` | Multi-HMR as body model (comparison) |
| `outputs/*_sam3d_v1`, `*_sam3d_v2` | earlier SAM 3D Body runs (v1: YOLO-only detection = ablation row; v2: without `single_conf`) |
| `outputs/*_v1` | first Multi-HMR run (single-threshold detection) |
| `outputs/cartoon/` | cartoon renders (figure removed) |

Output folders are chosen by `occlusion.config.out_dir()` from `OCC_BODY` (default `sam3d`).

## Paper (`paper_v1/` = first version, `paper_v2/` = current, Springer `sn-jnl` class, edited by a colleague)

- paper_v2 build: `cd paper_v2 && pdflatex main && bibtex main && pdflatex main && pdflatex main` (diagram: `cd figures && pdflatex diagram.tex`)

- **User decision: the old paper is unpublished; do NOT compare with it anywhere.** The movie frames are just another qualitative dataset.
- Authors: Ruben Tous (contact, ruben.tous@upc.edu), Beatriz Otero, Sebastià Nicolau.
- `paper/main.tex` (plain `article` class for now), `paper/references.bib`, `paper/figures/` (all generated). Build: `bash paper/build.sh`.
- **After every change to the paper, rebuild the PDF and commit `paper/main.pdf` with the sources.**
- Old paper text kept verbatim where still valid; **new/changed text is wrapped in `\new{...}` (blue)**, open items in `\todo{...}` (red; none left). Don't rewrite the user's kept (black) text without asking.
- Content: method (SAM 3D Body, cross-model detection), Table 1 (results), Table 2 (ablation), Table 3 + pros/cons list + 2 figures
  (Multi-HMR vs SAM 3D Body), Figure 1 = bench movie frame (occlusion row + de-occlusion row), COCO groups (kitchen `397354` first;
  user rejected the baseball team `247917` and the goats `008690`), movie-frame figures, timing, related work up to 2026
  (citations verified online: pix2gestalt, Zhan 2024, Xu 2024, SAMBA, Diffusion-VAS, DMAT, Noh & Chang, SAM 3D Body).

## Next steps (need the user)

1. User reviews the paper draft.
2. Target journal: move to its template; set `\showchangesfalse` in `main.tex` to print the blue text in black.
3. Optional improvements: refine meshes to fit the modal silhouettes (main remaining error, amodal IoU ~0.8);
   faster SAM 3D Body (`inference_type="body"`, skips the hand decoder).

## Environment gotchas

- Check `torch.cuda.is_available()`; the runtime has sometimes been CPU-only. Multi-HMR runs on CPU (~55 s/img); SAM 3D Body needs the GPU.
- Python 3.13: `simple-lama-inpainting` can't be installed; we load `big-lama.pt` with `torch.jit.load` (`occlusion/inpaint.py`).
- Multi-HMR uses relative paths (`./models/...`): `occlusion/body3d.py` chdirs into `/content/third_party/multi-hmr` while loading;
  DINOv2 code comes via `torch.hub` (internet on first load). xFormers / "missing keys" / "Mask-condition inference is not supported"
  / "default FOV" messages are harmless (we pass masks and the camera ourselves).
- Rendering is a small numpy/OpenCV rasteriser (`occlusion/render.py`), no OpenGL. Blender (cartoon scripts only) renders EEVEE via
  surfaceless EGL, ~1 min per character.
- Downloading and running binaries from the internet (e.g. the Blender tarball) was blocked by the permission system; apt packages work.

## Conventions

- All experiment code goes in the repo (`occlusion/` package + `scripts/`), runnable without redevelopment; commit after every working step.
- Steps cache outputs on Drive and are resumable (`--force` to recompute).
