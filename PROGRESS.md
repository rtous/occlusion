# Progress / handoff notes

Read this first when resuming (after `CLAUDE.md` and `PLAN.md`). Update it at the end of every session.

## Resume in a new Colab session

```python
from google.colab import drive; drive.mount('/content/drive')
```
```bash
cd /content/occlusion && git pull
bash setup_colab.sh      # deps, Multi-HMR, weights, SMPL files, images (all cached on Drive)
bash run_all.sh          # every step skips outputs already on Drive
```

If `git log` lacks recent work (commits that were never pushed), restore them from the Drive bundle:
`git pull /content/drive/MyDrive/occlusion_work/occlusion_repo.bundle main` (then push once credentials are set).

Everything that isn't code lives in `/content/drive/MyDrive/occlusion_work/`:

| Path | Content |
|---|---|
| `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`, `SMPLX_NEUTRAL.npz` | licensed body models; `setup_colab.sh` copies them to the repo root (gitignored) |
| `weights/` | yolo11m-seg, sam2.1_b, big-lama.pt, `multihmr/multiHMR_672_L.pt` + `smpl_mean_params.npz` |
| `coco/person_keypoints_val2017.json` | used by `scripts/select_coco.py` |
| `images/` (samples), `images/crowd/` (30), `images/single/` (30) | lists in `data/*.txt` |
| `images/oldframes/` (30 PNG) | movie frames (`scripts/fetch_old_frames.sh`), second qualitative set in the paper (films listed in the appendix) |
| `synth/` | synthetic benchmark (`make_synthetic_eval.py`) |
| `outputs/{modal,body3d,occlusion,compose,eval}` | per-step outputs; `eval/cache/` holds raw network outputs per synthetic image |
| `outputs/*_v1` | outputs of the first (single-threshold) detection, kept for the ablation (`eval.py --out eval_v1`) |

## Status (2026-11-23, end of session 2, GPU T4): method and paper draft complete

| Milestone | State | Notes |
|---|---|---|
| M1 environment | done | `setup_colab.sh` |
| M2+M3 detection | done | `scripts/run_detect.py` (replaces run_modal/run_body3d): YOLO11m-seg + SAM 2.1-b, Multi-HMR 672_L, **cross-model selection** (`occlusion.select`) + SAM fallback for meshes without mask (`pipeline.detect`) |
| M4 occlusion masks | done | `open_frac` tuned: **0.02** (default now); `dilate_frac` has no effect |
| M5 inpainting + composite | done | LaMa; large holes leave dark smudges (mentioned in paper) |
| M6 synthetic eval | done | 77 images (30 clean, 26 object, 21 human) |
| Movie frames | done | 30 frames (`images/oldframes`), used as a second qualitative set in the paper (`figures/movies*.jpg`); NOT compared with the old paper |
| Figures | done | `scripts/make_figures.py` -> `paper/figures/` (hand-picked COCO crowd images; `--name movies` for the movie frames, see `run_all.sh`), `diagram.tex` (TikZ, built by `paper/build.sh`) |
| Paper | draft complete | no `\todo` left; see Paper section |

## Results

`outputs/eval/summary*.txt` (final = `summary.txt`), ablation `eval_v1/summary_v1*.txt` (old single-threshold detection; cache kept in `outputs/eval_v1/cache`).

| config | obj occ IoU | hum occ IoU | hum amodal IoU | clean FP |
|---|---|---|---|---|
| single threshold, open 0.01 | 0.382 | 0.400 | 0.572 | 0.042 |
| cross-model, open 0.01 | 0.395 | 0.455 | 0.669 | 0.042 |
| **cross-model, open 0.02 (final)** | 0.385 | 0.452 | 0.669 | 0.021 |
| cross-model, open 0.03 / 0.05 | 0.335 / 0.221 | 0.373 / 0.147 | | 0.013 / 0.004 |

Label accuracy ~0.98. Remaining miss: `000000170191_human` (neither model detects the occluded person).
Timing on T4: detection 1.15 s/img, rasterise+occlusion 0.34 s, LaMa 0.85 s.
Old v1 outputs kept on Drive as `outputs/{modal,body3d,occlusion,compose,eval}_v1`.

## Resume here

```bash
cd /content/occlusion && git pull && bash setup_colab.sh && bash run_all.sh   # all cached, fast
bash paper/build.sh                                                           # rebuild paper/main.pdf
```
**In progress: SAM 3D Body as alternative body model** (user wants both reported; never delete Multi-HMR outputs).
- Code done (`occlusion/body3d_sam3d.py`, `pipeline.detect_topdown`, `OCC_BODY=sam3d` -> `outputs/*_sam3d`). Multi-HMR re-check gave identical numbers.
- Blocked on Hugging Face approval of facebook/sam-3d-body-vith for user `rtous` (requested 2026-11-23). In a new session: `! hf auth login` (device code), then `bash setup_colab.sh` caches the checkpoint in `weights/sam-3d-body-vith`.
- Then: check mesh/image alignment (`OCC_BODY=sam3d python scripts/run_detect.py`, look at `outputs/body3d_sam3d/*.jpg`; verts = pred_vertices + pred_cam_t assumed OpenCV camera frame), run
  `for D in $I $I/crowd $I/oldframes; do OCC_BODY=sam3d python scripts/run_detect.py --images $D; ... done; OCC_BODY=sam3d python scripts/eval.py`,
  tune open_frac for it, add a Multi-HMR vs SAM 3D Body row/table to the paper.

Open items (need the user):
1. Review the paper draft.
2. Target journal: move to its template; set `\showchangesfalse` in `main.tex` to print the blue (new) text in black.
3. Optional method improvement: refine meshes to fit the modal silhouettes (main error source, amodal IoU ~0.7); try SAM 3D Body instead of Multi-HMR.

## Paper (`paper/`)

- **User decision: the old paper is unpublished; do NOT compare with it anywhere.** The movie frames are just another qualitative dataset.
- Authors: Ruben Tous (contact, ruben.tous@upc.edu), Beatriz Otero, Sebastià Nicolau.
- `paper/main.tex` (plain `article` class for now), `paper/references.bib`, `paper/figures/` (all generated; `diagram.tex` source). Build: `bash paper/build.sh` (installs TeX Live on a fresh VM).
- **After every change to the paper, rebuild the PDF and commit `paper/main.pdf` with the sources.**
- Old paper text kept verbatim where still valid; **new/changed text is wrapped in `\new{...}` (blue)**, open items in `\todo{...}` (red; none left). Don't rewrite the user's kept (black) text without asking.
- Content: method with cross-model detection, Tables 1 (results) and 2 (ablation: detection, r_o sweep), COCO and movie-frame figures, de-occlusion examples (cartoon figure removed at user request: plain meshes are not cartoons; old cartoon models not available), timing, related work up to 2026 (all citations verified online: pix2gestalt, Zhan 2024, Xu 2024, SAMBA, Diffusion-VAS, DMAT, Noh & Chang, SAM 3D Body).

## Environment gotchas

- The runtime has sometimes been **CPU-only** despite CLAUDE.md (2 cores, 12 GB). Everything runs on CPU, just slowly: Multi-HMR ~55 s/img, SAM ~30 s/img. Check `torch.cuda.is_available()` and prefer a GPU runtime.
- Python 3.13: `simple-lama-inpainting` can't be installed (pins numpy 1.26 → source build). We load `big-lama.pt` with `torch.jit.load` instead (`occlusion/inpaint.py`).
- Multi-HMR uses relative paths (`./models/...`): `occlusion/body3d.py` chdirs into `/content/third_party/multi-hmr` while loading. It pulls DINOv2 code via `torch.hub` (internet needed on first load). xFormers warnings are harmless.
- Rendering is a small numpy/OpenCV painter's-algorithm rasteriser (`occlusion/render.py`), no OpenGL needed.
- `git push` needs credentials (the user provides a token per session; push with `https://rtous:<token>@github.com/rtous/occlusion.git`, never store it in the repo config); commits may be local only — check `git log origin/main..`. Until pushed, back them up:
  `git bundle create /content/drive/MyDrive/occlusion_work/occlusion_repo.bundle origin/main..main`

## Conventions

- All experiment code goes in the repo (`occlusion/` package + `scripts/`), runnable without redevelopment; commit after every working step.
- Paper: always rebuild `paper/main.pdf` after editing and commit it.
- Steps cache outputs on Drive and are resumable (`--force` to recompute).
