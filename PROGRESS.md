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
| `images/oldframes/` (30 PNG) | the old paper's movie frames (`scripts/fetch_old_frames.sh`); copyrighted, internal comparison only |
| `synth/` | synthetic benchmark (`make_synthetic_eval.py`) |
| `outputs/{modal,body3d,occlusion,compose,eval}` | per-step outputs; `eval/cache/` holds raw network outputs per synthetic image |

## Status (2026-11-23, session 2: GPU T4)

| Milestone | State | Notes |
|---|---|---|
| M1 environment | done | `setup_colab.sh` |
| M2+M3 detection | done | `scripts/run_detect.py` (replaces run_modal/run_body3d): YOLO11m-seg + SAM 2.1-b, Multi-HMR 672_L, **cross-model selection** (`occlusion.select`) + SAM fallback for meshes without mask (`pipeline.detect`) |
| M4 occlusion masks | done | `open_frac` tuned: **0.02** (default now); `dilate_frac` has no effect |
| M5 inpainting + composite | done | LaMa; large holes leave dark smudges (mentioned in paper) |
| M6 synthetic eval | done | 77 images (30 clean, 26 object, 21 human) |
| Old paper frames | done | outputs on Drive (`outputs/{occlusion,compose}/vlcsnap-*`), internal only |
| Figures | done | `scripts/make_figures.py` -> `paper/figures/` (hand-picked COCO crowd images), `diagram.tex` (TikZ) |

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
```
Open items (need the user):
1. Author list (`\todo` in title).
2. Comparison with the previous version on the movie frames: frames are copyrighted; the old paper showed them. Ask whether to include side-by-side figures (outputs exist on Drive) and whether old running times are known.
3. Optionally more 2024-2026 related work; journal template.
Possible method improvements (not required): mesh refinement to fit the modal silhouette (main error source = amodal IoU ~0.7).

## Paper (`paper/`)

- `paper/main.tex` (plain `article` class for now), `paper/references.bib` (old bib + new entries). Build: `bash paper/build.sh`.
- **After every change to the paper, rebuild the PDF (`bash paper/build.sh`) and commit `paper/main.pdf` with the sources.**
- Old paper text kept verbatim where still valid; **new/changed text is wrapped in `\new{...}` (blue)**, open items in `\todo{...}` (red). Don't rewrite kept text without asking the user: they wrote it by hand.
- Figures go in `paper/figures/` (missing ones show as a placeholder box). Needed: occlusion_ex.jpg, diagram.pdf, modal_masks.jpg, meshes.jpg, background.jpg, deocclusion.jpg, qualitative.jpg, cartoon.jpg.
- Tables 1-2 (results, ablation) filled; remaining `\todo`s: authors, comparison with previous version, old running time, optional related work.

## Next steps (old list, items 1-2 done)

1. Get synthetic eval numbers; sweep `--open-frac` (0.01, 0.02, 0.03, 0.05) and `--dilate-frac`; set the best as default.
2. `scripts/make_figures.py`: paper-style grid (original | modal | amodal | occlusion | composite) on `images/crowd`.
3. Optional: other ideas if false positives stay high: only keep object-occlusion components that touch a non-person image edge / strong image edge; or HMR2.0 fallback.
4. Write the paper (later).

## Environment gotchas

- The runtime has sometimes been **CPU-only** despite CLAUDE.md (2 cores, 12 GB). Everything runs on CPU, just slowly: Multi-HMR ~55 s/img, SAM ~30 s/img. Check `torch.cuda.is_available()` and prefer a GPU runtime.
- Python 3.13: `simple-lama-inpainting` can't be installed (pins numpy 1.26 → source build). We load `big-lama.pt` with `torch.jit.load` instead (`occlusion/inpaint.py`).
- Multi-HMR uses relative paths (`./models/...`): `occlusion/body3d.py` chdirs into `/content/third_party/multi-hmr` while loading. It pulls DINOv2 code via `torch.hub` (internet needed on first load). xFormers warnings are harmless.
- Rendering is a small numpy/OpenCV painter's-algorithm rasteriser (`occlusion/render.py`), no OpenGL needed.
- `git push` needs credentials set up by the user in the session (`gh auth login`); commits may be local only — check `git log origin/main..`. Until pushed, back them up:
  `git bundle create /content/drive/MyDrive/occlusion_work/occlusion_repo.bundle origin/main..main`

## Conventions

- All experiment code goes in the repo (`occlusion/` package + `scripts/`), runnable without redevelopment; commit after every working step.
- Paper: always rebuild `paper/main.pdf` after editing and commit it.
- Steps cache outputs on Drive and are resumable (`--force` to recompute).
