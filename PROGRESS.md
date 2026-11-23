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

## Status (2026-11-23)

| Milestone | State | Notes |
|---|---|---|
| M1 environment | done | `setup_colab.sh` |
| M2 modal masks | done | YOLO11m-seg + SAM 2.1-b box refinement; masks agree ~0.9 IoU |
| M3 3D bodies | done | Multi-HMR 672_L (SMPL-X); projection verified (heads align); poses approximate on tight crops |
| M4 occlusion masks | done, needs tuning | Hungarian IoU matching; human vs object label via mesh depth + other people's visible masks |
| M5 inpainting + composite | done | LaMa TorchScript; composite draws nearest non-occluded mesh per pixel |
| M6 synthetic eval | partial (11/47 scored, CPU run interrupted for GPU switch) | `object`, `human`, `clean` variants; metrics: occ IoU, false-positive ratio, amodal IoU, label acc |
| Old paper frames | partial (modal 17/30, body3d 0/30; interrupted) | same pipeline on the 30 movie frames from the old repo, to compare with the old figures |
| Crowd figure grid | todo | `images/crowd` downloaded; need `scripts/make_figures.py` |

## Results so far

- Sample images (`outputs/occlusion`, `outputs/compose`): street scene shows correct cuts (pole, trash bin); tennis crowd shows plausible human-human occlusions.
- **Main failure mode**: mesh pose misfit (mostly legs/arms) shows up as false "object occlusion" slivers (e.g. `bus.jpg`, no real occlusions). Plan: tune `open_frac` (morphological opening, `occlusion/occlusion.py:compute`) on the synthetic benchmark using `clean` false-positive ratio vs `object`/`human` occ IoU:
  `python scripts/eval.py --open-frac 0.03 --tag _open03` (re-scoring is fast, network outputs are cached).
- Synthetic eval numbers: see `outputs/eval/summary*.txt`. Partial (11 images, `summary_partial.txt`, open_frac 0.01):

  | variant | n | occ IoU | false-pos ratio | amodal IoU | label acc |
  |---|---|---|---|---|---|
  | object | 4 | 0.35 | 0.058 | 0.49 | 1.00 |
  | human | 2 | 0.51 | 0.048 | 0.69 | 0.97 |
  | clean | 5 | n/a | 0.032 | 0.69 | n/a |

  `000000000785_object`: target person not matched to any mesh (occ IoU 0) — investigate.

## Resume here (after switching to the GPU runtime)

```bash
cd /content/occlusion && git pull && bash setup_colab.sh
python -c "import torch; print(torch.cuda.is_available())"     # should be True now
bash run_all.sh     # finishes old frames + synthetic eval (cached parts are skipped)
python scripts/eval.py --cached-only                          # summary of everything scored
for o in 0.0 0.02 0.03 0.05; do python scripts/eval.py --open-frac $o --tag _open$o; done   # tune sliver filter
```
Then: set best `open_frac` default in `occlusion/occlusion.py`, investigate `000000000785_object`, write `scripts/make_figures.py`, fill the paper table/figures, rebuild PDF.

## Paper (`paper/`)

- `paper/main.tex` (plain `article` class for now), `paper/references.bib` (old bib + new entries). Build: `bash paper/build.sh`.
- **After every change to the paper, rebuild the PDF (`bash paper/build.sh`) and commit `paper/main.pdf` with the sources.**
- Old paper text kept verbatim where still valid; **new/changed text is wrapped in `\new{...}` (blue)**, open items in `\todo{...}` (red). Don't rewrite kept text without asking the user: they wrote it by hand.
- Figures go in `paper/figures/` (missing ones show as a placeholder box). Needed: occlusion_ex.jpg, diagram.pdf, modal_masks.jpg, meshes.jpg, background.jpg, deocclusion.jpg, qualitative.jpg, cartoon.jpg.
- Table 1 (synthetic results) and several TODOs wait for the eval numbers.

## Next steps

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
