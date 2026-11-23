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
| M6 synthetic eval | running / first numbers pending | `object`, `human`, `clean` variants; metrics: occ IoU, false-positive ratio, amodal IoU, label acc |
| Crowd figure grid | todo | `images/crowd` downloaded; need `scripts/make_figures.py` |

## Results so far

- Sample images (`outputs/occlusion`, `outputs/compose`): street scene shows correct cuts (pole, trash bin); tennis crowd shows plausible human-human occlusions.
- **Main failure mode**: mesh pose misfit (mostly legs/arms) shows up as false "object occlusion" slivers (e.g. `bus.jpg`, no real occlusions). Plan: tune `open_frac` (morphological opening, `occlusion/occlusion.py:compute`) on the synthetic benchmark using `clean` false-positive ratio vs `object`/`human` occ IoU:
  `python scripts/eval.py --open-frac 0.03 --tag _open03` (re-scoring is fast, network outputs are cached).
- Synthetic eval numbers: see `outputs/eval/summary*.txt` (copy them here when available).

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
- Steps cache outputs on Drive and are resumable (`--force` to recompute).
