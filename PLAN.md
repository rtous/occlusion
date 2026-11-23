# Plan: refreshed Human Body Occlusion Mask Estimation

Goal: a working, simple re-implementation of the method in `old_paper.tex`, with deprecated tools replaced. Good-enough results, not SOTA.

## Original method (old_paper.tex)

Per person: **occlusion mask = amodal mask − modal mask**

- Modal mask: Binary People Segmentation (U-Net, EfficientNet-B3).
- Amodal mask: 2D projection of the 3D body fitted with OpenPose + SMPLify-X (ORCNN was tried and discarded).
- Human–human occlusion: the occluding person's mesh is superimposed over the occluded one.
- Background: dilated modal mask + Stable Diffusion v1.4 inpainting.
- Application: render the meshes over the inpainted background, erasing the occluded parts (cartoonization).

## New pipeline

| Step | Old | New | Why |
|---|---|---|---|
| Modal mask (per instance) | People Segmentation (binary) | **YOLO11-seg** (ultralytics) for instances + **SAM 2** box-prompted refinement | pip-installable, robust, per-instance masks (needed for human–human occlusion) |
| 3D body → amodal mask | OpenPose + SMPLify-X | **SAM 3D Body** (final: per person, prompted with box + mask, MHR mesh); **Multi-HMR** (multi-person SMPL-X, single pass) as second person detector and comparison | no slow optimisation fitting; SAM 3D Body fits best |
| Projection | — | **pyrender** (EGL) or a small z-buffer rasteriser | per-person silhouette + joint depth/ID map |
| Mesh ↔ mask matching | implicit | Hungarian matching on IoU, with cross-model detection selection (low thresholds) and SAM fallback | simple, reliable; recovers heavily occluded people |
| Background | SD v1.4 inpainting | **LaMa** (`big-lama.pt` via TorchScript) | fast, deterministic, no prompts |

Occlusion mask for person i:

```
occ_i = amodal_i AND NOT modal_i
```

Each occluded pixel is labelled:
- *human-occluded* if another mesh lies in front of it in the joint depth map,
- *object-occluded* otherwise.

`modal_i` is clipped to a dilated `amodal_i` first, which removes stray segmentation pixels.

## Evaluation

- **Qualitative**: about 30 freely licensed images with crowds and occlusions (e.g. OCHuman / COCO val), shown as a figure grid like the paper's.
- **Quantitative, synthetic**: take non-occluded people (modal mask = ground-truth amodal), paste random object or person cut-outs over them to get a known occlusion mask, and report IoU of the predicted occlusion mask.
- Optional: AHP / COCOA amodal ground truth, if accessible.

## Repo layout

```
occlusion/        modal.py, body3d.py, render.py, occlusion.py, inpaint.py, compose.py
scripts/          run_detect.py, run_occlusion.py, run_compose.py, make_synthetic_eval.py, eval.py, make_figures.py
data/             image URL and license list (no images committed)
setup_colab.sh    installs everything, caches weights on Drive
```

Temporary files and checkpoints go in `/content/drive/MyDrive/occlusion_work/`.

## Milestones (all done as of 2026-11-23; see PROGRESS.md)

1. Environment: GPU runtime, `setup_colab.sh`, weights cached on Drive.
2. Modal masks: YOLO11 + SAM 2 on sample images, saved for a visual check.
3. 3D bodies: Multi-HMR first, then SAM 3D Body (final); meshes rendered and checked for alignment with the people.
4. Occlusion masks: matching, object vs. human labelling, visual outputs.
5. Inpainting and the final de-occlusion composite.
6. Synthetic evaluation (IoU numbers) and the figure grid.

Commit and push after each step that works.

## Decisions taken

- Body model: SAM 3D Body (main method, 2026-11-23); Multi-HMR (SMPL-X files on Drive) as second detector and comparison.
- GPU runtime (T4); everything also runs on CPU, slowly.
