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
| 3D body → amodal mask | OpenPose + SMPLify-X | **Multi-HMR** (multi-person SMPL-X + camera, single pass); fallback **HMR2.0 / 4D-Humans**; alternative **SAM 3D Body** (no SMPL license) | no slow optimisation fitting, one forward pass |
| Projection | — | **pyrender** (EGL) or a small z-buffer rasteriser | per-person silhouette + joint depth/ID map |
| Mesh ↔ mask matching | implicit | Hungarian matching on IoU | simple, reliable |
| Background | SD v1.4 inpainting | **LaMa** (`simple-lama-inpainting`), optional SDXL inpainting | fast, deterministic, no prompts |

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
scripts/          run_pipeline.py, make_synthetic_eval.py, eval.py, make_figures.py
data/             image URL and license list (no images committed)
setup_colab.sh    installs everything, caches weights on Drive
```

Temporary files and checkpoints go in `/content/drive/MyDrive/occlusion_work/`.

## Milestones

1. Environment: GPU runtime, `setup_colab.sh`, weights cached on Drive.
2. Modal masks: YOLO11 + SAM 2 on sample images, saved for a visual check.
3. 3D bodies: Multi-HMR (or HMR2.0) meshes rendered and checked for alignment with the people.
4. Occlusion masks: matching, object vs. human labelling, visual outputs.
5. Inpainting and the final de-occlusion composite.
6. Synthetic evaluation (IoU numbers) and the figure grid.

Commit and push after each step that works.

## Open decisions

- Body model: (a) SMPL-X files (free registration at smpl-x.is.tue.mpg.de, put in Drive; recommended), or (b) SAM 3D Body (MHR model, no registration).
- A GPU runtime is required.
