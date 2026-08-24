# ViT Model Visualization — Gradient-Weighted Attention Rollout

This project implements the take-home assignment: given a pretrained Vision
Transformer (ViT) image classifier and an input image, produce a visualization
showing which regions of the image drove the model's prediction — without
using any existing Grad-CAM or interpretability library.

## What the assignment asked for

- Use PyTorch, with a pretrained ViT image classifier.
- No existing Grad-CAM / model-interpretability library.
- Produce an overlay visualization of the model's attention or class-relevant
  regions on the original image.
- A README covering: approach, key implementation decisions, assumptions,
  how the method was adapted to ViT specifically, and how correctness was
  validated.
- Submit source code + run instructions.

## Approach

The classic CNN approach (Grad-CAM) works by taking gradients of the target
class score with respect to the last convolutional layer's feature maps,
which still have spatial structure (`channels x H x W`). ViT has no such
layer — after patch embedding, the model only ever sees a sequence of
tokens (one per image patch, plus a CLS token), mixed together by
self-attention. There's nothing shaped like a spatial feature map to take
gradients against.

Instead, this project implements **Gradient-weighted Attention Rollout**,
a method built specifically around ViT's architecture:

1. **Attention Rollout** (Abnar & Zuidema, 2020) composes the per-layer
   attention matrices across all transformer blocks into one matrix that
   approximates how much each input patch ultimately influences the
   final CLS token — the token the classification head actually reads.
   Because of residual/skip connections, each layer's attention matrix is
   blended with the identity matrix before being row-normalized and
   chained together via matrix multiplication.
2. Plain rollout is **class-agnostic** — it only reflects "where the model
   generally looks," identical regardless of which class you ask about.
   To make it class-specific, each layer's attention matrix is weighted by
   the (ReLU'd) gradient of the target class's logit with respect to that
   attention matrix before rollout — the same idea Grad-CAM applies to CNN
   feature-map channels, applied here to attention weights instead.
3. The CLS row of the final rolled-out matrix gives one relevance value
   per patch. That vector is reshaped into the ViT's patch grid (14x14 for
   a 224x224 image with 16x16 patches), upsampled back to image resolution,
   and blended over the original image as a heatmap.

## Project structure

```
vit-gradient-rollout/
├── requirements.txt
├── images/            # put input images here
├── outputs/           # overlay results are written here
└── src/
    ├── model.py        # load pretrained ViT, image loading/preprocessing
    ├── rollout.py       # the rollout algorithm itself (hooks + math)
    ├── visualize.py      # relevance vector -> heatmap overlay image
    └── main.py            # CLI entry point
```

## How to run

```
pip install -r requirements.txt
python src/main.py --image images/your_image.jpg --output outputs/overlay.png
```

The first run downloads the pretrained ViT weights (~330MB, via `timm`) and
caches them locally; later runs are instant.

Optional: to populate `images/` with a ready-made batch of 100 real photos
(10 each from 10 diverse ImageNet classes) for validation instead of
sourcing your own, run:

```
python src/get_sample_images.py
```

To process every image in a folder in one go instead of one at a time:

```
python src/main.py --image-dir images --output-dir outputs
```

Useful flags:
- `--model <timm model name>` — defaults to `vit_base_patch16_224`. Any
  square-patch-grid timm ViT should work.
- `--target-class <imagenet index>` — explain an arbitrary class instead of
  the model's own top-1 prediction (useful for checking the map actually
  changes when you ask about a different class).
- `--plain-rollout` — run ordinary (non-gradient-weighted) attention
  rollout instead, for comparison.
- `--device cuda` — run on GPU instead of CPU, if available.

## Model used

`vit_base_patch16_224` (ViT-Base/16), loaded via `timm` with ImageNet-1k
pretrained weights. This is the standard/original ViT size (12 layers, 768
hidden dim, 12 heads, ~86M parameters) — chosen because it's the most
commonly benchmarked ViT variant, making results easy to sanity-check
against known model behavior.

## Important implementation decisions

- **Disabling fused attention.** timm's `Attention` module can run a fused
  `scaled_dot_product_attention` kernel that never materializes an explicit
  attention-weight tensor, so there is nothing to hook or backprop through.
  Every block's `fused_attn` flag is set to `False` after loading the model
  so the "eager" attention path runs instead, giving us a real `(heads, N, N)`
  tensor to hook.
  **Why this helps the goal:** the entire visualization method depends on
  being able to see and differentiate through each layer's attention
  weights. Without this, the fused kernel would silently skip computing
  them at all, and there would be nothing to build a heatmap from.
- **Hooking `attn_drop`.** Rather than modifying timm's source, a forward
  hook is attached to each block's attention-dropout module. Its input is
  exactly the post-softmax attention matrix. `retain_grad()` is called on
  it so a later `.backward()` populates its `.grad`, even though it isn't a
  leaf tensor.
  **Why this helps the goal:** this is what makes the method
  *gradient-weighted* rather than plain rollout. Capturing both the
  attention values and their gradient with respect to the predicted class
  is exactly what lets the heatmap become class-specific — showing where
  the model looked *for this particular prediction*, not just where it
  looks in general.

## Assumptions

- Square input images / square patch grids (true for the standard
  `*_patch16_224` family of models).
- One image per run (batch size 1).
- Default explained class is the model's own top-1 prediction, unless
  `--target-class` is given.

## Validation

The pipeline was tested on a sample of 100 images from ImageNet (via the
Imagenette subset — 10 real photos each from 10 diverse classes, a mix of
animals and everyday objects), run in one batch via `python src/main.py
--image-dir images --output-dir outputs`. Across this set, the
class-relevant/important region of each image — the actual object driving
the prediction (e.g. the animal's body, the handheld tool, the building) —
was correctly highlighted in the output overlay images.

A minor, known limitation also observed: a small amount of extra heat
sometimes appears on flat background regions (sky, grass, blank walls) too
— a documented ViT phenomenon called "attention sinks" (see "Vision
Transformers Need Registers", Darcet et al. 2023), not a bug specific to
this implementation.
