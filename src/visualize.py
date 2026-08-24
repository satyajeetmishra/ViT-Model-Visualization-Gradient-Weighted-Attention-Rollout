"""
Turning a (num_patches,) relevance vector into an overlay image.

Plain functions only. Uses numpy/matplotlib/PIL -- general-purpose
image/array libraries, not an interpretability library.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.cm as cm
from PIL import Image


def relevance_to_grid(relevance):
    """
    Reshape a flat (num_patches,) relevance vector into its square patch
    grid, e.g. 196 patches -> 14x14. Assumes a square image / square patch
    grid, true for the standard vit_base_patch16_224 setup.
    """
    relevance = relevance.detach().cpu().numpy().reshape(-1)
    grid_size = int(math.sqrt(relevance.shape[0]))
    assert grid_size * grid_size == relevance.shape[0], (
        f"Relevance vector of length {relevance.shape[0]} is not a perfect "
        f"square -- can't reshape into a patch grid."
    )
    return relevance.reshape(grid_size, grid_size)


def normalize_map(grid):
    """Min-max normalize to [0, 1] so it can be used as a heatmap/alpha."""
    grid = grid - grid.min()
    max_val = grid.max()
    if max_val > 0:
        grid = grid / max_val
    return grid


def upsample_grid(grid, output_size):
    """
    Bilinearly upsample the coarse patch-grid map to the original image
    resolution (output_size = (height, width)).
    """
    tensor = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(
        tensor, size=output_size, mode="bilinear", align_corners=False
    )
    return upsampled.squeeze().numpy()


def overlay_heatmap(image, heatmap, alpha=0.5, colormap="jet"):
    """
    Blend a normalized [0, 1] heatmap (same H, W as the image) over the
    original PIL image and return the result as a PIL image.
    """
    image_np = np.array(image).astype(np.float32) / 255.0

    cmap = cm.get_cmap(colormap)
    colored_heatmap = cmap(heatmap)[:, :, :3]  # drop alpha channel from colormap

    blended = (1 - alpha) * image_np + alpha * colored_heatmap
    blended = np.clip(blended, 0, 1)

    return Image.fromarray((blended * 255).astype(np.uint8))


def build_overlay(image, relevance_vector, alpha=0.5, colormap="jet"):
    """
    Full pipeline: flat relevance vector -> square grid -> normalize ->
    upsample to image size -> blend over the original image.
    """
    grid = relevance_to_grid(relevance_vector)
    grid = normalize_map(grid)
    heatmap = upsample_grid(grid, output_size=(image.height, image.width))
    heatmap = normalize_map(heatmap)  # renormalize post-interpolation
    return overlay_heatmap(image, heatmap, alpha=alpha, colormap=colormap)


def save_image(image, output_path):
    image.save(output_path)
    return output_path
