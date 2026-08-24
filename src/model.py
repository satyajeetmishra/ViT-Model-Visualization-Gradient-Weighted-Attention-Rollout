"""
Loading a pretrained Vision Transformer and preparing it so its internal
attention weights can be captured with hooks.

Only plain functions here -- no custom classes. The model object itself is
of course a PyTorch nn.Module (that's the library, not our code), but
everything we write to drive it is procedural.
"""

import torch
import timm
from torchvision.models import ViT_B_16_Weights
from PIL import Image


def load_vit(model_name="vit_base_patch16_224", device="cpu"):
    """
    Load a pretrained ViT classifier from timm and put it in eval mode.

    Important detail: timm's Attention block can use a fused
    scaled_dot_product_attention kernel (flash-attention style) which never
    materializes an explicit (N, N) attention-weight tensor -- so there is
    nothing to hook. We force every block back onto the "eager" attention
    path (fused_attn = False) so the softmax attention matrix actually
    exists as a tensor we can attach hooks to and backprop through.
    """
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    model.to(device)

    for block in model.blocks:
        block.attn.fused_attn = False

    return model


def get_transform(model):
    """Build the exact preprocessing pipeline the pretrained weights expect."""
    config = timm.data.resolve_data_config({}, model=model)
    return timm.data.create_transform(**config)


def load_image(image_path):
    """Load an image from disk as RGB (drops alpha channel / palette modes)."""
    return Image.open(image_path).convert("RGB")


def preprocess_image(image, transform, device="cpu"):
    """Apply the model's transform and add a batch dimension."""
    tensor = transform(image).unsqueeze(0)
    return tensor.to(device)


def get_imagenet_labels():
    """
    Return the 1000 ImageNet class names, in the standard index order.

    Pulled from torchvision's bundled weights metadata (a static list baked
    into the library) rather than downloaded from anywhere at runtime, so
    this works even without a live download of an external labels file.
    """
    return ViT_B_16_Weights.IMAGENET1K_V1.meta["categories"]


def predict(model, input_tensor, top_k=1):
    """
    Run inference and return (logits, top_k class indices, top_k probabilities).
    Kept gradient-enabled on purpose -- rollout.py needs to backprop from the
    chosen class logit through the attention weights.
    """
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = probs.topk(top_k, dim=-1)
    return logits, top_indices[0], top_probs[0]
