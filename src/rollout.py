"""
Gradient-weighted Attention Rollout, implemented from scratch (no Grad-CAM /
interpretability library involved anywhere in this file).

Two pieces:
  1. Hooks that capture, per transformer block, the raw post-softmax
     attention matrix (forward pass) and its gradient w.r.t. the predicted
     class score (backward pass).
  2. The rollout math that turns that stack of per-layer attention matrices
     into a single class-relevance map over the input patches.

Everything is a plain function operating on lists of tensors -- no classes.
"""

import torch


def register_attention_hooks(model):
    """
    Attach a forward hook to every block's attention-dropout module.

    Why attn_drop and not something else: timm's Attention.forward only
    calls self.attn_drop(attn) on the "eager" (non-fused) path, and its
    input is exactly the post-softmax (num_heads, N, N) attention matrix we
    need. Hooking its input lets us grab that tensor without touching any
    timm source code, and retain_grad() on it lets a later .backward() call
    populate its .grad even though it isn't a leaf tensor.

    Returns:
        handles       -- hook handles, so they can be removed later.
        attention_maps -- list that gets refilled (in block order) every
                          time a forward pass runs.
    """
    attention_maps = []

    def save_attention(module, inputs, output):
        attn = inputs[0]
        attn.retain_grad()
        attention_maps.append(attn)

    handles = []
    for block in model.blocks:
        handle = block.attn.attn_drop.register_forward_hook(save_attention)
        handles.append(handle)

    return handles, attention_maps


def remove_hooks(handles):
    for handle in handles:
        handle.remove()


def collect_gradients(attention_maps):
    """After a backward() call, pull out each captured tensor's gradient."""
    return [attn.grad for attn in attention_maps]


def fuse_heads(attention, gradients=None, discard_negative=True):
    """
    Collapse the num_heads dimension of one layer's attention matrix down
    to a single (N, N) matrix.

    Plain rollout: average the heads.
    Gradient-weighted rollout: weight each head/position by its ReLU'd
    gradient before averaging, so heads/positions that don't actually help
    the predicted class are suppressed -- the same idea Grad-CAM applies to
    CNN feature-map channels, applied here to attention heads instead.
    """
    if gradients is None:
        return attention.mean(dim=1)

    weights = gradients
    if discard_negative:
        weights = torch.relu(weights)

    weighted = weights * attention
    return weighted.mean(dim=1)


def rollout_matrices(layer_matrices):
    """
    Compose a list of per-layer (N, N) attention matrices (one per
    transformer block, in forward order) into a single (N, N) matrix that
    approximates end-to-end token relevance through the whole network.

    Standard attention-rollout recipe (Abnar & Zuidema, 2020):
      - blend each layer's attention with the identity matrix to account
        for the residual/skip connection that runs alongside attention,
      - row-normalize so each row still sums to 1 (stays a valid
        "distribution of relevance over source tokens"),
      - multiply the layers together in order, so relevance composes
        transitively across depth.
    """
    batch_size, n_tokens, _ = layer_matrices[0].shape
    device = layer_matrices[0].device

    identity = torch.eye(n_tokens, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    result = identity.clone()

    for matrix in layer_matrices:
        blended = 0.5 * matrix + 0.5 * identity
        blended = blended / blended.sum(dim=-1, keepdim=True)
        result = torch.bmm(blended, result)

    return result


def class_relevance_map(attention_maps, gradients=None, cls_index=0):
    """
    Full pipeline: per-layer attention (+ optional gradients) -> fused
    per-layer matrices -> rolled-out matrix -> the row of that matrix that
    says how much each patch token ultimately fed into the CLS token,
    which is what the classification head actually reads.

    Returns a (batch, num_patches) relevance vector with the CLS-to-CLS
    entry dropped, ready to be reshaped into the patch grid.
    """
    if gradients is None:
        gradients = [None] * len(attention_maps)

    layer_matrices = [
        fuse_heads(attn, grad) for attn, grad in zip(attention_maps, gradients)
    ]

    rolled_out = rollout_matrices(layer_matrices)

    cls_row = rolled_out[:, cls_index, :]
    patch_relevance = torch.cat(
        [cls_row[:, :cls_index], cls_row[:, cls_index + 1:]], dim=1
    )
    return patch_relevance
