"""Layer Thinning for video DiT models (MiniMax H3, LTX, Wan).

Ports the h3-metal layer thinning technique: instead of running all DiT blocks,
select the N most important blocks and skip the rest. Works on any model that
stores transformer layers in nn.ModuleList.

Supported models:
- MiniMax H3: model.blocks (50 DiTBlocks)
- LTX: model.transformer_blocks (28 BasicTransformerBlocks)
- Wan: model.blocks (32 WanAttentionBlocks)
"""

import torch
import torch.nn as nn

from .model_detect import detect_model_type, get_blocks_attribute, get_total_blocks


def compute_block_importance(inner_model, model_type):
    """Compute block importance from model weights.

    For MiniMax H3: uses AdaLN gate norms.
    For LTX/Wan: uses attention output projection norms as proxy.
    """
    attr = get_blocks_attribute(model_type)
    blocks = getattr(inner_model, attr, None)
    if blocks is None:
        return []

    importance = []
    for i, block in enumerate(blocks):
        if model_type == "minimax_h3":
            if hasattr(block, "adaln_proj") and hasattr(block.adaln_proj, "linear"):
                score = block.adaln_proj.linear.weight.data.norm().item()
            else:
                score = 1.0
        elif model_type == "ltx":
            if hasattr(block, "attn1") and hasattr(block.attn1, "to_out"):
                score = block.attn1.to_out[0].weight.data.norm().item()
            elif hasattr(block, "ff") and hasattr(block.ff, "net"):
                score = block.ff.net[2].weight.data.norm().item() if len(block.ff.net) > 2 else 1.0
            else:
                score = 1.0
        elif model_type == "wan":
            if hasattr(block, "self_attn") and hasattr(block.self_attn, "to_out"):
                score = block.self_attn.to_out[0].weight.data.norm().item()
            elif hasattr(block, "ff") and hasattr(block.ff, "net"):
                score = block.ff.net[2].weight.data.norm().item() if len(block.ff.net) > 2 else 1.0
            else:
                score = 1.0
        else:
            score = 1.0
        importance.append((i, score))

    importance.sort(key=lambda x: x[1], reverse=True)
    return importance


def get_optimal_layers(n_keep, inner_model=None, model_type="minimax_h3"):
    """Select the optimal N blocks to keep.

    If model is provided, computes ranking from weight norms.
    Otherwise uses a heuristic spread across the network.
    """
    n_total = get_total_blocks(inner_model, model_type) if inner_model else 50

    if n_keep >= n_total:
        return list(range(n_total))

    if inner_model is not None:
        importance = compute_block_importance(inner_model, model_type)
        if importance:
            return sorted([idx for idx, _ in importance[:n_keep]])

    # Heuristic: keep first, last, and spread middle
    indices = [0, n_total - 1]
    remaining = n_keep - 2
    if remaining > 0:
        mid_start = max(1, n_total // 5)
        mid_end = min(n_total - 1, 4 * n_total // 5)
        mid_indices = list(range(mid_start, mid_end))
        step = max(1, len(mid_indices) // remaining)
        indices.extend(mid_indices[::step][:remaining])

    return sorted(list(set(indices)))[:n_keep]


def apply_layer_thinning(inner_model, keep_indices, model_type="minimax_h3"):
    """Apply layer thinning to a model.

    Modifies the blocks attribute in-place to only contain selected blocks.
    Returns the original block count.
    """
    attr = get_blocks_attribute(model_type)
    blocks = getattr(inner_model, attr, None)
    if blocks is None:
        return 0

    n_original = len(blocks)
    sorted_keep = sorted(keep_indices)

    # Store original for restoration
    setattr(inner_model, "_original_blocks_" + attr, blocks)
    inner_model._thinned_indices = sorted_keep
    inner_model._thinned_model_type = model_type

    thinned_blocks = nn.ModuleList([blocks[i] for i in sorted_keep])
    setattr(inner_model, attr, thinned_blocks)

    return n_original


def restore_layer_thinning(inner_model):
    """Restore original blocks after layer thinning."""
    model_type = getattr(inner_model, "_thinned_model_type", None)
    if model_type is None:
        return
    attr = get_blocks_attribute(model_type)
    orig_attr = "_original_blocks_" + attr
    if hasattr(inner_model, orig_attr):
        setattr(inner_model, attr, getattr(inner_model, orig_attr))
        delattr(inner_model, orig_attr)
        if hasattr(inner_model, "_thinned_indices"):
            del inner_model._thinned_indices
        if hasattr(inner_model, "_thinned_model_type"):
            del inner_model._thinned_model_type
