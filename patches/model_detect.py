"""Model type detection for multi-model support.

Detects MiniMax H3, LTX, Wan, and MiniMax Music3 models.
"""


def detect_model_type(inner_model):
    """Detect the model type from the inner model object.

    Returns: ("minimax_h3", "minimax_music3", "ltx", "wan", or "unknown")
    """
    if inner_model is None:
        return "unknown"

    cls_name = type(inner_model).__name__
    module = type(inner_model).__module__

    if "minimax_music" in module.lower() or "Music3" in cls_name:
        return "minimax_music3"

    if "minimax" in module.lower() or "MiniMaxH3" in cls_name:
        return "minimax_h3"

    if "lightricks" in module.lower() or "LTX" in cls_name:
        return "ltx"

    if "wan" in module.lower() or "Wan" in cls_name:
        return "wan"

    if hasattr(inner_model, "layers") and hasattr(inner_model, "project_in"):
        return "minimax_music3"

    if hasattr(inner_model, "blocks"):
        if hasattr(inner_model, "final_layer") or hasattr(inner_model, "token_refiner"):
            return "minimax_h3"
        return "wan"

    if hasattr(inner_model, "transformer_blocks"):
        return "ltx"

    return "unknown"


def get_blocks_attribute(model_type):
    """Return the attribute name that holds the transformer blocks."""
    if model_type == "ltx":
        return "transformer_blocks"
    if model_type == "minimax_music3":
        return "layers"
    return "blocks"


def get_total_blocks(inner_model, model_type):
    """Return the total number of transformer blocks."""
    attr = get_blocks_attribute(model_type)
    blocks = getattr(inner_model, attr, None)
    return len(blocks) if blocks is not None else 0


def supports_token_reduction(model_type):
    """Check if token reduction is supported for this model type."""
    return model_type in ("minimax_h3", "ltx", "wan")
