"""ComfyUI DiT Low VRAM: Low-VRAM optimization for DiT models.

Supports: MiniMax H3, MiniMax Music3, LTX, and Wan video/audio generation models.

Optimization techniques:
- Token Reduction: spatial token pooling in middle DiT blocks
- Layer Thinning: skip less important DiT blocks
- INT8 Quantization: 8-bit weight quantization
- Text Encoder Offload: layer-by-layer CPU offload
- Step Caching: reuse transformer output across denoising steps
"""

from .nodes.dit_lowvram_nodes import DiTLowVRAMOptimizer, VRAMMonitor

NODE_CLASS_MAPPINGS = {
    "DiTLowVRAMOptimizer": DiTLowVRAMOptimizer,
    "VRAMMonitor": VRAMMonitor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiTLowVRAMOptimizer": "DiT Low VRAM Optimizer",
    "VRAMMonitor": "VRAM Monitor",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
