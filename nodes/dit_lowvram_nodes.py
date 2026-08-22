"""ComfyUI nodes for low-VRAM optimization of DiT models.

Supports: MiniMax H3, MiniMax Music3, LTX, Wan video/audio generation models.
"""

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.ops
import nodes

from ..patches.model_detect import detect_model_type, get_total_blocks, supports_token_reduction
from ..patches.token_reduction import TokenReducer
from ..patches.layer_thinning import (
    compute_block_importance,
    apply_layer_thinning,
    restore_layer_thinning,
    get_optimal_layers,
)
from ..patches.weight_offload import TextEncoderOffloader, DiTBlockOffloader
from ..patches.step_cache import global_step_cache

CATEGORY = "dit-lowvram"

MODEL_INFO = {
    "minimax_h3": {"name": "MiniMax H3", "total_blocks": 50},
    "minimax_music3": {"name": "MiniMax Music3", "total_blocks": 36},
    "ltx": {"name": "LTX", "total_blocks": 28},
    "wan": {"name": "Wan", "total_blocks": 32},
}

AUTO_PRESETS = {
    "off":    {"n_keep_pct": 100, "token_reduction": False, "int8_dit": False, "int8_te": False, "te_offload": False, "step_cache": False},
    "6gb":    {"n_keep_pct": 60,  "token_reduction": True,  "int8_dit": True,  "int8_te": True,  "te_offload": True,  "step_cache": False},
    "8gb":    {"n_keep_pct": 70,  "token_reduction": True,  "int8_dit": True,  "int8_te": True,  "te_offload": True,  "step_cache": False},
    "10gb":   {"n_keep_pct": 80,  "token_reduction": True,  "int8_dit": True,  "int8_te": True,  "te_offload": True,  "step_cache": True},
    "12gb":   {"n_keep_pct": 86,  "token_reduction": True,  "int8_dit": True,  "int8_te": False, "te_offload": True,  "step_cache": True},
    "16gb":   {"n_keep_pct": 90,  "token_reduction": False, "int8_dit": True,  "int8_te": False, "te_offload": False, "step_cache": True},
}


def _fmt_gb(b):
    return "{:.2f} GB".format(b / (1024**3))


def _detect_and_get_total(model):
    """Detect model type and return (model_type, total_blocks)."""
    inner = model.get_model_object("model")
    model_type = detect_model_type(inner)
    total = get_total_blocks(inner, model_type) if inner else 50
    return model_type, total


class DiTLowVRAMOptimizer:
    @classmethod
    def INPUT_TYPES(cls):
        presets = list(AUTO_PRESETS.keys()) + ["manual"]
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "模型输入\n支持: MiniMax H3 / Music3 / LTX / Wan"}),
                "preset": (presets, {"default": "10gb", "tooltip": "按显存预算自动配置\nmanual = 手动设置下方各项"}),
            },
            "optional": {
                "n_keep": ("INT", {"default": 40, "min": 1, "max": 50, "step": 1,
                    "tooltip": "保留DiT层数，越少越省显存\n自动按模型总层数调整比例"}),
                "token_reduction": ("BOOLEAN", {"default": True,
                    "tooltip": "中间层空间token池化\n减少约40%计算量"}),
                "tr_start": ("INT", {"default": 10, "min": 0, "max": 49, "step": 1,
                    "tooltip": "Token降采样起始层"}),
                "tr_end": ("INT", {"default": 40, "min": 1, "max": 50, "step": 1,
                    "tooltip": "Token降采样结束层，此层恢复原始分辨率"}),
                "tr_pool": ("INT", {"default": 2, "min": 2, "max": 4, "step": 2,
                    "tooltip": "池化倍率: 2=减4倍token, 4=减16倍token"}),
                "int8_dit": ("BOOLEAN", {"default": True,
                    "tooltip": "INT8量化DiT权重，省约50%显存"}),
                "int8_te": ("BOOLEAN", {"default": True,
                    "tooltip": "INT8量化文本编码器，省约50%显存"}),
                "te_offload": ("BOOLEAN", {"default": True,
                    "tooltip": "文本编码器逐层卸载到CPU，省约98%显存"}),
                "step_cache": ("BOOLEAN", {"default": True,
                    "tooltip": "跳步缓存：跳过相邻去噪步的transformer计算\n复用缓存结果，加速约1.5-2x\n轻微质量损失，推荐搭配其他优化使用"}),
                "skip_interval": ("INT", {"default": 2, "min": 2, "max": 4, "step": 1,
                    "tooltip": "每隔N步跳过一次计算\n越小加速越大但质量下降越多"}),
                "warmup_steps": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1,
                    "tooltip": "前N步不跳过，让模型稳定\n避免开头质量差"}),
                "noise_scale": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 0.01, "step": 0.0005,
                    "tooltip": "跳步时添加的噪声大小\n防止重复伪影，越大越多样但可能引入噪声"}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = "DiT low-VRAM optimizer + step cache (MiniMax H3/Music3/LTX/Wan)."

    def apply(self, model, preset, n_keep=40, token_reduction=True,
              tr_start=10, tr_end=40, tr_pool=2, int8_dit=True, int8_te=True, te_offload=True,
              step_cache=True, skip_interval=2, warmup_steps=3, noise_scale=0.001):

        model_type, total_blocks = _detect_and_get_total(model)
        model_name = MODEL_INFO.get(model_type, {}).get("name", model_type)

        if preset != "manual":
            p = AUTO_PRESETS[preset]
            n_keep = max(1, int(total_blocks * p["n_keep_pct"] / 100))
            token_reduction = p["token_reduction"]
            int8_dit = p["int8_dit"]
            int8_te = p["int8_te"]
            te_offload = p["te_offload"]
            step_cache = p["step_cache"]

        m = model.clone()
        to = m.model_options.get("transformer_options", {}).copy()
        info_parts = ["[{}]".format(model_name)]

        if int8_dit or int8_te:
            to["h3_int8_quantize"] = True
            to["h3_quantize_dit"] = int8_dit
            to["h3_quantize_text_encoder"] = int8_te
            info_parts.append("INT8(dit={},te={})".format(int8_dit, int8_te))

        to["h3_text_encoder_offload"] = te_offload
        if te_offload:
            info_parts.append("TE Offload")

        if token_reduction and supports_token_reduction(model_type) and n_keep < total_blocks:
            reducer = TokenReducer(enabled=True, start_block=tr_start, end_block=tr_end, pool_factor=tr_pool)
            to["h3_token_reducer"] = reducer
            info_parts.append("TokenReduction({}-{})".format(tr_start, tr_end))

        if n_keep < total_blocks:
            inner = m.get_model_object("model")
            if inner is not None:
                keep_indices = get_optimal_layers(n_keep, inner, model_type)
                apply_layer_thinning(inner, keep_indices, model_type)
                to["h3_thinned_indices"] = keep_indices
                info_parts.append("Layers {}/{}".format(n_keep, total_blocks))

        m.model_options["transformer_options"] = to

        if step_cache:
            m, applied = global_step_cache.enable(m, skip_interval, warmup_steps, noise_scale)
            if applied:
                info_parts.append("StepCache(interval={},warmup={})".format(skip_interval, warmup_steps))

        info = " | ".join(info_parts)
        return (m, info)


class VRAMMonitor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "model": ("MODEL", {"tooltip": "可选：传入模型查看优化状态"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("vram_info",)
    FUNCTION = "monitor"
    CATEGORY = CATEGORY
    DESCRIPTION = "Monitor VRAM usage and active optimizations."

    def monitor(self, model=None):
        lines = ["=== VRAM Monitor ==="]
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_mem
            used = torch.cuda.memory_allocated(0)
            peak = torch.cuda.max_memory_allocated(0)
            lines.append("Total: {}".format(_fmt_gb(total)))
            lines.append("Used:  {}".format(_fmt_gb(used)))
            lines.append("Free:  {}".format(_fmt_gb(total - used)))
            lines.append("Peak:  {}".format(_fmt_gb(peak)))
            torch.cuda.reset_peak_memory_allocated(0)
        else:
            lines.append("CUDA not available")

        if model is not None:
            inner = model.get_model_object("model")
            model_type = detect_model_type(inner)
            model_name = MODEL_INFO.get(model_type, {}).get("name", model_type)
            total_blocks = get_total_blocks(inner, model_type) if inner else 0
            lines.append("Model: {} ({} blocks)".format(model_name, total_blocks))

            to = model.model_options.get("transformer_options", {})
            opts = []
            if to.get("h3_int8_quantize"):
                opts.append("INT8 Quantize")
            if to.get("h3_text_encoder_offload"):
                opts.append("TE Offload")
            if to.get("h3_token_reducer") is not None:
                opts.append("Token Reduction")
            if to.get("h3_thinned_indices") is not None:
                indices = to["h3_thinned_indices"]
                opts.append("Layer Thinning ({}/{})".format(len(indices), total_blocks))

            cache_stats = global_step_cache.get_stats(model)
            if cache_stats.get("enabled"):
                hits = cache_stats.get("hits", 0)
                calls = cache_stats.get("calls", 0)
                rate = cache_stats.get("hit_rate", 0)
                opts.append("Step Cache ({}/{}, {:.0f}%)".format(hits, calls, rate))

            lines.append("Active: {}".format(", ".join(opts)) if opts else "Active: none")

        return ("\n".join(lines),)
