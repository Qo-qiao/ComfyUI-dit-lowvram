# DiT Low VRAM Optimizer (ComfyUI-dit-lowvram)

**[📃中文](./README_zh.md)**

Low-VRAM optimization plugin for ComfyUI's DiT models. Supports **MiniMax H3, MiniMax Music3, LTX, and Wan** video/audio generation models.

Optimization techniques adapted from the [h3-metal](https://github.com/antirez/h3.c) native inference engine.

Run video/audio generation on Windows + NVIDIA GPUs with as little as **8GB VRAM**.

---

## Supported Models

| Model | Type | Layers | Token Reduction | Layer Thinning | INT8 Quant | Step Cache |
|-------|------|--------|-----------------|----------------|------------|------------|
| MiniMax H3 | Video+Audio | 50 | ✅ | ✅ | ✅ | ✅ |
| MiniMax Music3 | Audio | 36 | ❌ | ✅ | ✅ | ✅ |
| LTX | Video | 28 | ✅ | ✅ | ✅ | ✅ |
| Wan | Video | 32 | ✅ | ✅ | ✅ | ✅ |

Model type is auto-detected — no manual selection needed.

---

## Features

- **Auto Model Detection**: Identifies MiniMax H3 / Music3 / LTX / Wan automatically
- **Token Reduction**: Spatial token pooling in middle DiT blocks, ~40% compute savings
- **Layer Thinning**: Skips less important DiT blocks by weight importance, 30-50% weight savings
- **INT8 Quantization**: 8-bit weight quantization for DiT and text encoder, ~50% weight savings
- **Text Encoder Offload**: Layer-by-layer CPU offload, ~98% text encoder VRAM savings
- **Step Caching**: Reuses transformer output across denoising steps, ~1.5-2x speedup
- **Auto Presets**: One-click optimization profiles based on VRAM budget
- **VRAM Monitor**: Real-time GPU memory usage and optimization status

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xxx/ComfyUI-dit-lowvram.git
```

### Dependencies

```bash
cd ComfyUI/custom_nodes/ComfyUI-dit-lowvram
pip install -r requirements.txt
```

Required packages:
- `torch >= 2.1.0`
- `bitsandbytes >= 0.41.0`
- `psutil >= 5.9.0`

---

## Nodes

### 1. DiT Low VRAM Optimizer

All-in-one optimizer combining token reduction, layer thinning, INT8 quantization, text encoder offload, and step caching. Auto-detects model type.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | MODEL | - | Model input (MiniMax H3 / Music3 / LTX / Wan) |
| preset | COMBO | 10gb | Preset mode (see table below) |
| n_keep | INT | 40 | Number of DiT layers to keep (auto-scaled per model) |
| token_reduction | BOOLEAN | True | Enable spatial token pooling |
| tr_start | INT | 10 | Token reduction start layer |
| tr_end | INT | 40 | Token reduction end layer (restores original resolution) |
| tr_pool | INT | 2 | Pooling factor (2 or 4) |
| int8_dit | BOOLEAN | True | INT8 quantize DiT weights |
| int8_te | BOOLEAN | True | INT8 quantize text encoder weights |
| te_offload | BOOLEAN | True | Offload text encoder to CPU layer-by-layer |
| step_cache | BOOLEAN | True | Enable step caching (~1.5-2x speedup) |
| skip_interval | INT | 2 | Steps between cache skips |
| warmup_steps | INT | 3 | Initial steps computed without caching |
| noise_scale | FLOAT | 0.001 | Noise added to cached results |

**Outputs:**
- model: Optimized model
- info: Model type and active optimizations

---

### 2. VRAM Monitor

Displays current GPU VRAM usage, model type, and active optimization status.

| Parameter | Type | Description |
|-----------|------|-------------|
| model | MODEL | Optional: pass model to show optimization status |

**Outputs:**
- vram_info: VRAM usage information

---

## Presets

| Preset | Kept Layers | Token Reduction | INT8 | TE Offload | Step Cache | VRAM |
|--------|-------------|-----------------|------|------------|------------|------|
| **off** | 50/50 | Off | Off | Off | Off | 30GB+ |
| **6gb** | 30/50 | On | Full | On | Off | 6GB |
| **8gb** | 35/50 | On | Full | On | Off | 8GB |
| **10gb** | 40/50 | On | Full | On | On | 10GB |
| **12gb** | 43/50 | On | DiT | On | On | 12GB |
| **16gb** | 45/50 | Off | DiT | Off | On | 16GB |
| **manual** | Manual | Manual | Manual | Manual | Manual | Custom |

---

## How It Works

### Token Reduction

Adapted from h3-metal's `h3_dit.c`. Spatial tokens are average-pooled 2x2 in middle DiT blocks, reducing subsequent compute. Upsampled back to original resolution before the final blocks.

```
[Block 0-9]    Original resolution (100% tokens)
[Block 10-39]  Pooled resolution (25% tokens)   ← 75% compute reduction
[Block 40-49]  Restored to original resolution
```

### Layer Thinning

Computes per-block importance via L2 norm of AdaLN gate weights, keeps the N most important layers. Adapted from h3-metal's `!layers N` command.

### INT8 Quantization

Uses bitsandbytes to quantize DiT and text encoder weights to 8-bit, halving weight memory.

### Text Encoder Offload

Adapted from h3-metal's `h3_text_encoder.c`. Text encoder weights (e.g., 50 layers) stay on CPU; one layer is moved to GPU at a time for processing, then immediately released.

### Step Caching

Adapted from [comfyui-cache-dit](https://github.com/xxx/comfyui-cache-dit). Adjacent denoising steps produce similar transformer outputs. Step caching skips computation on selected steps, reusing the previous result with small noise perturbation to prevent visual artifacts.

```
Step 0:  Compute (warmup)       → cache result
Step 1:  Compute (warmup)       → cache result
Step 2:  Compute (warmup)       → cache result
Step 3:  Skip computation       → return Step2 result + noise
Step 4:  Compute                → cache result
Step 5:  Skip computation       → return Step4 result + noise
...
```

**Key parameters:**
- `skip_interval`: 2 = skip every other step (~2x speedup), 3 = skip every 2 steps
- `warmup_steps`: First N steps always computed
- `noise_scale`: Noise magnitude for artifact prevention, recommended 0.001

---

## Usage Examples

### Basic (auto preset, works with all models)

```
[MiniMax H3 / LTX / Wan Loader]
       ↓
[DiT Low VRAM Optimizer] → preset: 10gb
       ↓
[KSampler] → [VAE Decode] → [Save Video]
```

### Manual Configuration

```
[MiniMax H3 Loader]
       ↓
[DiT Low VRAM Optimizer] → preset: manual
       ↓                    n_keep: 35
                           token_reduction: True
                           int8_dit: True
                           te_offload: True
                           step_cache: True
[KSampler]
```

### Monitor VRAM

```
[VRAM Monitor] ← connect to Optimizer model output
       ↓
[Show Text]
```

---

## VRAM Estimates

| Optimization | Est. VRAM | Quality Impact |
|--------------|-----------|----------------|
| None | ~30 GB+ | None |
| INT8 + 45/50 layers | ~15-18 GB | Minimal |
| INT8 + 40/50 layers + Token Reduction | ~10-12 GB | Low |
| INT8 + 30/50 layers + Token Reduction + TE Offload | ~6-8 GB | Moderate |

---

## Notes

1. **Layer Thinning** skips DiT layers. Fewer layers = more quality loss. Recommended minimum: 30 layers.
2. **Token Reduction** operates in middle layers with minimal quality impact. Extreme settings may lose fine details.
3. **INT8 Quantization** may introduce slight numerical deviations in some scenarios.
4. **Text Encoder Offload** increases encoding time (layer-by-layer processing) but drastically reduces VRAM peak.
5. **Step Caching** provides ~1.5-2x speedup but may add slight noise in fine details. Best combined with other optimizations.
6. First run may require `bitsandbytes` to compile CUDA kernels — please wait.

---

## Compatibility

- ✅ MiniMax H3 video/audio generation
- ✅ MiniMax Music3 music generation
- ✅ LTX video generation
- ✅ Wan video generation
- ✅ ComfyUI native model nodes
- ✅ NVIDIA CUDA GPU (requires bitsandbytes)
- ⚠️ AMD ROCm GPU (some features may be limited)

---

## License

MIT License

---

## Acknowledgments

- [h3-metal](https://github.com/antirez/h3.c) — Salvatore Sanfilippo (antirez)'s MiniMax H3 native inference engine
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — Base framework
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — INT8 quantization support

---

## FAQ

**Q: Which models are supported?**
A: MiniMax H3, MiniMax Music3, LTX, and Wan. Model type is auto-detected.

**Q: I have 12GB VRAM. Which preset should I use?**
A: Use the `12gb` preset, or `manual` mode for fine-tuning.

**Q: Can Layer Thinning and Token Reduction be used together?**
A: Yes. They operate at different levels: layer thinning reduces total layers, token reduction reduces per-layer compute.

**Q: What hardware is needed for INT8 quantization?**
A: NVIDIA GPU with CUDA support. The bitsandbytes library handles INT8 quantization.

**Q: Will optimization reduce quality?**
A: Light presets (16gb/12gb) have negligible impact. Aggressive presets (6gb) may show some detail loss.

**Q: Can this be used with ComfyUI's --lowvram flag?**
A: Yes. They are complementary — ComfyUI's lowvram manages model loading/offloading, this plugin optimizes internal model computation.

**Q: Does step caching affect quality?**
A: Slightly. It reuses previous results with small noise to avoid artifacts. Most scenarios are imperceptible. Adjust `noise_scale` to control noise level.

**Q: Can step caching be combined with other optimizations?**
A: Yes. Step caching is fully compatible with layer thinning, token reduction, INT8 quantization, and all other features.
