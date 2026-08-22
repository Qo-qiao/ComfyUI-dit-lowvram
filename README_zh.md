# DiT 低显存优化插件 (ComfyUI-dit-lowvram)

为 ComfyUI 提供的 DiT 模型低显存优化插件，支持 **MiniMax H3 / MiniMax Music3 / LTX / Wan** 四大视频/音频生成模型。

核心优化技术参考 [h3-metal](https://github.com/antirez/h3.c) 原生加速引擎移植。

在 Windows + NVIDIA GPU 上实现低显存推理，**8GB 显存即可运行**。

---

## 支持的模型

| 模型 | 类型 | 总层数 | Token降采样 | 层裁剪 | INT8量化 | 跳步缓存 |
|------|------|--------|------------|--------|---------|---------|
| MiniMax H3 | 视频+音频 | 50 | ✅ | ✅ | ✅ | ✅ |
| MiniMax Music3 | 音频 | 36 | ❌ | ✅ | ✅ | ✅ |
| LTX | 视频 | 28 | ✅ | ✅ | ✅ | ✅ |
| Wan | 视频 | 32 | ✅ | ✅ | ✅ | ✅ |

插件自动检测模型类型，无需手动选择。

---

## 功能特性

- **自动模型检测**: 自动识别 MiniMax H3 / Music3 / LTX / Wan 模型
- **Token 降采样**: DiT 中间层空间 token 池化，减少约 40% 计算量
- **层裁剪**: 基于权重重要性排名跳过不重要的 DiT 层，减少 30-50% 权重显存
- **INT8 量化**: DiT 和文本编码器的 8-bit 权重量化，节省约 50% 权重显存
- **文本编码器卸载**: 逐层 CPU 卸载，节省约 98% 文本编码器显存
- **跳步缓存**: 跳过相邻去噪步的 transformer 计算，复用缓存结果，加速约 1.5-2x
- **自动预设**: 根据显存预算一键配置最优优化组合
- **显存监控**: 实时查看 GPU 显存使用和优化状态

---

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xxx/ComfyUI-dit-lowvram.git
```

### 依赖安装

```bash
cd ComfyUI/custom_nodes/ComfyUI-dit-lowvram
pip install -r requirements.txt
```

依赖项：
- `torch >= 2.1.0`
- `bitsandbytes >= 0.41.0`
- `psutil >= 5.9.0`

---

## 节点说明

### 1. DiT Low VRAM Optimizer

合一低显存优化器，集成了 Token 降采样、层裁剪、INT8 量化、文本编码器卸载和跳步缓存。自动检测模型类型。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | MODEL | - | 模型输入 (支持 MiniMax H3 / Music3 / LTX / Wan) |
| preset | COMBO | 10gb | 预设模式（见下表） |
| n_keep | INT | 40 | 保留的 DiT 层数（按模型自动调整） |
| token_reduction | BOOLEAN | True | 启用 Token 降采样 |
| tr_start | INT | 10 | 降采样起始层 |
| tr_end | INT | 40 | 降采样结束层 |
| tr_pool | INT | 2 | 池化倍率（2 或 4） |
| int8_dit | BOOLEAN | True | INT8 量化 DiT |
| int8_te | BOOLEAN | True | INT8 量化文本编码器 |
| te_offload | BOOLEAN | True | 文本编码器逐层卸载 |
| step_cache | BOOLEAN | True | 启用跳步缓存（加速约1.5-2x） |
| skip_interval | INT | 2 | 跳步间隔（每隔N步跳过一次） |
| warmup_steps | INT | 3 | 预热步数（前N步不跳过） |
| noise_scale | FLOAT | 0.001 | 跳步时添加的噪声大小 |

**输出:**
- model: 优化后的模型
- info: 模型类型和优化信息

---

### 2. VRAM Monitor

GPU 显存监控节点，显示当前显存使用量、模型类型和已启用的优化状态。

| 参数 | 类型 | 说明 |
|------|------|------|
| model | MODEL | 可选：输入模型查看优化状态 |

**输出:**
- vram_info: 显存使用信息

---

## 预设模式

| 预设 | 保留层 | Token降采样 | INT8 | TE卸载 | 跳步缓存 | 适用显存 |
|------|--------|------------|------|--------|---------|---------|
| **off** | 50/50 | 关 | 关 | 关 | 关 | 30GB+ |
| **6gb** | 30/50 | 开 | 全开 | 开 | 关 | 6GB |
| **8gb** | 35/50 | 开 | 全开 | 开 | 关 | 8GB |
| **10gb** | 40/50 | 开 | 全开 | 开 | 开 | 10GB |
| **12gb** | 43/50 | 开 | DiT | 开 | 开 | 12GB |
| **16gb** | 45/50 | 关 | DiT | 关 | 开 | 16GB |
| **manual** | 手动 | 手动 | 手动 | 手动 | 手动 | 自定义 |

---

## 优化技术原理

### Token 降采样

参考 h3-metal 的 `h3_dit.c` 实现。在 DiT 中间层将空间 token 进行 2x2 平均池化，减少后续层的计算量，最后在最终层前上采样恢复原始分辨率。

```
[Block 0-9]    原始分辨率 (100% token)
[Block 10-39]  池化分辨率 (25% token)   ← 计算量减少 75%
[Block 40-49]  恢复原始分辨率
```

### 层裁剪

基于 AdaLN 门控权重的 L2 范数计算每层重要性，保留最重要的 N/50 层。参考 h3-metal 的 `!layers N` 命令。

### INT8 量化

使用 bitsandbytes 对 DiT 和文本编码器权重进行 INT8 量化，权重体积减半。

### 文本编码器卸载

参考 h3-metal 的 `h3_text_encoder.c` 逐层处理策略。Qwen3-VL 50 层权重保留在 CPU，每次只将 1 层搬到 GPU 处理后立即释放。

### 跳步缓存 (Step Caching)

参考 [comfyui-cache-dit](https://github.com/xxx/comfyui-cache-dit)。在去噪循环中，相邻步的 transformer 输出高度相似。跳步缓存跳过部分步的计算，复用上一步结果 + 微小噪声扰动，避免重复伪影。

```
Step 0:  正常计算 (warmup)     → 缓存结果
Step 1:  正常计算 (warmup)     → 缓存结果
Step 2:  正常计算 (warmup)     → 缓存结果
Step 3:  跳过计算              → 返回 Step2 结果 + 噪声
Step 4:  正常计算              → 缓存结果
Step 5:  跳过计算              → 返回 Step4 结果 + 噪声
...
```

**关键参数:**
- `skip_interval`: 跳步间隔，2 = 每隔1步跳过1次（约2x加速），3 = 每隔2步跳过1次
- `warmup_steps`: 预热步数，前N步始终计算
- `noise_scale`: 噪声系数，防止视觉伪影，建议 0.001

---

## 使用示例

### 基础用法（自动预设，支持所有模型）

```
[MiniMax H3 / LTX / Wan Loader]
       ↓
[Low VRAM Optimizer] → preset: 10gb
       ↓
[KSampler] → [VAE Decode] → [Save Video]
```

### 手动配置

```
[MiniMax H3 Loader]
       ↓
[H3 Low VRAM Optimizer] → preset: manual
       ↓                    n_keep: 35
                           token_reduction: True
                           int8_dit: True
                           te_offload: True
[KSampler]
```

### 监控显存

```
[H3 Memory Monitor] ← 连接到 Optimizer 输出的 model
       ↓
[Show Text]
```

---

## 显存估算

| 优化组合 | 预估显存 | 质量影响 |
|---------|---------|---------|
| 无优化 | ~30 GB+ | 无 |
| INT8 + 45/50 层 | ~15-18 GB | 极小 |
| INT8 + 40/50 层 + Token 降采样 | ~10-12 GB | 较小 |
| INT8 + 30/50 层 + Token 降采样 + TE 卸载 | ~6-8 GB | 中等 |

---

## 注意事项

1. **层裁剪**会跳过部分 DiT 层，层数越少质量损失越明显，建议不低于 30 层
2. **Token 降采样**在中间层工作，对画质影响较小，但极端设置可能导致细节丢失
3. **INT8 量化**可能在某些场景下引入轻微的数值偏差
4. **文本编码器卸载**会增加编码时间（逐层处理），但大幅降低显存峰值
5. **跳步缓存**可加速约1.5-2x，但可能在细节上引入轻微噪声，建议搭配其他优化使用
6. 首次运行时 `bitsandbytes` 可能需要编译 CUDA kernel，请耐心等待

---

## 兼容性

- ✅ MiniMax H3 视频/音频生成
- ✅ MiniMax Music3 音乐生成
- ✅ LTX 视频生成
- ✅ Wan 视频生成
- ✅ ComfyUI 原生模型节点
- ✅ NVIDIA CUDA GPU（需要 bitsandbytes 支持）
- ⚠️ AMD ROCm GPU（部分功能可能受限）

---

## 许可证

MIT License

---

## 致谢

- [h3-metal](https://github.com/antirez/h3.c) - Salvatore Sanfilippo (antirez) 的 MiniMax H3 原生推理引擎
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 基础框架
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - INT8 量化支持

---

## 常见问题

**Q: 支持哪些模型？**
A: 支持 MiniMax H3、MiniMax Music3、LTX 和 Wan 四大视频/音频生成模型。插件自动检测模型类型。

**Q: 我的显存有 12GB，应该选哪个预设？**
A: 选择 `12gb` 预设，或使用 `manual` 模式微调参数。

**Q: 层裁剪和 Token 降采样可以同时使用吗？**
A: 可以，它们作用于不同层面：层裁剪减少总层数，Token 降采样减少每层的计算量。

**Q: INT8 量化需要什么硬件？**
A: 需要支持 CUDA 的 NVIDIA GPU，bitsandbytes 库提供 INT8 量化支持。

**Q: 优化后质量会下降吗？**
A: 轻度优化（如 16gb/12gb 预设）几乎无感知。激进优化（如 6gb 预设）在细节上可能有一定损失。

**Q: 可以和 ComfyUI 的 --lowvram 参数一起用吗？**
A: 可以，两者互补。ComfyUI 的 lowvram 管理模型加载/卸载，本插件优化模型内部计算。

**Q: 跳步缓存会影响质量吗？**
A: 轻微影响。跳步缓存通过复用上一步结果 + 微小噪声来避免伪影，大部分场景下几乎无感知。可通过调整 `noise_scale` 控制噪声大小。

**Q: 跳步缓存和其他优化可以同时使用吗？**
A: 可以。跳步缓存与层裁剪、Token 降采样、INT8 量化等完全兼容，可以叠加使用。
