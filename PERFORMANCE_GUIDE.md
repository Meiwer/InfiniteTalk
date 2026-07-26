# InfiniteTalk 推理性能优化指南

## 性能瓶颈分析

### 核心问题：`--num_persistent_param_in_dit 0`

该参数设为 0 时，所有 14B 模型参数常驻 CPU，每次前向传播都需要：

1. **`copy.deepcopy`** — 在 GPU 上分配临时内存并复制参数
2. **`.to(device="cuda")`** — CPU→GPU PCIe 传输
3. 计算完成后释放临时内存

相关代码位于 `src/vram_management/layers.py`：

```python
# AutoWrappedModule.forward() — num_persistent=0 时的执行路径
def forward(self, *args, **kwargs):
    if self.onload_device == self.computation_device:
        module = self.module  # 快速路径：参数已在 GPU
    else:
        # 慢速路径：每次前向都 deepcopy + 搬运（性能灾难）
        module = copy.deepcopy(self.module).to(
            dtype=self.computation_dtype, device=self.computation_device
        )
    return module(*args, **kwargs)
```

### 开销估算

| 项目 | 数值 |
|------|------|
| 模型参数量 | 14B（FP8 ≈ 14GB） |
| 每次扩散步 | 3 次前向（CFG: cond / drop_text / uncond） |
| 40 步总计 | 120 次前向传播 |
| 每次前向 PCIe 搬运 | ~14GB |
| 总 PCIe 搬运量 | **~1.5-2 TB** |
| PCIe 4.0 x16 带宽 | ~32 GB/s |
| 纯搬运耗时（理论下限） | ~50 秒 |
| 实际耗时（含 deepcopy + 计算） | **60+ 分钟**（clip 模式 81 帧） |

### GPU 利用率虚高

`nvidia-smi` 显示 98% 利用率，但大部分时间 GPU 在做**内存拷贝**（deepcopy），而非有效推理计算。

---

## 优化方案

### 1. 提高 `num_persistent_param_in_dit`（最关键）

让模型参数常驻 GPU，消除 PCIe 搬运和 deepcopy：

```bash
# 推荐：让全部参数常驻（FP8 模型 ~14GB，RTX 5070 Ti 16.3GB 可容纳）
--num_persistent_param_in_dit 15000000000

# 或者直接不传该参数（模型默认全部在 GPU）
# 去掉 --num_persistent_param_in_dit 0
```

**预期提速：10-20 倍**

> ⚠️ 如果 OOM，逐步降低该值（如 10000000000、7000000000），找到 VRAM 平衡点。

### 2. TeaCache（无需额外下载）

跳过冗余 DiT Block 计算，缓存相邻时间步的残差：

```bash
--use_teacache --teacache_thresh 0.2
```

- 提速约 **1.5-2 倍**
- `teacache_thresh` 越大跳过的步越多（速度更快，质量略降）
- 默认阈值 0.2 是速度与质量的平衡点

### 3. FusionX LoRA（需下载 ~2.5GB）

将采样步数从 40 降至 8，提速约 **5 倍**：

```bash
# 下载
huggingface-cli download vrgamedevgirl84/Wan14BT2VFusioniX \
    FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
    --local-dir weights/

# 运行
--lora_dir weights/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
--lora_scale 1.0 \
--sample_steps 8 \
--sample_shift 2 \
--sample_text_guide_scale 1.0 \
--sample_audio_guide_scale 2.0
```

> 注意：FusionX 会加剧长视频（>1分钟）的色偏，并降低 ID 保持度。

### 4. Lightx2v LoRA（需下载，仅 4 步）

```bash
# 下载
huggingface-cli download Kijai/WanVideo_comfy \
    Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors \
    --local-dir weights/

--lora_dir weights/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors \
--sample_steps 4
```

提速约 **10 倍**，适合快速验证。

### 5. 减少生成帧数（测试用）

```bash
--mode clip              # 只生成 1 个 chunk（81帧 ≈ 3.2秒视频）
--max_frame_num 161      # streaming 模式下限制总帧数
```

### 6. 减少采样步数（质量换速度）

```bash
--sample_steps 20        # 从 40 降到 20，速度翻倍，质量略降
```

---

## 推荐测试命令

### 快速验证（无需额外下载）

```bash
cd /root/InfiniteTalk && .venv/bin/python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir weights/chinese-wav2vec2-base \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode clip \
    --quant fp8 \
    --quant_dir weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors \
    --motion_frame 9 \
    --num_persistent_param_in_dit 15000000000 \
    --use_teacache --teacache_thresh 0.2 \
    --save_file infinitetalk_res_test
```

### 极速测试（FusionX + TeaCache + clip）

```bash
cd /root/InfiniteTalk && .venv/bin/python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir weights/chinese-wav2vec2-base \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 8 \
    --sample_shift 2 \
    --mode clip \
    --quant fp8 \
    --quant_dir weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors \
    --lora_dir weights/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
    --lora_scale 1.0 \
    --sample_text_guide_scale 1.0 \
    --sample_audio_guide_scale 2.0 \
    --motion_frame 9 \
    --num_persistent_param_in_dit 15000000000 \
    --use_teacache --teacache_thresh 0.2 \
    --save_file infinitetalk_res_fast
```

---

## 优化效果对比（预估）

| 配置 | 预计耗时（clip 81帧） | 相对基准 |
|------|------|------|
| 基准：num_persistent=0, 40步 | 60+ 分钟 | 1x |
| num_persistent=15B, 40步 | ~3-5 分钟 | ~15x |
| + TeaCache | ~2-3 分钟 | ~25x |
| + FusionX 8步 | ~30-60 秒 | ~60x |
| + Lightx2v 4步 | ~15-30 秒 | ~120x |

> 以上为 RTX 5070 Ti (16GB) 上的预估值，实际取决于 PCIe 带宽和系统内存。

---

## VRAM 使用参考

| 配置 | VRAM 占用 |
|------|------|
| FP8 模型全部常驻 GPU | ~14-15 GB |
| + 推理激活值 | ~15-16 GB |
| num_persistent=0（参数在 CPU） | ~15 GB（全是激活+临时拷贝） |

RTX 5070 Ti (16.3GB) 可以容纳 FP8 模型全部常驻。如遇 OOM：
1. 降低 `--num_persistent_param_in_dit`（如 10000000000）
2. 使用 `--mode clip` 减少激活值
3. 降低分辨率（480P 而非 720P）
