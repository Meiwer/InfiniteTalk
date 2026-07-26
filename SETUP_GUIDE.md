# InfiniteTalk 环境适配指南（RTX 5070 Ti / Blackwell 架构）

## 背景

项目原始要求为 PyTorch 2.4.1 + CUDA 12.1，但 RTX 5070 Ti 为 Blackwell 架构（sm_120），
PyTorch 2.4.1 仅支持到 sm_90。因此需要一系列适配才能跑通推理流程。

## 硬件环境

- GPU: NVIDIA GeForce RTX 5070 Ti (sm_120, Blackwell)
- CUDA Driver: 13.2
- 系统 CUDA Toolkit: /usr/local/cuda-13.3
- OS: Ubuntu 24.04

## 适配步骤

### 1. 创建 Python 3.10 虚拟环境

```bash
pyenv install 3.10.20
~/.pyenv/versions/3.10.20/bin/python -m venv /root/InfiniteTalk/.venv
source /root/InfiniteTalk/.venv/bin/activate
```

### 2. 安装 PyTorch（必须 2.11+ 以支持 sm_120）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> 实际安装版本: PyTorch 2.11.0+cu128, CUDA 12.8

### 3. 安装 xformers（匹配 PyTorch 版本）

```bash
pip install xformers --index-url https://download.pytorch.org/whl/cu128
```

> 实际安装版本: xformers 0.0.35

### 4. 安装 flash-attn（需跳过 CUDA 版本检查）

系统 nvcc 为 13.3，PyTorch 编译使用 CUDA 12.8，主版本号不同会触发 RuntimeError。

**修复方法**: 修改 `.venv/lib/python3.10/site-packages/torch/utils/cpp_extension.py`

将第 544-545 行:
```python
if cuda_ver.major != torch_cuda_version.major:
    raise RuntimeError(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)
```
改为:
```python
if cuda_ver.major != torch_cuda_version.major:
    logger.warning(CUDA_MISMATCH_WARN, cuda_str_version, torch.version.cuda)
```

然后编译安装（耗时约 40 分钟）:
```bash
pip install flash-attn --no-build-isolation
```

> 实际安装版本: flash-attn 2.8.3.post1

### 5. 安装 requirements.txt 及额外依赖

```bash
pip install -r requirements.txt
pip install wheel librosa soundfile "misaki[en]" ninja
```

### 6. 降级 diffusers 和 transformers（兼容 PyTorch 2.11）

```bash
pip install diffusers==0.33.0
pip install transformers==4.49.0 "tokenizers>=0.20.3,<0.22"
```

> - diffusers 0.39+ 的 attention_dispatch.py 与 PyTorch 2.4 不兼容，但 0.33 满足 xfuser 要求
> - transformers 5.x 引入了 DTensor 依赖，与当前环境不兼容

### 7. 升级 optimum-quanto

```bash
pip install --upgrade optimum-quanto  # 0.2.6 → 0.2.7
```

### 8. 修补 PyTorch C++ 头文件（CUDA 13.3 编译兼容）

optimum-quanto JIT 编译 CUDA 扩展时，PyTorch 头文件报错。

**修复方法**: 修改 `.venv/lib/python3.10/site-packages/torch/include/ATen/core/List_inl.h` 第 202 行:

```cpp
// 原始:
return {impl_->list.begin() + static_cast<typename decltype(impl_->list)::difference_type>(pos)};
// 改为:
return {impl_->list.begin() + static_cast<std::ptrdiff_t>(pos)};
```

### 9. 修补 xformers attention 的 dtype 兼容性

xformers 的 `memory_efficient_attention` 在 sm_120 上不支持 float32（fa3/cutlass 均不支持该架构的 fp32）。

**修复方法**: 修改 `wan/modules/attention.py`，在两处 xformers 调用前将张量转为 bfloat16:

```python
# 原始:
x = xformers.ops.memory_efficient_attention(q, encoder_k, encoder_v, attn_bias=attn_bias, op=None,)

# 改为:
_out_dtype = q.dtype
x = xformers.ops.memory_efficient_attention(
    q.to(torch.bfloat16), encoder_k.to(torch.bfloat16), encoder_v.to(torch.bfloat16),
    attn_bias=attn_bias, op=None,
).to(_out_dtype)
```

（共两处：SingleStreamAttention.forward 和 SingleStreamMutiAttention.forward）

### 10. 模型软链接

```bash
mkdir -p weights
ln -sf /root/.cache/modelscope/models/Wan-AI--Wan2.1-I2V-14B-480P/snapshots/master weights/Wan2.1-I2V-14B-480P
ln -sf /root/.cache/modelscope/models/TencentGameMate--chinese-wav2vec2-base/snapshots/master weights/chinese-wav2vec2-base
ln -sf /root/.cache/modelscope/models/MeiGen-AI--InfiniteTalk/snapshots/master weights/InfiniteTalk
```

### 11. 安装 FFmpeg

```bash
apt-get install -y ffmpeg
```

## 最终依赖版本汇总

| 包 | 版本 |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu128 |
| xformers | 0.0.35 |
| flash-attn | 2.8.3.post1 |
| diffusers | 0.33.0 |
| transformers | 4.49.0 |
| optimum-quanto | 0.2.7 |
| numpy | 1.26.4 |

## 运行命令示例（FP8 单人推理）

```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir weights/chinese-wav2vec2-base \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --quant fp8 \
    --quant_dir weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors \
    --motion_frame 9 \
    --num_persistent_param_in_dit 0 \
    --save_file infinitetalk_res_quant
```

## 注意事项

- flash-attn 编译约需 40 分钟，需确保 ninja 已安装
- streaming 模式 1000 帧 + 40 步采样在单卡 RTX 5070 Ti (16GB) 上耗时较长
- `--num_persistent_param_in_dit 0` 可显著降低显存占用
- 步骤 4、8、9 的补丁在重新安装对应包后需要重新应用
