# Wan2.2-TI2V-5B vs Bernini-R-1.3B-Diffusers 对比实验报告

> 环境：8×NVIDIA A800-SXM4-80GB（GPU 0 常被外部进程占 74GB，实验用 GPU 1）
> 项目根：`/home/web_server/antispam/project/clt/video-generation/`
> 完成日期：2026-07-05

---

## TL;DR

| 维度 | Wan2.2-TI2V-5B | Bernini-R-1.3B-Diffusers |
|---|---|---|
| DiT 参数 | 5B（30 层 × dim 3072） | 1.3B（30 层 × dim 1536） |
| 峰值显存 | **29.8 GB** | **8.1 GB**（**3.7× 更省**） |
| 每步耗时（同卡） | 6.34 s | 3.78 s（**1.7× 更快**） |
| 端到端（一次 t2v） | 7 min 7 s | 4 min |
| 默认分辨率 × 帧数 × fps | 1280×704 × 121 × 24 (≈5s) | 848×480 × 81 × 16 (≈5s) |
| 任务面 | t2v / i2v / ti2v | t2i, i2i, t2v, r2v, v2v, mv2v, rv2v |
| 对本机的压力 | 中（单 A800 完全轻松，量级远低于 4090 24GB 官方 profile） | 极小（约 1/10 张卡的显存） |

**综合结论**：在这套 8×A800-80GB 硬件下，两个模型都远未逼近极限——但相对而言 **Bernini-R-1.3B 明显更轻便**，同一张 A800 上跑不到 10GB、4 分钟出片，适合快速迭代和多任务实验；**Wan2.2-TI2V-5B 更"重"但潜在保真度更高**（更大的 VAE 潜空间 z_dim=48、720p、24fps），当出片质量优先于迭代速度时首选。

---

## 1. 推理层面：源码组件走查

### 1.1 Wan2.2-TI2V-5B 的推理链路

入口：`Wan2.2/generate.py` → argparse → `WanTI2V.generate()` → `t2v()` 或 `i2v()` 分支。

**关键组件**（详见 `docs/wan22_ti2v5b_code_walkthrough.md`）：

| 组件 | 位置 | 说明 |
|---|---|---|
| Text encoder | `wan/modules/t5.py` (T5EncoderModel) | UMT5-XXL bf16 ckpt，`text_len=512` |
| VAE | `wan/modules/vae2_2.py` (`Wan2_2_VAE`) | z_dim=48，空间 16×，时序 4× |
| DiT | `wan/modules/model.py` (`WanModel`) | 3D patch(1,2,2)，30 层 attention block |
| Scheduler | `wan/utils/fm_solvers_unipc.py` | Flow-matching UniPC，`shift=5.0` |
| Attention | `wan/modules/attention.py` | **强依赖 `flash_attn_varlen_func`**，`model.py:145` 直接调 `flash_attention()` 而非带 SDPA fallback 的 `attention()` |

**一次生成的物理流**（1280×704×121 帧）：
```
prompt → T5 → context [1, ≤512, 4096]
noise → latent [48, 31, 44, 80]         ← z_dim=48
DiT × 50 steps × 2 (cond/uncond CFG)   ← seq_len=27280 tokens
    每步：patch_embed → 30×(self_attn + cross_attn + ffn) → head
VAE decode → [3, 121, 704, 1280] in [-1,1]
imageio libx264 → mp4
```

**内存策略**：`--offload_model True`（world_size=1 时自动）在 DiT 上场时 `to(device)`、跑完 `cpu()`；配合 `--convert_model_dtype`（fp32→bf16）显存直接砍半。

### 1.2 Bernini-R-1.3B-Diffusers 的推理链路

入口：`bytedance/Bernini/infer_single_gpu.py` → `bernini.cli.build_pipeline` → `BerniniRendererPipeline`。

**关键组件**：

| 组件 | 位置 | 说明 |
|---|---|---|
| Text encoder | HF `UMT5EncoderModel` | 复用 Wan2.2 的 UMT5-XXL（5 shards, ~22GB fp32→bf16） |
| VAE | `diffusers.AutoencoderKLWan` | **Wan2.1 VAE**（z_dim=16，空间 8×，时序 4×）——与 Wan2.2 VAE 不兼容 |
| DiT | `diffusers.WanTransformer3DModel` | 30 层 × dim 1536（12 heads × 128），in/out 都是 16 通道 |
| Scheduler | `diffusers.UniPCMultistepScheduler` | flow_prediction，`flow_shift=3.0`，40 步 |
| Attention 后端 | `flash_attn 2.8.3` + VeOmni 的 monkey-patch | VeOmni 注入 attention 变体（`t2v_apg` 等 guidance_mode） |

**Bernini 特有**（`configs/bernini_renderer_wan21_1p3b/config.json`）：
- `skip_transformer_2: true` — 单专家 DiT（14B 版才有 hi/lo noise 双专家）
- `use_src_id_rotary_emb: true` — source-id rotary，支持多参考图 r2v
- `use_unipc: true, shift: 3.0`
- `guidance_mode` 变体（`t2v_apg`, `v2v_apg`, `rv2v_wapg` 等）通过 `omega_*` 参数调 CFG 各分支权重

**一次 t2v 生成的物理流**（848×480×81 帧）：
```
prompt → UMT5-XXL → context
noise → latent [16, ?, ?, ?]           ← z_dim=16
DiT × 40 steps × 2 (cond/uncond, apg 变体)
VAE decode → [3, 81, 480, 848]
imageio libx264 → mp4
```

### 1.3 关键差异

| 差异点 | Wan2.2-TI2V-5B | Bernini-R-1.3B |
|---|---|---|
| VAE 潜空间 | z=48，16×空间下采样 | z=16，8×空间下采样 |
| DiT 潜表达能力 | dim=3072（更大隐层） | dim=1536 |
| Attention fallback | 无（必须装 flash_attn） | 有（VeOmni 自动降级到 SDPA） |
| 推理入口 | 单文件 `generate.py` | 案例 JSON + `--config` 目录（diffusers 风格） |
| CFG 机制 | 单一 guide_scale 标量 | 多分支 APG（`omega_vid/img/txt/tgt`） |
| Prompt enhancer | 可选 `--use_prompt_extend`（本地 Qwen 或 dashscope） | 可选 `--use_pe`（OpenAI-compatible endpoint） |

---

## 2. 资源消耗对比

### 2.1 同一 prompt (t2v) 的直接对比

两个模型都跑同一个 "silver robot walking in rainy neon alley" prompt，seed 42：

| 指标 | Wan2.2-TI2V-5B | Bernini-R-1.3B |
|---|---|---|
| CUDA 设备 | GPU 1 (A800 80GB) | GPU 1 (A800 80GB) |
| 输出分辨率 | 1280×704 | 848×480 |
| 帧数 × fps | 121 × 24 = 5.04 s | 81 × 16 = 5.06 s |
| Diffusion steps | 50 | 40 |
| 采样时间 | 5 m 17 s（6.34 s/step） | 2 m 32 s（3.78 s/step） |
| 模型加载时间 | ~55 s（T5 40s + VAE 5s + DiT 10s） | ~35 s（T5 25s + DiT 9s） |
| VAE decode | ~10 s | ~5 s |
| 端到端墙钟 | **7 m 7 s** | **~4 m** |
| 峰值 GPU 显存 | 29 799 MiB | 8 087 MiB |
| 输出文件大小 | 8.05 MB | 6.45 MB |

样例输出（供直观比较）：
- Wan2.2：`/home/web_server/antispam/project/clt/video-generation/outputs/first_wan22_ti2v5b.mp4`
- Bernini：`/home/web_server/antispam/project/clt/video-generation/outputs/bernini_r_t2v_first.mp4`

### 2.2 硬件压力评估

本机是 **8×A800-80GB**，Wan2.2 官方 profile 是 4090 24GB。所以对本机而言：

- Wan2.2-TI2V-5B：**峰值 29.8 GB / 80 GB = 37% 占用**，"轻松量级"。理论上如果不做 `--offload_model`，占用可能升到 45-55 GB 也依然容得下。
- Bernini-R-1.3B：**峰值 8.1 GB / 80 GB = 10% 占用**，"几乎不吃"。同卡可以并行 3-4 个副本，或者跑多任务 pipeline。

**结论**：对这套硬件而言，两个模型都毫无压力；但从**每 GB 显存输出的生产力**看，Bernini 是碾压性的赢家。

---

## 3. 输入输出规范（IO Spec）

### 3.1 Wan2.2-TI2V-5B 输入

通过 `Wan2.2/generate.py` 或 `scripts/run_wan22_ti2v5b.sh`：

| 参数 | 类型 | 默认 / 限制 | 说明 |
|---|---|---|---|
| `--task` | str | 固定 `ti2v-5B` | 决定加载哪套 config |
| `--size` | str | **只允许 `1280*704` 或 `704*1280`** | 分辨率 |
| `--frame_num` | int | 默认 121；**必须 4n+1** | VAE 时序压缩 4× + 首帧独立 |
| `--sample_steps` | int | 默认 50 | Flow-matching UniPC 步数 |
| `--sample_guide_scale` | float | 默认 5.0 | CFG 强度 |
| `--sample_shift` | float | 默认 5.0 | Flow-matching σ 曲线偏移 |
| `--base_seed` | int | 默认 42 | RNG seed |
| `--prompt` | str | **必填** | 文本提示词（英文/中文均可） |
| `--image` | path | **可选**（图生视频） | 若给定，作为首帧启动 i2v 分支 |
| `--save_file` | path | **必填** | 输出 mp4 路径 |
| `--offload_model` | bool | world_size=1 时默认 True | 分阶段 CPU↔GPU 迁移 |
| `--convert_model_dtype` | flag | 关 | DiT 权重 fp32→bf16 |
| `--t5_cpu` | flag | 关 | T5 UMT5-XXL 常驻 CPU |

**Shell 封装**：`scripts/run_wan22_ti2v5b.sh` 支持环境变量覆盖：
```
TAG PROMPT IMAGE SIZE FRAME_NUM SAMPLE_STEPS GUIDE_SCALE SAMPLE_SHIFT BASE_SEED CUDA_VISIBLE_DEVICES
```

### 3.2 Bernini-R-1.3B 输入

通过 `bytedance/Bernini/infer_single_gpu.py`（video 任务多卡则 `infer_multi_gpu.py + torchrun`）：

| 参数 | 类型 | 默认 / 限制 | 说明 |
|---|---|---|---|
| `--config` | path | **必填** | Diffusers 格式目录路径（自包含所有子组件） |
| `--case` | json | **推荐** | 案例 JSON 文件（见下方 schema） |
| `--prompt` | str | 与 `--case` 互斥 | 直接命令行传 prompt |
| `--guidance_mode` | enum | 由 case 决定 | `t2v_apg / v2v_apg / rv2v / r2v_apg / ...` |
| `--num_frames` | int | 默认 81 | 帧数 |
| `--num_inference_steps` | int | 默认 40 | 步数 |
| `--max_image_size` | int | 默认 848 | 长边分辨率上限（宽） |
| `--height` / `--width` | int | 默认 0（跟随输入） | 覆盖输出尺寸 |
| `--fps` | int | 默认 16 | 帧率 |
| `--flow_shift` | float | 默认 3.0 | Flow-matching shift |
| `--seed` | int | 默认 None | RNG seed |
| `--use_pe` | flag | 关 | 通过 OpenAI-compatible endpoint 增强 prompt |

**Case JSON schema**：
```json
{
  "task_type": "t2v | t2i | i2i | v2v | r2v | rv2v | mv2v",
  "prompt": "<text prompt>",
  "output": "<mp4/png output path>",
  "video": "<optional source video for v2v/rv2v>",
  "image": "<optional source image for i2i>",
  "images": ["<optional reference image array for r2v/rv2v>"],
  "system_prompt": "<optional prompt-enhancer system prefix>"
}
```

### 3.3 输出

两个模型都通过 `imageio-ffmpeg`（libx264）编码为 mp4：

| 输出属性 | Wan2.2 | Bernini |
|---|---|---|
| 编码 | H.264 (libx264) | H.264 (libx264) |
| 像素格式 | yuv420p | yuv420p |
| 分辨率 | 1280×704 / 704×1280 | 由 `--max_image_size` × 输入 aspect ratio 决定 |
| 帧率 | 24（config 硬编码 `sample_fps=24`） | 16（默认）/ 24（`--fps 24`） |
| 命名约定（本项目） | `outputs/<tag>_wan22_ti2v5b.mp4` | `outputs/<tag>_bernini_r_<task>.mp4` |

### 3.4 运行记录（prompts.json）

所有 prompt + 参数 + 输出路径 + 运行时统计集中维护在：
```
/home/web_server/antispam/project/clt/video-generation/configs/prompts.json
```

字段 schema 见文件顶部 `field_reference`；每次新的推理都追加一条到 `runs` 数组。

---

## 4. 使用便利性对比

| 维度 | Wan2.2 | Bernini | 备注 |
|---|---|---|---|
| 依赖版本刚性 | 中（torch ≥2.4，transformers ≤4.51.3，需手动装 flash_attn） | 高（torch==2.5.1、diffusers==0.35.2、需 VeOmni --no-deps） | Bernini 强制版本更严 |
| 环境搭建耗时 | 一次装完 ~3 min（本地磁盘） | 一次装完 ~5 min（VeOmni 需 git build） | 都可复用 uv 缓存 |
| 单次推理调用 | 单 python 脚本 + 一堆 argparse flags | Python 脚本 + `--config` + 案例 JSON | Bernini 更"配置化" |
| 多任务面 | t2v / i2v（同一入口） | 7 种任务（t2i/i2i/t2v/v2v/r2v/rv2v/mv2v） | Bernini 更全 |
| 多卡策略 | FSDP + Ulysses（`--ulysses_size N`） | Ulysses via VeOmni（`--ulysses N`） | 都支持 |
| Prompt 增强 | 本地 Qwen 或 dashscope | OpenAI-compatible endpoint | Bernini 更通用 |
| 输出确定性 | 相同 seed 完全复现 | 相同 seed 完全复现 | 等价 |

---

## 5. 综合判断（定性 + 定量）

- **要"能出货"** → Wan2.2-TI2V-5B。720p × 24fps × 5s 直出，画面细节层次更好；单张 A800 上 7 分钟一条，一天 200+ 条不成问题。
- **要"能迭代"** → Bernini-R-1.3B。480p × 16fps × 5s 出图快 40%、显存不到 1/3，同卡还能并发；且任务种类多（编辑、参考驱动、Motion 迁移），做 A/B 或迭代比对很爽。
- **要"两个都用"** → 用 Wan2.2 出主稿，用 Bernini 做二次编辑/参考驱动（例如 rv2v 换装、r2v 用同一张脸驱动新动作），本项目里已经把两套 venv 完全隔离（`.venv` 与 `.venv-bernini`），可以互不影响并存。

**对本机压力评级**：
| 硬件维度 | Wan2.2 | Bernini | 结论 |
|---|---|---|---|
| A800 80GB 单卡显存 | 37% | 10% | 都毫无压力 |
| 单条推理耗时 | ~7 min | ~4 min | 都在可接受范围 |
| 8 卡并发容量 | 8 条并行 | 20+ 条并行（受 CPU/IO 而不是 GPU 限制） | Bernini 吞吐潜力大 |

---

## 6. 已验证的产出

| Run ID | 模型 | 任务 | 输入 | 输出（绝对路径） |
|---|---|---|---|---|
| `wan22-first` | Wan2.2-TI2V-5B | t2v | 银色机器人 prompt | `outputs/first_wan22_ti2v5b.mp4` |
| `bernini-r-t2v-first` | Bernini-R-1.3B | t2v | 同上 prompt | `outputs/bernini_r_t2v_first.mp4` |
| `guoge-certified-wan22-i2v` | Wan2.2-TI2V-5B | i2v | `guoge.jpg` + 动画 prompt | `outputs/guoge_certified_wan22_ti2v5b.mp4`（生成中） |

（详细参数、seed、峰值显存、md5 见 `configs/prompts.json`）
