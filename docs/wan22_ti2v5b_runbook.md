# Wan2.2-TI2V-5B on RTX 4090 24GB — Runbook

## 0. TL;DR

| 项 | 状态 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 24GB (driver 535.54.03, CUDA 12.2) |
| Python venv | `/home/web_server/antispam/project/clt/wan2.2/.venv`（uv-managed，Python 3.11.15）|
| Torch | 2.4.0+cu121 （从本机 `/root/venv-vllm-parent/venv-tf/` 复用整个 torch + nvidia + triton wheel 副本）|
| 代码 | `Wan2.2/`（`git clone https://github.com/Wan-Video/Wan2.2.git`, `wan/__init__.py` 有一处非破坏性 patch，见 §7）|
| 模型 | ⚠️ **仍在后台下载中**（ModelScope，见 §4），~34 GB 总量，实测 ~0.8-1 MB/s → ETA ~10-12h |
| 首次生成 | ⏳ **未执行**——模型未就绪。启动命令见 §5 |
| 峰值显存 | 待记录（模型就绪后运行 `scripts/run_wan22_ti2v5b_4090.sh` 时自动 sample 到 `logs/gpu_mem.csv`）|
| 输出视频 | 计划 `outputs/first_wan22_ti2v5b_4090.mp4` |

## 1. 环境事实

```
$ nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
NVIDIA GeForce RTX 4090, 24564 MiB, 24214 MiB, 535.54.03      # 释放后
CUDA (driver): 12.2                                            # nvcc 是 11.4，无关；PyTorch 用自带 cu121 runtime
$ .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
2.4.0+cu121 True
```

内存 1 TB / free 380G+；磁盘可用 4 TB（NFS 挂载 `10.80.193.230,10.80.193.231,10.80.193.232:/kcsonline/...`）。

## 2. 关键决策记录

- **Python 3.11 而不是 3.10**：本机 `/root/venv-vllm-parent/venv-tf/` 已装好完整 torch 2.4.0+cu121 stack（4.3 GB，含 `torch/`, `torchgen/`, `nvidia/*`, `triton/`），直接文件级 `cp -a` 到我们 venv 可以完全绕开 PyTorch cu121 wheel 的下载（~2GB，走 squid 出海代理时 40 分钟仅完成 9s CPU）。源 venv 是 3.11，因此我们 venv 也是 3.11。
- **不用 uv pip 装库**：`uv pip install` 在这台机器上单包能卡 >2 分钟，改用 `.venv/bin/python -m pip install`（走内部 pypi-proxy 可用）。
- **`wan/__init__.py` 打了一个非破坏性 patch**：把 `WanS2V` / `WanAnimate` 改成可选导入。原因是 `librosa` 安装 5 分钟仍超时（依赖树巨大），而 ti2v-5B 完全用不到它。恢复原始行为只需 `git checkout wan/__init__.py`。
- **模型下载改用 ModelScope 而不是 hf-cli**：（1）`oversea-squid2.ko.txyun:11080` 对 HF CDN body 掐流（大文件流跑不起来）；（2）`hf-cli` 被 kill 时会把 `.incomplete` 提升成 final 文件（本次曾造成 VAE 只有 15% 就被 rename 成 `Wan2.2_VAE.pth`）。ModelScope 走国内源，实测 0.8-1 MB/s，自带正确断点续传。

## 3. 依赖清单

已装（来自 `.venv/bin/python -m pip list`，节选）：

```
torch                    2.4.0+cu121   (copy from venv-tf)
torchvision              0.19.1+cu121
torchaudio               2.11.0        (--no-deps, ti2v-5B 用不到但装了)
numpy                    1.26.4
opencv-python            4.11.0.86
diffusers                0.38.0
transformers             4.51.3        (Wan2.2 要求 <=4.51.3)
tokenizers               0.21.4
accelerate               1.14.0
tqdm                     4.68.3
imageio                  2.37.3
imageio-ffmpeg           0.6.0
easydict                 1.13
ftfy                     6.3.1
dashscope                1.26.2
einops                   0.8.2
decord                   0.6.0         (s2v 传递依赖，import 需要)
huggingface_hub          (自带)
```

未装：`flash_attn`（跳过；`wan/modules/attention.py` 有 torch scaled_dot_product_attention fallback），`librosa`（s2v 才需要）。

## 4. 模型下载

**当前活跃的下载方式：ModelScope CLI**（`modelscope==1.38.0`）。

ModelScope 的对象存储在国内，走 kwai 出海 squid 代理**没有像 HF CDN 那样的 body 限流**——实测并行 4 流合计 ~0.8-1 MB/s，比 HF 提速 ~10 倍。

启动命令（已在跑）：

```bash
cd /home/web_server/antispam/project/clt/wan2.2
setsid nohup ./.venv/bin/modelscope download \
    --model Wan-AI/Wan2.2-TI2V-5B \
    --local_dir ./Wan2.2-TI2V-5B \
    > logs/modelscope_download.log 2>&1 < /dev/null &
disown
```

进度查看：

```bash
du -sh Wan2.2-TI2V-5B/
tail -f logs/modelscope_download.log       # 4 并行进度条
pgrep -af modelscope
```

**期望最终大小 ~34 GB**（HF/ModelScope 一致，通过 HEAD `content-length` 验证）：

| 文件 | 大小 |
|---|---|
| `models_t5_umt5-xxl-enc-bf16.pth` | **11.4 GB** |
| `diffusion_pytorch_model-00001-of-00003.safetensors` | **9.83 GB** |
| `diffusion_pytorch_model-00002-of-00003.safetensors` | **10.0 GB** |
| `diffusion_pytorch_model-00003-of-00003.safetensors` | **179 MB** |
| `Wan2.2_VAE.pth` | **2.82 GB** |
| tokenizer + assets + config | ~20 MB |

**ModelScope CLI 自带断点续传**，中断后重跑同样命令即可。**不用**再手动 kill 或清 lock。

### 备选：hf-mirror + hf_transfer（本机验证效果不如 ModelScope）

如果 ModelScope 卡了，可以切回 HF-mirror + Rust 加速：

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=60
./.venv/bin/huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
    --local-dir ./Wan2.2-TI2V-5B --resume-download
```

**注意 hf_transfer 不支持断点续传**——不带 `.incomplete` 文件的 resume，只能重头下。仅在真的稳能拉起 MB/s 时用。

### 已废弃：自写 safe_hf_download.py

早期 HF 走 squid 代理只有 ~95 KB/s，写过一个 `scripts/safe_hf_download.py`（4MiB range GET + `.part` 原子 rename）。ModelScope 可用后已停用。文件保留作为参考。

## 5. 首次生成命令

模型下载完成（`Wan2.2-TI2V-5B/*.pth` 和 3 个 `*.safetensors` 都非空且尺寸正确）后：

```bash
bash /home/web_server/antispam/project/clt/wan2.2/scripts/run_wan22_ti2v5b_4090.sh
```

底层等价命令：

```bash
export PATH=/home/web_server/antispam/project/clt/wan2.2/.venv/bin:$PATH
export VIRTUAL_ENV=/home/web_server/antispam/project/clt/wan2.2/.venv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd /home/web_server/antispam/project/clt/wan2.2/Wan2.2

python generate.py \
  --task ti2v-5B \
  --size 1280*704 \
  --ckpt_dir /home/web_server/antispam/project/clt/wan2.2/Wan2.2-TI2V-5B \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --save_file /home/web_server/antispam/project/clt/wan2.2/outputs/first_wan22_ti2v5b_4090.mp4 \
  --prompt "A cinematic tracking shot of a small silver robot walking through a rainy neon alley at night, wet pavement reflections, soft volumetric light, realistic camera movement, detailed background, smooth motion, high quality, no text, no watermark."
```

**参数含义**（详见 `docs/wan22_ti2v5b_code_walkthrough.md`）：

| flag | 作用 |
|---|---|
| `--task ti2v-5B` | 走 `wan.WanTI2V` pipeline（配置见 `wan/configs/wan_ti2v_5B.py`）|
| `--size 1280*704` | 5B 仅允许 `1280*704` / `704*1280`|
| `--offload_model True` | DiT/T5 分阶段迁 CPU，只有当前阶段占 GPU|
| `--convert_model_dtype` | DiT 权重 fp32 → bf16，直接砍半|
| `--t5_cpu` | T5 UMT5-XXL 全程 CPU（~10GB 常驻内存但不吃显存）|

默认 `frame_num=121`（≈5s @ 24fps），`sample_steps=50`，`guide_scale=5.0`，`sample_shift=5.0`。

**如果 OOM**：先确认三个 flag 都开了；再降 `--sample_steps 30`；smoke test 用 `--frame_num 81 --sample_steps 20`。

## 6. 期望性能（官方参考）

- 4090 24GB，720p (`1280*704`)，121 frames，50 steps → **~9 分钟量级**
- 峰值显存 offloading 下预期 **<22 GB**

实际数值：模型下载完成后运行 `scripts/run_wan22_ti2v5b_4090.sh` 会在 `logs/first_run.log` 里记录 `/usr/bin/time -v` 输出（wall time / max RSS），并在 `logs/gpu_mem.csv` 里以 2Hz 采样 `nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu`。脚本结尾自动打印峰值显存。

## 7. 变更清单（可回滚）

1. `Wan2.2/wan/__init__.py` — 把 `from .speech2video import WanS2V` 和 `from .animate import WanAnimate` 包成 `try/except ImportError: WanS2V=None; WanAnimate=None`。回滚：`git -C Wan2.2 checkout wan/__init__.py`。

其他所有文件都是新增，不修改任何已有仓库代码。

## 8. 遇到的问题 & 解决

| 问题 | 排查 | 解决 |
|---|---|---|
| GPU 起始时 18.5GB 已被占 | `nvidia-smi --query-compute-apps` 看到 `dflash.benchmark`（Qwen3-8B）pid | 用户授权 `kill 47654 47721` |
| Python 3.8 太旧 | Wan2.2 要求 3.10+ | `uv venv --python 3.11 .venv` |
| torch 2.4.1 cu121 wheel 出海代理下太慢（40min 只跑了 9s CPU） | `/proc/<pid>/net/tcp` 显示很多连接但接近零流量 | 从本机 `/root/venv-vllm-parent/venv-tf/` 拷 `torch/torchgen/nvidia/triton` 到新 venv |
| `uv pip install` 单个 30KB 包超时 30s | verbose 也没输出，卡在 resolver | 改用 `.venv/bin/python -m pip install`（走 kwai 内部 pypi-proxy 可行）|
| `librosa` pip 装 5min 超时 | 依赖树过大（scipy/numba/lazy_loader…）| 把 `WanS2V` / `WanAnimate` 改可选导入 |
| HF CDN (`us.aws.cdn.hf.co`) 大文件被 squid 掐流 | `curl --range 0-52428800` timeout 25s 得 0 字节；但 `--range 0-1048575` 能拿满 1MB@150KB/s；`curl -I` HEAD 完全正常，返回 `content-length: 2818839170` 和签名 CDN URL | 换 ModelScope CLI（国内源），并行 4 流合计 ~0.8-1 MB/s，比 HF 提速 ~10 倍 |
| `hf-cli` 被 kill 后把 `.incomplete` 提升成 final，得到截断文件 | 看到 `Wan2.2_VAE.pth` 只有 426MB（应 2.8GB），5 个大文件都被截断 | `rm` 掉截断文件；改用 ModelScope（`modelscope download` 自带正确断点续传） |
| `hf_transfer` 反而清空已下的 `.incomplete` | log 里 `Removing incomplete file ... (hf_transfer=True)` | 关掉：`HF_HUB_ENABLE_HF_TRANSFER=0` |

## 9. 目录索引

```
/home/web_server/antispam/project/clt/wan2.2/
├── .venv/                                  # uv-managed py3.11 venv
├── Wan2.2/                                 # git clone of Wan-Video/Wan2.2
│   └── wan/__init__.py                     # <-- non-destructive patch (§7)
├── Wan2.2-TI2V-5B/                         # HF model dir (in-progress)
├── scripts/
│   ├── run_wan22_ti2v5b_4090.sh            # T2V 首次生成
│   ├── safe_hf_download.py                 # 已废弃备用：4MiB range GET + .part 原子 rename
│   └── download_watchdog.sh                # hf-cli 版 watchdog（已弃用，见 §2 后半段解释）
├── logs/
│   ├── model_download.log                  # 早期 hf-cli 日志（含调试痕迹）
│   ├── safe_download.log                   # 当前活跃下载日志
│   ├── watchdog.log                        # 旧尝试
│   └── first_run.log                       # 生成时才写
├── outputs/                                # 视频输出目录
└── docs/
    ├── wan22_ti2v5b_runbook.md             # 本文
    └── wan22_ti2v5b_code_walkthrough.md    # 代码 + Mermaid 数据流
```

## 10. 下一步（模型下载完毕后）

1. 确认 5 个大文件 size 正确（比对上表）：
   ```bash
   ls -la Wan2.2-TI2V-5B/*.pth Wan2.2-TI2V-5B/*.safetensors
   ```
2. 停下载器：`pkill -f modelscope` （若切回 HF：`pkill -f huggingface-cli`）
3. 跑首次生成：`bash scripts/run_wan22_ti2v5b_4090.sh`
4. 回填本文档 §0 的"峰值显存"和"总耗时"字段。
