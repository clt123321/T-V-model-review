# Wan2.2-TI2V-5B vs Bernini-R-1.3B Video Generation Review

> 对比 **阿里 Wan2.2-TI2V-5B**（5B DiT，Wan2.2 VAE，UMT5-XXL）与 **ByteDance Bernini-R-1.3B-Diffusers**（1.3B DiT，Wan2.1 VAE，UMT5-XXL）在同一台机器上的 **文生视频 / 图生视频** 推理能力。含推理源码走查、资源开销对比、IO 规范、Web UI 展示，并给出他机可复现流程。

- 📊 **对比报告**：[`docs/model_comparison_report.md`](docs/model_comparison_report.md)
- 🎥 **样例产出**：`outputs/first_wan22_ti2v5b.mp4` (Wan2.2 720p) · `outputs/bernini_r_t2v_first.mp4` (Bernini 480p) — 同一 prompt "silver robot walking in rainy neon alley"
- 🌐 **Web UI**：`python app.py` → `http://<host>:15856`
- 🤖 **Agent Playbook**：[`docs/agent_constraints.md`](docs/agent_constraints.md) · [`docs/agent_actions_log.md`](docs/agent_actions_log.md)

---

## TL;DR

| | Wan2.2-TI2V-5B | Bernini-R-1.3B |
|---|---|---|
| DiT 参数 | 5 B | 1.3 B |
| 峰值 GPU 显存 | **29.8 GB** | **8.1 GB** |
| 一次 t2v 端到端墙钟 | ≈ 7 min | ≈ 4 min |
| 分辨率 × 帧数 × fps | 1280×704 × 121 × 24 | 848×480 × 81 × 16 |
| 任务面 | t2v / i2v / ti2v | t2i / i2i / t2v / v2v / r2v / rv2v / mv2v |
| 依赖版本 | torch 2.4.1+cu121, transformers ≤4.51.3, flash_attn 2.7.4 | torch 2.5.1+cu124, diffusers 0.35.2, transformers 4.57.3, flash_attn 2.8.3, VeOmni 0.1.10 |

**结论**：在 A800-80GB 上两者都毫无压力，但相对而言 **Bernini 更轻便**（显存 1/4、耗时 4/7），适合快速迭代与多任务实验；**Wan2.2 更"重"但保真度上限更高**（720p/24fps、VAE 潜空间 z=48）。详见 [`docs/model_comparison_report.md`](docs/model_comparison_report.md)。

---

## 硬件要求

| 场景 | 最低 | 推荐 |
|---|---|---|
| 仅浏览 Web UI | 任意 x86_64，2 GB RAM | 4 GB RAM |
| Bernini t2v/i2i/... | 1× 12 GB GPU | 1× 24 GB |
| Wan2.2 t2v/i2v | 1× 24 GB GPU（需 `--offload_model` `--t5_cpu` `--convert_model_dtype`） | 1× 48 GB+ |
| 多卡 Ulysses | 4× 24 GB+ | 8× 40 GB+ |

CUDA driver ≥ 535，CUDA runtime 12.1+。磁盘 ≥ 100 GB。

---

## 快速开始（他机复现）

### 1. 拉代码

```bash
git clone https://github.com/clt123321/T-V-model-review.git video-generation
cd video-generation
```

### 2. 拉权重（合计 ~64 GB）

```bash
pip install -U "huggingface_hub[cli]"
hf download Wan-AI/Wan2.2-TI2V-5B              --local-dir Wan2.2-TI2V-5B
hf download ByteDance/Bernini-R-1.3B-Diffusers --local-dir Bernini-R-1.3B-Diffusers
```

> **首次下载后强烈建议 SHA256 校验**——ModelScope 曾出现过"预分配文件但内容全 NULL"的情况（本项目 Bernini 就中过枪）。校验方法见 [`docs/agent_actions_log.md` §阶段 5](docs/agent_actions_log.md)。

### 3. 拉两份官方推理代码

```bash
git clone https://github.com/Wan-Video/Wan2.2.git    Wan2.2
git clone https://github.com/bytedance/Bernini.git   /root/Bernini
```

Wan2.2 需要一处非破坏性 patch（`wan/__init__.py`）以让 `WanS2V`/`WanAnimate` 变可选导入，跳过 librosa 依赖。见 [`docs/wan22_ti2v5b_runbook.md` §7](docs/wan22_ti2v5b_runbook.md)。

### 4. 建两个隔离 venv（**必须在本地盘**，不要放 ceph/NFS）

```bash
# --- Wan2.2 venv ---
uv venv --python 3.11 /root/wan22-venv
ln -sfn /root/wan22-venv .venv
/root/wan22-venv/bin/pip install -r requirements-wan22.txt
# flash_attn 手动拉预编译 wheel（避免 30 分钟本地编译）
curl -LO https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
/root/wan22-venv/bin/pip install --no-deps flash_attn-2.7.4.post1+*.whl

# --- Bernini venv ---
uv venv --python 3.11 /root/bernini-venv
ln -sfn /root/bernini-venv .venv-bernini
/root/bernini-venv/bin/pip install -r requirements-bernini.txt
curl -LO https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
/root/bernini-venv/bin/pip install --no-deps flash_attn-2.8.3+*.whl
/root/bernini-venv/bin/pip install --no-deps git+https://github.com/ByteDance-Seed/VeOmni.git@v0.1.10
```

> **注意**：两个 venv 的 torch 版本不兼容，务必隔离。

### 5. 启动 Web UI

```bash
uv venv --python 3.11 /root/app-venv
/root/app-venv/bin/pip install -r requirements-app.txt
/root/app-venv/bin/python app.py
```

打开浏览器：`http://<your-host>:15856`

Web UI 包含 5 个 Tab：Video Gallery / Comparison Report / Prompt Registry / Reproduce / Agent Playbook。

### 6. 命令行推理

```bash
# --- Wan2.2 文生视频 ---
CUDA_VISIBLE_DEVICES=1 TAG=demo \
  PROMPT="A cinematic shot of a cat walking on the moon." \
  bash scripts/run_wan22_ti2v5b.sh

# --- Wan2.2 图生视频 ---
CUDA_VISIBLE_DEVICES=1 TAG=demo \
  IMAGE=/abs/path/to/photo.jpg SIZE=704*1280 \
  PROMPT="The person waves and smiles at the camera." \
  bash scripts/run_wan22_ti2v5b.sh

# --- Bernini 文生视频 ---
CUDA_VISIBLE_DEVICES=1 TAG=demo \
  PROMPT="A cinematic shot of a cat walking on the moon." \
  bash scripts/run_bernini_r.sh

# --- Bernini 图像编辑 / 参考驱动视频 ---
# 见 scripts/run_bernini_r.sh 头部注释和 configs/bernini_cases/ 案例
```

每次推理会：
- 输出 mp4 到 `outputs/${TAG}_<model>_<task>.mp4`
- 写日志到 `logs/${TAG}_*.log`（含 2Hz GPU 显存采样 CSV）
- 记录一条到 `configs/prompts.json`（需手动追加，或在 UI 里操作）

---

## 目录结构

```
video-generation/
├── app.py                      ← Gradio Web UI (port 15856)
├── requirements-app.txt        ← 前端所需最小依赖
├── requirements-wan22.txt      ← Wan2.2 环境依赖清单
├── requirements-bernini.txt    ← Bernini 环境依赖清单
├── configs/
│   ├── prompts.json            ← 所有 run 的中央注册（schema 见文件顶部）
│   └── bernini_cases/          ← Bernini 案例 JSON
├── docs/
│   ├── model_comparison_report.md    ← 对比报告 + IO 规范
│   ├── agent_constraints.md          ← Agent 硬/软约束 + 命令模板
│   ├── agent_actions_log.md          ← 时间线 + 已发现的坑
│   ├── wan22_ti2v5b_runbook.md       ← Wan2.2 部署 runbook
│   └── wan22_ti2v5b_code_walkthrough.md ← Wan2.2 源码走查（含 Mermaid）
├── scripts/
│   ├── run_wan22_ti2v5b.sh     ← Wan2.2 t2v/i2v 封装
│   └── run_bernini_r.sh        ← Bernini 全任务封装
├── outputs/                    ← mp4 输出
├── logs/                       ← 运行日志 + GPU 采样
└── .gitignore                  ← 排除模型权重、上游 clone、私密媒体
```

## Web UI 截图指引

Tab 1 · **Video Gallery**：左侧选 run，右侧同步显示 mp4 + prompt + 参数 + 资源开销。
Tab 2 · **Comparison Report**：`docs/model_comparison_report.md` 全文渲染。
Tab 3 · **Prompt Registry**：`configs/prompts.json` 折叠展开查看。
Tab 4 · **Reproduce**：完整他机复现命令。
Tab 5 · **Agent Playbook**：约束 + 行动日志双嵌套 Tab。

## License / 致谢

- **本仓库**：MIT
- **Wan2.2**：见 [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)（Apache-2.0）
- **Bernini**：见 [bytedance/Bernini](https://github.com/bytedance/Bernini)（Apache-2.0）

## FAQ

**Q：他机跑 Web UI 但没有模型权重，还能用吗？**
A：能。Web UI 只依赖 `configs/prompts.json` + `docs/*.md` 就能展示报告和参数注册中心。`outputs/` 若为空，"Video Gallery" tab 会显示 ❌，其他 tab 完全可用。

**Q：Wan2.2 报 `AssertionError FLASH_ATTN_2_AVAILABLE`？**
A：Wan2.2 的 `wan/modules/model.py:145` 直接 `flash_attention()`，没有 SDPA fallback，flash_attn 是必须的。装预编译 wheel，别 `pip install` 触发源码编译。

**Q：venv `pip install` 卡 30 分钟？**
A：99% 是 venv 建在了 ceph/NFS 上。挪到本地盘。见 [`docs/agent_actions_log.md` §阶段 1](docs/agent_actions_log.md)。

**Q：ModelScope 下的权重能用吗？**
A：能，但 **一定要跑 SHA256 校验**——`.msc` 里没有内容 hash，见过下载出全 NULL 字节文件的坏 case。
