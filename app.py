#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wan2.2-TI2V-5B vs Bernini-R-1.3B-Diffusers Review Portal
--------------------------------------------------------
Gradio web app to browse the comparison report, watch generated videos,
inspect the prompt/run registry, and copy reproduction commands.

Port: 17024
Deps: gradio, Pillow  (see requirements-app.txt)

Run:
    /root/app-venv/bin/python app.py            # 本机
    python app.py                                # 其他机器（先 pip install -r requirements-app.txt）
"""

import json
import os
import socket
import sys
from pathlib import Path

import gradio as gr

# ── 常量 ─────────────────────────────────────────────────────
PORT = 17024
ROOT = Path(__file__).resolve().parent
PROMPTS_PATH = ROOT / "configs" / "prompts.json"
REPORT_PATH = ROOT / "docs" / "model_comparison_report.md"
CONSTRAINTS_PATH = ROOT / "docs" / "agent_constraints.md"
ACTIONS_PATH = ROOT / "docs" / "agent_actions_log.md"

# ── 数据加载 ─────────────────────────────────────────────────
def load_json(path):
    if not path.exists():
        return {"runs": [], "note": f"missing: {path}"}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path, fallback="_(file not present)_"):
    if not path.exists():
        return fallback
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_choices():
    """Return dropdown items: (display_label, run_id)."""
    data = load_json(PROMPTS_PATH)
    out = []
    for r in data.get("runs", []):
        title = r.get("video_title") or r.get("id")
        model_short = r.get("model", "?").split("-")[0]
        task = r.get("task", "?")
        label = f"[{model_short:>7s} · {task:>3s}] {title}"
        out.append((label, r["id"]))
    return out


def find_run(run_id):
    for r in load_json(PROMPTS_PATH).get("runs", []):
        if r["id"] == run_id:
            return r
    return None


def show_run(run_id):
    r = find_run(run_id)
    if r is None:
        return None, "_no such run_"
    out = r.get("output", {})
    p = r.get("params", {})
    rt = r.get("runtime", {})
    prm = r.get("prompt", {})
    video_path = out.get("path", "")
    exists = video_path and os.path.exists(video_path)

    input_media_lines = []
    if prm.get("image"):
        input_media_lines.append(f"- **Input image**: `{prm['image']}`")
    if prm.get("video"):
        input_media_lines.append(f"- **Input video**: `{prm['video']}`")
    if prm.get("audio"):
        input_media_lines.append(f"- **Input audio**: `{prm['audio']}`")
    if prm.get("images"):
        input_media_lines.append(f"- **Reference images**: {prm['images']}")

    info_md = f"""### {r.get("video_title") or r["id"]}

**Setup**
- Model: `{r.get('model')}`
- Task: `{r.get('task')}`
- Created: `{r.get('created_at', '?')}`

**Params**
- Size: `{p.get('size')}`
- Frames × fps: `{p.get('frame_num')} × {p.get('fps')}fps`
- Sampling: `{p.get('sample_steps')} steps, {p.get('sample_solver', '?')}, shift={p.get('sample_shift') or p.get('flow_shift')}, guide={p.get('guide_scale') or p.get('guidance_mode', '?')}`
- Seed: `{p.get('base_seed')}`

**Input**
{chr(10).join(input_media_lines) if input_media_lines else "- (text-only)"}

**Prompt**
> {prm.get("text", "").strip()}

**Output**
- Path: `{video_path}` {'✅' if exists else '❌ (not in this repo, see .gitignore)'}
- Size: `{out.get("size_bytes", 0):,} bytes`
- MD5: `{out.get("md5", "?")}`

**Runtime**
- GPU: `{rt.get('gpu', '?')}` (`CUDA_VISIBLE_DEVICES={rt.get('gpu_index', '?')}`)
- Wall time: **{rt.get('gen_seconds', '?')} s**
- Peak GPU memory: **{rt.get('peak_gpu_mib', '?')} MiB**
- Notes: {rt.get('notes', '—')}
"""
    return (video_path if exists else None), info_md


# ── 主页 ────────────────────────────────────────────────────
def hostname_hint():
    try:
        return socket.gethostname()
    except Exception:
        return "?"


REPRODUCE_MD = """## 复现指南（Other Cloud Servers）

### 0. 硬件

- 至少 1× NVIDIA GPU，显存 ≥ 24 GB（Wan2.2 t2v 单卡跑）
- Bernini t2v 单卡 ≥ 12 GB 显存即可
- CUDA driver ≥ 535 / CUDA 12.1+
- 磁盘 ≥ 100 GB（模型权重 ~64 GB + 环境 ~10 GB + 缓存）

### 1. 拉代码

```bash
git clone https://github.com/clt123321/T-V-model-review.git video-generation
cd video-generation
```

### 2. 拉两份权重（HF LFS）

```bash
pip install -U "huggingface_hub[cli]"
hf download Wan-AI/Wan2.2-TI2V-5B             --local-dir Wan2.2-TI2V-5B
hf download ByteDance/Bernini-R-1.3B-Diffusers --local-dir Bernini-R-1.3B-Diffusers
```

> 首次下载后建议 SHA256 校验：见 `docs/agent_actions_log.md` §阶段 2。

### 3. 拉两份官方推理仓库

```bash
git clone https://github.com/Wan-Video/Wan2.2.git    Wan2.2
git clone https://github.com/bytedance/Bernini.git   /root/Bernini
```

### 4. 建两个隔离 venv

```bash
# —— Wan2.2 ——（torch 2.4.1+cu121 + flash_attn 2.7.4）
uv venv --python 3.11 /root/wan22-venv
ln -sfn /root/wan22-venv .venv
/root/wan22-venv/bin/pip install -r requirements-wan22.txt

# —— Bernini ——（torch 2.5.1+cu124 + flash_attn 2.8.3 + VeOmni）
uv venv --python 3.11 /root/bernini-venv
ln -sfn /root/bernini-venv .venv-bernini
/root/bernini-venv/bin/pip install -r requirements-bernini.txt
/root/bernini-venv/bin/pip install --no-deps \\
    "git+https://github.com/ByteDance-Seed/VeOmni.git@v0.1.10"
```

> venv **必须**建在本地盘（如 `/root/`）；ceph/NFS 上会因 rename lock 慢 100 倍。

### 5. 启动 Web UI

```bash
uv venv --python 3.11 /root/app-venv
/root/app-venv/bin/pip install -r requirements-app.txt
/root/app-venv/bin/python app.py
# → http://<host>:17024
```

### 6. 命令行跑推理

**Wan2.2 T2V**:
```bash
CUDA_VISIBLE_DEVICES=1 TAG=mytest \\
  PROMPT="A cat walking on the moon" \\
  bash scripts/run_wan22_ti2v5b.sh
```

**Wan2.2 I2V** (图生视频):
```bash
CUDA_VISIBLE_DEVICES=1 TAG=mytest \\
  IMAGE=/abs/path/to/photo.jpg SIZE=704*1280 \\
  PROMPT="animate the person waving hello" \\
  bash scripts/run_wan22_ti2v5b.sh
```

**Bernini T2V**:
```bash
CUDA_VISIBLE_DEVICES=1 TAG=mytest \\
  PROMPT="A cat walking on the moon" \\
  bash scripts/run_bernini_r.sh
```

其他任务（i2i, v2v, r2v, ...）见 `docs/model_comparison_report.md §3 IO Spec`。

### 常见问题

- **flash_attn 装不上**：不要 `pip install flash-attn` 触发本地编译。到 GitHub releases 下预编译 wheel（选对 `torch` 版本 × `cxx11abi{TRUE|FALSE}`）。
- **权重完整性存疑**：ModelScope 曾出现"文件大小对但内容全 NULL"的 Bernini `*.index.json` / `vae/config.json`。跑一遍 SHA256 对照 HF LFS metadata。
- **ceph venv 慢**：`uv pip install torch` 卡 30 分钟？请把 venv 移到本地盘。
"""


# ── Gradio UI ────────────────────────────────────────────────
def build_ui():
    choices = get_choices()
    default_run = choices[0][1] if choices else None

    with gr.Blocks(
        title="Wan2.2 vs Bernini-R Video Review",
        theme=gr.themes.Soft(),
        css="""
        .video-card { padding: 12px; border-radius: 12px; background: rgba(0,0,0,0.02); }
        footer { visibility: hidden; }
        """,
    ) as demo:
        gr.Markdown(
            f"""# 🎬 Wan2.2-TI2V-5B vs Bernini-R-1.3B Video Review

对比 **Wan2.2-TI2V-5B**（阿里 5B 参数 DiT）与 **Bernini-R-1.3B-Diffusers**（ByteDance 1.3B 参数 DiT）
两个文生 / 图生视频模型在同硬件（本机为 `{hostname_hint()}`, 8×A800-80GB）下的推理表现。

* Repo: <https://github.com/clt123321/T-V-model-review>
* Port: `{PORT}`
"""
        )

        with gr.Tabs():
            # Tab 1: 视频画廊
            with gr.Tab("🎥 Video Gallery"):
                gr.Markdown(
                    "选择左侧一条 run，查看输出视频、prompt、参数、资源开销。"
                    "灰显项目 = 视频未包含在 git 仓库（受 .gitignore 保护）。"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        radio = gr.Radio(
                            choices=choices,
                            value=default_run,
                            label="Select a run",
                            interactive=True,
                        )
                        info_md = gr.Markdown()
                    with gr.Column(scale=1):
                        video = gr.Video(
                            label="Generated video",
                            elem_classes=["video-card"],
                            interactive=False,
                        )
                radio.change(show_run, inputs=radio, outputs=[video, info_md])

                if default_run:
                    initial_video, initial_info = show_run(default_run)
                    video.value = initial_video
                    info_md.value = initial_info

            # Tab 2: 对比报告
            with gr.Tab("📊 Comparison Report"):
                gr.Markdown(load_text(REPORT_PATH))

            # Tab 3: Prompts / Runs JSON
            with gr.Tab("📋 Prompt Registry"):
                gr.Markdown(
                    f"运行注册中心（源文件：`{PROMPTS_PATH}`）。每次推理都追加一条到 `runs` 数组。"
                )
                gr.JSON(load_json(PROMPTS_PATH), label="prompts.json", open=True)

            # Tab 4: 复现指南
            with gr.Tab("⚙️ Reproduce"):
                gr.Markdown(REPRODUCE_MD)

            # Tab 5: Agent docs
            with gr.Tab("🤖 Agent Playbook"):
                with gr.Tabs():
                    with gr.Tab("Constraints"):
                        gr.Markdown(load_text(CONSTRAINTS_PATH))
                    with gr.Tab("Actions & Findings"):
                        gr.Markdown(load_text(ACTIONS_PATH))

    return demo


if __name__ == "__main__":
    print(f"[INFO] Root: {ROOT}", flush=True)
    print(f"[INFO] Prompts: {PROMPTS_PATH} (exists={PROMPTS_PATH.exists()})", flush=True)
    print(f"[INFO] Port: {PORT}", flush=True)
    demo = build_ui()
    demo.queue(max_size=16).launch(
        server_name="0.0.0.0",
        server_port=PORT,
        show_error=True,
        share=False,
        favicon_path=None,
        allowed_paths=[str(ROOT / "outputs")],  # 允许 gr.Video 读 outputs/*.mp4
    )
