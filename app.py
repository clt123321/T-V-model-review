#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wan2.2-TI2V-5B vs Bernini-R-1.3B-Diffusers Review Portal
--------------------------------------------------------
Gradio web app to browse the comparison report, watch generated videos,
inspect the prompt/run registry, launch Wan2.2 t2v/i2v runs, and download
the resulting mp4s.

Port: 15856
Deps: gradio, Pillow  (see requirements-app.txt)

Run:
    /root/app-venv/bin/python app.py            # 本机
    python app.py                                # 其他机器（先 pip install -r requirements-app.txt）
"""

import datetime
import hashlib
import json
import os
import queue
import shlex
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import gradio as gr

PORT = 15856
ROOT = Path(__file__).resolve().parent
PROMPTS_PATH = ROOT / "configs" / "prompts.json"
REPORT_PATH = ROOT / "docs" / "model_comparison_report.md"
CONSTRAINTS_PATH = ROOT / "docs" / "agent_constraints.md"
ACTIONS_PATH = ROOT / "docs" / "agent_actions_log.md"

WAN22_VENV = Path("/root/wan22-venv")
WAN22_SCRIPT = ROOT / "scripts" / "run_wan22_ti2v5b.sh"
WAN22_WEIGHTS = ROOT / "Wan2.2-TI2V-5B"


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
        return None, "_no such run_", gr.DownloadButton(visible=False)
    out = r.get("output", {})
    p = r.get("params", {})
    rt = r.get("runtime", {})
    prm = r.get("prompt", {})
    video_path = out.get("path", "")
    exists = bool(video_path and os.path.exists(video_path))

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
    dl = (
        gr.DownloadButton(value=video_path, visible=True, label=f"⬇ Download {Path(video_path).name}")
        if exists
        else gr.DownloadButton(visible=False)
    )
    return (video_path if exists else None), info_md, dl


def hostname_hint():
    try:
        return socket.gethostname()
    except Exception:
        return "?"


def env_ready(deep: bool = False):
    py = WAN22_VENV / "bin" / "python"
    if not py.exists():
        return False, f"❌ Wan2.2 venv missing: `{WAN22_VENV}` — 请等待重建完成"
    if not WAN22_WEIGHTS.exists():
        return False, f"❌ Wan2.2 weights missing: `{WAN22_WEIGHTS}`"
    if not WAN22_SCRIPT.exists():
        return False, f"❌ launcher missing: `{WAN22_SCRIPT}`"
    if not deep:
        return True, "✅ Wan2.2 venv/weights/launcher all present (deep probe skipped)"
    try:
        r = subprocess.run(
            [str(py), "-c", "import torch, transformers, diffusers, flash_attn; print('ok')"],
            capture_output=True, timeout=15, text=True,
        )
        if r.returncode != 0:
            missing = (r.stderr or r.stdout).strip().splitlines()[-1][:220]
            return False, f"❌ Wan2.2 venv incomplete (pip 可能还在跑): `{missing}`"
    except subprocess.TimeoutExpired:
        return False, "❌ deep probe timed out (>15s) — venv 状态异常"
    except Exception as e:
        return False, f"❌ probe error: {e}"
    return True, "✅ Wan2.2 env ready (torch/transformers/diffusers/flash_attn all importable)"


def env_status_md():
    ok, msg = env_ready(deep=True)
    hint = "\n\n(watchdog: gradio 会在启动后 1 小时被自动 kill,防止占用别人机器)"
    return msg + hint


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def append_run_to_prompts(entry: dict):
    data = load_json(PROMPTS_PATH)
    if "runs" not in data or not isinstance(data["runs"], list):
        data["runs"] = []
    data["runs"].append(entry)
    tmp = PROMPTS_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(PROMPTS_PATH)


def peak_gpu_from_log(tag: str):
    csv_path = ROOT / "logs" / f"{tag}_gpu_mem.csv"
    if not csv_path.exists():
        return None
    try:
        peak = 0
        with open(csv_path) as f:
            next(f, None)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                mib = parts[1].strip().split()[0]
                if mib.isdigit():
                    peak = max(peak, int(mib))
        return peak or None
    except Exception:
        return None


def apply_preset(preset_label):
    """Preset radio -> steps, frame_num."""
    if preset_label.startswith("fast"):
        return gr.update(value=10), gr.update(value=41)
    return gr.update(value=50), gr.update(value=121)


def _stream_generate(task_type: str, prompt: str, size: str, gpu_idx: str,
                     steps: int, guide: float, shift: float, seed: int,
                     frame_num: int, tag: str, image_path: str | None):
    """Common streaming subprocess wrapper. Yields (status_md, video_or_None, dl_update, radio_update)."""
    ok, msg = env_ready(deep=True)
    hidden_dl = gr.DownloadButton(visible=False)
    if not ok:
        yield msg, None, hidden_dl
        return
    if not prompt or not prompt.strip():
        yield "❌ prompt is empty", None, hidden_dl
        return
    if task_type == "i2v" and (not image_path or not os.path.exists(image_path)):
        yield "❌ image is required for i2v", None, hidden_dl
        return
    if int(frame_num) % 4 != 1:
        yield f"❌ frame_num must be 4n+1 (got {frame_num})", None, hidden_dl
        return
    if size not in ("1280*704", "704*1280"):
        yield f"❌ size must be 1280*704 or 704*1280 (got {size})", None, hidden_dl
        return

    safe_tag = "".join(c for c in (tag or "webui") if c.isalnum() or c in "-_")[:24] or "webui"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_tag = f"{safe_tag}_{ts}"
    save_file = ROOT / "outputs" / f"{full_tag}_wan22_ti2v5b.mp4"

    env = os.environ.copy()
    env.update({
        "TAG": full_tag,
        "PROMPT": prompt.strip(),
        "SIZE": size,
        "SAMPLE_STEPS": str(int(steps)),
        "GUIDE_SCALE": str(float(guide)),
        "SAMPLE_SHIFT": str(float(shift)),
        "BASE_SEED": str(int(seed)),
        "FRAME_NUM": str(int(frame_num)),
        "CUDA_VISIBLE_DEVICES": str(gpu_idx),
    })
    if task_type == "i2v" and image_path:
        env["IMAGE"] = image_path
    else:
        env.pop("IMAGE", None)

    eta = "≈1 min" if int(steps) <= 12 else ("≈4 min" if int(steps) <= 25 else "≈7 min")
    header = (f"🚀 launching **Wan2.2 {task_type.upper()}** on GPU {gpu_idx} — expected {eta}\n\n"
              f"- tag: `{full_tag}`\n- size: `{size}`\n- steps/guide/shift/frames/seed: "
              f"`{steps}/{guide}/{shift}/{frame_num}/{seed}`\n")
    yield header + "\n_starting subprocess..._", None, hidden_dl

    start = time.time()
    proc = subprocess.Popen(
        ["bash", str(WAN22_SCRIPT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,  # unbuffered bytes; we manually split on \r/\n so tqdm progress is visible
    )

    # Threaded reader: tqdm uses `\r` to overwrite the same line without ever
    # emitting `\n` for minutes — plain `for line in proc.stdout` blocks that
    # long, so gradio SSE times out and the generator gets GeneratorExit'd
    # before we can append_run_to_prompts. Instead: reader thread pushes tokens
    # (split on \r OR \n) to a queue; the main coroutine drains + heartbeats
    # every ~1.5s so gradio's SSE stays alive.
    q: "queue.Queue[str | None]" = queue.Queue()

    def _reader():
        try:
            buf = b""
            fd = proc.stdout.fileno()
            while True:
                try:
                    chunk = os.read(fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while True:
                    n = buf.find(b"\n")
                    r = buf.find(b"\r")
                    idx = min([i for i in (n, r) if i >= 0], default=-1)
                    if idx < 0:
                        break
                    line = buf[:idx].decode("utf-8", errors="replace").rstrip()
                    buf = buf[idx + 1 :]
                    if line:
                        q.put(line)
            if buf:
                rem = buf.decode("utf-8", errors="replace").rstrip()
                if rem:
                    q.put(rem)
        finally:
            q.put(None)  # EOF sentinel

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    tail: list[str] = []
    last_yield = 0.0
    eof = False
    while not eof:
        # non-blocking drain
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                eof = True
                break
            tail.append(item)
            tail = tail[-10:]

        now = time.time()
        if now - last_yield >= 1.5:  # heartbeat keeps SSE alive
            elapsed = int(now - start)
            body = "\n".join(tail) if tail else "(waiting for output...)"
            yield f"{header}\n⏳ running ({elapsed}s)\n```\n{body}\n```", None, hidden_dl
            last_yield = now

        if not eof:
            time.sleep(0.3)

    proc.wait()
    reader_thread.join(timeout=2)
    elapsed = int(time.time() - start)

    if proc.returncode != 0 or not save_file.exists():
        body = "\n".join(tail[-15:])
        yield (f"{header}\n❌ **FAILED** (rc={proc.returncode}, {elapsed}s)\n\n"
               f"Last output:\n```\n{body}\n```"), None, hidden_dl
        return

    size_bytes = save_file.stat().st_size
    md5_hex = md5_of(save_file)
    peak = peak_gpu_from_log(full_tag)

    entry = {
        "id": full_tag,
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "Wan2.2-TI2V-5B",
        "task": task_type,
        "video_title": f"Web UI {task_type} · {ts}",
        "prompt": {
            "text": prompt.strip(),
            "negative_text": None,
            "system_prompt": None,
            "image": image_path if task_type == "i2v" else None,
            "images": [],
            "video": None,
            "audio": None,
            "reference_images": [],
        },
        "params": {
            "size": size,
            "frame_num": int(frame_num),
            "fps": 24,
            "sample_steps": int(steps),
            "sample_solver": "unipc",
            "guide_scale": float(guide),
            "sample_shift": float(shift),
            "base_seed": int(seed),
        },
        "output": {
            "path": str(save_file),
            "size_bytes": size_bytes,
            "md5": md5_hex,
        },
        "runtime": {
            "gpu": "NVIDIA A800-SXM4-80GB",
            "gpu_index": int(gpu_idx),
            "gen_seconds": elapsed,
            "peak_gpu_mib": peak,
            "notes": f"Web UI generated: task={task_type}, steps={steps}, frames={frame_num}",
        },
    }
    try:
        append_run_to_prompts(entry)
    except Exception as e:
        yield (f"{header}\n⚠ generated but failed to write prompts.json: {e}\n"
               f"Video is at `{save_file}`"), str(save_file), gr.DownloadButton(
                   value=str(save_file), visible=True, label=f"⬇ Download {save_file.name}")
        return

    ok_msg = (f"{header}\n✅ **DONE** in {elapsed}s\n\n"
              f"- file: `{save_file.name}`\n- size: `{size_bytes:,}` bytes\n"
              f"- md5: `{md5_hex[:12]}…`\n- peak GPU mem: `{peak} MiB`\n\n"
              f"Registered as run `{full_tag}`. **Switch to Video Gallery tab** to browse (it auto-refreshes on tab select).")
    yield (ok_msg,
           str(save_file),
           gr.DownloadButton(value=str(save_file), visible=True, label=f"⬇ Download {save_file.name}"))


def generate_t2v_stream(prompt, size_label, gpu_idx, steps, guide, shift, seed, frame_num, tag):
    size = size_label.split(" ")[0] if size_label else "1280*704"
    yield from _stream_generate("t2v", prompt, size, gpu_idx,
                                steps, guide, shift, seed, frame_num, tag, None)


def generate_i2v_stream(image_path, prompt, size_label, gpu_idx, steps, guide, shift, seed, frame_num, tag):
    size = size_label.split(" ")[0] if size_label else "704*1280"
    yield from _stream_generate("i2v", prompt, size, gpu_idx,
                                steps, guide, shift, seed, frame_num, tag, image_path)


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
# → http://<host>:15856
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
            with gr.Tab("🎥 Video Gallery") as gallery_tab:
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
                        gallery_dl = gr.DownloadButton(
                            label="⬇ Download mp4",
                            visible=False,
                        )
                radio.change(show_run, inputs=radio, outputs=[video, info_md, gallery_dl])

                # Auto-refresh gallery radio choices whenever user clicks into this tab.
                # We do NOT auto-select the newest run (that would auto-switch tabs and be jarring).
                def _refresh_gallery_choices():
                    return gr.update(choices=get_choices())
                gallery_tab.select(_refresh_gallery_choices, outputs=radio)

                if default_run:
                    v0, info0, dl0 = show_run(default_run)
                    video.value = v0
                    info_md.value = info0

            # Tab 2: Generate
            with gr.Tab("🎨 Generate (Wan2.2)"):
                gr.Markdown(
                    f"直接输入 prompt / 上传图片，后端会调用 **Wan2.2-TI2V-5B** 出片。\n\n"
                    f"{env_status_md()}"
                )
                with gr.Tabs():
                    # ---- T2V ----
                    with gr.Tab("T2V (文生视频)"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                t2v_prompt = gr.Textbox(
                                    lines=4,
                                    label="Prompt",
                                    placeholder="A cinematic shot of a small silver robot walking through a rainy neon alley at night...",
                                )
                                with gr.Row():
                                    t2v_size = gr.Radio(
                                        choices=["1280*704 (landscape)", "704*1280 (portrait)"],
                                        value="1280*704 (landscape)",
                                        label="Size",
                                    )
                                    t2v_gpu = gr.Dropdown(
                                        choices=[str(i) for i in range(8)],
                                        value="1",
                                        label="GPU (CUDA_VISIBLE_DEVICES)",
                                    )
                                t2v_preset = gr.Radio(
                                    choices=["fast (10 steps · 41 frames · ~1min)",
                                             "quality (50 steps · 121 frames · ~7min)"],
                                    value="fast (10 steps · 41 frames · ~1min)",
                                    label="Preset",
                                )
                                with gr.Accordion("Advanced params", open=False):
                                    t2v_steps = gr.Slider(4, 60, value=10, step=1, label="sample_steps")
                                    t2v_guide = gr.Slider(1, 15, value=5.0, step=0.5, label="sample_guide_scale")
                                    t2v_shift = gr.Slider(1, 12, value=5.0, step=0.5, label="sample_shift")
                                    t2v_seed = gr.Number(value=42, precision=0, label="base_seed")
                                    t2v_frames = gr.Number(value=41, precision=0, label="frame_num (must be 4n+1)")
                                t2v_tag = gr.Textbox(value="webui", label="Run tag (output filename prefix)")
                                t2v_btn = gr.Button("🚀 Generate T2V", variant="primary")
                            with gr.Column(scale=1):
                                t2v_status = gr.Markdown("_Ready. Click generate to start._")
                                t2v_video = gr.Video(label="Result", interactive=False, elem_classes=["video-card"])
                                t2v_dl = gr.DownloadButton(label="⬇ Download mp4", visible=False)

                        t2v_preset.change(apply_preset, inputs=t2v_preset, outputs=[t2v_steps, t2v_frames])
                        t2v_btn.click(
                            generate_t2v_stream,
                            inputs=[t2v_prompt, t2v_size, t2v_gpu, t2v_steps, t2v_guide, t2v_shift, t2v_seed, t2v_frames, t2v_tag],
                            outputs=[t2v_status, t2v_video, t2v_dl],
                        )

                    # ---- I2V ----
                    with gr.Tab("I2V (图生视频)"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                i2v_image = gr.Image(
                                    type="filepath",
                                    label="Input image (used as first frame)",
                                    height=320,
                                )
                                i2v_prompt = gr.Textbox(
                                    lines=4,
                                    label="Prompt",
                                    placeholder="The person in the image waves and smiles at the camera...",
                                )
                                with gr.Row():
                                    i2v_size = gr.Radio(
                                        choices=["1280*704 (landscape)", "704*1280 (portrait)"],
                                        value="704*1280 (portrait)",
                                        label="Size (should match image aspect)",
                                    )
                                    i2v_gpu = gr.Dropdown(
                                        choices=[str(i) for i in range(8)],
                                        value="1",
                                        label="GPU",
                                    )
                                i2v_preset = gr.Radio(
                                    choices=["fast (10 steps · 41 frames · ~1min)",
                                             "quality (50 steps · 121 frames · ~7min)"],
                                    value="fast (10 steps · 41 frames · ~1min)",
                                    label="Preset",
                                )
                                with gr.Accordion("Advanced params", open=False):
                                    i2v_steps = gr.Slider(4, 60, value=10, step=1, label="sample_steps")
                                    i2v_guide = gr.Slider(1, 15, value=5.0, step=0.5, label="sample_guide_scale")
                                    i2v_shift = gr.Slider(1, 12, value=5.0, step=0.5, label="sample_shift")
                                    i2v_seed = gr.Number(value=42, precision=0, label="base_seed")
                                    i2v_frames = gr.Number(value=41, precision=0, label="frame_num (must be 4n+1)")
                                i2v_tag = gr.Textbox(value="webui", label="Run tag")
                                i2v_btn = gr.Button("🚀 Generate I2V", variant="primary")
                            with gr.Column(scale=1):
                                i2v_status = gr.Markdown("_Ready. Upload an image, enter a prompt, click generate._")
                                i2v_video = gr.Video(label="Result", interactive=False, elem_classes=["video-card"])
                                i2v_dl = gr.DownloadButton(label="⬇ Download mp4", visible=False)

                        i2v_preset.change(apply_preset, inputs=i2v_preset, outputs=[i2v_steps, i2v_frames])
                        i2v_btn.click(
                            generate_i2v_stream,
                            inputs=[i2v_image, i2v_prompt, i2v_size, i2v_gpu, i2v_steps, i2v_guide, i2v_shift, i2v_seed, i2v_frames, i2v_tag],
                            outputs=[i2v_status, i2v_video, i2v_dl],
                        )

            # Tab 3: 对比报告
            with gr.Tab("📊 Comparison Report"):
                gr.Markdown(load_text(REPORT_PATH))

            # Tab 4: Prompts / Runs JSON
            with gr.Tab("📋 Prompt Registry"):
                gr.Markdown(
                    f"运行注册中心（源文件：`{PROMPTS_PATH}`）。每次推理都追加一条到 `runs` 数组。"
                )
                gr.JSON(load_json(PROMPTS_PATH), label="prompts.json", open=True)

            # Tab 5: 复现指南
            with gr.Tab("⚙️ Reproduce"):
                gr.Markdown(REPRODUCE_MD)

            # Tab 6: Agent docs
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
    print(f"[INFO] Wan2.2 env ready (shallow): {env_ready(deep=False)}", flush=True)
    demo = build_ui()
    demo.queue(max_size=16).launch(
        server_name="0.0.0.0",
        server_port=PORT,
        show_error=True,
        share=False,
        favicon_path=None,
        allowed_paths=[str(ROOT / "outputs"), "/tmp"],
    )
