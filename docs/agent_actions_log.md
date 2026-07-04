# Agent 行动日志与发现（Actions & Findings Log）

> 时间顺序记录本项目从环境接手到跑通 Wan2.2 + Bernini 双模型的关键动作、遇到的问题、以及诊断结论。之后再排查用。

---

## 阶段 0：接手时的初始状态（2026-07-04 21:00 前后）

- 项目根：`/home/web_server/antispam/project/clt/video-generation/`
- 已有：
  - `Wan2.2/`（git clone 好的官方代码）
  - `Wan2.2-TI2V-5B/`（权重已下载，5 个大文件按 size 看似完整）
  - `Bernini-R-1.3B-Diffusers/`（text_encoder 只有 shard 1/3，transformer 完整）
  - `.venv/`（uv 管理的 py3.11 venv，但 python 二进制指向 `/root/.local/share/uv/...` 已被卸载）
  - `docs/wan22_ti2v5b_runbook.md` 和 `wan22_ti2v5b_code_walkthrough.md`（老 doc，路径引用 `/clt/wan2.2/`）
  - `scripts/run_wan22_ti2v5b_4090.sh`（针对 4090 24GB 写的）
- 已知硬件差异：老 doc 说 4090 24GB，实际是 **8×A800-80GB**（`nvidia-smi` 确认）。

---

## 阶段 1：环境重建（21:00 - 22:00）

**动作 1.1**：`ls .venv/bin/python` → 悬挂软链，`/root/.local/share/uv/` 目录不存在。
**发现 1.1**：uv 曾装过、但被卸载。原 venv 无法使用。

**动作 1.2**：`python3 --version` → 3.12.3；测试 `import torch` → torch 2.8.0a0+nv 可用，8 卡 CUDA。
**发现 1.2**：系统 python 3.12 是可用的（NVIDIA 容器构建的 torch）。但 Wan2.2 官方要求 3.11。

**动作 1.3**（按用户要求）：重装 uv，创建 py3.11 venv。
```
pip install -i https://mirrors.aliyun.com/pypi/simple/ uv
uv venv --python 3.11 .venv
```
**发现 1.3**：清华 pypi 源在本机返回 403，aliyun 源正常。uv 会自动下 cpython-3.11.15 到 `/root/.local/share/uv/python/`。

**动作 1.4**：`uv pip install torch==2.4.1 torchvision==0.19.1`（走 pytorch cu121 index）。
**发现 1.4-A**：venv 位于 ceph（`10.80.193.230:.../project/`）时，wheel 解压极慢——单个 torch wheel 复制解压 34+ 分钟未完成。ceph rename 有 lock 竞争（4 个线程停在 `lock_rename`）。
**动作 1.4-B**：kill 掉，把 venv 移到本地 overlay `/root/wan22-venv/`，软链到项目 `.venv`。同样命令在本地跑 174 ms 完成。
**结论 1.4**：**venv 必须建在本地磁盘**，走软链暴露到项目路径。

**动作 1.5**：装剩下的 Wan2.2 依赖（transformers ≤4.51.3、diffusers、accelerate、opencv、imageio、decord、easydict、ftfy、dashscope、safetensors 等）。全部走 aliyun mirror，秒级完成。

---

## 阶段 2：权重校验（22:00 - 22:15）

**动作 2.1**：直接看 Wan2.2-TI2V-5B 文件大小 vs runbook 期望值，逐个匹配 → **5/5 sizes 都对**。

**动作 2.2**：ModelScope 的 `.msc` 里只有 git revision，没有 sha256。改从 HF API 拉参考值：
```
curl https://huggingface.co/api/models/Wan-AI/Wan2.2-TI2V-5B/tree/main | jq
```
拿到 5 个 LFS 文件的 `lfs.oid`（就是 sha256）。

**动作 2.3**：本地 5 个文件并行 `sha256sum`，34 GB **10.6 秒**跑完（ceph 读速 ~1.5 GB/s 缓存命中，实际并行读也很快）。**5/5 全部对上 HF 值**。

**动作 2.4**：额外用 `safetensors.safe_open` 检验 headers、`torch.load(weights_only=True)` 检验 `.pth` 反序列化——全部通过。

**结论 2**：Wan2.2-TI2V-5B 权重 100% 完整可用。

---

## 阶段 3：Wan2.2 首次推理（22:45 - 23:03）

**动作 3.1**：写 `scripts/run_wan22_ti2v5b.sh`（新脚本，替代老的 4090 脚本）。参数化 SIZE / FRAME_NUM / SAMPLE_STEPS / GUIDE_SCALE / SAMPLE_SHIFT / BASE_SEED / TAG / PROMPT，硬件档配到 A800（去掉 `--offload_model`/`--t5_cpu`，保留 `--convert_model_dtype`）。

**动作 3.2**：Smoke test（15 步 × 41 帧）。
**失败 3.2**：报错 `AssertionError` in `flash_attention()` at `attention.py:112`，说 `FLASH_ATTN_2_AVAILABLE=False`。

**发现 3.2**：读 `wan/modules/model.py` 行 145 发现 `WanAttentionBlock` 直接调 `flash_attention()`（要求 flash_attn），而不是走 `attention()` 那个带 SDPA fallback 的封装。**Wan2.2-TI2V-5B 实际上强依赖 flash_attn**。（Runbook 里"有 SDPA fallback" 的描述是错的。）

**动作 3.3**：装 flash_attn 2.7.4.post1 预编译 wheel（GitHub releases）。ABI 决策：`torch._C._GLIBCXX_USE_CXX11_ABI == False` → 选 `cxx11abiFALSE` 变体。3 秒完成安装。

**动作 3.4**：重跑 smoke test → **成功**。15 步 × 41 帧 × 1280×704 用时 25.5s 采样 + 加载/decode 共 118s。Peak 28.7 GB。

**动作 3.5**：全量 run（TAG=first，121 帧 × 50 步）→ **7 分 7 秒**，peak 29.8 GB。产出 `outputs/first_wan22_ti2v5b.mp4`（8.05 MB，md5 05c2e854...）。

**发现 3.5**：单 A800 跑 5B 模型完全从容。`--offload_model` 是脚本代码里默认开的（world_size==1 时自动），但显存本身也够。

---

## 阶段 4：SFTP 权限排查（23:15 - 23:30）

**问题**：用户用 chenglitao 的 SFTP 上传 mp4 到跳板机，`put` 到 `/` 时先写 32KB 然后 permission denied；`put` 到 `/home/chenglitao/` 直接 "No such file or directory"。

**诊断动作**：
- `getent passwd chenglitao` 本机查不到。
- `/share/` 列表下 chen* 家目录有 chengang06/chenjunwen/chenlu09/chenpengwei，**独缺 chenglitao**。
- 挂载表：本机挂了 `10.80.193.230:/rc-kml-ssd/pub → /share`（团队共享）和 `/rc-kml-ssd/ouyangshizhuang → /home/ouyangshizhuang`（per-user 挂载）。

**结论 4**：chenglitao 在 kml 账号系统里就没有 home 目录。32KB 是 SFTP client buffer，实际 flush 到 inode 时 ceph ACL 拒。

**方案**：把 mp4 cp 到 `/share/tmp_video_dropoff/`——如果跳板机也挂了 `/share`，直接就能看到；否则用 `/tmp/` 或让 admin 补 home。

---

## 阶段 5：Bernini 完整性 + 环境搭建（23:35 - 23:57）

**动作 5.1**：Bernini text_encoder 后来补齐 5/5 shards。跑完整性 check：
- text_encoder：5/5 shards OK，sum 22,723,671,744 ≈ index total 22,723,641,344（差 30,400 = safetensors header overhead）。
- transformer：2/2 shards OK。
- vae：1 safetensors OK。
- **但**：`transformer/*.index.json` 76,607 字节全 NULL；`vae/config.json` 724 字节全 NULL。

**发现 5.1**：ModelScope/HF 下载偶发"预分配文件但没写入内容"的坏 case。**光看 size 完全看不出来**。

**动作 5.2**：从 HF 直接 curl 两个坏 JSON：
```
curl -fsSL -o transformer/*.index.json https://huggingface.co/ByteDance/Bernini-R-1.3B-Diffusers/resolve/main/transformer/diffusion_pytorch_model.safetensors.index.json
curl -fsSL -o vae/config.json         https://huggingface.co/ByteDance/Bernini-R-1.3B-Diffusers/resolve/main/vae/config.json
```
**结果**：76,607 + 724 = 77,331 字节，parse OK，`weight_map` 覆盖两个 shards。

**动作 5.3**：读 Bernini README（架构、requirements）。发现：
- 强绑定 torch==2.5.1+cu124、diffusers==0.35.2、transformers==4.57.3、flash_attn==2.8.3
- 额外要 **VeOmni**（`git+https://github.com/ByteDance-Seed/VeOmni.git@v0.1.10` --no-deps）
- 与 wan22-venv 版本互不兼容 → **必须独立 venv**。

**动作 5.4**：建 `/root/bernini-venv`。torch 2.5.1 装完后 `torch.__version__` 显示 `2.5.1+cu124`（尽管 URL 指定 cu121，PyTorch 官方在 2.5.x 的 cu121 索引对某些 wheel 名做了 alias）。**cu124 wheel 在 A800 + driver 535.129.03 上依然工作**——CUDA 前向兼容 + Ampere SM 80 是 cu12.x 全线兼容的。

**动作 5.5**：装 flash_attn 2.8.3（cu12torch2.5 abiFALSE cp311 变体），装 VeOmni（git build ~5 分钟），装 setuptools（uv 默认不带 → 首次 import triton 挂）。

**动作 5.6**：`infer_single_gpu.py --help` 正常输出。Bernini 单卡 T2V 跑通：40 步 × 3.78 s = 2m32s 采样，`bernini_r_t2v_first.mp4` 6.45 MB 落盘。**峰值 GPU 仅 8087 MiB**。

---

## 阶段 6：项目工程化 + 图生视频（写这份日志的时刻）

- 清理：`/root/.cache/uv`（6 GB）删掉；smoke test mp4 + 老的 4090 脚本删掉；`/share/tmp_video_dropoff/` 保留一份 dropoff mp4 备份。
- 新增：`configs/prompts.json`（运行注册中心）、`configs/bernini_cases/`（Bernini 案例）、`docs/model_comparison_report.md`、`docs/agent_constraints.md`、`docs/agent_actions_log.md`。
- 更新：`scripts/run_wan22_ti2v5b.sh` 支持 `IMAGE` 环境变量。
- 正在跑：`guoge_certified_wan22_ti2v5b.mp4` 图生视频（guoge.jpg + 动画 prompt，i2v 分支）。

---

## 已发现的坑清单（快速索引）

| # | 症状 | 根因 | 处理 |
|---|---|---|---|
| 1 | `.venv/bin/python: No such file or directory` | uv 被卸载，符号链接指向 `/root/.local/share/uv/` 空目录 | 重装 uv 或改用系统 python |
| 2 | `uv pip install torch` 卡 30+ 分钟 | venv 建在 ceph，rename lock 竞争 | venv 移到本地 `/root/`，软链暴露 |
| 3 | 清华 pypi 403 | 出海代理策略 | 用 aliyun mirror |
| 4 | flash_attn 装 pip 触发本地编译 30+ 分钟 | 没找到匹配 wheel | 手动 curl GitHub releases 的预编译 wheel |
| 5 | `assert FLASH_ATTN_2_AVAILABLE` | Wan2.2 `model.py:145` 直接调 `flash_attention()`，无 SDPA fallback | 必须装 flash_attn |
| 6 | 权重 size 对但内容全 NULL | 下载器预分配文件但未写入 | 逐文件跑 NULL byte 检查，再从 HF 补 |
| 7 | SFTP `put /` permission denied | 目标用户 `chenglitao` 无 kml home 目录 | 走 `/share/` 团队共享 dropoff |
| 8 | GPU 0 长期占 74 GB | 外部进程 PID 1174614 | 脚本默认 `CUDA_VISIBLE_DEVICES=1` |

---

## 后续可能的动作（TODO 参考）

- [ ] 补 Bernini 的 i2v/r2v 单图测试案例（本次只跑了 t2v）。
- [ ] 尝试 Wan2.2 的 8 卡 Ulysses (`--ulysses_size 8`) 看提速比。
- [ ] 尝试 Bernini 的 `--use_pe` prompt enhancer（需要 OpenAI-compatible endpoint 环境变量）。
- [ ] 加一个 `scripts/run_bernini_r.sh` 通用封装（现在还是手写 python 命令）。
- [ ] 把 Wan2.2 用的 UMT5-XXL text encoder 权重和 Bernini 的对比一下——如果 tensor 完全一样，可以硬链接省 22 GB 磁盘。
