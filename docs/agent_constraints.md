# Agent 操作约束（Constraints）

> 用途：给之后接手的 agent（人或 LLM）一份可执行的"红线清单"，避免踩已经踩过的坑。

---

## 硬约束（Do NOT）

1. **不要跨账户/跨用户改文件权限**
   项目根 `/home/web_server/antispam/project/clt/video-generation/` 是 ceph 挂载点，属于 web_server 用户组共享空间。不要 `chown` / `chmod` 全目录、不要动 `.gitignore` 之外别人的文件。

2. **不要把 venv 建在 ceph 上**
   Ceph 对大量小文件（wheel 解压出的成千上万个 `.py`）rename/link 极慢，同样的 `torch` 装载在 ceph 上要 30+ 分钟，在本地 `/root/` 上只要 2 秒。**所有 venv 都必须放到 `/root/*-venv/`，再通过软链接暴露到项目目录。**

3. **不要动 GPU 0**
   本机 GPU 0 长期被外部进程（PID 1174614）占用 ~74GB。不要抢占，脚本默认 `CUDA_VISIBLE_DEVICES=1`。如需并发，选 GPU 1-7。

4. **不要绕过 SHA256 校验直接跑权重**
   ModelScope 下载偶尔会出现"文件大小对但内容全 0"的情况（Bernini 就中过枪：`transformer/*.index.json` 和 `vae/config.json` 都是 76607 字节和 724 字节全 NULL）。**每份新到的权重都要 SHA256 对照 HuggingFace LFS 元数据校验**。

5. **不要合并 `wan22-venv` 和 `bernini-venv`**
   两者 torch 版本不同（2.4.1+cu121 vs 2.5.1+cu124），flash_attn 也是不同 ABI 编译的（2.7.4 vs 2.8.3），合并环境必然破坏其中一个。**保持两个 venv 独立**。

6. **不要在没有明确用户授权时 `git commit / git push`**
   本项目不在 git 仓库里（`Is directory a git repo: NO`），也不要主动 `git init`。所有版本管理需求都要问用户。

7. **不要主动创建 README/主索引文档**
   仅在用户明确要求时创建 `.md` 文件。目前所有 doc 都在 `docs/`，创建新 doc 前先看是否可以扩展现有文件。

8. **不要在 sftp/scp 前假设远端路径存在**
   本机的 `/home/chenglitao` 在别人机器上未必是家目录（chenglitao 在 kml 账号系统里就没配家目录）。远端 `put` 要先 `pwd` 或 `cd ~`。

---

## 软约束（Prefer / Verify）

1. **优先用 aliyun mirror 装 pip 包**
   `--index-url https://mirrors.aliyun.com/pypi/simple/`。清华源 (`pypi.tuna.tsinghua.edu.cn`) 在本机返回 403。

2. **PyTorch wheel 走官方源**
   `--index-url https://download.pytorch.org/whl/cu121`（或 cu124）。aliyun 有 `/pytorch-wheels/` 但完整性欠佳。

3. **flash_attn 直接下预编译 wheel，不要 pip install 触发本地编译**
   本地编译 flash_attn 需 30+ 分钟。GitHub releases 有 `+cu12torch2.<X>cxx11abiFALSE-cp311` 全组合的 whl。**装前先查 `torch._C._GLIBCXX_USE_CXX11_ABI`** 决定 abiTRUE 还是 abiFALSE。

4. **HuggingFace 直连可用，但大文件用 ModelScope**
   本机能直接 curl `huggingface.co`（返回 200），API 也通。但大 LFS 文件（>1GB）走 HF CDN 会被 squid 限流，改用 ModelScope 更稳。**小 JSON/config 直接 curl HF 即可。**

5. **推理前跑 smoke test**
   全量参数（50 steps × 121 frames）跑一次要 7 分钟，出错就要重头。先用 `SAMPLE_STEPS=15 FRAME_NUM=41` 跑 ~2 分钟的 smoke test，验通了再跑正式。

6. **记录到 `configs/prompts.json`**
   每次成功推理后，追加一条 entry 到 `runs` 数组（schema 见文件顶部）。**md5 / size / peak_gpu_mib 必须实测填入**，别猜。

7. **观察 GPU 用量走势**
   `nvidia-smi --query-gpu=timestamp,memory.used,... --format=csv -lms 2000 > gpu_mem.csv &` 后台采样，跑完 `max()` 得到峰值。已写进 `scripts/run_wan22_ti2v5b.sh`。

---

## 目录约束

| 路径 | 允许写入 | 用途 |
|---|---|---|
| `/root/*-venv/` | 是 | venv 实体（本地 SSD） |
| `/root/Bernini/` | 是 | bytedance/Bernini github clone |
| `/root/.cache/uv/` | 是 | uv 缓存（可清） |
| `/root/.cache/huggingface/` | **否** | 其他用户/服务的 HF 缓存 |
| `/root/.cache/vllm/` | **否** | 其他用户/服务的 vllm 缓存 |
| `<project>/Wan2.2-TI2V-5B/` | **只读** | 模型权重，动就得重下 |
| `<project>/Bernini-R-1.3B-Diffusers/` | **只读** | 同上 |
| `<project>/Wan2.2/` | 可 patch | 官方仓库 clone，改动要记入 actions log |
| `<project>/scripts/` `configs/` `docs/` `outputs/` `logs/` | 是 | 项目本地产物 |
| `/share/` | 是 | 团队共享 ceph，可作为跨机 dropoff |
| 别人的 `/share/<user>/` | **否** | 尊重同事目录 |

---

## 命令模板

- 激活 wan22 venv：
  ```
  export PATH=/root/wan22-venv/bin:$PATH
  ```
- 激活 bernini venv：
  ```
  export PATH=/root/bernini-venv/bin:$PATH
  ```
- 跑 Wan2.2 t2v：
  ```
  CUDA_VISIBLE_DEVICES=1 TAG=<name> PROMPT="..." \
    bash scripts/run_wan22_ti2v5b.sh
  ```
- 跑 Wan2.2 i2v：
  ```
  CUDA_VISIBLE_DEVICES=1 TAG=<name> IMAGE=/abs/path.jpg SIZE=704*1280 PROMPT="..." \
    bash scripts/run_wan22_ti2v5b.sh
  ```
- 跑 Bernini t2v：
  ```
  cd /root/Bernini && CUDA_VISIBLE_DEVICES=1 /root/bernini-venv/bin/python infer_single_gpu.py \
    --config /home/web_server/antispam/project/clt/video-generation/Bernini-R-1.3B-Diffusers \
    --case <case.json> --guidance_mode t2v_apg
  ```
