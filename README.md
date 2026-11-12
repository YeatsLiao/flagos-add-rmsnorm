# flagos-add-rmsnorm

FlagOS 课时 18「Module II 期末项目」的 Add + RMSNorm 融合算子实现与实测。用 Triton 写一个单次读写的融合 kernel，与 PyTorch 分步实现、`add` + 原生 `rms_norm` 两 kernel 做正确性和性能对比。

## 实测结果

环境：NVIDIA RTX A4000 16GB（峰值带宽 448 GB/s），WSL2 Ubuntu 26.04，Python 3.14，PyTorch 2.9.1+cu128，Triton 3.5.1。

| 实现 | f32 4096×4096 | bf16 4096×4096 |
|---|---:|---:|
| 分步 eager | 2.052 ms | 2.660 ms |
| `add` + 原生 `rms_norm` | 1.224 ms | 0.608 ms |
| 融合 kernel | **0.828 ms** | **0.409 ms** |
| 融合有效带宽 | 324.4 GB/s | 328.2 GB/s |
| 相对两 kernel | **1.48×** | **1.49×** |

正确性：f32 最大绝对误差 1.907e-06，bf16 为 1.562e-02（bf16 输出量化量级）；N=1000 的 mask 路径同样通过。计时为预热 50 次、计时 200 次、CUDA Event 平均。

补充：N=8192 融合版 f32 1.875 ms、bf16 0.934 ms，约 1.51×；空 Triton kernel 单次 launch 约 13 µs（含 Python 调度）。

## 目录

```text
src/
  fused_add_rms_norm.py   融合 kernel、分块版本、参考实现、两条基线
  bench.py                正确性 + 性能
  block_sweep.py          num_warps / BLOCK 扫描
  launch_overhead.py      空 kernel launch 开销
  smoke_triton.py         Triton 编译链路冒烟测试
scripts/
  setup_wsl.sh            WSL 环境安装
  run_all.sh              跑数并归档日志
results/
  parse_logs.py           日志转 summary.md
  summary.md              实测数据汇总
requirements.txt          依赖版本
```

## 快速开始

### 1. 安装环境

需要 WSL2、NVIDIA GPU 和 Python 3.14：

```bash
git clone https://github.com/YeatsLiao/flagos-add-rmsnorm.git
cd flagos-add-rmsnorm
bash scripts/setup_wsl.sh
```

脚本会在 `~/flagos-add-rmsnorm` 准备运行目录和 `.venv`；如果仓库不在该目录，会把 `src/` 复制过去。随后安装 gcc、Python 头文件、torch 2.9.1+cu128、Triton 3.5.1 和 numpy，最后跑一次 Triton 冒烟测试。手动安装命令见 `scripts/setup_wsl.sh`。

### 2. 跑正确性和性能

```bash
bash scripts/run_all.sh
```

也可以手动执行：

```bash
cd ~/flagos-add-rmsnorm
source .venv/bin/activate
cd src
python bench.py
python block_sweep.py
python launch_overhead.py
```

### 3. 生成汇总

```bash
cd /path/to/flagos-add-rmsnorm
python results/parse_logs.py
```

## 实现要点

- 一个 program 处理一整行，`BLOCK = next_power_of_2(N)`，整行驻留寄存器，避免第二遍读 HBM。
- 平方和用 f32 累加；bf16 输入先转 f32，输出再转回。
- `new_residual` 单独写回，下一层残差依赖更新后的值。
- N 不是 2 的幂时用 mask，`other=0.0` 避免越界值污染平方和。
- `src/fused_add_rms_norm.py` 另有一个分块循环版本，用于 N 较大、整行放不进寄存器的场景。

## 数据说明

`results/summary.md` 是实测汇总。`scripts/run_all.sh` 会把原始日志写到本地 `logs/`，该目录不提交到仓库；文章和截图也不随仓库发布。README 中的数字来自同一组实测日志。FlagGems 的 90% 目标没有实测，文中没有给出达到该目标的结论。

## 测试条件

| 项 | 配置 |
|---|---|
| GPU | NVIDIA RTX A4000 16GB，峰值带宽 448 GB/s |
| 驱动 / CUDA | 616.92 / UMD 13.4 |
| 系统 | Windows 11 + WSL2，Ubuntu 26.04 |
| 软件 | Python 3.14，PyTorch 2.9.1+cu128，Triton 3.5.1 |
| 形状 | `[4096, 4096]`，边界用例 N=1000 |
| dtype / eps | float32、bfloat16 / 1e-6 |
| 计时 | 预热 50 次，计时 200 次，CUDA Event 平均 |

## 参考

- FlagOS 课程课时 18：Module II 期末项目（RMSNorm + Residual 融合算子）
- [FlagGems](https://github.com/flagos-ai/FlagGems)
- [Triton](https://triton-lang.org/)
- [LLaMA](https://arxiv.org/abs/2302.13971)
