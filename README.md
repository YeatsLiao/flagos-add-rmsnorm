# flagos-add-rmsnorm

> Triton 实现的 Add + RMSNorm 单次读写融合 kernel。4096×4096 f32 从 1.224 ms 降到 0.828 ms（1.48×），bf16 0.409 ms（1.49×）；含 FlagGems 5.0.2 对照、反向传播和 Nsight Compute 实测。

FlagOS 课程中的 Add + RMSNorm 融合算子实现与实测。用 Triton 写一个单次读写的融合 kernel，与 PyTorch 分步实现、`add` + 原生 `rms_norm` 两 kernel 做正确性和性能对比。

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

FlagGems 5.0.2 对照（同进程、重复 3 轮取中位数）：f32 4096×4096 为 0.977 ms 对 1.052 ms（0.93×），bf16 0.504 ms 对 0.486 ms（1.04×），f32 4096×8192 为 2.013 ms 对 2.042 ms（0.99×）。FlagGems 原地执行、不分配输出，本文每次调用都分配 `out` / `new_r`，因此这个比较对本文不利。该轮 GPU SM 频率为 1560 MHz，绝对值不能和上表混用，只比较同轮倍率。

Nsight Compute 2024.1.1 实测（`--clock-control none`，ECC 开启）：f32 4096×4096 的 DRAM 吞吐为 93.7%，物理读/写 154.5/155.4 MB，0.796 ms，43 寄存器/线程，79.1% 占用率；bf16 4096×4096 为 93.9%、0.402 ms；f32 4096×8192 为 92.5%，74 寄存器/线程，47.6% 占用率。手工 Roofline 的 72% 按 448 GB/s 规格值和 4-pass 理想流量计算；ncu 的 93.7% 按实际 DRAM 峰值（约 416 GB/s）和含 ECC 的物理流量计算。

反向传播：两个 Triton kernel，f32 下 `grad_x` / `grad_r` 最大绝对误差 4.8e-07，`grad_w` 6.1e-05（相对最大梯度 2.7e-07）；4096×4096 两个反向 kernel 耗时 2.044 ms（f32）、1.400 ms（bf16）。

## 目录

```text
src/
  fused_add_rms_norm.py    融合 kernel、分块版本、参考实现、两条基线
  bench.py                 正确性 + 性能
  bench_flag_gems.py       FlagGems 5.0.2 同口径对照
  block_sweep.py           num_warps / BLOCK 扫描
  launch_overhead.py       空 kernel launch 开销
  backward_add_rms_norm.py 反向 kernel 与 autograd 封装
  test_backward.py         反向正确性与耗时测试
  profile_kernel.py        Nsight Compute 单次采样入口
  smoke_triton.py          Triton 编译链路冒烟测试
scripts/
  setup_wsl.sh             WSL 环境安装
  run_all.sh               跑数并归档日志
results/
  parse_logs.py            日志转 summary.md
  make_roofline.py         Roofline 手工估算图
  summary.md               实测数据汇总
requirements.txt           依赖版本
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
python test_backward.py --shapes 64x256,4096x4096 --time
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

`results/summary.md` 是实测汇总。`scripts/run_all.sh` 会把原始日志写到本地 `logs/`，该目录不提交到仓库；文章和截图也不随仓库发布。README 中的数字来自同一组实测日志。

FlagGems 对照需要先安装 g++ 和 `flag-gems==5.0.2`（见 `src/bench_flag_gems.py` 文件头），再运行：

```bash
python src/bench_flag_gems.py --repeats 3 --log logs/flag_gems_bench.log
```

Nsight Compute 采样（需要 Windows 侧开启 GPU 性能计数器权限）：

```bash
cd src
ncu --clock-control none --kernel-name regex:fused_add_rms_norm_kernel \
  --launch-skip 3 --launch-count 1 \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  python profile_kernel.py --m 4096 --n 4096 --dtype float32
```

## 测试条件

| 项 | 配置 |
|---|---|
| GPU | NVIDIA RTX A4000 16GB，峰值带宽 448 GB/s |
| 驱动 / CUDA | KMD 616.92 / CUDA UMD 13.4 |
| 系统 | Windows 11 + WSL2，Ubuntu 26.04 |
| 软件 | Python 3.14，PyTorch 2.9.1+cu128，Triton 3.5.1，FlagGems 5.0.2 |
| 形状 | `[4096, 4096]`，补充 4096×8192，边界用例 N=1000；反向补充 64×256、64×1000、100×256 |
| dtype / eps | float32、bfloat16 / 1e-6 |
| 计时 | 预热 50 次，计时 200 次，CUDA Event 平均；对照实验重复 3 轮取中位数 |

## 参考

- FlagOS 课程：RMSNorm + Residual 融合算子项目
- [FlagGems](https://github.com/flagos-ai/FlagGems)
- [Triton](https://triton-lang.org/)
- [LLaMA](https://arxiv.org/abs/2302.13971)
