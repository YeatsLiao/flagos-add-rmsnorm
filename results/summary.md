# 实测结果汇总

## 正确性

| dtype | shape | max_abs | max_rel | new_r_abs |
|---|---|---:|---:|---:|
| float32 | 4096×4096 | 1.907e-06 | 3.471e-07 | 0.000e+00 |
| bfloat16 | 4096×4096 | 1.562e-02 | 7.519e-03 | 0.000e+00 |
| float32 | 4096×1000 | 1.907e-06 | 3.368e-07 | 0.000e+00 |

## 性能

| dtype | shape | stepwise | two_kernel | fused | fused_bw | speedup |
|---|---|---:|---:|---:|---:|---:|
| float32 | 4096×4096 | 2.052 ms | 1.224 ms | 0.828 ms | 324.4 GB/s | 1.48× |
| bfloat16 | 4096×4096 | 2.660 ms | 0.608 ms | 0.409 ms | 328.2 GB/s | 1.49× |
| float32 | 4096×1000 | 0.516 ms | 0.270 ms | 0.199 ms | 328.6 GB/s | 1.35× |

## 单次读写：num_warps

| dtype | BLOCK | warps | latency | bw |
|---|---:|---:|---:|---:|
| float32 | 4096 | 4 | 0.827 ms | 324.6 GB/s |
| float32 | 4096 | 8 | 0.820 ms | 327.5 GB/s |
| float32 | 4096 | 16 | 0.821 ms | 327.1 GB/s |
| bfloat16 | 4096 | 4 | 0.409 ms | 328.1 GB/s |
| bfloat16 | 4096 | 8 | 0.409 ms | 328.5 GB/s |
| bfloat16 | 4096 | 16 | 0.405 ms | 331.1 GB/s |

## 分块循环：BLOCK

表中 bw 按 4 pass 折算，便于和单次读写版本同口径比较；分块版本实际搬运 5 pass。

| dtype | BLOCK | warps | latency | bw (4-pass) |
|---|---:|---:|---:|---:|
| float32 | 64 | 8 | 1.122 ms | 239.3 GB/s |
| float32 | 128 | 8 | 1.041 ms | 257.8 GB/s |
| float32 | 256 | 8 | 1.037 ms | 258.8 GB/s |
| bfloat16 | 64 | 8 | 0.753 ms | 178.2 GB/s |
| bfloat16 | 128 | 8 | 0.522 ms | 257.2 GB/s |
| bfloat16 | 256 | 8 | 0.459 ms | 292.1 GB/s |

## 补充形状：N=8192

| dtype | shape | stepwise | two_kernel | fused | fused_bw | speedup |
|---|---|---:|---:|---:|---:|---:|
| float32 | 4096×8192 | 4.690 ms | 2.828 ms | 1.875 ms | 286.4 GB/s | 1.51× |
| bfloat16 | 4096×8192 | 6.070 ms | 1.414 ms | 0.934 ms | 287.6 GB/s | 1.51× |

## Nsight Compute 实测

Nsight Compute 2024.1.1，`--clock-control none`，ECC 开启，单次 kernel launch 计数。

| shape | dtype | duration | DRAM % | DRAM read / write | 物理/理想 4-pass | regs/thread | occupancy |
|---|---|---:|---:|---:|---:|---:|---:|
| 4096×4096 | float32 | 0.796 ms | 93.7% | 154.5 / 155.4 MB | 1.15× | 43 | 79.1% |
| 4096×4096 | bfloat16 | 0.402 ms | 93.9% | 78.6 / 78.2 MB | 1.17× | 40 | 95.0% |
| 4096×8192 | float32 | —（时钟浮动） | 92.5% | 333.1 / 338.1 MB | 1.25× | 74 | 47.6% |

ncu 报告 DRAM 频率 6.49 GHz（12.98 Gbps × 256 bit / 8 ≈ 415.6 GB/s）。ECC 开启使物理流量比理想 4-pass 高 15–17%；N=8192 高 25%，寄存器升到 74/线程，占用率降到 47.6%。手工 Roofline 的 72% 按 448 GB/s 规格峰值和理想流量计算，ncu 的 93.7% 按实际 DRAM 峰值和物理流量计算。

## FlagGems 5.0.2 对照

同进程、预热 50 次、计时 200 次、重复 3 轮取中位数。FlagGems 原地执行、不分配输出；本文为公开 API，每次调用分配 `out` / `new_r`。本轮 GPU SM 频率 1560 MHz，绝对延迟不能和上表混用，只比较同轮倍率。

| shape | dtype | FlagGems | 本文（公开 API） | 同轮本文（预分配） | 吞吐比（API） |
|---|---|---:|---:|---:|---:|
| 4096×4096 | float32 | 0.977 ms | 1.052 ms | 0.977 ms | 0.93× |
| 4096×4096 | bfloat16 | 0.504 ms | 0.486 ms | 0.524 ms | 1.04× |
| 4096×8192 | float32 | 2.013 ms | 2.042 ms | 1.978 ms | 0.99× |

吞吐比 = FlagGems 时间 / 本文时间。正确性：f32 最大绝对误差 FlagGems 2.861e-06、本文 1.907e-06；bf16 两者均为 1.562e-02。

## 反向传播

f32，loss=sum，与 PyTorch autograd 对照；相对误差按 max_abs / max|ref|。

| shape | grad_x / grad_r max_abs | grad_w max_abs | grad_w 相对误差 | 结论 |
|---|---:|---:|---:|---|
| 64×256 | 2.384e-07 | 3.815e-06 | 1.470e-07 | PASS |
| 4096×4096 | 4.768e-07 | 6.104e-05 | 2.712e-07 | PASS（相对口径） |
| 64×1000 | 4.768e-07 | 2.861e-06 | 1.207e-07 | PASS |
| 100×256 | 3.576e-07 | 3.815e-06 | 1.392e-07 | PASS |

bf16 梯度以 bf16 存储，相对误差约 4.9e-03；改用 f32 输出后降到 1.5e-07。4096×4096 的两个反向 kernel 耗时：f32 2.044 ms、bf16 1.400 ms。

## Kernel Launch 开销

独立一轮运行，空 kernel 计时 500 次，其余 200 次。

| 项 | 时间 |
|---|---:|
| 空 kernel，grid=1 | 13.19 µs |
| 空 kernel，grid=4096 | 13.00 µs |
| 融合 kernel | 0.962 ms |
| add + 原生 rms_norm | 1.420 ms |
| 分步 eager | 2.367 ms |
