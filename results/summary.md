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

| dtype | BLOCK | warps | latency | bw |
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

## Kernel Launch 开销

独立一轮运行，空 kernel 计时 500 次，其余 200 次。

| 项 | 时间 |
|---|---:|
| 空 kernel，grid=1 | 13.19 µs |
| 空 kernel，grid=4096 | 13.00 µs |
| 融合 kernel | 0.962 ms |
| add + 原生 rms_norm | 1.420 ms |
| 分步 eager | 2.367 ms |
