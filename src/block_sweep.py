import torch

from bench import bench
from fused_add_rms_norm import fused_add_rms_norm, fused_add_rms_norm_tiled


def measure(m, n, dtype, fn):
    torch.manual_seed(0)
    x = torch.randn(m, n, device="cuda", dtype=dtype)
    r = torch.randn_like(x)
    w = torch.randn(n, device="cuda", dtype=dtype)
    ms = bench(lambda: fn(x, r, w))
    es = x.element_size()
    bw = 4 * m * n * es / (ms * 1e-3) / 1e9
    return ms, bw


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    m, n = 4096, 4096
    print("single-pass: BLOCK = next_power_of_2(N), sweep num_warps")
    for dtype in (torch.float32, torch.bfloat16):
        name = str(dtype).replace("torch.", "")
        for warps in (4, 8, 16):
            ms, bw = measure(
                m, n, dtype,
                lambda x, r, w, _w=warps: fused_add_rms_norm(x, r, w, num_warps=_w),
            )
            print(f"[single] {name} BLOCK={n} warps={warps} "
                  f"latency={ms:.3f}ms bw={bw:.1f}GB/s")

    print("tiled: sweep BLOCK (num_warps=8)")
    for dtype in (torch.float32, torch.bfloat16):
        name = str(dtype).replace("torch.", "")
        for block in (64, 128, 256):
            ms, bw = measure(
                m, n, dtype,
                lambda x, r, w, _b=block: fused_add_rms_norm_tiled(x, r, w, block=_b),
            )
            print(f"[tiled] {name} BLOCK={block} warps=8 "
                  f"latency={ms:.3f}ms bw={bw:.1f}GB/s")


if __name__ == "__main__":
    main()
