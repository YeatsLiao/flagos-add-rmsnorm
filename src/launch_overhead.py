import torch
import triton
import triton.language as tl

from fused_add_rms_norm import (
    baseline_stepwise,
    baseline_two_kernel,
    fused_add_rms_norm,
)


@triton.jit
def _empty_kernel(X):
    tl.store(X + tl.program_id(0), 0.0)


def bench(fn, warmup=50, iters=500):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    scratch = torch.empty(4096, device="cuda", dtype=torch.float32)
    empty_1 = bench(lambda: _empty_kernel[(1,)](scratch))
    empty_4096 = bench(lambda: _empty_kernel[(4096,)](scratch))

    m = n = 4096
    torch.manual_seed(0)
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    r = torch.randn_like(x)
    w = torch.randn(n, device="cuda", dtype=torch.float32)

    t_fused = bench(lambda: fused_add_rms_norm(x, r, w), warmup=50, iters=200)
    t_two = bench(lambda: baseline_two_kernel(x, r, w), warmup=50, iters=200)
    t_step = bench(lambda: baseline_stepwise(x, r, w), warmup=50, iters=200)

    print(f"[launch] empty grid=1    {empty_1 * 1000:.2f} us")
    print(f"[launch] empty grid=4096 {empty_4096 * 1000:.2f} us")
    print(f"[launch] fused           {t_fused:.3f} ms")
    print(f"[launch] two_kernel      {t_two:.3f} ms")
    print(f"[launch] stepwise        {t_step:.3f} ms")


if __name__ == "__main__":
    main()
