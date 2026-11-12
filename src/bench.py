import argparse

import torch
import triton

from fused_add_rms_norm import (
    baseline_stepwise,
    baseline_two_kernel,
    fused_add_rms_norm,
    ref_fused_add_rms_norm,
)


def bench(fn, warmup=50, iters=200):
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


def errors(out, ref):
    diff = (out.float() - ref.float()).abs()
    rel = diff / ref.float().abs().clamp_min(1e-6)
    return diff.max().item(), rel.max().item()


def run(m, n, dtype, eps=1e-6):
    torch.manual_seed(0)
    x = torch.randn(m, n, device="cuda", dtype=dtype)
    r = torch.randn_like(x)
    w = torch.randn(n, device="cuda", dtype=dtype)

    out_ref, nr_ref = ref_fused_add_rms_norm(x, r, w, eps)
    out, new_r = fused_add_rms_norm(x, r, w, eps)
    torch.cuda.synchronize()

    max_abs, max_rel = errors(out, out_ref)
    nr_abs, _ = errors(new_r, nr_ref)

    t_fused = bench(lambda: fused_add_rms_norm(x, r, w, eps))
    t_two = bench(lambda: baseline_two_kernel(x, r, w, eps))
    t_step = bench(lambda: baseline_stepwise(x, r, w, eps))

    es = x.element_size()
    bandwidth = 4 * m * n * es / (t_fused * 1e-3) / 1e9

    name = str(dtype).replace("torch.", "")
    print(
        f"[correctness] {name} {m}x{n} "
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} new_r_abs={nr_abs:.3e}"
    )
    print(
        f"[bench] {name} {m}x{n} "
        f"stepwise={t_step:.3f}ms two_kernel={t_two:.3f}ms "
        f"fused={t_fused:.3f}ms fused_bw={bandwidth:.1f}GB/s "
        f"speedup_vs_two={t_two / t_fused:.2f}x"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    print("device:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)
    print("triton:", triton.__version__)

    for dtype in (torch.float32, torch.bfloat16):
        run(args.m, args.n, dtype)
    run(4096, 1000, torch.float32)


if __name__ == "__main__":
    main()
