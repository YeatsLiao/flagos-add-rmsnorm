"""Compare FlagGems fused_add_rms_norm against our Triton fused kernel.

FlagGems' API is in-place: fused_add_rms_norm(x, residual, normalized_shape, weight, eps)
mutates both x and residual and returns (x, residual). Inputs are cloned before the
correctness check and preallocated buffers are reused for the timing loop.
"""

import argparse
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fused_add_rms_norm import (  # noqa: E402
    baseline_two_kernel,
    fused_add_rms_norm,
    fused_add_rms_norm_kernel,
    ref_fused_add_rms_norm,
)

from flag_gems.fused import fused_add_rms_norm as gems_fused_add_rms_norm  # noqa: E402

EPS = 1e-6
WARMUP = 50
ITERS = 200
REPEATS = 3


def fused_add_rms_norm_into(x, r, w, out, new_r, eps=EPS):
    m, n = x.shape
    block = triton.next_power_of_2(n)
    fused_add_rms_norm_kernel[(m,)](
        x, r, w, out, new_r, n, eps, BLOCK=block, num_warps=8
    )
    return out, new_r


class Tee:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.file = None
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.file = open(path, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.terminal.flush()
        if self.file is not None:
            self.file.flush()


def bench(fn, warmup=WARMUP, iters=ITERS):
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


def bench_repeat(fn, repeats=None):
    repeats = REPEATS if repeats is None else repeats
    runs = [bench(fn) for _ in range(repeats)]
    runs_sorted = sorted(runs)
    median = runs_sorted[len(runs_sorted) // 2]
    return median, min(runs), runs


def errors(out, ref):
    diff = (out.float() - ref.float()).abs()
    rel = diff / ref.float().abs().clamp_min(1e-6)
    return diff.max().item(), rel.max().item()


def bandwidth_gbs(m, n, element_size, ms):
    # read x + read residual + write normalized out + write updated residual
    return 4 * m * n * element_size / (ms * 1e-3) / 1e9


def run(m, n, dtype):
    name = str(dtype).replace("torch.", "")
    torch.manual_seed(0)
    x = torch.randn(m, n, device="cuda", dtype=dtype)
    r = torch.randn_like(x)
    w = torch.randn(n, device="cuda", dtype=dtype)
    es = x.element_size()

    ref_out, ref_res = ref_fused_add_rms_norm(x, r, w, EPS)
    ours_out, ours_res = fused_add_rms_norm(x, r, w, EPS)

    # --- FlagGems correctness on cloned inputs ---
    xg = x.clone()
    rg = r.clone()
    xg_ptr, rg_ptr = xg.data_ptr(), rg.data_ptr()
    g_out, g_res = gems_fused_add_rms_norm(xg, rg, (n,), w, EPS)
    torch.cuda.synchronize()
    inplace = (g_out.data_ptr() == xg_ptr) and (g_res.data_ptr() == rg_ptr)
    out_abs, out_rel = errors(g_out, ref_out)
    res_abs, res_rel = errors(g_res, ref_res)
    ours_abs, ours_rel = errors(ours_out, ref_out)
    ours_res_abs, _ = errors(ours_res, ref_res)
    print(
        f"[fg-correctness] {name} {m}x{n} eps={EPS:g} "
        f"out_max_abs={out_abs:.3e} out_max_rel={out_rel:.3e} "
        f"res_max_abs={res_abs:.3e} res_max_rel={res_rel:.3e} "
        f"inplace={inplace} "
        f"ours_out_max_abs={ours_abs:.3e} ours_out_max_rel={ours_rel:.3e} "
        f"ours_res_max_abs={ours_res_abs:.3e}",
        flush=True,
    )

    # FlagGems is in-place: keep a dedicated buffer and call it repeatedly.
    x_fg = x.clone()
    r_fg = r.clone()
    out_pre = torch.empty_like(x)
    nr_pre = torch.empty_like(x)
    t_gems, t_gems_min, gems_run = bench_repeat(
        lambda: gems_fused_add_rms_norm(x_fg, r_fg, (n,), w, EPS)
    )
    t_pre, t_pre_min, pre_run = bench_repeat(
        lambda: fused_add_rms_norm_into(x, r, w, out_pre, nr_pre, EPS)
    )
    t_ours, t_ours_min, ours_run = bench_repeat(lambda: fused_add_rms_norm(x, r, w, EPS))
    t_two, t_two_min, two_run = bench_repeat(lambda: baseline_two_kernel(x, r, w, EPS))

    bw_gems = bandwidth_gbs(m, n, es, t_gems)
    bw_pre = bandwidth_gbs(m, n, es, t_pre)
    bw_ours = bandwidth_gbs(m, n, es, t_ours)
    bw_two = bandwidth_gbs(m, n, es, t_two)
    print(
        f"[fg-bench] {name} {m}x{n} "
        f"gems={t_gems:.4f}ms gems_bw={bw_gems:.1f}GB/s "
        f"ours_prealloc={t_pre:.4f}ms ours_pre_bw={bw_pre:.1f}GB/s "
        f"ours_fused={t_ours:.4f}ms ours_bw={bw_ours:.1f}GB/s "
        f"two_kernel={t_two:.4f}ms two_bw={bw_two:.1f}GB/s "
        f"gems_vs_pre={t_gems / t_pre:.2f}x gems_vs_fused={t_gems / t_ours:.2f}x "
        f"fused_vs_two={t_two / t_ours:.2f}x "
        f"warmup={WARMUP} iters={ITERS} repeats={REPEATS} "
        f"min_ms gems={t_gems_min:.4f} pre={t_pre_min:.4f} "
        f"ours={t_ours_min:.4f} two={t_two_min:.4f} "
        f"runs_ms gems={[round(v, 4) for v in gems_run]} "
        f"pre={[round(v, 4) for v in pre_run]} "
        f"ours={[round(v, 4) for v in ours_run]} "
        f"two={[round(v, 4) for v in two_run]}",
        flush=True,
    )
    return t_gems, t_pre, t_ours, t_two


def main():
    global REPEATS
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--skip-8192", action="store_true")
    args = parser.parse_args()
    REPEATS = args.repeats

    sys.stdout = Tee(args.log)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    import flag_gems

    print("device:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)
    print("triton:", triton.__version__)
    print("flag_gems:", flag_gems.__version__)
    print("eps:", EPS, "warmup:", WARMUP, "iters:", ITERS, "repeats:", REPEATS)
    print(
        "note: gems and ours_prealloc run in-place on preallocated buffers; "
        "ours_fused and two_kernel allocate out/new_r inside the call",
        flush=True,
    )

    run(4096, 4096, torch.float32)
    run(4096, 4096, torch.bfloat16)
    if not args.skip_8192:
        run(4096, 8192, torch.float32)
    print("[fg-bench] done", flush=True)


if __name__ == "__main__":
    main()
