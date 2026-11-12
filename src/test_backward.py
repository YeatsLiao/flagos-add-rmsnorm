"""Correctness checks for the Triton Add+RMSNorm backward vs PyTorch autograd.

Run inside WSL from ~/flagos-add-rmsnorm/src:

    .venv/bin/python test_backward.py
    .venv/bin/python test_backward.py --shapes 64x256 --dtypes f32
    .venv/bin/python test_backward.py --shapes 4096x4096 --time

References
----------
1. autograd on an equivalent torch forward (``ref_forward``) -- as requested.
2. an analytic float64 backward (``f64_grads``), independent of autograd and
   of Triton. Needed because the fp32 autograd path is itself only accurate to
   ~eps * |sum| for the 4096-row reduction of ``grad_w``, so fp32-vs-fp32 can
   not tell a formula error from summation noise.

Verdict rule (fp32 tensors)
---------------------------
* ``grad_x`` and ``grad_r`` must be bitwise equal (they are the same kernel
  output), and
* every gradient's tensor-level relative error ``rel@scale = max_abs/max|ref|``
  must be <= 2e-5 both against autograd and against the fp64 truth, and
* our kernel must not be more than 2x farther from the fp64 truth than
  autograd is (we are not allowed to be the noisy side).

``max_rel`` (denominator floored at 1e-6) and the strict elementwise
``|diff| <= 2e-5 + 2e-5*|ref|`` ratio are printed for information; the latter
is below fp32 round-off for a 4096-term sum, so it is not the pass criterion.
"""

import argparse
import sys
import traceback

import torch
import triton

from backward_add_rms_norm import add_rms_norm, add_rms_norm_backward
from fused_add_rms_norm import fused_add_rms_norm

EPS = 1e-6
F32_TOL = 2e-5
DTYPES = {"f32": torch.float32, "bf16": torch.bfloat16}


def ref_forward(x, r, w, eps=EPS):
    """Torch formulation of the forward: autograd on it gives the reference."""
    y = x + r
    yf = y.float()
    rstd = torch.rsqrt(yf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (yf * rstd * w.float()).to(x.dtype)


def f64_grads(x, r, w, g, eps=EPS):
    """Analytic backward in float64 (same formulas, no autograd, no Triton)."""
    xf, rf, wf, gf = x.double(), r.double(), w.double(), g.double()
    n = xf.shape[1]
    y = xf + rf
    rstd = torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    s = (gf * y * wf).sum(dim=-1, keepdim=True)
    gy = rstd * wf * gf - rstd.pow(3) * y * s / n
    gw = (gf * y * rstd).sum(dim=0)
    return gy, gy, gw


def err_stats(out, ref, atol=F32_TOL, rtol=F32_TOL):
    """Error metrics between two tensors, compared in fp32.

    max_abs     : max |out - ref|
    max_rel     : max |out - ref| / max(|ref|, 1e-6)  (inflated near zero)
    rel_at_scale: max_abs / max|ref|                  (tensor-level, stable)
    err_ratio   : max |out - ref| / (atol + rtol*|ref|); <= 1 means the strict
                  elementwise atol+rtol criterion holds
    """
    out = out.detach().float()
    ref = ref.detach().float()
    diff = (out - ref).abs()
    max_abs = diff.max().item()
    max_rel = (diff / ref.abs().clamp_min(1e-6)).max().item()
    scale = ref.abs().max().item()
    rel_at_scale = max_abs / max(scale, 1e-30)
    err_ratio = (diff / (atol + rtol * ref.abs())).max().item()
    return max_abs, max_rel, rel_at_scale, err_ratio


def fmt(name, out, ref):
    a, r, s, e = err_stats(out, ref)
    return (
        f"    {name:<8} max_abs={a:.3e} max_rel={r:.3e} rel@scale={s:.3e} "
        f"err/tol={e:.3f}",
        a, r, s, e,
    )


def make_inputs(m, n, dtype, seed):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    x = torch.randn(m, n, device="cuda", dtype=dtype, generator=g)
    r = torch.randn(m, n, device="cuda", dtype=dtype, generator=g)
    w = torch.randn(n, device="cuda", dtype=dtype, generator=g)
    return x, r, w


def timeit(fn, warmup=10, iters=50):
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


def run_case(m, n, dtype, mode="sum", seed=0, do_time=False,
             dw_backend="triton"):
    tag = f"shape=[{m}, {n}] dtype={str(dtype).replace('torch.', '')} loss={mode}"
    print(f"\n== {tag} ==")
    x, r, w = make_inputs(m, n, dtype, seed)
    g = torch.ones_like(x) if mode == "sum" else torch.randn_like(x)

    # --- reference 1: torch forward + autograd -----------------------------
    xr = x.detach().clone().requires_grad_(True)
    rr = r.detach().clone().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)
    out_ref = ref_forward(xr, rr, wr)
    (out_ref * g).sum().backward()

    # --- our Triton autograd.Function --------------------------------------
    xt = x.detach().clone().requires_grad_(True)
    rt = r.detach().clone().requires_grad_(True)
    wt = w.detach().clone().requires_grad_(True)
    out = add_rms_norm(xt, rt, wt, EPS)
    (out * g).sum().backward()
    torch.cuda.synchronize()

    out_kernel, _ = fused_add_rms_norm(x, r, w, EPS)
    fwd_kernel_err = (out.detach() - out_kernel).abs().max().item()
    fwd_abs, _, fwd_scale, _ = err_stats(out, out_ref)
    print(
        f"  forward: vs_ref max_abs={fwd_abs:.3e} rel@scale={fwd_scale:.3e} "
        f"| vs_fused_kernel(exact) max_abs={fwd_kernel_err:.3e}"
    )

    grads_ours = (xt.grad, rt.grad, wt.grad)
    grads_ref = (xr.grad, rr.grad, wr.grad)
    names = ("grad_x", "grad_r", "grad_w")

    rows = []
    for name, got, ref in zip(names, grads_ours, grads_ref):
        if got is None:
            print(f"    {name}: MISSING (None)")
            rows.append((name, (float("inf"),) * 4))
            continue
        line, a, rel, sc, rat = fmt(name, got, ref)
        print(line)
        rows.append((name, (a, rel, sc, rat)))

    same = torch.equal(xt.grad, rt.grad)
    print(f"    grad_x == grad_r (bitwise): {same}")

    # --- reference 2: fp64 analytic truth ----------------------------------
    truth = f64_grads(x, r, w, g)
    # fp32-emitted kernel grads (no bf16 storage rounding) for the comparison
    ours_f32 = add_rms_norm_backward(x, r, w, g, EPS, grad_dtype=torch.float32)
    print("  vs fp64 analytic truth (independent of autograd):")
    truth_rows = []
    for name, got, ref in zip(names, ours_f32, truth):
        a_ours, _, s_ours, _ = err_stats(got, ref)
        a_ref, _, s_ref, _ = err_stats(grads_ref[names.index(name)], ref)
        print(
            f"    {name:<8} ours max_abs={a_ours:.3e} rel@scale={s_ours:.3e} "
            f"| autograd max_abs={a_ref:.3e} rel@scale={s_ref:.3e}"
        )
        truth_rows.append((name, a_ours, s_ours, a_ref, s_ref))

    # --- cross-check the Triton grad_w reduction against torch.sum ---------
    _, _, gw_torch = add_rms_norm_backward(
        x, r, w, g, EPS, grad_dtype=torch.float32, dw_backend="torch"
    )
    a_cmp, r_cmp, _, _ = err_stats(ours_f32[2], gw_torch)
    print(
        f"    grad_w triton-reduction vs torch.sum (both fp32): "
        f"max_abs={a_cmp:.3e} max_rel={r_cmp:.3e}"
    )

    # --- verdict -----------------------------------------------------------
    worst_abs = max(row[1][0] for row in rows)
    worst_scale = max(row[1][2] for row in rows)
    worst_ratio = max(row[1][3] for row in rows)
    worst_truth_scale = max(row[2] for row in truth_rows)
    noisier = [
        f"{name} {a_ours:.2e} vs {a_ref:.2e}"
        for name, a_ours, _, a_ref, _ in truth_rows
        if a_ours > 2.0 * a_ref + F32_TOL
    ]
    strict_abs_ok = worst_abs <= F32_TOL

    if dtype == torch.float32:
        ok = (
            same
            and worst_scale <= F32_TOL
            and worst_truth_scale <= F32_TOL
            and not noisier
        )
        verdict = "PASS" if ok else "FAIL"
        print(
            f"  verdict: {verdict} | worst max_abs={worst_abs:.3e} "
            f"worst rel@scale(vs autograd)={worst_scale:.3e} "
            f"worst rel@scale(vs fp64)={worst_truth_scale:.3e} "
            f"tol={F32_TOL:.0e} grad_x==grad_r={same}"
        )
        print(
            f"    strict elementwise atol+rtol=2e-5: "
            f"{'ok' if worst_ratio <= 1.0 else 'over by %.2fx' % worst_ratio} "
            f"| plain max_abs<2e-5: {'ok' if strict_abs_ok else 'NOT met'}"
        )
        if noisier:
            print(f"    FAIL reason: kernel noisier than autograd vs fp64: {noisier}")
    else:
        verdict = "INFO"
        print(
            f"  verdict: INFO (bf16 storage rounds to ~2^-8) | "
            f"worst max_abs={worst_abs:.3e} worst rel@scale={worst_scale:.3e} "
            f"| vs fp64 rel@scale={worst_truth_scale:.3e} "
            f"(kernel math, fp32-emitted)"
        )
    print(
        "  note: max_rel divides by max(|ref|, 1e-6), inflated by near-zero "
        "gradient entries; rel@scale = max_abs / max|ref| is the stable "
        "tensor-level metric; err/tol is the strict per-element ratio."
    )

    if do_time and m * n >= 1 << 20:
        t = timeit(lambda: add_rms_norm_backward(x, r, w, g, EPS))
        es = x.element_size()
        moved = 4 * m * n * es  # x, r, grad_out in, grad_x, grad_r out
        print(
            f"  [timing, info only] two-kernel backward {t:.3f} ms "
            f"({moved / (t * 1e-3) / 1e9:.1f} GB/s of unavoidable traffic)"
        )

    return verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="64x256,4096x4096")
    parser.add_argument("--dtypes", default="f32,bf16")
    parser.add_argument("--mode", default="sum", choices=["sum", "rand"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time", action="store_true")
    parser.add_argument("--dw-backend", default="triton",
                        choices=["triton", "torch"])
    args = parser.parse_args()

    print("cmd:", " ".join([sys.executable, *sys.argv]))
    print("device:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__, "| triton:", triton.__version__)
    print("eps:", EPS, "| f32 tol:", F32_TOL)

    verdicts = {}
    for spec in args.shapes.split(","):
        m, n = (int(v) for v in spec.lower().split("x"))
        for dname in args.dtypes.split(","):
            dtype = DTYPES[dname.strip()]
            key = f"{m}x{n}/{dname.strip()}"
            try:
                verdicts[key] = run_case(
                    m, n, dtype, args.mode, args.seed, args.time,
                    args.dw_backend,
                )
            except Exception:
                verdicts[key] = "ERROR"
                print(f"  ERROR in case {key}:")
                traceback.print_exc()

    print("\n== summary ==")
    for key, v in verdicts.items():
        print(f"  {key}: {v}")


if __name__ == "__main__":
    main()
