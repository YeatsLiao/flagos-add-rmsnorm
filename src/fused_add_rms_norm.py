import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_add_rms_norm_kernel(
    X, R, W, OUT, NEW_R,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    off = row * N + cols

    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + off, mask=mask, other=0.0).to(tl.float32)
    y = x + r

    tl.store(NEW_R + off, y.to(X.dtype.element_ty), mask=mask)

    var = tl.sum(y * y, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + off, (y * rstd * w).to(X.dtype.element_ty), mask=mask)


@triton.jit
def fused_add_rms_norm_tiled_kernel(
    X, R, W, OUT, NEW_R,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * N

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        y = (
            tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
            + tl.load(R + base + cols, mask=mask, other=0.0).to(tl.float32)
        )
        tl.store(NEW_R + base + cols, y.to(X.dtype.element_ty), mask=mask)
        acc += y * y

    rstd = tl.rsqrt(tl.sum(acc, axis=0) / N + eps)

    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        y = tl.load(NEW_R + base + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(OUT + base + cols, (y * rstd * w).to(X.dtype.element_ty), mask=mask)


def fused_add_rms_norm(x, r, w, eps=1e-6, num_warps=8):
    m, n = x.shape
    out = torch.empty_like(x)
    new_r = torch.empty_like(x)
    block = triton.next_power_of_2(n)
    fused_add_rms_norm_kernel[(m,)](
        x, r, w, out, new_r, n, eps, BLOCK=block, num_warps=num_warps
    )
    return out, new_r


def fused_add_rms_norm_tiled(x, r, w, eps=1e-6, block=256, num_warps=8):
    m, n = x.shape
    out = torch.empty_like(x)
    new_r = torch.empty_like(x)
    fused_add_rms_norm_tiled_kernel[(m,)](
        x, r, w, out, new_r, n, eps, BLOCK=block, num_warps=num_warps
    )
    return out, new_r


def ref_fused_add_rms_norm(x, r, w, eps=1e-6):
    y = x.float() + r.float()
    var = y.pow(2).mean(dim=-1, keepdim=True)
    out = y * torch.rsqrt(var + eps) * w.float()
    return out.to(x.dtype), y.to(x.dtype)


def baseline_stepwise(x, r, w, eps=1e-6):
    y = x + r
    var = y.float().pow(2).mean(dim=-1, keepdim=True)
    out = (y.float() * torch.rsqrt(var + eps) * w.float()).to(x.dtype)
    return out, y


def baseline_two_kernel(x, r, w, eps=1e-6):
    y = x + r
    out = F.rms_norm(y, (y.shape[-1],), w, eps)
    return out, y
