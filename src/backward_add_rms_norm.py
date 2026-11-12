"""Triton backward for Add + RMSNorm (forward defined in fused_add_rms_norm.py).

Forward (unchanged):

    y    = x + r
    rstd = 1 / sqrt(mean(y^2) + eps)
    out  = y * rstd * w

Backward for a loss L that only depends on ``out`` (``new_residual`` is not
part of the loss), with ``grad_out = dL/dout``::

    s        = sum_k grad_out_k * y_k * w_k                   (per row)
    grad_y_j = rstd * w_j * grad_out_j - (rstd^3 * y_j / N) * s
    grad_x   = grad_y
    grad_r   = grad_y
    grad_w_j = sum_i grad_out_ij * y_ij * rstd_i              (reduction over rows)

What is Triton and what is PyTorch
----------------------------------
* ``_bwd_grad_y_kernel``  (Triton, one program per row):
  recomputes ``y = x + r`` in fp32, ``rstd``, ``s``, ``grad_y``, and writes
  ``grad_x``, ``grad_r`` and the per-row ``rstd`` (fp32 side buffer).
  ``grad_y`` is therefore fully Triton.
* ``_bwd_grad_w_kernel``  (Triton, one program per column block): streams the
  rows in ``[ROWS, BLOCK]`` tiles, tree-reduces each tile over the row axis and
  merges 4 fp32 accumulators as a tree at the end, then stores once. The
  cross-row reduction is Triton (no torch.sum on device).
* PyTorch is only used for: allocating output buffers, the (optional) reference
  ``dw_backend="torch"`` path that mirrors the reduction with ``torch.sum`` for
  debugging, and the autograd wiring in :class:`_AddRMSNormBackward`.

Precision
---------
All math is fp32. Only the final stores are cast: ``grad_x`` / ``grad_r`` to the
input dtype and ``grad_w`` to ``w.dtype`` (unless ``grad_dtype`` overrides it).
``y`` is recomputed from ``x`` and ``r`` exactly like the forward kernel does
(``float(x) + float(r)``), so no rounding of a bf16 ``new_residual`` leaks into
the gradient.

Limitation: ``_bwd_grad_w_kernel`` uses ``cdiv(N, BLOCK)`` programs, each of
which scans every row, so its parallelism is bounded by ``N / BLOCK``. That is
fine for correctness work but would need a split-K (+ atomics or a second
reduction) for production use.
"""

import torch
import triton
import triton.language as tl

from fused_add_rms_norm import fused_add_rms_norm


@triton.jit
def _bwd_grad_y_kernel(
    X, R, W, GO, DX, DR, RSTD,
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
    go = tl.load(GO + off, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(y * y, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    s = tl.sum(go * y * w, axis=0)

    gy = rstd * w * go - (rstd * rstd * rstd / N) * y * s
    tl.store(DX + off, gy.to(DX.dtype.element_ty), mask=mask)
    tl.store(DR + off, gy.to(DR.dtype.element_ty), mask=mask)
    tl.store(RSTD + row, rstd)


@triton.jit
def _grad_w_row_tile(X, R, GO, RSTD, M, N, base, rows, cols, col_mask,
                     ROWS: tl.constexpr):
    """Sum ``grad_out * y * rstd`` over a [ROWS, BLOCK] tile -> [BLOCK] fp32."""
    rr = base + rows
    row_mask = rr < M
    off = rr[:, None] * N + cols[None, :]
    tile_mask = row_mask[:, None] & col_mask[None, :]
    rstd = tl.load(RSTD + rr, mask=row_mask, other=0.0)
    x = tl.load(X + off, mask=tile_mask, other=0.0).to(tl.float32)
    r = tl.load(R + off, mask=tile_mask, other=0.0).to(tl.float32)
    go = tl.load(GO + off, mask=tile_mask, other=0.0).to(tl.float32)
    return tl.sum(go * (x + r) * rstd[:, None], axis=0)


@triton.jit
def _bwd_grad_w_kernel(
    X, R, GO, RSTD, GW,
    M, N,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    STEP: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK + tl.arange(0, BLOCK)
    col_mask = cols < N
    rows = tl.arange(0, ROWS)

    # Rows are consumed in [ROWS, BLOCK] tiles that are tree-reduced over the
    # row axis, and 4 independent accumulators are merged as a tree at the end.
    # That keeps the fp32 error of a 4096-row sum close to torch.sum's instead
    # of growing like a single sequential 4096-step accumulation.
    acc0 = tl.zeros([BLOCK], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK], dtype=tl.float32)
    # NOTE: the explicit step keeps Triton 3.5.1 off the `ast.Num` default-step
    # path, which is gone in Python 3.14 (two-arg range() fails to compile).
    for c in range(0, tl.cdiv(M, STEP), 1):
        base = c * STEP
        acc0 += _grad_w_row_tile(
            X, R, GO, RSTD, M, N, base, rows, cols, col_mask, ROWS=ROWS)
        acc1 += _grad_w_row_tile(
            X, R, GO, RSTD, M, N, base + ROWS, rows, cols, col_mask, ROWS=ROWS)
        acc2 += _grad_w_row_tile(
            X, R, GO, RSTD, M, N, base + 2 * ROWS, rows, cols, col_mask,
            ROWS=ROWS)
        acc3 += _grad_w_row_tile(
            X, R, GO, RSTD, M, N, base + 3 * ROWS, rows, cols, col_mask,
            ROWS=ROWS)

    acc = (acc0 + acc1) + (acc2 + acc3)
    tl.store(GW + cols, acc.to(GW.dtype.element_ty), mask=col_mask)


def _dw_block(n):
    return min(triton.next_power_of_2(n), 256)


_DW_ROWS = 16
_DW_ACCS = 4


def add_rms_norm_backward(
    x,
    r,
    w,
    grad_out,
    eps=1e-6,
    grad_dtype=None,
    dw_backend="triton",
    num_warps=8,
):
    """Backward of ``out = add_rms_norm(x, r, w)`` for ``loss = f(out)``.

    ``grad_x`` and ``grad_r`` are identical (both are ``grad_y``) but returned
    separately to mirror ``y = x + r``. Returns ``(grad_x, grad_r, grad_w)``.
    """
    x = x.contiguous()
    r = r.contiguous()
    grad_out = grad_out.contiguous()
    if x.shape != r.shape or x.ndim != 2:
        raise ValueError("x and r must be 2D tensors with the same shape")
    if grad_out.shape != x.shape:
        raise ValueError("grad_out must have the same shape as x")
    m, n = x.shape

    out_dtype = x.dtype if grad_dtype is None else grad_dtype
    grad_x = torch.empty((m, n), device=x.device, dtype=out_dtype)
    grad_r = torch.empty((m, n), device=x.device, dtype=out_dtype)
    rstd = torch.empty((m,), device=x.device, dtype=torch.float32)

    block = triton.next_power_of_2(n)
    _bwd_grad_y_kernel[(m,)](
        x, r, w, grad_out, grad_x, grad_r, rstd, n, eps, BLOCK=block,
        num_warps=num_warps,
    )

    if dw_backend == "triton":
        bw = _dw_block(n)
        grad_w = torch.empty((n,), device=x.device, dtype=out_dtype)
        _bwd_grad_w_kernel[(triton.cdiv(n, bw),)](
            x, r, grad_out, rstd, grad_w, m, n, BLOCK=bw, ROWS=_DW_ROWS,
            STEP=_DW_ACCS * _DW_ROWS, num_warps=4,
        )
    elif dw_backend == "torch":
        # Debug/reference path: the cross-row reduction done by torch.sum.
        y = x.float() + r.float()
        grad_w = (grad_out.float() * y * rstd[:, None]).sum(dim=0)
        grad_w = grad_w.to(out_dtype)
    else:
        raise ValueError(f"unknown dw_backend: {dw_backend}")

    return grad_x, grad_r, grad_w


class _AddRMSNormBackward(torch.autograd.Function):
    """Autograd wiring: same forward as :func:`fused_add_rms_norm`."""

    @staticmethod
    def forward(ctx, x, r, w, eps):
        out, _new_r = fused_add_rms_norm(x, r, w, eps)
        ctx.save_for_backward(x, r, w)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, r, w = ctx.saved_tensors
        grad_x, grad_r, grad_w = add_rms_norm_backward(
            x, r, w, grad_out, ctx.eps
        )
        return grad_x, grad_r, grad_w, None


def add_rms_norm(x, r, w, eps=1e-6):
    """Drop-in, autograd-capable version of ``fused_add_rms_norm``."""
    return _AddRMSNormBackward.apply(x, r, w, eps)
