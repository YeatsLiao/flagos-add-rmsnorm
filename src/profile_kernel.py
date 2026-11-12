"""Single-kernel launch used by Nsight Compute.

Run from src/ so the local import works:
    python profile_kernel.py --m 4096 --n 4096 --dtype float32 --warmup 3
Profile the last launch with:
    ncu --launch-skip 3 --launch-count 1 --kernel-name regex:fused_add_rms_norm_kernel ...
"""

import argparse

import torch
import triton

from fused_add_rms_norm import fused_add_rms_norm_kernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    x = torch.randn(args.m, args.n, device="cuda", dtype=dtype)
    r = torch.randn_like(x)
    w = torch.randn(args.n, device="cuda", dtype=dtype)
    out = torch.empty_like(x)
    new_r = torch.empty_like(x)
    block = triton.next_power_of_2(args.n)

    for _ in range(args.warmup):
        fused_add_rms_norm_kernel[(args.m,)](
            x, r, w, out, new_r, args.n, 1e-6, BLOCK=block, num_warps=8
        )
    torch.cuda.synchronize()

    # Profiled launch: ncu --launch-skip <warmup> --launch-count 1
    fused_add_rms_norm_kernel[(args.m,)](
        x, r, w, out, new_r, args.n, 1e-6, BLOCK=block, num_warps=8
    )
    torch.cuda.synchronize()
    print(f"warmup={args.warmup} shape={args.m}x{args.n} dtype={args.dtype}")


if __name__ == "__main__":
    main()
