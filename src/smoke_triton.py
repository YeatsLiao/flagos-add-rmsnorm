import torch
import triton
import triton.language as tl


@triton.jit
def add_one(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    x = tl.load(X + i, mask=m, other=0.0)
    tl.store(Y + i, x + 1.0, mask=m)


def main():
    n = 1024
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    add_one[(triton.cdiv(n, 256),)](x, y, n, BLOCK=256)
    torch.cuda.synchronize()
    ok = torch.allclose(y, x + 1.0)
    print("triton smoke:", ok)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
