import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
OUT = ROOT / "results" / "summary.md"

corr_re = re.compile(
    r"\[correctness\] (\S+) (\d+)x(\d+) max_abs=(\S+) max_rel=(\S+) new_r_abs=(\S+)"
)
bench_re = re.compile(
    r"\[bench\] (\S+) (\d+)x(\d+) stepwise=(\S+)ms two_kernel=(\S+)ms "
    r"fused=(\S+)ms fused_bw=(\S+)GB/s speedup_vs_two=(\S+)x"
)
single_re = re.compile(
    r"\[single\] (\S+) BLOCK=(\d+) warps=(\d+) latency=(\S+)ms bw=(\S+)GB/s"
)
tiled_re = re.compile(
    r"\[tiled\] (\S+) BLOCK=(\d+) warps=(\d+) latency=(\S+)ms bw=(\S+)GB/s"
)


def read(name):
    p = LOGS / name
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def main():
    bench = read("bench.log")
    sweep = read("block_sweep.log")

    corr = corr_re.findall(bench)
    perf = bench_re.findall(bench)
    single = single_re.findall(sweep)
    tiled = tiled_re.findall(sweep)

    lines = ["# 实测结果汇总", ""]

    lines += ["## 正确性", "", "| dtype | shape | max_abs | max_rel | new_r_abs |", "|---|---|---:|---:|---:|"]
    for dtype, m, n, a, r, nr in corr:
        lines.append(f"| {dtype} | {m}×{n} | {a} | {r} | {nr} |")

    lines += ["", "## 性能", "", "| dtype | shape | stepwise | two_kernel | fused | fused_bw | speedup |", "|---|---|---:|---:|---:|---:|---:|"]
    for dtype, m, n, s, t, f, bw, sp in perf:
        lines.append(f"| {dtype} | {m}×{n} | {s} ms | {t} ms | {f} ms | {bw} GB/s | {sp}× |")

    lines += ["", "## 单次读写：num_warps", "", "| dtype | BLOCK | warps | latency | bw |", "|---|---:|---:|---:|---:|"]
    for dtype, block, warps, ms, bw in single:
        lines.append(f"| {dtype} | {block} | {warps} | {ms} ms | {bw} GB/s |")

    lines += ["", "## 分块循环：BLOCK", "", "| dtype | BLOCK | warps | latency | bw |", "|---|---:|---:|---:|---:|"]
    for dtype, block, warps, ms, bw in tiled:
        lines.append(f"| {dtype} | {block} | {warps} | {ms} ms | {bw} GB/s |")

    if not corr and not perf and not single and not tiled:
        lines.append("日志还没有数据。先跑 scripts/run_all.sh。")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"correctness={len(corr)} bench={len(perf)} single={len(single)} tiled={len(tiled)}")


if __name__ == "__main__":
    main()
