from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images" / "05-roofline.png"
OUT.parent.mkdir(exist_ok=True)

PEAK_BW = 448.0  # GB/s
PEAK_FP32 = 19.17  # TFLOP/s
RIDGE = PEAK_FP32 / (PEAK_BW / 1000.0)  # FLOP/byte

ai = np.logspace(-2, 2.7, 400)
memory_roof = (PEAK_BW / 1000.0) * ai
compute_roof = np.full_like(ai, PEAK_FP32)
roof = np.minimum(memory_roof, compute_roof)

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
ax.loglog(ai, roof, color="#1f77b4", linewidth=2.5, label="A4000 roofline")
ax.axvline(RIDGE, color="#888888", linestyle="--", linewidth=1)
ax.text(RIDGE * 1.08, 0.02, f"ridge ≈ {RIDGE:.1f} FLOP/byte", fontsize=9, color="#555555")

# fused f32: 5 FLOPs (add, square, reduce-add, two muls) / 16 bytes, 324.4 GB/s
ai_f32 = 5.0 / 16
tflops_f32 = ai_f32 * (324.4 / 1000.0)
ax.scatter([ai_f32], [tflops_f32], color="#d62728", zorder=5, label="fused f32 measured")
ax.annotate(
    f"fused f32\nAI≈{ai_f32:.2f}, {324.4:.1f} GB/s\nmanual: 72% of 448 GB/s\nNsight DRAM: 93.7%",
    xy=(ai_f32, tflops_f32),
    xytext=(ai_f32 * 2.2, tflops_f32 * 0.18),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1),
)

ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
ax.set_ylabel("Throughput (TFLOP/s)")
ax.set_title("RTX A4000 roofline, manual estimate")
ax.set_xlim(1e-2, 5e2)
ax.set_ylim(1e-3, 3e1)
ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
