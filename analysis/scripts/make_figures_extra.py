"""Figures 3 and 5: flip rates with exact 95% CIs, and accuracy change vs. FNR change for all contrasts.
Usage: python3 -I make_figures_extra.py <stats.json> <out_dir>"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

S = json.load(open(sys.argv[1]))["single"]
OUT = Path(sys.argv[2])
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6})
DS = {"qwen_thinking": "Think", "toxicchat_prompt": "ToxicChat", "xstest_prompt": "XSTest", "beavertails_response": "BeaverTails"}

# ---- Figure: flip rates with Clopper-Pearson 95% CIs ----
VIOLET, AQUA = "#6250d6", "#1baf7a"   # validated pair; aqua carries direct labels (contrast relief)
rows = []
for ds in DS:
    rows.append((f"Qwen3Guard · {DS[ds]}", S["qwen"][ds]["pairs"]["nf4->fp4"], VIOLET, "o"))
    rows.append((f"Qwen3Guard · {DS[ds]}", S["qwen"][ds]["pairs"]["fp8->int8"], AQUA, "s"))
for ds in ("xstest_prompt", "beavertails_response"):
    rows.append((f"Llama Guard 3 · {DS[ds]}", S["llama"][ds]["pairs"]["nf4->fp4"], VIOLET, "o"))
fig, ax = plt.subplots(figsize=(3.45, 2.55))
labels, ypos, y = [], [], 0
prev = None
for lab, c, col, mk in rows:
    if lab != prev:
        y -= 1.0; labels.append(lab); ypos.append(y); off = 0.17
    else:
        off = -0.17
    yy = y + off
    lo, hi = 100 * c["flip_ci"][0], 100 * c["flip_ci"][1]
    ax.plot([lo, hi], [yy, yy], color=col, linewidth=1.2, solid_capstyle="round")
    ax.plot(100 * c["flip_rate"], yy, marker=mk, color=col, markersize=4.2, markeredgecolor="white", markeredgewidth=0.5)
    ax.text(hi + 0.08, yy, f"{100 * c['flip_rate']:.1f}", va="center", ha="left", fontsize=6.2, color=INK)
    prev = lab
ax.set_yticks(ypos); ax.set_yticklabels(labels, color=INK)
ax.set_xlim(0, 5.2); ax.set_xlabel("Changed decisions, % (exact 95% CI)", color=INK); ax.xaxis.set_label_coords(0.40, -0.12)
ax.grid(axis="x", color=GRID, linewidth=0.5); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.axhline(ypos[3] - 0.5, color=GRID, linewidth=0.8)
fig.legend(handles=[Line2D([], [], color=VIOLET, marker="o", label="NF4 vs. FP4 (4-bit)"),
                    Line2D([], [], color=AQUA, marker="s", label="FP8 vs. INT8 (8-bit)")],
           loc="upper center", ncol=2, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "fig_flip_rates.pdf")

# ---- Figure: accuracy change vs FNR change, all within-family contrasts ----
BLUE, ORANGE = "#2a78d6", "#eb6834"   # validated pair; families also differ by marker shape
LABPOS = {("bf16->fp4", "beavertails_response"): (-1.25, 5.35), ("nf4->fp4", "beavertails_response"): (-1.0, 4.6),
          ("nf4->fp4", "xstest_prompt"): (-1.0, 3.9), ("bf16->fp4", "xstest_prompt"): (-1.0, 3.2)}
fig, ax = plt.subplots(figsize=(3.45, 2.45))
for fam, col, mk in (("qwen", BLUE, "o"), ("llama", ORANGE, "^")):
    for ds, v in S[fam].items():
        for pair, c in v["pairs"].items():
            x, y = 100 * c["dAcc"], 100 * c["dFNR"]
            ax.plot(x, y, marker=mk, color=col, markersize=4.3, markeredgecolor="white", markeredgewidth=0.5, linestyle="none")
            if fam == "llama" and pair.endswith("fp4"):
                a, b = pair.split("->")
                tx, ty = LABPOS[(pair, ds)]
                ax.annotate(f"{a.upper()}→{b.upper()}, {DS[ds]}", (x, y), xytext=(tx, ty), textcoords="data",
                            fontsize=6.0, color=INK, va="center",
                            arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.5, shrinkA=0, shrinkB=2))
ax.axhline(0, color=INK2, linewidth=0.6); ax.axvline(0, color=INK2, linewidth=0.6)
ax.set_xlabel("Accuracy change (percentage points)", color=INK)
ax.set_ylabel("FNR change (percentage points)", color=INK)
ax.grid(color=GRID, linewidth=0.5); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_xlim(-3.0, 1.6); ax.set_ylim(-3.0, 5.8)
fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", linestyle="none", label="Qwen3Guard contrasts (40)"),
                    Line2D([], [], color=ORANGE, marker="^", linestyle="none", label="Llama Guard 3 contrasts (6)")],
           loc="upper center", ncol=2, frameon=False, fontsize=6.6, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.92))
fig.savefig(OUT / "fig_acc_vs_fnr.pdf")
