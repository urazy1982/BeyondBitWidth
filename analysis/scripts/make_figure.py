"""Figure 1: NF4 -> FP4 changes in unsafe-class decisions, per 100 unsafe inputs.
Usage: python3 -I make_figure.py <stats.json> <out.pdf>"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = json.load(open(sys.argv[1]))["single"]
cells = [("qwen", "qwen_thinking", "Qwen3Guard · Think"), ("qwen", "toxicchat_prompt", "Qwen3Guard · ToxicChat"),
         ("qwen", "xstest_prompt", "Qwen3Guard · XSTest"), ("qwen", "beavertails_response", "Qwen3Guard · BeaverTails"),
         ("llama", "xstest_prompt", "Llama Guard 3 · XSTest"), ("llama", "beavertails_response", "Llama Guard 3 · BeaverTails")]
MISSED, RECOV = "#eb6834", "#2a78d6"   # validated categorical slots 2 and 1 (CVD-safe pair)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.45, 2.25))
y = list(range(len(cells)))[::-1]
for yi, (fam, ds, lab) in zip(y, cells):
    c = S[fam][ds]["pairs"]["nf4->fp4"]; nu = S[fam][ds]["metrics"]["nf4"]["unsafe"]
    m, r = 100 * c["newly_missed"] / nu, 100 * c["recovered"] / nu
    ax.barh(yi, -m, height=0.55, color=MISSED, edgecolor="white", linewidth=0.6)
    ax.barh(yi, r, height=0.55, color=RECOV, edgecolor="white", linewidth=0.6)
    ax.text(-m - 0.12, yi, f"{c['newly_missed']}", va="center", ha="right", color=INK, fontsize=6.5)
    ax.text(r + 0.12, yi, f"{c['recovered']}", va="center", ha="left", color=INK, fontsize=6.5)
ax.axvline(0, color=INK2, linewidth=0.6)
ax.set_yticks(y); ax.set_yticklabels([c[2] for c in cells], color=INK)
ax.set_xlim(-5.4, 2.0)
ax.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2]); ax.set_xticklabels(["5", "4", "3", "2", "1", "0", "1", "2"], color=INK2)
ax.set_xlabel("Changed decisions per 100 unsafe inputs\n(bar labels: counts)", color=INK, fontsize=6.8); ax.xaxis.set_label_coords(0.42, -0.16)
ax.grid(axis="x", color=GRID, linewidth=0.5); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.axhline(1.5, color=GRID, linewidth=0.8)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=MISSED, label="Newly missed under FP4"), Patch(color=RECOV, label="Newly detected under FP4")],
          loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6.8, handlelength=1.2)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.92))
fig.savefig(sys.argv[2]); fig.savefig(sys.argv[2].replace(".pdf", ".png"), dpi=220)
