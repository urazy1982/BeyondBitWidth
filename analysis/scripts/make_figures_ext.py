"""Figures for the extended TIFS version.
Usage: python3 -I make_figures_ext.py <extra.json> <out_dir>
fig_codebook.pdf   - NF4/FP4 code points and blockwise-absmax reconstruction on synthetic weights
fig_pairwise.pdf   - pairwise decision disagreement between all deployments
fig_power.pdf      - unsafe inputs needed to detect a given rise in miss rate with an exact McNemar test
Writes power.json and codebook.json with the plotted numbers.
"""
import json, sys
from math import comb
from pathlib import Path
import numpy as np
from scipy.stats import binom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

X = json.load(open(sys.argv[1])); OUT = Path(sys.argv[2])
NF4C, FP4C = "#2a78d6", "#eb6834"            # categorical slots 1 and 2 (CVD-validated pair)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2})

# ---------------- codebooks (bitsandbytes get_4bit_type, normalized by absmax) ----------------
NF4 = np.array([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                0.7229568362236023, 1.0])
FP4raw = np.array([0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0, -0.0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0])
FP4 = FP4raw / np.abs(FP4raw).max()


def quant(w, code, block=64):
    w = w.reshape(-1, block); s = np.abs(w).max(axis=1, keepdims=True)
    x = w / s
    idx = np.abs(x[..., None] - code[None, None, :]).argmin(-1)
    return (code[idx] * s).ravel(), code[idx].ravel()


rng = np.random.default_rng(20261109)
n = 64 * 40000
dists = {"Gaussian": rng.standard_normal(n), "Laplace": rng.laplace(size=n) / np.sqrt(2),
         "Student-$t_3$": rng.standard_t(3, size=n) / np.sqrt(3)}
CB = {}
for name, w in dists.items():
    CB[name] = {}
    for cname, code in (("NF4", NF4), ("FP4", FP4)):
        wq, lev = quant(w, code)
        rel = float(np.sqrt(np.mean((wq - w) ** 2)) / np.sqrt(np.mean(w ** 2)))
        nearzero = float(np.mean(np.abs(lev) < 0.01))
        CB[name][cname] = dict(rel_rmse=rel, near_zero=nearzero)
CB["levels"] = dict(NF4=sorted(set(np.round(NF4, 6))), FP4=sorted(set(np.round(np.abs(FP4) * np.sign(FP4raw + 1e-12), 6))))
CB["distinct"] = dict(NF4=int(len(set(NF4))), FP4=int(len(set(FP4))))
(OUT / "codebook.json").write_text(json.dumps(CB, indent=1))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.45, 1.85), gridspec_kw=dict(width_ratios=[1.25, 1]))
for y, code, col, lab in ((1, NF4, NF4C, "NF4"), (0, np.unique(FP4), FP4C, "FP4")):
    a1.hlines(y, -1, 1, color=GRID, linewidth=0.8, zorder=1)
    a1.scatter(code, np.full(len(code), y), s=12, color=col, edgecolor="white", linewidth=0.5, zorder=3)
    a1.text(-1.08, y, lab, ha="right", va="center", color=INK, fontsize=7)
a1.set_ylim(-0.6, 1.6); a1.set_xlim(-1.05, 1.05); a1.set_yticks([])
a1.set_xticks([-1, -0.5, 0, 0.5, 1]); a1.set_xticklabels(["$-1$", "$-0.5$", "0", "0.5", "1"])
a1.set_xlabel("Code point / block absmax", color=INK, fontsize=6.8)
for s in ("top", "right", "left"):
    a1.spines[s].set_visible(False)
a1.set_title("(a) 4-bit code points", fontsize=7, color=INK, loc="left")

names = list(dists); x = np.arange(len(names)); wdt = 0.36
for k, (cname, col) in enumerate((("NF4", NF4C), ("FP4", FP4C))):
    v = [100 * CB[nm][cname]["near_zero"] for nm in names]
    a2.bar(x + (k - 0.5) * wdt, v, width=wdt - 0.03, color=col, label=cname)
    for xi, vi in zip(x, v):
        a2.text(xi + (k - 0.5) * wdt, vi + 0.6, f"{vi:.0f}", ha="center", va="bottom", fontsize=6, color=INK)
a2.set_xticks(x); a2.set_xticklabels(["Gauss.", "Laplace", "$t_3$"], fontsize=6.5)
a2.set_ylabel("Weights coded near zero (%)", color=INK, fontsize=6.5)
a2.grid(axis="y", color=GRID, linewidth=0.5); a2.set_axisbelow(True)
for s in ("top", "right"):
    a2.spines[s].set_visible(False)
a2.set_ylim(0, max(100 * CB[nm]["FP4"]["near_zero"] for nm in names) * 1.25)
a2.legend(frameon=False, fontsize=6.3, loc="upper left", handlelength=1)
a2.set_title("(b) Small weights zeroed", fontsize=7, color=INK, loc="left")
fig.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig(OUT / "fig_codebook.pdf"); fig.savefig(OUT / "fig_codebook.png", dpi=220)

# ---------------- pairwise disagreement heatmaps ----------------
SEQ = LinearSegmentedColormap.from_list("blue", ["#f7f7f5", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
panels = [("qwen/qwen_thinking", "Qwen3Guard · Think"), ("qwen/toxicchat_prompt", "Qwen3Guard · ToxicChat"),
          ("qwen/xstest_prompt", "Qwen3Guard · XSTest"), ("qwen/beavertails_response", "Qwen3Guard · BeaverTails"),
          ("llama/xstest_prompt", "Llama Guard 3 · XSTest"), ("llama/beavertails_response", "Llama Guard 3 · BeaverTails")]
AN = {"bf16": "BF16", "nf4": "NF4", "fp4": "FP4", "fp8": "FP8", "int8": "INT8"}
vmax = 3.5
fig, axes = plt.subplots(1, 6, figsize=(7.16, 1.72), gridspec_kw=dict(width_ratios=[5, 5, 5, 5, 3, 3]))
for ax, (key, title) in zip(axes, panels):
    M = X["pairwise"][key]
    arms = ["bf16", "nf4", "fp4", "fp8", "int8"] if key.startswith("qwen") else ["bf16", "nf4", "fp4"]
    k = len(arms); A = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(i):
            c, nn = M.get(f"{arms[j]}-{arms[i]}") or M.get(f"{arms[i]}-{arms[j]}")
            A[i, j] = 100 * c / nn
    ax.imshow(np.ma.masked_invalid(A), cmap=SEQ, vmin=0, vmax=vmax)
    for i in range(k):
        for j in range(i):
            v = A[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.6,
                    color="white" if v > 0.62 * vmax else INK)
            if {arms[i], arms[j]} == {"nf4", "fp4"}:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=FP4C, linewidth=1.2))
    ax.set_xticks(range(k - 1)); ax.set_xticklabels([AN[a] for a in arms[:-1]], fontsize=5.6, rotation=90)
    ax.set_yticks(range(1, k)); ax.set_yticklabels([AN[a] for a in arms[1:]], fontsize=5.6)
    ax.set_xlim(-0.5, k - 1.5); ax.set_ylim(k - 0.5, 0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title.replace(" · ", "\n"), fontsize=6.2, color=INK)
fig.tight_layout(pad=0.25, w_pad=0.5)
fig.savefig(OUT / "fig_pairwise.pdf"); fig.savefig(OUT / "fig_pairwise.png", dpi=220)

# ---------------- power / sample size for an exact McNemar test on the unsafe class ----------------
ALPHA, TARGET = 0.05, 0.80


def crit(D):
    """Smallest c with two-sided exact p(c, D-c) <= ALPHA for c >= D/2 (None if impossible)."""
    tot = 2 ** D; acc = 0
    for lo in range(0, D // 2 + 1):
        acc += comb(D, lo)
        if min(1.0, 2 * acc / tot) > ALPHA:
            return D - lo + 1 if lo > 0 else None
    return None

CRIT = {}
def power(N, psi, delta):
    p = 0.5 * (1 + delta / psi)
    Ds = np.arange(0, N + 1); w = binom.pmf(Ds, N, psi)
    keep = w > 1e-12; pw = 0.0
    for D, wd in zip(Ds[keep], w[keep]):
        D = int(D)
        if D not in CRIT:
            CRIT[D] = crit(D) if D < 1030 else None
        c = CRIT[D]
        if c is None:
            continue
        pw += wd * (binom.sf(c - 1, D, p) + binom.cdf(D - c, D, p))
    return pw


def n_needed(psi, delta, lo=20, hi=6000):
    # coarse-to-fine scan for the smallest N with power >= TARGET
    for step in (200, 20, 2):
        N = lo
        while N <= hi and power(N, psi, delta) < TARGET:
            N += step
        if N > hi:
            return None
        lo, hi = max(20, N - step), N
    return hi

PW = {}
shares = [0.7, 0.85, 1.0]          # share of unsafe-class flips that are newly missed inputs
deltas = [0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6]
for r in shares:
    PW[str(r)] = {}
    for d in deltas:
        psi = (d / 100) / (2 * r - 1)
        if psi < 0.3:
            PW[str(r)][str(d)] = n_needed(psi, d / 100, hi=30000)
(OUT / "power.json").write_text(json.dumps(PW, indent=1))
print(json.dumps(PW, indent=1))
print(json.dumps({k: v for k, v in CB.items() if k not in ("levels",)}, indent=1))

fig, ax = plt.subplots(figsize=(3.45, 2.05))
cols = ["#86b6ef", "#2a78d6", "#104281"]
labs = {0.7: "70% of flips are misses", 0.85: "85%", 1.0: "100% (one-directional)"}
for r, col in zip(shares, cols):
    xs = [d for d in deltas if PW[str(r)].get(str(d))]
    ys = [PW[str(r)][str(d)] for d in xs]
    ax.plot(xs, ys, color=col, linewidth=1.6, marker="o", markersize=3.5, markeredgecolor="white", markeredgewidth=0.5)
    ax.text(xs[-1] + 0.12, ys[-1], labs[r], fontsize=5.8, color=INK, va="center")
ax.set_yscale("log")
ax.set_xlabel("Rise in false-negative rate to detect (percentage points)", color=INK, fontsize=6.8)
ax.set_ylabel("Unsafe inputs needed (80% power)", color=INK, fontsize=6.8)
ax.set_xlim(0.3, 8.6)
from matplotlib.ticker import FixedLocator, NullLocator
ax.yaxis.set_major_locator(FixedLocator([100, 200, 500, 1000, 2000, 5000]))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_yticklabels(["100", "200", "500", "1,000", "2,000", "5,000"])
ax.set_ylim(100, 6000)
ax.grid(color=GRID, linewidth=0.5, which="major"); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for nu, dfnr, lab in ((200, 4.0, "Llama XSTest"), (750, 4.13, "Llama BeaverTails"), (1733, 0.46, "Qwen BeaverTails")):
    ax.scatter([dfnr], [nu], marker="D", s=14, color=FP4C, edgecolor="white", linewidth=0.5, zorder=4)
    pos = {"Llama XSTest": (2.55, 125), "Llama BeaverTails": (4.6, 900), "Qwen BeaverTails": (1.7, 3300)}[lab]
    ax.annotate(lab, (dfnr, nu), xytext=pos, fontsize=5.8, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.5, shrinkA=1, shrinkB=3))
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig_power.pdf"); fig.savefig(OUT / "fig_power.png", dpi=220)
