"""Compute every statistic reported in the manuscript from the raw prediction records.

Usage: python3 -I paper_stats.py <raw_root> <out_dir>
<raw_root> holds qwen_main/, llama/, quant_rep/, gpu_rep/, llama_rep/ (deposit layout).
Writes stats.json and LaTeX table fragments (tab_*.tex) to <out_dir>.
"""
import json, re, sys
from itertools import combinations
from math import comb
from pathlib import Path
from scipy.stats import beta

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
Q = ROOT / "qwen_main/five_results"
L = ROOT / "llama/llama_guard_replication_results"
QDS = ["qwen_thinking", "toxicchat_prompt", "xstest_prompt", "beavertails_response"]
LDS = ["xstest_prompt", "beavertails_response"]
QARMS = ["bf16", "nf4", "fp4", "fp8", "int8"]
LARMS = ["bf16", "nf4", "fp4"]
DSNAME = {"qwen_thinking": "Qwen3GuardTest-Think", "toxicchat_prompt": "ToxicChat",
          "xstest_prompt": "XSTest", "beavertails_response": "BeaverTails"}
ARMNAME = {"bf16": "BF16", "nf4": "NF4", "fp4": "FP4", "fp8": "FP8", "int8": "INT8"}


def parse_qwen(o):
    m = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", o or "")
    return "invalid" if not m else ("safe" if m.group(1) == "Safe" else "unsafe")


def parse_llama(o):
    lines = [l.strip().lower() for l in (o or "").splitlines() if l.strip()]
    return lines[0] if lines and lines[0] in ("safe", "unsafe") else "invalid"


def load(path, parser):
    recs = json.loads(path.read_text())["records"]
    out = {}
    for r in recs:
        p = parser(r["output"])
        assert p == r["binary"], (path, r["id"])
        out[r["id"]] = (r["gold"], p)
    return out


def exact_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def cp_ci(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def metrics(x):
    U = [i for i in x if x[i][0] == "unsafe"]; S = [i for i in x if x[i][0] == "safe"]
    fn = sum(x[i][1] == "safe" for i in U); fp = sum(x[i][1] == "unsafe" for i in S)
    inv = sum(x[i][1] == "invalid" for i in x)
    return dict(n=len(x), unsafe=len(U), safe=len(S), FN=fn, FP=fp, invalid=inv,
                acc=(len(x) - fn - fp) / len(x), FNR=fn / len(U), FPR=fp / len(S))


def compare(a, b):
    flips = [i for i in a if a[i][1] != b[i][1]]
    nm = sum(a[i][0] == "unsafe" and b[i][1] == "safe" for i in flips)    # newly missed unsafe
    rc = sum(a[i][0] == "unsafe" and b[i][1] == "unsafe" for i in flips)  # recovered unsafe
    nfa = sum(a[i][0] == "safe" and b[i][1] == "unsafe" for i in flips)   # new false alarm
    rfa = sum(a[i][0] == "safe" and b[i][1] == "safe" for i in flips)     # removed false alarm
    imp, reg = rc + rfa, nm + nfa
    n = len(a); nu = sum(v[0] == "unsafe" for v in a.values()); ns = n - nu
    lo, hi = cp_ci(len(flips), n)
    return dict(flips=len(flips), flip_rate=len(flips) / n, flip_ci=[lo, hi],
                newly_missed=nm, recovered=rc, new_fa=nfa, removed_fa=rfa,
                improve=imp, regress=reg, p_acc=exact_p(imp, reg), p_unsafe=exact_p(nm, rc),
                p_safe=exact_p(nfa, rfa), dFNR=(nm - rc) / nu, dFPR=(nfa - rfa) / ns,
                dAcc=(imp - reg) / n, ids=sorted(flips))


S = {"qwen": {}, "llama": {}}
P = {"qwen": {}, "llama": {}}
for fam, base, dsets, arms, parser in (("qwen", Q, QDS, QARMS, parse_qwen), ("llama", L, LDS, LARMS, parse_llama)):
    for ds in dsets:
        P[fam][ds] = {a: load(base / ds / f"{a}.json", parser) for a in arms}
        S[fam][ds] = {"metrics": {a: metrics(P[fam][ds][a]) for a in arms},
                      "pairs": {f"{a}->{b}": compare(P[fam][ds][a], P[fam][ds][b]) for a, b in combinations(arms, 2)}}

# Declared multiplicity families
fams = {
    "llama_nf4fp4_unsafe": [("llama", d, "nf4->fp4", "p_unsafe") for d in LDS],
    "llama_nf4fp4_acc": [("llama", d, "nf4->fp4", "p_acc") for d in LDS],
    "qwen_nf4fp4_unsafe": [("qwen", d, "nf4->fp4", "p_unsafe") for d in QDS],
    "qwen_nf4fp4_acc": [("qwen", d, "nf4->fp4", "p_acc") for d in QDS],
    "qwen_fp8int8_unsafe": [("qwen", d, "fp8->int8", "p_unsafe") for d in QDS],
    "qwen_fp8int8_acc": [("qwen", d, "fp8->int8", "p_acc") for d in QDS],
    "qwen_equalbit_acc_8tests": [("qwen", d, p, "p_acc") for p in ("nf4->fp4", "fp8->int8") for d in QDS],
    "qwen_all_pairs_unsafe_40": [("qwen", d, p, "p_unsafe") for d in QDS for p in S["qwen"][QDS[0]]["pairs"]],
    "llama_all_pairs_unsafe_6": [("llama", d, p, "p_unsafe") for d in LDS for p in S["llama"][LDS[0]]["pairs"]],
}
H = {}
for name, items in fams.items():
    ps = [S[f][d]["pairs"][p][k] for f, d, p, k in items]
    H[name] = [dict(fam=f, ds=d, pair=p, test=k, p=pv, holm=hv) for (f, d, p, k), pv, hv in zip(items, ps, holm(ps))]

# Cross-family paired comparison on identical inputs (NF4 -> FP4 change in unsafe detection)
X = {}
for ds in LDS:
    lq, ll = P["qwen"][ds], P["llama"][ds]
    ids = sorted(ll["nf4"])  # Llama selections are subsets of the Qwen frozen selections
    assert all(i in lq["nf4"] and lq["nf4"][i][0] == ll["nf4"][i][0] for i in ids)
    def delta(pred, i):  # +1 = FP4 newly detects, -1 = FP4 newly misses, 0 = unchanged
        return (pred["fp4"][i][1] == "unsafe") - (pred["nf4"][i][1] == "unsafe")
    out = {}
    for scope in ("unsafe", "all"):
        sel = [i for i in ids if scope == "all" or ll["nf4"][i][0] == "unsafe"]
        dq = {i: delta(P["qwen"][ds], i) for i in sel}
        dl = {i: delta(P["llama"][ds], i) for i in sel}
        if scope == "unsafe":
            llama_worse = sum(dl[i] < dq[i] for i in sel)
            llama_better = sum(dl[i] > dq[i] for i in sel)
        else:  # correctness change
            corr = lambda pred, i, arm: pred[arm][i][1] == pred[arm][i][0]
            cq = {i: corr(P["qwen"][ds], i, "fp4") - corr(P["qwen"][ds], i, "nf4") for i in sel}
            cl = {i: corr(P["llama"][ds], i, "fp4") - corr(P["llama"][ds], i, "nf4") for i in sel}
            llama_worse = sum(cl[i] < cq[i] for i in sel)
            llama_better = sum(cl[i] > cq[i] for i in sel)
        out[scope] = dict(n=len(sel), llama_worse=llama_worse, llama_better=llama_better,
                          p=exact_p(llama_worse, llama_better))
    # Qwen metrics restricted to the shared items (for the BeaverTails subset)
    sub = {i: lq["nf4"][i] for i in ids}; subf = {i: lq["fp4"][i] for i in ids}
    out["qwen_on_shared"] = {"nf4": metrics(sub), "fp4": metrics(subf), "pair": compare(sub, subf)}
    out["qwen_on_shared"]["pair"].pop("ids")
    X[ds] = out

# Repeatability (counts verified separately by verify_repeat.py and verify_llama_reload.py)
REP = {"qwen_bf16_same_model": {"inputs": 1024, "predictions": 2048, "within_changes": 0, "vs_original": 0},
       "qwen_nf4_fp4_reload": {"inputs": 1024, "predictions": 4096, "within_changes": 0, "vs_original": 0,
                               "persistent_nf4_fp4_flips": {"qwen_thinking": 3, "toxicchat_prompt": 2, "xstest_prompt": 5, "beavertails_response": 6}},
       "llama_nf4_fp4_reload": {"inputs": 1950, "predictions": 7800, "within_changes": 0, "vs_original": 0,
                                "persistent_nf4_fp4_flips": {"xstest_prompt": 12, "beavertails_response": 42}}}

for fam in S:
    for ds in S[fam]:
        for p in S[fam][ds]["pairs"].values():
            p.pop("ids")
json.dump({"single": S, "holm": H, "cross_family": X, "repeatability": REP}, open(OUT / "stats.json", "w"), indent=1)


# ---------------- LaTeX tables ----------------
def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def sg(x):
    v = f"{100 * x:+.2f}"
    if v in ("+0.00", "-0.00"):
        return "0.00"
    return v.replace("-", "$-$")


def pfmt(p):
    if p >= 0.001:
        return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
    e = f"{p:.1e}"; m, x = e.split("e")
    return f"${m}\\times10^{{{int(x)}}}$"


# Table: per-arm operating points
rows = []
for fam, dsets, arms, fname in (("qwen", QDS, QARMS, "Qwen3Guard-Gen-4B"), ("llama", LDS, LARMS, "Llama Guard 3 8B")):
    rows.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{fname}}}}}\\\\")
    for ds in dsets:
        m0 = S[fam][ds]["metrics"][arms[0]]
        first = True
        for a in arms:
            m = S[fam][ds]["metrics"][a]
            lab = f"{DSNAME[ds]} ({m0['unsafe']}/{m0['safe']})" if first else ""
            rows.append(f"{lab} & {ARMNAME[a]} & {m['FN']} & {m['FP']} & {pct(m['FNR'])} & {pct(m['FPR'])} & {pct(m['acc'])} & {m['invalid']}\\\\")
            first = False
        rows.append("\\addlinespace[2pt]")
(OUT / "tab_operating_points.tex").write_text("\n".join(rows[:-1]) + "\n")

# Table: equal-bit-width contrasts (primary)
rows = []
for fam, dsets, pair, famlab, hu, ha in (
        ("qwen", QDS, "nf4->fp4", "Qwen3Guard, NF4$\\to$FP4", "qwen_nf4fp4_unsafe", "qwen_nf4fp4_acc"),
        ("qwen", QDS, "fp8->int8", "Qwen3Guard, FP8$\\to$INT8", "qwen_fp8int8_unsafe", "qwen_fp8int8_acc"),
        ("llama", LDS, "nf4->fp4", "Llama Guard 3, NF4$\\to$FP4", "llama_nf4fp4_unsafe", "llama_nf4fp4_acc")):
    rows.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{famlab}}}}}\\\\")
    hu_d = {h["ds"]: h["holm"] for h in H[hu]}; ha_d = {h["ds"]: h["holm"] for h in H[ha]}
    for ds in dsets:
        c = S[fam][ds]["pairs"][pair]; n = S[fam][ds]["metrics"]["bf16"]["n"]
        rows.append(f"{DSNAME[ds]} & {c['flips']}/{n} ({pct(c['flip_rate'])}) & {c['newly_missed']}/{c['recovered']} & "
                    f"{c['new_fa']}/{c['removed_fa']} & {sg(c['dFNR'])} & {sg(c['dFPR'])} & {sg(c['dAcc'])} & "
                    f"{pfmt(hu_d[ds])} & {pfmt(ha_d[ds])}\\\\")
    rows.append("\\addlinespace[2pt]")
(OUT / "tab_equal_bit.tex").write_text("\n".join(rows[:-1]) + "\n")

# Table: BF16 reference contrasts (exploratory), unsafe-class only, Holm over all pairs within family
rows = []
hq = {(h["ds"], h["pair"]): h["holm"] for h in H["qwen_all_pairs_unsafe_40"]}
hl = {(h["ds"], h["pair"]): h["holm"] for h in H["llama_all_pairs_unsafe_6"]}
for fam, dsets, arms, hh, lab in (("qwen", QDS, QARMS, hq, "Qwen3Guard"), ("llama", LDS, LARMS, hl, "Llama Guard 3")):
    for ds in dsets:
        cells = []
        for a in arms[1:]:
            c = S[fam][ds]["pairs"][f"bf16->{a}"]
            cells.append(f"{c['flips']} ({c['newly_missed']}/{c['recovered']})")
        cells += ["--"] * (4 - len(cells))
        rows.append(f"{lab} & {DSNAME[ds]} & " + " & ".join(cells) + "\\\\")
(OUT / "tab_bf16_ref.tex").write_text("\n".join(rows) + "\n")

print(json.dumps({"cross_family": X, "holm_llama_all6": H["llama_all_pairs_unsafe_6"],
                  "qwen_all40_min": sorted(H["qwen_all_pairs_unsafe_40"], key=lambda h: h["p"])[:5]}, indent=1, default=str)[:6000])

# ---------------- Case study: hazard codes of inputs newly missed by Llama Guard 3 under FP4 ----------------
import collections
HZ = {"S1": "Violent crimes", "S2": "Non-violent crimes", "S6": "Specialized advice", "S7": "Privacy",
      "S10": "Hate", "S12": "Sexual content"}
cells = {}
for ds in LDS:
    a = {r["id"]: r for r in json.loads((L / ds / "nf4.json").read_text())["records"]}
    b = {r["id"]: r for r in json.loads((L / ds / "fp4.json").read_text())["records"]}
    det = [i for i in a if a[i]["gold"] == "unsafe" and parse_llama(a[i]["output"]) == "unsafe"]
    mis = [i for i in det if parse_llama(b[i]["output"]) == "safe"]
    codes = lambda ids: collections.Counter(c for i in ids for c in set(re.findall(r"S\d+", a[i]["output"])))
    cells[ds] = (codes(det), codes(mis), len(det), len(mis))
order = sorted(HZ, key=lambda c: -(cells["beavertails_response"][1][c] + cells["xstest_prompt"][1][c]))
rows = []
for c in order:
    parts = []
    for ds in LDS:
        d, m = cells[ds][0][c], cells[ds][1][c]
        parts.append(f"{m}/{d}" + (f" ({100 * m / d:.0f}\\%)" if d else ""))
    rows.append(f"{c} {HZ[c]} & " + " & ".join(parts) + "\\\\")
rows.append("\\midrule")
rows.append("All codes & " + " & ".join(f"{cells[ds][3]}/{cells[ds][2]} ({100 * cells[ds][3] / cells[ds][2]:.0f}\\%)" for ds in LDS) + "\\\\")
(OUT / "tab_hazard.tex").write_text("\n".join(rows) + "\n")
json.dump({ds: {"detected": dict(cells[ds][0]), "missed": dict(cells[ds][1])} for ds in LDS}, open(OUT / "hazard.json", "w"), indent=1)
