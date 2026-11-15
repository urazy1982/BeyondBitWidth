"""Extended analyses for the TIFS long version. All numbers come from the raw prediction records.

Usage: python3 -I extra_stats.py <raw_root> <out_dir>
Writes extra.json and LaTeX table fragments (tabx_*.tex) to <out_dir>.
"""
import json, re, sys
from collections import Counter, defaultdict
from itertools import combinations
from math import comb
from pathlib import Path
from scipy.stats import fisher_exact, mannwhitneyu, beta

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
Q = ROOT / "qwen_main/five_results"
L = ROOT / "llama/llama_guard_replication_results"
QDS = ["qwen_thinking", "toxicchat_prompt", "xstest_prompt", "beavertails_response"]
LDS = ["xstest_prompt", "beavertails_response"]
QARMS = ["bf16", "nf4", "fp4", "fp8", "int8"]
LARMS = ["bf16", "nf4", "fp4"]
DSN = {"qwen_thinking": "Think", "toxicchat_prompt": "ToxicChat", "xstest_prompt": "XSTest",
       "beavertails_response": "BeaverTails"}
AN = {"bf16": "BF16", "nf4": "NF4", "fp4": "FP4", "fp8": "FP8", "int8": "INT8"}


def exact_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def qwen_tier(o):
    return re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", o).group(1)


def qwen_cats(o):
    m = re.search(r"Categories:\s*(.*)", o)
    c = m.group(1).strip() if m else ""
    return frozenset() if c in ("", "None") else frozenset(x.strip() for x in c.split(","))


def llama_codes(o):
    lines = [l.strip() for l in o.splitlines() if l.strip()]
    return frozenset(x.strip() for x in lines[1].split(",")) if len(lines) > 1 else frozenset()


def load(path):
    return {r["id"]: r for r in json.loads(path.read_text())["records"]}


q = {ds: {a: load(Q / ds / f"{a}.json") for a in QARMS} for ds in QDS}
l = {ds: {a: load(L / ds / f"{a}.json") for a in LARMS} for ds in LDS}
qsel = {ds: {r["id"]: r for r in json.loads((Q / ds / "selection.json").read_text())} for ds in QDS}
lsel = {ds: {r["id"]: r for r in json.loads((L / ds / "llama_selection.json").read_text())} for ds in LDS}
for ds in QDS:
    for a in QARMS:
        assert set(q[ds][a]) == set(qsel[ds])
for ds in LDS:
    for a in LARMS:
        assert set(l[ds][a]) == set(lsel[ds])

X = {}

# ---------- A2: pairwise disagreement matrices ----------
X["pairwise"] = {}
for fam, data, arms, dss in [("qwen", q, QARMS, QDS), ("llama", l, LARMS, LDS)]:
    for ds in dss:
        M = {}
        for a, b in combinations(arms, 2):
            ids = data[ds][a]
            k = sum(data[ds][a][i]["binary"] != data[ds][b][i]["binary"] for i in ids)
            M[f"{a}-{b}"] = [k, len(ids)]
        X["pairwise"][f"{fam}/{ds}"] = M

# ---------- A3: Controversial tier ----------
X["controversial"] = {}
rows = []
fam_p = []
for ds in QDS:
    d = q[ds]
    cnt = {a: sum(qwen_tier(d[a][i]["output"]) == "Controversial" for i in d[a]) for a in QARMS}
    flips = [i for i in d["nf4"] if d["nf4"][i]["binary"] != d["fp4"][i]["binary"]]
    inv = sum("Controversial" in (qwen_tier(d["nf4"][i]["output"]), qwen_tier(d["fp4"][i]["output"])) for i in flips)
    # tier changes (any change of the 3-way label)
    tier_ch = sum(qwen_tier(d["nf4"][i]["output"]) != qwen_tier(d["fp4"][i]["output"]) for i in d["nf4"])
    # strict mapping: Controversial -> safe
    def strict(a, i):
        return "unsafe" if qwen_tier(d[a][i]["output"]) == "Unsafe" else "safe"
    U = [i for i in d["nf4"] if d["nf4"][i]["gold"] == "unsafe"]
    S = [i for i in d["nf4"] if d["nf4"][i]["gold"] == "safe"]
    nm = sum(strict("nf4", i) == "unsafe" and strict("fp4", i) == "safe" for i in U)
    rc = sum(strict("nf4", i) == "safe" and strict("fp4", i) == "unsafe" for i in U)
    sflips = sum(strict("nf4", i) != strict("fp4", i) for i in d["nf4"])
    fn_nf4 = sum(strict("nf4", i) == "safe" for i in U); fn_fp4 = sum(strict("fp4", i) == "safe" for i in U)
    fp_nf4 = sum(strict("nf4", i) == "unsafe" for i in S)
    p = exact_p(nm, rc); fam_p.append(p)
    X["controversial"][ds] = dict(count=cnt, flips=len(flips), flips_involving=inv, tier_changes=tier_ch,
                                  strict=dict(flips=sflips, nm=nm, rc=rc, p=p, FNR_nf4=fn_nf4 / len(U),
                                              FNR_fp4=fn_fp4 / len(U), FPR_nf4=fp_nf4 / len(S)))
for ds, pa in zip(QDS, holm(fam_p)):
    X["controversial"][ds]["strict"]["p_holm"] = pa

# ---------- A4: breakdowns ----------
def nf4fp4_by(data, ds, key):
    d = data[ds]; g = defaultdict(lambda: [0, 0, 0, 0])  # n, flips, nm, rc
    for i in d["nf4"]:
        r = d["nf4"][i]; k = key(r); g[k][0] += 1
        if r["binary"] != d["fp4"][i]["binary"]:
            g[k][1] += 1
            if r["gold"] == "unsafe":
                if r["binary"] == "unsafe": g[k][2] += 1
                else: g[k][3] += 1
    return {k: v for k, v in sorted(g.items())}

X["by_type"] = {
    "qwen/xstest": nf4fp4_by(q, "xstest_prompt", lambda r: r["category"]),
    "llama/xstest": nf4fp4_by(l, "xstest_prompt", lambda r: r["category"]),
    "qwen/think_gen": nf4fp4_by(q, "qwen_thinking", lambda r: r["subgroup"]),
    "qwen/think_cat": nf4fp4_by(q, "qwen_thinking", lambda r: r["category"] or "(safe)"),
    "qwen/toxic_jb": nf4fp4_by(q, "toxicchat_prompt", lambda r: f'{r["gold"]}/jb={r["subgroup"]}'),
}

# ---------- A5: input length ----------
def toklen(sel, i):
    r = sel[i]
    if "tokens" in r and r["tokens"]:
        return r["tokens"]
    return sum(len(m["content"]) for m in r["messages"])

X["length"] = {}
for fam, data, sel, dss in [("qwen", q, qsel, QDS), ("llama", l, lsel, LDS)]:
    for ds in dss:
        d = data[ds]; unit = "tokens" if "tokens" in next(iter(sel[ds].values())) else "chars"
        fl = [toklen(sel[ds], i) for i in d["nf4"] if d["nf4"][i]["binary"] != d["fp4"][i]["binary"]]
        st = [toklen(sel[ds], i) for i in d["nf4"] if d["nf4"][i]["binary"] == d["fp4"][i]["binary"]]
        u = mannwhitneyu(fl, st, alternative="two-sided")
        fl.sort(); st.sort()
        med = lambda v: (v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2)
        X["length"][f"{fam}/{ds}"] = dict(unit=unit, median_flip=med(fl), median_stable=med(st),
                                         n_flip=len(fl), p=float(u.pvalue))
lp = [X["length"][k]["p"] for k in X["length"]]
for k, pa in zip(X["length"], holm(lp)):
    X["length"][k]["p_holm"] = pa

# ---------- A6: are NF4/FP4 flips concentrated on inputs that are unstable elsewhere? ----------
X["instability"] = {}
for ds in QDS:
    d = q[ds]
    a = b = c = e = 0  # a: flip & unstable, b: flip & stable, c: noflip & unstable, e: noflip & stable
    for i in d["nf4"]:
        flip = d["nf4"][i]["binary"] != d["fp4"][i]["binary"]
        unst = len({d[x][i]["binary"] for x in ("bf16", "fp8", "int8")}) > 1
        if flip and unst: a += 1
        elif flip: b += 1
        elif unst: c += 1
        else: e += 1
    orr, p = fisher_exact([[a, b], [c, e]])
    X["instability"][ds] = dict(flip_unstable=a, flip_stable=b, noflip_unstable=c, noflip_stable=e,
                                rate_unstable=a / (a + c) if a + c else 0, rate_stable=b / (b + e),
                                OR=float(orr), p=float(p))
ip = [X["instability"][k]["p"] for k in QDS]
for k, pa in zip(QDS, holm(ip)):
    X["instability"][k]["p_holm"] = pa

# cross-family: Llama NF4/FP4 flips vs Qwen instability (any disagreement among 5 arms) on shared inputs
X["cross_overlap"] = {}
for ds in LDS:
    a = b = c = e = 0; both = 0
    for i in l[ds]["nf4"]:
        lf = l[ds]["nf4"][i]["binary"] != l[ds]["fp4"][i]["binary"]
        qu = len({q[ds][x][i]["binary"] for x in QARMS}) > 1
        qf = q[ds]["nf4"][i]["binary"] != q[ds]["fp4"][i]["binary"]
        both += lf and qf
        if lf and qu: a += 1
        elif lf: b += 1
        elif qu: c += 1
        else: e += 1
    orr, p = fisher_exact([[a, b], [c, e]])
    X["cross_overlap"][ds] = dict(llflip_qunstable=a, llflip_qstable=b, noflip_qunstable=c, noflip_qstable=e,
                                  both_nf4fp4=both, OR=float(orr), p=float(p),
                                  rate_qunstable=a / (a + c) if a + c else 0, rate_qstable=b / (b + e))

# Llama hard inputs: did Llama BF16-vs-NF4 flips coincide with NF4-vs-FP4 flips?
X["llama_overlap"] = {}
for ds in LDS:
    d = l[ds]
    f1 = {i for i in d["nf4"] if d["bf16"][i]["binary"] != d["nf4"][i]["binary"]}
    f2 = {i for i in d["nf4"] if d["nf4"][i]["binary"] != d["fp4"][i]["binary"]}
    f3 = {i for i in d["nf4"] if d["bf16"][i]["binary"] != d["fp4"][i]["binary"]}
    X["llama_overlap"][ds] = dict(bf16_nf4=len(f1), nf4_fp4=len(f2), bf16_fp4=len(f3), inter_12=len(f1 & f2))

# ---------- A7: secondary output (category) changes when the verdict agrees ----------
X["category_change"] = {}
for fam, data, cats, dss in [("qwen", q, qwen_cats, QDS), ("llama", l, llama_codes, LDS)]:
    for ds in dss:
        d = data[ds]
        both = [i for i in d["nf4"] if d["nf4"][i]["binary"] == "unsafe" and d["fp4"][i]["binary"] == "unsafe"]
        ch = sum(cats(d["nf4"][i]["output"]) != cats(d["fp4"][i]["output"]) for i in both)
        # BF16 vs NF4 for reference
        both_b = [i for i in d["nf4"] if d["bf16"][i]["binary"] == "unsafe" and d["nf4"][i]["binary"] == "unsafe"]
        ch_b = sum(cats(d["bf16"][i]["output"]) != cats(d["nf4"][i]["output"]) for i in both_b)
        X["category_change"][f"{fam}/{ds}"] = dict(both_unsafe=len(both), changed=ch, rate=ch / len(both),
                                                   bf16_nf4_both=len(both_b), bf16_nf4_changed=ch_b)

# ---------- A9: Clopper-Pearson interval for the share of discordant unsafe pairs that are newly missed ----------
def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)

X["direction_share"] = {}
for fam, data, dss in [("qwen", q, QDS), ("llama", l, LDS)]:
    for ds in dss:
        d = data[ds]
        U = [i for i in d["nf4"] if d["nf4"][i]["gold"] == "unsafe"]
        nm = sum(d["nf4"][i]["binary"] == "unsafe" and d["fp4"][i]["binary"] == "safe" for i in U)
        rc = sum(d["nf4"][i]["binary"] == "safe" and d["fp4"][i]["binary"] == "unsafe" for i in U)
        X["direction_share"][f"{fam}/{ds}"] = dict(nm=nm, rc=rc, share=nm / (nm + rc) if nm + rc else None,
                                                   ci=cp(nm, nm + rc) if nm + rc else None, NU=len(U))

(OUT / "extra.json").write_text(json.dumps(X, indent=1, default=str))

# ---------- LaTeX tables ----------
def pct(k, n):
    return f"{100 * k / n:.1f}"

# XSTest by prompt type (unsafe contrast types and safe types), both families
TYPES = ["homonyms", "figurative_language", "safe_targets", "safe_contexts", "definitions",
         "real_group_nons_discr", "nons_group_real_discr", "historical_events", "privacy_public", "privacy_fictional"]
TNAME = {"homonyms": "Homonyms", "figurative_language": "Figurative language", "safe_targets": "Safe targets",
         "safe_contexts": "Safe contexts", "definitions": "Definitions", "real_group_nons_discr": "Real group / nons. discr.",
         "nons_group_real_discr": "Nons. group / real discr.", "historical_events": "Historical events",
         "privacy_public": "Privacy (public figures)", "privacy_fictional": "Privacy (fictional)"}
CMAP = {"homonyms": "contrast_homonyms", "figurative_language": "contrast_figurative_language",
        "safe_targets": "contrast_safe_targets", "safe_contexts": "contrast_safe_contexts",
        "definitions": "contrast_definitions", "real_group_nons_discr": "contrast_discr",
        "historical_events": "contrast_historical_events", "privacy_public": "contrast_privacy"}
lines = []
for t in TYPES:
    qs, ls_ = X["by_type"]["qwen/xstest"][t], X["by_type"]["llama/xstest"][t]
    c = CMAP.get(t)
    if c:
        qc, lc = X["by_type"]["qwen/xstest"][c], X["by_type"]["llama/xstest"][c]
        cs = f"{qc[1]} & {lc[1]} ({lc[2]}/{lc[3]})"
    else:
        cs = "-- & --"
    lines.append(f"{TNAME[t]} & {qs[1]} & {ls_[1]} & {cs} \\\\")
tq = [X["by_type"]["qwen/xstest"][k] for k in X["by_type"]["qwen/xstest"]]
tl = [X["by_type"]["llama/xstest"][k] for k in X["by_type"]["llama/xstest"]]
sq = sum(v[1] for k, v in X["by_type"]["qwen/xstest"].items() if not k.startswith("contrast"))
sl = sum(v[1] for k, v in X["by_type"]["llama/xstest"].items() if not k.startswith("contrast"))
cq = sum(v[1] for k, v in X["by_type"]["qwen/xstest"].items() if k.startswith("contrast"))
cl = sum(v[1] for k, v in X["by_type"]["llama/xstest"].items() if k.startswith("contrast"))
lines.append("\\midrule")
lines.append(f"Total & {sq} & {sl} & {cq} & {cl} \\\\")
(OUT / "tabx_xstest_types.tex").write_text("\n".join(lines) + "\n")


# Mechanism-oriented summary table: instability, length, category change
lines = []
for ds in QDS:
    ins = X["instability"][ds]; ln = X["length"][f"qwen/{ds}"]; cc = X["category_change"][f"qwen/{ds}"]
    lines.append(f"Qwen & {DSN[ds]} & {ins['flip_unstable']}/{ins['flip_unstable'] + ins['noflip_unstable']} & "
                 f"{ins['flip_stable']}/{ins['flip_stable'] + ins['noflip_stable']} & "
                 f"{ln['median_flip']:.0f} / {ln['median_stable']:.0f} & {cc['changed']}/{cc['both_unsafe']} \\\\")
for ds in LDS:
    co = X["cross_overlap"][ds]; ln = X["length"][f"llama/{ds}"]; cc = X["category_change"][f"llama/{ds}"]
    lines.append(f"Llama & {DSN[ds]} & {co['llflip_qunstable']}/{co['llflip_qunstable'] + co['noflip_qunstable']}$^\\dagger$ & "
                 f"{co['llflip_qstable']}/{co['llflip_qstable'] + co['noflip_qstable']}$^\\dagger$ & "
                 f"{ln['median_flip']:.0f} / {ln['median_stable']:.0f} & {cc['changed']}/{cc['both_unsafe']} \\\\")
(OUT / "tabx_mechanism.tex").write_text("\n".join(lines) + "\n")


print(json.dumps({k: X[k] for k in ["controversial", "instability", "cross_overlap", "length", "category_change",
                                    "direction_share", "llama_overlap"]}, indent=1, default=str))
print(json.dumps(X["by_type"], default=str))
print(json.dumps(X["pairwise"]))

# ---------- A3b: three-tier transitions and ordinal shift (mapping-free) ----------
RANK = {"Safe": 0, "Controversial": 1, "Unsafe": 2}
X["tier"] = {}
for ds in QDS:
    d = q[ds]; X["tier"][ds] = {}
    for a, b in combinations(QARMS, 2):
        T = Counter(); down = up = 0; downU = upU = 0
        for i in d[a]:
            ta, tb = qwen_tier(d[a][i]["output"]), qwen_tier(d[b][i]["output"])
            T[(d[a][i]["gold"], ta, tb)] += 1
            if RANK[tb] < RANK[ta]:
                down += 1; downU += d[a][i]["gold"] == "unsafe"
            elif RANK[tb] > RANK[ta]:
                up += 1; upU += d[a][i]["gold"] == "unsafe"
        X["tier"][ds][f"{a}-{b}"] = dict(down=down, up=up, p=exact_p(down, up), downU=downU, upU=upU,
                                         pU=exact_p(downU, upU),
                                         T={"|".join(k): v for k, v in T.items() if k[1] != k[2]})
# strict-mapping nm/rc for every Qwen contrast
X["strict_all"] = {}
for ds in QDS:
    d = q[ds]; U = [i for i in d["nf4"] if d["nf4"][i]["gold"] == "unsafe"]
    st = lambda a, i: "unsafe" if qwen_tier(d[a][i]["output"]) == "Unsafe" else "safe"
    for a, b in combinations(QARMS, 2):
        nm = sum(st(a, i) == "unsafe" and st(b, i) == "safe" for i in U)
        rc = sum(st(a, i) == "safe" and st(b, i) == "unsafe" for i in U)
        X["strict_all"][f"{ds}/{a}-{b}"] = [nm, rc, exact_p(nm, rc)]
ks = list(X["strict_all"]); hp = holm([X["strict_all"][k][2] for k in ks])
for k, h in zip(ks, hp):
    X["strict_all"][k].append(h)
# generator gold counts
X["gen_gold"] = dict(Counter((r["subgroup"], r["gold"]) for r in q["qwen_thinking"]["nf4"].values()).items())
X["gen_gold"] = {"|".join(k): v for k, v in X["gen_gold"].items()}
orig = [i for i, r in q["qwen_thinking"]["nf4"].items() if r["subgroup"] in ("Qwen/Qwen3-8B", "zai-org/GLM-Z1-9B-0414")]
oth = [i for i in q["qwen_thinking"]["nf4"] if i not in set(orig)]
fl = lambda i: q["qwen_thinking"]["nf4"][i]["binary"] != q["qwen_thinking"]["fp4"][i]["binary"]
a1 = sum(map(fl, orig)); a2 = sum(map(fl, oth))
X["gen_test"] = dict(orig=[a1, len(orig)], other=[a2, len(oth)], p=float(fisher_exact([[a1, len(orig) - a1], [a2, len(oth) - a2]])[1]))
(OUT / "extra.json").write_text(json.dumps(X, indent=1, default=str))
print("TIER"); [print(ds, k, {kk: v[kk] for kk in ("down", "up", "p", "downU", "upU", "pU")}) for ds in QDS for k, v in X["tier"][ds].items()]
print("NF4-FP4 transitions"); [print(ds, X["tier"][ds]["nf4-fp4"]["T"]) for ds in QDS]
print("STRICT"); [print(k, v) for k, v in X["strict_all"].items()]
print(X["gen_gold"], X["gen_test"])

# ---------- Table: three-tier analysis of the NF4 -> FP4 switch (Qwen3Guard) ----------
pa_all = holm([X["tier"][ds]["nf4-fp4"]["p"] for ds in QDS])
for ds, h in zip(QDS, pa_all):
    X["tier"][ds]["nf4-fp4"]["p_holm"] = h
def pf(p):
    if p >= 0.995: return "1.00"
    if p >= 0.1: return f"{p:.2f}"
    if p >= 0.01: return f"{p:.3f}"
    if p >= 0.001: return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"
lines = []
for ds in QDS:
    t = X["tier"][ds]["nf4-fp4"]; c = X["controversial"][ds]; s = c["strict"]
    lines.append(f"{DSN[ds]} & {c['count']['nf4']}/{c['count']['fp4']} & {t['down']}/{t['up']} & {pf(t['p_holm'])} & "
                 f"{t['downU']}/{t['upU']} & {s['nm']}/{s['rc']} & {pf(s['p_holm'])} \\\\")
(OUT / "tabx_tier.tex").write_text("\n".join(lines) + "\n")
(OUT / "extra.json").write_text(json.dumps(X, indent=1, default=str))
print("TIERTAB"); print(open(OUT / "tabx_tier.tex").read())

# ---------- cross-family sign test under the strict Qwen mapping ----------
X["cross_strict"] = {}
for ds in LDS:
    worseL = worseQ = 0
    for i, r in l[ds]["nf4"].items():
        if r["gold"] != "unsafe":
            continue
        dl = (l[ds]["fp4"][i]["binary"] == "unsafe") - (r["binary"] == "unsafe")
        sq = lambda a: qwen_tier(q[ds][a][i]["output"]) == "Unsafe"
        dq = sq("fp4") - sq("nf4")
        if dl < dq: worseL += 1
        elif dq < dl: worseQ += 1
    X["cross_strict"][ds] = dict(worse_llama=worseL, worse_qwen=worseQ, p=exact_p(worseL, worseQ))
(OUT / "extra.json").write_text(json.dumps(X, indent=1, default=str))
print("CROSS_STRICT", X["cross_strict"])
