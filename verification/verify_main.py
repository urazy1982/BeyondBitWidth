"""Independent recomputation of Qwen five-arm and Llama three-arm results from raw records."""
import hashlib, json, re, sys
from itertools import combinations
from math import comb
from pathlib import Path

ROOT = Path(sys.argv[1])
Q = ROOT / "qwen_main/five_results"
L = ROOT / "llama/llama_guard_replication_results"
QREP = ROOT / "quant_rep/guard_quantized_repeatability_results"
GREP = ROOT / "gpu_rep/guard_repeatability_results"
QDS = ["qwen_thinking", "toxicchat_prompt", "xstest_prompt", "beavertails_response"]
QARMS = ["bf16", "nf4", "fp4", "fp8", "int8"]
LDS = ["xstest_prompt", "beavertails_response"]
LARMS = ["bf16", "nf4", "fp4"]
issues = []
out = {"qwen": {}, "llama": {}, "repeat_vs_original": {}}


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def sha(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def parse_qwen(o):
    m = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", o or "")
    return "invalid" if not m else ("safe" if m.group(1) == "Safe" else "unsafe")


def parse_llama(o):
    lines = [l.strip().lower() for l in (o or "").splitlines() if l.strip()]
    if not lines:
        return "invalid"
    first = lines[0]
    return first if first in ("safe", "unsafe") else "invalid"


def exact_p(a, b):
    """Two-sided exact McNemar (binomial, p=0.5) on discordant counts a,b."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0] * len(ps), 0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def load(path, sel_sha, n, parser, arm):
    d = json.loads(path.read_text())
    if not d.get("finished"):
        issues.append(f"{path}: not finished")
    if d.get("selection_sha256") != sel_sha:
        issues.append(f"{path}: selection hash mismatch")
    if d.get("arm") != arm:
        issues.append(f"{path}: arm field {d.get('arm')}")
    recs = d["records"]
    if len(recs) != n or len({r["id"] for r in recs}) != n:
        issues.append(f"{path}: {len(recs)} records / duplicates")
    res = {}
    for r in recs:
        p = parser(r.get("output"))
        if p != r["binary"]:
            issues.append(f"{path}/{r['id']}: reparsed {p} != stored {r['binary']}")
        res[r["id"]] = (r["gold"], p)
    return res


def metrics(x):
    ok = {i: v for i, v in x.items() if v[1] != "invalid"}
    U = [i for i in ok if ok[i][0] == "unsafe"]; S = [i for i in ok if ok[i][0] == "safe"]
    fn = sum(ok[i][1] == "safe" for i in U); fp = sum(ok[i][1] == "unsafe" for i in S)
    return dict(n=len(x), invalid=len(x) - len(ok), unsafe=len(U), safe=len(S), FN=fn, FP=fp,
                acc=(len(ok) - fn - fp) / len(x), FNR=fn / len(U), FPR=fp / len(S))


def compare(a, b):
    ids = [i for i in a if a[i][1] != "invalid" and b[i][1] != "invalid"]
    flips = [i for i in ids if a[i][1] != b[i][1]]
    improve = sum(b[i][1] == b[i][0] for i in flips)
    regress = len(flips) - improve
    newly_missed = sum(a[i][0] == "unsafe" and b[i][1] == "safe" for i in flips)
    recovered = sum(a[i][0] == "unsafe" and b[i][1] == "unsafe" for i in flips)
    return dict(flips=len(flips), improve=improve, regress=regress, p_acc=exact_p(improve, regress),
                newly_missed_unsafe=newly_missed, recovered_unsafe=recovered,
                p_unsafe=exact_p(newly_missed, recovered), flip_ids=sorted(flips))


def run_family(base, dsets, arms, parser, manifest_name, key):
    preds = {}
    for ds in dsets:
        man = json.loads((base / ds / manifest_name).read_text())
        sel_name = "selection.json" if key == "qwen" else "llama_selection.json"
        sel = json.loads((base / ds / sel_name).read_text())
        if sha(sel) != man["selection_sha256"]:
            issues.append(f"{key}/{ds}: recomputed selection SHA != manifest")
        if len(sel) != man["n"]:
            issues.append(f"{key}/{ds}: selection length")
        sel_ids = {r["id"] for r in sel}
        preds[ds] = {a: load(base / ds / f"{a}.json", man["selection_sha256"], man["n"], parser, a) for a in arms}
        for a in arms:
            if set(preds[ds][a]) != sel_ids:
                issues.append(f"{key}/{ds}/{a}: IDs differ from selection")
        out[key][ds] = {"manifest_n": man["n"], "metrics": {a: metrics(preds[ds][a]) for a in arms},
                        "pairs": {f"{a}->{b}": compare(preds[ds][a], preds[ds][b]) for a, b in combinations(arms, 2)},
                        "sel_ids": sel_ids, "selection_sha": man["selection_sha256"],
                        "source_sha": man.get("source_selection_sha256")}
    return preds


qp = run_family(Q, QDS, QARMS, parse_qwen, "manifest.json", "qwen")
lp = run_family(L, LDS, LARMS, parse_llama, "llama_manifest.json", "llama")

# Cross-family provenance: Llama selections drawn from the Qwen frozen selections?
for ds in LDS:
    src = out["llama"][ds]["source_sha"]
    if src != out["qwen"][ds]["selection_sha"]:
        issues.append(f"llama/{ds}: source selection SHA does not match Qwen selection SHA")
    if not out["llama"][ds]["sel_ids"] <= out["qwen"][ds]["sel_ids"]:
        issues.append(f"llama/{ds}: IDs not a subset of Qwen IDs")

# Original (main run) vs fresh repeatability records
for ds in QDS:
    for arm, base, files in (("bf16", GREP, ["bf16_repeat_A", "bf16_repeat_B"]),
                             ("nf4", QREP, ["nf4_repeat_A", "nf4_repeat_B"]),
                             ("fp4", QREP, ["fp4_repeat_A", "fp4_repeat_B"])):
        man = json.loads((base / ds / "subset_manifest.json").read_text())
        if man["frozen_sha256"] != out["qwen"][ds]["selection_sha"]:
            issues.append(f"repeat/{ds}: frozen SHA != main selection SHA")
        res = []
        for f in files:
            recs = json.loads((base / ds / f"{f}.json").read_text())["records"]
            ch = [r["id"] for r in recs if qp[ds][arm][r["id"]][1] != parse_qwen(r["output"])]
            golds = [r["id"] for r in recs if qp[ds][arm][r["id"]][0] != r["gold"]]
            if golds:
                issues.append(f"repeat/{ds}/{f}: gold label differs from main run for {len(golds)} ids")
            res.append(len(ch))
        out["repeat_vs_original"][f"{ds}/{arm}"] = res

# Holm families
def fam(key, dsets, pair, field):
    ps = [out[key][d]["pairs"][pair][field] for d in dsets]
    return dict(zip(dsets, zip(ps, holm(ps))))

holm_res = {
    "qwen nf4->fp4 accuracy (4 tests)": fam("qwen", QDS, "nf4->fp4", "p_acc"),
    "qwen nf4->fp4 unsafe (4 tests)": fam("qwen", QDS, "nf4->fp4", "p_unsafe"),
    "llama nf4->fp4 unsafe (2 primary)": fam("llama", LDS, "nf4->fp4", "p_unsafe"),
    "llama nf4->fp4 accuracy (2)": fam("llama", LDS, "nf4->fp4", "p_acc"),
}
ps8 = [out["qwen"][d]["pairs"][p]["p_acc"] for p in ("nf4->fp4", "fp8->int8") for d in QDS]
holm_res["qwen 4-bit + 8-bit accuracy (8 tests)"] = dict(
    zip([f"{p}/{d}" for p in ("nf4->fp4", "fp8->int8") for d in QDS], zip(ps8, holm(ps8))))

# Print
for key in ("qwen", "llama"):
    print(f"\n===== {key.upper()}")
    total = 0
    for ds, v in out[key].items():
        print(f"-- {ds} (n={v['manifest_n']})")
        for a, m in v["metrics"].items():
            total += m["n"]
            print(f"   {a:5s} n={m['n']} inv={m['invalid']} U={m['unsafe']} S={m['safe']} FN={m['FN']} FP={m['FP']} "
                  f"acc={m['acc']:.4f} FNR={m['FNR']:.4f} FPR={m['FPR']:.4f}")
        for p, c in v["pairs"].items():
            print(f"   {p:11s} flips={c['flips']:3d} improve={c['improve']:3d} regress={c['regress']:3d} p_acc={c['p_acc']:.3g} "
                  f"| missed+={c['newly_missed_unsafe']} recovered={c['recovered_unsafe']} p_unsafe={c['p_unsafe']:.3g}")
    print(f"   TOTAL arm-records: {total}")
print("\n===== HOLM")
for k, v in holm_res.items():
    print(k)
    for d, (p, a) in v.items():
        print(f"   {d:32s} p={p:.4g} holm={a:.4g}")
print("\n===== ORIGINAL vs FRESH REPEAT (changes in A, B)")
for k, v in out["repeat_vs_original"].items():
    print(f"   {k:32s} {v}")
print("\nISSUES:", len(issues))
for i in issues[:60]:
    print("  ", i)
