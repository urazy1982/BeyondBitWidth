"""Independent check of Llama Guard 3 NF4/FP4 independent-reload repeatability against the original run."""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(sys.argv[1])
REP = ROOT / "llama_rep/llama_guard_reload_repeatability"
ORIG = ROOT / "llama/llama_guard_replication_results"
DS = ["xstest_prompt", "beavertails_response"]
issues = []


def parse(o):
    lines = [l.strip().lower() for l in (o or "").splitlines() if l.strip()]
    return lines[0] if lines and lines[0] in ("safe", "unsafe") else "invalid"


def load(p, arm, rep=None, sel_sha=None, ids=None):
    d = json.loads(p.read_text())
    if not d.get("finished"): issues.append(f"{p}: not finished")
    if d.get("arm") != arm: issues.append(f"{p}: arm {d.get('arm')}")
    if rep and d.get("repeat") != rep: issues.append(f"{p}: repeat {d.get('repeat')}")
    if sel_sha and d.get("selection_sha256") != sel_sha: issues.append(f"{p}: selection sha")
    if d.get("model_revision") != "7327bd9f6efbbe6101dc6cc4736302b3cbb6e425": issues.append(f"{p}: revision")
    out = {}
    for r in d["records"]:
        b = parse(r["output"])
        if b != r["binary"]: issues.append(f"{p}/{r['id']}: reparsed {b} != {r['binary']}")
        out[r["id"]] = (r["gold"], b)
    if ids is not None and set(out) != ids: issues.append(f"{p}: ID set differs from selection")
    if len(out) != len(d["records"]): issues.append(f"{p}: duplicate IDs")
    return out, d.get("updated_utc")


def flips(a, b):
    return sorted(i for i in a if a[i][1] != b[i][1])


# 1) original-input hashes recorded by the reload run vs the original files we hold
env = json.loads((REP / "reload_environment.json").read_text())
for rel, h in env["original_sha256"].items():
    actual = hashlib.sha256((ORIG / rel).read_bytes()).hexdigest()
    print(f"orig hash {rel:42s} {'OK' if actual == h else 'MISMATCH'}")
    if actual != h: issues.append(f"original hash mismatch: {rel}")

# 2) per-dataset comparisons
for ds in DS:
    man = json.loads((ORIG / ds / "llama_manifest.json").read_text())
    ids = {r["id"] for r in json.loads((ORIG / ds / "llama_selection.json").read_text())}
    o = {a: load(ORIG / ds / f"{a}.json", a, sel_sha=man["selection_sha256"], ids=ids)[0] for a in ("nf4", "fp4")}
    f, ts = {}, {}
    for a in ("nf4", "fp4"):
        for rep in "AB":
            f[a, rep], ts[a, rep] = load(REP / ds / f"{a}_{rep}.json", a, rep, man["selection_sha256"], ids)
    print(f"\n== {ds} (n={len(ids)})  timestamps: " + ", ".join(f"{a}_{r}={ts[a, r]}" for a, r in ts))
    for a in ("nf4", "fp4"):
        print(f"  {a}: A vs B = {len(flips(f[a,'A'], f[a,'B']))}, orig vs A = {len(flips(o[a], f[a,'A']))}, "
              f"orig vs B = {len(flips(o[a], f[a,'B']))}")
        fn = {k: sum(v[0] == 'unsafe' and v[1] == 'safe' for v in x.values())
              for k, x in (("orig", o[a]), ("A", f[a, 'A']), ("B", f[a, 'B']))}
        print(f"       FN counts {fn}")
    so, sa, sb = flips(o["nf4"], o["fp4"]), flips(f["nf4", "A"], f["fp4", "A"]), flips(f["nf4", "B"], f["fp4", "B"])
    print(f"  NF4-vs-FP4 flips: orig={len(so)} A={len(sa)} B={len(sb)} identical_sets={so == sa == sb}")
    nm = sum(f["nf4", "A"][i][0] == "unsafe" and f["fp4", "A"][i][1] == "safe" for i in sa)
    rc = sum(f["nf4", "A"][i][0] == "unsafe" and f["fp4", "A"][i][1] == "unsafe" for i in sa)
    print(f"  fresh A NF4->FP4: newly missed unsafe={nm}, recovered unsafe={rc}")
    if not so == sa == sb: issues.append(f"{ds}: NF4-FP4 flip sets differ across passes")

print("\nISSUES:", len(issues))
for i in issues[:40]: print("  ", i)
