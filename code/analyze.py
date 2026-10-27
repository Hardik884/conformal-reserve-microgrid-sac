"""Aggregate SAC runs + baselines, paired statistics, and the data behind every table/figure."""
import json, glob, itertools, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from common import *
HOMES = ["Building_8", "Building_2", "Building_10"]
ROLE = {"Building_8": "Donor (B8)", "Building_2": "Balanced (B2)", "Building_10": "Spiky (B10)"}
VARS = ["SAC", "SAC-P", "SAC-QR", "SAC-CQR", "SAC-CQR-S"]

def load_runs():
    rows = []
    for f in glob.glob(os.path.join(OUT, "sac", "*.json")):
        r = json.load(open(f))
        base = dict(variant=r["variant"], building=r["building"], seed=r["seed"])
        rows.append({**base, "eval": "default", **r["test"]})
        for k, v in r.get("extra", {}).items():
            rows.append({**base, "eval": k, **v})
    return pd.DataFrame(rows)

def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj

def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d)
    m = rng.choice(d, (n, len(d))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def paired(df, a, b, metric, eval_a="default", eval_b="default"):
    A = df[(df.variant == a) & (df["eval"] == eval_a)].set_index(["building", "seed"])[metric]
    B = df[(df.variant == b) & (df["eval"] == eval_b)].set_index(["building", "seed"])[metric]
    j = A.index.intersection(B.index)
    d = (A[j] - B[j]).values
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return dict(a=a + ("" if eval_a == "default" else ":" + eval_a), b=b + ("" if eval_b == "default" else ":" + eval_b),
                metric=metric, n=len(d), mean_diff=float(d.mean()), ci=boot_ci(d), p=float(p),
                wins=int((d < 0).sum()))

EXTRA = [  # (a, eval_a, b, eval_b) evaluation-time and post-hoc comparisons; Holm within each metric
    ("SAC-P", "posthoc_layer_aci", "SAC-P", "default"),
    ("SAC-P", "posthoc_layer_cqr", "SAC-P", "default"),
    ("SAC-P", "posthoc_layer_aci", "SAC-P", "posthoc_layer_cqr"),
    ("SAC-CQR", "posthoc_layer_raw", "SAC-CQR", "posthoc_layer_cqr"),
    ("SAC-CQR", "posthoc_layer_aci", "SAC-CQR", "posthoc_layer_cqr"),
    ("SAC-CQR-S", "default", "SAC-CQR-S", "layer_off"),
    ("SAC-CQR-S", "aci_inputs_aci_layer", "SAC-CQR-S", "default"),
    ("SAC-CQR", "posthoc_layer_cqr", "SAC-CQR-S", "default"),              # like-for-like: same inputs + same layer, added after vs during training
    ("SAC-P", "posthoc_layer_aci", "SAC-CQR-S", "aci_inputs_aci_layer"),
]

if __name__ == "__main__":
    df = load_runs()
    df.to_csv(os.path.join(OUT, "sac_all.csv"), index=False)
    d0 = df[df["eval"] == "default"]
    print(d0.groupby(["building", "variant"])[["cost", "energy_cost", "viol", "over_kwh"]].agg(["mean", "std"]).round(1).to_string())
    comps = [("SAC-P", "SAC"), ("SAC-QR", "SAC-P"), ("SAC-CQR", "SAC-P"), ("SAC-CQR", "SAC-QR"),
             ("SAC-CQR-S", "SAC-CQR"), ("SAC-CQR-S", "SAC-P")]
    main, extra = [], []
    for metric in ["cost", "viol"]:
        rs = [paired(df, a, b, metric) for a, b in comps]
        for r, q in zip(rs, holm([r["p"] for r in rs])): r["p_holm"] = float(q)
        main += rs
        rs = [paired(df, a, b, metric, ea, eb) for a, ea, b, eb in EXTRA]
        for r, q in zip(rs, holm([r["p"] for r in rs])): r["p_holm"] = float(q)
        extra += rs
    for r in main + extra: print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
    json.dump(dict(paired=main, paired_extra=extra), open(os.path.join(OUT, "stats.json"), "w"), indent=1)
