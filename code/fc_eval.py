import numpy as np, json, sys
from common import *
def evaluate(name):
    d = np.load(os.path.join(OUT, f"fc_{name}.npz"))
    df, _ = load_building(name); tr, ca, te = split_masks(df)
    out = {}
    for split, m in [("train_oof", tr), ("test", te)]:
        rows = np.where(m & ~np.isnan(d["med"][:, 0]) & ~np.isnan(d["y"][:, -1]))[0]
        y = d["y"][rows]
        r = {"n": len(rows), "mae_med_h1": float(np.abs(d["med"][rows, 0] - y[:, 0]).mean()),
             "mae_med_all": float(np.abs(d["med"][rows] - y).mean())}
        meths = ["raw", "cqr"] + (["aci"] if split == "test" else [])
        for mth in meths:
            lo, hi = d["lo_" + mth][rows], d["hi_" + mth][rows]
            cov = ((y >= lo) & (y <= hi)).mean(0)
            w = (hi - lo).mean(0)
            a = ALPHA
            wink = (hi - lo) + 2 / a * (lo - y) * (y < lo) + 2 / a * (y - hi) * (y > hi)
            r[mth] = dict(cov_mean=float(cov.mean()), cov_h1=float(cov[0]), cov_h6=float(cov[5]), cov_h24=float(cov[-1]),
                          upper_miss=float((y > hi).mean()), width_mean=float(w.mean()), winkler=float(wink.mean()))
        out[split] = r
    return out
if __name__ == "__main__":
    for n in sys.argv[1:]:
        print(n, json.dumps(evaluate(n), indent=1))
