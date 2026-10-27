"""Home statistics and forecast-quality numbers behind Tables 3 and 5 and Figure 2 -> out/table_homes_fc.json"""
import json, numpy as np
from common import *
from env import HomeData
from fc_eval import evaluate
rows = []
for role, b in [("donor", "Building_8"), ("balanced", "Building_2"), ("spiky", "Building_10")]:
    df, meta = load_building(b); tr, ca, te = split_masks(df); h = HomeData(b); x = df[tr]; net = df.net.values
    T = h.test_idx; y = h.fc["y"][T]
    pers = np.stack([np.full(H, net[t - 1]) for t in T]); sn = np.stack([net[t - 24:t] for t in T]); ev = evaluate(b)
    r = dict(role=role, b=b, pv_kw=meta["pv_kw"], load_mean=x.load.mean(), load_std=x.load.std(), surplus=(x.net < 0).mean(),
             net_mean=x.net.mean(), p_cap=h.p_cap, test_viol_nobat=int((net[T] > h.p_cap).sum()),
             mae_model=float(np.abs(h.fc["med"][T] - y).mean()), mae_pers=float(np.abs(pers - y).mean()), mae_sn=float(np.abs(sn - y).mean()),
             mae_model_h1=float(np.abs(h.fc["med"][T, 0] - y[:, 0]).mean()), mae_pers_h1=float(np.abs(pers[:, 0] - y[:, 0]).mean()))
    for m in ["raw", "cqr", "aci"]:
        for k in ["cov_mean", "upper_miss", "width_mean", "winkler"]:
            r[f"{m}_{k}"] = ev["test"][m][k]
        lo, hi = h.fc["lo_" + m][T], h.fc["hi_" + m][T]; cov = ((y >= lo) & (y <= hi)).mean(1); mon = h.month[T]
        r[f"{m}_cov_by_month"] = {int(mm): float(cov[mon == mm].mean()) for mm in [5, 6, 7]}
    r["load_mean_by_month"] = {int(mm): float(df.load[df.month == mm].mean()) for mm in [4, 5, 6, 7]}
    rows.append(r)
json.dump(rows, open(OUT + "/table_homes_fc.json", "w"), indent=1)
