"""Risk dial: the conformal level alpha used by the reserve layer, swept for training-free
controllers (RBC-SC, MPC-P). Uses only April (calibration) + test months of the FINAL forecaster."""
import json, numpy as np
from common import *
from env import HomeData, track, safety_layer, reserve_soc
from forecast import aci_intervals
from baselines import lp_plan, summarize

ALPHAS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]

def bounds_for(h, alpha):
    f = h.fc; y = f["y"]
    cal = h.val_idx                    # April decision times (final model, not used for fitting)
    s = np.maximum(f["lo_raw"][cal] - y[cal], y[cal] - f["hi_raw"][cal])
    n = len(cal); lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q = np.quantile(s, lvl, axis=0, method="higher")
    hi_cqr = f["hi_raw"] + q
    lo_aci, hi_aci = aci_intervals(f["lo_raw"], f["hi_raw"], y, h.test_idx, s, alpha=alpha)
    return hi_cqr, hi_aci

def run(h, hi, ctrl):
    soc = 0.5; log = []; held = 0; need = 0
    for t in h.test_idx:
        if ctrl == "RBC-SC":
            g_lo = g_hi = 0.0
        else:
            plan = lp_plan(h.fc["med"][t], h.price[t:t + H], soc, h.cap_kwh, h.p_max, h.p_cap)
            g_lo = g_hi = min(h.p_cap, h.fc["med"][t, 0] + plan[0])
        # reserve condition for THIS hour, based on the bound issued one hour earlier (horizon 2)
        if t - 1 in set_idx:
            e = min(h.p_max, max(0.0, hi[t - 1, 1] - h.p_cap))
            if e > 0:
                need += 1
                held += soc >= SOC_MIN + e / (ETA_D * h.cap_kwh) - 1e-9
        g_lo, g_hi, s_req, c_lim = safety_layer(g_lo, g_hi, soc, hi[t, :H_AGENT], h.cap_kwh, h.p_cap, h.p_max)
        soc, info = track(g_lo, g_hi, soc, h.net[t], h.price[t], h.cap_kwh, h.p_max, h.p_cap, soc_res=s_req, c_lim=c_lim)
        log.append(info)
    r = summarize(log); r["reserve_held_share"] = held / max(need, 1); r["reserve_needed_hours"] = need
    return r

out = {}
for b in ["Building_8", "Building_2", "Building_10"]:
    h = HomeData(b); set_idx = set(h.test_idx.tolist())
    y = h.fc["y"]; T = h.test_idx
    for a in ALPHAS:
        hc, ha = bounds_for(h, a)
        for src, hi in [("cqr", hc), ("aci", ha)]:
            miss1 = float((y[T, 1] > hi[T, 1]).mean())   # horizon-2 upper miss (Proposition 1)
            for ctrl in ["RBC-SC", "MPC-P"]:
                r = run(h, hi, ctrl); r.update(upper_miss_h2=miss1, viol_rate=r["viol"] / len(T))
                out[f"{b}|{src}|{a}|{ctrl}"] = r
                print(b, src, a, ctrl, "cost %.1f viol %d miss_h2 %.3f viol_rate %.3f held %.2f" % (r["cost"], r["viol"], miss1, r["viol_rate"], r["reserve_held_share"]), flush=True)
json.dump(out, open(os.path.join(OUT, "dial.json"), "w"), indent=1)
