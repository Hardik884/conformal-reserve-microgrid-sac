"""Non-learning controllers evaluated on the same physics (env.apply_power)."""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from env import HomeData, apply_power, safety_layer, track
from common import *

V_TERM = 0.21 * ETA_D   # $/kWh value of energy left in the battery at the MPC horizon end


def lp_plan(y_hat, prices, soc0, cap, p_max, p_cap, y_up=None, terminal_value=V_TERM):
    """Linear programme over len(y_hat) hours. Variables per hour: c, d, imp, exp, ov.
    If y_up is given the overrun term is evaluated on the upper bound (conformal MPC)."""
    n = len(y_hat)
    nv = 5 * n
    C, D, I, E, O = (np.arange(n) + k * n for k in range(5))
    cost = np.zeros(nv)
    cost[I] = prices; cost[E] = -FEED_IN; cost[O] = LAMBDA_CAP
    # terminal value of stored energy: soc_n = soc0 + sum(eta c - d/eta)/cap
    cost[C] -= terminal_value * ETA_C; cost[D] += terminal_value / ETA_D
    rows, cols, vals, b_eq = [], [], [], []
    for h in range(n):  # balance: imp - exp - c + d = y
        for j, v in ((I[h], 1), (E[h], -1), (C[h], -1), (D[h], 1)):
            rows.append(h); cols.append(j); vals.append(v)
        b_eq.append(y_hat[h])
    A_eq = sp.csr_matrix((vals, (rows, cols)), shape=(n, nv))
    r2, c2, v2, b_ub = [], [], [], []
    k = 0
    yu = y_hat if y_up is None else y_up
    for h in range(n):
        # overrun: ov >= (yu + c - d) - p_cap  ->  c - d - ov <= p_cap - yu
        for j, v in ((C[h], 1), (D[h], -1), (O[h], -1)):
            r2.append(k); c2.append(j); v2.append(v)
        b_ub.append(p_cap - yu[h]); k += 1
        # SoC bounds after hour h: cumulative sum
        for j in range(h + 1):
            r2 += [k, k]; c2 += [C[j], D[j]]; v2 += [ETA_C / cap, -1 / (ETA_D * cap)]
        b_ub.append(SOC_MAX - soc0); k += 1
        for j in range(h + 1):
            r2 += [k, k]; c2 += [C[j], D[j]]; v2 += [-ETA_C / cap, 1 / (ETA_D * cap)]
        b_ub.append(soc0 - SOC_MIN); k += 1
    A_ub = sp.csr_matrix((v2, (r2, c2)), shape=(k, nv))
    bounds = [(0, p_max)] * (2 * n) + [(0, None)] * (3 * n)
    r = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if r.status != 0:
        return np.zeros(n)
    return r.x[C] - r.x[D]


def rollout(home, policy, layer_src=None):
    """policy(t, soc) -> (g_lo, g_hi) thresholds, executed by the same low-level tracker."""
    soc = 0.5
    log = []
    for t in home.test_idx:
        g_lo, g_hi = policy(t, soc)
        s_req = c_lim = None
        if layer_src is not None:
            g_lo, g_hi, s_req, c_lim = safety_layer(g_lo, g_hi, soc, home.upper(layer_src, t), home.cap_kwh, home.p_cap, home.p_max)
        soc, info = track(g_lo, g_hi, soc, home.net[t], home.price[t], home.cap_kwh, home.p_max, home.p_cap,
                          soc_res=s_req, c_lim=c_lim)
        info.update(t=int(t), g_lo=g_lo, g_hi=g_hi, soc=soc)
        log.append(info)
    return log


def make_policies(home):
    f = home.fc
    pol = {}
    pol["NoBattery"] = lambda t, soc: (-1e9, 1e9)
    pol["RBC-SC"] = lambda t, soc: (0.0, 0.0)
    def mpc(src):
        def p(t, soc):
            yu = None if src is None else f["hi_" + src][t]
            plan = lp_plan(f["med"][t], home.price[t:t + H], soc, home.cap_kwh, home.p_max, home.p_cap, y_up=yu)
            g = min(home.p_cap, f["med"][t, 0] + plan[0])
            return (g, g)
        return p
    pol["MPC-P"] = mpc(None)
    pol["MPC-CQR"] = mpc("cqr")
    pol["MPC-ACI"] = mpc("aci")
    return pol


def oracle(home):
    """perfect-foresight LP over the whole test period, executed as direct power set-points."""
    idx = home.test_idx
    plan = lp_plan(home.net[idx], home.price[idx], 0.5, home.cap_kwh, home.p_max, home.p_cap, terminal_value=0.0)
    soc, log = 0.5, []
    for t, p in zip(idx, plan):
        soc, info = apply_power(float(p), soc, home.net[t], home.price[t], home.cap_kwh, home.p_max, home.p_cap)
        info.update(t=int(t), soc=soc); log.append(info)
    return log


def summarize(log):
    c = lambda k: float(sum(x[k] for x in log))
    return dict(cost=c("cost"), energy_cost=c("energy_cost"), viol=c("viol"), over_kwh=c("over"),
                imp=c("imp"), exp=c("exp"), peak=float(max(x["imp"] for x in log)),
                throughput=c("pc") + c("pd"))
