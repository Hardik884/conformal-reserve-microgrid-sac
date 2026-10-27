"""Multi-horizon quantile CNN-LSTM forecaster + conformalised quantile regression (CQR)
+ adaptive conformal inference (ACI).

Forecast issued at decision time t (start of hour t) for net load in hours t..t+H-1,
using observations up to hour t-1 and the dataset's weather predictions issued at t-1.

Outputs (saved per building to out/fc_<building>.npz), arrays of shape [T, H]:
  med, lo_raw, hi_raw        raw quantile model (5%, 50%, 95%)
  lo_cqr, hi_cqr             split-CQR intervals (90% nominal)
For TRAINING months the forecasts are out-of-fold (blocked 4-fold cross-fitting, each fold
calibrated on the 30 days next to it), so the RL agent trains on realistic forecast errors.
For CALIBRATION + TEST months they come from one final model trained on all training months
and calibrated on April.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from common import *

torch.set_num_threads(1)
CAL_LEN = 720  # 30 days


# ----------------------------------------------------------------- features
def make_features(df):
    hr = df.hour.values
    dow = (df.day_type.values - 1) % 7
    cal = np.stack([np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24),
                    np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                    (dow >= 5).astype(float)], 1)
    wcols = ["w_outdoor_dry_bulb_temperature", "w_diffuse_solar_irradiance", "w_direct_solar_irradiance"]
    pcols = [c for c in df.columns if c.startswith("w_") and "predicted" in c and "humidity" not in c]
    seq = np.column_stack([df.net.values, df.load.values, df.pv.values, df[wcols].values, cal])
    exo = np.column_stack([df[pcols].values, cal])  # weather predictions + calendar
    return seq.astype(np.float32), exo.astype(np.float32)


def build_samples(seq, exo, net, idx):
    """idx: decision times t. X: seq[t-L:t], E: exo[t-1] (latest info) + calendar at t, Y: net[t:t+H]."""
    X = np.stack([seq[t - LOOKBACK:t] for t in idx])
    E = np.stack([np.concatenate([exo[t - 1, :-5], exo[t, -5:]]) for t in idx])
    Y = np.stack([net[t:t + H] for t in idx])
    return X, E, Y


def valid_times(allowed, n):
    """decision times t whose whole window [t-L, t+H) lies inside 'allowed' rows."""
    ok = []
    for t in range(LOOKBACK, n - H + 1):
        if allowed[t - LOOKBACK:t + H].all():
            ok.append(t)
    return np.array(ok)


# ----------------------------------------------------------------- model
class QuantileCNNLSTM(nn.Module):
    """Conv1D(64)-Conv1D(32)-LSTM(64) encoder as in the preliminary study, with a multi-horizon,
    non-crossing 3-quantile head."""
    def __init__(self, f_seq, f_exo):
        super().__init__()
        self.c1 = nn.Conv1d(f_seq, 64, 3, padding=1); self.b1 = nn.BatchNorm1d(64)
        self.c2 = nn.Conv1d(64, 32, 3, padding=1); self.b2 = nn.BatchNorm1d(32)
        self.drop = nn.Dropout(0.2)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64 + f_exo, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 3 * H))

    def forward(self, x, e):
        z = x.transpose(1, 2)
        z = self.drop(self.b1(torch.relu(self.c1(z))))
        z = self.b2(torch.relu(self.c2(z)))
        _, (h, _) = self.lstm(z.transpose(1, 2))
        o = self.head(torch.cat([h[-1], e], 1)).view(-1, H, 3)
        med = o[..., 1]
        lo = med - nn.functional.softplus(o[..., 0])
        hi = med + nn.functional.softplus(o[..., 2])
        return torch.stack([lo, med, hi], -1)


def pinball(pred, y):
    q = torch.tensor(QUANTILES, dtype=pred.dtype)
    err = y.unsqueeze(-1) - pred
    return torch.maximum(q * err, (q - 1) * err).mean()


def fit_model(Xtr, Etr, Ytr, seed, max_epochs=80, patience=10):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(Xtr); nv = int(0.1 * n)
    # scalers fitted on the fitting block only (no leakage)
    xm, xs = Xtr[:-nv].reshape(-1, Xtr.shape[2]).mean(0), Xtr[:-nv].reshape(-1, Xtr.shape[2]).std(0) + 1e-6
    em, es = Etr[:-nv].mean(0), Etr[:-nv].std(0) + 1e-6
    ys = Ytr[:-nv].std() + 1e-6
    sc = dict(xm=xm, xs=xs, em=em, es=es, ys=ys)
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    X, Ee, Y = T((Xtr - xm) / xs), T((Etr - em) / es), T(Ytr / ys)
    model = QuantileCNNLSTM(X.shape[2], Ee.shape[1])
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    best, bad, best_state = 1e9, 0, None
    # chronological validation tail (last 10%)
    Xt, Et, Yt, Xv, Ev, Yv = X[:-nv], Ee[:-nv], Y[:-nv], X[-nv:], Ee[-nv:], Y[-nv:]
    g = torch.Generator().manual_seed(seed)
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), 128):
            b = perm[i:i + 128]
            loss = pinball(model(Xt[b], Et[b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = pinball(model(Xv, Ev), Yv).item()
        if vl < best - 1e-5:
            best, bad, best_state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, sc, ep + 1, best


def predict(model, sc, X, E):
    with torch.no_grad():
        p = model(torch.tensor((X - sc["xm"]) / sc["xs"], dtype=torch.float32),
                  torch.tensor((E - sc["em"]) / sc["es"], dtype=torch.float32)).numpy()
    return p * sc["ys"]  # [N, H, 3] in kW


def cqr_qhat(lo, hi, y, alpha=ALPHA):
    """per-horizon CQR correction (Romano et al., 2019)."""
    s = np.maximum(lo - y, y - hi)  # [N, H]
    n = s.shape[0]
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(s, lvl, axis=0, method="higher"), s


# ----------------------------------------------------------------- ACI
def aci_intervals(lo, hi, y, valid_t, cal_scores, alpha=ALPHA, gamma=0.005, window=CAL_LEN):
    """Adaptive conformal inference (Gibbs & Candes, 2021) with a rolling score window and
    delayed feedback: the score for a forecast issued at t for hour t+h-1 is only used once that
    hour has been observed (at decision time t+h)."""
    T_, Hh = lo.shape
    out_lo, out_hi = np.full_like(lo, np.nan), np.full_like(hi, np.nan)
    a = np.full(Hh, alpha)
    buf = [list(cal_scores[-window:, h]) for h in range(Hh)]
    issued = {}  # (t, h) -> (lo, hi)
    tset = set(valid_t.tolist())
    for t in valid_t:
        # 1) learn from every interval whose target hour (t-1) has now been observed
        for h in range(Hh):
            ts = t - 1 - h
            if (ts, h) in issued:
                l, u = issued.pop((ts, h))
                yt = y[ts, h]
                err = float(yt < l or yt > u)
                a[h] = a[h] + gamma * (alpha - err)
                buf[h].append(max(lo[ts, h] - yt, yt - hi[ts, h]))
                if len(buf[h]) > window:
                    buf[h].pop(0)
        # 2) issue new intervals
        for h in range(Hh):
            s = np.asarray(buf[h])
            if a[h] <= 0:
                q = s.max() * 2  # infinite interval in theory; use a very wide one
            elif a[h] >= 1:
                q = -np.inf
            else:
                n = len(s)
                q = np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - a[h])) / n), method="higher")
            l, u = lo[t, h] - q, hi[t, h] + q
            if u < l:
                l = u = (lo[t, h] + hi[t, h]) / 2
            out_lo[t, h], out_hi[t, h] = l, u
            issued[(t, h)] = (l, u)
    return out_lo, out_hi


# ----------------------------------------------------------------- main per building
def run_building(name, seed=0):
    df, meta = load_building(name)
    n = len(df)
    tr, ca, te = split_masks(df)
    seq, exo = make_features(df)
    net = df.net.values.astype(np.float32)
    res = {k: np.full((n, H), np.nan, np.float32) for k in ["med", "lo_raw", "hi_raw", "lo_cqr", "hi_cqr"]}
    Ytrue = np.full((n, H), np.nan, np.float32)
    for t in range(n - H + 1):
        Ytrue[t] = net[t:t + H]
    log = {"building": name, "folds": []}

    # ---- out-of-fold forecasts for training months (blocked 4-fold cross-fitting)
    tr_rows = np.where(tr)[0]
    folds = np.array_split(tr_rows, 4)
    for k, f in enumerate(folds):
        fold_mask = np.zeros(n, bool); fold_mask[f] = True
        # calibration chunk: 30 days right before the fold (fold 0: right after it)
        if k > 0:
            cal_rows = np.arange(f[0] - CAL_LEN, f[0])
        else:
            cal_rows = np.arange(f[-1] + 1, f[-1] + 1 + CAL_LEN)
        cal_mask = np.zeros(n, bool); cal_mask[cal_rows] = True
        fit_mask = tr & ~fold_mask & ~cal_mask
        t_fit = valid_times(fit_mask, n)
        t_cal = valid_times(cal_mask, n)
        t_fold = np.array([t for t in f if LOOKBACK <= t <= n - H])
        model, sc, ep, vl = fit_model(*build_samples(seq, exo, net, t_fit), seed=seed + k)
        pc = predict(model, sc, *build_samples(seq, exo, net, t_cal)[:2])
        qhat, _ = cqr_qhat(pc[..., 0], pc[..., 2], Ytrue[t_cal])
        pf = predict(model, sc, *build_samples(seq, exo, net, t_fold)[:2])
        res["med"][t_fold] = pf[..., 1]; res["lo_raw"][t_fold] = pf[..., 0]; res["hi_raw"][t_fold] = pf[..., 2]
        res["lo_cqr"][t_fold] = pf[..., 0] - qhat; res["hi_cqr"][t_fold] = pf[..., 2] + qhat
        log["folds"].append(dict(k=k, n_fit=len(t_fit), n_cal=len(t_cal), epochs=ep, val_pinball=vl,
                                 qhat_h1=float(qhat[0]), qhat_mean=float(qhat.mean())))
        print(name, "fold", k, log["folds"][-1], flush=True)

    # ---- final model: fit on training months, calibrate on April, forecast April + test
    t_fit = valid_times(tr, n)
    t_cal = valid_times(ca, n)
    t_test = np.array([t for t in np.where(te)[0] if t <= n - H])
    model, sc, ep, vl = fit_model(*build_samples(seq, exo, net, t_fit), seed=seed + 10)
    pc = predict(model, sc, *build_samples(seq, exo, net, t_cal)[:2])
    qhat, cal_scores = cqr_qhat(pc[..., 0], pc[..., 2], Ytrue[t_cal])
    for tt, pp in [(t_cal, pc), (t_test, predict(model, sc, *build_samples(seq, exo, net, t_test)[:2]))]:
        res["med"][tt] = pp[..., 1]; res["lo_raw"][tt] = pp[..., 0]; res["hi_raw"][tt] = pp[..., 2]
        res["lo_cqr"][tt] = pp[..., 0] - qhat; res["hi_cqr"][tt] = pp[..., 2] + qhat
    log["final"] = dict(n_fit=len(t_fit), n_cal=len(t_cal), n_test=len(t_test), epochs=ep, val_pinball=vl,
                        qhat=qhat.tolist())
    # ---- ACI on the test period (rolling window, delayed feedback)
    lo_aci, hi_aci = aci_intervals(res["lo_raw"], res["hi_raw"], Ytrue, t_test, cal_scores)
    res["lo_aci"], res["hi_aci"] = lo_aci.astype(np.float32), hi_aci.astype(np.float32)
    np.savez_compressed(os.path.join(OUT, f"fc_{name}.npz"), y=Ytrue, **res)
    json.dump(log, open(os.path.join(OUT, f"fc_{name}.json"), "w"), indent=1)
    print(name, "final", {k: v for k, v in log["final"].items() if k != "qhat"}, flush=True)


if __name__ == "__main__":
    t0 = time.time()
    run_building(sys.argv[1])
    print("done in %.0fs" % (time.time() - t0))
