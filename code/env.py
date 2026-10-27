"""Energy-conserving residential PV + battery environment with a two-level controller and the
conformal reserve (safety) layer.

Two-level control (as in commercial home batteries):
  * high level, once per hour: choose a charge threshold g_lo and a discharge threshold g_hi
    on the household's grid import (kW), g_lo <= g_hi <= P_cap;
  * low level, within the hour: the inverter tracks the realised net load y_t:
        y_t > g_hi -> discharge (y_t - g_hi);  y_t < g_lo -> charge (g_lo - y_t);  else idle,
    within power and state-of-charge limits.
  g_lo = g_hi = 0 is the familiar self-consumption mode; g_hi = P_cap is pure peak shaving.

Settlement per hour: grid = y + p_c - p_d, import = max(grid,0), export = max(-grid,0),
cost = price*import - FEED_IN*export + LAMBDA_CAP*max(0, import - P_cap).
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from common import *


class HomeData:
    """Everything a controller may use for one building, as plain arrays."""
    def __init__(self, name):
        df, meta = load_building(name)
        self.name = name
        self.net = df.net.values.astype(np.float64)
        self.price = df.price.values.astype(np.float64)
        self.hour = df.hour.values
        self.month = df.month.values
        self.dow = (df.day_type.values - 1) % 7
        self.cap_kwh, self.p_max = meta["cap_kwh"], meta["p_max"]
        tr, ca, te = split_masks(df)
        self.tr, self.ca, self.te = tr, ca, te
        self.p_cap = float(np.percentile(self.net[tr], CAP_PCTL))
        fc = np.load(os.path.join(OUT, f"fc_{name}.npz"))
        self.fc = {k: fc[k].astype(np.float64) for k in fc.files}
        ok = ~np.isnan(self.fc["med"]).any(1) & ~np.isnan(self.fc["y"]).any(1)
        idx = np.arange(len(self.net))
        self.train_starts = idx[tr & ok & (idx >= 24)]
        self.test_idx = idx[te & ok]
        self.val_idx = idx[ca & ok]

    def upper(self, src, t, n=H_AGENT):
        if src == "med":
            return self.fc["med"][t, :n]
        return self.fc["hi_" + src][t, :n]


def battery_limits(soc, cap, p_max):
    """max charge / discharge power (kW) that keeps SoC inside [SOC_MIN, SOC_MAX] over 1 h."""
    c_max = min(p_max, max(0.0, (SOC_MAX - soc) * cap / ETA_C))
    d_max = min(p_max, max(0.0, (soc - SOC_MIN) * cap * ETA_D))
    return c_max, d_max


def settle(pc, pd, soc, y, price, cap, p_cap):
    soc_new = soc + (ETA_C * pc - pd / ETA_D) / cap
    grid = y + pc - pd
    imp, exp_ = max(grid, 0.0), max(-grid, 0.0)
    over = max(0.0, imp - p_cap)
    return soc_new, dict(cost=price * imp - FEED_IN * exp_ + LAMBDA_CAP * over,
                         energy_cost=price * imp - FEED_IN * exp_, imp=imp, exp=exp_,
                         over=over, viol=float(over > 1e-6), pc=pc, pd=pd)


def apply_power(p, soc, y, price, cap, p_max, p_cap):
    """direct power set-point (used only by the perfect-foresight oracle)."""
    c_max, d_max = battery_limits(soc, cap, p_max)
    pc = min(max(p, 0.0), c_max); pd = min(max(-p, 0.0), d_max)
    return settle(pc, pd, soc, y, price, cap, p_cap)


def track(g_lo, g_hi, soc, y, price, cap, p_max, p_cap, soc_res=None, c_lim=None):
    """low-level threshold tracking within the hour, optionally honouring a reserve SoC:
    energy below soc_res may only be used to shave import ABOVE P_cap."""
    c_max, d_max = battery_limits(soc, cap, p_max)
    pc = pd = 0.0
    if y > g_hi:
        want = y - g_hi
        if soc_res is None:
            pd = min(want, d_max)
        else:
            pk = min(want, max(0.0, y - p_cap))           # part that would overrun the cap
            free = max(0.0, (soc - soc_res) * cap * ETA_D)  # energy above the reserve
            pd = min(d_max, pk + min(want - pk, free))
    elif y < g_lo:
        pc = min(g_lo - y, c_max)
        if c_lim is not None:
            pc = min(pc, max(c_lim, -y if y < 0 else 0.0))  # PV surplus is always stored
    return settle(pc, pd, soc, y, price, cap, p_cap)


def reserve_soc(U, cap, p_max, p_cap):
    """SoC needed to shave the predicted exceedances of the coming hours (U = upper bounds for
    hours t+1..t+k)."""
    e = sum(min(p_max, max(0.0, u - p_cap)) for u in U) / ETA_D
    return min(SOC_MAX, SOC_MIN + e / cap)


def safety_layer(g_lo, g_hi, soc, U, cap, p_cap, p_max=5.0):
    """Conformal reserve layer. U = upper bounds for hours t..t+H_AGENT-1 issued at t.
    Returns modified thresholds + the reserve SoC and charge limit for the tracker."""
    s_req = reserve_soc(U[1:], cap, p_max, p_cap)
    c_lim = None
    if soc < s_req:                         # rebuild the reserve with grid energy, but never above the cap
        g_lo = p_cap
        g_hi = max(g_hi, g_lo)
        c_lim = (s_req - soc) * cap / ETA_C
    return g_lo, g_hi, s_req, c_lim


def decode(action, p_cap, p_max):
    """Prior-centred action map: a = (0, 0) is the self-consumption rule g_lo = g_hi = 0.
    a_i > 0 scales a threshold up towards P_cap, a_i < 0 down towards -P_max.
    g_lo = charge threshold, g_hi = discharge threshold, g_lo <= g_hi <= P_cap."""
    m = lambda a: a * p_cap if a >= 0 else a * p_max
    g_lo = m(float(np.clip(action[0], -1, 1)))
    g_hi = m(float(np.clip(action[1], -1, 1)))
    return min(g_lo, g_hi), g_hi


VARIANTS = {
    # name: (forecast features in the state, safety-layer bound source or None)
    "SAC":       (None, None),
    "SAC-P":     ("med", None),
    "SAC-QR":    ("raw", None),
    "SAC-CQR":   ("cqr", None),
    "SAC-CQR-S": ("cqr", "cqr"),
}


class MicrogridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, home: HomeData, variant="SAC-CQR", mode="train", episode_len=168,
                 interval_src=None, layer_src=None, seed=0):
        super().__init__()
        self.h = home
        self.feat, self.layer = VARIANTS[variant]
        if interval_src is not None:   # evaluation-time swap of the interval source (e.g. 'aci')
            self.feat = interval_src
        if layer_src is not None:
            self.layer = None if layer_src == "none" else layer_src
        self.mode, self.L = mode, episode_len
        self.rng = np.random.default_rng(seed)
        self.valid = set(home.train_starts.tolist())
        self.r_scale = float(os.environ.get("R_SCALE", 10.0))
        self.dayahead = os.environ.get("DAYAHEAD", "1") == "1" and self.feat is not None
        nf = {None: 0, "med": H_AGENT}.get(self.feat, 3 * H_AGENT) + (2 if self.dayahead else 0)
        self.observation_space = spaces.Box(-10, 10, (13 + nf,), np.float32)
        self.action_mode = os.environ.get("ACTION", "reserve")
        self.action_space = spaces.Box(-1, 1, (1 if self.action_mode == "reserve" else 2,), np.float32)

    def _obs(self):
        h, t, s = self.h, self.t, self.h.p_cap
        hr, dw = h.hour[t], h.dow[t]
        pr = h.price[t:t + 6]
        if len(pr) < 6:
            pr = np.pad(pr, (0, 6 - len(pr)), mode="edge")
        o = [self.soc, h.net[t - 1] / s, h.net[t - 2] / s, h.net[t - 24] / s, *(pr / 0.5),
             np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24), float(dw >= 5)]
        f = h.fc
        if self.feat == "med":
            o += list(f["med"][t, :H_AGENT] / s)
        elif self.feat is not None:
            o += list(f["med"][t, :H_AGENT] / s) + list(f["lo_" + self.feat][t, :H_AGENT] / s) + \
                 list(f["hi_" + self.feat][t, :H_AGENT] / s)
        if self.dayahead:   # day-ahead outlook from the 24 h median forecast
            m = f["med"][t, :H]
            o += [m.sum() / (H * s), np.maximum(-m, 0).sum() / (H * s)]
        return np.clip(np.nan_to_num(np.array(o, np.float32)), -10, 10)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.mode == "train":
            st = self.h.train_starts
            st = st[st + self.L < st.max()]
            self.t0 = int(self.rng.choice(st))
            self.soc = float(self.rng.uniform(0.2, 0.8))
            self.end = self.t0 + self.L
        else:
            ix = {"test": self.h.test_idx, "val": self.h.val_idx, "trainfull": self.h.train_starts}[self.mode]
            self.t0 = int(ix[0]); self.end = int(ix[-1]) + 1
            self.soc = 0.5
        self.t = self.t0
        return self._obs(), {}

    def step(self, action):
        h, t = self.h, self.t
        s_req = c_lim = None
        if self.action_mode == "reserve":
            # action = reserve state-of-charge target s*; a = -1 is the self-consumption rule
            s_star = SOC_MIN + (float(np.clip(action[0], -1, 1)) + 1) / 2 * (SOC_MAX - SOC_MIN)
            g_lo, g_hi = 0.0, 0.0
            if self.layer is not None:
                _, _, s_lay, _ = safety_layer(0.0, 0.0, self.soc, h.upper(self.layer, t), h.cap_kwh, h.p_cap, h.p_max)
                s_star = max(s_star, s_lay)
            s_req = s_star
            if self.soc < s_star:     # refill towards the target from the grid, never above P_cap
                g_lo, g_hi = h.p_cap, h.p_cap
                c_lim = (s_star - self.soc) * h.cap_kwh / ETA_C
        else:
            g_lo, g_hi = decode(action, h.p_cap, h.p_max)
            if self.layer is not None:
                g_lo, g_hi, s_req, c_lim = safety_layer(g_lo, g_hi, self.soc, h.upper(self.layer, t), h.cap_kwh, h.p_cap, h.p_max)
        self.soc, info = track(g_lo, g_hi, self.soc, h.net[t], h.price[t], h.cap_kwh, h.p_max, h.p_cap,
                               soc_res=s_req, c_lim=c_lim)
        info.update(t=t, g_lo=g_lo, g_hi=g_hi, soc=self.soc, soc_req=s_req if s_req is not None else np.nan)
        self.t += 1
        done = self.t >= self.end or (self.mode == "train" and self.t not in self.valid)
        obs = self._obs() if self.t < len(h.net) - H else np.zeros(self.observation_space.shape, np.float32)
        # episodes end on a time limit, not a terminal state -> report as truncation
        return obs, -self.r_scale * info["cost"], False, done, info
