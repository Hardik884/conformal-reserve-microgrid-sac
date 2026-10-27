"""Shared configuration and data loading.

Data: CityLearn 2022 challenge dataset (17 single-family homes, Fontana CA, hourly,
one year Aug-Jul). Files live in ../data (copied from the CityLearn GitHub repo, MIT licence).
"""
import json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

# ---------------- problem constants (fixed before any test-set look) ----------------
H = 24            # forecast horizon (hours) used by MPC
H_AGENT = 6       # horizons exposed to the SAC agent / safety layer
LOOKBACK = 24     # hours of history fed to the forecaster
ALPHA = 0.10      # target miscoverage (90% intervals)
QUANTILES = (ALPHA / 2, 0.5, 1 - ALPHA / 2)
FEED_IN = 0.05    # $/kWh export credit (net-billing style, well below retail)
LAMBDA_CAP = 1.00 # $/kWh penalty on import above the subscribed capacity
CAP_PCTL = 90     # subscribed capacity = 90th pct of TRAINING-period hourly net load
ETA_C = ETA_D = 0.95   # one-way efficiencies (round trip ~0.90, as in the dataset schema)
SOC_MIN, SOC_MAX = 0.10, 0.95

TRAIN_MONTHS = [8, 9, 10, 11, 12, 1, 2, 3]   # Aug-Mar
CALIB_MONTHS = [4]                           # Apr
TEST_MONTHS = [5, 6, 7]                      # May-Jul


def load_schema():
    return json.load(open(os.path.join(DATA, "schema.json")))


def load_building(name):
    """Return an hourly DataFrame for one building with load, pv, net (kW) + exogenous."""
    s = load_schema()["buildings"][name]
    d = pd.read_csv(os.path.join(DATA, s["energy_simulation"]))
    w = pd.read_csv(os.path.join(DATA, "weather.csv"))
    p = pd.read_csv(os.path.join(DATA, "pricing.csv"))
    pv_kw = s["pv"]["attributes"]["nominal_power"]
    df = pd.DataFrame({
        "month": d.month, "hour": d.hour % 24, "day_type": d.day_type,
        "load": d.non_shiftable_load.astype(float),
        "pv": d.solar_generation.astype(float) * pv_kw / 1000.0,
        "price": p.electricity_pricing.astype(float),
    })
    df["net"] = df.load - df.pv
    for c in w.columns:
        df["w_" + c] = w[c].astype(float)
    # drop the single leading row that belongs to the previous July (hour-ending 24:00)
    df = df.iloc[1:].reset_index(drop=True)
    bat = s["electrical_storage"]["attributes"]
    meta = dict(pv_kw=pv_kw, cap_kwh=bat["capacity"], p_max=bat["nominal_power"])
    return df, meta


def split_masks(df):
    m = df.month.values
    return (np.isin(m, TRAIN_MONTHS), np.isin(m, CALIB_MONTHS), np.isin(m, TEST_MONTHS))


def building_stats(names):
    rows = []
    for n in names:
        df, meta = load_building(n)
        tr, _, _ = split_masks(df)
        x = df[tr]
        rows.append(dict(building=n, pv_kw=meta["pv_kw"], surplus_share=(x.net < 0).mean(),
                         load_std=x.load.std(), net_mean=x.net.mean(), pv_mean=x.pv.mean()))
    return pd.DataFrame(rows).set_index("building")


def select_buildings():
    """Archetype selection carried over from the preliminary study (donor / balanced / spiky),
    computed on TRAINING months only. Buildings without PV are excluded."""
    sch = load_schema()["buildings"]
    names = [n for n in sch if sch[n].get("pv")]
    st = building_stats(names)
    st = st[st.pv_mean > 0.1]              # drop homes with (almost) no PV
    zero = {}
    for n in st.index:                     # screened on TRAINING months only
        d_, _ = load_building(n); tr_, _, _ = split_masks(d_)
        zero[n] = (d_.load[tr_] <= 1e-3).mean()
    st["zero_load_share"] = pd.Series(zero)
    st = st[st.zero_load_share <= 0.02]    # data-quality filter: <=2% zero-load (meter-gap) hours
    donor = st[st.net_mean < st.net_mean.median()].surplus_share.idxmax()
    rest = st.drop(index=donor)
    spiky = rest.load_std.idxmax()
    rest = rest.drop(index=spiky)
    z = (rest - st.mean()) / st.std()
    balanced = (z[["surplus_share", "load_std", "net_mean"]] ** 2).sum(axis=1).idxmin()
    return {"donor": donor, "balanced": balanced, "spiky": spiky}, st


if __name__ == "__main__":
    sel, st = select_buildings()
    print(st.round(3).to_string())
    print(sel)
    json.dump(sel, open(os.path.join(OUT, "selected.json"), "w"), indent=1)
