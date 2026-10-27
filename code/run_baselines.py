"""All training-free controllers on the test months, with and without the reserve layer."""
import json
from env import HomeData
from baselines import make_policies, rollout, oracle, summarize
from common import *
out = {}
for b in ["Building_8", "Building_2", "Building_10"]:
    h = HomeData(b); pols = make_policies(h)
    for k, pol in pols.items():
        for lay in [None, "med", "raw", "cqr", "aci"]:
            if k == "NoBattery" and lay: continue
            if k in ("MPC-CQR", "MPC-ACI") and lay not in (None, k.split("-")[1].lower()): continue
            name = k + ("" if lay is None else "+L(" + lay + ")")
            out[f"{b}|{name}"] = summarize(rollout(h, pol, lay))
            print(b, name, round(out[f"{b}|{name}"]["cost"], 1), out[f"{b}|{name}"]["viol"], flush=True)
    out[f"{b}|Oracle"] = summarize(oracle(h))
json.dump(out, open(os.path.join(OUT, "baselines.json"), "w"), indent=1)
