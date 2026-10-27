"""Re-price RBC-SC rollouts (with / without the ACI reserve layer) under other penalties and export credits."""
import json
from env import HomeData
from baselines import make_policies, rollout
from common import OUT
out = {}
for b in ["Building_8", "Building_2", "Building_10"]:
    h = HomeData(b); p = make_policies(h)
    L0 = rollout(h, p["RBC-SC"]); L1 = rollout(h, p["RBC-SC"], "aci")
    for lam in [0.5, 1.0, 2.0]:
        for fi in [0.0, 0.05, 0.10]:
            c = lambda L: sum(h.price[x["t"]] * x["imp"] - fi * x["exp"] + lam * x["over"] for x in L)
            a, bb = c(L0), c(L1); out[f"{b}|{lam}|{fi}"] = dict(rbc=a, layer=bb, change=(bb / a - 1) * 100)
json.dump(out, open(OUT + "/sensitivity.json", "w"), indent=1)
