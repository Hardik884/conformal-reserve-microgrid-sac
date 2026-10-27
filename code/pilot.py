"""Design pilot on the TRAINING home's April validation month only (no test data touched)."""
import os, sys, json, time, torch
from stable_baselines3 import SAC
from env import HomeData, MicrogridEnv
from run_sac import HP, ValCurve, evaluate
torch.set_num_threads(1)
variant, building, seed, steps, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
home = HomeData(building)
env = MicrogridEnv(home, variant, mode="train", seed=seed)
hp = dict(HP)
if os.environ.get("NET"): hp["policy_kwargs"] = dict(net_arch=[int(x) for x in os.environ["NET"].split(",")])
if os.environ.get("BS"): hp["batch_size"] = int(os.environ["BS"])
model = SAC("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **hp)
cb = ValCurve(home, variant, every=10000); t0 = time.time(); model.learn(steps, callback=cb)
v, _ = evaluate(model, home, variant, "val")
out = dict(tag=tag, variant=variant, building=building, env=dict(R=os.environ.get("R_SCALE"), D=os.environ.get("DAYAHEAD")),
           sec=time.time() - t0, curve=cb.curve, val=v)
json.dump(out, open(f"../out/pilot/{tag}.json", "w"), indent=1); print(json.dumps(out))
tr, _ = evaluate(model, home, variant, "trainfull")
print("TRAINFULL", json.dumps(tr))
