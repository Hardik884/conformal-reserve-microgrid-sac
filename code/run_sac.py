"""Train one SAC agent (variant x building x seed) with a fixed budget, then evaluate the final
policy on the held-out test months. April (calibration month) is used only for a diagnostic
learning curve, never for model selection."""
import os, sys, json, time
import numpy as np, torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from env import HomeData, MicrogridEnv, VARIANTS
from common import *

torch.set_num_threads(1)
STEPS = int(os.environ.get("SAC_STEPS", 40000))
HP = dict(learning_rate=3e-4, buffer_size=200_000, batch_size=128, learning_starts=2_000, train_freq=1,
          gradient_steps=1, gamma=0.99, tau=0.005, ent_coef="auto", policy_kwargs=dict(net_arch=[128, 128]))


def evaluate(model, home, variant, mode="test", **kw):
    env = MicrogridEnv(home, variant, mode=mode, **kw)
    obs, _ = env.reset(); log = []
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        log.append(info)
        if term or trunc:
            break
    c = lambda k: float(sum(x[k] for x in log))
    return dict(cost=c("cost"), energy_cost=c("energy_cost"), viol=c("viol"), over_kwh=c("over"),
                imp=c("imp"), exp=c("exp"), peak=float(max(x["imp"] for x in log)),
                throughput=c("pc") + c("pd"), hours=len(log)), log


class ValCurve(BaseCallback):
    def __init__(self, home, variant, every=10_000):
        super().__init__(); self.home, self.variant, self.every, self.curve = home, variant, every, []
    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            s, _ = evaluate(self.model, self.home, self.variant, mode="val")
            self.curve.append(dict(step=self.num_timesteps, cost=s["cost"], viol=s["viol"]))
        return True


def run(variant, building, seed):
    tag = f"{variant}__{building}__s{seed}"
    path = os.path.join(OUT, "sac", tag + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    home = HomeData(building)
    env = MicrogridEnv(home, variant, mode="train", seed=seed)
    model = SAC("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **HP)
    cb = ValCurve(home, variant)
    t0 = time.time()
    model.learn(STEPS, callback=cb)
    res = dict(variant=variant, building=building, seed=seed, steps=STEPS, train_sec=time.time() - t0,
               val_curve=cb.curve)
    res["test"], log = evaluate(model, home, variant, "test")
    np.savez_compressed(os.path.join(OUT, "sac", tag + "_log.npz"),
                        **{k: np.array([x[k] for x in log]) for k in ["t", "cost", "viol", "over", "soc", "g_lo", "g_hi", "imp"]})
    # evaluation-time swaps (no retraining): ACI intervals, and the reserve layer on/off with other bounds
    extra = {}
    feat, layer = VARIANTS[variant]
    if feat in ("cqr",):
        extra["aci_inputs"] = evaluate(model, home, variant, "test", interval_src="aci")[0]
    if variant == "SAC-CQR":
        for src in ["med", "raw", "cqr", "aci"]:
            extra["posthoc_layer_" + src] = evaluate(model, home, variant, "test", layer_src=src)[0]
    if variant in ("SAC", "SAC-P"):
        for src in ["cqr", "aci"]:
            extra["posthoc_layer_" + src] = evaluate(model, home, variant, "test", layer_src=src)[0]
    if variant == "SAC-CQR-S":
        extra["aci_inputs_aci_layer"] = evaluate(model, home, variant, "test", interval_src="aci", layer_src="aci")[0]
        extra["layer_off"] = evaluate(model, home, variant, "test", layer_src="none")[0]
    res["extra"] = extra
    model.save(os.path.join(OUT, "sac", tag + "_model"))
    json.dump(res, open(path, "w"), indent=1)
    return res


if __name__ == "__main__":
    v, b, s = sys.argv[1], sys.argv[2], int(sys.argv[3])
    r = run(v, b, s)
    print(json.dumps({k: r[k] for k in ["variant", "building", "seed", "train_sec"]}), json.dumps(r["test"]), r["val_curve"])
