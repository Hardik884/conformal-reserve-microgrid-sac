"""Validation-only hyper-parameter selection for the SAC-P baseline (April of each home).
The chosen configuration is then frozen for every SAC variant."""
import os, sys, json, time, torch
from stable_baselines3 import SAC
from env import HomeData, MicrogridEnv
from run_sac import HP, evaluate
torch.set_num_threads(1)
CONFIGS = {
    "A_base":   dict(),
    "B_r1":     dict(_R="1"),
    "C_g95":    dict(gamma=0.95),
    "D_lr1e3":  dict(learning_rate=1e-3),
    "E_tent4":  dict(target_entropy=-4.0),
    "F_g95_r1": dict(gamma=0.95, _R="1"),
}
os.makedirs("../out/tune", exist_ok=True)
for name in sys.argv[1:]:
    cfg = CONFIGS[name]
    for b in ["Building_8", "Building_2", "Building_10"]:
        path = f"../out/tune/{name}__{b}.json"
        if os.path.exists(path):
            continue
        os.environ["R_SCALE"] = cfg.get("_R", "10")
        hp = dict(HP); hp.update({k: v for k, v in cfg.items() if not k.startswith("_")})
        home = HomeData(b)
        env = MicrogridEnv(home, "SAC-P", mode="train", seed=0)
        m = SAC("MlpPolicy", env, seed=0, device="cpu", verbose=0, **hp)
        t0 = time.time(); m.learn(40000)
        v, _ = evaluate(m, home, "SAC-P", "val")
        json.dump(dict(cfg=name, building=b, val=v, sec=time.time() - t0), open(path, "w"))
        print(name, b, round(v["cost"], 2), v["viol"], flush=True)
