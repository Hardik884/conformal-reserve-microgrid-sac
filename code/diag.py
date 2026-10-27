import numpy as np, torch, json
from stable_baselines3 import SAC
from env import HomeData, MicrogridEnv
from run_sac import HP
torch.set_num_threads(1)
h=HomeData("Building_8"); env=MicrogridEnv(h,"SAC-P",mode="train",seed=0)
hp=dict(HP); m=SAC("MlpPolicy",env,seed=0,device="cpu",verbose=0,**hp); m.learn(20000)
m.save("../out/pilot/diag_sacp")
ev=MicrogridEnv(h,"SAC-P",mode="val"); o,_=ev.reset(); L=[]
while True:
    a,_=m.predict(o,deterministic=True); o,r,te,tr,info=ev.step(a); info["a"]=a.tolist(); L.append(info)
    if te or tr: break
hr=np.array([h.hour[x["t"]] for x in L])
for k in ["g_lo","g_hi","soc","imp","pc","pd"]:
    v=np.array([x[k] for x in L]); print(k, np.round([v[hr==i].mean() for i in range(24)],2).tolist())
print("cost", sum(x["cost"] for x in L), "energy", sum(x["energy_cost"] for x in L), "viol", sum(x["viol"] for x in L), "exp", sum(x["exp"] for x in L), "imp", sum(x["imp"] for x in L))
