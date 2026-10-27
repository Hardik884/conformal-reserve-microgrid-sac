"""Pull jobs (variant, home, seed) from a fixed list; resume-safe via result files + lock files."""
import os, sys, time, json, itertools
from common import OUT
VARS = ["SAC", "SAC-P", "SAC-QR", "SAC-CQR", "SAC-CQR-S"]
HOMES = ["Building_8", "Building_2", "Building_10"]
jobs = [(v, b, s) for s in range(5) for b in HOMES for v in VARS]
os.makedirs(os.path.join(OUT, "sac"), exist_ok=True)
import run_sac
for v, b, s in jobs:
    tag = f"{v}__{b}__s{s}"
    res = os.path.join(OUT, "sac", tag + ".json"); lock = res + ".lock"
    if os.path.exists(res):
        continue
    if os.path.exists(lock) and time.time() - os.path.getmtime(lock) < 1800:
        continue
    open(lock, "w").write(str(os.getpid()))
    t0 = time.time()
    r = run_sac.run(v, b, s)
    os.remove(lock)
    print(f"{tag} cost {r['test']['cost']:.1f} viol {r['test']['viol']:.0f} ({time.time()-t0:.0f}s)", flush=True)
