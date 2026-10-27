"""Publication figures (PNG 300 dpi + PDF) from the saved result files."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from common import OUT
FIG = "../figures"
BLUE, GREY, INK, QUIET = "#2a78d6", "#8a8f96", "#1f1f1e", "#5f5e5a"
GOOD, BAD = "#0ca30c", "#d03b3b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#bdbdb8",
                     "axes.labelcolor": QUIET, "xtick.color": QUIET, "ytick.color": QUIET,
                     "axes.spines.top": False, "axes.spines.right": False})
HOMES = [("Building_8", "Donor (B8)"), ("Building_2", "Balanced (B2)"), ("Building_10", "Spiky (B10)")]
def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Figure 2: coverage
T = {r["b"]: r for r in json.load(open(f"{OUT}/table_homes_fc.json"))}
fig, ax = plt.subplots(figsize=(6.5, 3.0))
meth = [("raw", "Raw QR", GREY, 0.45), ("cqr", "Static CQR", GREY, 1.0), ("aci", "ACI", BLUE, 1.0)]
yt, yl = [], []
for i, (b, lab) in enumerate(HOMES):
    for j, (m, ml, c, a) in enumerate(meth):
        y = -(i * 4 + j); v = T[b][f"{m}_cov_mean"]
        ax.hlines(y, 0.5, v, color=c, alpha=a * 0.6, lw=1.2); ax.plot(v, y, "o", color=c, alpha=a, ms=6)
        ax.text(v + (0.008 if v > 0.89 else -0.008), y + 0.35, f"{v*100:.1f}%", ha="left" if v > 0.89 else "right", fontsize=8, color=INK if m == "aci" else QUIET)
        yt.append(y); yl.append(f"{lab} · {ml}" if j == 1 else ml)
ax.axvline(0.9, color="#8a8a86", ls="--", lw=1); ax.text(0.902, 0.6, "90% target", fontsize=8, color=QUIET)
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8); ax.set_xlim(0.5, 1.0)
ax.set_xticks(np.arange(0.5, 1.01, 0.1)); ax.set_xticklabels([f"{round(x*100)}%" for x in np.arange(0.5, 1.01, 0.1)])
ax.set_xlabel("Empirical coverage of nominal 90% intervals, test months (mean over 24 horizons)")
ax.set_title("Only ACI holds the 90% target in every home", loc="left", fontsize=10, color=INK, fontweight="bold")
save(fig, "fig2_coverage")

# ---------- Figure 3: risk dial
D = json.load(open(f"{OUT}/dial.json")); B = json.load(open(f"{OUT}/baselines.json"))
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
for ax, (b, lab) in zip(axs, HOMES):
    for src, c, ml in [("cqr", GREY, "static CQR"), ("aci", BLUE, "ACI")]:
        al = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
        xs = [D[f"{b}|{src}|{a}|RBC-SC"]["viol"] for a in al]; ys = [D[f"{b}|{src}|{a}|RBC-SC"]["cost"] for a in al]
        ax.plot(xs, ys, "-o", color=c, ms=3.5, lw=1.6, label=ml)
        if src == "aci":
            for a, x, y in zip(al, xs, ys):
                if a in (0.02, 0.1, 0.5): ax.annotate(str(a), (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7, color=INK)
    nb = B[f"{b}|RBC-SC"]; ax.plot(nb["viol"], nb["cost"], "o", mfc="none", mec="#7a7a76", ms=6)
    ax.annotate("no layer", (nb["viol"], nb["cost"]), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7, color=QUIET)
    ax.set_title(lab, fontsize=9, color=INK, loc="left"); ax.set_xlim(0, 200); ax.set_xlabel("violation hours")
axs[0].set_ylabel("total test cost ($)"); axs[1].legend(frameon=False, fontsize=7, loc="lower right")
fig.suptitle("A stricter conformal level buys fewer violations for little extra cost (RBC-SC + reserve layer)", x=0.01, ha="left", fontsize=9.5, color=INK, fontweight="bold")
fig.tight_layout(); save(fig, "fig3_risk_dial")

# ---------- Figure 4: paired effects
S = json.load(open(f"{OUT}/stats.json")); P = {(r["a"], r["b"], r["metric"]): r for r in S["paired"] + S["paired_extra"]}
eff = [("Point forecasts in the state", "SAC-P", "SAC"), ("Raw intervals added to the state", "SAC-QR", "SAC-P"),
       ("State intervals conformalised", "SAC-CQR", "SAC-QR"), ("Reserve layer used in training", "SAC-CQR-S", "SAC-CQR"),
       ("ACI layer added after training", "SAC-P:posthoc_layer_aci", "SAC-P"),
       ("ACI instead of static CQR at test", "SAC-CQR-S:aci_inputs_aci_layer", "SAC-CQR-S")]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, m, xl in [(axs[0], "cost", "total cost change ($)"), (axs[1], "viol", "violation-hour change")]:
    for i, (lab, a, b) in enumerate(eff):
        r = P[(a, b, m)]; lo, hi = r["ci"]; c = GOOD if hi < 0 else BAD if lo > 0 else GREY
        ax.hlines(-i, lo, hi, color=c, lw=2); ax.plot(r["mean_diff"], -i, "o", color=c, ms=6)
    ax.axvline(0, color="#8a8a86", lw=1); ax.set_xlabel(xl)
axs[0].set_yticks([-i for i in range(len(eff))]); axs[0].set_yticklabels([e[0] for e in eff], fontsize=8)
fig.suptitle("Calibrated intervals helped through the reserve layer, not as extra state", x=0.01, ha="left", fontsize=10, color=INK, fontweight="bold")
fig.text(0.01, -0.04, "Mean paired change over 15 (home, seed) pairs, bootstrap 95% CI. Green: CI below zero (better); red: above zero (worse); grey: crosses zero.", fontsize=7, color=QUIET)
fig.tight_layout(); save(fig, "fig4_effects")

# ---------- Figure 1: architecture
fig, ax = plt.subplots(figsize=(7.2, 3.6)); ax.set_xlim(0, 760); ax.set_ylim(370, 0); ax.axis("off")
def box(x, y, w, h, name, lines, hot=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8", fc="#e6effa" if hot else "white", ec=BLUE if hot else "#9a9a96", lw=1.6 if hot else 1))
    ax.text(x + w / 2, y + 20, name, va="center", ha="center", fontsize=7.6, fontweight="bold", color=INK)
    for k, l in enumerate(lines): ax.text(x + w / 2, y + 42 + 15 * k, l, ha="center", va="center", fontsize=6.6, color=INK if hot else QUIET)
box(32, 56, 208, 88, "Inputs at hour t", ["24 h of net load, load, PV", "weather predictions", "and calendar"])
box(280, 56, 208, 88, "Quantile CNN-LSTM", ["median plus 5% and 95%", "quantiles, 24 h ahead", "(non-crossing head)"])
box(528, 56, 208, 88, "Conformal calibration", ["static CQR (April scores)", "or online ACI", "90% intervals"], True)
box(32, 208, 160, 88, "SAC policy", ["SoC, prices, lags,", "median, interval", "out: reserve target"])
box(216, 208, 160, 88, "Reserve layer", ["keeps energy for", "predicted peaks", "above P_cap"], True)
box(400, 208, 160, 88, "Inverter tracker", ["follows realised", "load above the", "reserve floor"])
box(584, 208, 160, 88, "Home", ["PV, load, battery", "grid import", "billed hourly"])
arr = dict(arrowstyle="-|>", color="#7a7a76", lw=1)
for (x1, y1, x2, y2) in [(240, 100, 280, 100), (488, 100, 528, 100), (192, 252, 216, 252), (376, 252, 400, 252), (560, 252, 584, 252), (112, 176, 112, 208), (296, 176, 296, 208), (16, 100, 32, 100)]:
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=arr)
ax.plot([632, 632, 112], [144, 176, 176], color="#7a7a76", lw=1); ax.plot([664, 664, 16, 16], [296, 328, 328, 100], color="#7a7a76", lw=1)
ax.text(120, 198, "intervals into the state", fontsize=7, color=QUIET); ax.text(304, 198, "upper bounds U", fontsize=7, color=QUIET)
ax.text(340, 348, "measured net load updates the history and the ACI level", ha="center", fontsize=7, color=QUIET)
ax.text(32, 30, "Calibrated intervals feed both the policy and the reserve layer", fontsize=10, fontweight="bold", color=INK)
save(fig, "fig1_architecture")
print("ok")
