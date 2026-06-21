"""Generate summary figures for the spectral-library / aerosol report.
All data are the recorded results from the session's experiments (hardcoded),
so this just renders — no experiments re-run."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/users/marcyin/surface_prior/spectral_aerosol_report"
plt.rcParams.update({"figure.dpi": 110, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}", bbox_inches="tight"); plt.close(fig)


# F1 — library intrinsic dimensionality (hyperspectral VNIR cumulative variance)
k = [1, 2, 3, 4, 5, 6, 8]
cum = [85.96, 97.16, 99.07, 99.73, 99.89, 99.93, 99.97]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(k, cum, "o-", color="#2c7fb8", lw=2)
ax.axhline(99.7, ls="--", color="grey", lw=1)
for x, y in zip(k, cum):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -14), fontsize=8, ha="center")
ax.set(xlabel="number of principal components", ylabel="cumulative surface variance (%)",
       title="Library intrinsic dimensionality (hyperspectral VNIR, 77k spectra)", ylim=(84, 100.3))
save(fig, "f1_intrinsic_dim.png")

# F2 — kNN source_fit_rmse, HLS vs S2, per band (decomposition)
bands = ["coastal", "blue", "green", "red", "nir", "swir16", "swir22"]
s2 = [76.0, 59.7, 47.0, 37.0, 34.8, 29.3, 30.9]
hls = [34.9, 23.9, 49.9, 31.2, 30.7, 28.4, 28.2]
x = np.arange(len(bands)); w = 0.38
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - w/2, s2, w, label="S2 L2A", color="#d95f02")
ax.bar(x + w/2, hls, w, label="HLS", color="#1b9e77")
ax.set_xticks(x); ax.set_xticklabels(bands, rotation=20)
ax.set(ylabel="kNN reconstruction residual (DN)",
       title="How well the spectral library explains the surface (per band)\nS2 blue/coastal are 2-3x harder to fit than HLS")
ax.legend()
save(fig, "f2_library_fit_perband.png")

# F6 — LOYO localization, blue prediction RMSE per config
cfgs = ["global\nlibrary", "LOYO-Jul", "LOYO\nall-month", "same-year\nrest", "all-but\ntarget"]
blue = [121, 42, 42, 43, 42]
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(cfgs, blue, color=["#7570b3"] + ["#1b9e77"]*4)
for b, v in zip(bars, blue):
    ax.annotate(f"{v}", (b.get_x()+b.get_width()/2, v), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=9)
ax.set(ylabel="blue prediction RMSE (DN)",
       title="Localization (leave-one-year-out): any scene-local pool ties\nglobal library is the only thing that's worse")
save(fig, "f6_localization.png")

# F7 — AOD retrieval: retrieved vs true, non-iterated vs iterated (the key result)
truth = [0.10, 0.30, 0.50, 0.80]
noniter = [0.02, 0.22, 0.46, 0.78]
itr = [0.10, 0.30, 0.50, 0.78]
fig, ax = plt.subplots(figsize=(6, 5.2))
ax.plot([0, 0.9], [0, 0.9], "k--", lw=1, label="1:1 (perfect)")
ax.plot(truth, noniter, "s-", color="#d95f02", lw=2, label="raw anchor (biased low)")
ax.plot(truth, itr, "o-", color="#1b9e77", lw=2, label="anchor-iterated (unbiased)")
ax.set(xlabel="true AOD550", ylabel="retrieved AOD550 (60-cell pooled)",
       title="Closed-loop AOD retrieval\nscene-local dictionary + red+nir+swir + anchor iteration + pooling",
       xlim=(0, 0.9), ylim=(0, 0.9), aspect="equal")
ax.legend(loc="upper left")
save(fig, "f7_aod_retrieval.png")

# F8 — low-AOD selection null: AOD vs surface cleanliness, Egypt + IGP
eg_a = [0.139, 0.142, 0.150, 0.151, 0.152, 0.193, 0.200, 0.231, 0.233, 0.240, 0.305, 0.335]
eg_f = [35.2, 38.7, 34.2, 34.3, 39.6, 33.4, 42.7, 36.3, 41.1, 35.0, 30.7, 37.2]
ig_a = [0.283, 0.296, 0.357, 0.370, 0.455, 0.504, 0.561, 0.633, 0.735, 0.769, 0.900, 1.060]
ig_f = [92.6, 104.4, 44.1, 68.2, 46.6, 93.4, 84.6, 59.6, 79.7, 85.4, 62.1, 85.6]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a, (xa, ya, ti, cr) in enumerate([(eg_a, eg_f, "Nile Delta, Jul 2020 (r=-0.15)", "#2c7fb8"),
                                       (ig_a, ig_f, "Indo-Gangetic Plain, Nov 2022 (r=-0.01)", "#d95f02")]):
    ax[a].scatter(xa, ya, color=cr, s=45)
    z = np.polyfit(xa, ya, 1); xs = np.array([min(xa), max(xa)])
    ax[a].plot(xs, np.polyval(z, xs), "k--", lw=1)
    ax[a].set(xlabel="AOD550 (CAMS+MERRA)", ylabel="surface library residual (DN)", title=ti)
fig.suptitle("Does haze make the L2A surface dirtier? — No AOD correlation, even at AOD~1", y=1.02)
save(fig, "f8_aod_selection_null.png")

# F9 — deep-blue (443) SNR: aerosol signal vs surface-prediction noise
vb = ["coastal\n(443)", "blue\n(490)", "green\n(560)"]
sig = [899, 572, 257]; noise = [117, 117, 161]; snr = [7.7, 4.9, 1.6]
x = np.arange(3); w = 0.38
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - w/2, sig, w, label="aerosol signal @AOD0.3", color="#d95f02")
ax.bar(x + w/2, noise, w, label="surface-prediction noise", color="#7570b3")
for i in range(3):
    ax.annotate(f"SNR {snr[i]}", (i, max(sig[i], noise[i])), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=9, weight="bold")
ax.set_xticks(x); ax.set_xticklabels(vb)
ax.set(ylabel="DN", title="The 443 'deep-blue analog' band has the best aerosol SNR")
ax.legend()
save(fig, "f9_deepblue_snr.png")

# F10 — pluggable AOD prior: retrieval by mode + prior-source on a gradient
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
truth = [0.10, 0.30, 0.50, 0.80]
raw = [0.02, 0.22, 0.46, 0.78]; pri = [0.10, 0.30, 0.50, 0.82]; itr = [0.10, 0.30, 0.50, 0.78]
ax[0].plot([0, 0.9], [0, 0.9], "k--", lw=1, label="1:1")
ax[0].plot(truth, raw, "s-", color="#d95f02", label="raw TOA anchor (biased)")
ax[0].plot(truth, pri, "o-", color="#1b9e77", label="AOD-prior pre-correction")
ax[0].plot(truth, itr, "^--", color="#7570b3", label="full iteration")
ax[0].set(xlabel="true AOD550", ylabel="retrieved AOD550", title="Anchor pre-correction = full iteration",
          xlim=(0, 0.9), ylim=(0, 0.9), aspect="equal"); ax[0].legend(fontsize=8, loc="upper left")
srcs = ["TOA\n(none)", "coarse\n(CAMS/MERRA)", "coarse\n+0.1", "highres\n(MAIAC)", "oracle"]
rmse = [0.074, 0.041, 0.045, 0.039, 0.039]
bars = ax[1].bar(srcs, rmse, color=["#d95f02", "#1b9e77", "#66a61e", "#1f78b4", "#7570b3"])
for b, v in zip(bars, rmse):
    ax[1].annotate(f"{v:.3f}", (b.get_x()+b.get_width()/2, v), textcoords="offset points",
                   xytext=(0, 3), ha="center", fontsize=8)
ax[1].set(ylabel="AOD pooled RMSE (gradient scene)", title="Prior SOURCE barely matters\n(the solve provides the resolution)")
fig.suptitle("Pluggable AOD prior: any rough source pre-corrects the anchor; coarse model AOD suffices", y=1.03)
save(fig, "f10_aod_prior.png")

print("figures written to", OUT)
