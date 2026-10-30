"""Motor-imagery BCI: band-pass + Common Spatial Patterns (CSP) + LDA on simulated EEG.

Simulation: 16 channels, 250 Hz, 3 s trials. Left/right imagery suppresses the 10 Hz mu rhythm
(event-related desynchronisation) over the contralateral sensorimotor source. Sources are mixed into
channels through a random mixing matrix with strong ongoing alpha, 1/f-like background and sensor noise.
Evaluation: stratified 10-fold CV with CSP fitted inside each training fold (no leakage).
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold

FS, NCH, T = 250, 16, 3.0

def simulate(n_per_class=100, erd=0.3, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(FS * T); t = np.arange(n) / FS
    A = rng.standard_normal((NCH, 8))                       # mixing: 2 mu sources + 6 background
    X, y = [], []
    for label in (0, 1):
        for _ in range(n_per_class):
            src = np.zeros((8, n)); trial_gain = np.exp(0.4 * rng.standard_normal())   # trial-to-trial variability
            for k in range(2):                               # k=0 left-hemisphere (C3), k=1 right (C4)
                ph = rng.uniform(0, 2 * np.pi)
                f = 10 + 0.4 * rng.standard_normal()
                amp = trial_gain * ((1 - erd) if k == (1 - label) else 1.0)   # imagery of hand `label` suppresses contralateral
                src[k] = 3 * amp * np.sin(2 * np.pi * f * t + ph) + 0.5 * rng.standard_normal(n)
            for k in range(2, 8):                            # background: alpha-ish + noise, independent of class
                ph = rng.uniform(0, 2 * np.pi)
                src[k] = 2 * np.sin(2 * np.pi * (9 + 3 * rng.random()) * t + ph) + rng.standard_normal(n)
            X.append(A @ src + noise * 2 * rng.standard_normal((NCH, n))); y.append(label)
    return np.array(X), np.array(y)

def bandpass(X, lo=8, hi=30):
    return sosfiltfilt(butter(4, [lo, hi], "bandpass", fs=FS, output="sos"), X, axis=-1)

def cov(X):
    C = np.stack([x @ x.T / np.trace(x @ x.T) for x in X]); return C.mean(0)

def fit_csp(X, y, n_pairs=3):
    C0, C1 = cov(X[y == 0]), cov(X[y == 1])
    w, V = eigh(C0, C0 + C1)                                 # generalized eigenproblem
    idx = np.r_[np.arange(n_pairs), np.arange(len(w) - n_pairs, len(w))]
    return V[:, idx].T

def feats(W, X):
    Z = np.einsum("fc,ect->eft", W, X); v = Z.var(-1)
    return np.log(v / v.sum(1, keepdims=True))

def cv_accuracy(X, y, use_csp=True, seed=0):
    accs = []
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=seed).split(X, y):
        if use_csp:
            W = fit_csp(X[tr], y[tr]); Ftr, Fte = feats(W, X[tr]), feats(W, X[te])
        else:                                                # baseline: per-channel log band power
            Ftr, Fte = np.log(X[tr].var(-1)), np.log(X[te].var(-1))
        accs.append(LDA().fit(Ftr, y[tr]).score(Fte, y[te]))
    return float(np.mean(accs))

if __name__ == "__main__":
    out = {}
    for noise in (3.0, 6.0, 10.0):
        rows = []
        for seed in range(5):
            X, y = simulate(noise=noise, seed=seed); Xf = bandpass(X)
            rows.append((cv_accuracy(Xf, y, True, seed), cv_accuracy(Xf, y, False, seed)))
        a = np.array(rows)
        out[f"sensor_noise_{noise}"] = dict(csp_lda=float(a[:, 0].mean()), csp_sd=float(a[:, 0].std()),
                                           channel_logpower_lda=float(a[:, 1].mean()))
        print(noise, out[f"sensor_noise_{noise}"])
    json.dump(out, open("results.json", "w"), indent=2)
    ks = list(out); fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar(np.arange(3) - 0.2, [out[k]["csp_lda"] for k in ks], 0.4, label="CSP + LDA")
    ax.bar(np.arange(3) + 0.2, [out[k]["channel_logpower_lda"] for k in ks], 0.4, label="channel log-power + LDA")
    ax.axhline(0.5, color="k", lw=0.7, ls="--"); ax.set_xticks(range(3)); ax.set_xticklabels([k.split("_")[-1] for k in ks])
    ax.set_xlabel("sensor noise level"); ax.set_ylabel("10-fold CV accuracy"); ax.legend(fontsize=7); plt.tight_layout()
    plt.savefig("accuracy.png", dpi=150)
