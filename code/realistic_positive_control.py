"""
realistic_positive_control.py  (reviewer point 7)

A more realistic synthetic positive control than a pure Gaussian model:
the data contain a dominant global size mode, gene-module structure, a
per-cell size factor and Poisson count noise. We inject a gene-specific
response of controlled strength (mixed with the global mode) and ask whether
the pipeline recovers the specific (global-mode-orthogonal) component.

If the pipeline recovers the specific component in this realistic setting, the
main analysis is not simply "blind to any signal".

Output: realistic_positive_control.png, realistic_positive_control.csv
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def simulate(n_genes=400, n_cells=3000, n_modules=8, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.exp(rng.normal(1.5, 0.8, n_genes))
    M = rng.normal(size=(n_genes, n_modules)) * (rng.random((n_genes, n_modules)) < 0.15)
    Z = rng.normal(size=(n_cells, n_modules))
    s = np.exp(rng.normal(0, 0.35, n_cells))            # size factor -> global mode
    lam = s[:, None] * mu[None, :] * np.exp(0.6 * (Z @ M.T))
    X = rng.poisson(np.clip(lam, 1e-6, None)).astype(np.float32)
    return X, mu, M, s, Z, lam, rng


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else float("nan")


def run(alpha, seed=0, n_pert=60):
    X, mu, M, s, Z, lam, rng = simulate(seed=seed)
    Xc = X - X.mean(0)
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    v = V[:, np.argmax(w)]; v = v / np.linalg.norm(v)
    gfrac = float(w.max() / w.sum())
    # true (population) covariance from the model
    Lc = lam - lam.mean(0)
    cov_true = (Lc.T @ Lc) / (Lc.shape[0] - 1)

    n_genes = X.shape[1]
    pert = rng.choice(n_genes, n_pert, replace=False)
    cs = []
    for g in pert:
        # gene-specific direction = the TRUE covariance column of g (orthogonalised)
        spec = cov_true[:, g].astype(float)
        spec = spec - (spec @ v) * v
        spec = spec / (np.linalg.norm(spec) + 1e-9)
        # response = alpha * true linear response + (1-alpha) * noise
        noise = rng.normal(size=n_genes)
        noise = noise - (noise @ v) * v
        d = alpha * spec + (1 - alpha) * 0.5 * noise
        d = d - (d @ v) * v
        # pipeline prediction: ESTIMATED covariance column of g (orthogonalised)
        sig = cov[:, g]; sig = sig - (sig @ v) * v
        dd = float(sig @ sig)
        a = float(sig @ d / dd) if dd else 0
        cs.append(cos(d, a * sig))
    return float(np.nanmean(cs)), gfrac


def main():
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    real, shuf = [], []
    for a in alphas:
        vals = [run(a, seed=s)[0] for s in range(5)]
        real.append(np.mean(vals))
    gfrac = run(0.3, seed=0)[1]

    with open("realistic_positive_control.csv", "w") as fh:
        fh.write("alpha,specific_corr\n")
        for a, r in zip(alphas, real):
            fh.write("{:.1f},{:.4f}\n".format(a, r))

    plt.figure(figsize=(7, 5))
    plt.plot(alphas, real, "o-")
    plt.xlabel("injected gene-specific signal strength $\\alpha$")
    plt.ylabel("recovered specific correlation")
    plt.title("Realistic positive control (global mode + count noise)\n"
              "global-mode fraction = {:.2f}".format(gfrac))
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("realistic_positive_control.png", dpi=150)
    print("global-mode fraction:", round(gfrac, 3))
    print("alpha:", alphas)
    print("recovered:", ["{:.3f}".format(x) for x in real])
    print("wrote realistic_positive_control.png, realistic_positive_control.csv")


if __name__ == "__main__":
    main()
