"""
positive_control.py

Validates that the pipeline CAN detect a perturbation-specific signal when one
is present. Synthetic data:

  * control cells are drawn from a covariance Sigma with a dominant global mode
    plus gene-module structure
  * for each perturbation g the "true" response is a mixture
        dX_g = alpha * Sigma[:,g] + (1-alpha) * noise
  * alpha = 1  -> response lies in the covariance structure (should be detected)
  * alpha = 0  -> pure noise (should give ~0)
  * a shuffled control uses Sigma[:,g'] from a DIFFERENT gene

Reports the orthogonal specific correlation vs alpha. If the pipeline is sound,
it should rise with alpha and be ~0 for the shuffled control.

Output: positive_control.png, positive_control.csv
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else float("nan")


def simulate(n_genes=500, n_cells=2000, n_pert=80, n_modules=10, seed=0):
    rng = np.random.default_rng(seed)
    # global mode (all-positive direction) + modules
    g = rng.random(n_genes) + 0.5
    v = g / np.linalg.norm(g)
    M = rng.normal(size=(n_genes, n_modules)) * (rng.random((n_genes, n_modules)) < 0.2)
    Sigma = 0.7 * np.outer(v, v) + 0.3 * (M @ M.T) / n_modules
    Sigma += 0.1 * np.eye(n_genes)
    X = rng.multivariate_normal(np.zeros(n_genes), Sigma, size=n_cells)
    return Sigma, X, v, rng


def run(alpha, shuffled=False, seed=0):
    Sigma, X, v, rng = simulate(seed=seed)
    Xc = X - X.mean(0)
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    v_hat = V[:, np.argmax(w)]; v_hat = v_hat / np.linalg.norm(v_hat)

    n = Sigma.shape[0]
    pert = rng.choice(n, size=80, replace=False)
    cs = []
    for i, g in enumerate(pert):
        target = Sigma[:, g]
        noise = rng.normal(scale=np.linalg.norm(target) / 3, size=n)
        d = alpha * target + (1 - alpha) * noise
        if shuffled:
            g2 = pert[(i + 1) % len(pert)]
            sig = cov[:, g2]
        else:
            sig = cov[:, g]
        d = d - float(d @ v_hat) * v_hat
        dd = float(sig @ sig)
        a = float(sig @ d / dd) if dd else 0
        cs.append(pearson(d, a * sig))
    return float(np.nanmean(cs))


def main():
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    real = [run(a) for a in alphas]
    shuf = [run(a, shuffled=True) for a in alphas]

    with open("positive_control.csv", "w") as fh:
        fh.write("alpha,real,shuffled\n")
        for a, r, s in zip(alphas, real, shuf):
            fh.write("{:.1f},{:.4f},{:.4f}\n".format(a, r, s))

    plt.figure(figsize=(7, 5))
    plt.plot(alphas, real, "o-", label="true response (specific signal)")
    plt.plot(alphas, shuf, "s--", label="shuffled control")
    plt.xlabel("signal strength $\\alpha$")
    plt.ylabel("recovered specific correlation")
    plt.title("Positive control: the pipeline recovers a specific signal when present")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("positive_control.png", dpi=150)
    print("alpha:", alphas)
    print("real :", ["{:.3f}".format(x) for x in real])
    print("shuf :", ["{:.3f}".format(x) for x in shuf])
    print("wrote positive_control.png, positive_control.csv")


if __name__ == "__main__":
    main()
