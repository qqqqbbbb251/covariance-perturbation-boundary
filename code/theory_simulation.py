"""
theory_simulation.py -- numerical check of the identifiability sketch in
理论_可识别性与生成模型.md.

Generative model per "dataset":
    Sigma = lam * v v^T + Sigma_perp          (control covariance, spike + bulk)
    dX_i  = b_i * v + c_i * w + eps_i + noise  (perturbation i's mean shift)
The forward model predicts dX_i from Sigma[:, target_i] (rank-1 least squares).  We
check:
  * forward R^2 tracks shared_frac = cos^2(dX, v) as the spike dominates;
  * the operator column cannot predict the specific part eps (off-axis);
  * a second dataset with correlated eps recovers it via Procrustes.

Outputs: ../results/theory_simulation.csv, ../figures/fig28_theory_simulation.png
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG, exist_ok=True)
P, N, SEED = 400, 120, 0


def cos2(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b) ** 2 / (na * na * nb * nb) if na and nb else np.nan


def one_run(rng, lam, mu, s_spec, s_shared, noise, rho):
    v = rng.normal(size=P); v /= np.linalg.norm(v)
    B = rng.normal(size=(P, P)); Sperp = (B @ B.T) / P * mu
    w = rng.normal(size=P); w -= (w @ v) * v; w /= np.linalg.norm(w)
    Sigma = lam * np.outer(v, v) + Sperp
    Sigma = 0.5 * (Sigma + Sigma.T)

    def spec(n_):
        e = rng.normal(size=(n_, P))
        e -= np.outer(e @ v, v)                       # orthogonal to the shared axis
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        return e
    eps = spec(N)
    if rho < 1:
        eps2 = rho * eps + np.sqrt(1 - rho ** 2) * spec(N)
    else:
        eps2 = eps
    b = rng.normal(size=(N, 1)) * s_shared
    c = rng.normal(size=(N, 1)) * s_shared
    dX = b * v + c * w + s_spec * eps + noise * rng.normal(size=(N, P))

    # forward prediction from Sigma[:, target gene i]
    r2, shared, off_cos2 = [], [], []
    Rt, Rp = [], []
    for i in range(N):
        col = Sigma[:, i % P]
        a = float(col @ dX[i]) / float(col @ col)
        pred = a * col
        r2.append(1 - float((dX[i] - pred) @ (dX[i] - pred)) / float(dX[i] @ dX[i]))
        shared.append(cos2(dX[i], v))
        rt = dX[i] - float(dX[i] @ v) * v
        rp = pred - float(pred @ v) * v
        off_cos2.append(cos2(rt, rp))
        Rt.append(rt); Rp.append(rp)
    return (float(np.mean(r2)), float(np.mean(shared)), float(np.mean(off_cos2)),
            float(np.mean([cos2(Rt[i], eps2[i]) for i in range(N)])))


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    # sweep the bulk magnitude mu at fixed spike lambda: theory says forward R^2 ->
    # shared_frac as mu/lambda -> 0, and the operator stays blind to eps.
    for mu in [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
        r2, sf, offc, _ = one_run(rng, lam=1.0, mu=mu, s_spec=1.0, s_shared=1.0,
                                  noise=0.1, rho=0.0)
        rows.append({"mu": mu, "forward_r2": r2, "shared_frac": sf, "operator_off_cos2": offc})
        print("mu=%-8s R2=%.3f shared=%.3f off_cos2=%.4f" % (mu, r2, sf, offc), flush=True)
    # Procrustes recovery of eps across two datasets
    proc = []
    for rho in [0.0, 0.3, 0.6, 0.9]:
        rng2 = np.random.default_rng(1)
        _, _, _, rec = one_run(rng2, lam=0.2, mu=0.1, s_spec=1.0, s_shared=1.0,
                               noise=0.1, rho=rho)
        proc.append((rho, rec))
        print("rho=%.1f  eps recovery cos2=%.3f" % (rho, rec), flush=True)

    with open(os.path.join(RES, "theory_simulation.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "x", "y", "shared_frac", "operator_off_cos2"])
        for r in rows:
            w.writerow(["forward_vs_mu", r["mu"], r["forward_r2"], r["shared_frac"],
                        r["operator_off_cos2"]])
        for rho, rec in proc:
            w.writerow(["procrustes_eps", rho, rec, "", ""])

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    xs = [max(r["mu"], 1e-7) for r in rows]
    ax[0].plot(xs, [r["forward_r2"] for r in rows], "o-", label="forward R²")
    ax[0].plot(xs, [r["shared_frac"] for r in rows], "k--", label="shared fraction")
    ax[0].plot(xs, [r["operator_off_cos2"] for r in rows], "s-", color="#c53030", label="operator off-axis cos²")
    ax[0].set_xscale("log"); ax[0].set_xlabel("bulk magnitude μ (spike λ=1)"); ax[0].set_ylabel("value")
    ax[0].set_title("A. Forward R² → shared_frac as Σ becomes rank-1;\noperator stays blind to the specific part")
    ax[0].legend(fontsize=8)
    ax[1].plot([rho for rho, _ in proc], [rec for _, rec in proc], "o-", color="#276749")
    ax[1].set_xlabel("cross-dataset ε correlation ρ"); ax[1].set_ylabel("recovered ε cos²")
    ax[1].set_ylim(-0.05, 1.0); ax[1].set_title("B. Specific part recoverable across datasets\n(not from Σ)")
    fig.suptitle("Synthetic check of the identifiability sketch", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIG, "fig28_theory_simulation.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", os.path.join(RES, "theory_simulation.csv"), "and", out)


if __name__ == "__main__":
    main()
