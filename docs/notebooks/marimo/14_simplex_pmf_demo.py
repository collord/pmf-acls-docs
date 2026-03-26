# /// script
# requires-python = ">=3.11"
# dependencies = [
# ]
# ///
import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
async def _():
    import sys
    if "pyodide" in sys.modules or "pyodide_js" in sys.modules:
        import pyodide_js
        await pyodide_js.loadPackage(["numpy", "scipy", "matplotlib"])
        import micropip
        await micropip.install("pmf-acls", deps=False)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Simplex-Constrained PMF

        `simplex_pmf()` produces factor contributions (G) whose columns
        sum to 1 — each observation is decomposed into **fractional source
        contributions** that form a valid composition on the simplex.

        This is useful when the application requires:

        - End-member mixing fractions (sediment fingerprinting)
        - Regulatory reporting of source contribution percentages
        - Direct comparison of relative source importance across samples

        ### How it works

        The algorithm alternates between:

        1. **ACLS iterations** — minimize weighted reconstruction error
           Q = Σ((X - FG)/σ)², warm-started from the previous outer step
        2. **Simplex projection** — column-normalize G so each observation's
           contributions sum to 1, with approximate compensation in F

        ### Important tradeoffs

        The simplex projection is **mathematically inexact**. Normalizing G
        columns is a per-observation operation that cannot be exactly
        compensated in F (which has no observation index). Each projection
        introduces error that subsequent ACLS steps only partially repair.

        Consequences:
        - **Q increases** relative to unconstrained PMF (the constraint
          restricts the feasible set)
        - **Q/Q_exp loses its standard interpretation** (it absorbs
          projection error, not just data-model misfit)
        - Source profiles in F are distorted by the compensation

        **When to prefer standard `pmf()` instead:** if you need absolute
        contributions (µg/m³), run unconstrained PMF and compute fractions
        post-hoc via `G / G.sum(axis=0)`. This introduces zero factorization
        error and defers normalization to reporting.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["axes.edgecolor"] = "black"

    from pmf_acls import pmf, simplex_pmf

    return mo, np, plt, pmf, simplex_pmf


# --- Configuration ---


@app.cell
def _(mo):
    mo.md("## 1. Configuration")
    return


@app.cell
def _(mo):
    n_vars_slider = mo.ui.slider(8, 25, step=1, value=12, label="Variables (m)")
    n_obs_slider = mo.ui.slider(40, 200, step=20, value=80, label="Observations (n)")
    n_sources_slider = mo.ui.slider(2, 5, step=1, value=3, label="True sources")
    noise_slider = mo.ui.slider(0.02, 0.20, step=0.02, value=0.06, label="Noise fraction")
    seed_number = mo.ui.number(value=42, label="Random seed")

    mo.vstack([
        mo.hstack(
            [n_vars_slider, n_obs_slider, n_sources_slider],
            justify="start", gap=1.5,
        ),
        mo.hstack(
            [noise_slider, seed_number],
            justify="start", gap=1.5,
        ),
    ])
    return n_obs_slider, n_sources_slider, n_vars_slider, noise_slider, seed_number


# --- Synthetic data with simplex-constrained G ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. Synthetic Data

        Ground truth has G columns summing to 1 — each observation is a
        mixture of sources with known fractional contributions. F absorbs
        the absolute scale (total mass per source).
        """
    )
    return


@app.cell
def _(
    mo,
    n_obs_slider,
    n_sources_slider,
    n_vars_slider,
    noise_slider,
    np,
    seed_number,
):
    _rng = np.random.default_rng(int(seed_number.value))
    _m = n_vars_slider.value
    _n = n_obs_slider.value
    _p = n_sources_slider.value

    # True source profiles (arbitrary non-negative, not on the simplex)
    F_true = _rng.exponential(1.0, size=(_m, _p))
    # Make profiles more distinct by concentrating mass in different variables
    for _k in range(_p):
        _dominant = _rng.choice(_m, size=max(2, _m // _p), replace=False)
        F_true[_dominant, _k] *= 5.0

    # True fractional contributions on the simplex (columns sum to 1)
    _alpha = np.ones(_p) * 2.0
    G_true = _rng.dirichlet(_alpha, size=_n).T  # (p, n)

    # Clean signal + noise
    X_clean = F_true @ G_true
    _sigma_true = noise_slider.value * np.maximum(X_clean, 0.01) + 0.001
    X = np.maximum(X_clean + _rng.normal(0, _sigma_true), 1e-8)
    sigma = noise_slider.value * np.maximum(X, 0.01) + 0.001

    var_names = [f"V{_i+1:02d}" for _i in range(_m)]

    mo.md(
        f"**Data**: {_m} variables x {_n} observations, "
        f"**{_p} true sources**, noise = {noise_slider.value:.0%}\n\n"
        f"True G column sums: all = 1.0 (by construction)"
    )
    return F_true, G_true, X, sigma, var_names


# --- Run simplex_pmf and standard pmf ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. Simplex PMF vs Standard PMF

        We run both and compare:
        - **Q values** (simplex Q ≥ standard Q, since the constraint
          restricts the feasible set)
        - **G column sums** (simplex ≈ 1.0, standard unconstrained)
        - **Factor profile recovery** (Hungarian-matched to truth)
        """
    )
    return


@app.cell
def _(F_true, X, mo, np, pmf, sigma, simplex_pmf):
    _p = F_true.shape[1]
    _m, _n = X.shape

    # Standard PMF (best of 5 seeds)
    _best_Q = np.inf
    r_std = None
    for _seed in range(5):
        _r = pmf(X, sigma, _p, max_iter=2000, conv_tol=0.001, random_seed=_seed)
        if _r.Q < _best_Q:
            _best_Q = _r.Q
            r_std = _r

    # Simplex PMF — warm-start from best standard PMF
    r_simplex = simplex_pmf(
        X, sigma, _p,
        F_init=r_std.F.copy(), G_init=r_std.G.copy(),
        max_outer=50, inner_iter=200,
        simplex_tol=1e-4, conv_tol=0.001,
    )

    _col_sums_std = r_std.G.sum(axis=0)
    _col_sums_sim = r_simplex.G.sum(axis=0)

    mo.md(
        f"| | Standard PMF | Simplex PMF |\n"
        f"|---|---:|---:|\n"
        f"| Q | {r_std.Q:.2f} | {r_simplex.Q:.2f} |\n"
        f"| Q/mn | {r_std.Q / (_m * _n):.4f} | {r_simplex.Q / (_m * _n):.4f} |\n"
        f"| Converged | {r_std.converged} | {r_simplex.converged} |\n"
        f"| G col-sum mean | {_col_sums_std.mean():.4f} | {_col_sums_sim.mean():.4f} |\n"
        f"| G col-sum std | {_col_sums_std.std():.4f} | {_col_sums_sim.std():.4f} |\n"
        f"| G col-sum range | [{_col_sums_std.min():.3f}, {_col_sums_std.max():.3f}] "
        f"| [{_col_sums_sim.min():.4f}, {_col_sums_sim.max():.4f}] |\n"
        f"\nQ increases under the simplex constraint — this is expected. "
        f"The constraint restricts the feasible set, so the optimizer "
        f"cannot reach the same minimum."
    )
    return r_simplex, r_std


# --- G column sums histogram ---


@app.cell
def _(mo):
    mo.md("### Distribution of G Column Sums")
    return


@app.cell
def _(np, plt, r_simplex, r_std):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    _cs_std = r_std.G.sum(axis=0)
    _cs_sim = r_simplex.G.sum(axis=0)

    _ax1.hist(_cs_std, bins=20, color="coral", alpha=0.7, edgecolor="black")
    _ax1.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="Target = 1.0")
    _ax1.set_xlabel("Column sum of G")
    _ax1.set_ylabel("Count")
    _ax1.set_title(f"Standard PMF (mean={_cs_std.mean():.3f})")
    _ax1.legend(fontsize=8)

    _ax2.hist(_cs_sim, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
    _ax2.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="Target = 1.0")
    _ax2.set_xlabel("Column sum of G")
    _ax2.set_ylabel("Count")
    _ax2.set_title(f"Simplex PMF (mean={_cs_sim.mean():.4f})")
    _ax2.legend(fontsize=8)

    # Use same x-axis range for fair comparison
    _lo = min(_cs_std.min(), _cs_sim.min()) * 0.95
    _hi = max(_cs_std.max(), _cs_sim.max()) * 1.05
    _ax1.set_xlim(_lo, _hi)
    _ax2.set_xlim(np.min([0.99, _cs_sim.min() * 0.999]),
                  np.max([1.01, _cs_sim.max() * 1.001]))

    _fig.suptitle("G Column Sums: Unconstrained vs Simplex-Constrained", fontsize=12)
    _fig.tight_layout()
    _fig
    return


# --- Factor profile comparison ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 4. Factor Profile Recovery

        Both results are Hungarian-matched to truth. Since `simplex_pmf`
        constrains G (not F), the source profiles in F absorb both the
        true profile shape and the total mass that was previously in G.
        We L1-normalize F columns for visual comparison.
        """
    )
    return


@app.cell
def _(F_true, np, plt, r_simplex, r_std, var_names):
    from scipy.optimize import linear_sum_assignment

    def _hungarian_match(F_est, F_ref):
        """Hungarian match columns of F_est to F_ref."""
        _p = F_est.shape[1]
        _corr = np.zeros((_p, _p))
        for _i in range(_p):
            for _j in range(_p):
                _c = np.corrcoef(F_est[:, _i], F_ref[:, _j])[0, 1]
                _corr[_i, _j] = _c if np.isfinite(_c) else 0.0
        _ri, _ci = linear_sum_assignment(-np.abs(_corr))
        return _ri, _ci, _corr

    def _normalize_cols(F):
        """L1-normalize columns of F."""
        _sums = F.sum(axis=0, keepdims=True)
        _sums[_sums == 0] = 1.0
        return F / _sums

    _p = F_true.shape[1]
    _m = F_true.shape[0]
    _x = np.arange(_m)
    _F_true_norm = _normalize_cols(F_true)

    _fig, _axes = plt.subplots(2, _p, figsize=(3.5 * _p, 6), sharey=False)
    if _p == 1:
        _axes = _axes.reshape(2, 1)

    for _row, (_result, _label, _color) in enumerate([
        (r_std, "Standard PMF", "coral"),
        (r_simplex, "Simplex PMF", "steelblue"),
    ]):
        _ri, _ci, _corr = _hungarian_match(_result.F, F_true)
        _order = np.empty(_p, dtype=int)
        for _k in range(_p):
            _order[_ci[_k]] = _ri[_k]
        _F_norm = _normalize_cols(_result.F[:, _order])

        for _k in range(_p):
            _ax = _axes[_row, _k]
            _c = np.corrcoef(_F_norm[:, _k], _F_true_norm[:, _k])[0, 1]
            _r = abs(_c) if np.isfinite(_c) else 0.0

            _ax.bar(_x, _F_norm[:, _k], color=_color, alpha=0.7, label=_label)
            _ax.hlines(
                _F_true_norm[:, _k], _x - 0.35, _x + 0.35,
                colors="black", linewidths=2, label="True",
            )
            _ax.set_title(f"Factor {_k+1} (|r|={_r:.3f})", fontsize=9)
            _ax.set_xticks(_x)
            _ax.set_xticklabels(var_names, rotation=90, fontsize=6)
            if _k == 0:
                _ax.set_ylabel(_label, fontsize=9)
                _ax.legend(fontsize=6)

    _fig.suptitle("L1-Normalized Factor Profiles vs Truth", fontsize=12)
    _fig.tight_layout()
    _fig
    return


# --- Contribution time series ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 5. Source Contribution Time Series

        With simplex-constrained G, each column is a valid composition —
        the contributions directly show what fraction of each observation
        comes from each source, without post-hoc normalization.
        """
    )
    return


@app.cell
def _(F_true, G_true, np, plt, r_simplex, r_std):
    from scipy.optimize import linear_sum_assignment as _lsa

    def _match_G(G_est, G_ref):
        _p = G_est.shape[0]
        _corr = np.zeros((_p, _p))
        for _i in range(_p):
            for _j in range(_p):
                _c = np.corrcoef(G_est[_i, :], G_ref[_j, :])[0, 1]
                _corr[_i, _j] = _c if np.isfinite(_c) else 0.0
        _ri, _ci = _lsa(-np.abs(_corr))
        _order = np.empty(_p, dtype=int)
        for _k in range(_p):
            _order[_ci[_k]] = _ri[_k]
        return _order

    _p = F_true.shape[1]
    _n = G_true.shape[1]
    _obs = np.arange(_n)

    # Normalize standard PMF G for comparison
    _G_std_frac = r_std.G / r_std.G.sum(axis=0, keepdims=True)

    # Match both to truth
    _order_sim = _match_G(r_simplex.G, G_true)
    _order_std = _match_G(_G_std_frac, G_true)

    _fig, _axes = plt.subplots(_p, 1, figsize=(10, 2.5 * _p), sharex=True)
    if _p == 1:
        _axes = [_axes]

    for _k in range(_p):
        _ax = _axes[_k]
        _ax.plot(_obs, G_true[_k, :], "k-", linewidth=1.5,
                 label="True", alpha=0.7)
        _ax.plot(_obs, r_simplex.G[_order_sim[_k], :], "-",
                 color="steelblue", linewidth=1, label="Simplex PMF", alpha=0.8)
        _ax.plot(_obs, _G_std_frac[_order_std[_k], :], "--",
                 color="coral", linewidth=1, label="Standard (post-hoc)", alpha=0.8)
        _ax.set_ylabel(f"Source {_k+1}", fontsize=9)
        if _k == 0:
            _ax.legend(fontsize=7, ncol=3, loc="upper right")

    _axes[-1].set_xlabel("Observation")
    _fig.suptitle("Fractional Source Contributions", fontsize=12)
    _fig.tight_layout()
    _fig
    return


# --- Q cost tradeoff ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 6. The Q Tradeoff

        The simplex constraint increases Q because the optimizer can no
        longer freely adjust G column sums. The gap between unconstrained
        and simplex Q quantifies the cost of enforcing fractional contributions.

        For well-separated sources with moderate noise, the gap is small.
        For overlapping sources or high noise, the gap grows — the constraint
        forces the optimizer to sacrifice reconstruction accuracy.
        """
    )
    return


@app.cell
def _(X, mo, np, plt, r_simplex, r_std, sigma):
    _m, _n = X.shape
    _mn = _m * _n

    # --- Reconstruction error metrics ---
    _R_std = X - r_std.F @ r_std.G
    _R_sim = X - r_simplex.F @ r_simplex.G

    # RMSE (root mean squared error, in data units)
    _rmse_std = np.sqrt(np.mean(_R_std ** 2))
    _rmse_sim = np.sqrt(np.mean(_R_sim ** 2))

    # Normalized Euclidean distance: ||X - FG||_F / ||X||_F
    _X_norm = np.linalg.norm(X, "fro")
    _ned_std = np.linalg.norm(_R_std, "fro") / _X_norm
    _ned_sim = np.linalg.norm(_R_sim, "fro") / _X_norm

    # Q/mn (chi-squared per element)
    _qmn_std = r_std.Q / _mn
    _qmn_sim = r_simplex.Q / _mn

    mo.md(
        f"### Reconstruction Error Summary\n\n"
        f"| Metric | Standard PMF | Simplex PMF | Ratio (simplex/std) |\n"
        f"|--------|---:|---:|---:|\n"
        f"| Q | {r_std.Q:.2f} | {r_simplex.Q:.2f} | {r_simplex.Q / max(r_std.Q, 1e-30):.3f} |\n"
        f"| Q/mn | {_qmn_std:.4f} | {_qmn_sim:.4f} | {_qmn_sim / max(_qmn_std, 1e-30):.3f} |\n"
        f"| RMSE | {_rmse_std:.4e} | {_rmse_sim:.4e} | {_rmse_sim / max(_rmse_std, 1e-30):.3f} |\n"
        f"| Norm. Euclidean dist. | {_ned_std:.4f} | {_ned_sim:.4f} | {_ned_sim / max(_ned_std, 1e-30):.3f} |\n"
        f"\n"
        f"- **Q** = Σ((X - FG)/σ)² — the uncertainty-weighted objective; "
        f"the simplex constraint increases it because the feasible set is smaller\n"
        f"- **RMSE** = √(mean((X - FG)²)) — reconstruction error in data units, "
        f"unweighted by σ\n"
        f"- **Norm. Euclidean dist.** = ‖X - FG‖_F / ‖X‖_F — "
        f"fractional reconstruction error (0 = perfect, 1 = no signal recovered)\n"
        f"\nA ratio near 1.0 means the simplex constraint costs little; "
        f"a ratio >> 1 means the constraint is significantly degrading fit."
    )
    return


@app.cell
def _(np, plt, r_simplex, r_std):
    # Q convergence plots
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4))

    _ax1.plot(r_std.Q_history, color="coral", label="Standard PMF")
    _ax1.set_xlabel("ACLS iteration")
    _ax1.set_ylabel("Q")
    _ax1.set_title("Standard PMF: Q Convergence")
    _ax1.set_yscale("log")
    _ax1.legend(fontsize=8)

    _ax2.plot(r_simplex.Q_history, color="steelblue", label="Simplex PMF")
    _ax2.set_xlabel("ACLS iteration (across all outer steps)")
    _ax2.set_ylabel("Q")
    _ax2.set_title("Simplex PMF: Q Convergence")
    _ax2.set_yscale("log")
    _ax2.legend(fontsize=8)

    _fig.suptitle("Convergence Comparison", fontsize=12)
    _fig.tight_layout()
    _fig
    return


# --- When to use simplex_pmf ---


@app.cell
def _(mo):
    mo.md(
        """
        ## When to Use `simplex_pmf`

        | Scenario | Recommended approach |
        |----------|---------------------|
        | Need fractional contributions for reporting | `simplex_pmf()` |
        | End-member mixing (sediment, water) | `simplex_pmf()` |
        | Comparing relative source importance across samples | `simplex_pmf()` |
        | Need best reconstruction accuracy | `pmf()` (standard) |
        | Need reliable Q/Q_exp diagnostics | `pmf()` (standard) |
        | Absolute concentrations (µg/m³) | `pmf()`, compute fractions post-hoc |
        | Bootstrap / DISP uncertainty analysis | `pmf()` (diagnostics assume unconstrained Q) |

        ### Warm-starting tip

        `simplex_pmf` benefits from warm-starting with a converged
        standard PMF solution:

        ```python
        base = pmf(X, sigma, p=4, max_iter=2000)
        result = simplex_pmf(X, sigma, p=4,
                             F_init=base.F, G_init=base.G)
        ```

        This gives the inner ACLS solver a good starting point,
        so the outer projection loop converges faster.
        """
    )
    return


if __name__ == "__main__":
    app.run()
