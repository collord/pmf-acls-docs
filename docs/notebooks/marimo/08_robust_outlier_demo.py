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
        # Robust Bayesian NMF — Outlier Detection via Student's t Likelihood

        Standard PMF assumes Gaussian residuals, which makes it sensitive to
        outliers (instrument spikes, contamination events, sample errors).
        A single bad data point can distort an entire factor profile.

        The **robust mode** replaces the Gaussian likelihood with a Student's t,
        implemented as a Gaussian-scale mixture: each element (i,j) receives an
        auxiliary precision weight $\eta_{ij}$.  Large residuals drive $\eta$
        toward zero, automatically down-weighting outliers without manual
        screening.

        $$X_{ij} \sim \mathcal{N}([\mathbf{GF}]_{ij},\; \sigma_{ij}^2 / \eta_{ij}),
        \quad \eta_{ij} \sim \mathrm{Gamma}(\tfrac{v}{2}, \tfrac{v}{2})$$

        The degrees-of-freedom parameter $v$ controls tail heaviness:
        small $v$ (3-5) aggressively rejects outliers; $v \to \infty$ recovers
        the standard Gaussian.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import linear_sum_assignment

    from pmf_acls import pmf_bayes

    return linear_sum_assignment, mo, np, plt, pmf_bayes


@app.cell
def _(linear_sum_assignment, np):
    def make_synthetic(rng, m=15, n=100, p=3, noise_frac=0.05):
        """Sparse-profile synthetic data."""
        F = np.full((m, p), 0.005)
        for k in range(p):
            n_load = max(2, int(round(0.3 * m)))
            vars_k = rng.choice(m, size=n_load, replace=False)
            F[vars_k, k] += rng.exponential(0.5, size=n_load)
        for k in range(p):
            F[:, k] /= F[:, k].sum()

        scales = np.logspace(np.log10(40), np.log10(8), p)
        G = np.zeros((p, n))
        for k in range(p):
            G[k, :] = rng.exponential(scales[k], size=n)

        X_true = F @ G
        sigma = noise_frac * np.maximum(X_true, 0.01) + 0.005
        noise = rng.normal(0, sigma)
        X_clean = np.maximum(X_true + noise, 0.0)
        return X_clean, sigma, F, G

    def inject_outliers(X, rng, frac=0.05, magnitude=10.0):
        """Inject outliers by multiplying random elements."""
        m, n = X.shape
        X_out = X.copy()
        n_outliers = max(1, int(frac * m * n))
        rows = rng.choice(m, size=n_outliers)
        cols = rng.choice(n, size=n_outliers)
        mask = np.zeros((m, n), dtype=bool)
        for i, j in zip(rows, cols):
            X_out[i, j] *= magnitude
            mask[i, j] = True
        return X_out, mask

    def match_factors(F_est, F_true):
        """Hungarian matching by absolute correlation."""
        p_est, p_true = F_est.shape[1], F_true.shape[1]
        corr = np.zeros((p_est, p_true))
        for i in range(p_est):
            for j in range(p_true):
                c = np.corrcoef(F_est[:, i], F_true[:, j])[0, 1]
                corr[i, j] = c if np.isfinite(c) else 0.0
        row_ind, col_ind = linear_sum_assignment(-np.abs(corr))
        return row_ind, col_ind, corr

    return make_synthetic, inject_outliers, match_factors


# ── Controls ──────────────────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md("## 1. Setup")
    return


@app.cell
def _(mo):
    m_slider = mo.ui.slider(10, 30, step=5, value=15, label="Variables (m)")
    n_slider = mo.ui.slider(50, 200, step=25, value=100, label="Observations (n)")
    p_slider = mo.ui.slider(2, 5, step=1, value=3, label="Factors (p)")
    noise_slider = mo.ui.slider(0.02, 0.20, step=0.02, value=0.05, label="Noise fraction")
    outlier_frac_slider = mo.ui.slider(0.01, 0.15, step=0.01, value=0.05, label="Outlier fraction")
    outlier_mag_slider = mo.ui.slider(3.0, 20.0, step=1.0, value=10.0, label="Outlier magnitude")
    df_slider = mo.ui.slider(1.0, 30.0, step=1.0, value=5.0, label="Degrees of freedom (v)")
    seed_number = mo.ui.number(value=2026, label="Seed")
    run_btn = mo.ui.run_button(label="Run Comparison")
    _controls = mo.vstack([
        mo.hstack([m_slider, n_slider, p_slider, noise_slider], justify="start", gap=1.0),
        mo.hstack([outlier_frac_slider, outlier_mag_slider, df_slider, seed_number, run_btn], justify="start", gap=1.0),
    ])
    _controls
    return (
        m_slider, n_slider, p_slider, noise_slider,
        outlier_frac_slider, outlier_mag_slider, df_slider,
        seed_number, run_btn,
    )


# ── Generate data + run solvers ───────────────────────────────────────────


@app.cell
def _(
    df_slider,
    inject_outliers,
    m_slider,
    make_synthetic,
    mo,
    n_slider,
    noise_slider,
    np,
    outlier_frac_slider,
    outlier_mag_slider,
    p_slider,
    pmf_bayes,
    run_btn,
    seed_number,
):
    mo.stop(not run_btn.value, mo.md("*Click **Run Comparison** above*"))

    _rng = np.random.default_rng(seed_number.value)
    X_clean, sigma, F_true, G_true = make_synthetic(
        _rng, m=m_slider.value, n=n_slider.value,
        p=p_slider.value, noise_frac=noise_slider.value,
    )
    X_outlier, outlier_mask_true = inject_outliers(
        X_clean, _rng,
        frac=outlier_frac_slider.value,
        magnitude=outlier_mag_slider.value,
    )
    _n_out = outlier_mask_true.sum()
    _m, _n = X_outlier.shape

    # Standard Bayesian NMF (no robust)
    res_std = pmf_bayes(
        X_outlier, sigma, p=p_slider.value,
        n_samples=1500, n_burnin=800,
        random_seed=seed_number.value, verbose=False,
    )

    # Robust Bayesian NMF
    res_rob = pmf_bayes(
        X_outlier, sigma, p=p_slider.value,
        n_samples=1500, n_burnin=800,
        random_seed=seed_number.value, verbose=False,
        robust=True, robust_df=df_slider.value,
    )

    # Also run on clean data for reference Q
    res_clean = pmf_bayes(
        X_clean, sigma, p=p_slider.value,
        n_samples=500, n_burnin=300,
        random_seed=seed_number.value, verbose=False,
    )

    mo.md(
        f"""
        **Data**: {_m} vars x {_n} obs, {p_slider.value} factors
        &nbsp;|&nbsp; **{_n_out} outliers** ({100*_n_out/(_m*_n):.1f}% of elements,
        {outlier_mag_slider.value:.0f}x magnitude)

        | Run | Q | Explained var | chi2 |
        |-----|---|---------------|------|
        | Clean (no outliers) | {res_clean.Q:.2e} | {res_clean.explained_variance:.2%} | {res_clean.chi2:.3f} |
        | Standard (with outliers) | {res_std.Q:.2e} | {res_std.explained_variance:.2%} | {res_std.chi2:.3f} |
        | **Robust** (with outliers) | {res_rob.Q:.2e} | {res_rob.explained_variance:.2%} | {res_rob.chi2:.3f} |
        """
    )
    return X_clean, X_outlier, sigma, F_true, G_true, outlier_mask_true, res_std, res_rob, res_clean


# ── Factor profile comparison ─────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. Factor Profiles — Standard vs Robust

        Both solvers return a multi-seed ACLS point estimate as `result.F`.
        Outliers distort the standard profiles (red). The robust solver
        (blue) recovers profiles closer to the truth (black lines) because
        outlier elements are automatically down-weighted during sampling.
        """
    )
    return


@app.cell
def _(F_true, match_factors, np, plt, res_rob, res_std):
    _row_s, _col_s, _corr_s = match_factors(res_std.F, F_true)
    _row_r, _col_r, _corr_r = match_factors(res_rob.F, F_true)

    _m, _p = F_true.shape
    _x = np.arange(_m)
    _var_labels = [f"V{i+1}" for i in range(_m)]

    _order_s = np.argsort(_col_s)
    _row_s, _col_s = _row_s[_order_s], _col_s[_order_s]
    _order_r = np.argsort(_col_r)
    _row_r, _col_r = _row_r[_order_r], _col_r[_order_r]

    _fig, _axes = plt.subplots(
        _p, 2, figsize=(max(8, 0.6 * _m), 3 * _p),
        constrained_layout=True, squeeze=False,
    )
    _fig.suptitle("Factor Profiles — Standard (left) vs Robust (right)", fontsize=13)

    for idx in range(_p):
        # Standard
        ks, kt = _row_s[idx], _col_s[idx]
        ax = _axes[idx, 0]
        ax.bar(_x, res_std.F[:, ks], 0.6, color="none", edgecolor="C3", linewidth=1.2)
        ax.hlines(F_true[:, kt], _x - 0.3, _x + 0.3, colors="0.2", linewidth=2.5)
        rs = abs(_corr_s[ks, kt])
        ax.set_title(f"Factor {kt} — Standard (r={rs:.3f})", fontsize=9)
        ax.set_xticks(_x)
        ax.set_xticklabels(_var_labels, rotation=90, fontsize=6)

        # Robust
        kr, kt_r = _row_r[idx], _col_r[idx]
        ax2 = _axes[idx, 1]
        ax2.bar(_x, res_rob.F[:, kr], 0.6, color="none", edgecolor="C0", linewidth=1.2)
        ax2.hlines(F_true[:, kt_r], _x - 0.3, _x + 0.3, colors="0.2", linewidth=2.5)
        rr = abs(_corr_r[kr, kt_r])
        ax2.set_title(f"Factor {kt_r} — Robust (r={rr:.3f})", fontsize=9)
        ax2.set_xticks(_x)
        ax2.set_xticklabels(_var_labels, rotation=90, fontsize=6)

    _axes[0, 0].legend(["Standard", "Truth"], fontsize=7, loc="upper right")
    _axes[0, 1].legend(["Robust", "Truth"], fontsize=7, loc="upper right")
    _fig
    return


# ── Eta heatmap ───────────────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. Eta Precision Weights — Outlier Map

        Each element's posterior mean $\\bar{\\eta}_{ij}$ indicates data quality:
        values near 1.0 are trusted; values near 0 are detected outliers.
        Red crosses mark the ground-truth outlier positions.
        """
    )
    return


@app.cell
def _(np, outlier_mask_true, plt, res_rob):
    _eta = res_rob.eta_posterior_mean
    _m, _n = _eta.shape

    _fig, _ax = plt.subplots(figsize=(min(12, 0.12 * _n), max(3, 0.25 * _m)),
                              constrained_layout=True)
    _im = _ax.imshow(_eta, aspect="auto", cmap="viridis", vmin=0, vmax=2.0)
    _fig.colorbar(_im, ax=_ax, label=r"$\bar{\eta}_{ij}$ (posterior mean)")

    # Overlay true outlier positions
    _oi, _oj = np.where(outlier_mask_true)
    _ax.scatter(_oj, _oi, marker="x", color="red", s=40, linewidths=1.5,
                label="True outliers", zorder=5)

    _ax.set_xlabel("Observation")
    _ax.set_ylabel("Variable")
    _ax.set_title("Eta Precision Weights (dark = down-weighted)")
    _ax.legend(fontsize=8, loc="upper right")

    # Precision / recall
    _detected = res_rob.outlier_mask
    _tp = (outlier_mask_true & _detected).sum()
    _fp = (~outlier_mask_true & _detected).sum()
    _fn = (outlier_mask_true & ~_detected).sum()
    _prec = _tp / max(_tp + _fp, 1)
    _rec = _tp / max(_tp + _fn, 1)
    _f1 = 2 * _prec * _rec / max(_prec + _rec, 1e-10)
    _ax.text(
        0.01, 0.01,
        f"Detection (eta<0.5):  precision={_prec:.2f}  recall={_rec:.2f}  F1={_f1:.2f}",
        transform=_ax.transAxes, fontsize=8, va="bottom",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )
    _fig
    return


# ── Eta distribution ──────────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        """
        ## 4. Eta Distribution — Outliers vs Clean

        Non-outlier elements cluster near $\\eta \\approx 1$; true outliers
        are pushed toward zero by the heavy-tailed likelihood.
        """
    )
    return


@app.cell
def _(np, outlier_mask_true, plt, res_rob):
    _eta_flat = res_rob.eta_posterior_mean.ravel()
    _mask_flat = outlier_mask_true.ravel()

    _fig, _ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    _bins = np.linspace(0, max(2.0, np.percentile(_eta_flat, 99)), 50)
    _ax.hist(_eta_flat[~_mask_flat], bins=_bins, alpha=0.6, color="C0",
             label="Clean elements", density=True)
    _ax.hist(_eta_flat[_mask_flat], bins=_bins, alpha=0.7, color="C3",
             label="True outliers", density=True)
    _ax.axvline(0.5, color="k", ls="--", lw=1, alpha=0.5, label="Detection threshold")
    _ax.set_xlabel(r"$\bar{\eta}_{ij}$")
    _ax.set_ylabel("Density")
    _ax.set_title("Distribution of Eta Weights")
    _ax.legend(fontsize=9)
    _fig
    return


# ── Residual vs eta scatter ───────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        """
        ## 5. Residual Magnitude vs Eta

        The precision weight $\\eta$ is inversely related to the squared
        residual — large residuals are automatically down-weighted.
        """
    )
    return


@app.cell
def _(X_outlier, np, outlier_mask_true, plt, res_rob, sigma):
    _R = X_outlier - res_rob.F @ res_rob.G
    _abs_resid = (np.abs(_R) / sigma).ravel()
    _eta_flat = res_rob.eta_posterior_mean.ravel()
    _mask_flat = outlier_mask_true.ravel()

    _fig, _ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    _ax.scatter(_abs_resid[~_mask_flat], _eta_flat[~_mask_flat],
                s=8, alpha=0.3, color="C0", label="Clean", zorder=2)
    _ax.scatter(_abs_resid[_mask_flat], _eta_flat[_mask_flat],
                s=30, alpha=0.8, color="C3", marker="x", linewidths=1.5,
                label="True outliers", zorder=3)
    _ax.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.5)
    _ax.set_xlabel("|Residual| / sigma")
    _ax.set_ylabel(r"$\bar{\eta}_{ij}$")
    _ax.set_title("Residual vs Precision Weight")
    _ax.legend(fontsize=9)
    _ax.set_xlim(left=0)
    _ax.set_ylim(bottom=0)
    _fig
    return


# ── Degrees of freedom sweep ─────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        """
        ## 6. Degrees-of-Freedom Sweep

        Sweep $v$ from 2 to 50 and measure outlier detection F1 and
        mean factor correlation with truth. There is a sweet spot:
        too small $v$ over-rejects good data, too large $v$ misses outliers.
        """
    )
    return


@app.cell
def _(mo):
    df_sweep_btn = mo.ui.run_button(label="Run DF Sweep")
    df_sweep_btn
    return (df_sweep_btn,)


@app.cell
def _(
    F_true,
    X_outlier,
    df_sweep_btn,
    match_factors,
    mo,
    np,
    outlier_mask_true,
    p_slider,
    plt,
    pmf_bayes,
    seed_number,
    sigma,
):
    mo.stop(not df_sweep_btn.value, mo.md("*Click **Run DF Sweep** above*"))

    _dfs = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50]
    _f1s = []
    _corrs = []

    for _v in _dfs:
        _res = pmf_bayes(
            X_outlier, sigma, p=p_slider.value,
            n_samples=800, n_burnin=400,
            random_seed=seed_number.value, verbose=False,
            robust=True, robust_df=float(_v),
        )
        # F1
        _det = _res.outlier_mask
        _tp = (outlier_mask_true & _det).sum()
        _fp = (~outlier_mask_true & _det).sum()
        _fn = (outlier_mask_true & ~_det).sum()
        _prec = _tp / max(_tp + _fp, 1)
        _rec = _tp / max(_tp + _fn, 1)
        _f1s.append(2 * _prec * _rec / max(_prec + _rec, 1e-10))

        # Mean correlation
        _r, _c, _corr = match_factors(_res.F, F_true)
        _corrs.append(np.mean([abs(_corr[_r[j], _c[j]]) for j in range(len(_r))]))

    _fig, _ax1 = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    _ax1.plot(_dfs, _f1s, "s-", color="C3", label="Outlier F1")
    _ax1.set_xlabel("Degrees of freedom (v)")
    _ax1.set_ylabel("F1 Score", color="C3")
    _ax1.set_xscale("log")
    _ax1.legend(fontsize=9, loc="center left")
    _ax1.set_title("Outlier Detection & Factor Quality vs DF")

    _ax2 = _ax1.twinx()
    _ax2.plot(_dfs, _corrs, "o--", color="C0", label="Mean |r| with truth")
    _ax2.set_ylabel("Mean factor correlation", color="C0")
    _ax2.legend(fontsize=9, loc="center right")

    _fig
    return


# ── Takeaways ─────────────────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        """
        ## Takeaways

        1. **Automatic outlier identification**: The robust Student's t likelihood
           assigns low precision weights ($\\eta \\approx 0$) to outlier elements
           without any manual screening or threshold tuning.

        2. **Better factor profiles under contamination**: By down-weighting
           outliers, the robust solver recovers profiles much closer to truth
           than the standard Gaussian likelihood.

        3. **Degrees of freedom (v) controls sensitivity**: $v = 3$-$5$ for
           aggressive outlier rejection; $v = 10$-$20$ for mild robustness;
           $v \\to \\infty$ recovers the standard Gaussian.

        4. **Zero overhead when not needed**: `robust=False` (default) skips the
           eta sampling entirely, so existing workflows are unaffected.

        5. **Complements hierarchical sigma**: Hierarchical sigma addresses
           uncertainty about the *measurement error scale*. Robust mode addresses
           *occasional gross errors* within a given measurement. Both can be
           active simultaneously.
        """
    )
    return


if __name__ == "__main__":
    app.run()
