# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pmf-acls",
# ]
# ///
import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Informative F Prior — ACLS → Bayesian UQ

        A common workflow: solve with ACLS to get a point estimate of the
        factor profiles (F), then pass that F as an informative prior to
        the Bayesian solver to obtain credible intervals on both F and G.

        This avoids two problems:

        1. **Cold-start collapse** — the Gibbs sampler can merge factors
           if started from random init.  An ACLS-derived F anchors the
           posterior around well-separated profiles.
        2. **Warm-start rigidity** — the default Bayesian warm-start
           reruns ACLS internally, which may find a different local
           minimum.  Passing F directly ensures the Bayesian solver
           explores around *your* solution.

        The `F_prior_scale` parameter controls how tightly the posterior
        is anchored to the ACLS profiles.  Smaller values = tighter.
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
    from scipy.optimize import linear_sum_assignment

    from pmf_acls import pmf, pmf_bayes

    return linear_sum_assignment, mo, np, plt, pmf, pmf_bayes


# --- Synthetic data generation ---


@app.cell
def _(np):
    def make_synthetic(rng, m=25, n=80, noise_frac=0.10, scales=None):
        """Generate synthetic NMF data with configurable source strengths."""
        if scales is None:
            scales = np.array([50.0, 40.0, 15.0, 3.0, 2.0])
        scales = np.asarray(scales, dtype=float)
        p = len(scales)
        source_names = [f"Source {k+1}" for k in range(p)]
        var_names = [f"V{i+1:02d}" for i in range(m)]

        F = np.full((m, p), 0.01)
        for k in range(p):
            frac = max(0.15, min(0.5, 0.1 + 0.01 * scales[k]))
            n_load = max(2, int(round(frac * m)))
            n_load = min(n_load, m)
            chosen = rng.choice(m, size=n_load, replace=False)
            F[chosen, k] += rng.exponential(
                max(0.1, 0.01 * scales[k]), size=n_load,
            )
        for k in range(p):
            F[:, k] /= F[:, k].sum()

        G = np.zeros((p, n))
        for k in range(p):
            G[k, :] = rng.exponential(scales[k], size=n)

        X_true = F @ G
        sigma = noise_frac * np.maximum(X_true, 0.01) + 0.005
        noise = rng.normal(0, sigma)
        X = np.maximum(X_true + noise, 0.0)

        return X, sigma, F, G, source_names, var_names

    def match_factors(F_est, F_true, linear_sum_assignment):
        """Hungarian matching of estimated to true factors by correlation."""
        p_est = F_est.shape[1]
        p_true = F_true.shape[1]
        corr = np.zeros((p_est, p_true))
        for i in range(p_est):
            for j in range(p_true):
                c = np.corrcoef(F_est[:, i], F_true[:, j])[0, 1]
                corr[i, j] = c if np.isfinite(c) else 0.0
        row_ind, col_ind = linear_sum_assignment(-np.abs(corr))
        return row_ind, col_ind, corr

    return make_synthetic, match_factors


# --- Configuration ---


@app.cell
def _(mo):
    mo.md("## 1. Configuration")
    return


@app.cell
def _(mo):
    noise_slider = mo.ui.slider(
        0.02, 0.30, step=0.02, value=0.10, label="Noise fraction"
    )
    n_vars_slider = mo.ui.slider(
        10, 50, step=5, value=25, label="Variables (m)"
    )
    n_obs_slider = mo.ui.slider(
        40, 200, step=20, value=80, label="Observations (n)"
    )
    seed_number = mo.ui.number(value=2026, label="Random seed")
    prior_scale_slider = mo.ui.slider(
        0.05, 5.0, step=0.05, value=1.0, label="F_prior_scale"
    )
    n_samples_slider = mo.ui.slider(
        200, 2000, step=200, value=1000, label="Bayesian n_samples"
    )
    n_burnin_slider = mo.ui.slider(
        100, 1000, step=100, value=500, label="Bayesian n_burnin"
    )

    mo.vstack([
        mo.hstack(
            [noise_slider, n_vars_slider, n_obs_slider, seed_number],
            justify="start",
        ),
        mo.hstack(
            [prior_scale_slider, n_samples_slider, n_burnin_slider],
            justify="start",
        ),
    ])
    return (
        n_burnin_slider,
        n_obs_slider,
        n_samples_slider,
        n_vars_slider,
        noise_slider,
        prior_scale_slider,
        seed_number,
    )


@app.cell
def _(
    mo,
    n_burnin_slider,
    n_obs_slider,
    n_samples_slider,
    n_vars_slider,
    noise_slider,
    prior_scale_slider,
    seed_number,
):
    mo.md(
        f"**Data**: m={n_vars_slider.value}, n={n_obs_slider.value}, "
        f"noise={noise_slider.value:.0%}, seed={seed_number.value} / "
        f"**Bayesian**: F_prior_scale={prior_scale_slider.value:.2f}, "
        f"n_samples={n_samples_slider.value}, n_burnin={n_burnin_slider.value}"
    )
    return


# --- Generate data ---


@app.cell
def _(make_synthetic, mo, n_obs_slider, n_vars_slider, noise_slider, np, seed_number):
    _rng = np.random.default_rng(int(seed_number.value))
    X, sigma, F_true, G_true, source_names, var_names = make_synthetic(
        _rng,
        m=n_vars_slider.value,
        n=n_obs_slider.value,
        noise_frac=noise_slider.value,
    )
    p_true = F_true.shape[1]

    _X_true = F_true @ G_true
    _snr_power = np.sum(_X_true**2) / np.sum(sigma**2)
    _snr_db = 10 * np.log10(_snr_power)

    mo.md(
        f"**Data**: {X.shape[0]} variables × {X.shape[1]} observations, "
        f"**{p_true} true sources**, noise = {noise_slider.value:.0%}, "
        f"SNR = {_snr_db:.1f} dB"
    )
    return F_true, G_true, X, p_true, sigma, source_names, var_names


# --- Step 1: ACLS multi-seed solve ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. ACLS Point Estimate (Best of 30 Seeds)

        We run 30 random-seed ACLS solves and keep the one with the
        lowest Q (weighted residual).  This gives us the best available
        point estimate of F.
        """
    )
    return


@app.cell
def _(X, mo, np, p_true, pmf, sigma):
    _best_Q = np.inf
    _best_result = None
    for _seed in range(30):
        try:
            _r = pmf(
                X, sigma, p_true,
                algorithm="acls",
                max_iter=1000,
                conv_tol=0.005,
                random_seed=_seed,
            )
            if _r.Q < _best_Q:
                _best_Q = _r.Q
                _best_result = _r
        except Exception:
            continue

    acls_result = _best_result
    mo.md(
        f"**Best ACLS result**: Q = {acls_result.Q:.4e}, "
        f"converged = {acls_result.converged}, "
        f"Q/mn = {acls_result.Q / (X.shape[0] * X.shape[1]):.3f}"
    )
    return (acls_result,)


@app.cell
def _(
    F_true,
    acls_result,
    linear_sum_assignment,
    match_factors,
    np,
    plt,
    source_names,
    var_names,
):
    # ACLS factor chart
    _row, _col, _corr = match_factors(
        acls_result.F, F_true, linear_sum_assignment
    )
    _order = np.argsort(_col)
    _row = _row[_order]
    _col = _col[_order]

    _m, _p = acls_result.F.shape
    _p_show = min(_p, len(source_names))
    _x = np.arange(_m)

    _ncols = min(_p_show, 2)
    _nrows = (_p_show + _ncols - 1) // _ncols
    _fig = plt.figure(
        figsize=(max(5, 0.6 * _m) * _ncols, 4 * _nrows),
        constrained_layout=True,
    )
    _fig.suptitle("ACLS Point Estimate vs Truth", fontsize=14, fontweight="bold")

    for _idx in range(_p_show):
        if _idx >= len(_row):
            break
        _k_est = _row[_idx]
        _k_true = _col[_idx]
        _ax = _fig.add_subplot(_nrows, _ncols, _idx + 1)

        _f_est = acls_result.F[:, _k_est]

        _ax.bar(
            _x, _f_est, 0.6,
            color="C0", edgecolor="C0", alpha=0.6,
            label="ACLS estimate",
        )
        _ax.hlines(
            F_true[:, _k_true], _x - 0.3, _x + 0.3,
            colors="0.2", linewidth=2.0, label="Truth",
        )
        _r = abs(_corr[_k_est, _k_true])
        _ax.set_title(f"{source_names[_k_true]}  (r = {_r:.3f})", fontsize=11)
        _ax.set_xticks(_x)
        _ax.set_xticklabels(var_names, rotation=90, fontsize=7)
        if _idx % _ncols == 0:
            _ax.set_ylabel("F profile")
        if _idx == 0:
            _ax.legend(fontsize=8, loc="upper right")

    _fig
    return


# --- Step 2: Bayesian solve with informative F prior ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. Bayesian UQ with Informative F Prior

        We pass the ACLS F as `F_prior_mean` to the Bayesian solver.
        The solver initializes from this F and samples both F and G,
        with a Gaussian prior pulling F toward the ACLS profiles.

        `F_prior_scale` controls anchoring strength:
        - **Small** (0.1–0.5): tight — posterior stays close to ACLS
        - **Medium** (1.0): balanced — data can pull F away if warranted
        - **Large** (2.0+): loose — prior is weak, more like free sampling

        The credible intervals reflect genuine posterior uncertainty
        given both the data and the prior belief about F.
        """
    )
    return


@app.cell
def _(
    X,
    acls_result,
    mo,
    n_burnin_slider,
    n_samples_slider,
    p_true,
    pmf_bayes,
    prior_scale_slider,
    sigma,
):
    bayes_result = pmf_bayes(
        X, sigma, p_true,
        F_prior_mean=acls_result.F,
        F_prior_scale=prior_scale_slider.value,
        n_samples=n_samples_slider.value,
        n_burnin=n_burnin_slider.value,
        warm_start=False,
        random_seed=42,
        verbose=False,
    )

    _cd = bayes_result.convergence_details or {}
    _advisory = []
    if not _cd.get("geweke_ok", True):
        _advisory.append(f"Geweke |z|={_cd.get('geweke_z_abs', '?'):.2f}")
    if not _cd.get("ess_ok", True):
        _advisory.append(f"min ESS={_cd.get('min_factor_ess', '?'):.0f}")
    if not _cd.get("labels_ok", True):
        _advisory.append(f"label gap={bayes_result.label_switch_gap:.3f}")
    _adv_str = f" — advisory: {', '.join(_advisory)}" if _advisory else ""

    mo.md(
        f"**Bayesian result**: converged = {bayes_result.converged}{_adv_str}, "
        f"Q = {bayes_result.Q:.4e}, "
        f"F_prior_scale = {prior_scale_slider.value:.2f}"
    )
    return (bayes_result,)


@app.cell
def _(
    F_true,
    bayes_result,
    linear_sum_assignment,
    match_factors,
    np,
    plt,
    source_names,
    var_names,
):
    # Bayesian factor chart with credible intervals
    _row, _col, _corr = match_factors(
        bayes_result.F_posterior_mean, F_true, linear_sum_assignment
    )
    _order = np.argsort(_col)
    _row = _row[_order]
    _col = _col[_order]

    _m, _p = bayes_result.F_posterior_mean.shape
    _p_show = min(_p, len(source_names))
    _x = np.arange(_m)

    _ncols = min(_p_show, 2)
    _nrows = (_p_show + _ncols - 1) // _ncols
    _fig = plt.figure(
        figsize=(max(5, 0.6 * _m) * _ncols, 4 * _nrows),
        constrained_layout=True,
    )
    _fig.suptitle(
        "Bayesian Posterior Mean ± 95% CI vs Truth",
        fontsize=14, fontweight="bold",
    )

    for _idx in range(_p_show):
        if _idx >= len(_row):
            break
        _k_est = _row[_idx]
        _k_true = _col[_idx]
        _ax = _fig.add_subplot(_nrows, _ncols, _idx + 1)

        _f_mean = bayes_result.F_posterior_mean[:, _k_est]
        _f_std = bayes_result.F_std[:, _k_est]

        # Hollow bars for posterior mean with 95% CI whiskers
        _ax.bar(
            _x, _f_mean, 0.6,
            yerr=2 * _f_std, capsize=2,
            color="none", edgecolor="C0", linewidth=1.2,
            error_kw=dict(ecolor="C0", lw=0.8),
            label="Posterior mean ± 95% CI",
        )
        # Truth as heavy horizontal lines
        _ax.hlines(
            F_true[:, _k_true], _x - 0.3, _x + 0.3,
            colors="0.2", linewidth=2.0, label="Truth",
        )
        _r = abs(_corr[_k_est, _k_true])
        _ax.set_title(f"{source_names[_k_true]}  (r = {_r:.3f})", fontsize=11)
        _ax.set_xticks(_x)
        _ax.set_xticklabels(var_names, rotation=90, fontsize=7)
        if _idx % _ncols == 0:
            _ax.set_ylabel("F profile")
        if _idx == 0:
            _ax.legend(fontsize=8, loc="upper right")

    _fig
    return


# --- Comparison summary ---


@app.cell
def _(
    F_true,
    acls_result,
    bayes_result,
    linear_sum_assignment,
    match_factors,
    mo,
    np,
):
    mo.md("## 4. Comparison")

    _rows = []
    for _label, _F_est in [
        ("ACLS", acls_result.F),
        ("Bayesian posterior mean", bayes_result.F_posterior_mean),
    ]:
        _row, _col, _corr = match_factors(_F_est, F_true, linear_sum_assignment)
        _mean_r = np.mean([abs(_corr[_row[i], _col[i]]) for i in range(len(_row))])
        _rows.append(f"| {_label} | {_mean_r:.4f} |")

    # Coverage: fraction of truth within 95% CI
    _row_b, _col_b, _ = match_factors(
        bayes_result.F_posterior_mean, F_true, linear_sum_assignment
    )
    _n_covered = 0
    _n_total = 0
    for _idx in range(len(_row_b)):
        _k_est = _row_b[_idx]
        _k_true = _col_b[_idx]
        _lo = bayes_result.F_posterior_mean[:, _k_est] - 2 * bayes_result.F_std[:, _k_est]
        _hi = bayes_result.F_posterior_mean[:, _k_est] + 2 * bayes_result.F_std[:, _k_est]
        _covered = (F_true[:, _k_true] >= _lo) & (F_true[:, _k_true] <= _hi)
        _n_covered += _covered.sum()
        _n_total += len(_covered)
    _coverage = _n_covered / _n_total if _n_total > 0 else 0

    # Standardized displacement: how far did Bayes pull F from ACLS (in σ units)?
    # Match ACLS and Bayes factors to truth, then compare corresponding columns.
    _row_a, _col_a, _corr_a = match_factors(acls_result.F, F_true, linear_sum_assignment)
    _order_a = np.argsort(_col_a)
    _row_a = _row_a[_order_a]

    _order_b = np.argsort(_col_b)
    _row_b_sorted = _row_b[_order_b]
    _col_b_sorted = _col_b[_order_b]

    _pull_rows = []
    for _idx in range(len(_row_a)):
        _k_acls = _row_a[_idx]
        _k_bayes = _row_b_sorted[_idx]
        _k_true = _col_b_sorted[_idx]

        # Normalize both to unit L1 for comparable shape
        _f_acls = acls_result.F[:, _k_acls].copy()
        _f_acls_norm = _f_acls / (_f_acls.sum() + 1e-30)
        _f_bayes = bayes_result.F_posterior_mean[:, _k_bayes].copy()
        _f_bayes_norm = _f_bayes / (_f_bayes.sum() + 1e-30)
        _f_std = bayes_result.F_std[:, _k_bayes].copy()
        _f_std_safe = np.maximum(_f_std, 1e-30)

        # Per-element displacement in σ units
        _displ = (_f_bayes - _f_acls_norm) / _f_std_safe
        _rms = float(np.sqrt(np.mean(_displ ** 2)))

        # Cosine similarity (shape preservation)
        _cos = float(np.dot(_f_acls_norm, _f_bayes_norm) / (
            np.linalg.norm(_f_acls_norm) * np.linalg.norm(_f_bayes_norm) + 1e-30
        ))

        _pull_rows.append(
            f"| {source_names[_k_true]} | {_rms:.2f}σ | {_cos:.4f} |"
        )

    _pull_table = "\n".join(_pull_rows)

    mo.md(
        f"| Method | Mean |r| vs truth |\n"
        f"|---|:-:|\n"
        + "\n".join(_rows)
        + f"\n\n**95% CI coverage of truth**: {_coverage:.0%} "
        f"({_n_covered}/{_n_total} elements)\n\n"
        f"Coverage near 95% indicates well-calibrated credible intervals. "
        f"Coverage well above 95% suggests the prior is too loose "
        f"(wide intervals); well below suggests it is too tight.\n\n"
        f"### Posterior Displacement from ACLS\n\n"
        f"How far the Bayesian posterior mean moved from the ACLS point estimate, "
        f"in units of posterior standard deviations.\n\n"
        f"| Factor | RMS pull | Cosine sim |\n"
        f"|---|:-:|:-:|\n"
        f"{_pull_table}\n\n"
        f"**RMS pull** < 0.5σ: ACLS and data agree (prior dominates) / "
        f"0.5–1.5σ: healthy data-informed adjustment / "
        f"1.5–3σ: meaningful disagreement / "
        f"> 3σ: ACLS solution substantially revised.\n\n"
        f"**Cosine sim** near 1.0: profile shape preserved; "
        f"below 0.95: shape changed meaningfully."
    )
    return


if __name__ == "__main__":
    app.run()
