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
        # Bayesian PMF Demo

        Interactive exploration of the Bayesian NMF system: posterior uncertainty
        for major and minor factors, ARD for automatic factor selection, WAIC model
        comparison, and MCMC convergence diagnostics.

        The synthetic problem has **5 sources** with very different magnitudes:
        two dominant factors, one moderate, and two minor trace-level factors.
        The Bayesian approach quantifies how uncertain the minor components
        are — something point-estimate solvers (ACLS, Newton) cannot do.
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

    from pmf_acls import (
        pmf_bayes,
        effective_sample_size,
        gelman_rubin,
        compute_waic,
    )

    return (
        compute_waic,
        effective_sample_size,
        gelman_rubin,
        linear_sum_assignment,
        mo,
        np,
        plt,
        pmf_bayes,
    )


@app.cell
def _(np):
    def make_synthetic(rng, m=25, n=80, noise_frac=0.10):
        """
        5-source problem with realistic major/minor structure.

        Each source loads on a random subset of variables (no index
        autocorrelation), mimicking real PMF where variable ordering
        is arbitrary (e.g. chemical species).

        Sources:
          0  Combustion       — dominant, loads on ~40% of variables
          1  Mineral dust     — dominant, loads on ~30% of variables
          2  Sea salt         — moderate, loads on ~25% of variables
          3  Industrial trace — minor, loads on 2-3 variables
          4  Biomass burning  — minor, loads on 2-3 variables
        """
        p = 5
        source_names = [
            "Combustion", "Mineral dust", "Sea salt",
            "Industrial trace", "Biomass burning",
        ]
        var_names = [f"V{i+1:02d}" for i in range(m)]

        F = np.full((m, p), 0.01)  # small baseline

        # Major: Combustion — ~40% of variables with varying loads
        _n_load = max(2, int(round(0.4 * m)))
        _vars0 = rng.choice(m, size=_n_load, replace=False)
        F[_vars0, 0] += rng.exponential(0.5, size=_n_load)

        # Major: Mineral dust — ~30% of variables
        _n_load = max(2, int(round(0.3 * m)))
        _vars1 = rng.choice(m, size=_n_load, replace=False)
        F[_vars1, 1] += rng.exponential(0.4, size=_n_load)

        # Moderate: Sea salt — ~25% of variables
        _n_load = max(2, int(round(0.25 * m)))
        _vars2 = rng.choice(m, size=_n_load, replace=False)
        F[_vars2, 2] += rng.exponential(0.3, size=_n_load)

        # Minor: Industrial trace — 2-3 variables, sharp spikes
        _n_load = min(3, m)
        _vars3 = rng.choice(m, size=_n_load, replace=False)
        F[_vars3, 3] += np.array([0.6, 0.3, 0.2])[:_n_load]

        # Minor: Biomass burning — 2-3 variables, sharp spikes
        _n_load = min(3, m)
        _vars4 = rng.choice(m, size=_n_load, replace=False)
        F[_vars4, 4] += np.array([0.4, 0.2, 0.15])[:_n_load]

        for k in range(p):
            F[:, k] /= F[:, k].sum()

        scales = np.array([50.0, 40.0, 15.0, 3.0, 2.0])
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


# --- Section 1: Synthetic Data ---


@app.cell
def _(mo):
    mo.md("## 1. Synthetic Data")
    return


@app.cell
def _(mo):
    noise_slider = mo.ui.slider(
        0.02, 0.30, step=0.02, value=0.10, label="Noise fraction"
    )
    n_vars_slider = mo.ui.slider(
        10, 30, step=1, value=20, label="Variables (m)"
    )
    n_obs_slider = mo.ui.slider(
        40, 200, step=20, value=100, label="Observations (n)"
    )
    seed_number = mo.ui.number(value=2026, label="Random seed")
    data_controls = mo.hstack(
        [noise_slider, n_vars_slider, n_obs_slider, seed_number], justify="start", gap=1.5
    )
    data_controls
    return noise_slider, n_vars_slider, n_obs_slider, seed_number


@app.cell
def _(make_synthetic, mo, noise_slider, n_vars_slider, n_obs_slider, np, seed_number):
    _rng = np.random.default_rng(seed_number.value)
    X, sigma, F_true, G_true, source_names, var_names = make_synthetic(
        _rng, m=n_vars_slider.value, n=n_obs_slider.value, noise_frac=noise_slider.value
    )
    _m, _n = X.shape
    _p_true = F_true.shape[1]

    # Compute SNR: power ratio of true signal to noise standard deviation
    _X_true = F_true @ G_true
    _snr_power = np.sum(_X_true**2) / np.sum(sigma**2)
    _snr_db = 10 * np.log10(_snr_power)

    mo.md(
        f"""
        **Data**: {_m} variables × {_n} observations, {_p_true} true sources
        &nbsp;|&nbsp; **Noise fraction = {noise_slider.value:.2f}
        → SNR = {_snr_power:.1f} ({_snr_db:.1f} dB)**

        | Source | Mean contribution |
        |--------|:-----------------:|
        | {source_names[0]} (major) | {G_true[0].mean():.0f} |
        | {source_names[1]} (major) | {G_true[1].mean():.0f} |
        | {source_names[2]} (moderate) | {G_true[2].mean():.0f} |
        | {source_names[3]} (**minor**) | {G_true[3].mean():.1f} |
        | {source_names[4]} (**minor**) | {G_true[4].mean():.1f} |
        """
    )
    return X, sigma, F_true, G_true, source_names, var_names


# --- Section 2: Bayesian NMF with full posterior ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. Bayesian NMF — Posterior Mean with Credible Intervals

        The Bayesian solver runs a multi-seed ACLS point estimate (stored in
        `result.F`) and then refines via Gibbs sampling.  The plots below use
        `result.F_posterior_mean` / `result.G_posterior_mean` as the central
        value so that the ±2σ credible intervals (from `result.F_std`) are on
        the same scale.  Minor components should have much wider credible
        intervals.
        """
    )
    return


@app.cell
def _(mo):
    p_slider = mo.ui.slider(2, 8, step=1, value=5, label="Factors (p)")
    ns_slider = mo.ui.slider(
        200, 4000, step=200, value=2000, label="Posterior samples"
    )
    nb_slider = mo.ui.slider(
        200, 4000, step=200, value=2000, label="Burn-in sweeps"
    )
    run_btn = mo.ui.run_button(label="Run Bayesian NMF")
    bayes_controls = mo.hstack(
        [p_slider, ns_slider, nb_slider, run_btn], justify="start", gap=1.5
    )
    bayes_controls
    return p_slider, ns_slider, nb_slider, run_btn


@app.cell
def _(
    X,
    sigma,
    effective_sample_size,
    mo,
    nb_slider,
    ns_slider,
    p_slider,
    pmf_bayes,
    run_btn,
    seed_number,
):
    mo.stop(not run_btn.value, mo.md("*Click **Run Bayesian NMF** above*"))
    result = pmf_bayes(
        X, sigma,
        p=p_slider.value,
        n_samples=ns_slider.value,
        n_burnin=nb_slider.value,
        store_samples=True,
        random_seed=seed_number.value,
        verbose=False,
    )
    _ess = effective_sample_size(result.Q_samples)
    _cd = result.convergence_details or {}
    mo.md(
        f"""
        **Result**: Q = {result.Q:.4e} &nbsp;|&nbsp;
        Explained variance = {result.explained_variance:.2%} &nbsp;|&nbsp;
        chi² = {result.chi2:.3f} &nbsp;|&nbsp;
        ESS = {_ess:.0f}/{len(result.Q_samples)} &nbsp;|&nbsp;
        Converged = {result.converged}
        {f"&nbsp;|&nbsp; Geweke z = {_cd['geweke_z']:.2f}" if 'geweke_z' in _cd else ''}
        {f"&nbsp;|&nbsp; Min factor ESS = {_cd['min_factor_ess']:.0f}" if 'min_factor_ess' in _cd else ''}
        {f"&nbsp;|&nbsp; Label gap = {result.label_switch_gap:.3f}" if result.label_switch_gap is not None else ''}
        """
    )
    return (result,)


@app.cell
def _(
    F_true,
    linear_sum_assignment,
    match_factors,
    np,
    plt,
    result,
    source_names,
    var_names,
):
    # Factor profiles with 95% credible intervals — bar chart
    _row, _col, _corr = match_factors(
        result.F, F_true, linear_sum_assignment
    )
    # Sort by true factor index so subplot order is consistent across solvers
    _order = np.argsort(_col)
    _row = _row[_order]
    _col = _col[_order]

    _m, _p = result.F_posterior_mean.shape
    _p_show = min(_p, len(source_names))
    _x = np.arange(_m)

    # Export the canonical true-factor ordering for other cells
    factor_plot_order = _col[:_p_show].copy()

    _ncols = min(_p_show, 2)
    _nrows = (_p_show + _ncols - 1) // _ncols
    _fig = plt.figure(figsize=(max(5, 0.8 * _m) * _ncols, 8 * _nrows),
                       constrained_layout=True)
    _fig.suptitle(
        "Factor Profiles — Posterior Mean ± 95% CI vs Truth", fontsize=39
    )

    for _idx in range(_p_show):
        if _idx >= len(_row):
            break
        _k_est = _row[_idx]
        _k_true = _col[_idx]
        _ax = _fig.add_subplot(_nrows, _ncols, _idx + 1)

        _f_mean = result.F_posterior_mean[:, _k_est]
        _f_std = result.F_std[:, _k_est]

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
            colors="0.2", linewidth=2.5, label="Truth",
        )
        _r = abs(_corr[_k_est, _k_true])
        _ax.set_title(f"{source_names[_k_true]}\nr = {_r:.3f}", fontsize=30)
        _ax.set_xticks(_x)
        _ax.set_xticklabels(var_names, rotation=90, fontsize=18)
        if _idx % _ncols == 0:
            _ax.set_ylabel("F profile")
        if _idx == 0:
            _ax.legend(fontsize=21, loc="upper right")

    _fig
    return (factor_plot_order,)


@app.cell
def _(
    X,
    sigma,
    effective_sample_size,
    mo,
    nb_slider,
    ns_slider,
    p_slider,
    pmf_bayes,
    run_btn,
    seed_number,
):
    mo.stop(not run_btn.value, mo.md(""))

    result_full = pmf_bayes(
        X, sigma,
        p=p_slider.value,
        n_samples=ns_slider.value,
        n_burnin=nb_slider.value,
        warm_start=False,
        volume_alpha=3.0,
        store_samples=True,
        random_seed=seed_number.value,
        verbose=False,
    )
    _ess = effective_sample_size(result_full.Q_samples)
    _vol_rate = result_full.mh_volume_acceptance_rate
    _cd = result_full.convergence_details or {}
    mo.md(
        f"""
        **Full Bayesian** (warm_start=False, volume_alpha=3.0):
        Q = {result_full.Q:.4e} &nbsp;|&nbsp;
        Explained variance = {result_full.explained_variance:.2%} &nbsp;|&nbsp;
        chi² = {result_full.chi2:.3f} &nbsp;|&nbsp;
        ESS = {_ess:.0f}/{len(result_full.Q_samples)} &nbsp;|&nbsp;
        Vol MH accept = {_vol_rate:.0%} &nbsp;|&nbsp;
        Converged = {result_full.converged}
        {f"&nbsp;|&nbsp; Geweke z = {_cd['geweke_z']:.2f}" if 'geweke_z' in _cd else ''}
        {f"&nbsp;|&nbsp; Min factor ESS = {_cd['min_factor_ess']:.0f}" if 'min_factor_ess' in _cd else ''}
        {f"&nbsp;|&nbsp; Label gap = {result_full.label_switch_gap:.3f}" if result_full.label_switch_gap is not None else ''}
        """
    )
    return (result_full,)


@app.cell
def _(
    F_true,
    factor_plot_order,
    linear_sum_assignment,
    match_factors,
    np,
    plt,
    result_full,
    source_names,
    var_names,
):
    # Full Bayesian factor profiles — bar chart with credible intervals
    _row, _col, _corr = match_factors(
        result_full.F_posterior_mean, F_true, linear_sum_assignment
    )
    _true_to_est = dict(zip(_col, _row))

    _m = result_full.F_posterior_mean.shape[0]
    _p_show = len(factor_plot_order)
    _x = np.arange(_m)

    _ncols = min(_p_show, 2)
    _nrows = (_p_show + _ncols - 1) // _ncols
    _fig = plt.figure(
        figsize=(max(5, 0.8 * _m) * _ncols, 8 * _nrows),
        constrained_layout=True,
    )
    _fig.suptitle(
        "Full Bayesian (volume prior) — Posterior Mean ± 95% CI vs Truth",
        fontsize=39,
    )

    for _idx, _k_true in enumerate(factor_plot_order):
        _k_est = _true_to_est.get(_k_true)
        _ax = _fig.add_subplot(_nrows, _ncols, _idx + 1)

        if _k_est is not None:
            _f_mean = result_full.F_posterior_mean[:, _k_est]
            _f_std = result_full.F_std[:, _k_est]
            _ax.bar(
                _x, _f_mean, 0.6,
                yerr=2 * _f_std, capsize=2,
                color="none", edgecolor="C5", linewidth=1.2,
                error_kw=dict(ecolor="C5", lw=0.8),
                label="Full Bayes ± 95% CI",
            )
            _r = abs(_corr[_k_est, _k_true])
        else:
            _r = 0.0

        _ax.hlines(
            F_true[:, _k_true], _x - 0.3, _x + 0.3,
            colors="0.2", linewidth=2.5, label="Truth",
        )
        _ax.set_title(f"{source_names[_k_true]}\nr = {_r:.3f}", fontsize=30)
        _ax.set_xticks(_x)
        _ax.set_xticklabels(var_names, rotation=90, fontsize=18)
        if _idx % _ncols == 0:
            _ax.set_ylabel("F profile")
        if _idx == 0:
            _ax.legend(fontsize=21, loc="upper right")

    _fig
    return


# --- Section 3: Minor factor zoom ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. Minor Factor Zoom — Posterior Credible Intervals

        The two minor sources (Industrial trace, Biomass burning) contribute
        ~3-5% of the signal. The credible intervals from the Gibbs posterior
        show how uncertainty spreads out where the signal is weak.
        """
    )
    return


@app.cell
def _(
    F_true,
    linear_sum_assignment,
    match_factors,
    mo,
    np,
    plt,
    result,
    source_names,
    var_names,
):
    mo.stop(
        result.F_samples is None,
        mo.md("*No stored samples — increase sample count and re-run.*"),
    )

    _row, _col, _corr = match_factors(
        result.F, F_true, linear_sum_assignment
    )
    _minor_pairs = sorted(
        [(r, c) for r, c in zip(_row, _col) if c >= 3],
        key=lambda x: x[1],
    )
    _m = result.F_posterior_mean.shape[0]
    _x = np.arange(_m)

    _fig, _axes = plt.subplots(
        1, len(_minor_pairs),
        figsize=(max(6, 0.6 * _m) * len(_minor_pairs), 4.5),
        constrained_layout=True,
    )
    _fig.suptitle(
        "Minor Factor Profiles — Posterior Mean ± 95% CI",
        fontsize=39, fontweight="bold",
    )
    if len(_minor_pairs) == 1:
        _axes = [_axes]

    for _ax, (_k_est, _k_true) in zip(_axes, _minor_pairs):
        _f_mean = result.F_posterior_mean[:, _k_est]
        _f_std = result.F_std[:, _k_est]

        # Same hollow bars + 95% CI whiskers as the Section 2 charts
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
            colors="0.2", linewidth=2.5, label="Truth",
        )
        _r = abs(_corr[_k_est, _k_true])
        _ax.set_title(
            f"{source_names[_k_true]}  (r={_r:.3f})", fontsize=33
        )
        _ax.set_xticks(_x)
        _ax.set_xticklabels(var_names, rotation=90, fontsize=18)
        _ax.set_ylabel("F profile value")
        _ax.legend(fontsize=21, loc="upper left")

        _sig_mask = F_true[:, _k_true] > 0.05
        if _sig_mask.any():
            _ms = result.F_posterior_mean[_sig_mask, _k_est].mean()
            _ss = result.F_std[_sig_mask, _k_est].mean()
            _cv = _ss / max(_ms, 1e-10)
            _ax.text(
                0.98, 0.95, f"Signal-region CV = {_cv:.0%}",
                transform=_ax.transAxes, ha="right", va="top", fontsize=27,
                bbox=dict(
                    boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9
                ),
            )

    _fig
    return


@app.cell
def _(
    F_true,
    linear_sum_assignment,
    match_factors,
    mo,
    np,
    plt,
    result_full,
    source_names,
    var_names,
):
    mo.stop(
        result_full.F_samples is None,
        mo.md("*No stored samples — increase sample count and re-run.*"),
    )

    _row, _col, _corr = match_factors(
        result_full.F_posterior_mean, F_true, linear_sum_assignment
    )
    _minor_pairs = sorted(
        [(r, c) for r, c in zip(_row, _col) if c >= 3],
        key=lambda x: x[1],
    )
    _m = result_full.F_posterior_mean.shape[0]
    _x = np.arange(_m)

    _fig, _axes = plt.subplots(
        1, len(_minor_pairs),
        figsize=(max(6, 0.6 * _m) * len(_minor_pairs), 4.5),
        constrained_layout=True,
    )
    _fig.suptitle(
        "Minor Factor Profiles — Full Bayesian (vol \u03b1=3) ± 95% CI",
        fontsize=39, fontweight="bold",
    )
    if len(_minor_pairs) == 1:
        _axes = [_axes]

    for _ax, (_k_est, _k_true) in zip(_axes, _minor_pairs):
        _f_mean = result_full.F_posterior_mean[:, _k_est]
        _f_std = result_full.F_std[:, _k_est]

        _ax.bar(
            _x, _f_mean, 0.6,
            yerr=2 * _f_std, capsize=2,
            color="none", edgecolor="C5", linewidth=1.2,
            error_kw=dict(ecolor="C5", lw=0.8),
            label="Full Bayes ± 95% CI",
        )
        _ax.hlines(
            F_true[:, _k_true], _x - 0.3, _x + 0.3,
            colors="0.2", linewidth=2.5, label="Truth",
        )
        _r = abs(_corr[_k_est, _k_true])
        _ax.set_title(
            f"{source_names[_k_true]}  (r={_r:.3f})", fontsize=33
        )
        _ax.set_xticks(_x)
        _ax.set_xticklabels(var_names, rotation=90, fontsize=18)
        _ax.set_ylabel("F profile value")
        _ax.legend(fontsize=21, loc="upper left")

        _sig_mask = F_true[:, _k_true] > 0.05
        if _sig_mask.any():
            _ms = result_full.F_posterior_mean[_sig_mask, _k_est].mean()
            _ss = result_full.F_std[_sig_mask, _k_est].mean()
            _cv = _ss / max(_ms, 1e-10)
            _ax.text(
                0.98, 0.95, f"Signal-region CV = {_cv:.0%}",
                transform=_ax.transAxes, ha="right", va="top", fontsize=27,
                bbox=dict(
                    boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9
                ),
            )

    _fig
    return


# --- Section 4: Uncertainty comparison ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 4. Uncertainty Comparison — Major vs Minor

        The coefficient of variation (CV = σ/μ) quantifies relative
        uncertainty. Minor factors have much higher CV: the Bayesian
        posterior honestly reports that weak signals are less certain.
        """
    )
    return


@app.cell
def _(
    F_true,
    linear_sum_assignment,
    match_factors,
    np,
    plt,
    result,
    source_names,
):
    _row, _col, _ = match_factors(result.F, F_true, linear_sum_assignment)

    _names, _g_cvs, _f_cvs = [], [], []
    for _ri, _ci in sorted(zip(_row, _col), key=lambda x: x[1]):
        _names.append(source_names[_ci])
        _gm = result.F_posterior_mean[:, _ri]
        _gs = result.F_std[:, _ri]
        _g_cvs.append(np.mean(_gs / np.maximum(_gm, 1e-10)))
        _fm = result.G_posterior_mean[_ri, :]
        _fs = result.G_std[_ri, :]
        _f_cvs.append(np.mean(_fs / np.maximum(_fm, 1e-10)))

    _colors = ["C0", "C1", "C2", "C4", "C5"][: len(_names)]

    _fig, (_ax1, _ax2) = plt.subplots(
        1, 2, figsize=(11, 3.5), constrained_layout=True
    )
    _fig.suptitle(
        "Uncertainty Scales with Factor Magnitude",
        fontsize=39, fontweight="bold",
    )

    _ax1.barh(range(len(_names)), _g_cvs, color=_colors)
    _ax1.set_yticks(range(len(_names)))
    _ax1.set_yticklabels(_names)
    _ax1.set_xlabel("Mean CV (σ/μ) of F profile")
    _ax1.set_title("Profile uncertainty")
    _ax1.invert_yaxis()
    for _i, _v in enumerate(_g_cvs):
        _ax1.text(_v + 0.01, _i, f"{_v:.0%}", va="center", fontsize=27)

    _ax2.barh(range(len(_names)), _f_cvs, color=_colors)
    _ax2.set_yticks(range(len(_names)))
    _ax2.set_yticklabels(_names)
    _ax2.set_xlabel("Mean CV (σ/μ) of G contributions")
    _ax2.set_title("Contribution uncertainty")
    _ax2.invert_yaxis()
    for _i, _v in enumerate(_f_cvs):
        _ax2.text(_v + 0.01, _i, f"{_v:.0%}", va="center", fontsize=27)

    _fig
    return


# --- Section 5: ARD factor selection ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 5. ARD — Automatic Factor Selection

        Over-specify the number of factors and let Automatic Relevance
        Determination prune the unnecessary ones. Each factor gets its own
        prior rate; factors not supported by data have their rates driven
        large, effectively zeroing them.

        Both the **ACLS warm-start** (default) and **Full Bayesian** (volume
        prior, random init) approaches are shown side-by-side for comparison.
        """
    )
    return


@app.cell
def _(mo):
    ard_p_slider = mo.ui.slider(5, 12, step=1, value=8, label="Max factors")
    ard_shape_slider = mo.ui.slider(
        0.1, 2.0, step=0.1, value=0.5, label="hyperparam_shape (< 1 = aggressive)"
    )
    ard_run_btn = mo.ui.run_button(label="Run ARD")
    ard_controls = mo.hstack(
        [ard_p_slider, ard_shape_slider, ard_run_btn], justify="start", gap=1.5
    )
    ard_controls
    return ard_p_slider, ard_shape_slider, ard_run_btn


@app.cell
def _(
    X,
    sigma,
    ard_p_slider,
    ard_run_btn,
    ard_shape_slider,
    mo,
    np,
    plt,
    pmf_bayes,
    seed_number,
):
    mo.stop(not ard_run_btn.value, mo.md("*Click **Run ARD** above*"))

    # --- ACLS warm-start ARD ---
    ard_result = pmf_bayes(
        X, sigma,
        p=ard_p_slider.value,
        n_samples=1500,
        n_burnin=1000,
        ard=True,
        hyperparam_shape=ard_shape_slider.value,
        random_seed=seed_number.value,
        verbose=False,
    )

    # --- Full Bayesian ARD (volume prior, random init) ---
    ard_result_full = pmf_bayes(
        X, sigma,
        p=ard_p_slider.value,
        n_samples=1500,
        n_burnin=1000,
        ard=True,
        hyperparam_shape=ard_shape_slider.value,
        warm_start=False,
        volume_alpha=3.0,
        random_seed=seed_number.value,
        verbose=False,
    )

    _p_ard = ard_result.F.shape[1]

    # Helper to compute contributions and colors
    def _ard_contribs_colors(res):
        _p = res.F.shape[1]
        contribs = np.array([
            res.F[:, k].sum() * res.G[k, :].sum() for k in range(_p)
        ])
        colors = [
            "C0" if res.active_factors[k] else "0.75" for k in range(_p)
        ]
        return contribs, colors

    _contribs_ws, _colors_ws = _ard_contribs_colors(ard_result)
    _contribs_fb, _colors_fb = _ard_contribs_colors(ard_result_full)

    _fig, _axes = plt.subplots(
        2, 2, figsize=(13, 7), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.5, 1]},
    )

    # --- Top row: ACLS warm-start ---
    _axes[0, 0].barh(range(_p_ard), _contribs_ws, color=_colors_ws)
    _axes[0, 0].set_yticks(range(_p_ard))
    _axes[0, 0].set_yticklabels([
        f"Factor {k} {'✓' if ard_result.active_factors[k] else '✗'}"
        for k in range(_p_ard)
    ])
    _axes[0, 0].set_xlabel("Total contribution (F_col × G_row)")
    _axes[0, 0].set_title(
        f"ACLS warm-start: {ard_result.effective_p}/{_p_ard} active",
        fontweight="bold",
    )
    _axes[0, 0].invert_yaxis()

    _x_k = np.arange(_p_ard)
    _axes[0, 1].bar(_x_k - 0.15, ard_result.ard_lambda_F, 0.3, label="λ_G", color="C1")
    _axes[0, 1].bar(_x_k + 0.15, ard_result.ard_lambda_G, 0.3, label="λ_F", color="C2")
    _axes[0, 1].set_xticks(_x_k)
    _axes[0, 1].set_xlabel("Factor")
    _axes[0, 1].set_ylabel("Rate parameter λ")
    _axes[0, 1].set_title("Prior rates (ACLS warm-start)")
    _axes[0, 1].legend(fontsize=27)

    # --- Bottom row: Full Bayesian ---
    _axes[1, 0].barh(range(_p_ard), _contribs_fb, color=_colors_fb)
    _axes[1, 0].set_yticks(range(_p_ard))
    _axes[1, 0].set_yticklabels([
        f"Factor {k} {'✓' if ard_result_full.active_factors[k] else '✗'}"
        for k in range(_p_ard)
    ])
    _axes[1, 0].set_xlabel("Total contribution (F_col × G_row)")
    _axes[1, 0].set_title(
        f"Full Bayesian (vol α=3): {ard_result_full.effective_p}/{_p_ard} active",
        fontweight="bold",
    )
    _axes[1, 0].invert_yaxis()

    _axes[1, 1].bar(_x_k - 0.15, ard_result_full.ard_lambda_F, 0.3, label="λ_G", color="C1")
    _axes[1, 1].bar(_x_k + 0.15, ard_result_full.ard_lambda_G, 0.3, label="λ_F", color="C2")
    _axes[1, 1].set_xticks(_x_k)
    _axes[1, 1].set_xlabel("Factor")
    _axes[1, 1].set_ylabel("Rate parameter λ")
    _axes[1, 1].set_title("Prior rates (Full Bayesian)")
    _axes[1, 1].legend(fontsize=27)

    _fig.suptitle("ARD Factor Selection Comparison", fontsize=42, fontweight="bold")
    _fig
    return ard_result, ard_result_full


# --- Section 6: WAIC model comparison ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 6. WAIC — Model Comparison

        Compare models with different numbers of factors using the Widely
        Applicable Information Criterion. Lower WAIC is better; models
        within 1 standard error of the best are statistically equivalent.

        Both **ACLS warm-start** and **Full Bayesian** (volume prior) curves
        are shown side-by-side for comparison.
        """
    )
    return


@app.cell
def _(mo):
    waic_pmin = mo.ui.slider(2, 5, step=1, value=3, label="Min p")
    waic_pmax = mo.ui.slider(5, 10, step=1, value=8, label="Max p")
    waic_run_btn = mo.ui.run_button(label="Run WAIC comparison")
    waic_controls = mo.hstack(
        [waic_pmin, waic_pmax, waic_run_btn], justify="start", gap=1.5
    )
    waic_controls
    return waic_pmin, waic_pmax, waic_run_btn


@app.cell
def _(
    X,
    sigma,
    compute_waic,
    mo,
    np,
    plt,
    pmf_bayes,
    seed_number,
    waic_pmax,
    waic_pmin,
    waic_run_btn,
):
    mo.stop(
        not waic_run_btn.value, mo.md("*Click **Run WAIC comparison** above*")
    )

    _p_range = range(waic_pmin.value, waic_pmax.value + 1)

    # --- ACLS warm-start WAIC ---
    waic_results = {}
    _rows_ws = []
    for _p in _p_range:
        _r = pmf_bayes(
            X, sigma, _p,
            n_samples=500, n_burnin=500,
            store_samples=True,
            random_seed=seed_number.value,
            verbose=False,
        )
        _w = compute_waic(X, sigma, _r.F_samples, _r.G_samples)
        waic_results[_p] = _w
        _min_ess = _w.get('min_ess', '—')
        _min_ess_str = f"{_min_ess:.0f}" if isinstance(_min_ess, (int, float)) else _min_ess
        _rows_ws.append(
            f"| {_p} | {_w['waic']:.1f} | {_w['p_waic']:.1f} | {_w['se']:.1f} | {_min_ess_str} | {_r.Q:.2e} |"
        )

    # --- Full Bayesian WAIC ---
    waic_results_full = {}
    _rows_fb = []
    for _p in _p_range:
        _r = pmf_bayes(
            X, sigma, _p,
            n_samples=500, n_burnin=500,
            store_samples=True,
            warm_start=False,
            volume_alpha=3.0,
            random_seed=seed_number.value,
            verbose=False,
        )
        _w = compute_waic(X, sigma, _r.F_samples, _r.G_samples)
        waic_results_full[_p] = _w
        _min_ess = _w.get('min_ess', '—')
        _min_ess_str = f"{_min_ess:.0f}" if isinstance(_min_ess, (int, float)) else _min_ess
        _rows_fb.append(
            f"| {_p} | {_w['waic']:.1f} | {_w['p_waic']:.1f} | {_w['se']:.1f} | {_min_ess_str} | {_r.Q:.2e} |"
        )

    # --- Best-p for each ---
    _best_p_ws = min(waic_results, key=lambda k: waic_results[k]["waic"])
    _best_waic_ws = waic_results[_best_p_ws]["waic"]
    _best_se_ws = waic_results[_best_p_ws]["se"]
    _equiv_ws = sorted(
        [p_ for p_, w in waic_results.items() if w["waic"] <= _best_waic_ws + _best_se_ws]
    )

    _best_p_fb = min(waic_results_full, key=lambda k: waic_results_full[k]["waic"])
    _best_waic_fb = waic_results_full[_best_p_fb]["waic"]
    _best_se_fb = waic_results_full[_best_p_fb]["se"]
    _equiv_fb = sorted(
        [p_ for p_, w in waic_results_full.items() if w["waic"] <= _best_waic_fb + _best_se_fb]
    )

    _ps = sorted(waic_results.keys())
    _waics_ws = [waic_results[p_]["waic"] for p_ in _ps]
    _ses_ws = [waic_results[p_]["se"] for p_ in _ps]
    _waics_fb = [waic_results_full[p_]["waic"] for p_ in _ps]
    _ses_fb = [waic_results_full[p_]["se"] for p_ in _ps]

    # --- Side-by-side plots ---
    _fig, (_ax1, _ax2) = plt.subplots(
        1, 2, figsize=(12, 4), constrained_layout=True, sharey=True,
    )

    _ax1.errorbar(_ps, _waics_ws, yerr=_ses_ws, fmt="o-", capsize=5, color="C3")
    _ax1.axvline(_best_p_ws, ls="--", color="C3", alpha=0.4)
    _ax1.set_xlabel("Number of factors (p)")
    _ax1.set_ylabel("WAIC (lower is better)")
    _ax1.set_title(f"ACLS warm-start — best p={_best_p_ws} (1 SE: {_equiv_ws})")

    _ax2.errorbar(_ps, _waics_fb, yerr=_ses_fb, fmt="s-", capsize=5, color="C5")
    _ax2.axvline(_best_p_fb, ls="--", color="C5", alpha=0.4)
    _ax2.set_xlabel("Number of factors (p)")
    _ax2.set_title(f"Full Bayesian (vol α=3) — best p={_best_p_fb} (1 SE: {_equiv_fb})")

    _fig.suptitle("WAIC Model Comparison", fontsize=42, fontweight="bold")

    # --- Tables ---
    _table_ws = (
        "**ACLS warm-start**\n\n"
        "| p | WAIC | p_waic | SE | min ESS | Q |\n"
        "|:-:|-----:|-------:|---:|--------:|--:|\n"
        + "\n".join(_rows_ws)
    )
    _table_fb = (
        "**Full Bayesian (volume α=3)**\n\n"
        "| p | WAIC | p_waic | SE | min ESS | Q |\n"
        "|:-:|-----:|-------:|---:|--------:|--:|\n"
        + "\n".join(_rows_fb)
    )

    mo.vstack([
        _fig,
        mo.hstack([mo.md(_table_ws), mo.md(_table_fb)], gap=2),
    ])
    return waic_results, waic_results_full


# --- Section 7: Convergence diagnostics ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 7. Convergence Diagnostics

        Run multiple independent chains and compare with Gelman-Rubin Rhat
        (should be < 1.05) and per-chain Effective Sample Size.
        """
    )
    return


@app.cell
def _(mo):
    diag_chains_slider = mo.ui.slider(2, 6, step=1, value=4, label="Chains")
    diag_run_btn = mo.ui.run_button(label="Run diagnostics")
    diag_controls = mo.hstack(
        [diag_chains_slider, diag_run_btn], justify="start", gap=1.5
    )
    diag_controls
    return diag_chains_slider, diag_run_btn


@app.cell
def _(
    X,
    sigma,
    diag_chains_slider,
    diag_run_btn,
    effective_sample_size,
    gelman_rubin,
    mo,
    np,
    p_slider,
    plt,
    pmf_bayes,
):
    mo.stop(
        not diag_run_btn.value, mo.md("*Click **Run diagnostics** above*")
    )

    _n_chains = diag_chains_slider.value
    _chains = []
    _rows = []
    for _i in range(_n_chains):
        _r = pmf_bayes(
            X, sigma,
            p=p_slider.value,
            n_samples=1000, n_burnin=500,
            random_seed=100 + _i,
            verbose=False,
        )
        _chains.append(_r)
        _ess = effective_sample_size(_r.Q_samples)
        _cd = _r.convergence_details or {}
        _mfe = f"{_cd['min_factor_ess']:.0f}" if 'min_factor_ess' in _cd else '—'
        _lg = f"{_r.label_switch_gap:.3f}" if _r.label_switch_gap is not None else '—'
        _rows.append(
            f"| {_i} | {_r.Q:.4e} | {_ess:.0f} / {len(_r.Q_samples)} | {_mfe} | {_lg} | {_r.converged} |"
        )

    _rhat = gelman_rubin(*[c.Q_samples for c in _chains])
    _rhat_ok = "OK" if _rhat < 1.05 else "NOT CONVERGED"

    _fig, _axes = plt.subplots(
        1, _n_chains, figsize=(3.5 * _n_chains, 2.5),
        constrained_layout=True, sharey=True,
    )
    _fig.suptitle(
        f"Q Traces — Rhat = {_rhat:.4f} ({_rhat_ok})", fontsize=36
    )
    if _n_chains == 1:
        _axes = [_axes]
    for _i, (_ax, _c) in enumerate(zip(_axes, _chains)):
        _ax.plot(_c.Q_samples, lw=0.4, color=f"C{_i}")
        _ess = effective_sample_size(_c.Q_samples)
        _ax.set_title(f"Chain {_i}  ESS={_ess:.0f}", fontsize=27)
        _ax.set_xlabel("Sample")
        if _i == 0:
            _ax.set_ylabel("Q")

    _table_md = (
        "| Chain | Q | ESS | Min factor ESS | Label gap | Converged |\n"
        "|:-----:|--:|----:|:--------------:|:---------:|:---------:|\n"
        + "\n".join(_rows)
        + f"\n\n**Gelman-Rubin Rhat = {_rhat:.4f}** ({_rhat_ok})"
    )

    mo.vstack([_fig, mo.md(_table_md)])
    return


# --- Section 8: Log-normal likelihood ---


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 8. Log-Normal Likelihood

        Environmental concentration data is often approximately log-normal:
        values span orders of magnitude and measurement uncertainty is
        proportional to the concentration (constant CV).

        The `likelihood='lognormal'` option fits
        $\log X_{ij} \sim \mathcal{N}\!\bigl(\log (GF)_{ij},\; \tau_{ij}^2\bigr)$
        where $\tau_{ij} = \sigma_{ij} / X_{ij}$ (coefficient of variation,
        computed internally via the delta method).

        **Sigma preparation is unchanged** — pass the same absolute-uncertainty
        matrix you would use for the Gaussian path.  The solver converts it
        internally.  The only requirement is that **X > 0 everywhere** (no
        zeros or negatives).

        Below we generate synthetic log-normal data and compare Gaussian vs
        log-normal fits.
        """
    )
    return


@app.cell
def _(mo):
    ln_run_btn = mo.ui.run_button(label="Run Gaussian vs Log-Normal comparison")
    ln_run_btn
    return (ln_run_btn,)


@app.cell
def _(ln_run_btn, make_synthetic, mo, np, pmf_bayes, seed_number):
    mo.stop(
        not ln_run_btn.value,
        mo.md("*Click **Run Gaussian vs Log-Normal comparison** above*"),
    )

    # Generate data with log-normal noise (multiplicative)
    _rng = np.random.default_rng(seed_number.value + 1)
    X_ln_true, _, F_ln_true, G_ln_true, ln_source_names, _ = make_synthetic(
        _rng, m=25, n=80, noise_frac=0.0,
    )
    # Apply multiplicative (log-normal) noise instead of additive
    _cv = 0.15  # 15% coefficient of variation
    X_ln = X_ln_true * np.exp(_rng.normal(0, _cv, size=X_ln_true.shape))
    sigma_ln = _cv * X_ln  # uncertainty proportional to value

    # Gaussian fit
    result_gauss = pmf_bayes(
        X_ln, sigma_ln, p=5,
        n_samples=1000, n_burnin=500,
        store_samples=True,
        random_seed=seed_number.value,
        likelihood="gaussian",
        verbose=False,
    )

    # Log-normal fit
    result_logn = pmf_bayes(
        X_ln, sigma_ln, p=5,
        n_samples=1000, n_burnin=500,
        store_samples=True,
        random_seed=seed_number.value,
        likelihood="lognormal",
        verbose=False,
    )

    mo.md(
        f"""
        **Synthetic log-normal data** (CV = {_cv:.0%}, 5 sources)

        | | Gaussian | Log-Normal |
        |--|:--:|:--:|
        | Explained variance | {result_gauss.explained_variance:.2%} | {result_logn.explained_variance:.2%} |
        | Q | {result_gauss.Q:.2e} | {result_logn.Q:.2e} |
        | Converged | {result_gauss.converged} | {result_logn.converged} |
        | MH acceptance | — | {result_logn.mh_acceptance_rate:.1%} |
        """
    )
    return (
        F_ln_true,
        X_ln,
        ln_source_names,
        result_gauss,
        result_logn,
        sigma_ln,
    )


@app.cell
def _(
    F_ln_true,
    linear_sum_assignment,
    ln_source_names,
    match_factors,
    np,
    plt,
    result_gauss,
    result_logn,
):
    # Side-by-side factor profiles: Gaussian vs Log-Normal — bar charts
    _m = F_ln_true.shape[0]
    _x = np.arange(_m)
    _p_true = F_ln_true.shape[1]

    _row_g, _col_g, _corr_g = match_factors(
        result_gauss.F, F_ln_true, linear_sum_assignment
    )
    _row_l, _col_l, _corr_l = match_factors(
        result_logn.F, F_ln_true, linear_sum_assignment
    )

    _p_show = min(_p_true, len(_row_g), len(_row_l))
    _ncols = min(_p_show, 2)
    _nrows_per = (_p_show + _ncols - 1) // _ncols
    _fig, _axes = plt.subplots(
        2 * _nrows_per, _ncols,
        figsize=(max(5, 0.8 * _m) * _ncols, 7 * 2 * _nrows_per),
        constrained_layout=True,
    )
    if _axes.ndim == 1:
        _axes = _axes.reshape(-1, 1)
    _fig.suptitle(
        "Factor Profiles — Gaussian vs Log-Normal on Log-Normal Data",
        fontsize=39, fontweight="bold",
    )

    for _idx in range(_p_show):
        _gr = _idx // _ncols
        _gc = _idx % _ncols
        # Gaussian row
        _k_g = _row_g[_idx]
        _k_true_g = _col_g[_idx]
        _ax = _axes[2 * _gr, _gc]
        _f_mean = result_gauss.F[:, _k_g]
        _f_std = result_gauss.F_std[:, _k_g]
        _ax.bar(
            _x, _f_mean, 0.6,
            yerr=2 * _f_std, capsize=2,
            color="none", edgecolor="C0", linewidth=1.0,
            error_kw=dict(ecolor="C0", lw=0.8),
        )
        _ax.hlines(
            F_ln_true[:, _k_true_g], _x - 0.3, _x + 0.3,
            colors="0.2", linewidth=2.5,
        )
        _r = abs(_corr_g[_k_g, _k_true_g])
        _ax.set_title(f"{ln_source_names[_k_true_g]}\nr={_r:.3f}", fontsize=27)
        _ax.set_xticks(_x)
        _ax.set_xticklabels([f"V{i+1:02d}" for i in range(_m)], rotation=90, fontsize=15)
        if _gc == 0:
            _ax.set_ylabel("Gaussian")

        # Log-normal row
        _k_l = _row_l[_idx]
        _k_true_l = _col_l[_idx]
        _ax2 = _axes[2 * _gr + 1, _gc]
        _f_mean2 = result_logn.F[:, _k_l]
        _f_std2 = result_logn.F_std[:, _k_l]
        _ax2.bar(
            _x, _f_mean2, 0.6,
            yerr=2 * _f_std2, capsize=2,
            color="none", edgecolor="C2", linewidth=1.0,
            error_kw=dict(ecolor="C2", lw=0.8),
        )
        _ax2.hlines(
            F_ln_true[:, _k_true_l], _x - 0.3, _x + 0.3,
            colors="0.2", linewidth=2.5,
        )
        _r2 = abs(_corr_l[_k_l, _k_true_l])
        _ax2.set_title(f"r={_r2:.3f}", fontsize=27)
        _ax2.set_xticks(_x)
        _ax2.set_xticklabels([f"V{i+1:02d}" for i in range(_m)], rotation=90, fontsize=15)
        if _gc == 0:
            _ax2.set_ylabel("Log-Normal")

    # Hide unused axes
    for _r_idx in range(2 * _nrows_per):
        for _c_idx in range(_ncols):
            _flat = (_r_idx // 2) * _ncols + _c_idx
            if _flat >= _p_show:
                _axes[_r_idx, _c_idx].set_visible(False)

    _fig
    return


@app.cell
def _(
    F_ln_true,
    linear_sum_assignment,
    ln_source_names,
    match_factors,
    np,
    plt,
    result_gauss,
    result_logn,
):
    # CV comparison: Gaussian vs Log-Normal by factor
    _row_g, _col_g, _ = match_factors(
        result_gauss.F, F_ln_true, linear_sum_assignment
    )
    _row_l, _col_l, _ = match_factors(
        result_logn.F, F_ln_true, linear_sum_assignment
    )

    _names_g, _cv_gauss, _cv_logn = [], [], []
    _pairs = sorted(zip(_row_g, _col_g), key=lambda x: x[1])
    _pairs_l = dict(zip(_col_l, _row_l))

    for _rg, _cg in _pairs:
        _names_g.append(ln_source_names[_cg])
        _gm = result_gauss.F[:, _rg]
        _gs = result_gauss.F_std[:, _rg]
        _cv_gauss.append(np.mean(_gs / np.maximum(_gm, 1e-10)))

        if _cg in _pairs_l:
            _rl = _pairs_l[_cg]
            _gm2 = result_logn.F[:, _rl]
            _gs2 = result_logn.F_std[:, _rl]
            _cv_logn.append(np.mean(_gs2 / np.maximum(_gm2, 1e-10)))
        else:
            _cv_logn.append(0.0)

    _y = np.arange(len(_names_g))
    _fig, _ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    _ax.barh(_y - 0.15, _cv_gauss, 0.3, label="Gaussian", color="C0")
    _ax.barh(_y + 0.15, _cv_logn, 0.3, label="Log-Normal", color="C2")
    _ax.set_yticks(_y)
    _ax.set_yticklabels(_names_g)
    _ax.set_xlabel("Mean CV (σ/μ) of F profile")
    _ax.set_title("Profile Uncertainty: Gaussian vs Log-Normal Likelihood")
    _ax.legend()
    _ax.invert_yaxis()
    _fig
    return


# --- Section 9: Summary ---


@app.cell
def _(mo):
    mo.md(
        """
        ## Key Takeaways

        1. **Posterior uncertainty scales with signal strength**: minor factors
           have CV (σ/μ) several times larger than major factors. This is a
           genuine Bayesian benefit — bootstrap from ACLS underestimates
           uncertainty for weak components.

        2. **ARD determines the number of factors**: over-specify *p*, and
           irrelevant factors are driven to zero. No ad-hoc Q/Q_expected
           threshold needed.

        3. **WAIC enables principled model comparison**: select the simplest
           model within 1 SE of the best WAIC. Unlike Q-based selection, WAIC
           penalizes model complexity.

        4. **ESS and Rhat are MCMC hygiene**: always check that chains have
           mixed (Rhat < 1.05) and that ESS is reasonable relative to sample
           count.

        5. **Log-normal likelihood for concentration data**: when data spans
           orders of magnitude with proportional uncertainty, `likelihood='lognormal'`
           fits in log-space. Pass the same sigma matrix — the solver converts
           it internally via `tau = sigma / X`. Requires X > 0 everywhere.
        """
    )
    return


if __name__ == "__main__":
    app.run()
