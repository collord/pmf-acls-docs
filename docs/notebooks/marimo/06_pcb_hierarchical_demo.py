import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # PCB Source Apportionment — Bayesian PMF with Hierarchical Uncertainty

        Synthetic PCB congener data with realistic Aroclor source profiles.
        This demo highlights **hierarchical sigma** — placing a prior on the
        measurement uncertainty itself and marginalizing over it during
        Gibbs sampling.

        Standard PMF (EPA PMF, ME-2) treats sigma as fixed and known. In
        practice, uncertainty estimates are themselves uncertain — especially
        for congeners near detection limits or estimated from analogous
        studies. The hierarchical extension propagates that
        uncertainty-about-uncertainty into the factor posteriors.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import linear_sum_assignment

    from pmf_acls import (
        pmf_bayes,
        SIGMA_CATEGORY_PRESETS,
        effective_sample_size,
        gelman_rubin,
        compute_waic,
    )

    return (
        SIGMA_CATEGORY_PRESETS,
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
    # --- PCB congener definitions ---
    # 18 congeners spanning tri-CB through deca-CB, representing a typical
    # sediment or water monitoring panel.
    CONGENER_NAMES = [
        "PCB-8", "PCB-18", "PCB-28", "PCB-44", "PCB-52",
        "PCB-66", "PCB-77", "PCB-101", "PCB-105", "PCB-118",
        "PCB-126", "PCB-128", "PCB-138", "PCB-153", "PCB-170",
        "PCB-180", "PCB-195", "PCB-206",
    ]

    SOURCE_NAMES = [
        "Aroclor 1242",
        "Aroclor 1254",
        "Aroclor 1260",
        "Atmospheric deposition",
    ]

    def make_pcb_synthetic(rng, n=60):
        """
        4-source PCB problem with realistic congener profiles.

        Sources:
          0  Aroclor 1242    — dominated by lighter congeners (tri/tetra-CB)
          1  Aroclor 1254    — mid-range congeners (penta/hexa-CB)
          2  Aroclor 1260    — heavier congeners (hexa/hepta-CB)
          3  Atmospheric dep — minor, diffuse mix biased toward lighter congeners

        Variable ordering is NOT correlated with chlorination level —
        the congener list is randomly permuted internally and the
        returned names track the permutation.
        """
        m = len(CONGENER_NAMES)
        p = len(SOURCE_NAMES)

        # Chlorine number per congener (approximate, for profile construction)
        cl_number = np.array([
            2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 8, 9,
        ], dtype=float)

        # Randomly permute variable order so index has no physical meaning
        perm = rng.permutation(m)
        cl_perm = cl_number[perm]
        var_names = [CONGENER_NAMES[i] for i in perm]

        F = np.full((m, p), 0.005)  # small baseline everywhere

        # Aroclor 1242: peaks at tri/tetra-CB (cl 2-4)
        for i in range(m):
            if cl_perm[i] <= 4:
                F[i, 0] += rng.exponential(0.4)
            elif cl_perm[i] <= 5:
                F[i, 0] += rng.exponential(0.05)

        # Aroclor 1254: peaks at penta/hexa-CB (cl 5-6)
        for i in range(m):
            if 4 <= cl_perm[i] <= 6:
                F[i, 1] += rng.exponential(0.4)
            elif cl_perm[i] == 7:
                F[i, 1] += rng.exponential(0.05)

        # Aroclor 1260: peaks at hexa/hepta-CB (cl 6-8)
        for i in range(m):
            if 6 <= cl_perm[i] <= 8:
                F[i, 2] += rng.exponential(0.4)
            elif cl_perm[i] == 9:
                F[i, 2] += rng.exponential(0.1)

        # Atmospheric deposition: minor, diffuse, biased lighter
        for i in range(m):
            weight = max(0.0, 1.0 - (cl_perm[i] - 3) * 0.15)
            F[i, 3] += rng.exponential(0.05 * weight + 0.01)

        # Normalize columns to unit L1
        for k in range(p):
            F[:, k] /= F[:, k].sum()

        # Source contributions (G): Aroclors are major, atm dep is minor
        scales = np.array([80.0, 60.0, 50.0, 8.0])
        G = np.zeros((p, n))
        for k in range(p):
            G[k, :] = rng.exponential(scales[k], size=n)

        X_true = F @ G
        return X_true, F, G, var_names

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

    return CONGENER_NAMES, SOURCE_NAMES, make_pcb_synthetic, match_factors


# --- Section 1: Synthetic Data ---


@app.cell
def _(mo):
    mo.md("## 1. Synthetic PCB Data")
    return


@app.cell
def _(mo):
    n_obs_slider = mo.ui.slider(
        20, 200, step=10, value=60, label="Samples (n)"
    )
    seed_number = mo.ui.number(value=2026, label="Random seed")
    data_controls = mo.hstack(
        [n_obs_slider, seed_number], justify="start", gap=1.5
    )
    data_controls
    return n_obs_slider, seed_number


@app.cell
def _(SOURCE_NAMES, make_pcb_synthetic, mo, np, n_obs_slider, seed_number):
    _rng = np.random.default_rng(seed_number.value)
    X_true, F_true, G_true, var_names = make_pcb_synthetic(
        _rng, n=n_obs_slider.value,
    )
    _m, _n = X_true.shape
    _p_true = F_true.shape[1]

    _snr_note = (
        "*(Noise and uncertainty are configured in Section 2 below — "
        "this is the clean signal.)*"
    )

    _rows = "\n".join(
        f"        | {SOURCE_NAMES[k]}"
        f" {'(major)' if k < 3 else '(**minor**)'}"
        f" | {G_true[k].mean():.1f} |"
        for k in range(_p_true)
    )

    mo.md(
        f"""
        **Data**: {_m} congeners x {_n} samples, {_p_true} true sources

        {_snr_note}

        | Source | Mean contribution |
        |--------|:-----------------:|
{_rows}
        """
    )
    return X_true, F_true, G_true, var_names


# --- Section 2: Per-variable uncertainty controls ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. Per-Congener Uncertainty & Category Assignment

        Set the **noise fraction** (sigma = fraction x value) and
        **uncertainty category** for each congener.  The category controls
        how tightly the hierarchical prior anchors sigma to the initial
        estimate during Bayesian sampling.

        - **measured** (alpha=10): well-calibrated, sigma barely moves
        - **mdl** (alpha=5): near detection limit, moderate flexibility
        - **analogous** (alpha=2): borrowed from similar analyte, loose
        - **guess** (alpha=1): wild guess, data dominates

        Use the **group defaults** to quickly assign all congeners, then
        override individual ones below.
        """
    )
    return


@app.cell
def _(mo):
    # Group-level default controls
    default_frac = mo.ui.slider(
        0.05, 0.50, step=0.05, value=0.10,
        label="Default noise fraction",
    )
    default_cat = mo.ui.dropdown(
        options={"measured": "measured", "mdl": "mdl",
                 "analogous": "analogous", "guess": "guess"},
        value="analogous",
        label="Default category",
    )
    mo.hstack([default_frac, default_cat], justify="start", gap=1.5)
    return default_frac, default_cat


@app.cell
def _(default_cat, default_frac, mo, var_names):
    # Per-congener overrides — fraction slider + category dropdown per var
    _cat_options = {
        "measured": "measured", "mdl": "mdl",
        "analogous": "analogous", "guess": "guess",
    }
    frac_sliders = {}
    cat_dropdowns = {}
    _rows = []
    for vn in var_names:
        frac_sliders[vn] = mo.ui.slider(
            0.05, 0.60, step=0.05, value=default_frac.value,
            label="",
        )
        cat_dropdowns[vn] = mo.ui.dropdown(
            options=_cat_options,
            value=default_cat.value,
            label="",
        )
        _rows.append(
            {
                "Congener": vn,
                "Noise fraction": frac_sliders[vn],
                "Category": cat_dropdowns[vn],
            }
        )

    unc_table = mo.ui.table(
        _rows,
        label="Per-congener uncertainty settings",
        selection=None,
    )
    unc_table
    return frac_sliders, cat_dropdowns, unc_table


@app.cell
def _(
    SIGMA_CATEGORY_PRESETS,
    X_true,
    cat_dropdowns,
    frac_sliders,
    mo,
    np,
    var_names,
):
    # Build sigma and category arrays from the UI controls
    _m, _n = X_true.shape
    _cat_name_to_int = {"measured": 0, "mdl": 1, "analogous": 2, "guess": 3}

    noise_fracs = np.array([frac_sliders[vn].value for vn in var_names])
    sigma_categories = np.array(
        [_cat_name_to_int[cat_dropdowns[vn].value] for vn in var_names],
        dtype=int,
    )
    sigma_prior_params = {
        v: SIGMA_CATEGORY_PRESETS[k]
        for k, v in _cat_name_to_int.items()
    }

    # Build sigma matrix: per-variable fraction of the signal
    sigma = noise_fracs[:, np.newaxis] * np.maximum(X_true, 0.01) + 0.005

    # Generate noisy data
    _rng_noise = np.random.default_rng(12345)
    noise = _rng_noise.normal(0, sigma)
    X = np.maximum(X_true + noise, 0.0)

    # SNR
    _snr_power = np.sum(X_true ** 2) / np.sum(sigma ** 2)
    _snr_db = 10 * np.log10(_snr_power)

    # Category summary table
    _cat_names = ["measured", "mdl", "analogous", "guess"]
    _summary_rows = []
    for ci, cn in enumerate(_cat_names):
        _mask = sigma_categories == ci
        _count = int(_mask.sum())
        if _count > 0:
            _avg_frac = noise_fracs[_mask].mean()
            _alpha = SIGMA_CATEGORY_PRESETS[cn][0]
            _congeners = ", ".join(
                vn for vn, cat in zip(var_names, sigma_categories) if cat == ci
            )
            _summary_rows.append(
                f"| {cn} | {_alpha:.0f} | {_count} | {_avg_frac:.0%} | {_congeners} |"
            )

    _table = (
        "| Category | alpha | Count | Avg fraction | Congeners |\n"
        "|----------|:-----:|:-----:|:------------:|:---------|\n"
        + "\n".join(_summary_rows)
    )

    mo.md(
        f"""
        **Noisy data generated** &nbsp;|&nbsp;
        SNR = {_snr_power:.1f} ({_snr_db:.1f} dB)

{_table}
        """
    )
    return X, sigma, noise_fracs, sigma_categories, sigma_prior_params


# --- Section 3: Bayesian NMF with hierarchical sigma ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. Full Bayesian NMF — Fixed vs Hierarchical Sigma

        Run the full Bayesian solver (volume prior, random init) twice:
        once with **fixed sigma** and once with **hierarchical sigma**
        (InvGamma prior on sigma^2, sampled per variable).  The plots use
        the posterior mean (`result.F_posterior_mean`) with Gibbs-sampled
        posterior uncertainty (`result.F_std`) so that central values and
        intervals are on the same scale.  Compare how the posterior
        uncertainty changes — especially for congeners in the "guess" and
        "analogous" categories.
        """
    )
    return


@app.cell
def _(mo):
    p_slider = mo.ui.slider(2, 8, step=1, value=4, label="Factors (p)")
    ns_slider = mo.ui.slider(
        200, 4000, step=200, value=2000, label="Posterior samples"
    )
    nb_slider = mo.ui.slider(
        200, 2000, step=200, value=1000, label="Burn-in sweeps"
    )
    run_btn = mo.ui.run_button(label="Run both models")
    bayes_controls = mo.hstack(
        [p_slider, ns_slider, nb_slider, run_btn], justify="start", gap=1.5
    )
    bayes_controls
    return p_slider, ns_slider, nb_slider, run_btn


@app.cell
def _(
    X,
    sigma,
    sigma_categories,
    sigma_prior_params,
    effective_sample_size,
    mo,
    nb_slider,
    np,
    ns_slider,
    p_slider,
    pmf_bayes,
    run_btn,
    seed_number,
):
    mo.stop(not run_btn.value, mo.md("*Click **Run both models** above*"))

    # Fixed sigma (full Bayesian — volume prior, random init)
    result_fixed = pmf_bayes(
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

    # Hierarchical sigma (full Bayesian — volume prior, random init)
    result_hier = pmf_bayes(
        X, sigma,
        p=p_slider.value,
        n_samples=ns_slider.value,
        n_burnin=nb_slider.value,
        warm_start=False,
        volume_alpha=3.0,
        store_samples=True,
        random_seed=seed_number.value,
        sigma_prior="per_variable",
        sigma_categories=sigma_categories,
        sigma_prior_params=sigma_prior_params,
        verbose=False,
    )

    _ess_f = effective_sample_size(result_fixed.Q_samples)
    _ess_h = effective_sample_size(result_hier.Q_samples)
    _cd_f = result_fixed.convergence_details or {}
    _cd_h = result_hier.convergence_details or {}
    _mfe_f = f"{_cd_f['min_factor_ess']:.0f}" if 'min_factor_ess' in _cd_f else '—'
    _mfe_h = f"{_cd_h['min_factor_ess']:.0f}" if 'min_factor_ess' in _cd_h else '—'
    _lg_f = f"{result_fixed.label_switch_gap:.3f}" if result_fixed.label_switch_gap is not None else '—'
    _lg_h = f"{result_hier.label_switch_gap:.3f}" if result_hier.label_switch_gap is not None else '—'

    mo.md(
        f"""
        | | Fixed sigma | Hierarchical sigma |
        |--|:--:|:--:|
        | Q | {result_fixed.Q:.2e} | {result_hier.Q:.2e} |
        | Explained variance | {result_fixed.explained_variance:.2%} | {result_hier.explained_variance:.2%} |
        | chi² | {result_fixed.chi2:.3f} | {result_hier.chi2:.3f} |
        | ESS | {_ess_f:.0f}/{len(result_fixed.Q_samples)} | {_ess_h:.0f}/{len(result_hier.Q_samples)} |
        | Min factor ESS | {_mfe_f} | {_mfe_h} |
        | Label gap | {_lg_f} | {_lg_h} |
        | Converged | {result_fixed.converged} | {result_hier.converged} |

        Hierarchical sigma posterior mean per variable: min={np.min(result_hier.sigma_posterior_mean):.4f},
        max={np.max(result_hier.sigma_posterior_mean):.4f}
        (initial sigma² range: {np.min(sigma**2):.4f} – {np.max(sigma**2):.4f})
        """
    )
    return result_fixed, result_hier


# --- Section 4: Profile comparison ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 4. Factor Profiles — Fixed vs Hierarchical Sigma

        Side-by-side comparison: top row is fixed sigma (standard Bayesian),
        bottom row is hierarchical sigma. The heavy black lines are truth.
        Wider credible intervals under hierarchical sigma indicate congeners
        where the model is honestly less certain because the *uncertainty
        estimate itself* was uncertain.
        """
    )
    return


@app.cell
def _(
    F_true,
    SOURCE_NAMES,
    linear_sum_assignment,
    match_factors,
    np,
    plt,
    result_fixed,
    result_hier,
    var_names,
):
    _m = F_true.shape[0]
    _x = np.arange(_m)
    _p_true = F_true.shape[1]

    _row_f, _col_f, _corr_f = match_factors(
        result_fixed.F, F_true, linear_sum_assignment
    )
    _row_h, _col_h, _corr_h = match_factors(
        result_hier.F, F_true, linear_sum_assignment
    )

    _p_show = min(_p_true, len(_row_f), len(_row_h))
    _ncols = min(_p_show, 4)
    _nrows = 2
    _fig, _axes = plt.subplots(
        _nrows, _ncols,
        figsize=(max(5, 0.7 * _m) * _ncols, 4 * _nrows),
        constrained_layout=True,
    )
    if _axes.ndim == 1:
        _axes = _axes.reshape(1, -1)
    _fig.suptitle(
        "Factor Profiles — Fixed vs Hierarchical Sigma",
        fontsize=13, fontweight="bold",
    )

    for _idx in range(_p_show):
        # Fixed sigma row
        _k_f = _row_f[_idx]
        _k_true_f = _col_f[_idx]
        _ax = _axes[0, _idx]
        _f_mean = result_fixed.F_posterior_mean[:, _k_f]
        _f_std = result_fixed.F_std[:, _k_f]
        _ax.bar(
            _x, _f_mean, 0.6,
            yerr=2 * _f_std, capsize=2,
            color="none", edgecolor="C0", linewidth=1.0,
            error_kw=dict(ecolor="C0", lw=0.8),
        )
        _ax.hlines(
            F_true[:, _k_true_f], _x - 0.3, _x + 0.3,
            colors="0.2", linewidth=2.5,
        )
        _r = abs(_corr_f[_k_f, _k_true_f])
        _ax.set_title(f"{SOURCE_NAMES[_k_true_f]}\nr={_r:.3f}", fontsize=9)
        _ax.set_xticks(_x)
        _ax.set_xticklabels(var_names, rotation=90, fontsize=5)
        if _idx == 0:
            _ax.set_ylabel("Fixed sigma")

        # Hierarchical sigma row
        _k_h = _row_h[_idx]
        _k_true_h = _col_h[_idx]
        _ax2 = _axes[1, _idx]
        _f_mean2 = result_hier.F_posterior_mean[:, _k_h]
        _f_std2 = result_hier.F_std[:, _k_h]
        _ax2.bar(
            _x, _f_mean2, 0.6,
            yerr=2 * _f_std2, capsize=2,
            color="none", edgecolor="C2", linewidth=1.0,
            error_kw=dict(ecolor="C2", lw=0.8),
        )
        _ax2.hlines(
            F_true[:, _k_true_h], _x - 0.3, _x + 0.3,
            colors="0.2", linewidth=2.5,
        )
        _r2 = abs(_corr_h[_k_h, _k_true_h])
        _ax2.set_title(f"r={_r2:.3f}", fontsize=9)
        _ax2.set_xticks(_x)
        _ax2.set_xticklabels(var_names, rotation=90, fontsize=5)
        if _idx == 0:
            _ax2.set_ylabel("Hierarchical sigma")

    _fig
    return


# --- Section 5: Sigma posterior ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 5. Posterior Sigma — How Much Did the Data Adjust Uncertainty?

        The ratio of posterior sigma to initial sigma shows where the model
        found the input uncertainties were too tight (ratio > 1) or too
        loose (ratio < 1).  Congeners in the "guess" category (alpha=1)
        should show the most movement.
        """
    )
    return


@app.cell
def _(np, plt, result_hier, sigma, sigma_categories, var_names):
    _m = len(var_names)
    _x = np.arange(_m)
    _cat_names = ["measured", "mdl", "analogous", "guess"]
    _cat_colors = ["C0", "C1", "C4", "C3"]

    # Posterior mean sigma (stored as sigma^2, take sqrt)
    _sigma_post = np.sqrt(result_hier.sigma_posterior_mean)
    # Initial sigma (mean across observations for each variable)
    _sigma_init = sigma.mean(axis=1)
    _ratio = _sigma_post / _sigma_init

    _colors = [_cat_colors[c] for c in sigma_categories]

    _fig, (_ax1, _ax2) = plt.subplots(
        2, 1, figsize=(max(8, 0.6 * _m), 6), constrained_layout=True,
    )
    _fig.suptitle("Posterior vs Initial Sigma", fontsize=13, fontweight="bold")

    # Top: absolute values
    _ax1.bar(_x - 0.2, _sigma_init, 0.4, color="0.7", label="Initial sigma")
    _ax1.bar(_x + 0.2, _sigma_post, 0.4, color=_colors, label="Posterior sigma")
    _ax1.set_xticks(_x)
    _ax1.set_xticklabels(var_names, rotation=90, fontsize=6)
    _ax1.set_ylabel("sigma (std dev)")
    _ax1.set_title("Absolute sigma: initial vs posterior mean")
    _ax1.legend(fontsize=8)

    # Bottom: ratio
    _bars = _ax2.bar(_x, _ratio, 0.6, color=_colors)
    _ax2.axhline(1.0, color="0.5", ls="--", lw=0.8)
    _ax2.set_xticks(_x)
    _ax2.set_xticklabels(var_names, rotation=90, fontsize=6)
    _ax2.set_ylabel("Posterior / Initial sigma")
    _ax2.set_title("Sigma adjustment ratio by category")
    # Legend for categories
    import matplotlib.patches as mpatches
    _patches = [
        mpatches.Patch(color=_cat_colors[i], label=_cat_names[i])
        for i in range(4)
        if np.any(sigma_categories == i)
    ]
    _ax2.legend(handles=_patches, fontsize=8)

    _fig
    return


# --- Section 6: CV comparison ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 6. Uncertainty Comparison — Fixed vs Hierarchical

        The coefficient of variation (CV = std/mean) of each factor's
        F profile, comparing fixed-sigma and hierarchical-sigma runs.
        Hierarchical sigma should widen posteriors for factors that load
        heavily on poorly-measured congeners.
        """
    )
    return


@app.cell
def _(
    F_true,
    SOURCE_NAMES,
    linear_sum_assignment,
    match_factors,
    np,
    plt,
    result_fixed,
    result_hier,
):
    _row_f, _col_f, _ = match_factors(
        result_fixed.F, F_true, linear_sum_assignment
    )
    _row_h, _col_h, _ = match_factors(
        result_hier.F, F_true, linear_sum_assignment
    )

    _names, _cv_fixed, _cv_hier = [], [], []
    _pairs_f = sorted(zip(_row_f, _col_f), key=lambda x: x[1])
    _pairs_h = dict(zip(_col_h, _row_h))

    for _rf, _cf in _pairs_f:
        _names.append(SOURCE_NAMES[_cf])
        _gm = result_fixed.F_posterior_mean[:, _rf]
        _gs = result_fixed.F_std[:, _rf]
        _cv_fixed.append(np.mean(_gs / np.maximum(_gm, 1e-10)))

        if _cf in _pairs_h:
            _rh = _pairs_h[_cf]
            _gm2 = result_hier.F_posterior_mean[:, _rh]
            _gs2 = result_hier.F_std[:, _rh]
            _cv_hier.append(np.mean(_gs2 / np.maximum(_gm2, 1e-10)))
        else:
            _cv_hier.append(0.0)

    _y = np.arange(len(_names))
    _fig, _ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
    _ax.barh(_y - 0.15, _cv_fixed, 0.3, label="Fixed sigma", color="C0")
    _ax.barh(_y + 0.15, _cv_hier, 0.3, label="Hierarchical sigma", color="C2")
    _ax.set_yticks(_y)
    _ax.set_yticklabels(_names)
    _ax.set_xlabel("Mean CV (std/mean) of F profile")
    _ax.set_title(
        "Profile Uncertainty: Fixed vs Hierarchical Sigma",
        fontsize=13, fontweight="bold",
    )
    _ax.legend(fontsize=9)
    _ax.invert_yaxis()
    for _i in range(len(_names)):
        _ax.text(
            _cv_fixed[_i] + 0.01, _i - 0.15, f"{_cv_fixed[_i]:.0%}",
            va="center", fontsize=8, color="C0",
        )
        _ax.text(
            _cv_hier[_i] + 0.01, _i + 0.15, f"{_cv_hier[_i]:.0%}",
            va="center", fontsize=8, color="C2",
        )
    _fig
    return


# --- Section 7: Contributions ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 7. Source Contributions — Hierarchical Model
        """
    )
    return


@app.cell
def _(
    F_true,
    SOURCE_NAMES,
    linear_sum_assignment,
    match_factors,
    np,
    plt,
    result_hier,
):
    _row, _col, _ = match_factors(
        result_hier.F, F_true, linear_sum_assignment
    )
    _p_show = min(result_hier.G_posterior_mean.shape[0], len(SOURCE_NAMES), len(_row))
    _n_obs = result_hier.G_posterior_mean.shape[1]
    _x_obs = np.arange(_n_obs)

    _fig, _axes = plt.subplots(
        1, _p_show, figsize=(4 * _p_show, 3), constrained_layout=True
    )
    _fig.suptitle(
        "Contributions G — Hierarchical Sigma Posterior Mean ± 95% CI",
        fontsize=13,
    )
    if _p_show == 1:
        _axes = [_axes]

    for _idx in range(_p_show):
        _k_est = _row[_idx]
        _k_true = _col[_idx]
        _ax = _axes[_idx]

        _g_mean = result_hier.G_posterior_mean[_k_est, :]
        _g_std = result_hier.G_std[_k_est, :]

        _ax.fill_between(
            _x_obs, _g_mean - 2 * _g_std, _g_mean + 2 * _g_std,
            alpha=0.2, color="C1",
        )
        _ax.plot(_x_obs, _g_mean, "C1-", lw=0.8)
        _ax.set_title(SOURCE_NAMES[_k_true], fontsize=10)
        _cv = np.mean(_g_std / np.maximum(_g_mean, 1e-10))
        _ax.text(
            0.98, 0.95, f"CV = {_cv:.0%}",
            transform=_ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )
        if _idx == 0:
            _ax.set_ylabel("Contribution")
        _ax.set_xlabel("Sample")

    _fig
    return


# --- Section 8: ARD ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 8. ARD — Automatic Factor Selection with Hierarchical Sigma

        Over-specify the number of factors and let ARD prune.  When
        combined with hierarchical sigma, the model can simultaneously
        determine the number of sources and honestly propagate
        measurement uncertainty.
        """
    )
    return


@app.cell
def _(mo):
    ard_p_slider = mo.ui.slider(4, 10, step=1, value=7, label="Max factors")
    ard_run_btn = mo.ui.run_button(label="Run ARD + hierarchical sigma")
    ard_controls = mo.hstack(
        [ard_p_slider, ard_run_btn], justify="start", gap=1.5
    )
    ard_controls
    return ard_p_slider, ard_run_btn


@app.cell
def _(
    X,
    sigma,
    sigma_categories,
    sigma_prior_params,
    ard_p_slider,
    ard_run_btn,
    mo,
    np,
    plt,
    pmf_bayes,
    seed_number,
):
    mo.stop(not ard_run_btn.value, mo.md("*Click **Run ARD** above*"))

    ard_result = pmf_bayes(
        X, sigma,
        p=ard_p_slider.value,
        n_samples=1500,
        n_burnin=1000,
        ard=True,
        hyperparam_shape=0.5,
        warm_start=False,
        volume_alpha=3.0,
        random_seed=seed_number.value,
        sigma_prior="per_variable",
        sigma_categories=sigma_categories,
        sigma_prior_params=sigma_prior_params,
        verbose=False,
    )

    _p_ard = ard_result.F.shape[1]
    _contribs = np.array([
        ard_result.F[:, k].sum() * ard_result.G[k, :].sum()
        for k in range(_p_ard)
    ])
    _colors = [
        "C0" if ard_result.active_factors[k] else "0.75"
        for k in range(_p_ard)
    ]

    _fig, (_ax1, _ax2) = plt.subplots(
        1, 2, figsize=(11, 3.5), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.5, 1]},
    )
    _fig.suptitle(
        f"ARD + Hierarchical Sigma: {ard_result.effective_p} / {_p_ard} active",
        fontsize=13, fontweight="bold",
    )

    _ax1.barh(range(_p_ard), _contribs, color=_colors)
    _ax1.set_yticks(range(_p_ard))
    _ax1.set_yticklabels([
        f"Factor {k} {'active' if ard_result.active_factors[k] else 'pruned'}"
        for k in range(_p_ard)
    ])
    _ax1.set_xlabel("Total contribution")
    _ax1.set_title("Factor contributions")
    _ax1.invert_yaxis()

    _lam_g = ard_result.ard_lambda_F
    _lam_f = ard_result.ard_lambda_G
    _x_k = np.arange(_p_ard)
    _ax2.bar(_x_k - 0.15, _lam_g, 0.3, label="lambda_F", color="C1")
    _ax2.bar(_x_k + 0.15, _lam_f, 0.3, label="lambda_G", color="C2")
    _ax2.set_xticks(_x_k)
    _ax2.set_xlabel("Factor")
    _ax2.set_ylabel("Rate parameter")
    _ax2.set_title("Per-factor prior rates")
    _ax2.legend(fontsize=9)

    _fig
    return (ard_result,)


# --- Section 9: Summary ---


@app.cell
def _(mo):
    mo.md(
        """
        ## Key Takeaways

        1. **Hierarchical sigma propagates uncertainty-about-uncertainty**:
           congeners in the "guess" and "analogous" categories have their
           sigma adjusted by the data, widening factor posteriors where the
           input uncertainties were suspect.

        2. **Per-variable sigma pooling** (recommended): each congener's
           sigma is informed by all n samples, giving tight posteriors with
           negligible computational cost.

        3. **Category presets map to PMF practice**: "measured" (alpha=10)
           for well-calibrated analytes, down to "guess" (alpha=1) for
           values that are essentially placeholders. The alpha parameter
           controls how much the data can override the initial sigma.

        4. **Combined with ARD**: hierarchical sigma + ARD simultaneously
           determines the number of sources and honestly propagates all
           sources of uncertainty into the factor posteriors.

        5. **Backward compatible**: `sigma_prior=None` (default) recovers
           the standard fixed-sigma Bayesian NMF.
        """
    )
    return


if __name__ == "__main__":
    app.run()
