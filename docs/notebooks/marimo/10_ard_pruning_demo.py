import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # ARD Factor Pruning — Default vs Full Bayesian

        Automatic Relevance Determination (ARD) over-specifies the number of
        factors and lets per-factor hyperparameters prune unnecessary ones.
        Factors not supported by the data have their exponential rates driven
        large, effectively zeroing them out.

        **The problem**: ARD uses random initialization (ACLS warm-start is
        disabled because it distributes mass across all factors, defeating
        pruning).  Without an identifiability constraint, factors can collapse
        or merge, causing ARD to prune real factors or retain duplicates.

        **The fix**: The **volume prior** (`volume_alpha > 0`) penalizes
        collinear F columns, keeping factors geometrically separated so ARD
        can cleanly decide which are real and which are noise.

        This notebook compares:

        1. **ARD alone** — `ard=True` (random init, no volume prior)
        2. **ARD + volume prior** — `ard=True, volume_alpha=2.0`

        Both start from the same random seed and over-specify p.
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

    from pmf_acls import pmf_bayes, factor_count_posterior

    return factor_count_posterior, linear_sum_assignment, mo, np, plt, pmf_bayes


@app.cell
def _(np):
    def make_synthetic(rng, m=25, n=80, noise_frac=0.10, scales=None):
        """
        Generate synthetic NMF data with configurable source strengths.

        Parameters
        ----------
        scales : array-like, optional
            Mean contribution (G scale) for each source.  Length determines
            the number of true factors.  Defaults to [50, 40, 15, 3, 2].
        """
        if scales is None:
            scales = np.array([50.0, 40.0, 15.0, 3.0, 2.0])
        scales = np.asarray(scales, dtype=float)
        p = len(scales)
        source_names = [f"Source {k+1}" for k in range(p)]
        var_names = [f"V{i+1:02d}" for i in range(m)]

        F = np.full((m, p), 0.01)

        # Each source loads on a random subset of variables
        for k in range(p):
            # Fraction of variables loaded scales with strength
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


# --- Section 1: Data ---


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
        5, 25, step=1, value=10, label="Variables (m)"
    )
    n_obs_slider = mo.ui.slider(
        40, 200, step=20, value=80, label="Observations (n)"
    )
    seed_number = mo.ui.number(value=2026, label="Random seed")
    n_sources_slider = mo.ui.slider(
        2, 8, step=1, value=5, label="Number of true sources"
    )
    data_controls = mo.hstack(
        [noise_slider, n_vars_slider, n_obs_slider, n_sources_slider, seed_number],
        justify="start", gap=1.5,
    )
    data_controls
    return noise_slider, n_vars_slider, n_obs_slider, n_sources_slider, seed_number


@app.cell
def _(mo, n_sources_slider):
    # Default scales: 1 strong, 2 intermediate, rest weak
    _n = n_sources_slider.value
    _defaults = [50.0, 20.0, 15.0, 3.0, 2.0, 1.5, 1.0, 0.5][:_n]
    source_scale_sliders = mo.ui.array([
        mo.ui.slider(
            0.5, 100.0, step=0.5, value=_defaults[k],
            label=f"Source {k+1} strength",
        )
        for k in range(_n)
    ])
    mo.vstack([
        mo.md(f"**Source strengths** ({_n} sources) — "
               "controls mean G contribution per source"),
        source_scale_sliders,
    ])
    return (source_scale_sliders,)


@app.cell
def _(
    make_synthetic, mo, noise_slider, np,
    n_obs_slider, n_vars_slider, seed_number,
    source_scale_sliders,
):
    _scales = np.array([s.value for s in source_scale_sliders])
    _rng = np.random.default_rng(seed_number.value)
    X, sigma, F_true, G_true, source_names, var_names = make_synthetic(
        _rng, m=n_vars_slider.value, n=n_obs_slider.value,
        noise_frac=noise_slider.value, scales=_scales,
    )
    p_true = F_true.shape[1]

    _X_true = F_true @ G_true
    _snr_power = np.sum(_X_true**2) / np.sum(sigma**2)
    _snr_db = 10 * np.log10(_snr_power)

    _rows = []
    for _k in range(p_true):
        _strength = "strong" if _scales[_k] >= 30 else "medium" if _scales[_k] >= 8 else "weak"
        _rows.append(
            f"| {source_names[_k]} | {_strength} | {G_true[_k].mean():.1f} |"
        )

    mo.md(
        f"""
        **Data**: {X.shape[0]} variables x {X.shape[1]} observations,
        **{p_true} true sources**, noise = {noise_slider.value:.0%},
        SNR = {_snr_power:.1f} ({_snr_db:.1f} dB)

        | Source | Type | Mean G |
        |--------|------|:------:|
        {"".join(_rows)}
        """
    )
    return X, sigma, F_true, G_true, source_names, var_names, p_true


# --- Section 2: ARD configuration ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. ARD Configuration

        Over-specify **p** well beyond the true number of factors and let ARD
        prune.  The `hyperparam_shape` controls pruning aggressiveness (< 1 =
        more aggressive).

        > **Note:** When `ard=True`, ACLS warm-start is **automatically
        > disabled** in both runs (see `bayes.py` line 1336).  ACLS distributes
        > mass across all *p* factors, which defeats ARD's ability to prune —
        > so both "ARD alone" and "ARD + volume prior" start from **random
        > initialization** with the same seed.  The only difference between
        > the two is the volume prior.
        """
    )
    return


@app.cell
def _(mo, n_vars_slider):
    _m = n_vars_slider.value
    ard_p_slider = mo.ui.slider(
        6, _m, step=1, value=min(10, _m), label=f"Max factors (p ≤ {_m})"
    )
    ard_shape_slider = mo.ui.slider(
        0.1, 2.0, step=0.1, value=0.5,
        label="hyperparam_shape (< 1 = aggressive)",
    )
    vol_alpha_slider = mo.ui.slider(
        0.5, 8.0, step=0.5, value=2.0,
        label="volume_alpha (for full Bayesian)",
    )
    activity_threshold_slider = mo.ui.slider(
        0.001, 0.10, step=0.001, value=0.01,
        label="Activity threshold (fraction)",
    )
    ard_controls = mo.hstack(
        [ard_p_slider, ard_shape_slider, vol_alpha_slider, activity_threshold_slider],
        justify="start", gap=1.5,
    )
    ard_controls
    return ard_p_slider, ard_shape_slider, vol_alpha_slider, activity_threshold_slider


# --- Section 3: Run both ---


@app.cell
def _(
    X,
    ard_p_slider,
    ard_shape_slider,
    mo,
    np,
    p_true,
    pmf_bayes,
    seed_number,
    sigma,
    vol_alpha_slider,
):
    _p = ard_p_slider.value
    _shape = ard_shape_slider.value
    _valpha = vol_alpha_slider.value

    mo.md(f"Running ARD with p={_p}, shape={_shape}...")

    # --- ARD alone (random init — warm-start auto-disabled for ARD) ---
    ard_default = pmf_bayes(
        X, sigma, p=_p,
        n_samples=1500, n_burnin=1000,
        ard=True,
        hyperparam_shape=_shape,
        random_seed=seed_number.value,
    )

    # --- ARD + volume prior (also random init) ---
    ard_volume = pmf_bayes(
        X, sigma, p=_p,
        n_samples=1500, n_burnin=1000,
        ard=True,
        hyperparam_shape=_shape,
        volume_alpha=_valpha,
        random_seed=seed_number.value,
    )

    _def_active = int(ard_default.effective_p) if ard_default.effective_p is not None else 0
    _vol_active = int(ard_volume.effective_p) if ard_volume.effective_p is not None else 0
    _cd_d = ard_default.convergence_details or {}
    _cd_v = ard_volume.convergence_details or {}
    _mfe_d = f"{_cd_d['min_factor_ess']:.0f}" if 'min_factor_ess' in _cd_d else '—'
    _mfe_v = f"{_cd_v['min_factor_ess']:.0f}" if 'min_factor_ess' in _cd_v else '—'
    _lg_d = f"{ard_default.label_switch_gap:.3f}" if ard_default.label_switch_gap is not None else '—'
    _lg_v = f"{ard_volume.label_switch_gap:.3f}" if ard_volume.label_switch_gap is not None else '—'

    _pt = p_true
    _n_real_d = np.sum(ard_default.active_factors[:_pt]) if ard_default.active_factors is not None and len(ard_default.active_factors) >= _pt else '?'
    _n_real_v = np.sum(ard_volume.active_factors[:_pt]) if ard_volume.active_factors is not None and len(ard_volume.active_factors) >= _pt else '?'

    mo.md(
        f"""
        ### Results

        | | ARD alone | ARD + volume prior (alpha={_valpha}) |
        |---|:-:|:-:|
        | **Active factors** | **{_def_active}** / {_p} | **{_vol_active}** / {_p} |
        | True factors | {_n_real_d} of {_pt} real sources active | {_n_real_v} of {_pt} real sources active |
        | Converged | {'Yes' if ard_default.converged else 'No'} | {'Yes' if ard_volume.converged else 'No'} |
        | Min factor ESS | {_mfe_d} | {_mfe_v} |
        | Label gap | {_lg_d} | {_lg_v} |
        | Q | {ard_default.Q:.2e} | {ard_volume.Q:.2e} |
        | Vol MH accept | n/a | {ard_volume.mh_volume_acceptance_rate:.0%} |
        """
    )
    return ard_default, ard_volume


# --- Section 4: Factor count posterior ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. Posterior Distribution Over Number of Factors

        Using `factor_count_posterior`, we compute the full posterior PMF over
        the number of active factors at each Gibbs sample.  This lets us make
        probabilistic statements like *"there is only a 15% chance that more
        than 5 factors are identifiable in this dataset"*.
        """
    )
    return


@app.cell
def _(activity_threshold_slider, ard_default, ard_p_slider, ard_volume, factor_count_posterior, mo, np, plt, p_true):
    _pairs = [
        ("ARD alone", ard_default),
        ("ARD + volume prior", ard_volume),
    ]

    _fig, _axes = plt.subplots(
        1, 2, figsize=(12, 4), constrained_layout=True,
        sharey=True,
    )
    _fig.suptitle(
        "Posterior P(number of active factors = k)",
        fontsize=13, fontweight="bold",
    )

    _summaries = []
    for _ax, (_title, _res) in zip(_axes, _pairs):
        _fc = factor_count_posterior(_res.factor_activity_samples, threshold=activity_threshold_slider.value)
        _p = _res.F.shape[1]
        _ks = np.arange(_p + 1)

        _colors = [
            "C0" if k == p_true else "0.70" for k in _ks
        ]
        _ax.bar(_ks, _fc["pmf"], color=_colors, edgecolor="0.4", linewidth=0.5)
        _ax.axvline(p_true, color="red", ls="--", lw=1.5, label=f"True p = {p_true}")
        _ax.set_xlabel("Number of active factors")
        _ax.set_ylabel("Posterior probability")
        _lo, _hi = _fc["ci_90"]
        _ax.set_title(
            f"{_title}\n"
            f"mean={_fc['mean']:.1f}  median={_fc['median']}  "
            f"90% CI=[{_lo}, {_hi}]",
            fontsize=9,
        )
        _ax.legend(fontsize=8)
        _ax.set_xticks(_ks)

        _summaries.append((_title, _fc))

    _fig

    # Sweep range: 2 to max factors (from ARD config slider)
    _p_max_slider = ard_p_slider.value
    _sweep = list(range(2, _p_max_slider + 1))

    # Build per-method probability tables
    _prob_tables = []
    for _title, _fc in _summaries:
        _lo, _hi = _fc["ci_90"]
        _cols = " | ".join(f"**{_k}**{'*' if _k == p_true else ''}" for _k in _sweep)
        _align = " | ".join([":-:"] * len(_sweep))
        _p_exact = " | ".join(f"{_fc['pmf'][_k]:.0%}" for _k in _sweep)
        _p_gt = " | ".join(f"{_fc['prob_gt'][_k]:.0%}" for _k in _sweep)
        _prob_tables.append(
            f"**{_title}** — mean={_fc['mean']:.1f}, "
            f"median={_fc['median']}, 90% CI=[{_lo}, {_hi}]\n\n"
            f"| p | {_cols} |\n"
            f"|---|{_align}|\n"
            f"| P(= p) | {_p_exact} |\n"
            f"| P(> p) | {_p_gt} |"
        )

    _tables_md = "\n\n".join(_prob_tables)

    # Build threshold sensitivity table
    _sens_rows = []
    for _title, _fc in _summaries:
        _ts = _fc.get("threshold_sensitivity", {})
        if _ts:
            _vals = " | ".join(f"{v:.1f}" for v in _ts.values())
            _sens_rows.append(f"| {_title} | {_vals} |")

    _sens_md = ""
    if _sens_rows:
        _ts_keys = list(_summaries[0][1].get("threshold_sensitivity", {}).keys())
        _thresh_cols = " | ".join(f"{t}" for t in _ts_keys)
        _thresh_align = " | ".join([":-:"] * len(_ts_keys))
        _sens_md = (
            "\n\n### Threshold Sensitivity\n\n"
            "How does the median active factor count change with the activity threshold?\n\n"
            f"| | {_thresh_cols} |\n"
            f"|---|{_thresh_align}|\n"
            + "\n".join(_sens_rows)
            + "\n\nStable counts across thresholds indicate a robust result."
        )

    mo.md(
        f"### Factor Count Summary\n\n"
        f"Posterior probability of exactly *p* active factors and of more than *p* "
        f"active factors, for p = {_sweep[0]}–{_sweep[-1]} "
        f"(true p = {p_true}, marked with *).\n\n"
        f"{_tables_md}\n\n"
        f"**P(> p)** answers: *\"what is the probability that more than p factors are "
        f"identifiable?\"*  Lower is better for the target count."
        f"{_sens_md}"
    )
    return


# --- Section 5: Factor contributions ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 5. Factor Contributions — Which Factors Survived?

        Horizontal bars show each factor's total contribution
        (F column sum x G row sum).  Active factors are colored;
        pruned factors are gray.
        """
    )
    return


@app.cell
def _(ard_default, ard_volume, np, plt):
    _pairs = [
        ("ARD alone", ard_default),
        ("ARD + volume prior", ard_volume),
    ]

    _fig, _axes = plt.subplots(
        1, 2, figsize=(12, 4), constrained_layout=True,
        sharey=True,
    )
    _fig.suptitle(
        "Factor Contributions — Active vs Pruned",
        fontsize=13, fontweight="bold",
    )

    for _ax, (_title, _res) in zip(_axes, _pairs):
        _p = _res.F.shape[1]
        _contribs = np.array([
            _res.F[:, k].sum() * _res.G[k, :].sum()
            for k in range(_p)
        ])
        _colors = [
            "C0" if _res.active_factors[k] else "0.80"
            for k in range(_p)
        ]
        _ax.barh(range(_p), _contribs, color=_colors)
        _ax.set_yticks(range(_p))
        _ax.set_yticklabels([
            f"F{k}" for k in range(_p)
        ])
        _ax.set_xlabel("Total contribution")
        _eff = int(_res.effective_p) if _res.effective_p is not None else 0
        _ax.set_title(f"{_title}\n{_eff} / {_p} active")
        _ax.invert_yaxis()

    _fig
    return


# --- Section 5: Per-factor lambda rates ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 6. Per-Factor Prior Rates (lambda)

        ARD drives lambda large for inactive factors (strong shrinkage toward
        zero) and keeps lambda small for active ones.  Compare how cleanly the
        two approaches separate active from pruned.
        """
    )
    return


@app.cell
def _(ard_default, ard_volume, np, plt):
    _pairs = [
        ("ARD alone", ard_default),
        ("ARD + volume prior", ard_volume),
    ]

    _fig, _axes = plt.subplots(
        1, 2, figsize=(12, 4), constrained_layout=True,
        sharey=True,
    )
    _fig.suptitle(
        "Per-Factor ARD Rates",
        fontsize=13, fontweight="bold",
    )

    for _ax, (_title, _res) in zip(_axes, _pairs):
        _p = _res.F.shape[1]
        _x_k = np.arange(_p)
        _lam_f = _res.ard_lambda_F
        _lam_g = _res.ard_lambda_G
        _active = _res.active_factors

        _colors_f = ["C1" if _active[k] else "0.80" for k in range(_p)]
        _colors_g = ["C2" if _active[k] else "0.80" for k in range(_p)]

        _ax.bar(_x_k - 0.15, _lam_f, 0.3, color=_colors_f, label="lambda_F")
        _ax.bar(_x_k + 0.15, _lam_g, 0.3, color=_colors_g, label="lambda_G")
        _ax.set_xticks(_x_k)
        _ax.set_xticklabels([f"F{k}" for k in range(_p)])
        _ax.set_ylabel("Rate parameter")
        _ax.set_title(_title)
        _ax.legend(fontsize=8)
        _ax.set_yscale("log")

    _fig
    return


# --- Section 6: Matched profiles ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 7. Recovered Profiles — Active Factors vs Truth

        For each active factor, we match to the closest true source by
        correlation.  Bars show the posterior mean; whiskers are +/- 2 sigma.
        Black lines are truth.  Only active factors are shown.
        """
    )
    return


@app.cell
def _(
    F_true,
    ard_default,
    ard_volume,
    linear_sum_assignment,
    match_factors,
    np,
    plt,
    source_names,
    var_names,
):
    _pairs = [
        ("ARD alone", ard_default),
        ("ARD + volume prior", ard_volume),
    ]

    _m = F_true.shape[0]
    _x = np.arange(_m)

    for _title, _res in _pairs:
        _F_pm = _res.F_posterior_mean
        _active = _res.active_factors
        _active_idx = np.where(_active)[0]
        _n_active = len(_active_idx)

        if _n_active == 0:
            continue

        # Match only active factors to truth
        _F_active = _F_pm[:, _active_idx]
        _row, _col, _corr = match_factors(
            _F_active, F_true, linear_sum_assignment,
        )
        _order = np.argsort(_col)
        _row = _row[_order]
        _col = _col[_order]
        _n_show = len(_row)

        _ncols = min(_n_show, 3)
        _nrows = (_n_show + _ncols - 1) // _ncols

        _fig = plt.figure(
            figsize=(max(5, 0.7 * _m) * _ncols, 3.5 * _nrows),
            constrained_layout=True,
        )
        _fig.suptitle(
            f"{_title} — {_n_active} active factors matched to truth",
            fontsize=13, fontweight="bold",
        )

        for _idx in range(_n_show):
            _k_active_local = _row[_idx]
            _k_global = _active_idx[_k_active_local]
            _k_true = _col[_idx]
            _ax = _fig.add_subplot(_nrows, _ncols, _idx + 1)

            _f_mean = _F_pm[:, _k_global]
            _f_std = _res.F_std[:, _k_global]
            _r = abs(_corr[_k_active_local, _k_true])

            _color = "C0" if _title.startswith("ARD alone") else "C2"
            _ax.bar(
                _x, _f_mean, 0.6,
                yerr=2 * _f_std, capsize=2,
                color="none", edgecolor=_color, linewidth=1.0,
                error_kw=dict(ecolor=_color, lw=0.8),
            )
            _ax.hlines(
                F_true[:, _k_true], _x - 0.3, _x + 0.3,
                colors="0.2", linewidth=2.5,
            )
            _ax.set_title(
                f"{source_names[_k_true]}  r={_r:.3f}", fontsize=9,
            )
            _ax.set_xticks(_x)
            _ax.set_xticklabels(var_names, rotation=90, fontsize=5)

        _fig

    return


# --- Section 7: Q trace ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 8. Q Trace — Convergence

        Both start from random initialization.  The volume-prior run may take
        longer to settle but should converge to a similar or better Q.
        """
    )
    return


@app.cell
def _(ard_default, ard_volume, plt):
    _fig, _ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
    _ax.plot(
        ard_default.Q_samples, color="C0", lw=1.2, alpha=0.8,
        label="ARD alone",
    )
    _ax.plot(
        ard_volume.Q_samples, color="C2", lw=1.2, alpha=0.8,
        label="ARD + volume prior",
    )
    _ax.set_xlabel("Sample index")
    _ax.set_ylabel("Q")
    _ax.set_title("Q Trace", fontsize=13, fontweight="bold")
    _ax.legend(fontsize=9)
    _ax.set_yscale("log")
    _fig
    return


# --- Section 8: Pairwise similarity ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 9. Factor Separation — Pairwise Cosine Similarity

        Heatmaps of cosine similarity between all F posterior-mean columns.
        Off-diagonal values close to 1 indicate collapsed or duplicate factors.
        The volume prior should produce a cleaner block-diagonal structure
        (distinct factors with low cross-similarity).
        """
    )
    return


@app.cell
def _(ard_default, ard_volume, np, plt):
    _pairs = [
        ("ARD alone", ard_default),
        ("ARD + volume prior", ard_volume),
    ]

    _fig, _axes = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True,
    )
    _fig.suptitle(
        "Pairwise Cosine Similarity of F Columns",
        fontsize=13, fontweight="bold",
    )

    for _ax, (_title, _res) in zip(_axes, _pairs):
        _F = _res.F_posterior_mean
        _p = _F.shape[1]
        _norms = np.linalg.norm(_F, axis=0, keepdims=True)
        _F_norm = _F / np.maximum(_norms, 1e-30)
        _sim = _F_norm.T @ _F_norm

        _im = _ax.imshow(_sim, vmin=0, vmax=1, cmap="RdYlBu_r")
        _ax.set_xticks(range(_p))
        _ax.set_yticks(range(_p))
        _ax.set_xticklabels([f"F{k}" for k in range(_p)], fontsize=7)
        _ax.set_yticklabels([f"F{k}" for k in range(_p)], fontsize=7)

        # Annotate active/pruned
        for k in range(_p):
            marker = "o" if _res.active_factors[k] else "x"
            color = "green" if _res.active_factors[k] else "red"
            _ax.plot(k, -0.7, marker, color=color, markersize=6,
                     clip_on=False, transform=_ax.transData)

        # Max off-diagonal
        _sim_offdiag = _sim.copy()
        np.fill_diagonal(_sim_offdiag, 0)
        _ax.set_title(
            f"{_title}\nmax off-diag = {_sim_offdiag.max():.3f}",
        )
        _fig.colorbar(_im, ax=_ax, shrink=0.8)

    _fig
    return


# --- Takeaways ---


@app.cell
def _(mo):
    mo.md(
        """
        ## Takeaways

        1. **ARD alone can miscount factors** — without the volume prior,
           random-init factors may collapse or merge, causing ARD to prune
           real sources or retain near-duplicate noise factors.

        2. **Volume prior + ARD is the cleanest pruning path** — the volume
           prior keeps factors geometrically separated, giving ARD a clear
           signal for which factors carry real data support and which don't.

        3. **`factor_count_posterior` quantifies uncertainty about p** — instead
           of a single point estimate for the number of factors, you get
           the full posterior distribution: P(p_active = k) for each k,
           exceedance probabilities P(p_active > k), and credible intervals.

        4. **The similarity heatmap is diagnostic** — if you see large
           off-diagonal blocks in the ARD-alone heatmap, factor collapse is
           likely.  The volume-prior heatmap should be closer to diagonal.

        5. **Tuning**: `hyperparam_shape < 1` makes pruning more aggressive.
           `volume_alpha` of 2-3 is usually sufficient.  If the volume MH
           acceptance rate drops below 10%, reduce `volume_alpha`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
