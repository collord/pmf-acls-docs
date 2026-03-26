import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # F-Element Displacement Analysis (DISP)

        The displacement method (Paatero 2013) establishes confidence
        intervals on each element of the factor profile matrix F by
        probing the Q surface:

        1. Fix all F elements except one
        2. Perturb that element away from its base value
        3. Re-optimize remaining factors (G only, or full F+G)
        4. Find the boundaries where Q increases by dQ_max

        dQ=4 corresponds to a ~95% CI (chi-squared with 1 d.f.).

        **Swap detection**: at each boundary, the G rows are compared
        to the base G via R² correlation.  If any factor's best match
        is off-diagonal, a factor swap (rotational instability) has
        occurred — meaning the solution is not uniquely determined.

        This is a **solver-agnostic** diagnostic: it only needs X, sigma,
        and a point estimate of F.
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

    from pmf_acls import pmf, displacement_test_F

    return displacement_test_F, linear_sum_assignment, mo, np, plt, pmf


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
        5, 25, step=1, value=10, label="Variables (m)"
    )
    n_obs_slider = mo.ui.slider(
        40, 200, step=20, value=80, label="Observations (n)"
    )
    seed_number = mo.ui.number(value=2026, label="Random seed")
    n_sources_slider = mo.ui.slider(
        2, 6, step=1, value=3, label="True sources"
    )
    dq_choices = mo.ui.multiselect(
        options={"1": 1.0, "4": 4.0, "8": 8.0, "16": 16.0},
        value=["4", "8", "16"],
        label="dQ thresholds",
    )
    reopt_dropdown = mo.ui.dropdown(
        options={"g_only (fast)": "g_only", "acls (Paatero 2014)": "acls", "nnls": "nnls"},
        value="g_only",
        label="Re-optimization method",
    )

    mo.vstack([
        mo.hstack(
            [noise_slider, n_vars_slider, n_obs_slider, seed_number],
            justify="start", gap=1.5,
        ),
        mo.hstack(
            [n_sources_slider, dq_choices, reopt_dropdown],
            justify="start", gap=1.5,
        ),
    ])
    return (
        dq_choices,
        n_obs_slider,
        n_sources_slider,
        n_vars_slider,
        noise_slider,
        reopt_dropdown,
        seed_number,
    )


@app.cell
def _(
    dq_choices,
    mo,
    n_obs_slider,
    n_sources_slider,
    n_vars_slider,
    noise_slider,
    reopt_dropdown,
    seed_number,
):
    _dq_str = ", ".join(str(v) for v in sorted(dq_choices.value))
    mo.md(
        f"**Data**: m={n_vars_slider.value}, n={n_obs_slider.value}, "
        f"p={n_sources_slider.value}, noise={noise_slider.value:.0%}, "
        f"seed={seed_number.value} / "
        f"**DISP**: dQ = [{_dq_str}], reopt = {reopt_dropdown.value}"
    )
    return


# --- Generate data ---


@app.cell
def _(
    make_synthetic,
    mo,
    n_obs_slider,
    n_sources_slider,
    n_vars_slider,
    noise_slider,
    np,
    seed_number,
):
    _rng = np.random.default_rng(int(seed_number.value))
    _default_scales = [50.0, 40.0, 15.0, 3.0, 2.0, 1.0]
    _scales = _default_scales[:n_sources_slider.value]
    X, sigma, F_true, G_true, source_names, var_names = make_synthetic(
        _rng,
        m=n_vars_slider.value,
        n=n_obs_slider.value,
        noise_frac=noise_slider.value,
        scales=_scales,
    )
    p_true = F_true.shape[1]

    _X_true = F_true @ G_true
    _snr_power = np.sum(_X_true**2) / np.sum(sigma**2)
    _snr_db = 10 * np.log10(_snr_power)

    mo.md(
        f"**Data**: {X.shape[0]} variables x {X.shape[1]} observations, "
        f"**{p_true} true sources**, noise = {noise_slider.value:.0%}, "
        f"SNR = {_snr_db:.1f} dB"
    )
    return F_true, G_true, X, p_true, sigma, source_names, var_names


# --- ACLS multi-seed solve ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. ACLS Point Estimate (Best of 30 Seeds)

        We run 30 random-seed ACLS solves and keep the one with the
        lowest Q (weighted residual).  This gives us the best available
        point estimate of F to feed into DISP.
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
    var_names,
):
    _row_ind, _col_ind, _ = match_factors(
        acls_result.F, F_true, linear_sum_assignment
    )
    _p = acls_result.F.shape[1]
    _m = acls_result.F.shape[0]
    _x = np.arange(_m)

    _fig, _axes = plt.subplots(1, _p, figsize=(4 * _p, 3.5), sharey=False)
    if _p == 1:
        _axes = [_axes]
    for _idx in range(_p):
        _ax = _axes[_idx]
        _k_est = _row_ind[_idx]
        _k_true = _col_ind[_idx]
        _corr = np.corrcoef(
            acls_result.F[:, _k_est], F_true[:, _k_true]
        )[0, 1]
        _ax.bar(
            _x, acls_result.F[:, _k_est],
            color="steelblue", edgecolor="steelblue",
            fill=False, linewidth=1.5, label="ACLS",
        )
        _ax.hlines(
            F_true[:, _k_true], _x - 0.35, _x + 0.35,
            colors="black", linewidths=2.5, label="True",
        )
        _ax.set_title(f"Factor {_idx+1} (|r|={abs(_corr):.3f})")
        _ax.set_xticks(_x)
        _ax.set_xticklabels(var_names, rotation=90, fontsize=7)
        _ax.legend(fontsize=7)
    _fig.suptitle("ACLS Factor Profiles vs Truth", fontsize=13)
    _fig.tight_layout()
    _fig
    return


# --- DISP analysis on ACLS ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. DISP on ACLS Solution

        For each element F[i,k], we probe the Q surface by perturbing
        that single element and re-optimizing the remaining factors.

        - **g_only**: re-optimize only G (fast, conservative intervals)
        - **acls**: re-optimize all F and G (Paatero 2014, wider intervals)
        - **nnls**: like acls but uses scipy NNLS sub-problems

        The resulting intervals show how far each element can move
        before Q increases by dQ_max.  Swap detection at boundaries
        reveals rotational instability.
        """
    )
    return


@app.cell
def _(X, acls_result, dq_choices, displacement_test_F, mo, reopt_dropdown, sigma):
    _dq_list = sorted(dq_choices.value)
    disp_result = displacement_test_F(
        X, sigma, acls_result.F,
        dQ_thresholds=_dq_list,
        reopt_method=reopt_dropdown.value,
        verbose=False,
    )

    _mn = X.shape[0] * X.shape[1]
    mo.md(
        f"**ACLS DISP complete**: Q_base = {disp_result.Q_base:.4e} "
        f"(Q/mn = {disp_result.Q_base / _mn:.3f}), "
        f"thresholds = {_dq_list}, "
        f"reopt = {disp_result.reopt_method}"
    )
    return (disp_result,)


# --- Factor profiles with DISP bounds ---


@app.cell
def _(mo):
    mo.md("### Factor Profiles with DISP Bounds (dQ=4)")
    return


@app.cell
def _(
    F_true,
    disp_result,
    linear_sum_assignment,
    match_factors,
    np,
    plt,
    var_names,
):
    _F = disp_result.F_base
    _m, _p = _F.shape
    _x = np.arange(_m)

    # Use dQ=4 if available, else smallest threshold
    _dq_plot = 4.0 if 4.0 in disp_result.F_lo else disp_result.dQ_thresholds[0]
    _F_lo = disp_result.F_lo[_dq_plot]
    _F_hi = disp_result.F_hi[_dq_plot]

    _row_ind, _col_ind, _ = match_factors(_F, F_true, linear_sum_assignment)

    _fig, _axes = plt.subplots(1, _p, figsize=(4 * _p, 4), sharey=False)
    if _p == 1:
        _axes = [_axes]
    for _idx in range(_p):
        _ax = _axes[_idx]
        _k_est = _row_ind[_idx]
        _k_true = _col_ind[_idx]
        _base = _F[:, _k_est]
        _lo = _F_lo[:, _k_est]
        _hi = _F_hi[:, _k_est]
        _yerr_lo = np.maximum(_base - _lo, 0)
        _yerr_hi = np.maximum(_hi - _base, 0)

        _ax.bar(_x, _base, color="steelblue", alpha=0.7)
        _ax.errorbar(
            _x, _base,
            yerr=[_yerr_lo, _yerr_hi],
            fmt="none", ecolor="black", capsize=3, linewidth=1.5,
        )
        _ax.hlines(
            F_true[:, _k_true], _x - 0.35, _x + 0.35,
            colors="black", linewidths=2.5, label="True",
        )
        _ax.set_title(f"Factor {_idx+1}")
        _ax.set_xticks(_x)
        _ax.set_xticklabels(var_names, rotation=90, fontsize=7)
        _ax.set_ylabel("F value")
        if _idx == 0:
            _ax.legend(fontsize=7)
    _fig.suptitle(
        f"DISP Bounds on F (dQ={_dq_plot})", fontsize=13
    )
    _fig.tight_layout()
    _fig
    return


# --- Multi-threshold comparison ---


@app.cell
def _(mo):
    mo.md("### Multi-Threshold Interval Widths")
    return


@app.cell
def _(disp_result, np, plt):
    _F = disp_result.F_base
    _m, _p = _F.shape
    _thresholds = sorted(disp_result.dQ_thresholds)

    _fig, _axes = plt.subplots(1, _p, figsize=(4 * _p, 3.5), sharey=False)
    if _p == 1:
        _axes = [_axes]
    _colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(_thresholds)))

    for _k in range(_p):
        _ax = _axes[_k]
        for _ci, _dq in enumerate(_thresholds):
            _width = disp_result.F_hi[_dq][:, _k] - disp_result.F_lo[_dq][:, _k]
            _ax.plot(
                _width, "o-", color=_colors[_ci], markersize=4,
                label=f"dQ={_dq}", alpha=0.8,
            )
        _ax.set_title(f"Factor {_k+1}")
        _ax.set_xlabel("Variable index")
        _ax.set_ylabel("Interval width")
        _ax.legend(fontsize=7)
    _fig.suptitle("DISP Interval Width by dQ Threshold", fontsize=13)
    _fig.tight_layout()
    _fig
    return


# --- Swap detection summary ---


@app.cell
def _(mo):
    mo.md(
        """
        ### Swap Detection Summary

        Factor swaps at the displacement boundary indicate rotational
        ambiguity — the perturbed element can trade identity with
        another factor while keeping Q within the threshold.

        **Swaps at dQ=4 are serious**: they mean the 95% CI boundary
        involves a factor interchange, so the factor profiles are
        not uniquely determined at that confidence level.
        """
    )
    return


@app.cell
def _(disp_result, mo, np):
    _thresholds = sorted(disp_result.dQ_thresholds)
    _F = disp_result.F_base
    _m, _p = _F.shape

    _header = "| dQ | Total swaps | Elements with swaps | Fraction | Converged |"
    _sep = "|---:|---:|---:|---:|---:|"
    _rows = [_header, _sep]
    for _dq in _thresholds:
        _sc = disp_result.swap_counts[_dq]
        _total = int(np.sum(_sc))
        _n_elements = int(np.sum(_sc > 0))
        _frac = _n_elements / (_m * _p) if _m * _p > 0 else 0.0
        _conv = disp_result.converged[_dq]
        _n_conv = int(np.sum(_conv))
        _rows.append(
            f"| {_dq} | {_total} | {_n_elements} / {_m * _p} "
            f"| {_frac:.1%} | {_n_conv} / {_m * _p} |"
        )

    _stable = all(
        int(np.sum(disp_result.swap_counts[_dq])) == 0
        for _dq in _thresholds
    )

    _verdict = (
        "No swaps detected at any threshold — solution is rotationally stable."
        if _stable
        else "Swaps detected — solution has rotational ambiguity at one or more thresholds."
    )

    mo.md("\n".join(_rows) + f"\n\n**Verdict**: {_verdict}")
    return


# --- Per-factor swap heatmap ---


@app.cell
def _(mo):
    mo.md("### Per-Factor Swap Heatmap (dQ=4)")
    return


@app.cell
def _(disp_result, np, plt, var_names):
    _dq_heat = 4.0 if 4.0 in disp_result.swap_counts else disp_result.dQ_thresholds[0]
    _sc = disp_result.swap_counts[_dq_heat]
    _m, _p = _sc.shape

    _fig, _ax = plt.subplots(figsize=(max(6, _p * 2), max(4, _m * 0.3)))
    _im = _ax.imshow(_sc, aspect="auto", cmap="Reds", vmin=0, vmax=2)
    _ax.set_xticks(range(_p))
    _ax.set_xticklabels([f"Factor {k+1}" for k in range(_p)])
    _ax.set_yticks(range(_m))
    _ax.set_yticklabels(var_names, fontsize=7)
    _ax.set_title(f"Swap Counts per Element (dQ={_dq_heat})")
    _fig.colorbar(_im, ax=_ax, label="Swaps (0=stable, 1=lo or hi, 2=both)")
    _fig.tight_layout()
    _fig
    return


# --- Approximate standard deviations ---


@app.cell
def _(mo):
    mo.md(
        """
        ### Approximate Standard Deviations (from dQ=4)

        The dQ=4 interval approximates a 95% CI (chi-squared with 1 d.f.,
        ~2-sigma).  Dividing the half-width by 2 gives an approximate
        1-sigma equivalent for each F element.
        """
    )
    return


@app.cell
def _(disp_result, np, plt, var_names):
    _F_std = disp_result.F_std_approx
    if _F_std is not None:
        _F = disp_result.F_base
        _m, _p = _F.shape
        _x = np.arange(_m)

        _fig, _axes = plt.subplots(1, _p, figsize=(4 * _p, 3.5), sharey=False)
        if _p == 1:
            _axes = [_axes]
        for _k in range(_p):
            _ax = _axes[_k]
            _rel_std = np.where(
                _F[:, _k] > 1e-10,
                _F_std[:, _k] / _F[:, _k],
                0.0,
            )
            _ax.bar(_x, _rel_std * 100, color="coral", alpha=0.7)
            _ax.set_title(f"Factor {_k+1}")
            _ax.set_xticks(_x)
            _ax.set_xticklabels(var_names, rotation=90, fontsize=7)
            _ax.set_ylabel("Relative std (%)")
        _fig.suptitle(
            "DISP Approximate Relative Uncertainty (% of base)", fontsize=13
        )
        _fig.tight_layout()
        _fig
    return


# --- Comparison with true factors (ACLS) ---


@app.cell
def _(mo):
    mo.md(
        """
        ### DISP Coverage of True F (ACLS)

        For each true factor element, check whether the DISP interval
        [F_lo, F_hi] at dQ=4 contains the true value.  This is a
        calibration check: the dQ=4 interval should cover ~95% of
        well-estimated elements.
        """
    )
    return


@app.cell
def _(F_true, disp_result, linear_sum_assignment, match_factors, mo, np):
    _dq_cov = 4.0 if 4.0 in disp_result.F_lo else disp_result.dQ_thresholds[0]
    _F_lo = disp_result.F_lo[_dq_cov]
    _F_hi = disp_result.F_hi[_dq_cov]
    _F_base = disp_result.F_base
    _m, _p = _F_base.shape

    _row_ind, _col_ind, _ = match_factors(
        _F_base, F_true, linear_sum_assignment
    )

    _header = "| Factor | Mean |r| vs True | Coverage (%) | Median rel. width |"
    _sep = "|---:|---:|---:|---:|"
    _rows = [_header, _sep]
    _total_covered = 0
    _total_elements = 0
    for _idx in range(_p):
        _k_est = _row_ind[_idx]
        _k_true = _col_ind[_idx]
        _corr = abs(np.corrcoef(_F_base[:, _k_est], F_true[:, _k_true])[0, 1])
        _covered = np.sum(
            (_F_lo[:, _k_est] <= F_true[:, _k_true] + 1e-10)
            & (_F_hi[:, _k_est] >= F_true[:, _k_true] - 1e-10)
        )
        _width = _F_hi[:, _k_est] - _F_lo[:, _k_est]
        _rel_w = np.where(
            _F_base[:, _k_est] > 1e-10,
            _width / _F_base[:, _k_est],
            0.0,
        )
        _total_covered += _covered
        _total_elements += _m
        _rows.append(
            f"| {_idx+1} | {_corr:.3f} | {_covered}/{_m} ({100*_covered/_m:.0f}%) "
            f"| {np.median(_rel_w):.2f} |"
        )

    _overall_cov = _total_covered / _total_elements if _total_elements > 0 else 0.0
    _rows.append(
        f"| **All** | | {_total_covered}/{_total_elements} "
        f"({100*_overall_cov:.0f}%) | |"
    )

    mo.md(
        f"**ACLS dQ={_dq_cov} coverage check**\n\n"
        + "\n".join(_rows)
    )
    return



if __name__ == "__main__":
    app.run()
