import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Volume Prior Demo — Full Bayesian NMF vs ACLS Warm-Start

        The standard Bayesian solver initializes from ACLS (a point estimate)
        and quantifies uncertainty *around* that solution.  The **volume prior**
        (`volume_alpha > 0`) adds a log-determinant penalty on F that penalizes
        collinear factor profiles, enabling the Gibbs sampler to maintain
        distinct factors from random initialization — a truly standalone
        Bayesian solver that explores the full posterior.

        This notebook compares:

        1. **Default** — ACLS warm-start + Gibbs UQ (the standard path)
        2. **Volume prior sweep** — `warm_start=False`, `volume_alpha` from 1 to 5

        We measure factor recovery (correlation with truth), posterior
        uncertainty (CV), and factor separation (pairwise cosine similarity).
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

    from pmf_acls import pmf_bayes

    return linear_sum_assignment, mo, np, plt, pmf_bayes


@app.cell
def _(np):
    def make_synthetic(rng, m=25, n=80, noise_frac=0.10):
        """
        5-source problem with realistic major/minor structure.

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


# --- Section 1: Data Generation ---


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
    data_controls = mo.hstack(
        [noise_slider, n_vars_slider, n_obs_slider, seed_number],
        justify="start", gap=1.5,
    )
    data_controls
    return noise_slider, n_vars_slider, n_obs_slider, seed_number


@app.cell
def _(make_synthetic, mo, noise_slider, n_vars_slider, n_obs_slider, np, seed_number):
    _rng = np.random.default_rng(seed_number.value)
    X, sigma, F_true, G_true, source_names, var_names = make_synthetic(
        _rng, m=n_vars_slider.value, n=n_obs_slider.value,
        noise_frac=noise_slider.value,
    )
    _m, _n = X.shape
    _p_true = F_true.shape[1]
    _X_true = F_true @ G_true
    _snr_power = np.sum(_X_true**2) / np.sum(sigma**2)
    _snr_db = 10 * np.log10(_snr_power)

    mo.md(
        f"""
        **Data**: {_m} variables x {_n} observations, {_p_true} true sources
        &nbsp;|&nbsp; **Noise fraction = {noise_slider.value:.2f}
        -> SNR = {_snr_power:.1f} ({_snr_db:.1f} dB)**

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


# --- Section 2: Run solvers ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. Solver Runs

        We run the **default** Bayesian solver (ACLS warm-start, `volume_alpha=0`)
        and then a sweep of `volume_alpha` values from 1 to 5 with
        `warm_start=False`.  All use the same random seed for the Gibbs
        sampler so differences come purely from the initialization and
        volume prior strength.
        """
    )
    return


@app.cell
def _(F_true, X, linear_sum_assignment, match_factors, mo, np, pmf_bayes, sigma):
    _p = F_true.shape[1]

    # --- Default: ACLS warm-start ---
    result_default = pmf_bayes(
        X, sigma, _p,
        n_samples=1000, n_burnin=500, random_seed=42,
        warm_start=True,
    )

    # --- Volume prior sweep: alpha = 1, 2, 3, 4, 5 ---
    alpha_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    results_vol = {}
    for _alpha in alpha_values:
        results_vol[_alpha] = pmf_bayes(
            X, sigma, _p,
            n_samples=1000, n_burnin=500, random_seed=42,
            warm_start=False, volume_alpha=_alpha,
        )

    # --- Compute metrics for each run ---
    def compute_metrics(result, label):
        F_pm = result.F_posterior_mean
        G_pm = result.G_posterior_mean
        row, col, corr = match_factors(F_pm, F_true, linear_sum_assignment)

        # Mean absolute correlation with truth
        mean_r = np.mean([abs(corr[r, c]) for r, c in zip(row, col)])

        # Mean CV of F profiles (posterior mean scale)
        cvs = []
        for r, c in zip(row, col):
            fm = F_pm[:, r]
            fs = result.F_std[:, r]
            cvs.append(np.mean(fs / np.maximum(fm, 1e-10)))
        mean_cv = np.mean(cvs)

        # Pairwise cosine similarity (lower = better separation)
        norms = np.linalg.norm(F_pm, axis=0)
        sim = (F_pm.T @ F_pm) / np.maximum(np.outer(norms, norms), 1e-30)
        np.fill_diagonal(sim, 0)
        max_cosine = sim.max()

        vol_rate = result.mh_volume_acceptance_rate

        return {
            "label": label,
            "mean_r": mean_r,
            "mean_cv": mean_cv,
            "max_cosine": max_cosine,
            "Q": result.Q,
            "converged": result.converged,
            "vol_accept": vol_rate,
        }

    all_metrics = [compute_metrics(result_default, "Default\n(warm-start)")]
    for _alpha in alpha_values:
        all_metrics.append(
            compute_metrics(results_vol[_alpha], f"alpha={_alpha:.0f}")
        )

    # Summary table
    _rows = []
    for m_dict in all_metrics:
        _va = f"{m_dict['vol_accept']:.0%}" if m_dict['vol_accept'] is not None else "n/a"
        _rows.append(
            f"| {m_dict['label'].replace(chr(10), ' ')} | "
            f"{m_dict['mean_r']:.3f} | "
            f"{m_dict['mean_cv']:.0%} | "
            f"{m_dict['max_cosine']:.3f} | "
            f"{m_dict['Q']:.2e} | "
            f"{'Y' if m_dict['converged'] else 'N'} | "
            f"{_va} |"
        )
    _table = "\n".join(_rows)

    mo.md(
        f"""
        ### Summary

        | Solver | Mean |r| | Mean CV | Max cos sim | Q | Conv | Vol MH |
        |--------|:-------:|:-------:|:-----------:|:---------:|:----:|:------:|
        {_table}

        **Mean |r|** = average absolute correlation of matched F columns with truth (higher = better recovery).
        **Mean CV** = average coefficient of variation across F profiles (lower = more certain).
        **Max cos sim** = maximum pairwise cosine similarity between F columns (lower = better separation).
        """
    )
    return result_default, results_vol, alpha_values, all_metrics


# --- Section 3: Factor profiles comparison ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. Factor Profiles — Default vs Volume Prior

        Each row shows one matched factor.  Left column is the default
        (ACLS warm-start), right column is the best volume-prior run.
        Bars show the posterior mean; whiskers are +/-2 sigma credible intervals.
        Black horizontal lines are truth.
        """
    )
    return


@app.cell
def _(
    F_true,
    all_metrics,
    alpha_values,
    linear_sum_assignment,
    match_factors,
    np,
    plt,
    result_default,
    results_vol,
    source_names,
    var_names,
):
    # Pick the alpha with best mean_r (skip index 0 which is default)
    _vol_metrics = all_metrics[1:]
    _best_idx = np.argmax([m["mean_r"] for m in _vol_metrics])
    _best_alpha = alpha_values[_best_idx]
    _best_vol = results_vol[_best_alpha]

    _pairs = [
        ("Default (warm-start)", result_default),
        (f"Volume prior (alpha={_best_alpha:.0f})", _best_vol),
    ]

    _m = F_true.shape[0]
    _p = F_true.shape[1]
    _x = np.arange(_m)
    _p_show = min(_p, len(source_names))

    _fig, _axes = plt.subplots(
        _p_show, 2,
        figsize=(max(5, 0.7 * _m) * 2, 3.5 * _p_show),
        constrained_layout=True,
    )
    if _p_show == 1:
        _axes = _axes.reshape(1, -1)

    for _col_idx, (_title, _res) in enumerate(_pairs):
        _F_pm = _res.F_posterior_mean
        _row, _col, _corr = match_factors(_F_pm, F_true, linear_sum_assignment)
        _order = np.argsort(_col)
        _row = _row[_order]
        _col = _col[_order]

        for _idx in range(_p_show):
            _k_est = _row[_idx]
            _k_true = _col[_idx]
            _ax = _axes[_idx, _col_idx]

            _f_mean = _F_pm[:, _k_est]
            _f_std = _res.F_std[:, _k_est]
            _r = abs(_corr[_k_est, _k_true])

            _ax.bar(
                _x, _f_mean, 0.6,
                yerr=2 * _f_std, capsize=2,
                color="none",
                edgecolor="C0" if _col_idx == 0 else "C2",
                linewidth=1.0,
                error_kw=dict(
                    ecolor="C0" if _col_idx == 0 else "C2", lw=0.8
                ),
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
            if _idx == 0:
                _ax.set_title(
                    f"{_title}\n{source_names[_k_true]}  r={_r:.3f}",
                    fontsize=9,
                )

    _fig.suptitle(
        "Factor Profiles — Posterior Mean +/- 95% CI vs Truth",
        fontsize=13, fontweight="bold",
    )
    _fig
    return


# --- Section 4: Metrics sweep plot ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 4. Metrics vs Volume Alpha

        How does factor recovery, uncertainty, and separation change as
        `volume_alpha` increases?  The dashed line shows the default
        (ACLS warm-start) baseline.
        """
    )
    return


@app.cell
def _(all_metrics, alpha_values, np, plt):
    _default = all_metrics[0]
    _alphas = np.array(alpha_values)
    _vol = all_metrics[1:]

    _mean_rs = [m["mean_r"] for m in _vol]
    _mean_cvs = [m["mean_cv"] for m in _vol]
    _max_cos = [m["max_cosine"] for m in _vol]

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(
        1, 3, figsize=(14, 4), constrained_layout=True,
    )
    _fig.suptitle(
        "Effect of Volume Alpha on Recovery, Uncertainty, and Separation",
        fontsize=13, fontweight="bold",
    )

    # Mean |r|
    _ax1.plot(_alphas, _mean_rs, "o-C2", lw=2, label="Volume prior")
    _ax1.axhline(
        _default["mean_r"], ls="--", color="C0", lw=1.5,
        label="Default (warm-start)",
    )
    _ax1.set_xlabel("volume_alpha")
    _ax1.set_ylabel("Mean |r| with truth")
    _ax1.set_title("Factor Recovery")
    _ax1.legend(fontsize=8)
    _ax1.set_ylim(0, 1.05)

    # Mean CV
    _ax2.plot(_alphas, _mean_cvs, "o-C2", lw=2, label="Volume prior")
    _ax2.axhline(
        _default["mean_cv"], ls="--", color="C0", lw=1.5,
        label="Default (warm-start)",
    )
    _ax2.set_xlabel("volume_alpha")
    _ax2.set_ylabel("Mean CV (std/mean)")
    _ax2.set_title("Posterior Uncertainty")
    _ax2.legend(fontsize=8)

    # Max cosine sim
    _ax3.plot(_alphas, _max_cos, "o-C2", lw=2, label="Volume prior")
    _ax3.axhline(
        _default["max_cosine"], ls="--", color="C0", lw=1.5,
        label="Default (warm-start)",
    )
    _ax3.set_xlabel("volume_alpha")
    _ax3.set_ylabel("Max pairwise cosine sim")
    _ax3.set_title("Factor Separation")
    _ax3.legend(fontsize=8)

    _fig
    return


# --- Section 5: Q trace comparison ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 5. Q Trace — Convergence Comparison

        The Q objective trace for each solver.  The default (ACLS warm-start)
        typically starts at a lower Q and stabilizes quickly.  The volume-prior
        runs start from random init and must find good solutions from scratch.
        """
    )
    return


@app.cell
def _(alpha_values, plt, result_default, results_vol):
    _fig, _ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    _ax.plot(
        result_default.Q_samples, color="C0", lw=1.5, alpha=0.9,
        label="Default (warm-start)",
    )
    _cmap = plt.cm.Greens(
        [0.4 + 0.12 * i for i in range(len(alpha_values))]
    )
    for _i, _alpha in enumerate(alpha_values):
        _ax.plot(
            results_vol[_alpha].Q_samples, color=_cmap[_i], lw=1.0,
            alpha=0.8, label=f"alpha={_alpha:.0f}",
        )
    _ax.set_xlabel("Sample index")
    _ax.set_ylabel("Q")
    _ax.set_title("Q Trace — Convergence Comparison", fontsize=13, fontweight="bold")
    _ax.legend(fontsize=8, ncol=2)
    _ax.set_yscale("log")
    _fig
    return


# --- Section 6: Contribution time series ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 6. Contributions — Default vs Best Volume Prior

        Time series of G (factor contributions) with posterior 95% credible
        intervals.  The volume-prior solver explores the posterior from
        scratch, so its uncertainty may differ from the ACLS-anchored default.
        """
    )
    return


@app.cell
def _(
    F_true,
    all_metrics,
    alpha_values,
    linear_sum_assignment,
    match_factors,
    np,
    plt,
    result_default,
    results_vol,
    source_names,
):
    _vol_metrics = all_metrics[1:]
    _best_idx = np.argmax([m["mean_r"] for m in _vol_metrics])
    _best_alpha = alpha_values[_best_idx]
    _best_vol = results_vol[_best_alpha]

    _pairs = [
        ("Default (warm-start)", result_default),
        (f"Volume prior (alpha={_best_alpha:.0f})", _best_vol),
    ]
    _p = F_true.shape[1]
    _p_show = min(_p, len(source_names))

    _fig, _axes = plt.subplots(
        2, _p_show, figsize=(3.5 * _p_show, 6), constrained_layout=True,
    )
    if _p_show == 1:
        _axes = _axes.reshape(2, 1)

    for _row_idx, (_title, _res) in enumerate(_pairs):
        _G_pm = _res.G_posterior_mean
        _row, _col, _ = match_factors(
            _res.F_posterior_mean, F_true, linear_sum_assignment,
        )
        _order = np.argsort(_col)
        _row = _row[_order]
        _col = _col[_order]

        _n_obs = _G_pm.shape[1]
        _x_obs = np.arange(_n_obs)
        _color = "C0" if _row_idx == 0 else "C2"

        for _idx in range(_p_show):
            _k_est = _row[_idx]
            _k_true = _col[_idx]
            _ax = _axes[_row_idx, _idx]

            _g_mean = _G_pm[_k_est, :]
            _g_std = _res.G_std[_k_est, :]

            _ax.fill_between(
                _x_obs, _g_mean - 2 * _g_std, _g_mean + 2 * _g_std,
                alpha=0.2, color=_color,
            )
            _ax.plot(_x_obs, _g_mean, f"{_color}-", lw=0.8)
            _cv = np.mean(_g_std / np.maximum(_g_mean, 1e-10))
            _ax.text(
                0.98, 0.95, f"CV={_cv:.0%}",
                transform=_ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )
            if _row_idx == 0:
                _ax.set_title(source_names[_k_true], fontsize=9)
            if _idx == 0:
                _ax.set_ylabel(_title, fontsize=8)

    _fig.suptitle(
        "Contributions G — Posterior Mean +/- 95% CI",
        fontsize=13, fontweight="bold",
    )
    _fig
    return


# --- Section 7: Takeaways ---


@app.cell
def _(mo):
    mo.md(
        """
        ## Takeaways

        1. **ACLS warm-start is a strong default** — it gives sharp, well-separated
           profiles and fast convergence.  The Bayesian layer adds honest
           uncertainty quantification around that solution.

        2. **Volume prior enables standalone Bayesian NMF** — with `warm_start=False`
           and `volume_alpha > 0`, the solver discovers factors purely from the
           data + prior, without relying on a point estimate for initialization.

        3. **Moderate alpha (2-3) is often sufficient** — too low and factors may
           still collapse; too high and the prior dominates, potentially biasing
           profiles away from the data.

        4. **The posterior mean + std are on the same scale** — always use
           `result.F_posterior_mean` / `result.G_posterior_mean` with
           `result.F_std` / `result.G_std` for plotting credible intervals and
           computing CVs.  `result.F` / `result.G` are the ACLS point estimate
           (when `warm_start=True`) and live on a different normalization scale.

        5. **`mh_volume_acceptance_rate`** tracks how often the volume-prior MH
           correction accepts.  Very low rates (< 10%) suggest `volume_alpha` is
           too aggressive for the data geometry.
        """
    )
    return


if __name__ == "__main__":
    app.run()
