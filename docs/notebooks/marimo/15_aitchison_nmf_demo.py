import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Aitchison NMF: Compositional Source Apportionment

        Standard PMF minimizes Euclidean residuals weighted by uncertainty:
        $Q = \sum_{ij} \left(\frac{X_{ij} - R_{ij}}{\sigma_{ij}}\right)^2$

        **Aitchison NMF** instead minimizes weighted centered log-residuals:
        $Q_A = \sum_{ij} w_{ij} \left(C_{ij}^w\right)^2$

        where $w_{ij} = X_{ij}^2 / \sigma_{ij}^2$ (squared SNR),
        $Z_{ij} = \ln(X_{ij}/R_{ij})$, and
        $C_{ij}^w = Z_{ij} - \bar{Z}_j^w$ (weighted-centered log-residuals).

        This operates in the **Aitchison geometry** of the simplex, penalizing
        *ratio* errors rather than *absolute* errors. It is the natural metric
        when compositional ratios carry the scientific meaning.

        ### When does it matter?

        Toggle **"Close data"** to see the difference. When data is closed
        (rows sum to a constant), Euclidean distances are distorted by the
        closure constraint. Aitchison NMF recovers the correct factor
        *proportions* regardless of closure, while standard PMF's profile
        recovery degrades.
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

    from pmf_acls import pmf
    from pmf_acls.core import simplex_pmf
    from pmf_acls.coda import aitchison_nmf, closure

    return aitchison_nmf, closure, mo, np, plt, pmf, simplex_pmf


# --- Configuration ---


@app.cell
def _(mo):
    mo.md("## 1. Configuration")
    return


@app.cell
def _(mo):
    n_vars_slider = mo.ui.slider(6, 25, step=1, value=12, label="Variables (m)")
    n_obs_slider = mo.ui.slider(40, 200, step=20, value=100, label="Observations (n)")
    n_sources_slider = mo.ui.slider(2, 8, step=1, value=4, label="True sources")
    noise_slider = mo.ui.slider(0.02, 0.30, step=0.02, value=0.08, label="Noise fraction")
    seed_number = mo.ui.number(value=42, label="Random seed")
    close_toggle = mo.ui.switch(value=True, label="Close data (sum-to-1)")
    n_starts_slider = mo.ui.slider(1, 20, step=1, value=5, label="Multi-start seeds")
    anchor_number = mo.ui.number(value=0.0, start=0.0, step=0.001,
                                  label="Anchor — in data units, e.g. MDL (0 = strict Aitchison)")

    mo.vstack([
        mo.hstack(
            [n_vars_slider, n_obs_slider, n_sources_slider],
            justify="start", gap=1.5,
        ),
        mo.hstack(
            [noise_slider, seed_number, n_starts_slider],
            justify="start", gap=1.5,
        ),
        mo.hstack(
            [close_toggle, anchor_number],
            justify="start", gap=1.5,
        ),
    ])
    return (
        anchor_number,
        close_toggle,
        n_obs_slider,
        n_sources_slider,
        n_starts_slider,
        n_vars_slider,
        noise_slider,
        seed_number,
    )


@app.cell
def _(mo, n_sources_slider):
    # Default scales: 1 dominant, 1–2 intermediate, rest weak/trace
    _n = n_sources_slider.value
    _defaults = [50.0, 20.0, 10.0, 3.0, 2.0, 1.5, 1.0, 0.5][:_n]
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


# --- Synthetic data ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. Synthetic Data

        True source profiles live on the simplex (each column of F sums to 1
        over variables). Contributions G are gamma-distributed. When **"Close
        data"** is on, each observation (column of X) is normalized to a
        constant total — simulating mass fraction or percentage data where
        only relative abundances are measured.
        """
    )
    return


@app.cell
def _(
    close_toggle,
    closure,
    mo,
    n_obs_slider,
    n_vars_slider,
    noise_slider,
    np,
    seed_number,
    source_scale_sliders,
):
    _rng = np.random.default_rng(int(seed_number.value))
    _m = n_vars_slider.value
    _n = n_obs_slider.value
    _scales = np.array([s.value for s in source_scale_sliders])
    _p = len(_scales)

    # True source profiles on the simplex (columns sum to 1)
    # Each source loads on a random subset of variables (sparser for weaker sources)
    F_true = np.full((_m, _p), 0.01)
    for _k in range(_p):
        _frac = max(0.15, min(0.5, 0.1 + 0.01 * _scales[_k]))
        _n_load = max(2, int(round(_frac * _m)))
        _n_load = min(_n_load, _m)
        _chosen = _rng.choice(_m, size=_n_load, replace=False)
        F_true[_chosen, _k] += _rng.exponential(
            max(0.1, 0.01 * _scales[_k]), size=_n_load,
        )
    # Normalize columns to simplex
    for _k in range(_p):
        F_true[:, _k] /= F_true[:, _k].sum()

    # Source contributions with configurable strengths
    G_true = np.zeros((_p, _n))
    for _k in range(_p):
        G_true[_k, :] = _rng.exponential(_scales[_k], size=_n)

    # Clean signal + noise
    X_clean = F_true @ G_true
    _sigma_frac = noise_slider.value
    _sigma_true = _sigma_frac * np.maximum(X_clean, 0.01) + 0.001
    X_raw = np.maximum(X_clean + _rng.normal(0, _sigma_true), 1e-8)

    # Optionally close
    if close_toggle.value:
        _col_totals = X_raw.sum(axis=0, keepdims=True)
        _mean_total = _col_totals.mean()
        X = closure(X_raw.T).T * _mean_total
    else:
        X = X_raw

    sigma = _sigma_frac * np.maximum(X, 0.01) + 0.001
    var_names = [f"V{_i+1:02d}" for _i in range(_m)]
    source_names = [f"Source {_k+1}" for _k in range(_p)]

    _rows = []
    for _k in range(_p):
        _strength = "strong" if _scales[_k] >= 30 else "medium" if _scales[_k] >= 8 else "weak"
        _rows.append(
            f"| {source_names[_k]} | {_strength} | {_scales[_k]:.1f} | {G_true[_k].mean():.1f} |"
        )

    _X_true = F_true @ G_true
    _snr_power = np.sum(_X_true**2) / np.sum(sigma**2)
    _snr_db = 10 * np.log10(_snr_power)

    mo.md(
        f"**Data**: {_m} variables × {_n} observations, "
        f"**{_p} true sources**, noise = {_sigma_frac:.0%}, "
        f"closed = {close_toggle.value}, "
        f"SNR = {_snr_power:.1f} ({_snr_db:.1f} dB)\n\n"
        f"| Source | Type | Scale | Mean G |\n"
        f"|--------|------|:-----:|:------:|\n"
        + "\n".join(_rows)
    )
    return F_true, G_true, X, sigma, source_names, var_names


# --- Run both solvers ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. Standard PMF vs Aitchison NMF

        Both solvers use multi-start random initialization (same seeds).
        We compare:
        - **Factor profile recovery** (Hungarian-matched correlation to truth)
        - **Convergence** behavior
        - **Sensitivity to closure** (toggle data closure on/off)
        """
    )
    return


@app.cell
def _(
    F_true, X, aitchison_nmf, anchor_number, close_toggle, mo,
    n_starts_slider, np, pmf, sigma, simplex_pmf,
):
    _p = F_true.shape[1]
    _m, _n = X.shape
    _n_starts = n_starts_slider.value
    _anchor = anchor_number.value
    _closed = close_toggle.value

    # Standard PMF — best of n_starts seeds
    # When data is closed, use simplex_pmf so G columns sum to 1
    _best_Q_std = np.inf
    r_std = None
    for _seed in range(_n_starts):
        if _closed:
            _r = simplex_pmf(X, sigma, _p, random_seed=_seed)
        else:
            _r = pmf(X, sigma, _p, max_iter=2000, conv_tol=0.001, random_seed=_seed)
        if _r.Q < _best_Q_std:
            _best_Q_std = _r.Q
            r_std = _r

    # Aitchison NMF — best of n_starts seeds
    # Aitchison geometry is naturally closure-invariant, no projection needed
    _best_Q_ait = np.inf
    r_ait = None
    for _seed in range(_n_starts):
        _r = aitchison_nmf(X, sigma, _p, max_iter=2000, tol=1e-6,
                           anchor=_anchor, random_seed=_seed)
        if _r.Q < _best_Q_ait:
            _best_Q_ait = _r.Q
            r_ait = _r

    _Q_exp = _m * _n
    _anchor_str = f", anchor = {_anchor:.2f}" if _anchor > 0 else ""
    _std_label = "Simplex PMF" if _closed else "Standard PMF"
    mo.md(
        f"**{_std_label}**: Q = {r_std.Q:.2f}, "
        f"Q/Q_exp = {r_std.Q / _Q_exp:.4f}, "
        f"converged = {r_std.converged}, "
        f"iters = {r_std.n_iter}\n\n"
        f"**Aitchison NMF**: Q_A = {r_ait.Q:.4f}{_anchor_str}, "
        f"converged = {r_ait.converged}, "
        f"iters = {r_ait.n_iter}\n\n"
        f"*(best of {_n_starts} random seeds each)*\n\n"
        + ("Note: Simplex PMF enforces G column sums = 1 (closed solution). "
           "Aitchison NMF handles closure natively via log-ratio geometry.\n\n"
           if _closed else "")
        + "Note: Q and Q_A are different cost functions and cannot be "
        "compared numerically. Compare factor recovery below."
    )
    return r_ait, r_std


# --- Factor profiles ---


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Log-Ratio Profile Recovery

        The key advantage of Aitchison NMF is **ratio recovery** — getting
        the proportions between species right within each profile. Bar charts
        of absolute profile values don't show this clearly; CLR coordinates do.

        **CLR scatter plot:** For each factor, we plot
        $\text{CLR}(\hat{F}_k)$ vs $\text{CLR}(F_k^{\text{true}})$
        for every species. Points on the 1:1 line mean perfect ratio recovery.
        Spread away from the diagonal means distorted inter-species ratios.

        **Pairwise log-ratio scatter:** For each factor, we plot
        $\ln(\hat{F}_{ik}/\hat{F}_{jk})$ vs $\ln(F^{\text{true}}_{ik}/F^{\text{true}}_{jk})$
        for all species pairs $(i,j)$. This is the direct object that the
        Aitchison cost optimizes. Tighter cloud on the 1:1 line = better.
        """
    )
    return


@app.cell
def _(F_true, closure, np, plt, r_ait, r_std):
    from scipy.optimize import linear_sum_assignment
    from pmf_acls.coda import clr_transform

    def _hungarian_match(_F_est, _F_ref):
        _p = _F_est.shape[1]
        _corr = np.zeros((_p, _p))
        for _i in range(_p):
            for _j in range(_p):
                _c = np.corrcoef(_F_est[:, _i], _F_ref[:, _j])[0, 1]
                _corr[_i, _j] = _c if np.isfinite(_c) else 0.0
        _row_ind, _col_ind = linear_sum_assignment(-np.abs(_corr))
        return _row_ind, _col_ind

    _p = F_true.shape[1]
    _m = F_true.shape[0]

    # Align both to truth independently
    _ri_std, _ci_std = _hungarian_match(r_std.F, F_true)
    _ri_ait, _ci_ait = _hungarian_match(r_ait.F, F_true)

    _order_std = np.empty(_p, dtype=int)
    _order_ait = np.empty(_p, dtype=int)
    for _k in range(_p):
        _order_std[_ci_std[_k]] = _ri_std[_k]
        _order_ait[_ci_ait[_k]] = _ri_ait[_k]

    # L1-normalize (close) profiles for compositional comparison
    _F_true_c = closure(F_true.T).T
    _F_std_c = closure(r_std.F[:, _order_std].T).T
    _F_ait_c = closure(r_ait.F[:, _order_ait].T).T

    # CLR-transform each column (profile)
    _clr_true = np.zeros_like(_F_true_c)
    _clr_std = np.zeros_like(_F_std_c)
    _clr_ait = np.zeros_like(_F_ait_c)
    for _k in range(_p):
        _clr_true[:, _k] = clr_transform(_F_true_c[:, _k])
        _clr_std[:, _k] = clr_transform(_F_std_c[:, _k])
        _clr_ait[:, _k] = clr_transform(_F_ait_c[:, _k])

    # --- Top row: CLR scatter (one point per species per factor) ---
    # --- Bottom row: Pairwise log-ratio scatter ---
    _fig, _axes = plt.subplots(2, _p, figsize=(3.5 * _p, 7))
    if _p == 1:
        _axes = _axes.reshape(2, 1)

    clr_rmse_std = []
    clr_rmse_ait = []
    lr_rmse_std = []
    lr_rmse_ait = []

    for _k in range(_p):
        # --- CLR scatter ---
        _ax = _axes[0, _k]
        _ax.scatter(_clr_true[:, _k], _clr_std[:, _k],
                    color="coral", s=40, alpha=0.7, edgecolors="white",
                    linewidths=0.5, label="Standard", zorder=3)
        _ax.scatter(_clr_true[:, _k], _clr_ait[:, _k],
                    color="steelblue", s=40, alpha=0.7, edgecolors="white",
                    linewidths=0.5, label="Aitchison", zorder=4)

        _lim = max(abs(_clr_true[:, _k]).max(), 1.0) * 1.3
        _ax.plot([-_lim, _lim], [-_lim, _lim], "k--", alpha=0.3, linewidth=1)
        _ax.set_xlim(-_lim, _lim)
        _ax.set_ylim(-_lim, _lim)
        _ax.set_aspect("equal")
        _ax.set_xlabel("CLR(True)", fontsize=8)
        if _k == 0:
            _ax.set_ylabel("CLR(Estimated)", fontsize=8)
            _ax.legend(fontsize=7)

        _rmse_s = np.sqrt(np.mean((_clr_std[:, _k] - _clr_true[:, _k]) ** 2))
        _rmse_a = np.sqrt(np.mean((_clr_ait[:, _k] - _clr_true[:, _k]) ** 2))
        clr_rmse_std.append(_rmse_s)
        clr_rmse_ait.append(_rmse_a)
        _ax.set_title(f"Factor {_k+1}\nCLR RMSE: S={_rmse_s:.2f}  A={_rmse_a:.2f}",
                      fontsize=8)

        # --- Pairwise log-ratio scatter ---
        _ax2 = _axes[1, _k]
        _lr_true = []
        _lr_std_k = []
        _lr_ait_k = []
        for _i in range(_m):
            for _j in range(_i + 1, _m):
                _lr_true.append(np.log(_F_true_c[_i, _k] / _F_true_c[_j, _k]))
                _lr_std_k.append(np.log(_F_std_c[_i, _k] / _F_std_c[_j, _k]))
                _lr_ait_k.append(np.log(_F_ait_c[_i, _k] / _F_ait_c[_j, _k]))
        _lr_true = np.array(_lr_true)
        _lr_std_k = np.array(_lr_std_k)
        _lr_ait_k = np.array(_lr_ait_k)

        _ax2.scatter(_lr_true, _lr_std_k,
                     color="coral", s=15, alpha=0.5, edgecolors="none",
                     label="Standard", zorder=3)
        _ax2.scatter(_lr_true, _lr_ait_k,
                     color="steelblue", s=15, alpha=0.5, edgecolors="none",
                     label="Aitchison", zorder=4)

        _lim2 = max(abs(_lr_true).max(), 1.0) * 1.3
        _ax2.plot([-_lim2, _lim2], [-_lim2, _lim2], "k--", alpha=0.3, linewidth=1)
        _ax2.set_xlim(-_lim2, _lim2)
        _ax2.set_ylim(-_lim2, _lim2)
        _ax2.set_aspect("equal")
        _ax2.set_xlabel("ln(True ratio)", fontsize=8)
        if _k == 0:
            _ax2.set_ylabel("ln(Estimated ratio)", fontsize=8)
            _ax2.legend(fontsize=7)

        _rmse_lr_s = np.sqrt(np.mean((_lr_std_k - _lr_true) ** 2))
        _rmse_lr_a = np.sqrt(np.mean((_lr_ait_k - _lr_true) ** 2))
        lr_rmse_std.append(_rmse_lr_s)
        lr_rmse_ait.append(_rmse_lr_a)
        _ax2.set_title(f"LR RMSE: S={_rmse_lr_s:.2f}  A={_rmse_lr_a:.2f}",
                       fontsize=8)

    _fig.text(0.01, 0.75, "CLR\nscatter", ha="left", va="center",
              fontsize=9, fontstyle="italic", color="gray")
    _fig.text(0.01, 0.28, "Pairwise\nlog-ratio", ha="left", va="center",
              fontsize=9, fontstyle="italic", color="gray")
    _fig.suptitle("Log-Ratio Recovery: Estimated vs True", fontsize=12)
    _fig.tight_layout(rect=[0.04, 0, 1, 0.95])
    _fig
    return clr_rmse_ait, clr_rmse_std, lr_rmse_ait, lr_rmse_std


# --- Quantitative summary ---


@app.cell
def _(F_true, clr_rmse_ait, clr_rmse_std, lr_rmse_ait, lr_rmse_std, mo, np):
    _p = F_true.shape[1]
    _rows = [
        "| Factor | CLR RMSE (Std) | CLR RMSE (Ait) | LR RMSE (Std) | LR RMSE (Ait) | Ait better? |",
        "|---:|---:|---:|---:|---:|:---|",
    ]
    for _k in range(_p):
        _better_clr = clr_rmse_ait[_k] < clr_rmse_std[_k] - 0.01
        _better_lr = lr_rmse_ait[_k] < lr_rmse_std[_k] - 0.01
        _tag = "Yes" if (_better_clr or _better_lr) else ("No" if (
            clr_rmse_std[_k] < clr_rmse_ait[_k] - 0.01) else "Tie")
        _rows.append(
            f"| {_k+1} | {clr_rmse_std[_k]:.3f} | {clr_rmse_ait[_k]:.3f} "
            f"| {lr_rmse_std[_k]:.3f} | {lr_rmse_ait[_k]:.3f} | {_tag} |"
        )
    _rows.append(
        f"| **Mean** | **{np.mean(clr_rmse_std):.3f}** | **{np.mean(clr_rmse_ait):.3f}** "
        f"| **{np.mean(lr_rmse_std):.3f}** | **{np.mean(lr_rmse_ait):.3f}** | |"
    )

    mo.md(
        "### Log-Ratio Recovery Metrics\n\n"
        + "\n".join(_rows)
        + "\n\n**CLR RMSE**: RMSE of centered log-ratio coordinates (one per species). "
        "Lower = better compositional structure recovery.\n\n"
        "**LR RMSE**: RMSE of all pairwise log-ratios ln(F_i/F_j). "
        "This is what the Aitchison cost directly optimizes — the natural "
        "error metric in simplex geometry."
    )
    return


# --- Convergence ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 5. Convergence

        Left: Standard PMF Q history. Right: Aitchison Q_A history.
        These are *different* cost functions — the y-axis scales are
        not comparable. What matters is that both converge smoothly.
        """
    )
    return


@app.cell
def _(plt, r_ait, r_std):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    _ax1.plot(r_std.Q_history, color="coral", linewidth=1)
    _ax1.set_xlabel("Iteration")
    _ax1.set_ylabel("Q (Euclidean)")
    _ax1.set_title("Standard PMF")
    _ax1.set_yscale("log")

    _ax2.plot(r_ait.Q_history, color="steelblue", linewidth=1)
    _ax2.set_xlabel("Iteration")
    _ax2.set_ylabel("Q_A (Aitchison)")
    _ax2.set_title("Aitchison NMF")
    _ax2.set_yscale("log")

    _fig.suptitle("Convergence Traces", fontsize=12)
    _fig.tight_layout()
    _fig
    return


# --- Contribution time series ---


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6. Source Mixing Fractions

        For each observation, the fractional contribution of each source
        is $g_{kj} / \sum_k g_{kj}$ — the share of total signal from source $k$.
        This is the quantity that matters for closed data: not "how much of
        source $k$" but "what fraction is source $k$."

        **Stacked area plots** show the estimated source mix across observations,
        compared to the true mix. On closed data, Aitchison NMF should
        recover these fractions more faithfully because it optimizes ratios.
        """
    )
    return


@app.cell
def _(F_true, G_true, np, plt, r_ait, r_std):
    from scipy.optimize import linear_sum_assignment as _lsa

    def _match_G(_G_est, _G_ref):
        _p = _G_est.shape[0]
        _corr = np.zeros((_p, _p))
        for _i in range(_p):
            for _j in range(_p):
                _c = np.corrcoef(_G_est[_i, :], _G_ref[_j, :])[0, 1]
                _corr[_i, _j] = _c if np.isfinite(_c) else 0.0
        _ri, _ci = _lsa(-np.abs(_corr))
        _order = np.empty(_p, dtype=int)
        for _k in range(_p):
            _order[_ci[_k]] = _ri[_k]
        return _order

    _p = F_true.shape[1]
    _n = G_true.shape[1]

    _order_std = _match_G(r_std.G, G_true)
    _order_ait = _match_G(r_ait.G, G_true)

    # Compute fractional contributions (columns sum to 1)
    _frac_true = G_true / G_true.sum(axis=0, keepdims=True)
    _G_std_matched = r_std.G[_order_std, :]
    _frac_std = _G_std_matched / _G_std_matched.sum(axis=0, keepdims=True)
    _G_ait_matched = r_ait.G[_order_ait, :]
    _frac_ait = _G_ait_matched / _G_ait_matched.sum(axis=0, keepdims=True)

    # Sort observations by dominant source for visual clarity
    _dominant = np.argmax(_frac_true, axis=0)
    _sort_idx = np.lexsort((_frac_true[0, :], _dominant))

    _colors = plt.cm.Set2(np.linspace(0, 1, _p))

    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    _obs = np.arange(_n)

    for _col, (_frac, _title) in enumerate([
        (_frac_true[:, _sort_idx], "True"),
        (_frac_std[:, _sort_idx], "Standard PMF"),
        (_frac_ait[:, _sort_idx], "Aitchison NMF"),
    ]):
        _ax = _axes[_col]
        _bottom = np.zeros(_n)
        for _k in range(_p):
            _ax.fill_between(_obs, _bottom, _bottom + _frac[_k, :],
                             color=_colors[_k], alpha=0.8,
                             label=f"Source {_k+1}")
            _bottom += _frac[_k, :]
        _ax.set_xlim(0, _n - 1)
        _ax.set_ylim(0, 1)
        _ax.set_xlabel("Observation (sorted)", fontsize=8)
        _ax.set_title(_title, fontsize=10)
        if _col == 0:
            _ax.set_ylabel("Fractional contribution", fontsize=8)
            _ax.legend(fontsize=6, loc="upper left")

    _fig.suptitle("Source Mixing Fractions per Observation", fontsize=12)
    _fig.tight_layout()
    _fig

    # Compute fraction RMSE for the summary
    frac_rmse_std = np.sqrt(np.mean((_frac_std - _frac_true) ** 2))
    frac_rmse_ait = np.sqrt(np.mean((_frac_ait - _frac_true) ** 2))

    return frac_rmse_ait, frac_rmse_std


@app.cell
def _(frac_rmse_ait, frac_rmse_std, mo):
    _winner = "Aitchison" if frac_rmse_ait < frac_rmse_std - 0.001 else (
        "Standard" if frac_rmse_std < frac_rmse_ait - 0.001 else "Tie")
    mo.md(
        f"**Mixing fraction RMSE** (lower = better): "
        f"Standard = {frac_rmse_std:.4f}, "
        f"Aitchison = {frac_rmse_ait:.4f} "
        f"(**{_winner}**)"
    )
    return


# --- Log-ratio residual comparison ---


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 7. Residual Diagnostics

        **Left:** Euclidean residuals $(X - R)/\sigma$ — what standard PMF
        minimizes.

        **Right:** Log-ratio residuals $\ln(X/R)$ — what Aitchison NMF
        minimizes (before centering/weighting).

        Well-specified models should show residuals centered at zero with
        no systematic structure. Aitchison NMF's log-ratio residuals
        should be more symmetric (log-ratios treat over- and
        under-estimation symmetrically).
        """
    )
    return


@app.cell
def _(X, np, plt, r_ait, r_std, sigma):
    _R_std = r_std.F @ r_std.G
    _R_ait = r_ait.F @ r_ait.G

    _fig, _axes = plt.subplots(2, 2, figsize=(10, 6))

    # Standard PMF — Euclidean residuals
    _resid_std = (X - _R_std) / sigma
    _axes[0, 0].hist(_resid_std.ravel(), bins=50, color="coral", alpha=0.7,
                      edgecolor="white", linewidth=0.5)
    _axes[0, 0].set_title("Std PMF: (X-R)/σ", fontsize=9)
    _axes[0, 0].axvline(0, color="black", linewidth=0.5)

    # Aitchison NMF — Euclidean residuals (for comparison)
    _resid_ait_euc = (X - _R_ait) / sigma
    _axes[0, 1].hist(_resid_ait_euc.ravel(), bins=50, color="steelblue", alpha=0.7,
                      edgecolor="white", linewidth=0.5)
    _axes[0, 1].set_title("Aitchison NMF: (X-R)/σ", fontsize=9)
    _axes[0, 1].axvline(0, color="black", linewidth=0.5)

    # Standard PMF — log-ratio residuals
    _eps = 1e-30
    _logr_std = np.log(np.maximum(X, _eps) / np.maximum(_R_std, _eps))
    _axes[1, 0].hist(_logr_std.ravel(), bins=50, color="coral", alpha=0.7,
                      edgecolor="white", linewidth=0.5)
    _axes[1, 0].set_title("Std PMF: ln(X/R)", fontsize=9)
    _axes[1, 0].axvline(0, color="black", linewidth=0.5)

    # Aitchison NMF — log-ratio residuals
    _logr_ait = np.log(np.maximum(X, _eps) / np.maximum(_R_ait, _eps))
    _axes[1, 1].hist(_logr_ait.ravel(), bins=50, color="steelblue", alpha=0.7,
                      edgecolor="white", linewidth=0.5)
    _axes[1, 1].set_title("Aitchison NMF: ln(X/R)", fontsize=9)
    _axes[1, 1].axvline(0, color="black", linewidth=0.5)

    for _ax in _axes.ravel():
        _ax.set_ylabel("Count", fontsize=8)

    _fig.suptitle("Residual Distributions", fontsize=12)
    _fig.tight_layout()
    _fig
    return


# --- Weight distribution ---


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 8. Weight Distribution

        Aitchison NMF uses weights $w_{ij} = X_{ij}^2 / \sigma_{ij}^2$
        (squared signal-to-noise ratio), while standard PMF uses
        $1/\sigma_{ij}^2$. The Aitchison weights naturally down-weight
        near-detection-limit species — elements with small $X_{ij}$ get
        near-zero weight regardless of their assigned uncertainty.

        This is arguably better behavior for BDL species: in standard
        PMF, a below-detection value at the detection limit still
        receives substantial weight via $1/\sigma^2$.
        """
    )
    return


@app.cell
def _(X, np, plt, sigma):
    _w_std = 1.0 / sigma ** 2
    _w_ait = (X / sigma) ** 2

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    _ax1.hist(np.log10(_w_std.ravel()), bins=50, color="coral", alpha=0.7,
              edgecolor="white", linewidth=0.5)
    _ax1.set_xlabel("log₁₀(weight)")
    _ax1.set_ylabel("Count")
    _ax1.set_title("Standard PMF: 1/σ²")

    _ax2.hist(np.log10(np.maximum(_w_ait.ravel(), 1e-30)), bins=50,
              color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.5)
    _ax2.set_xlabel("log₁₀(weight)")
    _ax2.set_ylabel("Count")
    _ax2.set_title("Aitchison NMF: X²/σ² (squared SNR)")

    _fig.suptitle("Weight Distributions", fontsize=12)
    _fig.tight_layout()
    _fig
    return


# --- Summary ---


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        | | Standard PMF | Aitchison NMF |
        |---|---|---|
        | **Cost function** | Euclidean: $\sum (X-R)^2/\sigma^2$ | Aitchison: $\sum w \cdot C^2$ (centered log-ratios) |
        | **Geometry** | Euclidean (absolute errors) | Simplex / Aitchison (ratio errors) |
        | **Weights** | $1/\sigma_{ij}^2$ | $X_{ij}^2/\sigma_{ij}^2$ (squared SNR) |
        | **BDL handling** | Weight from $1/\sigma^2$ can be large | Near-zero X → near-zero weight (natural suppression) |
        | **Closure invariance** | No — closure distorts Euclidean distances | Yes — log-ratios + centering remove closure effects |
        | **Scale sensitivity** | Absolute concentrations matter | Only ratios matter (scale-invariant) |
        | **Best for** | Absolute concentration data | Mass fractions, percentages, closed/normalized data |

        ### When to use Aitchison NMF

        - Data is **compositional** (mass fractions, percentages, normalized)
        - You care about **relative source proportions**, not absolute levels
        - Species span **orders of magnitude** (log-space errors are more
          meaningful than absolute errors)
        - BDL species should be **naturally down-weighted** without manual
          uncertainty inflation

        ### When to stick with standard PMF

        - Data is in **absolute concentrations** with meaningful totals
        - Uncertainties are well-calibrated and $Q/Q_{exp} \approx 1$ matters
        - You need compatibility with EPA PMF / ESAT conventions
        - No closure or normalization has been applied

        ### Usage

        ```python
        from pmf_acls import aitchison_nmf

        result = aitchison_nmf(X, sigma, p=3, random_seed=42)
        # result.F — factor profiles (m, p)
        # result.G — contributions (p, n)
        # result.Q — Aitchison cost (not comparable to standard Q)
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
