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
        # Complete PMF Workflow

        This notebook walks through the **end-to-end Positive Matrix Factorization
        workflow** that an environmental scientist would follow when analysing
        a multivariate dataset of chemical concentrations (or any non-negative
        receptor data).

        The seven steps are:

        1. **Generate / load data** -- here we synthesise a realistic dataset with
           missing values and below-detection-limit entries.
        2. **Data preparation** -- impute missing values, handle BDL, and estimate
           measurement uncertainties.
        3. **Factor selection** -- scan a range of factor counts and pick the best
           *p* using the Q_robust diagnostic.
        4. **PMF analysis** -- run the solver with the chosen *p*.
        5. **Uncertainty estimation** -- bootstrap displacement to quantify how
           stable the solution is.
        6. **Factor contributions** -- decompose the reconstructed signal by
           source.
        7. **Diagnostics** -- residual analysis and summary statistics.
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

    from pmf_acls import (
        prepare_data,
        select_factors,
        pmf,
        bootstrap_uncertainty,
        print_diagnostics,
        compute_contributions,
    )

    return (
        compute_contributions,
        bootstrap_uncertainty,
        mo,
        np,
        plt,
        pmf,
        prepare_data,
        print_diagnostics,
        select_factors,
    )


# ── Step 1: Generate Synthetic Data ──────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 1 -- Synthetic Data Generation

        We create a small but realistic problem: **10 chemical species** measured
        across **100 samples**, generated from **3 true source profiles**.

        Real-world data always contains quality issues, so we inject:

        - **Missing values** (NaN) in a few cells -- simulating instrument
          dropouts or flagged observations.
        - **Below-detection-limit (BDL)** entries -- concentrations too low for
          the instrument to quantify reliably, replaced with a negative sentinel
          value.

        These are exactly the data-quality problems that `prepare_data` is
        designed to handle in Step 2.
        """
    )
    return


@app.cell
def _(np):
    rng = np.random.default_rng(42)

    n, m, p_true = 100, 10, 3
    var_names = [f"Spec_{i+1}" for i in range(m)]

    # True source profiles and contributions
    G_true = rng.random((n, p_true)) + 0.5
    F_true = rng.random((p_true, m)) + 0.5

    # Observed data with heteroscedastic noise
    X_raw = G_true @ F_true + 0.15 * rng.standard_normal((n, m))

    # Inject data-quality issues
    X_raw[50:55, 2] = np.nan       # missing values
    X_raw[X_raw < 0.2] = -0.1      # below detection limit

    detection_limits = 0.2 * np.ones(m)

    n_missing = int(np.sum(np.isnan(X_raw)))
    n_bdl = int(np.sum(X_raw[~np.isnan(X_raw)] < 0))

    return (
        F_true,
        G_true,
        X_raw,
        detection_limits,
        m,
        n,
        n_bdl,
        n_missing,
        p_true,
        rng,
        var_names,
    )


@app.cell
def _(m, mo, n, n_bdl, n_missing, p_true):
    mo.md(
        f"""
        **Dataset summary**

        | Property | Value |
        |----------|-------|
        | Variables (species) | {m} |
        | Observations (samples) | {n} |
        | True source factors | {p_true} |
        | Missing values | {n_missing} |
        | Below-detection-limit entries | {n_bdl} |
        """
    )
    return


# ── Step 2: Data Preparation ─────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 2 -- Data Preparation

        Before running PMF we need a clean, non-negative data matrix **X** and a
        matching uncertainty matrix **sigma**.

        `prepare_data` handles:

        - **Missing values** -- replaced by the column median (a robust choice
          that avoids pulling the imputed value toward outliers).
        - **Below-detection-limit values** -- replaced by half the detection
          limit, with an inflated uncertainty (5/6 of the DL) following the
          Polissar et al. convention widely used in EPA PMF.
        - **Uncertainty estimation** -- automatic method selection based on the
          signal magnitude and detection limits.
        """
    )
    return


@app.cell
def _(X_raw, detection_limits, np, prepare_data):
    X_clean, sigma = prepare_data(
        X_raw,
        detection_limit=detection_limits,
        missing_method="median",
        bdl_replacement="half_dl",
        uncertainty_method="auto",
        verbose=False,
    )

    snr_median = float(np.median(X_clean / sigma))

    return X_clean, sigma, snr_median


@app.cell
def _(X_clean, mo, np, snr_median):
    mo.md(
        f"""
        **Prepared data**

        - Data range: [{X_clean.min():.3f}, {X_clean.max():.3f}]
        - All values finite and non-negative: {bool(np.all(np.isfinite(X_clean)) and np.all(X_clean >= 0))}
        - Median signal-to-noise ratio: {snr_median:.1f}

        A median S/N around 5--20 is typical for ambient air-quality data.
        Lower S/N means the solver must lean more on the uncertainty weighting
        to down-weight noisy observations.
        """
    )
    return


# ── Step 3: Factor Selection ─────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 3 -- Factor Selection

        Choosing the right number of factors *p* is the most consequential
        decision in a PMF study. Too few factors and distinct sources get merged;
        too many and the solver splits real sources or fits noise.

        `select_factors` runs PMF for each candidate *p* with multiple random
        initialisations and reports:

        - **Q** -- the objective function (weighted sum of squared residuals).
        - **Q_robust** = Q / E[Q] -- the ratio of the achieved Q to its expected
          value under a perfect model (degrees of freedom = m*n - p*(m+n)).
          A Q_robust near **1.0** signals a good fit; much below 1.0 suggests
          overfitting, much above 1.0 suggests underfitting.

        We look for the **smallest *p*** whose Q_robust is closest to 1.0.
        """
    )
    return


@app.cell
def _(X_clean, select_factors, sigma):
    selection = select_factors(
        X_clean,
        sigma,
        p_range=(2, 6),
        n_runs=5,
        random_seed=42,
        max_iter=100,
        verbose=False,
    )
    return (selection,)


@app.cell
def _(mo, np, plt, selection):
    _p_vals = selection.p_values
    _q_robust = [selection.Q_robust[p] for p in _p_vals]
    _q_means = [np.mean(selection.Q_values[p]) for p in _p_vals]
    _q_stds = [np.std(selection.Q_values[p]) for p in _p_vals]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # -- Q_robust vs p --
    _ax1.plot(_p_vals, _q_robust, "o-", color="steelblue", linewidth=2, markersize=8)
    _ax1.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="ideal Q_robust = 1")
    _ax1.axvline(selection.best_p, color="tomato", linestyle=":", linewidth=1.5,
                 label=f"selected p = {selection.best_p}")
    _ax1.set_xlabel("Number of factors (p)")
    _ax1.set_ylabel("Q_robust  (Q / E[Q])")
    _ax1.set_title("Factor Selection: Q_robust")
    _ax1.legend(fontsize=9)
    _ax1.set_xticks(_p_vals)

    # -- Q (raw) vs p --
    _ax2.errorbar(_p_vals, _q_means, yerr=_q_stds, fmt="s-", color="darkorange",
                  linewidth=2, markersize=7, capsize=4)
    _ax2.axvline(selection.best_p, color="tomato", linestyle=":", linewidth=1.5,
                 label=f"selected p = {selection.best_p}")
    _ax2.set_xlabel("Number of factors (p)")
    _ax2.set_ylabel("Q (objective)")
    _ax2.set_title("Factor Selection: Raw Q")
    _ax2.legend(fontsize=9)
    _ax2.set_xticks(_p_vals)

    _fig.tight_layout()
    mo.md(
        f"""
        **Result:** The recommended number of factors is **p = {selection.best_p}**
        (Q_robust = {selection.Q_robust[selection.best_p]:.4f}).

        The left panel shows Q_robust approaching 1.0 as *p* increases.  The right
        panel shows the raw Q values (mean +/- std across random starts) -- Q
        should decrease with *p*, but the marginal improvement flattens once the
        true number of sources is reached.
        """
    )
    return


# ── Step 4: Run PMF ──────────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 4 -- PMF Analysis

        With the number of factors chosen, we run PMF in `full` mode (which
        enables all internal convergence checks and rotational diagnostics).

        The solver minimises the Paatero Q objective:

        $$
        Q = \sum_{i,j} \left(\frac{X_{ij} - (FG)_{ij}}{\sigma_{ij}}\right)^2
        $$

        subject to the constraint that **F** (profiles) and **G** (contributions)
        are non-negative.  The per-element uncertainty weighting lets the solver
        automatically down-weight noisy or imputed data points.
        """
    )
    return


@app.cell
def _(X_clean, pmf, selection, sigma):
    result = pmf(
        X_clean,
        sigma,
        p=selection.best_p,
        mode="full",
        max_iter=200,
        random_seed=42,
        verbose=False,
    )
    return (result,)


@app.cell
def _(mo, result, selection):
    mo.md(
        f"""
        **PMF result**

        | Diagnostic | Value |
        |------------|-------|
        | Converged | {result.converged} |
        | Iterations | {result.n_iter} |
        | Final Q | {result.Q:.2f} |
        | Explained variance | {result.explained_variance:.2%} |
        | Chi-square | {result.chi2:.4f} |
        | Factors | {selection.best_p} |
        """
    )
    return


# ── Step 5: Uncertainty Estimation ───────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 5 -- Bootstrap Uncertainty

        A single PMF solution tells you the *best-fit* profiles and contributions,
        but not how **stable** they are.  Bootstrap displacement creates perturbed
        versions of the dataset and re-solves PMF on each, building up a
        distribution of F and G values.

        From the bootstrap ensemble we extract:

        - **Mean profiles** (F_mean) and **standard deviations** (F_std) --
          the spread tells you which species loadings are well-constrained and
          which are uncertain.
        - **Relative uncertainty** -- F_std / F_mean expressed as a percentage.
          Values below ~25% are generally considered well-determined.

        For a publication-quality analysis you would use 100--200 bootstrap
        replicates; here we use 20 for speed.
        """
    )
    return


@app.cell
def _(X_clean, bootstrap_uncertainty, np, result, selection, sigma):
    uncertainty = bootstrap_uncertainty(
        X_clean,
        sigma,
        p=selection.best_p,
        base_result=result,
        n_bootstrap=20,
        method="displacement",
        random_seed=42,
        max_iter=100,
        verbose=False,
    )

    rel_unc_F = float(np.mean(uncertainty.F_std / (uncertainty.F_mean + 1e-10)))
    rel_unc_G = float(np.mean(uncertainty.G_std / (uncertainty.G_mean + 1e-10)))

    return rel_unc_F, rel_unc_G, uncertainty


@app.cell
def _(mo, rel_unc_F, rel_unc_G, uncertainty):
    mo.md(
        f"""
        **Bootstrap summary**

        - Replicates completed: {len(uncertainty.bootstrap_results)}
        - Mean relative uncertainty (F): {rel_unc_F:.1%}
        - Mean relative uncertainty (G): {rel_unc_G:.1%}

        Lower relative uncertainty means the factor profiles are well-constrained
        by the data.  If specific species show very high uncertainty, that is a
        signal to investigate whether they carry useful source-apportionment
        information or are dominated by noise.
        """
    )
    return


@app.cell
def _(mo, np, plt, result, selection, uncertainty, var_names):
    _p = selection.best_p
    _fig, _axes = plt.subplots(1, _p, figsize=(4 * _p, 4), sharey=True)
    if _p == 1:
        _axes = [_axes]

    _x = np.arange(result.F.shape[0])
    _colors = plt.cm.Set2(np.linspace(0, 1, _p))

    for k in range(_p):
        _ax = _axes[k]
        _ax.bar(
            _x,
            uncertainty.F_mean[:, k],
            yerr=uncertainty.F_std[:, k],
            color=_colors[k],
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
            error_kw={"linewidth": 1.2},
        )
        _ax.set_title(f"Factor {k + 1}", fontsize=12, fontweight="bold")
        _ax.set_xlabel("Variable")
        _ax.set_xticks(_x)
        _ax.set_xticklabels(var_names, rotation=45, ha="right", fontsize=8)
        if k == 0:
            _ax.set_ylabel("Profile loading")

    _fig.suptitle(
        "Factor Profiles with Bootstrap Uncertainty",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    _fig.tight_layout()
    mo.md(
        r"""
        Each bar shows the **mean profile loading** across bootstrap replicates,
        with error bars representing +/- 1 standard deviation.  Narrow error bars
        indicate species that are tightly associated with a given source; wide
        bars flag species whose attribution is ambiguous.
        """
    )
    return


# ── Step 6: Factor Contributions ─────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 6 -- Factor Contributions

        `compute_contributions` decomposes the reconstructed matrix into
        per-factor slices:

        $$
        X \approx \sum_{k=1}^{p} F_{:,k} \, G_{k,:}
        $$

        Each slice `contributions[k]` has shape (m, n) and represents how much
        of the signal at every variable and sample is explained by factor *k*.
        Summing over variables gives the **total mass contribution** of each
        source.
        """
    )
    return


@app.cell
def _(compute_contributions, np, result, selection):
    contributions = compute_contributions(result.F, result.G)

    total_mass = float(np.sum(result.F @ result.G))
    factor_pcts = []
    for k in range(selection.best_p):
        pct = float(np.sum(contributions[k]) / total_mass * 100)
        factor_pcts.append(pct)

    return contributions, factor_pcts, total_mass


@app.cell
def _(contributions, factor_pcts, mo, np, plt, selection, var_names):
    _p = selection.best_p
    _labels = [f"Factor {k + 1}\n({factor_pcts[k]:.1f}%)" for k in range(_p)]
    _colors = plt.cm.Set2(np.linspace(0, 1, _p))

    _fig, (_ax_pie, _ax_bar) = plt.subplots(1, 2, figsize=(12, 4.5))

    # -- Pie chart: overall source contributions --
    _wedges, _texts, _autotexts = _ax_pie.pie(
        factor_pcts,
        labels=_labels,
        colors=_colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10},
    )
    for _t in _autotexts:
        _t.set_fontsize(9)
    _ax_pie.set_title("Overall Source Contributions", fontsize=12, fontweight="bold")

    # -- Stacked bar: per-variable contribution --
    _m = contributions.shape[1]
    _x = np.arange(_m)
    _bottom = np.zeros(_m)
    for k in range(_p):
        # contributions[k] is (m, n); sum over samples for per-variable totals
        _var_k = np.sum(contributions[k], axis=1)
        _ax_bar.bar(_x, _var_k, bottom=_bottom, color=_colors[k],
                    edgecolor="black", linewidth=0.3, label=f"Factor {k + 1}")
        _bottom += _var_k

    _ax_bar.set_xticks(_x)
    _ax_bar.set_xticklabels(var_names, rotation=45, ha="right", fontsize=8)
    _ax_bar.set_ylabel("Reconstructed signal (summed over samples)")
    _ax_bar.set_title("Per-Variable Source Apportionment", fontsize=12, fontweight="bold")
    _ax_bar.legend(fontsize=9)

    _fig.tight_layout()

    mo.md(
        f"""
        The pie chart shows the **fraction of total reconstructed mass**
        attributed to each factor.  The stacked bar chart breaks this down
        by variable, showing which species are dominated by which source --
        the key output for source apportionment.
        """
    )
    return


# ── Step 7: Diagnostics ──────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 7 -- Residual Diagnostics

        Good PMF solutions should have **scaled residuals**
        $e_{ij} = (X_{ij} - (FG)_{ij}) / \sigma_{ij}$
        that look like standard-normal white noise:

        - The heatmap should show no spatial structure (no rows or columns
          with systematically large residuals).
        - The histogram should be roughly bell-shaped and centred near zero.

        Patterns in the residuals indicate model mis-specification -- for
        example, a missing source, incorrect uncertainties, or non-linear
        mixing.
        """
    )
    return


@app.cell
def _(X_clean, mo, np, plt, result, sigma, var_names):
    _residuals = (X_clean - result.F @ result.G) / sigma

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # -- Residual heatmap --
    _im = _ax1.imshow(_residuals, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    _ax1.set_xlabel("Sample index")
    _ax1.set_ylabel("Variable")
    _ax1.set_yticks(range(len(var_names)))
    _ax1.set_yticklabels(var_names, fontsize=8)
    _ax1.set_title("Scaled Residuals (e = (X - FG) / sigma)", fontsize=11)
    _cb = _fig.colorbar(_im, ax=_ax1, shrink=0.8)
    _cb.set_label("Scaled residual")

    # -- Histogram --
    _ax2.hist(_residuals.ravel(), bins=40, density=True, color="steelblue",
              edgecolor="black", linewidth=0.5, alpha=0.8, label="Residuals")
    _xx = np.linspace(-4, 4, 200)
    _ax2.plot(_xx, np.exp(-_xx**2 / 2) / np.sqrt(2 * np.pi), "r-",
              linewidth=2, label="N(0,1)")
    _ax2.set_xlabel("Scaled residual")
    _ax2.set_ylabel("Density")
    _ax2.set_title("Residual Distribution", fontsize=11)
    _ax2.legend()

    _fig.tight_layout()

    _rmse = float(np.sqrt(np.mean(_residuals**2)))
    _frac_gt2 = float(np.mean(np.abs(_residuals) > 2) * 100)

    mo.md(
        f"""
        **Residual statistics**

        - Root-mean-square scaled residual: {_rmse:.3f} (ideal: 1.0)
        - Fraction of residuals with |e| > 2: {_frac_gt2:.1f}% (expect ~5% for normal)

        If the RMSE is substantially above 1.0 or the tails are heavy, consider
        adding more factors or using the robust solver (`robust=True` in the
        Bayesian backend) to down-weight outliers.
        """
    )
    return


# ── Summary ───────────────────────────────────────────────────────────────


@app.cell
def _(mo, p_true, rel_unc_F, result, selection):
    mo.md(
        f"""
        ---

        ## Workflow Summary

        | Step | Outcome |
        |------|---------|
        | Data preparation | Missing values imputed, BDL handled, uncertainties estimated |
        | Factor selection | Tested p = {selection.p_values[0]}..{selection.p_values[-1]}; selected **p = {selection.best_p}** (true: {p_true}) |
        | PMF solution | Converged in {result.n_iter} iterations, explained variance {result.explained_variance:.1%} |
        | Uncertainty | Mean relative uncertainty on profiles: {rel_unc_F:.1%} |
        | Diagnostics | Residuals checked for structure and normality |

        This completes the standard PMF analysis pipeline.  From here a
        practitioner would typically:

        - Interpret the factor profiles by matching species loadings to known
          source signatures (e.g. crustal dust, vehicle exhaust, sea salt).
        - Apply **FPEAK rotation** to explore rotational ambiguity.
        - Compare with the **Bayesian PMF** backend for posterior uncertainty
          and automatic relevance determination (ARD) -- see
          `05_bayesian_demo.py`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
