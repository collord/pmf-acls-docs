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
        # Factor Selection in PMF

        Choosing the right number of factors (*p*) is one of the most important
        decisions in Positive Matrix Factorization.  Too few factors blend distinct
        sources together; too many split real sources or fit noise.

        This notebook demonstrates the `select_factors()` helper, which runs PMF
        for a range of *p* values and reports the key diagnostic **Q/Q_exp**
        (also called *Q_robust*).

        ## What is Q/Q_exp?

        The PMF objective function **Q** sums the squared, uncertainty-weighted
        residuals:

        $$Q = \sum_{i,j}\left(\frac{x_{ij} - \sum_k f_{ik}\,g_{kj}}{\sigma_{ij}}\right)^2$$

        If the model is correct and the uncertainties are well-calibrated, Q should
        follow a chi-squared distribution whose expectation is roughly the number
        of degrees of freedom:

        $$Q_{\text{exp}} \approx m \times n - p\,(m + n)$$

        The ratio **Q / Q_exp** (reported as `Q_robust`) should therefore be close
        to **1.0** for the correct *p*.  Values much greater than 1 mean the model
        under-fits; values well below 1 suggest over-fitting.
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

    from pmf_acls import select_factors

    return mo, np, plt, select_factors


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Generate synthetic data with a known number of factors

        We create an 8-variable, 40-observation dataset generated from exactly
        **3 true factors** plus Gaussian noise.  Because we know the ground truth
        we can verify that the selection procedure recovers *p = 3*.
        """
    )
    return


@app.cell
def _(np):
    rng = np.random.default_rng(42)

    m, n, p_true = 8, 40, 3

    F_true = rng.random((m, p_true)) + 0.5
    G_true = rng.random((p_true, n)) + 0.5
    X = F_true @ G_true + 0.1 * rng.standard_normal((m, n))
    sigma = 0.1 * np.ones_like(X)

    return F_true, G_true, X, m, n, p_true, sigma


@app.cell
def _(m, mo, n, p_true):
    mo.md(
        f"""
        **Data dimensions:** {m} variables x {n} observations

        **True number of factors:** {p_true}

        **Noise level:** constant sigma = 0.1 (homoscedastic)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Why multiple seeds per *p*?

        PMF is a non-convex optimisation problem -- different random
        initialisations can converge to different local minima.  Running several
        seeds for each candidate *p* lets us:

        1. **Find a better minimum** -- the run with the lowest Q is kept as the
           representative solution for that *p*.
        2. **Gauge stability** -- if all seeds give similar Q values the solution
           is robust; large spread hints at rotational ambiguity or a poor model.

        Below we test *p* from 1 to 6 with 5 random seeds each.
        """
    )
    return


@app.cell
def _(X, select_factors, sigma):
    selection_result = select_factors(
        X,
        sigma,
        p_range=(1, 6),
        n_runs=5,
        random_seed=42,
        max_iter=100,
        verbose=True,
    )
    return (selection_result,)


@app.cell
def _(mo, np, p_true, selection_result):
    rows = []
    for p in selection_result.p_values:
        q_robust = selection_result.Q_robust[p]
        q_vals = selection_result.Q_values[p]
        q_mean = np.mean(q_vals)
        q_std = np.std(q_vals)
        ev = selection_result.best_results[p].explained_variance

        tag = ""
        if p == selection_result.best_p:
            tag += " **Recommended**"
        if p == p_true:
            tag += " *(true p)*"

        rows.append(
            f"| {p} | {q_robust:.4f} | {q_mean:.3e} +/- {q_std:.3e} "
            f"| {ev:.2%} |{tag} |"
        )

    table = "\n".join(rows)
    mo.md(
        f"""
        ## Results

        | p | Q_robust (Q/Q_exp) | Q (mean +/- std) | Explained variance | Note |
        |---|---|---|---|---|
        {table}

        **Recommended number of factors: {selection_result.best_p}**

        Q_robust closest to 1.0 indicates the best trade-off between fit quality
        and model complexity.  Values well above 1 mean under-fitting; values
        well below 1 suggest over-fitting or over-parameterisation.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Diagnostic plots

        Two complementary views of the same information:

        1. **Q / Q_exp vs *p*** -- look for the value nearest 1.0 (dashed line).
        2. **Explained variance vs *p*** -- look for diminishing returns after the
           correct *p* (the "elbow").
        """
    )
    return


@app.cell
def _(np, p_true, plt, selection_result):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ps = selection_result.p_values
    q_robust_vals = [selection_result.Q_robust[p] for p in ps]
    ev_vals = [
        selection_result.best_results[p].explained_variance for p in ps
    ]

    # --- Q / Q_exp plot ---
    ax = axes[0]
    ax.plot(ps, q_robust_vals, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Q/Q_exp = 1")

    best_idx = ps.index(selection_result.best_p)
    ax.plot(
        selection_result.best_p,
        q_robust_vals[best_idx],
        "s",
        color="crimson",
        markersize=14,
        zorder=5,
        label=f"Recommended p={selection_result.best_p}",
    )

    # Annotate the true p if different from best
    if p_true != selection_result.best_p:
        true_idx = ps.index(p_true)
        ax.annotate(
            f"True p={p_true}",
            xy=(p_true, q_robust_vals[true_idx]),
            xytext=(p_true + 0.3, q_robust_vals[true_idx] + 0.05),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black"),
        )

    ax.set_xlabel("Number of factors (p)")
    ax.set_ylabel("Q / Q_exp")
    ax.set_title("Q / Q_exp vs number of factors")
    ax.set_xticks(ps)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Explained variance bar chart ---
    ax = axes[1]
    colors = [
        "crimson" if p == selection_result.best_p else "steelblue" for p in ps
    ]
    bars = ax.bar(ps, [v * 100 for v in ev_vals], color=colors, edgecolor="white")

    for bar, val in zip(bars, ev_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Number of factors (p)")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("Explained variance vs number of factors")
    ax.set_xticks(ps)
    ax.set_ylim(0, np.max([v * 100 for v in ev_vals]) * 1.08)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Interpreting the results

        * **Q/Q_exp plot:** The recommended *p* (red square) should sit near the
          dashed line at 1.0.  Smaller *p* values have Q/Q_exp >> 1 (under-fit),
          while larger values drop below 1 (over-fit).

        * **Explained variance:** After the true number of factors, additional
          factors yield diminishing gains -- the "elbow" in the bar chart.

        * **Multi-seed spread:** If Q values for a given *p* show large variance
          across seeds, the solution landscape has many local minima and the
          results may be sensitive to initialisation.  This is common when *p* is
          too large.

        ### Practical guidance

        1. Start with the Q/Q_exp ~ 1 criterion.
        2. Cross-check with the explained-variance elbow.
        3. **Always** inspect the resulting factor profiles for physical
           interpretability -- a statistically optimal *p* is useless if the
           factors are not meaningful.
        """
    )
    return


if __name__ == "__main__":
    app.run()
