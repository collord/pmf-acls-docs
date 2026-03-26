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
        # Basic PMF Example

        **Positive Matrix Factorization (PMF)** decomposes a data matrix
        $X$ (variables x observations) into two non-negative matrices:

        $$X \approx F \cdot G$$

        where:

        - **F** (m x p) — *factor profiles*: each column describes one source's
          fingerprint across the measured variables.
        - **G** (p x n) — *factor contributions*: each row describes how strongly
          one source contributes to each observation.

        Unlike ordinary NMF, PMF uses **per-element uncertainty weights**
        $\sigma_{ij}$. The objective it minimises is the weighted chi-squared
        statistic:

        $$Q = \sum_{i,j} \left(\frac{X_{ij} - (FG)_{ij}}{\sigma_{ij}}\right)^2$$

        A good fit has $Q / (m \cdot n) \approx 1$, meaning residuals are
        on average one standard deviation of measurement uncertainty.

        This notebook walks through the simplest possible workflow:
        generate synthetic data with known factors, run PMF, and inspect
        how well it recovers the truth.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import linear_sum_assignment

    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["axes.edgecolor"] = "black"

    from pmf_acls import pmf

    return linear_sum_assignment, mo, np, plt, pmf


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. Generate synthetic data

        We create a small problem with **10 variables**, **50 observations**,
        and **3 true factors**. Each element of the true F and G is drawn
        uniformly from [0.5, 1.5] so every factor has a clear, non-trivial
        profile. Gaussian noise at 10% of the signal is added to X.
        """
    )
    return


@app.cell
def _(np):
    rng = np.random.default_rng(42)

    m, n, p_true = 10, 50, 3  # 10 variables, 50 observations, 3 true factors

    # True factor profiles (F) and contributions (G)
    F_true = rng.uniform(0.5, 1.5, size=(m, p_true))
    G_true = rng.uniform(0.5, 1.5, size=(p_true, n))

    # Generate data: X = F @ G + noise
    noise = 0.1 * rng.standard_normal(size=(m, n))
    X = F_true @ G_true + noise

    # Uncertainties (constant for simplicity)
    sigma = 0.1 * np.ones_like(X)

    return F_true, G_true, X, m, n, p_true, sigma


@app.cell
def _(X, m, mo, n, p_true):
    mo.md(
        f"""
        **Data generated.**

        | Property | Value |
        |----------|-------|
        | Variables (m) | {m} |
        | Observations (n) | {n} |
        | True factors (p) | {p_true} |
        | Data shape | {X.shape} |
        | Uncertainty model | constant $\\sigma = 0.1$ |
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Run PMF

        We call `pmf()` requesting 3 factors (matching the truth here).
        The default algorithm is **ACLS** (Alternating Constrained Least
        Squares), which iteratively updates F and G while enforcing
        non-negativity.

        Key parameters:

        - `p` — number of factors to extract
        - `max_iter` — iteration budget
        - `conv_tol` — stop when the relative change in Q falls below this
        - `random_seed` — for reproducible random initialisation
        """
    )
    return


@app.cell
def _(X, pmf, sigma):
    result = pmf(
        X, sigma,
        p=3,
        max_iter=10000,
        conv_tol=1e-6,
        random_seed=42,
        verbose=True,
    )
    return (result,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Inspect results

        The `result` object exposes convergence diagnostics and the
        recovered matrices. The most important numbers are:

        - **Converged** — did Q stabilise within the iteration budget?
        - **Q** — the weighted chi-squared objective. Divide by $m \times n$
          to get the average weighted-squared residual.
        - **Explained variance** — fraction of variance in X captured
          by the reconstruction $FG$.
        """
    )
    return


@app.cell
def _(X, m, mo, n, np, result):
    X_reconstructed = result.F @ result.G
    mse = np.mean((X - X_reconstructed) ** 2)

    mo.md(
        f"""
        ### Summary

        | Metric | Value |
        |--------|-------|
        | Converged | {result.converged} |
        | Iterations | {result.n_iter} |
        | Final Q | {result.Q:.4e} |
        | Q / (m x n) | {result.Q / (m * n):.4f} |
        | Explained variance | {result.explained_variance:.2%} |
        | Chi-square | {result.chi2:.4f} |
        | F shape | {result.F.shape} |
        | G shape | {result.G.shape} |
        | Mean squared reconstruction error | {mse:.4e} |

        A $Q/(m \\times n)$ close to 1.0 means the residuals are
        well-calibrated against the stated uncertainties.
        """
    )
    return X_reconstructed, mse


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Compare recovered profiles to truth

        PMF does not guarantee that the order of factors matches the
        generating order. We use **Hungarian matching** (minimum-cost
        assignment on cosine distances) to align recovered factors to the
        true factors before comparing them.
        """
    )
    return


@app.cell
def _(F_true, linear_sum_assignment, np, result):
    def match_factors(F_est, F_ref):
        """Return column permutation of F_est that best matches F_ref."""
        p = F_ref.shape[1]
        cost = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                cos_sim = (
                    np.dot(F_ref[:, i], F_est[:, j])
                    / (np.linalg.norm(F_ref[:, i]) * np.linalg.norm(F_est[:, j]) + 1e-12)
                )
                cost[i, j] = 1.0 - cos_sim
        row_ind, col_ind = linear_sum_assignment(cost)
        return col_ind

    perm = match_factors(result.F, F_true)
    F_matched = result.F[:, perm]
    return F_matched, match_factors, perm


@app.cell
def _(F_matched, F_true, m, np, plt):
    fig_bar, axes_bar = plt.subplots(1, F_true.shape[1], figsize=(12, 4), sharey=True)
    x_pos = np.arange(m)
    bar_width = 0.35

    for k in range(F_true.shape[1]):
        ax = axes_bar[k]
        ax.bar(x_pos - bar_width / 2, F_true[:, k], bar_width, label="True", color="#4c72b0")
        ax.bar(x_pos + bar_width / 2, F_matched[:, k], bar_width, label="Recovered", color="#dd8452")
        ax.set_xlabel("Variable")
        if k == 0:
            ax.set_ylabel("Loading")
        ax.set_title(f"Factor {k + 1}")
        ax.legend(fontsize=8)

    fig_bar.suptitle("True vs Recovered Factor Profiles (Hungarian-matched)", fontsize=13, y=1.02)
    fig_bar.tight_layout()
    fig_bar
    return (fig_bar,)


@app.cell
def _(mo):
    mo.md(
        r"""
        The bar charts show that PMF closely recovers the true factor
        profiles despite the added noise. Small differences are expected
        due to rotational ambiguity and measurement noise.

        ## 5. Reconstruction error heatmap

        The matrix of residuals $(X - FG)$ reveals whether any particular
        variable-observation combination is poorly explained. Ideally the
        residuals look like spatially-uniform noise with magnitudes
        comparable to $\sigma$.
        """
    )
    return


@app.cell
def _(X, X_reconstructed, plt):
    fig_heat, ax_heat = plt.subplots(figsize=(10, 4))
    im = ax_heat.imshow(
        X - X_reconstructed,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-0.3,
        vmax=0.3,
    )
    ax_heat.set_xlabel("Observation")
    ax_heat.set_ylabel("Variable")
    ax_heat.set_title("Reconstruction Error  (X - FG)")
    fig_heat.colorbar(im, ax=ax_heat, label="Residual")
    fig_heat.tight_layout()
    fig_heat
    return (fig_heat,)


@app.cell
def _(mo):
    mo.md(
        r"""
        A well-fit model shows a heatmap dominated by low-magnitude,
        randomly scattered residuals (no visible structure). Streaks or
        blocks would indicate a missing factor or model misspecification.

        ---

        **Next steps:**

        - Try changing `p` to 2 or 4 and see how Q and the profiles change.
        - Replace the constant $\sigma$ with heteroscedastic uncertainties
          (e.g., proportional to X) for a more realistic scenario.
        - Explore `02_factor_selection.py` for systematic selection of the
          number of factors.
        """
    )
    return


if __name__ == "__main__":
    app.run()
