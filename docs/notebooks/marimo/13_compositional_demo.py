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
        # Compositional Data Analysis for PMF

        When source profiles or observations are **compositional** (mass fractions,
        percentages, or any data normalized to a constant sum), standard PMF
        operates in the wrong geometry. The simplex constraint introduces:

        1. **Spurious correlations** (Pearson 1897) — closure forces negative
           correlation even among independent variables
        2. **Sub-compositional incoherence** — dropping a species and re-closing
           changes distances, so results depend on which species you measure
        3. **Incorrect rank estimation** — PCA in Euclidean space may find
           spurious or miss real components due to closure artifacts

        The **Aitchison geometry** (Aitchison 1982, Egozcue et al. 2003) is the
        correct framework for the simplex. The **Isometric Log-Ratio (ILR)**
        transform maps compositions to unconstrained Euclidean space while
        preserving distances.

        This notebook demonstrates:
        - **ILR scree analysis** for compositional rank estimation (the primary use case)
        - Comparison with standard (Euclidean) PCA — why they diverge on closed data
        - ILR-PCA initialization for PMF (a convenience, not a guarantee of better factors)
        - Low-level CoDA transforms
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

    from pmf_acls import pmf, ilr_pca_init, ilr_scree
    from pmf_acls.coda import (
        closure,
        clr_transform,
        ilr_transform,
        ilr_inverse,
        aitchison_distance,
        multiplicative_replacement,
    )

    return (
        aitchison_distance,
        closure,
        clr_transform,
        ilr_inverse,
        ilr_pca_init,
        ilr_scree,
        ilr_transform,
        mo,
        multiplicative_replacement,
        np,
        plt,
        pmf,
    )


# --- Configuration ---


@app.cell
def _(mo):
    mo.md("## 1. Configuration")
    return


@app.cell
def _(mo):
    n_vars_slider = mo.ui.slider(8, 30, step=1, value=15, label="Variables (m)")
    n_obs_slider = mo.ui.slider(40, 200, step=20, value=100, label="Observations (n)")
    n_sources_slider = mo.ui.slider(2, 6, step=1, value=4, label="True sources")
    noise_slider = mo.ui.slider(0.02, 0.30, step=0.02, value=0.08, label="Noise fraction")
    seed_number = mo.ui.number(value=42, label="Random seed")
    close_data_toggle = mo.ui.switch(value=True, label="Close data (sum-to-1)")

    mo.vstack([
        mo.hstack(
            [n_vars_slider, n_obs_slider, n_sources_slider],
            justify="start", gap=1.5,
        ),
        mo.hstack(
            [noise_slider, seed_number, close_data_toggle],
            justify="start", gap=1.5,
        ),
    ])
    return (
        close_data_toggle,
        n_obs_slider,
        n_sources_slider,
        n_vars_slider,
        noise_slider,
        seed_number,
    )


# --- Synthetic data ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. Synthetic Compositional Data

        We generate data from known source profiles on the simplex
        (each profile sums to 1 over variables). When "Close data" is on,
        the observations are also closed — simulating mass fraction data
        where closure artifacts are strongest.
        """
    )
    return


@app.cell
def _(
    close_data_toggle,
    closure,
    mo,
    n_obs_slider,
    n_sources_slider,
    n_vars_slider,
    noise_slider,
    np,
    seed_number,
):
    _rng = np.random.default_rng(int(seed_number.value))
    _m = n_vars_slider.value
    _n = n_obs_slider.value
    _p = n_sources_slider.value

    # True source profiles on the simplex
    F_true = _rng.dirichlet(np.ones(_m) * 0.5, size=_p).T  # (m, p)

    # Source contributions (gamma-distributed, varying strengths)
    _scales = np.logspace(1.5, 0, _p)  # decreasing strength
    G_true = np.zeros((_p, _n))
    for k in range(_p):
        G_true[k, :] = _rng.exponential(_scales[k], size=_n)

    # Clean signal + noise
    X_clean = F_true @ G_true
    _sigma_true = noise_slider.value * np.maximum(X_clean, 0.01) + 0.001
    X_raw = np.maximum(X_clean + _rng.normal(0, _sigma_true), 1e-8)

    # Optionally close (normalize columns to sum to 1)
    if close_data_toggle.value:
        X = closure(X_raw.T).T * X_raw.sum(axis=0, keepdims=True).mean()
    else:
        X = X_raw

    sigma = noise_slider.value * np.maximum(X, 0.01) + 0.001
    var_names = [f"V{i+1:02d}" for i in range(_m)]

    mo.md(
        f"**Data**: {_m} variables x {_n} observations, "
        f"**{_p} true sources**, noise = {noise_slider.value:.0%}, "
        f"closed = {close_data_toggle.value}"
    )
    return F_true, G_true, X, sigma, var_names


# --- ILR Scree vs Standard PCA ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. Scree Analysis: ILR-PCA vs Standard PCA

        The **ILR scree plot** performs PCA in the Aitchison geometry
        (after ILR transform), giving rank estimates that are invariant
        to sub-composition selection and free of closure artifacts.

        The **standard scree plot** performs PCA on the raw data.
        When data is closed (compositional), standard PCA may show
        spurious components or miss real ones.

        ### What to look for

        **Left panel (singular values):**

        1. **The elbow.** Look for a sharp bend where singular values
           transition from "signal" (large, steep decline) to "noise"
           (small, flat tail). The number of components *before* the
           elbow is your rank estimate.
        2. **ILR vs Standard divergence.** If the two curves suggest
           different elbows, the standard PCA is likely being distorted
           by closure. Trust the ILR curve for compositional data.
        3. **Gradual decay vs sharp drop.** A clean break means
           well-separated sources; a gradual tail means overlapping
           sources or high noise — consider a range of p values rather
           than a single choice.

        **Right panel (cumulative explained variance):**

        1. **The 90-95% threshold.** A common heuristic: choose the
           smallest p where cumulative variance exceeds ~95%. This is
           a rough guide, not a rule — for noisy environmental data,
           80-90% may be more appropriate.
        2. **Diminishing returns.** If adding component p+1 explains
           < 2-3% additional variance, the marginal source may not
           be identifiable and you risk overfitting.
        3. **ILR vs Standard cumulative.** On closed data, standard
           PCA often reaches 95% with *fewer* components (the closure
           constraint artificially reduces apparent dimensionality).
           The ILR curve gives the honest rank.

        **General guidance:** The scree plot narrows the plausible
        range of p (e.g., "3 to 5 sources"). Final selection should
        combine the scree with PMF diagnostics — Q/Q_expected,
        bootstrap stability, and physical interpretability of profiles.
        """
    )
    return


@app.cell
def _(F_true, X, ilr_scree, np, plt):
    _p_true = F_true.shape[1]

    # ILR scree
    S_ilr, explained_ilr = ilr_scree(X)

    # Standard PCA scree (on centered data)
    _X_centered = X - X.mean(axis=1, keepdims=True)
    _, S_std, _ = np.linalg.svd(_X_centered.T, full_matrices=False)
    _total_var_std = np.sum(S_std ** 2)
    explained_std = S_std ** 2 / _total_var_std

    _max_k = min(12, len(S_ilr), len(S_std))
    _k = np.arange(1, _max_k + 1)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: singular values
    _ax1.plot(_k, S_ilr[:_max_k], "o-", label="ILR (Aitchison)", color="steelblue")
    _ax1.plot(_k, S_std[:_max_k], "s--", label="Standard (Euclidean)", color="coral")
    _ax1.axvline(_p_true, color="gray", linestyle=":", label=f"True rank = {_p_true}")
    _ax1.set_xlabel("Component")
    _ax1.set_ylabel("Singular value")
    _ax1.set_title("Scree Plot")
    _ax1.legend(fontsize=8)

    # Right: cumulative explained variance
    _ax2.plot(_k, np.cumsum(explained_ilr[:_max_k]), "o-",
              label="ILR", color="steelblue")
    _ax2.plot(_k, np.cumsum(explained_std[:_max_k]), "s--",
              label="Standard", color="coral")
    _ax2.axvline(_p_true, color="gray", linestyle=":", label=f"True rank = {_p_true}")
    _ax2.axhline(0.95, color="black", linestyle=":", alpha=0.3, label="95%")
    _ax2.set_xlabel("Component")
    _ax2.set_ylabel("Cumulative explained variance")
    _ax2.set_title("Cumulative Variance")
    _ax2.legend(fontsize=8)
    _ax2.set_ylim(0, 1.05)

    _fig.suptitle("Rank Estimation: ILR vs Standard PCA", fontsize=13)
    _fig.tight_layout()
    _fig
    return S_ilr, explained_ilr


@app.cell
def _(F_true, S_ilr, explained_ilr, mo, np):
    _p_true = F_true.shape[1]
    _header = "| Component | Singular Value | Explained | Cumulative |"
    _sep = "|---:|---:|---:|---:|"
    _rows = [_header, _sep]
    _cumsum = 0.0
    for _i in range(min(8, len(S_ilr))):
        _cumsum += explained_ilr[_i]
        _marker = " <--" if _i + 1 == _p_true else ""
        _rows.append(
            f"| {_i+1} | {S_ilr[_i]:.2f} | {explained_ilr[_i]:.1%} "
            f"| {_cumsum:.1%}{_marker} |"
        )

    # Find elbow: biggest drop in explained variance
    _diffs = np.diff(explained_ilr[:8])
    _elbow = int(np.argmin(_diffs)) + 1  # component after biggest drop

    # Ratio of consecutive singular values (signal/noise contrast)
    _ratio_rows = []
    for _j in range(min(7, len(S_ilr) - 1)):
        _ratio = S_ilr[_j] / S_ilr[_j + 1] if S_ilr[_j + 1] > 0 else float("inf")
        _ratio_rows.append(f"| {_j+1} → {_j+2} | {_ratio:.2f}x |")

    mo.md(
        f"**ILR Scree Table**\n\n"
        + "\n".join(_rows)
        + f"\n\nBiggest drop after component **{_elbow}** "
        + f"(true rank = {_p_true})"
        + f"\n\n**Consecutive singular value ratios** "
        + f"(large ratio = clear signal/noise gap):\n\n"
        + "| Transition | Ratio |\n|---:|---:|\n"
        + "\n".join(_ratio_rows)
        + f"\n\nLook for the largest ratio — a value > 2x suggests a "
        + f"clear separation between signal and noise at that transition."
    )
    return


# --- ILR-PCA Initialization ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 4. ILR-PCA Initialization vs Random Initialization

        `ilr_pca_init(X, p)` derives initial factor matrices from the
        leading ILR-PCA singular vectors, back-transformed to the simplex.

        **Important caveat:** PCA finds orthogonal directions of maximum
        variance, while NMF finds non-negative additive components — these
        are fundamentally different decompositions. Back-transforming PCA
        directions to the simplex gives valid compositions, but they need
        not resemble physical source profiles. Multi-start random
        initialization often recovers better factors because it explores
        more of the solution landscape.

        The primary value of `ilr_scree` is **rank estimation** (Section 3),
        not factor initialization. `ilr_pca_init` is a convenience for
        cases where a deterministic starting point is desired.
        """
    )
    return


@app.cell
def _(F_true, X, ilr_pca_init, mo, np, pmf, sigma):
    _p = F_true.shape[1]
    _m, _n = X.shape

    # ILR-PCA initialized PMF
    F_init_ilr, G_init_ilr = ilr_pca_init(X, _p)
    r_ilr = pmf(X, sigma, _p, F_init=F_init_ilr, G_init=G_init_ilr,
                max_iter=2000, conv_tol=0.001)

    # Random init PMF (best of 10 seeds)
    _best_Q = np.inf
    r_random = None
    for _seed in range(10):
        _r = pmf(X, sigma, _p, max_iter=2000, conv_tol=0.001, random_seed=_seed)
        if _r.Q < _best_Q:
            _best_Q = _r.Q
            r_random = _r

    mo.md(
        f"**ILR-PCA init**: Q = {r_ilr.Q:.2f}, "
        f"Q/mn = {r_ilr.Q / (_m * _n):.4f}, "
        f"converged = {r_ilr.converged}, "
        f"iterations = {r_ilr.n_iter}\n\n"
        f"**Random init** (best of 10): Q = {r_random.Q:.2f}, "
        f"Q/mn = {r_random.Q / (_m * _n):.4f}, "
        f"converged = {r_random.converged}, "
        f"iterations = {r_random.n_iter}"
    )
    return r_ilr, r_random


# --- Factor Profile Comparison ---


@app.cell
def _(mo):
    mo.md(
        """
        ### Factor Profiles: ILR-PCA Init vs Random Init vs Truth

        Both results are Hungarian-matched to truth for visual comparison.
        Multi-start random init (best of 10) typically matches truth as
        well or better than ILR-PCA init — this is expected, since PCA
        singular vectors are not NMF sources.
        """
    )
    return


@app.cell
def _(F_true, closure, np, plt, r_ilr, r_random, var_names):
    from scipy.optimize import linear_sum_assignment

    def _hungarian_match(F_est, F_ref):
        """Hungarian match columns of F_est to F_ref, return permutation."""
        _p = F_est.shape[1]
        _corr = np.zeros((_p, _p))
        for _i in range(_p):
            for _j in range(_p):
                _c = np.corrcoef(F_est[:, _i], F_ref[:, _j])[0, 1]
                _corr[_i, _j] = _c if np.isfinite(_c) else 0.0
        _row_ind, _col_ind = linear_sum_assignment(-np.abs(_corr))
        return _row_ind, _col_ind, _corr

    _p = F_true.shape[1]
    _m = F_true.shape[0]
    _x = np.arange(_m)

    # Align both results to truth independently — each gets its own permutation
    _ri_ilr, _ci_ilr, _ = _hungarian_match(r_ilr.F, F_true)
    _ri_rand, _ci_rand, _ = _hungarian_match(r_random.F, F_true)

    # Build reorder arrays: position k gets the estimated factor that matched true factor k
    _order_ilr = np.empty(_p, dtype=int)
    _order_rand = np.empty(_p, dtype=int)
    for _k in range(_p):
        _order_ilr[_ci_ilr[_k]] = _ri_ilr[_k]
        _order_rand[_ci_rand[_k]] = _ri_rand[_k]

    # L1-normalize for visual comparison
    _F_true_norm = closure(F_true.T).T
    _F_ilr_norm = closure(r_ilr.F[:, _order_ilr].T).T
    _F_rand_norm = closure(r_random.F[:, _order_rand].T).T

    _fig, _axes = plt.subplots(2, _p, figsize=(3.5 * _p, 6), sharey=False)
    if _p == 1:
        _axes = _axes.reshape(2, 1)

    for _row, (_F_norm, _label) in enumerate([
        (_F_ilr_norm, "ILR-PCA init"),
        (_F_rand_norm, "Random init"),
    ]):
        for _k in range(_p):
            _ax = _axes[_row, _k]
            _c = np.corrcoef(_F_norm[:, _k], _F_true_norm[:, _k])[0, 1]
            _r = abs(_c) if np.isfinite(_c) else 0.0

            _ax.bar(_x, _F_norm[:, _k], color="steelblue", alpha=0.7,
                    label=_label)
            _ax.hlines(
                _F_true_norm[:, _k], _x - 0.35, _x + 0.35,
                colors="black", linewidths=2, label="True",
            )
            _ax.set_title(f"Factor {_k+1} (|r|={_r:.3f})", fontsize=9)
            _ax.set_xticks(_x)
            _ax.set_xticklabels(var_names, rotation=90, fontsize=6)
            if _k == 0:
                _ax.set_ylabel(_label, fontsize=9)
                _ax.legend(fontsize=6)

    _fig.suptitle("Normalized Factor Profiles vs Truth", fontsize=12)
    _fig.tight_layout()
    _fig
    return


# --- CoDA Transform Demo ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 5. CoDA Transform Walkthrough

        Demonstration of the core transforms on a single composition.
        The ILR transform maps D-component compositions to (D-1)-dimensional
        Euclidean space while preserving Aitchison distances.
        """
    )
    return


@app.cell
def _(
    aitchison_distance,
    closure,
    clr_transform,
    ilr_inverse,
    ilr_transform,
    mo,
    np,
):
    # Two example compositions
    x = np.array([0.50, 0.30, 0.15, 0.05])
    y = np.array([0.25, 0.25, 0.25, 0.25])

    x_closed = closure(x)
    y_closed = closure(y)

    x_clr = clr_transform(x_closed)
    y_clr = clr_transform(y_closed)

    x_ilr = ilr_transform(x_closed)
    y_ilr = ilr_transform(y_closed)

    d_aitchison = aitchison_distance(x, y)
    d_euclidean_ilr = float(np.linalg.norm(x_ilr - y_ilr))
    d_euclidean_raw = float(np.linalg.norm(x - y))

    # Roundtrip
    x_roundtrip = ilr_inverse(x_ilr)

    _fmt = lambda a: "[" + ", ".join(f"{v:.4f}" for v in a) + "]"

    mo.md(
        f"**Composition x**: {_fmt(x)}\n\n"
        f"**Composition y**: {_fmt(y)} (uniform)\n\n"
        f"---\n\n"
        f"**CLR(x)**: {_fmt(x_clr)} (sums to {x_clr.sum():.1e})\n\n"
        f"**ILR(x)**: {_fmt(x_ilr)} ({len(x_ilr)} coordinates for {len(x)} components)\n\n"
        f"**ILR roundtrip**: {_fmt(x_roundtrip)} (matches closure(x))\n\n"
        f"---\n\n"
        f"| Distance | Value |\n"
        f"|----------|------:|\n"
        f"| Aitchison d(x, y) | {d_aitchison:.4f} |\n"
        f"| Euclidean in ILR space | {d_euclidean_ilr:.4f} |\n"
        f"| Euclidean in raw space | {d_euclidean_raw:.4f} |\n\n"
        f"The Aitchison and ILR-Euclidean distances are **identical** "
        f"(the ILR transform is isometric). The raw Euclidean distance "
        f"is different and does not respect the simplex geometry."
    )
    return


# --- Scale Invariance ---


@app.cell
def _(mo):
    mo.md(
        """
        ### Scale Invariance of Aitchison Distance

        A key property: Aitchison distance is **invariant to scaling**.
        Multiplying all components by a constant (changing total mass)
        does not change the distance. This is essential for compositional
        data where the total is arbitrary.
        """
    )
    return


@app.cell
def _(aitchison_distance, mo, np):
    _x = np.array([0.5, 0.3, 0.2])
    _y = np.array([0.1, 0.6, 0.3])

    _rows = ["| Scale factor | Aitchison d | Euclidean d |", "|---:|---:|---:|"]
    for _s in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        _da = aitchison_distance(_x * _s, _y * _s)
        _de = float(np.linalg.norm(_x * _s - _y * _s))
        _rows.append(f"| {_s:g} | {_da:.4f} | {_de:.4f} |")

    mo.md(
        "Scaling both compositions by the same factor:\n\n"
        + "\n".join(_rows)
        + "\n\nAitchison distance is constant; Euclidean scales linearly."
    )
    return


# --- Zero Handling ---


@app.cell
def _(mo):
    mo.md(
        """
        ## 6. Handling Zeros

        Log-ratio transforms require strictly positive data. Real compositional
        data often has zeros (species below detection limit). Multiplicative
        replacement (Martin-Fernandez et al. 2003) substitutes a small value
        for zeros while preserving row sums.
        """
    )
    return


@app.cell
def _(mo, multiplicative_replacement, np):
    _X = np.array([
        [0.50, 0.30, 0.0, 0.20],
        [0.0, 0.40, 0.35, 0.25],
        [0.10, 0.0, 0.0, 0.90],
    ])
    _X_rep = multiplicative_replacement(_X)

    _fmt_row = lambda r: "[" + ", ".join(f"{v:.4f}" for v in r) + "]"

    _lines = ["| Row | Original | Replaced | Sum preserved? |", "|---:|---|---|---:|"]
    for _i in range(3):
        _ok = abs(_X[_i].sum() - _X_rep[_i].sum()) < 1e-10
        _lines.append(
            f"| {_i} | {_fmt_row(_X[_i])} | {_fmt_row(_X_rep[_i])} | {_ok} |"
        )

    mo.md(
        "**Multiplicative replacement** in action:\n\n"
        + "\n".join(_lines)
        + "\n\nZeros are replaced with small values; non-zero entries are "
        "adjusted so row sums are preserved exactly."
    )
    return


# --- Summary ---


@app.cell
def _(mo):
    mo.md(
        """
        ## Summary

        | Function | Purpose |
        |----------|---------|
        | `ilr_scree(X)` | **Rank estimation in Aitchison geometry** — the primary use case; compare with standard PCA scree to detect closure artifacts |
        | `ilr_pca_init(X, p)` | Deterministic initialization for `pmf()` from ILR-PCA singular vectors; a convenience, not a substitute for multi-start |
        | `closure(X)` | Normalize rows to sum to 1 |
        | `clr_transform(X)` | Centered log-ratio transform (D -> D, rows sum to 0) |
        | `ilr_transform(X)` | Isometric log-ratio transform (D -> D-1, isometric) |
        | `ilr_inverse(X_ilr)` | Back-transform from ILR to simplex |
        | `aitchison_distance(X, Y)` | Scale-invariant simplex distance |
        | `multiplicative_replacement(X)` | Handle zeros for log-ratio transforms |

        **When to use `ilr_scree`:**
        - Data is in mass fractions, percentages, or has been normalized
        - You suspect closure artifacts are distorting factor selection
        - Standard PCA scree gives an ambiguous or implausible rank estimate

        **When standard PMF is fine:**
        - Data is in absolute concentrations with meaningful totals
        - No normalization or closure has been applied
        """
    )
    return


if __name__ == "__main__":
    app.run()
