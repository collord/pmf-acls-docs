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
        # Data Preparation for PMF

        Real environmental data is messy: instruments have detection limits,
        samples go missing, and measurement uncertainty varies across species
        and concentrations.  Proper data preparation is **critical** for PMF
        because the objective function

        $$Q = \sum_{i,j} \left(\frac{X_{ij} - [\mathbf{FG}]_{ij}}{\sigma_{ij}}\right)^2$$

        weights every residual by $1/\sigma_{ij}^2$.  If uncertainties are
        wrong — too small for noisy measurements, or missing entirely for
        replaced values — the solver will chase noise in some cells and ignore
        real signal in others.

        This notebook walks through the standard EPA PMF preparation pipeline:

        1. **Missing values** — replace with species medians, inflate $\sigma$
        2. **Below-detection-limit (BDL)** — replace with DL/2, set $\sigma = \tfrac{5}{6}\,\text{DL}$
        3. **Uncertainty estimation** — propagate measurement error so the Q objective is well-scaled
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

    from pmf_acls import prepare_data, check_data_quality, pmf

    return check_data_quality, mo, np, plt, pmf, prepare_data


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. Generate Realistic Messy Data

        We simulate PM2.5 chemical composition data for 12 species across
        100 samples, then inject the two most common real-world problems:

        - **Missing values (NaN)** — 5 % of cells, representing lost samples
          or failed analyses.
        - **Below-detection-limit (BDL) measurements** — 10-20 % per species,
          where the instrument reports a value below its reliable range.
        """
    )
    return


@app.cell
def _(np):
    rng = np.random.default_rng(42)

    species_names = [
        "SO4", "NO3", "NH4", "OC", "EC", "Al", "Si",
        "Ca", "Fe", "Zn", "Pb", "K",
    ]
    m, n = len(species_names), 100  # 12 species, 100 samples

    # Base concentrations (ug/m3) — realistic range for urban PM2.5
    X_base = rng.uniform(1, 15, size=(m, n))

    # --- Inject missing values (5 %) ---
    n_missing = int(0.05 * m * n)
    missing_flat_idx = rng.choice(m * n, size=n_missing, replace=False)
    X_messy = X_base.copy()
    X_messy.flat[missing_flat_idx] = np.nan

    # --- Inject below-detection-limit values (10-20 % per species) ---
    detection_limits = rng.uniform(0.1, 0.6, size=m)
    for i in range(m):
        n_bdl = rng.integers(int(0.1 * n), int(0.2 * n))
        bdl_cols = rng.choice(n, size=n_bdl, replace=False)
        X_messy[i, bdl_cols] = rng.uniform(-0.1, detection_limits[i] / 2, size=n_bdl)

    n_nan = int(np.sum(np.isnan(X_messy)))

    return X_messy, detection_limits, m, n, n_nan, rng, species_names


@app.cell
def _(X_messy, m, mo, n, n_nan):
    mo.md(
        f"""
        **Simulated data:** {m} species x {n} samples

        - Missing values (NaN): **{n_nan}** ({n_nan / (m * n) * 100:.1f} % of cells)
        - Negative / sub-DL values present in every species

        Before we can run PMF, every cell needs a finite, non-negative
        concentration and a positive uncertainty.  The next step handles that.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Data Preparation Pipeline

        `prepare_data()` applies the standard EPA PMF 5.0 guidance:

        | Problem | Replacement | Uncertainty |
        |---------|------------|-------------|
        | **Missing (NaN)** | Species median | 4x the median concentration |
        | **Below DL** | DL / 2 | 5/6 x DL |
        | **Above DL** | Keep as-is | Measurement error (>= 10 % of value) |

        The **5/6 DL** uncertainty for BDL values comes from EPA PMF guidance
        (Norris et al., 2014).  It reflects that we know very little about the
        true value — only that it is somewhere between 0 and DL — so the
        uncertainty should be wide relative to the replacement value of DL/2.
        """
    )
    return


@app.cell
def _(X_messy, detection_limits, prepare_data):
    X_clean, sigma = prepare_data(
        X_messy,
        detection_limit=detection_limits,
        missing_method="median",
        bdl_replacement="half_dl",
        uncertainty_method="auto",
        min_uncertainty_fraction=0.1,
        verbose=True,
    )
    return X_clean, sigma


@app.cell
def _(X_clean, check_data_quality, mo, np, sigma):
    diagnostics = check_data_quality(X_clean, sigma, verbose=False)

    _has_nan = np.any(np.isnan(X_clean))
    _all_pos = np.all(X_clean >= 0)
    _sigma_pos = np.all(sigma > 0)

    mo.md(
        f"""
        ### Data quality after preparation

        | Check | Result |
        |-------|--------|
        | No missing values | {"PASS" if not _has_nan else "FAIL"} |
        | All values non-negative | {"PASS" if _all_pos else "FAIL"} |
        | All uncertainties positive | {"PASS" if _sigma_pos else "FAIL"} |
        | Median signal-to-noise | {diagnostics['signal_to_noise']:.1f} |
        | Fraction with S/N < 3 | {diagnostics['high_uncertainty']:.1%} |

        The data is now ready for PMF.
        """
    )
    return diagnostics,


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Before / After Heatmaps

        The left panel shows the raw data with NaN gaps (red x markers) and
        negative BDL values (dark patches).  The right panel shows the cleaned
        matrix — all cells filled, all values non-negative.
        """
    )
    return


@app.cell
def _(X_clean, X_messy, np, plt, species_names):
    fig_heatmap, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(14, 5))

    # Before
    _vmax = np.nanpercentile(X_messy, 98)
    im0 = ax_before.imshow(
        X_messy, aspect="auto", cmap="viridis", vmin=-0.5, vmax=_vmax,
    )
    ax_before.set_title("Raw data (before preparation)")
    ax_before.set_xlabel("Sample index")
    ax_before.set_ylabel("Species")
    ax_before.set_yticks(range(len(species_names)))
    ax_before.set_yticklabels(species_names, fontsize=8)
    fig_heatmap.colorbar(im0, ax=ax_before, label=r"Concentration ($\mu$g/m$^3$)")

    # Overlay NaN locations
    _nan_mask = np.isnan(X_messy)
    _nan_y, _nan_x = np.where(_nan_mask)
    ax_before.scatter(
        _nan_x, _nan_y, marker="x", color="red", s=12, linewidths=0.6, label="NaN",
    )
    ax_before.legend(loc="upper right", fontsize=7)

    # After
    im1 = ax_after.imshow(
        X_clean, aspect="auto", cmap="viridis", vmin=0, vmax=_vmax,
    )
    ax_after.set_title("Cleaned data (after preparation)")
    ax_after.set_xlabel("Sample index")
    ax_after.set_ylabel("Species")
    ax_after.set_yticks(range(len(species_names)))
    ax_after.set_yticklabels(species_names, fontsize=8)
    fig_heatmap.colorbar(im1, ax=ax_after, label=r"Concentration ($\mu$g/m$^3$)")

    fig_heatmap.tight_layout()
    fig_heatmap
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Signal-to-Noise Ratio by Species

        The signal-to-noise ratio $\text{S/N}_i = \text{median}(X_i / \sigma_i)$
        tells us how informative each species is.  EPA guidance classifies
        species as:

        - **Strong** (S/N > 2): full weight in the fit
        - **Weak** (0.5 < S/N < 2): down-weight by increasing $\sigma$ by 3x
        - **Bad** (S/N < 0.5): exclude from analysis

        Species with low S/N contribute mostly noise to the Q objective.
        Down-weighting them prevents the solver from fitting noise.
        """
    )
    return


@app.cell
def _(X_clean, np, plt, sigma, species_names):
    snr_per_species = np.median(X_clean / sigma, axis=1)

    fig_snr, ax_snr = plt.subplots(figsize=(10, 4))
    colors = []
    for s in snr_per_species:
        if s > 2:
            colors.append("#2ecc71")   # strong — green
        elif s > 0.5:
            colors.append("#f39c12")   # weak — amber
        else:
            colors.append("#e74c3c")   # bad — red

    ax_snr.bar(species_names, snr_per_species, color=colors, edgecolor="black", linewidth=0.5)
    ax_snr.axhline(2, color="#2ecc71", linestyle="--", linewidth=1, label="Strong (S/N > 2)")
    ax_snr.axhline(0.5, color="#e74c3c", linestyle="--", linewidth=1, label="Bad (S/N < 0.5)")
    ax_snr.set_ylabel("Median S/N ratio")
    ax_snr.set_title("Signal-to-Noise Ratio by Species (EPA classification)")
    ax_snr.legend(fontsize=8)
    fig_snr.tight_layout()
    fig_snr
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. Uncertainty vs. Concentration

        Good uncertainties should scale with concentration: larger values
        have larger absolute uncertainty, but roughly constant *relative*
        uncertainty.  BDL replacements (clustered at low concentrations)
        should have disproportionately large $\sigma$ — this is how PMF
        "knows" not to trust those cells.

        The dashed lines show constant relative-uncertainty contours at
        10 %, 50 %, and 100 %.  BDL-replaced cells should sit near or
        above the 100 % line.
        """
    )
    return


@app.cell
def _(X_clean, np, plt, sigma):
    fig_uq, ax_uq = plt.subplots(figsize=(8, 6))

    ax_uq.scatter(
        X_clean.ravel(), sigma.ravel(),
        alpha=0.25, s=10, color="#3498db", edgecolors="none",
    )

    # Reference lines for relative uncertainty
    _xref = np.linspace(0.01, X_clean.max(), 200)
    for frac, ls in [(0.1, ":"), (0.5, "--"), (1.0, "-")]:
        ax_uq.plot(
            _xref, frac * _xref, ls, color="grey", linewidth=0.8,
            label=f"{frac:.0%} relative",
        )

    ax_uq.set_xlabel(r"Concentration ($\mu$g/m$^3$)")
    ax_uq.set_ylabel(r"Uncertainty $\sigma$ ($\mu$g/m$^3$)")
    ax_uq.set_title("Uncertainty vs. Concentration")
    ax_uq.legend(fontsize=8)
    ax_uq.set_xlim(left=0)
    ax_uq.set_ylim(bottom=0)
    fig_uq.tight_layout()
    fig_uq
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6. Run PMF on the Prepared Data

        With clean data and well-calibrated uncertainties, PMF can solve
        for the factor profiles (F) and contributions (G).  The Q/Q_exp
        ratio should be close to 1.0 if the uncertainties are correct —
        meaning the model fits the data to within the expected noise level.
        """
    )
    return


@app.cell
def _(X_clean, m, n, pmf, sigma):
    result = pmf(
        X_clean, sigma,
        p=3,
        max_iter=10000,
        conv_tol=1e-6,
        random_seed=42,
        verbose=False,
    )

    _q_exp = m * n  # expected Q under correct model
    return result, _q_exp


@app.cell
def _(X_clean, _q_exp, mo, np, result, sigma):
    _resid = (X_clean - result.F @ result.G) / sigma
    _q_per_species = np.sum(_resid ** 2, axis=1)

    mo.md(
        f"""
        ### PMF Results

        | Metric | Value |
        |--------|-------|
        | Converged | {result.converged} |
        | Iterations | {result.n_iter} |
        | Q (objective) | {result.Q:.1f} |
        | Q / Q_expected | {result.Q / _q_exp:.3f} |
        | Explained variance | {result.explained_variance:.2%} |

        A Q/Q_exp near 1.0 means the model fits the data to within the
        measurement uncertainty — exactly what well-calibrated $\\sigma$ values
        should produce.  Values much greater than 1 suggest the uncertainties
        are too small (or the model needs more factors); values much less than 1
        suggest the uncertainties are too large.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        Data preparation is not a preliminary chore — it is an integral part of
        the PMF analysis.  The choices made here flow directly into the Q
        objective:

        - **Missing values** replaced with medians and inflated $\sigma$ prevent
          fabricated data from dominating the fit.
        - **BDL values** replaced with DL/2 and $\sigma = \frac{5}{6}$DL follow
          EPA guidance, ensuring the solver does not over-fit sub-detection noise.
        - **Uncertainty estimation** that scales with concentration keeps the
          Q weighting balanced across species with very different magnitudes.

        Getting these right is the difference between factors that represent
        real emission sources and factors that represent data artifacts.
        """
    )
    return


if __name__ == "__main__":
    app.run()
