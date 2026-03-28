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
        # Shrinkage & Truncation Bias: ACLS vs Bayesian NMF

        ACLS systematically **overestimates dominant variables** in factor profiles
        relative to Bayesian NMF. This notebook isolates three reinforcing mechanisms:

        1. **Non-negativity truncation bias** — ACLS clips near-zero least-squares
           solutions to zero. The lost mass migrates to the dominant entries.
        2. **Exponential-prior shrinkage** — Bayesian NMF's exponential prior acts
           as L1 regularisation, gently pulling large values toward zero.
        3. **Posterior mean vs mode** — The Gibbs-sampled posterior mean is a better
           point estimate than the constrained-least-squares mode for skewed
           (truncated) distributions.

        We explore how the bias depends on SNR, profile sparsity, and prior strength.

        > **Note**: `pmf_bayes` returns an ACLS point estimate as `result.F` by
        > default (for sharp, well-separated profiles). This demo uses
        > `result.F_posterior_mean` to compare the *Bayesian posterior mean*
        > against ACLS — the quantity where shrinkage effects are visible.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import linear_sum_assignment

    from pmf_acls import pmf, pmf_bayes

    return linear_sum_assignment, mo, np, plt, pmf, pmf_bayes


@app.cell
def _(linear_sum_assignment, np):
    # ── Utilities ──────────────────────────────────────────────────────────

    def make_sparse_synthetic(rng, m=25, n=80, p=4, sparsity=0.3, noise_frac=0.10):
        """
        Generate a p-factor problem with tuneable sparsity.

        Each factor profile (F column) loads on `int(sparsity * m)` randomly
        chosen variables (exponential magnitudes), with a small baseline for the
        rest.  Two factors are "dominant" (high G scale), the rest moderate/minor.
        """
        F = np.full((p, m), 0.005)
        n_load = max(2, int(round(sparsity * m)))
        for k in range(p):
            vars_k = rng.choice(m, size=n_load, replace=False)
            F[k, vars_k] += rng.exponential(0.5, size=n_load)

        for k in range(p):
            F[k, :] /= F[k, :].sum()

        # Dominant-to-minor gradient in G scales
        scales = np.logspace(np.log10(50), np.log10(5), p)
        G = np.zeros((n, p))
        for k in range(p):
            G[:, k] = rng.exponential(scales[k], size=n)

        X_true = G @ F
        sigma = noise_frac * np.maximum(X_true, 0.01) + 0.005
        noise = rng.normal(0, sigma)
        X = np.maximum(X_true + noise, 0.0)
        return X, sigma, F, G

    def match_factors(F_est, F_true):
        """Hungarian matching by absolute correlation."""
        p_est, p_true = F_est.shape[0], F_true.shape[0]
        corr = np.zeros((p_est, p_true))
        for i in range(p_est):
            for j in range(p_true):
                c = np.corrcoef(F_est[i, :], F_true[j, :])[0, 1]
                corr[i, j] = c if np.isfinite(c) else 0.0
        row_ind, col_ind = linear_sum_assignment(-np.abs(corr))
        return row_ind, col_ind, corr

    def normalize_l1(F):
        """L1-normalise each row of F to sum to 1."""
        return F / F.sum(axis=1, keepdims=True)

    def compute_bias(F_est, F_true, row_ind, col_ind, top_k=3):
        """
        Signed relative bias on the `top_k` dominant variables per factor.

        Returns (dominant_biases, minor_biases) — arrays of per-variable biases
        pooled across all matched factors.
        """
        dom, minor = [], []
        for ke, kt in zip(row_ind, col_ind):
            g_true = F_true[kt, :]
            g_est = F_est[ke, :]
            order = np.argsort(-g_true)
            mask = g_true > 0.01  # only non-trivial vars
            top = order[:top_k]
            rest = order[top_k:]
            for v in top:
                if mask[v]:
                    dom.append((g_est[v] - g_true[v]) / g_true[v])
            for v in rest:
                if mask[v]:
                    minor.append((g_est[v] - g_true[v]) / g_true[v])
        return np.array(dom), np.array(minor)

    return make_sparse_synthetic, match_factors, normalize_l1, compute_bias


# ── Section 1: Single-problem comparison ─────────────────────────────────


@app.cell
def _(mo):
    mo.md("## 1. Profile Comparison at a Single SNR")
    return


@app.cell
def _(mo):
    noise_slider = mo.ui.slider(
        0.02, 0.30, step=0.02, value=0.10, label="Noise fraction"
    )
    m_slider = mo.ui.slider(10, 40, step=5, value=25, label="Variables (m)")
    n_slider = mo.ui.slider(40, 200, step=20, value=80, label="Observations (n)")
    p_slider = mo.ui.slider(3, 6, step=1, value=4, label="Factors (p)")
    sparsity_slider = mo.ui.slider(
        0.15, 0.70, step=0.05, value=0.30, label="Sparsity (frac vars per factor)"
    )
    seed_number = mo.ui.number(value=2026, label="Random seed")
    run_btn = mo.ui.run_button(label="Run Comparison")
    _controls = mo.hstack(
        [noise_slider, m_slider, n_slider, p_slider, sparsity_slider, seed_number, run_btn],
        justify="start", gap=1.0,
    )
    _controls
    return noise_slider, m_slider, n_slider, p_slider, sparsity_slider, seed_number, run_btn


@app.cell
def _(
    make_sparse_synthetic,
    match_factors,
    mo,
    noise_slider,
    m_slider,
    n_slider,
    normalize_l1,
    np,
    p_slider,
    pmf,
    pmf_bayes,
    run_btn,
    seed_number,
    sparsity_slider,
):
    mo.stop(not run_btn.value, mo.md("*Click **Run Comparison** above*"))

    _rng = np.random.default_rng(seed_number.value)
    X, sigma, F_true, G_true = make_sparse_synthetic(
        _rng,
        m=m_slider.value,
        n=n_slider.value,
        p=p_slider.value,
        sparsity=sparsity_slider.value,
        noise_frac=noise_slider.value,
    )
    _snr = np.sum((F_true @ G_true) ** 2) / np.sum(sigma ** 2)

    acls_res = pmf(X, sigma, p=p_slider.value, algorithm="acls")
    bayes_res = pmf_bayes(
        X, sigma, p=p_slider.value,
        n_samples=2000, n_burnin=1000,
        random_seed=seed_number.value, verbose=False,
    )

    # Normalise both to L1 for fair comparison
    G_true_n = normalize_l1(F_true)
    F_acls_n = normalize_l1(acls_res.F)
    F_bayes_n = normalize_l1(bayes_res.F_posterior_mean)

    acls_row, acls_col, acls_corr = match_factors(F_acls_n, G_true_n)
    bayes_row, bayes_col, bayes_corr = match_factors(F_bayes_n, G_true_n)

    mo.md(
        f"""
        **Data**: {X.shape[0]} vars x {X.shape[1]} obs, {p_slider.value} factors
        &nbsp;|&nbsp; **SNR = {_snr:.0f} ({10*np.log10(_snr):.1f} dB)**

        | Solver | Q | Explained var | chi2 |
        |--------|---|---------------|------|
        | ACLS | {acls_res.Q:.2e} | {acls_res.explained_variance:.2%} | {acls_res.chi2:.3f} |
        | Bayesian | {bayes_res.Q:.2e} | {bayes_res.explained_variance:.2%} | {bayes_res.chi2:.3f} |
        """
    )
    return (
        X, sigma, F_true, G_true,
        acls_res, bayes_res,
        G_true_n, F_acls_n, F_bayes_n,
        acls_row, acls_col, acls_corr,
        bayes_row, bayes_col, bayes_corr,
    )


@app.cell
def _(
    G_true_n,
    F_acls_n,
    F_bayes_n,
    acls_col,
    acls_corr,
    acls_row,
    bayes_col,
    bayes_corr,
    bayes_row,
    np,
    plt,
):
    # Side-by-side profile bar charts: ACLS (left) vs Bayesian (right)
    _p = len(acls_row)
    _m = G_true_n.shape[0]
    _x = np.arange(_m)
    _var_labels = [f"V{i+1}" for i in range(_m)]

    _fig, _axes = plt.subplots(
        _p, 2, figsize=(max(8, 0.6 * _m), 3 * _p),
        constrained_layout=True, squeeze=False,
    )
    _fig.suptitle("Factor Profiles — ACLS vs Bayesian vs Truth", fontsize=13)

    for idx in range(_p):
        # ACLS
        ka, kt_a = acls_row[idx], acls_col[idx]
        ax_a = _axes[idx, 0]
        ax_a.bar(_x, F_acls_n[:, ka], 0.6, color="none", edgecolor="C3", linewidth=1.2)
        ax_a.hlines(G_true_n[:, kt_a], _x - 0.3, _x + 0.3, colors="0.2", linewidth=2.5)
        ra = abs(acls_corr[ka, kt_a])
        ax_a.set_title(f"Factor {kt_a}  (ACLS, r={ra:.3f})", fontsize=9)
        ax_a.set_xticks(_x)
        ax_a.set_xticklabels(_var_labels, rotation=90, fontsize=6)

        # Bayesian
        kb, kt_b = bayes_row[idx], bayes_col[idx]
        ax_b = _axes[idx, 1]
        ax_b.bar(_x, F_bayes_n[:, kb], 0.6, color="none", edgecolor="C0", linewidth=1.2)
        ax_b.hlines(G_true_n[:, kt_b], _x - 0.3, _x + 0.3, colors="0.2", linewidth=2.5)
        rb = abs(bayes_corr[kb, kt_b])
        ax_b.set_title(f"Factor {kt_b}  (Bayesian, r={rb:.3f})", fontsize=9)
        ax_b.set_xticks(_x)
        ax_b.set_xticklabels(_var_labels, rotation=90, fontsize=6)

    _axes[0, 0].legend(
        ["ACLS", "Truth"], fontsize=7, loc="upper right"
    )
    _axes[0, 1].legend(
        ["Bayesian", "Truth"], fontsize=7, loc="upper right"
    )
    _fig
    return


# ── Section 2: Shrinkage scatter ─────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. Shrinkage Scatter

        Each point is one variable in one matched factor:
        **true profile value** (x) vs **estimated profile value** (y).

        A perfect solver would land on the 1:1 diagonal.
        ACLS bends **above** the line for dominant variables (over-estimation)
        and sits **at zero** for minor ones (truncation). Bayesian hugs the diagonal
        thanks to exponential-prior shrinkage.
        """
    )
    return


@app.cell
def _(
    G_true_n,
    F_acls_n,
    F_bayes_n,
    acls_col,
    acls_row,
    bayes_col,
    bayes_row,
    np,
    plt,
):
    # Pool all (true, est) pairs across matched factors
    _true_a, _est_a = [], []
    for ke, kt in zip(acls_row, acls_col):
        _true_a.append(G_true_n[:, kt])
        _est_a.append(F_acls_n[:, ke])
    _true_a = np.concatenate(_true_a)
    _est_a = np.concatenate(_est_a)

    _true_b, _est_b = [], []
    for ke, kt in zip(bayes_row, bayes_col):
        _true_b.append(G_true_n[:, kt])
        _est_b.append(F_bayes_n[:, ke])
    _true_b = np.concatenate(_true_b)
    _est_b = np.concatenate(_est_b)

    # Only plot variables with non-trivial true value
    _mask_a = _true_a > 0.005
    _mask_b = _true_b > 0.005

    _fig, _ax = plt.subplots(figsize=(6, 6))
    _ax.scatter(
        _true_a[_mask_a], _est_a[_mask_a],
        s=18, alpha=0.5, color="C3", label="ACLS", zorder=3,
    )
    _ax.scatter(
        _true_b[_mask_b], _est_b[_mask_b],
        s=18, alpha=0.5, color="C0", label="Bayesian", zorder=3,
    )
    _lim = max(_true_a[_mask_a].max(), _true_b[_mask_b].max(),
               _est_a[_mask_a].max(), _est_b[_mask_b].max()) * 1.05
    _ax.plot([0, _lim], [0, _lim], "k--", lw=1, alpha=0.5, label="1:1")
    _ax.set_xlim(0, _lim)
    _ax.set_ylim(0, _lim)
    _ax.set_xlabel("True profile value (L1-normalised)")
    _ax.set_ylabel("Estimated profile value")
    _ax.set_title("Shrinkage Plot")
    _ax.legend(fontsize=9)
    _ax.set_aspect("equal")

    # Annotate mean signed error
    _mse_a = np.mean(_est_a[_mask_a] - _true_a[_mask_a])
    _mse_b = np.mean(_est_b[_mask_b] - _true_b[_mask_b])
    _ax.text(
        0.03, 0.95,
        f"Mean signed error\nACLS: {_mse_a:+.4f}\nBayesian: {_mse_b:+.4f}",
        transform=_ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )
    _fig
    return


# ── Section 3: Bias vs SNR sweep ─────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. Dominant-Variable Bias vs Noise Level

        We sweep the noise fraction and measure the **mean signed relative bias**
        on the top-3 variables per factor.  Positive bias = overestimation.

        ACLS bias should grow with noise (more truncation), while Bayesian
        stays near zero.
        """
    )
    return


@app.cell
def _(mo):
    sweep_reps_slider = mo.ui.slider(1, 5, step=1, value=3, label="Reps per level")
    sweep_btn = mo.ui.run_button(label="Run SNR Sweep")
    mo.hstack([sweep_reps_slider, sweep_btn], justify="start", gap=1.5)
    return sweep_reps_slider, sweep_btn


@app.cell
def _(
    compute_bias,
    make_sparse_synthetic,
    match_factors,
    mo,
    m_slider,
    n_slider,
    normalize_l1,
    np,
    p_slider,
    plt,
    pmf,
    pmf_bayes,
    sparsity_slider,
    sweep_btn,
    sweep_reps_slider,
):
    mo.stop(not sweep_btn.value, mo.md("*Click **Run SNR Sweep** above*"))

    _noise_fracs = np.linspace(0.03, 0.30, 8)
    _reps = sweep_reps_slider.value
    _m, _n, _p = m_slider.value, n_slider.value, p_slider.value
    _sp = sparsity_slider.value

    _acls_dom = np.zeros((len(_noise_fracs), _reps))
    _bayes_dom = np.zeros_like(_acls_dom)
    _acls_min = np.zeros_like(_acls_dom)
    _bayes_min = np.zeros_like(_acls_dom)

    for i, nf in enumerate(_noise_fracs):
        for r in range(_reps):
            rng = np.random.default_rng(1000 + r)
            X_, s_, Gt_, Ft_ = make_sparse_synthetic(
                rng, m=_m, n=_n, p=_p, sparsity=_sp, noise_frac=nf,
            )
            Gt_n = normalize_l1(Gt_)

            a_res = pmf(X_, s_, p=_p, algorithm="acls")
            b_res = pmf_bayes(
                X_, s_, p=_p,
                n_samples=500, n_burnin=500,
                random_seed=1000 + r, verbose=False,
            )
            Ga_n = normalize_l1(a_res.F)
            Gb_n = normalize_l1(b_res.F_posterior_mean)

            ar, ac, _ = match_factors(Ga_n, Gt_n)
            br, bc, _ = match_factors(Gb_n, Gt_n)

            ad, am = compute_bias(Ga_n, Gt_n, ar, ac)
            bd, bm = compute_bias(Gb_n, Gt_n, br, bc)

            _acls_dom[i, r] = ad.mean() if len(ad) else 0
            _bayes_dom[i, r] = bd.mean() if len(bd) else 0
            _acls_min[i, r] = am.mean() if len(am) else 0
            _bayes_min[i, r] = bm.mean() if len(bm) else 0

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    _fig.suptitle("Signed Relative Bias vs Noise Fraction", fontsize=13)

    for _ax, _ad, _bd, title in [
        (_ax1, _acls_dom, _bayes_dom, "Dominant variables (top-3)"),
        (_ax2, _acls_min, _bayes_min, "Minor variables"),
    ]:
        _mu_a = _ad.mean(axis=1)
        _mu_b = _bd.mean(axis=1)
        _sd_a = _ad.std(axis=1)
        _sd_b = _bd.std(axis=1)

        _ax.plot(_noise_fracs, _mu_a, "o-", color="C3", label="ACLS")
        _ax.fill_between(_noise_fracs, _mu_a - _sd_a, _mu_a + _sd_a, color="C3", alpha=0.15)
        _ax.plot(_noise_fracs, _mu_b, "s-", color="C0", label="Bayesian")
        _ax.fill_between(_noise_fracs, _mu_b - _sd_b, _mu_b + _sd_b, color="C0", alpha=0.15)
        _ax.axhline(0, color="k", ls="--", lw=0.8)
        _ax.set_xlabel("Noise fraction")
        _ax.set_ylabel("Mean signed relative bias")
        _ax.set_title(title)
        _ax.legend(fontsize=9)

    _fig
    return


# ── Section 4: Role of the prior ─────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        """
        ## 4. Role of the Exponential Prior

        The Bayesian solver uses Exp(lambda) priors on F and G elements.
        Larger lambda = stronger shrinkage toward zero.

        With `learn_hyperparams=False` we fix lambda and sweep its value.
        At very small lambda (weak prior) the Bayesian solver should approach
        ACLS-like behaviour. At moderate lambda, shrinkage optimally counteracts
        the truncation bias. At very large lambda, over-shrinkage pulls
        everything toward zero.
        """
    )
    return


@app.cell
def _(mo):
    prior_btn = mo.ui.run_button(label="Run Prior Sweep")
    prior_btn
    return (prior_btn,)


@app.cell
def _(
    X,
    sigma,
    acls_res,
    compute_bias,
    G_true_n,
    match_factors,
    mo,
    normalize_l1,
    np,
    p_slider,
    plt,
    pmf_bayes,
    prior_btn,
    acls_row,
    acls_col,
):
    mo.stop(not prior_btn.value, mo.md("*Click **Run Prior Sweep** above*"))

    _lambdas = np.logspace(-2, 1.5, 14)

    _acls_dom_bias, _ = compute_bias(
        normalize_l1(acls_res.F), G_true_n, acls_row, acls_col,
    )
    _acls_ref = _acls_dom_bias.mean()

    _bayes_biases = np.zeros(len(_lambdas))
    _bayes_corrs = np.zeros(len(_lambdas))

    for i, lam in enumerate(_lambdas):
        _res = pmf_bayes(
            X, sigma, p=p_slider.value,
            n_samples=800, n_burnin=500,
            lambda_F=lam, lambda_G=lam,
            learn_hyperparams=False,
            random_seed=42, verbose=False,
        )
        _Gn = normalize_l1(_res.F_posterior_mean)
        _r, _c, _corr = match_factors(_Gn, G_true_n)
        _bd, _ = compute_bias(_Gn, G_true_n, _r, _c)
        _bayes_biases[i] = _bd.mean() if len(_bd) else 0
        # Average absolute correlation across matched factors
        _bayes_corrs[i] = np.mean([abs(_corr[_r[j], _c[j]]) for j in range(len(_r))])

    _fig, _ax1 = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    _ax1.semilogx(_lambdas, _bayes_biases, "s-", color="C0", label="Bayesian bias")
    _ax1.axhline(_acls_ref, color="C3", ls="--", lw=1.5, label=f"ACLS bias ({_acls_ref:+.2f})")
    _ax1.axhline(0, color="k", ls=":", lw=0.8)
    _ax1.set_xlabel("lambda (exponential prior rate)")
    _ax1.set_ylabel("Mean signed relative bias (dominant vars)", color="C0")
    _ax1.legend(fontsize=9, loc="upper left")
    _ax1.set_title("Dominant-Variable Bias vs Prior Strength")

    _ax2 = _ax1.twinx()
    _ax2.semilogx(_lambdas, _bayes_corrs, "o--", color="C2", alpha=0.6, label="Mean |r|")
    _ax2.set_ylabel("Mean factor correlation with truth", color="C2")
    _ax2.legend(fontsize=9, loc="upper right")

    _fig
    return


# ── Section 5: Sparsity effect ───────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        """
        ## 5. Sparsity Effect

        Sparser true profiles (fewer non-zero variables per factor) mean more
        variables are near zero — more truncation at the non-negativity boundary,
        and more mass redistributed to the dominant entries.

        The ACLS overestimation should **increase** as sparsity decreases
        (i.e. fewer variables load on each factor), while Bayesian remains stable.
        """
    )
    return


@app.cell
def _(mo):
    sparsity_reps_slider = mo.ui.slider(1, 5, step=1, value=3, label="Reps per level")
    sparsity_btn = mo.ui.run_button(label="Run Sparsity Sweep")
    mo.hstack([sparsity_reps_slider, sparsity_btn], justify="start", gap=1.5)
    return sparsity_reps_slider, sparsity_btn


@app.cell
def _(
    compute_bias,
    make_sparse_synthetic,
    match_factors,
    mo,
    m_slider,
    n_slider,
    noise_slider,
    normalize_l1,
    np,
    p_slider,
    plt,
    pmf,
    pmf_bayes,
    sparsity_btn,
    sparsity_reps_slider,
):
    mo.stop(not sparsity_btn.value, mo.md("*Click **Run Sparsity Sweep** above*"))

    _sparsities = np.linspace(0.15, 0.70, 8)
    _reps = sparsity_reps_slider.value
    _m, _n, _p = m_slider.value, n_slider.value, p_slider.value
    _nf = noise_slider.value

    _acls_dom = np.zeros((len(_sparsities), _reps))
    _bayes_dom = np.zeros_like(_acls_dom)

    for i, sp in enumerate(_sparsities):
        for r in range(_reps):
            rng = np.random.default_rng(2000 + r)
            X_, s_, Gt_, Ft_ = make_sparse_synthetic(
                rng, m=_m, n=_n, p=_p, sparsity=sp, noise_frac=_nf,
            )
            Gt_n = normalize_l1(Gt_)

            a_res = pmf(X_, s_, p=_p, algorithm="acls")
            b_res = pmf_bayes(
                X_, s_, p=_p,
                n_samples=500, n_burnin=500,
                random_seed=2000 + r, verbose=False,
            )
            Ga_n = normalize_l1(a_res.F)
            Gb_n = normalize_l1(b_res.F_posterior_mean)

            ar, ac, _ = match_factors(Ga_n, Gt_n)
            br, bc, _ = match_factors(Gb_n, Gt_n)

            ad, _ = compute_bias(Ga_n, Gt_n, ar, ac)
            bd, _ = compute_bias(Gb_n, Gt_n, br, bc)

            _acls_dom[i, r] = ad.mean() if len(ad) else 0
            _bayes_dom[i, r] = bd.mean() if len(bd) else 0

    _fig, _ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

    _mu_a = _acls_dom.mean(axis=1)
    _mu_b = _bayes_dom.mean(axis=1)
    _sd_a = _acls_dom.std(axis=1)
    _sd_b = _bayes_dom.std(axis=1)

    _ax.plot(_sparsities, _mu_a, "o-", color="C3", label="ACLS")
    _ax.fill_between(_sparsities, _mu_a - _sd_a, _mu_a + _sd_a, color="C3", alpha=0.15)
    _ax.plot(_sparsities, _mu_b, "s-", color="C0", label="Bayesian")
    _ax.fill_between(_sparsities, _mu_b - _sd_b, _mu_b + _sd_b, color="C0", alpha=0.15)
    _ax.axhline(0, color="k", ls="--", lw=0.8)
    _ax.set_xlabel("Sparsity (fraction of variables loading per factor)")
    _ax.set_ylabel("Mean signed relative bias (dominant vars)")
    _ax.set_title("Dominant-Variable Bias vs Profile Sparsity")
    _ax.legend(fontsize=9)

    _fig
    return


# ── Takeaways ─────────────────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        """
        ## Takeaways

        1. **ACLS overestimates dominant variables** because non-negativity truncation
           at zero redistributes the lost mass to the remaining large coefficients.
           This is a systematic bias, not random noise.

        2. **Bayesian NMF's exponential prior** provides L1-like shrinkage that
           counteracts the redistribution. The posterior mean sits closer to the
           truth across the full range of profile values.

        3. The ACLS bias **worsens with noise** (more variables are pushed to the
           truncation boundary) and with **sparser profiles** (more true zeros).

        4. The prior strength (lambda) has an optimal range: too weak and the Bayesian
           solver behaves like ACLS; too strong and it over-shrinks everything toward
           zero. The default `learn_hyperparams=True` adapts lambda to the data.

        5. **Practical implication**: When interpreting PMF profiles from ACLS, be
           cautious about the absolute magnitude of dominant species contributions.
           Bootstrap CIs from ACLS will not capture this systematic bias. The
           Bayesian posterior mean (`result.F_posterior_mean`) provides a less
           biased alternative. For profile interpretation, `result.F` returns the
           ACLS point estimate (sharp, well-separated); for shrinkage-aware
           profiles, use `result.F_posterior_mean` with `result.F_std` for
           uncertainty.
        """
    )
    return


if __name__ == "__main__":
    app.run()
