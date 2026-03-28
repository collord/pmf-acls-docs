# Algorithms: When to Use Which

The `pmf()` entry point provides access to five complementary solvers. Each addresses a different use case—some prioritize speed, others prioritize rigor or specific uncertainty quantification. This guide explains the tradeoffs so you can choose the right tool.

## ACLS (Alternating Constrained Least Squares) — **Use this by default**

**What it does:** Alternates between solving weighted least-squares problems for $F$ and $G$. Each factor element is updated via a $k \times k$ normal equation. Non-negativity is enforced via clamping (setting negative values to a small $\epsilon$).

*Note: ACLS is terminology internal to this package. The broader NMF literature refers to this approach as ANLS (Alternating Non-negative Least Squares) or ALS with non-negativity constraints. Searching for "ACLS" in the research literature will yield few results outside this package.*

**Why it's the default:**
- Fast (scales well to large datasets)
- Robust (converges reliably from random initialization)
- Supports FPEAK rotation analysis (unique among the point-estimate solvers)
- The clamped non-negativity is approximate but works well in practice

**When to choose it:** Most of the time. For routine source apportionment, ACLS is the right starting point. It converges in seconds, handles large datasets, and provides all EPA PMF diagnostics (DISP, bootstrap, Q/Qexp).

**Key parameters:**
- `fpeak`: Rotation penalty (positive pushes toward peaked profiles, negative toward diffuse). Use `fpeak_sweep()` to explore rotational ambiguity.
- `lambda_W`, `lambda_H`: Tikhonov regularization (rarely needed; defaults work for most problems).

```python
from pmf_acls import pmf, fpeak_sweep

# Quick run
result = pmf(X, sigma, p=3, algorithm="acls")

# Explore rotational ambiguity
sweep = fpeak_sweep(X, sigma, p=3, fpeak_values=np.linspace(-2, 2, 9), verbose=True)
```

---

## LS-PMF (Least Squares PMF) — **Use when monotone decrease matters**

**What it does:** Weighted multiplicative update rules that monotonically decrease the objective $Q$ at each iteration. This package names this algorithm "LS-PMF"; the literature standard name is "LS-NMF" (Wang et al. 2006), and it is also used by ESAT (EPA's open-source successor to PMF 5.0).

**Why it exists:**
- **Monotone decrease guarantee:** Each iteration provably reduces (or leaves unchanged) the objective $Q$. This ensures descent-based progress and prevents oscillation, but does not guarantee escape from saddle points—only convergence to a stationary point. The NMF landscape is non-convex with many local minima and saddle points, so monotone decrease is about descent direction, not solution quality.
- **ESAT compatibility:** If you're comparing against ESAT results or migrating from ESAT, using the same algorithm simplifies interpretation.
- **Conservative:** Slower per-iteration but sometimes finds solutions that ACLS's aggressive normal equations miss, especially in ill-conditioned problems.

**When to choose it:** When you need to verify that the algorithm is making progress toward a local minimum, or when you're validating results against ESAT. Not recommended for routine use (slower, no FPEAK support, convergence is slower).

**Limitation:** Does not support FPEAK rotation analysis. If rotational ambiguity is a concern, use ACLS with `fpeak_sweep()` instead.

```python
result = pmf(X, sigma, p=3, algorithm="ls-nmf", max_iter=5000)

# LS-PMF converges more slowly; you may need higher max_iter
```

---

## Newton — **Use when you want the most rigorous point estimate**

**What it does:** Gauss-Newton solver that uses second-derivative information to solve for factor updates. Paatero's PMF2 used Newton-like methods; EPA PMF 5.0 uses the proprietary ME-2 (Multilinear Engine v2), whose exact algorithms are not published.

**Why it exists:**
- **Second-derivative step control:** Each iteration uses approximate Hessian information for step-size determination.
- **Validation studies:** When benchmarking against published PMF2 results or attempting to understand ME-2 behavior, Newton-based solvers provide a reference point (though ME-2's exact approach is undocumented).
- **Conceptual precedent:** Gauss-Newton methods are well-studied and theoretically grounded, but this does not guarantee better results than ACLS on practical problems.

**When to choose it:** For validation studies comparing against PMF2/ME-2, or when you need to publish results and want the most defensible solver. Not recommended for routine use (slow, poor scaling with size, still doesn't provide uncertainty bands).

**Limitation:** Scales poorly. For a $100 \times 30 \times 5$ problem (100 species, 30 samples, 5 factors), Newton builds a $650 \times 650$ system each iteration. It is impractical for large datasets.

```python
result = pmf(X, sigma, p=3, algorithm="newton", mode="regularized", max_iter=100)
```

---

## Bayesian (Gibbs Sampling) — **Use when you need posterior uncertainty or automatic factor count**

**What it does:** Full Bayesian inference via Gibbs sampling. Draws samples from the posterior distribution $P(F, G | X, \sigma)$ using exponential priors (non-negativity) and a Gaussian likelihood. See {doc}`bayesian` for detailed explanation.

**Why it's valuable:**
- **Posterior uncertainty:** Returns credible intervals on all factor elements, capturing joint uncertainty in $(F, G)$ that point-estimate methods cannot quantify.
- **ARD factor pruning:** Optional automatic relevance determination learns per-factor precision, pruning unnecessary factors based on data evidence.
- **Factor count posterior:** With ARD, get a probability distribution over factor count, not just a single point estimate.
- **Warm-starting from ACLS:** Can augment an ACLS solution with Bayesian uncertainty without re-optimizing from scratch.

**When to choose it:**
- You need credible intervals on profiles or contributions.
- You want data-driven factor count determination (ARD posterior).
- Your factors are overlapping and you need to characterize the joint uncertainty coupling.
- You have extra computational budget (Bayesian inference is slower).

**Two modes:**
- **`warm_start=True` (default):** Augments ACLS with posterior uncertainty. Fast, gives credible intervals around the ACLS solution.
- **`warm_start=False, ard=True`:** Full Bayesian factor determination. Slower, but factor count is a data-driven posterior, not analyst judgment.

See {doc}`bayesian` for interpretation of convergence diagnostics (Geweke z-score, ESS, label-switch gap).

```python
# Bayesian uncertainty on ACLS solution
result = pmf(X, sigma, p=3, algorithm="bayes", warm_start=True, n_samples=2000)
print(f"Profile uncertainty: {result.F_std}")

# Full Bayesian with ARD factor count
result = pmf(X, sigma, p=5, algorithm="bayes", warm_start=False, ard=True,
             n_samples=2000, n_burnin=1000)
print(f"Factor count posterior: {result.factor_count_posterior}")
```

---

## LDA (Latent Dirichlet Allocation variant)

**What it does:** Bayesian inference with Dirichlet-distributed factor profiles. Each profile is a probability distribution over species (rows sum to 1), similar to LDA in text mining.

**When to choose it:** When you want profiles normalized as probability distributions and need Bayesian posterior inference. Rarely used; included for completeness and for researchers familiar with LDA.

```python
from pmf_acls import pmf_lda

result = pmf_lda(X, sigma, p=3, n_samples=1000, alpha=1.0)
```

---

## Decision Guide: Choosing an Algorithm

| Your situation | Recommended algorithm | Why |
|---|---|---|
| Routine source apportionment, < 1000 samples | ACLS | Fast, robust, supports FPEAK diagnostics |
| Large dataset (>5000 samples) or real-time application | ACLS | Only practical choice for speed |
| Need rotational uncertainty (DISP, FPEAK) | ACLS | Only solver with FPEAK; use with `fpeak_sweep()` |
| Need posterior credible intervals | Bayesian (warm_start=True) | Adds UQ layer without re-optimizing |
| Need automated factor count determination | Bayesian with ARD | Data-driven factor count posterior |
| Validating against EPA PMF 5.0 / ESAT | Newton or LS-PMF | ESAT uses LS-PMF; Newton mimics PMF2 |
| Publishing and want most defensible solver | Newton | Most mathematically rigorous |
| Small dataset (<30 samples) and want rigor | Newton | Poor scaling doesn't matter; precision matters |

---

## References

- Langville, A. N., Meyer, C. D., Albright, R., Cox, J., & Duling, D. (2014). Algorithms, initializations, and convergence for the nonnegative matrix factorization. *arXiv preprint arXiv:1407.7299*.
- Wang, G., Kossenkov, A.V., Bhatt, N.N., & Ochs, M.F. (2006). LS-NMF: A modified non-negative matrix factorization algorithm utilizing uncertainty estimates. *BMC Bioinformatics*, 7, 175.
- Schmidt, M. N., Winther, O., & Hansen, L. K. (2009). Bayesian non-negative matrix factorization. *International Conference on Independent Component Analysis and Signal Separation*, 540–547.
