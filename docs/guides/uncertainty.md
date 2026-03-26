# Uncertainty Quantification: Complementary Methods

Point estimates from ACLS, LS-NMF, or Newton solvers are not enough. Different uncertainty quantification methods answer *different questions* about solution quality, and EPA PMF tradition recommends using all three. The Bayesian posterior adds a fourth, complementary layer.

## Bootstrap: Does the data structure support these factors?

**What it does:** Resamples observations (with replacement, using block resampling to preserve temporal/spatial structure if present) and re-runs PMF on each bootstrap sample. Compares the resolved factors across samples.

**What it answers:** If I re-collect my data, would I get the same factors? Does the data structure reliably support the factorization?

**Physical interpretation:** Sampling uncertainty. How stable are the factor profiles and contributions under resampling of the observations?

```python
from pmf_acls import bootstrap_uncertainty

bs = bootstrap_uncertainty(X, sigma, p=3, n_bootstrap=100)
print(f"Profile CIs: {bs.F_ci_low} to {bs.F_ci_high}")
print(f"Contribution CIs: {bs.G_ci_low} to {bs.G_ci_high}")
```

**When to use:** For all analyses. Bootstrap is model-light—it doesn't assume a specific error distribution, just resamples the observed data. It tests whether the data itself supports the solution.

---

## Displacement (DISP): How far can profiles move before Q degrades?

**What it does:** For each element of the profile matrix $F_{ik}$, pins it to progressively different values and re-optimizes all other elements. Measures how far $F_{ik}$ can be displaced before the objective $Q$ increases beyond a threshold (typically ΔQ = 4, 8, or 16).

**What it answers:** Which profile elements are tightly determined by the data? Which are weakly constrained (large uncertainty)? Are there factors that become confused under slight displacement (rotational instability)?

**Physical interpretation:** Local Q-surface geometry. How rugged is the valley around the solution? Wide displacement ranges indicate flexible factors; narrow ranges indicate tight constraints.

```python
from pmf_acls import displacement_test_F

disp = displacement_test_F(X, sigma, F_base, reopt_method='acls')
print(f"Profile uncertainties (dQ=4): {disp.F_std_approx}")
print(f"Factor swaps (rotational instability): {disp.swap_counts}")
```

**Key parameter:** `reopt_method` controls what gets re-optimized after pinning each $F_{ik}$:
- `'g_only'` (fast): Only $G$ is adjusted; other $F$ elements held fixed. Conservative (narrower CIs).
- `'acls'`, `'nnls'` (standard): All remaining elements of both $F$ and $G$ re-optimized. More accurate.
- `'newton'` (most rigorous): Full Gauss-Newton re-optimization. Matches EPA PMF2/ME-2 behavior.

**When to use:** For factor interpretability assessment. DISP reveals which factors are solid (narrow ranges, no swaps) vs. fragile (wide ranges, swaps at moderate ΔQ). Not a sample-level uncertainty metric—it's about solution robustness.

---

## Multistart: Do different random initializations converge to the same solution?

**What it does:** Runs PMF from 10-20 different random initial factor matrices and compares the converged solutions.

**What it answers:** Is the solution unique, or do different starting points converge to different local minima? Is the problem well-conditioned?

**Physical interpretation:** Local minimum uniqueness. If all random seeds converge to identical factors (up to factor order), the solution is robust. If different seeds yield visibly different factors, the problem is ill-conditioned or underdetermined.

```python
from pmf_acls import multistart_test

ms = multistart_test(X, sigma, p=3, n_seeds=20)
print(f"Solution stability: {ms.stability_metric}")
# High stability = all seeds converge to same factors
# Low stability = different seeds find different factors
```

**When to use:** To detect ill-conditioned problems. If multistart reveals alternative factorizations, use DISP to assess which is more stable, or add constraints (FPEAK, Bayesian priors).

---

## The Three-Method EPA Recommendation

EPA PMF 5.0 guidance recommends running all three classical methods and interpreting their **agreement or disagreement**:

| Bootstrap CI | DISP range | Multistart stability | Interpretation |
|---|---|---|---|
| Narrow | Narrow | High | Solution is tight; factor is well-determined |
| Wide | Narrow | High | Factor is uncertain globally (sampling) but not rotational |
| Narrow | Wide | High | Factor is rotational but data support is strong |
| Wide | Wide | Low | Solution is fragile; factor may be spurious |

If the three methods agree (narrow across the board, high multistart stability), you have high confidence in the solution. If they conflict, the solution is uncertain or the problem is under-determined.

**Practical workflow:**

```python
from pmf_acls import pmf, bootstrap_uncertainty, displacement_test_F, multistart_test

# Base solution
result = pmf(X, sigma, p=3, algorithm='acls')

# All three
bs = bootstrap_uncertainty(X, sigma, p=3, n_bootstrap=100)
disp = displacement_test_F(X, sigma, result.F, reopt_method='acls')
ms = multistart_test(X, sigma, p=3, n_seeds=20)

# Compare
for k in range(3):
    print(f"\nFactor {k}:")
    print(f"  Bootstrap CI width: {bs.F_ci_high[k] - bs.F_ci_low[k]}")
    print(f"  DISP range (dQ=4): {disp.F_hi[4.0][k] - disp.F_lo[4.0][k]}")
    print(f"  Multistart swaps: {ms.swap_counts[k]}")
```

---

## Bayesian Posterior: What is the joint distribution over (F, G)?

The Bayesian solver provides a fourth, complementary layer: the *joint posterior distribution* over all factor elements.

**What it does:** Gibbs sampling explores the posterior $P(F, G | X, \sigma)$ under a specified prior and likelihood. Returns posterior standard deviations, credible intervals, and full posterior chains.

**What it answers:** Given the data and my prior assumptions, what is the full distribution of plausible factorizations? How does uncertainty in one factor element correlate with uncertainty in others?

**Physical interpretation:** Posterior uncertainty. Unlike bootstrap (data resampling), DISP (local Q-surface geometry), and multistart (local minimum search), the posterior captures the *joint* uncertainty in $(F, G)$ conditional on the assumed model.

```python
from pmf_acls import pmf_bayes

result = pmf_bayes(X, sigma, p=3, n_samples=2000, n_burnin=1000)
print(f"Profile posterior std: {result.F_std}")
print(f"Contribution posterior std: {result.G_std}")
```

**Complementarity:** The posterior does NOT replace the classical methods. It answers a different question:
- **Bootstrap/DISP/Multistart:** "What does the data structure robustly support?"
- **Bayesian posterior:** "What is the joint distribution of parameters under my model and prior?"

See {doc}`bayesian` for interpretation of convergence diagnostics and honest limitations.

---

## Recommended Practice

1. **Always run bootstrap.** It's model-light and tests data robustness.
2. **Run DISP for factor interpretability.** Use `reopt_method='acls'` for publication-quality results.
3. **Run multistart to detect ill-conditioning.** If stability is low, investigate.
4. **If Bayesian tools available,** run the posterior to get joint uncertainty. Compare posterior std to bootstrap/DISP intervals—discrepancies reveal factor coupling.
5. **Report all three.** "Profiles were constrained via bootstrap (95% CI), DISP (ΔQ=4), and multistart (20 seeds). All three methods agreed, supporting factor robustness."

---

## References

- Brown, S. G., Eberly, S., Paatero, P., & Norris, G. A. (2015). Methods for estimating uncertainty in PMF solutions. *Science of the Total Environment*, 518–519, 626–635.
- Paatero, P., Eberly, S., Brown, S. G., & Norris, G. A. (2014). Methods for estimating uncertainty in factor analytic solutions. *Atmospheric Measurement Techniques*, 7(3), 781–797.
- Norris, G. A., et al. (2014). EPA Positive Matrix Factorization (PMF) 5.0 Fundamentals and User Guide. EPA/600/R-14/108.
