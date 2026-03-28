# Factor Selection: The Diagnostic Discrepancy Principle

Choosing the number of factors $p$ is one of the most important decisions in source apportionment. Yet factor count determination is fundamentally a *two-question problem*, and conflating them leads to wrong answers.

## What Q/Qexp Actually Is (and Isn't)

The statistic $Q / Q_{\text{exp}}$ is **a noise model validation metric**, not a factor count determination metric. To understand why, we need to look at the theory.

### The Statistical Basis

Under the assumption that the model is correct and the uncertainty estimates $\sigma_{ij}$ are well-calibrated, each standardized residual $(X_{ij} - (FG)_{ij}) / \sigma_{ij}$ should follow approximately $\mathcal{N}(0, 1)$. The sum of squared standardized residuals, $Q$, should thus be $\chi^2$ distributed with degrees of freedom equal to:

$$\nu = (n_{\text{obs}} \times n_{\text{species}}) - p(n_{\text{obs}} + n_{\text{species}} - p)$$

(This is Paatero's formula; the EPA simplification nm - p(n + m) is slightly different and less accurate when constraints are active.) Under this model, $E[Q] = \nu$, so $Q / Q_{\text{exp}} \approx 1$ indicates that the observed residuals match the expected noise level.

### Why the Theory Breaks Down for Factor Determination

This chi-squared derivation requires three assumptions that **all three are violated in practice**:

1. **Linear model:** The derivation assumes linear least squares (y = Xβ). PMF is *bilinear* in the parameters (F, G), and the chi-squared distribution does not transfer cleanly to nonlinear problems.

2. **Unconstrained optimization:** The chi-squared result assumes the minimum is in the interior of parameter space. PMF's non-negativity constraints create *active constraints*, where elements of F or G are exactly zero. This changes the effective degrees of freedom in a data-dependent way that neither DOF formula accounts for.

3. **Known variances:** The derivation assumes σ are the *true* error standard deviations, not estimates. In practice, σ are always estimated from equations, replicate measurements, or ad-hoc formulas—introducing uncertainty that chi-squared theory ignores.

Paatero (1997) himself acknowledged these violations, calling Q "asymptotically" chi-squared. He continued to use Q/Qexp as a useful *heuristic*, but warned that Q curves can produce "spurious shapes" and "unfounded choices of dimension." His own practice was to treat Q/Qexp as one diagnostic among several, not the final arbiter of factor count.

### What Q/Qexp Actually Tells You

| Observation | Interpretation |
|---|---|
| $Q/Q_{\text{exp}} \approx 1$ | Your stated uncertainties align with observed residuals. Good fit at the noise level you claimed. |
| $Q/Q_{\text{exp}} > 1$ | Residuals are larger than expected. Either the model is underfitting (missing factors) or your uncertainties are too optimistic. |
| $Q/Q_{\text{exp}} < 1$ | Residuals are smaller than expected. Either you have too many factors (overfitting) or your uncertainties are pessimistic. |

**Crucially:** Two models with different $p$ can *both* have $Q/Q_{\text{exp}} \approx 1$. Q alone does not determine which is correct.

## The Diagnostic Discrepancy Principle

This section introduces the **Diagnostic Discrepancy Principle**—a framework developed for this package that synthesizes two complementary diagnostic methods. While the individual components (scree/elbow testing and Q/Qexp analysis) are established practices in the literature, their specific combination under this name and the interpretive framework below are novel to this package. The approach is methodologically sound but should be understood as a package-proposed heuristic rather than a validated statistical criterion from the peer-reviewed literature.

Factor selection requires **two independent metrics** that ask different questions:

### 1. Structural Question: How many dimensions of variation exist?

Use **norm-standardized reconstruction error** (or equivalently, Euclidean distance inflection, or scree test on singular values):

$$\text{RMSE}_{\text{norm}} = \frac{1}{\sqrt{mn}} \left\| W^{-1/2} (X - FG) \right\|_F$$

where $W^{-1/2}$ is the square root of the inverse weight matrix. This produces an uncertainty-independent "elbow" in the reconstruction error curve. The elbow point $p_{\text{struct}}$ is the number of factors needed to explain the data's structured variation, independent of any noise model assumptions.

Practical implementation:

```python
import numpy as np
from pmf_acls import pmf

p_range = range(2, 9)
rmses = []
for p in p_range:
    result = pmf(X, sigma, p=p)
    # Compute norm-standardized RMSE
    residual = X - result.F @ result.G
    weighted_residual = residual / sigma  # Element-wise weighting
    rmse = np.linalg.norm(weighted_residual) / np.sqrt(X.size)
    rmses.append(rmse)

# Look for elbow in rmses vs p_range
```

### 2. Noise Model Question: Are my uncertainties well-calibrated?

Use $Q / Q_{\text{exp}}$ (or WAIC for Bayesian models):

$$p_{\text{noise}} = \arg_p (Q / Q_{\text{exp}} \approx 1)$$

This is the factor count at which your stated uncertainties are consistent with observed residuals.

```python
from pmf_acls import pmf

p_range = range(2, 9)
q_ratios = []
for p in p_range:
    result = pmf(X, sigma, p=p)
    dof = X.shape[0] * X.shape[1] - p * (X.shape[0] + X.shape[1] - p)
    q_ratio = result.Q / dof if dof > 0 else np.nan
    q_ratios.append(q_ratio)

# Look for p where Q/Qexp ≈ 1
```

### 3. Interpret the Discrepancy

| Discrepancy | Candidate explanation | Action |
|---|---|---|
| $p_{\text{struct}} = p_{\text{noise}}$ | Convergent evidence. Data structure agrees with noise model. | Choose $p$; high confidence. |
| $p_{\text{struct}} > p_{\text{noise}}$ | Your uncertainties may be too generous (overstated). The data structurally needs more factors, but noise model says enough. | Test sensitivity to uncertainty specification (see {doc}`why_error_weighting`). If conclusion doesn't change with 2× tighter σ, use $p_{\text{struct}}$. |
| $p_{\text{struct}} < p_{\text{noise}}$ | Your uncertainties may be too tight (understated). Noise model wants more factors, but data structure doesn't need them. | Re-examine your error specification. Or accept the overfitting: sometimes Q/Qexp can be less than 1 if the problem is simply well-measured. |

## Bayesian Automated Factor Count

When Bayesian tools are available, the **ARD (Automatic Relevance Determination) factor count posterior** provides a data-driven alternative to this manual negotiation:

```python
from pmf_acls import pmf

# Full Bayesian with ARD
result = pmf(X, sigma, p=5, algorithm="bayes", warm_start=False, ard=True,
             n_samples=2000, n_burnin=1000)

# Factor count posterior
print(f"P(p=3 | data) = {result.factor_count_posterior.get(3, 0):.2%}")
print(f"P(p=4 | data) = {result.factor_count_posterior.get(4, 0):.2%}")
print(f"P(p=5 | data) = {result.factor_count_posterior.get(5, 0):.2%}")
```

The ARD posterior learns which factors are "active" (high precision) vs. "pruned" (low precision). This provides a probability distribution over factor count conditioned on the data and your prior assumptions. The threshold sensitivity (how much does the posterior change if you vary the pruning threshold?) indicates whether the factor count is robust or borderline.

## WAIC: Predictive Model Comparison

When using Bayesian inference, WAIC (Widely Applicable Information Criterion) provides a principled predictive comparison across factor counts:

```python
from pmf_acls import pmf_bayes, compute_waic

waics = {}
for p in range(2, 9):
    result = pmf_bayes(X, sigma, p=p, store_samples=True, n_samples=2000)
    waics[p] = compute_waic(X, sigma, result.F_samples, result.G_samples)

best_p = min(waics, key=waics.get)
print(f"Best p by WAIC: {best_p}")
for p, waic in sorted(waics.items()):
    print(f"  p={p}  WAIC={waic:.0f}")
```

WAIC estimates out-of-sample predictive accuracy. Lower WAIC indicates better generalization. It is preferred over AIC/BIC for hierarchical models like PMF because it accounts for uncertainty in the hyperparameters.

## Practical Workflow

1. **Run both $p_{\text{struct}}$ and $p_{\text{noise}}$**: Plot RMSE vs. $p$ and Q/Qexp vs. $p$ on the same figure.
2. **Interpret the discrepancy**: Do they agree? If not, investigate your uncertainty specification (see {doc}`why_error_weighting`).
3. **If Bayesian tools available**: Run ARD and check factor count posterior distribution. Use threshold sensitivity analysis.
4. **Sensitivity test**: Re-run with alternative error specifications (1.5× tighter, 1.5× looser σ). Do factors appear/disappear? If yes, your choice is uncertain.
5. **Report with caveats**: "Based on the Diagnostic Discrepancy Principle, we selected p=4. Q/Qexp shows convergence, but RMSE inflection occurs at p=5. Sensitivity testing with ±50% uncertainty variation does not change qualitative factor structure."

---

## References

- Paatero, P. (1997). Least squares formulation of robust non-negative factor analysis. *Chemometrics and Intelligent Laboratory Systems*, 37(1), 23–35.
- Paatero, P., Eberly, S., Brown, S. G., & Norris, G. A. (2014). Methods for estimating uncertainty in factor analytic solutions. *Atmospheric Measurement Techniques*, 7(3), 781–797.
- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman and Hall/CRC.
