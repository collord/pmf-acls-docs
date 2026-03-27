# Bayesian PMF: Theory and Practice

Bayesian inference provides full posterior uncertainty quantification and automated factor count determination. But it also has real limitations that must be acknowledged. This guide explains what the solver genuinely provides, what it cannot do, and how it complements EPA PMF diagnostics.

## The Two Modes

The Bayesian solver operates in two fundamentally different modes:

### Mode 1: Warm-Start Uncertainty Augmentation (default)

**What it does:** Warm-starts from the ACLS solution and runs Gibbs sampling nearby. Returns posterior credible intervals around the point estimate.

**What it provides:**
- Joint credible intervals on all factor elements
- Uncertainty that accounts for coupling between $F$ and $G$ (unlike conditional analytical errors)
- Rapid convergence (warm start means brief burn-in)

**What it doesn't provide:** A new point estimate. The reported $F$ and $G$ are the initialization values, not the posterior means.

```python
from pmf_acls import pmf

# Bayesian UQ on ACLS solution
result = pmf(X, sigma, p=3, algorithm="bayes", warm_start=True, n_samples=1000)
print(f"Profiles: {result.F}")  # ACLS solution
print(f"Profile uncertainty: {result.F_std}")  # Posterior std (uncertainty only)
```

**Use case:** You have an ACLS solution you trust. You want uncertainty bands to communicate confidence, but don't need a new factorization.

### Mode 2: Full Bayesian with ARD Factor Count Inference

**What it does:** Starts from scratch (or from multiple ACLS seeds) and runs long Gibbs chains. Automatic Relevance Determination learns per-factor precision, pruning weak factors.

**What it provides:**
- Full posterior over $(F, G)$ without anchoring to ACLS
- Factor count posterior (probability distribution over number of active factors)
- Data-driven factor selection independent of analyst judgment

**What it doesn't provide:** Rotational analysis (DISP remains necessary) or model adequacy assessment independent of sigma learning.

```python
from pmf_acls import pmf

# Full Bayesian with ARD factor count inference
result = pmf(
    X, sigma, p=6,  # Start with p=6; ARD may prune some
    algorithm="bayes",
    warm_start=False,  # Don't anchor to ACLS
    ard=True,
    n_samples=2000,
    n_burnin=1000,
)
print(f"Factor count posterior: {result.factor_count_posterior}")
```

**Use case:** You want the data to determine factor count. You're willing to spend extra computation for uncertainty quantification and automated dimensionality.

## The Model

The likelihood is a heteroscedastic Gaussian:

$$X_{ij} \mid F, G \sim \mathcal{N}\left(\sum_k F_{ik}\, G_{kj},\; \sigma_{ij}^2\right)$$

The priors are truncated exponential (equivalent to imposing non-negativity):

$$F_{ik} \sim \text{Exp}(\lambda_F), \quad G_{kj} \sim \text{Exp}(\lambda_G)$$

where $\lambda_F$ and $\lambda_G$ control the prior strength (smaller λ pushes harder toward zero). By default, these are learned from data via hierarchical Bayes. Optional: use hierarchical Gamma priors on λ (`learn_hyperparams=True`, default) or fix them (`learn_hyperparams=False`).

## Convergence Diagnostics: How to Read Them

Three diagnostics assess whether the Gibbs sampler has converged:

### Gelman-Rubin Statistic (Rhat)

Run multiple chains from different starting points. If the between-chain variance equals the within-chain variance, chains have converged. Rhat is the ratio of total variance to within-chain variance.

- **Rhat < 1.05:** Chains have converged. Reliable inference.
- **1.05 < Rhat < 1.1:** Marginal. Consider longer burn-in or more samples.
- **Rhat > 1.1:** Chains have not converged. The posterior samples are not from the stationary distribution.

```python
from pmf_acls import gelman_rubin

rhat = gelman_rubin(result.G_samples)  # Must have run multiple chains
print(f"Max Rhat: {rhat.max():.3f}")  # All should be < 1.05
```

### Effective Sample Size (ESS)

Due to autocorrelation, posterior samples are not independent. ESS estimates the equivalent number of independent draws.

- **ESS / n_samples > 0.1:** Good. Enough independent information.
- **ESS / n_samples < 0.05:** Poor. Chains are highly autocorrelated; consider thinning or longer chains.

```python
from pmf_acls import effective_sample_size

ess = effective_sample_size(result.G_samples)
for k in range(result.p):
    print(f"Factor {k}: ESS = {ess[k] / result.G_samples.shape[0]:.1%}")
```

### Label-Switch Gap

In Bayesian NMF, the sampler can permute factor labels (swap factors) during sampling. Ideally this doesn't happen (gap near 1.0). Large gaps indicate that the sampler found multiple local solutions with different factor orderings—a sign of label switching.

```python
print(f"Label switch gap: {result.label_switch_gap:.2f}")  # Should be close to 1.0
```

## ARD (Automatic Relevance Determination)

**What it does:** Learns a per-factor precision hyperparameter $\alpha_k$. Factors with high α are precise (kept); factors with low α are vague (pruned).

**What it provides:** A data-driven factor count, not analyst judgment. The `factor_count_posterior` gives P(k active factors | data).

**What it doesn't guarantee:** That the pruned factors are truly absent. ARD can spuriously prune factors if the data are sparse, or falsely retain factors if the model is misspecified.

**Key parameter:** `ard_threshold` (default 0.01). Factors with $\alpha_k$ below this threshold are considered pruned. Test sensitivity: if factor count changes substantially between threshold=0.005 and threshold=0.05, the result is threshold-dependent and should be reported as uncertain.

```python
result = pmf(X, sigma, p=6, algorithm="bayes", ard=True, ard_threshold=0.01, n_samples=2000)
print(f"Active factors: {result.n_active_factors}")
print(f"Factor count posterior:\n{result.factor_count_posterior}")

# Test threshold sensitivity
for threshold in [0.005, 0.01, 0.02, 0.05]:
    n_active = sum(result.factor_alpha >= threshold)
    print(f"  Threshold {threshold}: {n_active} factors")
```

## Honest Limitations

### 1. Label Switching

In Bayesian NMF, factors can permute (swap labels) during sampling. The solver tries to prevent this via Hungarian matching to the warm-start solution, but it's not perfect. Large label-switch gaps or bimodal posteriors indicate that the sampler found multiple orderings—making posterior means unreliable.

**What to do:** If label switching is detected, increase burn-in or reduce the sampler's exploration parameter.

### 2. Sigma Learning Can Mask Model Inadequacy

If `sigma_prior` is active (learning noise from data), large residuals are absorbed into inflated σ estimates rather than flagged as evidence for additional factors. This means $Q/Q_{\text{exp}}$ on the learned-σ solution is uninformative.

**Required dependency:** When using sigma learning, you *must* run Q/Qexp on the *fixed-σ* baseline solution (from ACLS) to validate the noise model independently.

```python
from pmf_acls import pmf

# Fixed sigma: model adequacy check
result_fixed = pmf(X, sigma, p=3, algorithm='acls')
q_ratio = result_fixed.Q / (X.size - 3 * (X.shape[0] + X.shape[1] - 3))
print(f"Q/Qexp (fixed sigma): {q_ratio:.2f}")  # Should be ≈ 1 for adequate model

# Bayesian with sigma learning: posterior UQ
result_bayes = pmf(X, sigma, p=3, algorithm='bayes', learn_sigma=True)
print(f"Posterior σ: {result_bayes.sigma_posterior_mean}")  # May differ from input sigma
```

### 3. Bayesian Propagates Uncertainty Within a Model Family, Not About It

The posterior is computed under the assumption that the Gaussian likelihood and exponential priors are correct. If the true noise is heavy-tailed or the prior family is wrong, the posterior will be tight and wrong—the Bayesian framework doesn't detect this misspecification.

**Mitigation:** Use the robust likelihood (`robust=True, robust_df=5.0`) for heavy-tailed noise. But this only partially addresses model misspecification.

## Complementarity with EPA PMF

Bayesian inference and EPA PMF diagnostics address **non-overlapping** uncertainty layers:

| Tool | Addresses | Cannot address |
|---|---|---|
| **DISP (rotational ambiguity)** | One-at-a-time factor element sensitivity | Joint factor coupling |
| **Bootstrap (sampling stability)** | Resampling uncertainty | Model-specific uncertainty |
| **Bayesian posterior (joint uncertainty)** | Joint $(F, G)$ distribution under the model | Systematic enumeration of rotations (though label switching and multimodality partially explore rotational freedom) |
| **Q/Qexp (noise model validation)** | Are uncertainties calibrated to residuals? | Factor count (conflates structure and noise) |

**You need both:** Bayesian posterior for credible intervals, EPA PMF tools (DISP, Q/Qexp) for rotational stability and model adequacy.

```python
from pmf_acls import pmf, fpeak_sweep, displacement_test_F, pmf_bayes

# Complementary workflow
acls_result = pmf(X, sigma, p=3)

# EPA PMF diagnostics
disp = displacement_test_F(X, sigma, acls_result.F)  # Rotational stability
sweep = fpeak_sweep(X, sigma, p=3)  # Rotational ambiguity

# Bayesian posterior uncertainty
bayes_result = pmf_bayes(X, sigma, p=3, warm_start=True, n_samples=2000)

# Report
print(f"Rotational stability (DISP): {disp.F_std_approx}")
print(f"Rotational freedom (FPEAK range): {max(sweep.Q_values) - min(sweep.Q_values)} (ΔQ)")
print(f"Joint posterior uncertainty: {bayes_result.F_std}")
```

---

## References

- Schmidt, M. N., Winther, O., & Hansen, L. K. (2009). Bayesian non-negative matrix factorization. *International Conference on Independent Component Analysis and Signal Separation*, 540–547.
- Brouwer, T., Frellsen, J., & Liò, P. (2017). Comparative study of inference methods for Bayesian nonnegative matrix factorisation. In *Machine Learning and Knowledge Discovery in Databases* (ECML PKDD 2017). Lecture Notes in Computer Science, vol. 10534. Springer.
- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman and Hall/CRC.
