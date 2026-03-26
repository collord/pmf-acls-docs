# Bayesian PMF

The Bayesian solver uses Gibbs sampling to draw from the posterior distribution
over factor matrices $F$ and $G$, providing full uncertainty quantification.

## Model

$$X_{ij} \sim \mathcal{N}\!\left(\sum_k F_{ik}\, G_{kj},\; \sigma_{ij}^2\right)$$

with truncated-exponential priors:

$$F_{ik} \sim \text{Exp}(\lambda_F), \quad G_{kj} \sim \text{Exp}(\lambda_G)$$

## Basic usage

```python
from pmf_acls import pmf_bayes

result = pmf_bayes(
    X, sigma, p=3,
    n_samples=1000,
    n_burnin=500,
    n_thin=2,
    store_samples=True,
)
```

## Warm starting

By default, the sampler runs multiple short ACLS seeds and initializes from the
best one (`warm_start=True`, `warm_start_seeds=5`).

## Automatic Relevance Determination (ARD)

ARD learns a per-factor precision and prunes irrelevant factors:

```python
result = pmf_bayes(X, sigma, p=6, ard=True, ard_threshold=0.01)
# result.G may have fewer than 6 active rows
```

## Convergence diagnostics

```python
from pmf_acls import gelman_rubin, effective_sample_size, compute_waic

ess = effective_sample_size(result.G_samples)
rhat = gelman_rubin(result.G_samples)
waic = compute_waic(X, sigma, result.F_samples, result.G_samples)
```

## Robust likelihood

For data with outliers, use a Student-t likelihood:

```python
result = pmf_bayes(X, sigma, p=3, robust=True, robust_df=5.0)
```
