# Quick Start

## Basic PMF

```python
import numpy as np
from pmf_acls import pmf

# Generate synthetic data
rng = np.random.default_rng(42)
F_true = rng.gamma(2.0, 1.0, size=(100, 3))
G_true = rng.gamma(1.0, 1.0, size=(3, 20))
X = F_true @ G_true
sigma = 0.1 * X + 0.01

# Solve
result = pmf(X, sigma, p=3)
print(f"Converged: {result.converged}")
print(f"Q: {result.Q:.1f}")
print(f"Explained variance: {result.explained_variance:.3f}")
```

## Bayesian PMF

```python
from pmf_acls import pmf_bayes

result = pmf_bayes(
    X, sigma, p=3,
    n_samples=1000,
    n_burnin=500,
    store_samples=True,
)

# Posterior mean profiles
G_mean = result.G        # (p, n_vars)
G_std = result.G_std     # posterior uncertainty

# Full posterior chains (if store_samples=True)
G_samples = result.G_samples  # (n_samples, p, n_vars)
```

## Factor selection

```python
from pmf_acls import select_factors

selection = select_factors(X, sigma, p_range=(2, 8), n_runs=5)
print(f"Best p: {selection.best_p}")
```

## FPEAK rotation sweep

```python
from pmf_acls import fpeak_sweep

sweep = fpeak_sweep(X, sigma, p=3, fpeak_values=[-1, -0.5, 0, 0.5, 1])
# Flat Q-vs-FPEAK → large rotational ambiguity
```
