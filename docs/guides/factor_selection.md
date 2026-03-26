# Factor Selection

Choosing the number of factors $p$ is one of the most important decisions in
PMF analysis. `pmf-acls` provides automated tools to guide this choice.

## Q-based selection

```python
from pmf_acls import select_factors

selection = select_factors(X, sigma, p_range=(2, 8), n_runs=5)
print(f"Best p: {selection.best_p}")
```

The selection considers:
- **Q/E[Q]**: should decrease with increasing $p$ and stabilize near 1.0
- **Explained variance**: marginal improvement diminishes at the correct $p$
- **Factor interpretability**: degenerate or split factors indicate over-fitting

## Manual sweep

```python
from pmf_acls import pmf

for p in range(2, 9):
    result = pmf(X, sigma, p=p)
    dof = X.shape[0] * X.shape[1] - p * sum(X.shape)
    q_robust = result.Q / dof if dof > 0 else float("nan")
    print(f"p={p}  Q={result.Q:.0f}  Q/E[Q]={q_robust:.2f}  R²={result.explained_variance:.3f}")
```

## Bayesian model comparison

WAIC (Widely Applicable Information Criterion) provides a principled
comparison across factor counts:

```python
from pmf_acls import pmf_bayes, compute_waic

waics = {}
for p in range(2, 9):
    result = pmf_bayes(X, sigma, p=p, store_samples=True)
    waics[p] = compute_waic(X, sigma, result.F_samples, result.G_samples)
# Lower WAIC = better predictive fit
```
