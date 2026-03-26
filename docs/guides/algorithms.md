# Algorithms

`pmf-acls` provides four algorithms through the unified `pmf()` entry point, plus
a standalone Bayesian LDA variant.

## ACLS (default)

Alternating Constrained Least Squares {cite}`langville2014`. Solves weighted
$k \times k$ normal equations per column/row. Fast and robust.

```python
result = pmf(X, sigma, p=3, algorithm="acls")
```

Key parameters: `lambda_W`, `lambda_H` (Tikhonov regularization), `fpeak` (rotation).

## LS-NMF

Weighted multiplicative update rules {cite}`wang2006`. Equivalent to the algorithm
used in ESAT. Slower convergence but sometimes avoids local minima that ACLS hits.

```python
result = pmf(X, sigma, p=3, algorithm="ls-nmf")
```

## Newton

Newton-based solver with exact Hessian. Solves the full $(mp + np) \times (mp + np)$
system. Handles uncertainties naturally but scales poorly with matrix size.

```python
result = pmf(X, sigma, p=3, algorithm="newton", mode="regularized")
```

## Bayesian (Gibbs)

Full Bayesian inference via Gibbs sampling {cite}`schmidt2009`. Provides posterior
uncertainty on all factor elements. See {doc}`bayesian` for details.

```python
result = pmf(X, sigma, p=3, algorithm="bayes", n_samples=1000)
```

## LDA variant

Dirichlet-constrained profiles (each source profile sums to 1):

```python
from pmf_acls import pmf_lda

result = pmf_lda(X, sigma, p=3, n_samples=1000, alpha=1.0)
```

## Comparison

| Algorithm | Speed | Uncertainty weighting | Posterior uncertainty | Rotation control |
|-----------|-------|-----------------------|----------------------|------------------|
| ACLS      | Fast  | Per-element           | No                   | FPEAK            |
| LS-NMF    | Medium| Per-element           | No                   | No               |
| Newton    | Slow  | Exact Hessian         | No                   | No               |
| Bayes     | Slow  | Per-element           | Yes                  | Via prior        |
| LDA       | Slow  | Per-element           | Yes                  | Simplex          |
