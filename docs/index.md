# pmf-acls

**Positive Matrix Factorization for environmental data analysis.**

`pmf-acls` provides uncertainty-weighted non-negative matrix factorization with
five solver backends, full Bayesian inference, rotational analysis, and
bootstrap/displacement uncertainty quantification.

$$Q = \sum_{i,j} \left[\frac{x_{ij} - \sum_k f_{ik}\, g_{kj}}{\sigma_{ij}}\right]^2$$

## Quick start

```python
from pmf_acls import pmf

result = pmf(X, sigma, p=3)          # ACLS (default)
result = pmf(X, sigma, p=3,
             algorithm="bayes")       # Bayesian via Gibbs sampling
```

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

guides/algorithms
guides/bayesian
guides/uncertainty
guides/factor_selection
guides/rotation
```

```{toctree}
:maxdepth: 2
:caption: Examples

notebooks/index
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: Reference

changelog
references
```
