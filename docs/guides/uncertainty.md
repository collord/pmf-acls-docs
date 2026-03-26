# Uncertainty Quantification

`pmf-acls` provides three complementary approaches to uncertainty estimation,
following EPA PMF 5.0 methodology {cite}`norris2014`.

## Bootstrap uncertainty

Block bootstrap resampling of observations:

```python
from pmf_acls import bootstrap_uncertainty

bs = bootstrap_uncertainty(X, sigma, p=3, n_bootstrap=100)
# bs.F_ci_low, bs.F_ci_high — confidence intervals on profiles
# bs.G_ci_low, bs.G_ci_high — confidence intervals on contributions
```

## Displacement test

Perturbs each factor element and re-fits to assess local sensitivity:

```python
from pmf_acls import displacement_test

disp = displacement_test(X, sigma, p=3)
```

## Multistart test

Multiple random initializations to assess solution stability:

```python
from pmf_acls import multistart_test

ms = multistart_test(X, sigma, p=3, n_seeds=20)
```

## Bayesian posterior

The Bayesian solver provides posterior standard deviations directly:

```python
from pmf_acls import pmf_bayes

result = pmf_bayes(X, sigma, p=3, store_samples=True)
# result.G_std — posterior standard deviation per element
# result.G_samples — full posterior chain for custom credible intervals
```
