# pmf-acls

**Positive Matrix Factorization for environmental data analysis.**

## What is source apportionment?

Environmental source apportionment answers a fundamental question: *what pollution sources contributed to the chemical composition of my samples?* Given a matrix of measured concentrations (species × observations) and estimates of measurement uncertainty, source apportionment decomposes the data into factor profiles (what each source looks like chemically) and contributions (how much each source affected each sample). The technique originated in atmospheric chemistry but is now widely used in water quality, sediment geochemistry, and forensic analysis. The goal is not just statistical dimension reduction — it is source apportionment: attributing measured concentrations to their origin via unique chemical fingerprints, interpreted and validated with domain expertise.

## Why error weighting matters

The core innovation of PMF (versus plain NMF) is **heteroscedastic uncertainty weighting**: different measurements have different reliability, and the factorization must respect that. A trace element measured near its detection limit carries less information than the same element measured well above it. The weighted objective $Q$ (below) ensures that precise measurements exert more influence on the solution than noisy ones. This is not a cosmetic choice — error weighting is fundamental to the method. The package can operate at different knowledge levels: from coarse error classes (when you know little about reliability) to observation-level uncertainties (when measurement QA is well-characterized).

$$Q = \sum_{i,j} \left[\frac{x_{ij} - \sum_k f_{ik}\, g_{kj}}{\sigma_{ij}}\right]^2$$

## What this package provides

`pmf-acls` is a Python-native implementation of PMF and NMF with modern uncertainty quantification. It serves researchers working in two distinct traditions:

- **Air quality & receptor modeling (EPA PMF tradition):** Uses absolute mass contributions (µg/m³) and source profiles normalized per-factor. The classical ACLS solver is the default; Bayesian inference provides posterior uncertainty bands and automatic factor count determination via ARD.
- **Sediment, water, and geochemistry (unmixing tradition):** Works with fractional end-member abundances summing to 1. The package provides `simplex_pmf()` for hard simplex constraints and `aitchison_nmf()` for compositional data in the Aitchison geometry, where fingerprint and magnitude information are naturally separated.

**Solver landscape:** EPA PMF 5.0 (a free tool implementing Paatero's proprietary ME-2 algorithm) was the standard; ESAT is its open-source successor. This package offers a complementary approach: Python-native, open-source, with built-in Bayesian uncertainty quantification, factor count inference via ARD, and full support for compositional data analysis. It is not positioned as a replacement for EPA PMF, but as an alternative for users who need Bayesian posterior inference, automated factor selection, or the ability to work directly in compositional space.

## Quick start

```python
from pmf_acls import pmf

# ACLS solver (default, fastest)
result = pmf(X, sigma, p=3)

# Bayesian inference: posterior uncertainty + factor count distribution
result = pmf(X, sigma, p=3, algorithm="bayes", ard=True)
print(f"Factor count posterior: {result.factor_count_posterior}")

# Compositional data (sediment, geochemistry)
from pmf_acls import aitchison_nmf
result = aitchison_nmf(X, sigma, p=3)  # Works in CLR space with delta-method weights
```

## Convention Note

**Matrix notation:** This package uses $X$ as (species × observations), which transposes the standard EPA PMF convention (observations × species). When comparing against EPA PMF outputs or publications following the Paatero & Tapper (1994) notation, transpose accordingly. Internally, $G$ is the contributions matrix (factors × observations) and $F$ is the profiles matrix (species × factors).

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

guides/why_error_weighting
guides/algorithms
guides/factor_selection
guides/uncertainty
guides/bayesian
guides/rotation
guides/compositional
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
