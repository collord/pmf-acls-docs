# Cross-Algorithm Comparison

```{note}
The interactive cross-algorithm comparison notebook with sliders for matrix size,
noise fraction, and factor sweep lives in the
[downstream-pmf](https://github.com/collord/downstream-pmf) repo and requires
additional dependencies (ESAT, rpy2, scikit-learn, wine for ME2/PMF2).

See `downstream-pmf/notebooks/cross_algorithm_comparison.ipynb` for the full
interactive version with all 15+ solvers.
```

## Available solvers

The comparison notebook benchmarks these NMF/PMF implementations:

| Solver | Uncertainty weighting | Source |
|--------|----------------------|--------|
| **PMF_ACLS** | Per-element | `pmf_acls` |
| **PMF_Bayes** | Per-element | `pmf_acls` |
| **ESAT** (multiseed) | Per-element | `esat` |
| **sklearn NMF** | None | `scikit-learn` |
| **PMF2** | Per-element | EPA reference (wine) |
| **ME2** | Per-element | EPA reference (wine) |
| **R NMF** (9 methods) | Varies | `NMF` R package via `rpy2` |

## Metrics evaluated

- **MAE / RMSE**: reconstruction error against true signal
- **R²**: global explained variance
- **Q**: weighted sum of squared residuals $\sum (X - FG)^2 / \sigma^2$
- **Q/E[Q]**: Q normalized by degrees of freedom (ideal ≈ 1.0)
- **Cosine similarity**: profile shape recovery
- **Scale error**: relative total-mass error per factor
- **Contribution correlation**: temporal pattern recovery
- **Normed Euclidean distance**: on both X and profile space
