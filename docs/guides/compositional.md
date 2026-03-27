# Compositional Data Analysis: The Aitchison Geometry

## The Community Divide

Two environmental science communities use matrix factorization for source apportionment, but they work in fundamentally different physical spaces:

### Air Quality (PMF tradition)

- **What is measured:** Absolute concentrations (µg/m³)
- **What varies:** Total mass, which depends on meteorology
- **Contributions:** Absolute mass flows (e.g., "vehicle exhaust contributed 5 µg/m³ today")
- **Scaling convention:** Normalize source profiles per-factor; total contributions sum to a varying total
- **Standard tools:** EPA PMF 5.0, ESAT, `pmf()` in this package

### Sediment, Geochemistry, Remote Sensing (Unmixing tradition)

- **What is measured:** Relative compositions or spectral reflectances
- **What is fixed:** Total mass or reflectance (the sample "as a whole")
- **Contributions:** Fractional end-member abundances summing to 1 (e.g., "this sample is 40% granite, 30% basalt, 30% sedimentary rock")
- **Scaling convention:** Contributions constrained to the simplex; each sample is a mixture of fixed end members
- **Physical model:** Mixing of end members; the sample IS the mixture, not a collection of independent source flows
- **Standard tools:** EMMA (End-Member Mixing Analysis), spectral unmixing, geological fingerprinting

**These are not variations of the same method—they encode different physical models.** The choice is not about statistics, but about what your data represents.

## When to Use Each

| Your data | Physical model | Use this |
|---|---|---|
| Air quality monitoring (PM₂.₅, VOC, etc.) | Independent sources contribute absolute mass | `pmf()` (ACLS, Bayesian) |
| Sediment fingerprinting | Samples are mixtures of fixed end-member sources | `simplex_pmf()` |
| Compositional data (rocks, alloys, mixtures) | Relative fingerprints matter; absolute magnitudes are sample-dependent | `aitchison_nmf()` |

## The Geometry Problem: Why It Matters

Standard PMF works in **Euclidean space** ($\mathbb{R}^+$) on raw concentrations. For compositional data, this is the wrong geometry.

### The Problem with Euclidean Factorization

Consider two sediment samples:
- Sample A: 40% granite, 60% basalt (measured as absolute measurements, with noise)
- Sample B: 80% granite, 120% basalt (measured with different noise structure)

In Euclidean space, these look like different compositions. But if we normalize:
- Sample A: 40/(40+60) = 0.4, 0.6
- Sample B: 80/(80+120) = 0.4, 0.6

**They are the same composition.** Yet unconstrained PMF on the raw measurements will treat them as different sources, absorbing the magnitude structure into the solution.

**Equally problematic:** The uncertainty structure is distorted. If σ_A ≠ σ_B (different measurement precision), then after simplex projection, the per-element uncertainties become intertwined through a shared denominator. The Bayesian propagation of variance—the delta method—becomes complex and often mishandled.

### The Crude Simplex Solution (and its problems)

Row-normalizing contribution matrix G so that each sample's contributions sum to 1:

$$G'_{kj} = \frac{G_{kj}}{\sum_k G_{kj}}$$

Enforces the constraint, but:
1. **Reconstruction breaks:** Post-hoc normalization changes the fit; $F (G') \neq X$ in general.
2. **Iterative projection needed:** The colleague's unmixing approach runs NMF in a loop, projecting onto the simplex each iteration. This converges, but slowly.
3. **Uncertainty distortion:** The normalization tangles uncertainties across analytes through the shared denominator.

## The Aitchison Resolution: Working in the Correct Geometry

The simplex has its own intrinsic geometry: **the Aitchison geometry**. Working in this space from the start sidesteps all the crude simplex problems.

### Centered Log-Ratio (CLR) Transform

For a composition with $D$ parts (species), the CLR transform is:

$$\text{CLR}_i(x) = \log\left(\frac{x_i}{\text{GM}(x)}\right)$$

where $\text{GM}(x) = \sqrt[D]{\prod_i x_i}$ is the geometric mean.

**Key properties:**
1. **Geometric embedding:** The CLR maps the simplex to a $(D-1)$-dimensional hyperplane in $\mathbb{R}^D$ (since CLR components sum to zero). The map is isometric with respect to the Aitchison inner product but is not injective as a map into $\mathbb{R}^D$ (covariance matrices will be singular). For full-rank representations, see the ILR (Isometric Log-Ratio) transform, which is a bijection to $\mathbb{R}^{D-1}$ (Egozcue et al., 2003).
2. **Separates magnitude from composition:** Scaling all parts by a constant $c$ does not change the CLR (geometric mean cancels out). The CLR sees only the relative structure.
3. **Aitchison-isometric:** Distances in CLR space respect the Aitchison inner product on the simplex.
4. **Recoverable:** Inverse CLR maps back to the simplex (compositions summing to a constant).

### Weighted Aitchison Factorization

`aitchison_nmf()` implements the full approach:

1. **Transform to CLR space:** $\text{CLR}(X)$ maps compositions to the $(D-1)$-dimensional hyperplane in $\mathbb{R}^D$.
2. **Weight via delta method:** Original per-element uncertainties σ propagate to CLR space via:
$$w_{ij} = \frac{X^2_{ij}}{\sigma^2_{ij}}$$
These weights preserve heteroscedastic information. Note: This is a diagonal approximation of the delta-method Jacobian and assumes independent measurement errors across species. For data with correlated measurement errors (common in XRF or ICP-MS due to matrix effects), the full covariance propagation would include off-diagonal terms. In practice, the diagonal approximation is widely used and usually adequate.
3. **Factorize in CLR space:** Minimize weighted distance, discovering profiles in the correct geometry.
4. **Inverse transform:** $\text{CLR}^{-1}(F, G)$ recovers profiles as simplex-valid compositions and contributions.

```python
from pmf_acls import aitchison_nmf

# Data is compositional (rows sum to 1 or to a fixed total)
result = aitchison_nmf(X, sigma, p=3)

# Profiles are on the simplex: row sums ≈ 1
print(f"Profiles (sum to): {result.F.sum(axis=0)}")

# Contributions are absolute quantities (recovered via inverse transform)
print(f"Contributions: {result.G}")

# Uncertainty fully accounts for compositional geometry
print(f"Profile std: {result.F_std}")
```

## The `anchor` Parameter: Geometric Purity vs. Sensitivity

The CLR transform has a potential pitfall for trace species: if a species is near zero, its log is large (and noisy). The `anchor` parameter softens this problem by adding a small pseudo-count before log-transformation.

$$\text{CLR}_{\text{anchor}}(x) = \log\left(\frac{x_i + \text{anchor}}{\text{GM}(x + \text{anchor})}\right)$$

| `anchor` value | Effect | Use when |
|---|---|---|
| 0 (default) | Pure CLR, strict Aitchison geometry | All species well above detection, or analyzing only major components |
| 0.01-MDL | Moderate softening | Trace species present; want to balance composition purity with minor-source sensitivity |
| MDL | Strong softening | Many trace species at/near detection limits; minor sources are scientifically important |

Trade-off: higher anchor → softer log transform → minor species less down-weighted → better sensitivity to trace-driven factors. But higher anchor → breaks scale invariance slightly → deviates from strict Aitchison geometry.

**Recommendation:** Start with `anchor=0` for well-measured data. If trace species factors are suppressed, test `anchor=0.5*MDL` and `anchor=MDL` and compare factor structures. Report the chosen value.

```python
result_strict = aitchison_nmf(X, sigma, p=3, anchor=0.0)
result_moderate = aitchison_nmf(X, sigma, p=3, anchor=0.5)
result_soft = aitchison_nmf(X, sigma, p=3, anchor=1.0)

# Compare factor profiles
import matplotlib.pyplot as plt
for k in range(3):
    plt.figure()
    plt.plot(result_strict.F[:, k], label='anchor=0')
    plt.plot(result_moderate.F[:, k], label='anchor=0.5')
    plt.plot(result_soft.F[:, k], label='anchor=1.0')
    plt.legend()
    plt.title(f"Factor {k}")
```

## `simplex_pmf()`: The Alternative for Hard Simplex Constraints

If you want contributions *strictly* summing to 1 (hard constraint, not soft via geometry), use `simplex_pmf()`:

```python
from pmf_acls import simplex_pmf

# Hard simplex constraint: each sample's contributions sum exactly to 1
result = simplex_pmf(X, sigma, p=3)

# Contributions are normalized
print(f"Contribution sums: {result.G.sum(axis=0)}")
```

`simplex_pmf()` enforces the constraint during optimization (constrained NNLS solver), not post-hoc. It's more direct but less geometrically elegant than Aitchison NMF.

**When to use:**
- You need contributions to be *exactly* fractional (regulatory/physical requirement)
- You're familiar with FCLS (Fully Constrained Least Squares) from remote sensing
- You don't want to think about log transformations or geometric means

**When to prefer Aitchison NMF:**
- You want to preserve heteroscedastic uncertainty structure
- You have trace species and want good minor-source sensitivity
- You want a principled geometric approach (not ad-hoc constraint)

## Choosing Between Methods

| Question | Answer | Use this |
|---|---|---|
| Are contributions absolute quantities (µg/m³)? | Yes | `pmf()` |
| Are contributions fractional (%, ratios)? | Yes | `simplex_pmf()` or `aitchison_nmf()` |
| Do you have many trace species? | Yes | `aitchison_nmf()` (with `anchor` tuning) |
| Do you need hard simplex constraint? | Yes | `simplex_pmf()` |
| Do you want principled geometry? | Yes | `aitchison_nmf()` |

---

## References

- Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*. Chapman and Hall.
- Egozcue, J. J., Pawlowsky-Glahn, V., Mateu-Figueras, G., & Barceló-Vidal, C. (2003). Isometric logratio transformations for compositional data analysis. *Mathematical Geology*, 35(3), 279–300.
- Gillis, N., & Vavasis, S. A. (2015). Fast and robust recursive algorithms for separable nonnegative matrix factorization. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 37(11), 2275–2287.
