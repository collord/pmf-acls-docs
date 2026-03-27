# Rotational Ambiguity & FPEAK

## The Fundamental Problem: Non-Uniqueness

Non-negative matrix factorization solutions are *not unique*. For any invertible matrix $T$ such that both $FT$ and $T^{-1}G$ remain non-negative, the products are identical:

$$(FT)(T^{-1}G) = F(TT^{-1})G = FG$$

This means infinitely many $(F', G') = (FT, T^{-1}G)$ pairs give the same data fit. Different rotations can represent qualitatively different source profiles and contributions — yet they all explain the data equally well.

**Why is this a problem?** In source apportionment, the rotational freedom is not just mathematical ambiguity—it is genuine physical ambiguity. Two sources with different profiles that happen to covary can appear as one source or two, depending on the rotation. A factor that is "peaked" (dominates a few species) or "diffuse" (present in all species) depends on how the solution is rotated.

Paatero's PMF includes the **FPEAK** rotation parameter to manage this ambiguity. This guide explains how it works and what the results mean.

## How FPEAK Works Mechanically

FPEAK adds a rotation penalty to the objective:

$$Q_{\text{FPEAK}} = Q + \text{FPEAK} \cdot \sum_k \left(\sum_i F_{ik}\right)^2$$

**Positive FPEAK:** Penalizes the sum of each profile column. The solver is encouraged to make profiles *peaked*: each source dominates a small set of species.

**Negative FPEAK:** Penalizes the *difference* in column sums. The solver is encouraged to make profiles *diffuse*: each source contributes to all species.

By sweeping FPEAK across a range (e.g., -2 to +2), you explore how the solution rotates and how robust the rotation choice is.

## FPEAK Sweep and Interpretation

Run a sweep to explore rotational freedom:

```python
from pmf_acls import fpeak_sweep
import matplotlib.pyplot as plt
import numpy as np

sweep = fpeak_sweep(
    X, sigma, p=3,
    fpeak_values=np.linspace(-2, 2, 9),
    verbose=True,
)

# Plot Q vs FPEAK
plt.figure(figsize=(8, 5))
plt.plot(sweep.fpeak_values, sweep.Q_values, "o-", linewidth=2)
plt.axhline(y=sweep.Q_values[len(sweep.Q_values)//2], color='r', linestyle='--', alpha=0.5, label='Q at FPEAK=0')
plt.xlabel("FPEAK", fontsize=12)
plt.ylabel("Q (objective)", fontsize=12)
plt.title("Rotational Ambiguity: Q vs FPEAK")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# How much does Q change?
q_range = max(sweep.Q_values) - min(sweep.Q_values)
print(f"Q range: {q_range:.0f} (ΔQ = {q_range / sweep.Q_values[len(sweep.Q_values)//2] * 100:.1f}%)")
```

### Reading the Curve

| Curve shape | Interpretation | Action |
|---|---|---|
| **Sharp V-shaped valley** | FPEAK=0 is clearly optimal. Rotation is well-determined. | Use FPEAK=0. Factors are well-constrained by the data. |
| **Broad, flat plateau** | Q is insensitive to FPEAK. Large rotational freedom. | Factors are under-determined. Report profiles with caveats about rotational ambiguity. Use DISP to assess stability. |
| **Asymmetric (e.g., valley at FPEAK=+0.5)** | The data prefer a particular rotation direction. | Run factorization with the optimal FPEAK value. Peaked profiles are favored by the data structure. |
| **Multiple local minima** | Multiple rotations are locally optimal. | Choose one and report the ambiguity. Consider using Bayesian priors instead. |

## Interpreting Peaked vs. Diffuse Profiles

**Peaked profiles (positive FPEAK):** Each source dominates a small set of species. In air quality, this corresponds to distinct emission sources with unique fingerprints (e.g., lead in particles → vehicle emissions, vanadium → oil combustion).

**Diffuse profiles (negative FPEAK):** Each source contributes to all species similarly. This can indicate secondary formation (e.g., sulfate, which is formed from multiple sources), or poor resolution due to weak constraints.

**Physically:** The data structure itself prefers one rotation over another. If the sweep shows a sharp minimum at FPEAK > 0, the source profiles are genuinely distinct and well-separated. If the sweep is flat, the sources are overlapping or inter-correlated.

```python
# Get factors at different FPEAK values
import numpy as np
sweep = fpeak_sweep(X, sigma, p=3, fpeak_values=np.linspace(-1, 1, 5))

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for k in range(3):
    ax = axes[k]
    for fpeak_idx, fpeak_val in enumerate(sweep.fpeak_values):
        ax.plot(sweep.results[fpeak_val].F[:, k], label=f"FPEAK={fpeak_val:+.1f}", alpha=0.7)
    ax.set_title(f"Factor {k}: Profile across rotations")
    ax.set_xlabel("Species")
    ax.set_ylabel("Profile loading")
    ax.legend(fontsize=8)
plt.tight_layout()
```

## DISP and Rotational Stability

FPEAK sweep reveals *global* rotational freedom (is the problem generally under-determined?). DISP (displacement test) reveals *local* rotational stability (can the resolved factors be pushed apart without breaking the fit?).

**Complementary use:**
- **FPEAK sweep flat:** Global rotational ambiguity. Factors are under-determined by data structure.
- **DISP reveals factor swaps at moderate ΔQ:** Local rotational instability. Factors become confused under perturbation.
- **FPEAK sharp minimum + DISP shows no swaps:** Factors are rotationally well-determined. High confidence.

See {doc}`uncertainty` for detailed DISP interpretation.

## The Bayesian Approach to Rotation

The Bayesian solver doesn't use FPEAK. Instead, rotational preference is encoded in the prior on $G$:

$$G_{kj} \sim \text{Exp}(\lambda_G)$$

**Stronger (smaller) λ:** Pushes toward sparse, peaked profiles (similar to positive FPEAK).

**Weaker (larger) λ:** Allows diffuse profiles (similar to negative FPEAK).

If you enable hierarchical Bayes (`learn_hyperparams=True`, default), λ is learned from data—automatically finding the rotation that best explains the data under the assumed prior family.

```python
from pmf_acls import pmf

# Default: learns λ, automatically balances rotation
result_auto = pmf(X, sigma, p=3, algorithm="bayes", warm_start=False, learn_hyperparams=True)

# Fixed λ: user-specified rotation preference
# Smaller lambda_G → more peaked
result_peaked = pmf(X, sigma, p=3, algorithm="bayes", warm_start=False,
                    lambda_G=0.5, learn_hyperparams=False)

# Larger lambda_G → more diffuse
result_diffuse = pmf(X, sigma, p=3, algorithm="bayes", warm_start=False,
                     lambda_G=2.0, learn_hyperparams=False)

# Compare profiles
import numpy as np
for k in range(3):
    peaked_sum = np.sum(result_peaked.F[result_peaked.F[:, k] > 0.01, k])
    diffuse_sum = np.sum(result_diffuse.F[result_diffuse.F[:, k] > 0.01, k])
    print(f"Factor {k}: Peaked={peaked_sum:.2f}, Diffuse={diffuse_sum:.2f}")
```

**Qualitative similarity, different mechanisms:** Bayesian λ_G and ACLS FPEAK both influence the peaked/diffuse trade-off, but through fundamentally different mechanisms. FPEAK is a deterministic rotation control that modifies the objective surface during optimization. λ_G is a sparsity prior that operates through the posterior sampler. While smaller λ_G qualitatively favors peaked profiles (similar to positive FPEAK) and larger λ_G favors diffuse profiles, they are not interchangeable: tuning λ_G will *not* replicate an FPEAK sweep, and the sampled posteriors will explore the rotation manifold differently than a deterministic rotation penalty would.

## Practical Workflow

1. **Run FPEAK sweep** on your ACLS solution. Look for the curve shape.
2. **If sweep is sharp (V-shaped):** Rotation is well-determined. Use FPEAK=0 or the optimal value.
3. **If sweep is flat:** Rotational ambiguity is large. Use DISP to check local stability. Consider using Bayesian inference with a strong prior (small λ_G) to regularize the rotation.
4. **Report the ambiguity:** "The Q-vs-FPEAK curve was flat (Q range = 50, ~1% of Q), indicating substantial rotational freedom. DISP found no factor swaps, but profile differences across FPEAK±1.0 exceeded measurement uncertainty. We report results at FPEAK=0 with this caveat."
5. **If factors are important, use multiple rotation metrics:** Report profiles at FPEAK=-1, 0, +1 to show the range of plausible solutions.

---

## References

- Paatero, P. (1997). Least squares formulation of robust non-negative factor analysis. *Chemometrics and Intelligent Laboratory Systems*, 37(1), 23–35.
- Paatero, P., & Hopke, P. K. (2003). Discarding or downweighting high-noise variables in factor analytic models. *Atmospheric Environment*, 37(21), 2891–2902.
- Norris, G. A., et al. (2014). EPA Positive Matrix Factorization (PMF) 5.0 Fundamentals and User Guide. EPA/600/R-14/108.
