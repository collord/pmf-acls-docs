# Rotational Ambiguity & FPEAK

NMF solutions are not unique — any rotation $F' = FT$, $G' = T^{-1}G$ that
preserves non-negativity yields an equally valid factorization. FPEAK controls
this rotational degree of freedom.

## FPEAK sweep

```python
from pmf_acls import fpeak_sweep

sweep = fpeak_sweep(
    X, sigma, p=3,
    fpeak_values=[-2, -1, -0.5, 0, 0.5, 1, 2],
)

# Plot Q vs FPEAK
import matplotlib.pyplot as plt
plt.plot(sweep.fpeak_values, sweep.Q_values, "o-")
plt.xlabel("FPEAK")
plt.ylabel("Q")
```

## Interpretation

- **FPEAK > 0**: biases toward peaked (sparse) profiles
- **FPEAK < 0**: biases toward diffuse profiles
- **Flat Q-vs-FPEAK curve**: large rotational ambiguity — interpret factors cautiously
- **Sharp Q increase at FPEAK ≠ 0**: well-determined rotation

## Bayesian approach

The Bayesian solver handles rotational ambiguity through the prior:

```python
from pmf_acls import pmf_bayes

# Stronger G prior → sparser profiles (similar to positive FPEAK)
result = pmf_bayes(X, sigma, p=3, lambda_G=2.0, learn_hyperparams=False)
```
