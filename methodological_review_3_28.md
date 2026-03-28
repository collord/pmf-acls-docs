# Methodological Review — External Docs vs. Implementation (2026-03-28)

Issues found comparing `/Users/collord/pmf-acls-docs/docs/` against the actual `pmf_acls` codebase. The code was recently refactored to Paatero 1994 convention and NMF→PMF naming. Many of these are stale references from before those changes.

---

## Critical: Convention Note Is Inverted

**File:** `index.md`, line ~43

The Convention Note says X is "(species × observations)" with G as "(factors × observations)" and F as "(species × factors)". This is the **old Lee & Seung convention** — the exact opposite of what the code now implements.

**Code uses Paatero 1994:**
- X is (n, m) — observations × variables
- G is (n, p) — observations × factors
- F is (p, m) — factors × variables
- X ≈ G @ F

This will confuse every user who reads the convention note first.

---

## Critical: Quickstart Synthetic Data Uses Old Shapes

**File:** `quickstart.md`, lines ~11-12

```python
F_true = rng.gamma(2.0, 1.0, size=(100, 3))   # should be (3, 100)
G_true = rng.gamma(1.0, 1.0, size=(3, 20))     # should be (20, 3)
```

Also line ~36: `G_mean = result.G` with comment `# (p, n_vars)` — should be `(n, p)`.

---

## Critical: `algorithm="ls-nmf"` Will Crash

**File:** `guides/algorithms.md`, line ~49

Code example passes `algorithm="ls-nmf"`. The actual valid string is `"ls-pmf"`. The surrounding prose correctly mentions "LS-PMF" but the code example contradicts it.

---

## Stale Names (NMF → PMF rename)

| File | Issue |
|------|-------|
| `index.md` lines 20, 37-38 | `aitchison_nmf()` → `aitchison_pmf()` |
| `guides/compositional.md` lines 32, 83, 93, 96, 125-127, 171-174 | `aitchison_nmf()` → `aitchison_pmf()` |
| `api/data_structures.md` line ~11 | `BayesNMFResult` → `BayesPMFResult` |
| `changelog.md` line ~5 | "LS-NMF" → "LS-PMF" |
| `notebooks/marimo_bayesian.md` line 1 | "Bayesian NMF" → "Bayesian PMF" |
| `notebooks/marimo_compositional.md` line 25 | "Aitchison NMF" → "Aitchison PMF" |

Note: backward-compatible aliases exist (`aitchison_nmf`, `BayesNMFResult`), so these won't crash, but docs should use canonical names.

---

## Incorrect Matrix Shapes (Old Convention)

| File | Line | Issue |
|------|------|-------|
| `guides/bayesian.md` ~63 | Formula uses $F_{ik} G_{kj}$ (X=FG). Should be $G_{jk} F_{ki}$ (X=GF) |
| `guides/compositional.md` ~54-56 | G normalization formula uses G(p,n) indexing. Should be G(n,p) |
| `guides/compositional.md` ~99 | `result.F.sum(axis=0)` — should be `axis=1` for row sums of F(p,m) |
| `guides/compositional.md` ~133-136 | `result.F[:, k]` — should be `result.F[k, :]` for k-th factor profile |
| `guides/compositional.md` ~151 | `result.G.sum(axis=0)` — should be `axis=1` for per-observation sums |
| `guides/factor_selection.md` ~66 | `result.F @ result.G` — should be `result.G @ result.F` |
| `guides/rotation.md` ~93 | `sweep.results[fpeak_val].F[:, k]` — should be `.F[k, :]` |

---

## API Examples That Won't Work

### `result.factor_count_posterior` doesn't exist as an attribute

**Files:** `guides/bayesian.md` lines ~54, 105, 116, 128; `guides/factor_selection.md` line ~133

`factor_count_posterior` is a standalone function in `bayes_diagnostics.py`, not an attribute on `BayesPMFResult`. Correct usage:

```python
from pmf_acls import factor_count_posterior
fcp = factor_count_posterior(result.factor_activity_samples)
```

### `result.n_active_factors` doesn't exist

**File:** `guides/bayesian.md` line ~127

The actual attribute is `result.effective_p`.

### `result.factor_alpha` doesn't exist

**File:** `guides/bayesian.md` line ~132

ARD attributes are `result.ard_lambda_F` and `result.ard_lambda_G`.

### `gelman_rubin(result.G_samples)` — wrong signature

**File:** `guides/bayesian.md` line ~88

`gelman_rubin(*chains)` expects two or more 1-D traces, not a 3-D sample array. Correct:

```python
results = [pmf_bayes(X, sigma, p, random_seed=s) for s in range(4)]
rhat = gelman_rubin(*[r.Q_samples for r in results])
```

### `effective_sample_size(result.G_samples)` — wrong input type

**File:** `guides/bayesian.md` line ~102

Expects a 1-D trace `(n_samples,)`, not a 3-D array. Use `result.Q_samples`.

### `multistart_test(..., n_seeds=20)` — wrong parameter name

**File:** `guides/uncertainty.md` line ~68

The parameter is `n_displacements`, not `n_seeds`. Return type is `dict`, not a named result class. Attribute references like `ms.stability_metric` and `ms.swap_counts[k]` won't work on a dict.

### `compute_waic(...)` returns a dict, not a scalar

**File:** `guides/factor_selection.md` line ~133

`waics[p] = compute_waic(...)` stores a dict. Needs `compute_waic(...)['waic']`.

---

## Implementation Mismatches

### `aitchison_pmf` CLR pipeline description doesn't match code

**File:** `guides/compositional.md` lines ~83-91

Describes a 4-step pipeline: "Transform to CLR space → Weight → Factorize in CLR → Inverse transform." The actual code (`coda.py`) minimizes a weighted Aitchison cost via projected gradient descent on the original non-negative factors using per-observation centered log-residuals. It does not perform an explicit CLR→factorize→inverse pipeline.

### `result.F_std` on aitchison_pmf result is None

**File:** `guides/compositional.md` line ~105

`aitchison_pmf()` returns `PMFResult` but does not compute analytical uncertainty. `result.F_std` will be `None`.

---

## Minor: `api/uncertainty.md` Missing `displacement_test_F`

**File:** `api/uncertainty.md`

Only documents `displacement_test` (G-element displacement). Should also document `displacement_test_F` (F-element displacement, Paatero 2014).
