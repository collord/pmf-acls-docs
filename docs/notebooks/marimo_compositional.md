# Compositional Analysis

Methods for compositional data: simplex constraints and Aitchison geometry.

## Compositional Data Analysis

When sources are defined as fractional end-member abundances (e.g., sediment fingerprinting, spectral unmixing, geological mixtures), standard PMF works in the wrong mathematical space. This notebook introduces the geometry problem: why Euclidean distance conflates magnitude with composition, and how the Aitchison geometry (via CLR transform) recovers the correct structure. See how sample composition is preserved under scale-invariance and why heteroscedastic weights distort under post-hoc simplex projection.

```{marimo} 13_compositional_demo.py
:height: 800px
:click-to-load: overlay
:load-button-text: Launch Compositional Demo
```

## Simplex-Constrained PMF

A hard constraint approach: force contributions to sum exactly to 1 (the simplex) during optimization using constrained NNLS solvers. Learn when hard constraints are necessary (regulatory requirements, remote sensing standards) and how `simplex_pmf()` compares to Aitchison NMF. Test how the constraint affects profile recovery and compare factorization quality when the simplex is physically required vs. when geometric methods are preferred.

```{marimo} 14_simplex_pmf_demo.py
:height: 800px
:click-to-load: overlay
:load-button-text: Launch Simplex PMF Demo
```

## Aitchison NMF

The principled geometric approach: work entirely in CLR (Centered Log-Ratio) space where composition is separated from magnitude, apply delta-method weights to preserve heteroscedastic uncertainty, and recover profiles as valid simplex compositions. Explore how the `anchor` parameter balances geometric purity against minor-source sensitivity—essential when trace species are scientifically important. Compare recovered profiles and uncertainty structure to Euclidean PMF and hard simplex methods.

```{marimo} 15_aitchison_nmf_demo.py
:height: 800px
:click-to-load: overlay
:load-button-text: Launch Aitchison NMF Demo
```
