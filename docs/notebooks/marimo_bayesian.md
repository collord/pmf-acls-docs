# Bayesian NMF

Posterior uncertainty, hierarchical models, priors, ARD pruning, and robust estimation.

## Bayesian PMF

Introductory Bayesian inference: run Gibbs sampling to generate posterior uncertainty on factor profiles and contributions. Compare warm-start augmentation of ACLS vs. full Bayesian factor determination with ARD. Understand convergence diagnostics and factor count posteriors.

```{marimo} 05_bayesian_demo.py
:height: 900px
:click-to-load: overlay
:load-button-text: Launch Bayesian PMF Demo
```

## PCB Hierarchical Model

Advanced hierarchical Bayesian modeling: place hyperpriors on the noise level (sigma learning) so that measurement uncertainties are estimated from data rather than assumed fixed. Learn when and how sigma learning improves results.

```{marimo} 06_pcb_hierarchical_demo.py
:height: 900px
:click-to-load: overlay
:load-button-text: Launch PCB Hierarchical Demo
```

## Shrinkage & Truncation Bias

The Bayesian posterior mean is shrunk toward the origin compared to the maximum-likelihood (ACLS) solution. Explore this shrinkage effect: when is it desirable (noise mitigation)? When is it problematic (bias on true signals)? Understand the truncated-exponential prior.

```{marimo} 07_shrinkage_exploration.py
:height: 800px
:click-to-load: overlay
:load-button-text: Launch Shrinkage Exploration
```

## Robust Outlier Detection

Replace the Gaussian likelihood with a Student-t distribution (heavier tails) to downweight outliers without losing sensitivity to true signals. Learn when robust inference helps and when it masks model inadequacy.

```{marimo} 08_robust_outlier_demo.py
:height: 800px
:click-to-load: overlay
:load-button-text: Launch Robust Outlier Demo
```

## Volume Prior

A geometric prior (volume prior on F, also called the "proper" volume prior) prevents factors from collapsing to zero or becoming redundant. Explore how the volume prior stabilizes factor identifiability in high-dimensional problems.

```{marimo} 09_volume_prior_demo.py
:height: 800px
:click-to-load: overlay
:load-button-text: Launch Volume Prior Demo
```

## ARD Factor Pruning

Automatic Relevance Determination learns which factors are "active" (high precision) vs. "pruned" (low precision). Visualize how ARD infers the factor count posterior. Test sensitivity to the pruning threshold.

```{marimo} 10_ard_pruning_demo.py
:height: 800px
:click-to-load: overlay
:load-button-text: Launch ARD Pruning Demo
```

## Informative Priors

Use trusted external profiles (e.g., EPA SPECIATE, literature values) as Bayesian priors to regularize the solution. Learn how to set prior strength (`F_prior_scale`) to balance trust in the prior vs. letting data dominate.

```{marimo} 11_informative_prior_demo.py
:height: 800px
:click-to-load: overlay
:load-button-text: Launch Informative Prior Demo
```
