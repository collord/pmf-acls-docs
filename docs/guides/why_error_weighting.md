# Why Error Weighting Matters

## The Core Problem

Environmental measurements are not all equally trustworthy. A trace element measured via X-ray fluorescence at concentrations well above its detection limit carries far more information than the same element measured near the limit with large relative uncertainty. Yet standard matrix factorization (NMF) treats all measurements equally, implicitly assuming each data point has the same variance—a false and damaging assumption.

**Positive Matrix Factorization (PMF) solves this by incorporating a per-element uncertainty matrix $\sigma_{ij}$, the standard deviation of measurement $X_{ij}$.** Each residual is divided by its uncertainty before squaring:

$$Q = \sum_{i,j} \left[\frac{X_{ij} - (FG)_{ij}}{\sigma_{ij}}\right]^2$$

Precise measurements (small $\sigma_{ij}$) exert more influence. Noisy measurements (large $\sigma_{ij}$) contribute less. This heteroscedastic weighting is not a cosmetic refinement—it is constitutive of the method. The question is not whether to use error information, but how much precision you can justify.

## The Knowledge-Level Spectrum

Different datasets support different error-specification strategies. The appropriate method depends on **how much you actually know about measurement reliability**:

| Knowledge Level | What you know | Appropriate method | Example |
|---|---|---|---|
| **Level 1** | Species-level patterns only | Unweighted NMF + feature scaling | Novel analyte, no QA history, non-routine monitoring |
| **Level 2** | Dominant heteroscedasticity structure | LS-NMF with coarse weight classes | Water quality monitoring, soil surveys, mixed-method data |
| **Level 3** | Equation-based observation-level estimates | PMF with *mandatory sensitivity testing* | CSN/IMPROVE PM2.5 (XRF, IC with decades of QA) |
| **Level 4** | Replicate-validated uncertainties | PMF with full diagnostic suite | AMS with Poisson error model, high-end speciation |
| **Level 5** | Want to learn error structure | Bayesian NMF with sigma learning | Research-driven, novel datasets, heteroscedastic exploration |

**Critical revision:** The field often assumes all environmental monitoring is Level 3-4, but this is wrong for many applications. Speciated PM₂.₅ from EPA's CSN and IMPROVE networks uses X-ray fluorescence and ion chromatography with uncertainties characterized through decades of QA programs, interlaboratory comparisons, and certified reference materials—genuine Level 3. But water quality monitoring from different agencies, soil contamination surveys, and novel analytes are often Level 1-2, where observation-level weighting is unjustified overspecification.

## The "Off by 2× vs. 100×" Argument

Even if your equation-based uncertainty estimates are off by a factor of 2, they capture something important: **measurements near the detection limit are far less reliable than measurements well above it.** The ratio between near-MDL and above-MDL concentrations can easily be 100:1 in magnitude, but the relative uncertainty can differ by an order of magnitude as well. Ignoring this structure (Level 0: uniform weighting) is off by 100×. Using coarse weight classes (Level 2: "near-MDL" vs. "above-MDL") is off by ~2×. The difference is substantial.

This is why **Level 2 is often the sweet spot** when you lack precise QA characterization. Two weight classes capture the dominant heteroscedasticity structure and require no detailed analytical justification.

## The Sensitivity-Testing Imperative

Regardless of knowledge level, **you must test sensitivity to error specification.** Run your factorization with two or three alternative uncertainty specifications:

1. Your best estimate (e.g., equations from analytical methods)
2. A conservative overestimate (wider error bars → lower weighting on noisy data)
3. A less stringent specification (coarser weight classes or uniform weights)

If the resolved factor structures **differ qualitatively** across these specifications (factors appear/disappear, profiles change), your knowledge level is lower than you assumed. Retreat to a simpler method.

If the factor structures **agree**, you have genuine evidence that the data support the solution independent of error specification details.

**This is the field's failure mode:** Not using error information, but using it without sensitivity testing. The choice between PMF and unweighted NMF is less important than testing whether the choice matters for your conclusions.

## Q/Qexp as a Noise Model Check

The statistic $Q / Q_{\text{exp}}$ (your weighted objective divided by the degrees-of-freedom-adjusted expected value) is often misused for factor count selection. This is wrong. **Q/Qexp is a noise model validation metric, not a factor selection metric.** It answers: "Are my stated uncertainties consistent with the observed residuals?"

- $Q / Q_{\text{exp}} \approx 1$: Your uncertainties are well-calibrated to the data. The model fits at the noise level you claimed.
- $Q / Q_{\text{exp}} \gg 1$: Either the model is inadequate (missing factors) or your uncertainties are too small (overconfident).
- $Q / Q_{\text{exp}} \ll 1$: Either you have too many factors or your uncertainties are too large (underconfident).

**But:** Two models with different factor counts can both have $Q / Q_{\text{exp}} \approx 1$. Q alone does not determine which is "right." For factor count selection, use the Diagnostic Discrepancy Principle in {doc}`factor_selection` — a separate, uncertainty-independent metric.

## Practical Guidance

1. **Assess your knowledge level honestly.** Do you have published analytical QA characterization? Interlaboratory comparison data? Replicate analyses? If not, Level 2-3 is more realistic than Level 4.

2. **Don't skip coarse error classes.** If you lack detailed QA, use two weight classes: one for high-quality data (XRF, internal standards) and one for trace/estimated values. This is Level 2 and captures the dominant structure.

3. **Always test sensitivity.** Run with your best error estimate, then re-run with 2-3× uncertainty bands. If conclusions change, flag it as uncertain.

4. **Use Q/Qexp to validate noise assumptions, not to select factors.** If $Q / Q_{\text{exp}} \gg 1$, your uncertainties may be too optimistic. If $Q / Q_{\text{exp}} \ll 1$, you may be overfitting—but the answer to "overfitting how much?" requires a separate metric.

5. **Report uncertainty sources.** When publishing factor profiles and contributions, document your uncertainty specification (equation-based, replicate-validated, coarse classes, etc.) so readers can judge credibility.

---

## References

- Polissar, A. V., et al. (1998). PMF and EM for source apportionment of oceanic fine particles. *Journal of Geophysical Research*, 103(D3), 3601–3609.
- Norris, G. A., et al. (2014). EPA Positive Matrix Factorization (PMF) 5.0 Fundamentals and User Guide. EPA/600/R-14/108.
- Reff, A., Eberly, S. I., & Bhave, P. V. (2007). Receptor modeling of ambient particulate matter in Memphis, Tennessee. *Journal of the Air & Waste Management Association*, 57(5), 594–605.
