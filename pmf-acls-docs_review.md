# Critical Review: `pmf-acls-docs` Documentation

**Repository:** `github.com/collord/pmf-acls-docs` (branch: `main/docs`)
**Reviewer scope:** Factual accuracy, bibliographic integrity, and identification of underpinning conjectures. Source-code alignment is excluded per request.
**Date:** 26 March 2026

---

## 1. Summary of the Documentation

The `pmf-acls` documentation describes a Python package for Positive Matrix Factorization (PMF) and Non-negative Matrix Factorization (NMF) with heteroscedastic error weighting. It positions itself as a complement to EPA PMF 5.0 and ESAT, offering five solvers (ACLS, LS-NMF, Newton, Bayesian Gibbs, and an LDA variant), full Bayesian uncertainty quantification with ARD-based factor count inference, and support for compositional data analysis via Aitchison geometry. The documentation spans seven guide pages, a quickstart, API stubs, and a references file.

Overall, the conceptual exposition is unusually strong for package documentation: the treatment of error weighting philosophy, the separation of structural vs. noise-model diagnostics for factor selection, and the honest-limitations sections in the Bayesian guide reflect genuine domain expertise. However, the documentation contains a number of verifiable factual errors—several of which are consequential—alongside conjectures presented as established facts.

---

## 2. Factual Errors

### 2.1 CLR Is Not a Bijection to R^(D−1)

**Location:** `guides/compositional.md`, lines 76–79

The documentation states:

> **Bijection:** The CLR is a one-to-one map from the simplex to R^{D−1}. No information is lost.

This is incorrect on two counts. The Centered Log-Ratio transform takes a D-part composition and produces a D-dimensional vector (not D−1), whose components sum to zero. Its image is therefore a (D−1)-dimensional hyperplane embedded in R^D. Critically, the CLR is *not injective* as a map into R^D: the `compositions` R package documentation (maintained by the CoDa community) states explicitly that "the clr-transform maps a composition in the D-part Aitchison-simplex isometrically to a D-dimensional Euclidean vector subspace: consequently, the transformation is not injective. Thus resulting covariance matrices are always singular." The transform that *is* a bijection to R^{D−1} is the ILR (Isometric Log-Ratio) transform introduced by Egozcue et al. (2003)—which the documentation itself cites in the references but apparently confuses with CLR.

**Severity:** High. This is a foundational mathematical claim in the compositional data chapter. Users who factorize in CLR space relying on full-rank covariance matrices will encounter singular systems.

### 2.2 Paatero (1993) Date Error

**Location:** `guides/factor_selection.md`, line 151

The citation reads:

> Paatero, P. (1993). Least squares formulation of robust non-negative factor analysis. *Chemometrics and Intelligent Laboratory Systems*, 37(1), 23–35.

Volume 37 of *Chemometrics and Intelligent Laboratory Systems* was published in **1997**, not 1993. The paper was received July 1996 and accepted March 1997. Paatero's 1993 work (with Tapper) is a different paper: "Analysis of different modes of factor analysis as least squares fit problems" in *CILS* 18, 183–194. The rotation guide (`guides/rotation.md`, line 152) correctly dates this same paper to 1997, creating an internal inconsistency.

**Severity:** Medium. Incorrect citation dates undermine traceability and may cause confusion when readers attempt to locate the original source.

### 2.3 Fabricated Brouwer et al. Reference

**Location:** `guides/bayesian.md`, line 204

The citation reads:

> Brouwer, T., Frellsen, J., & Liò, P. (2017). Variational auto-encoded deep Gaussian processes. In *International Conference on Learning Representations*.

The actual 2017 paper by these authors is titled "Comparative Study of Inference Methods for Bayesian Nonnegative Matrix Factorisation," published at ECML PKDD 2017 (Lecture Notes in Computer Science, vol. 10534, Springer). Both the title and the venue are wrong. "Variational auto-encoded deep Gaussian processes" appears to be a hallucinated or confused title—no paper with that exact name appears in the ICLR proceedings.

**Severity:** High. This is a wholly incorrect citation that no reader could locate.

### 2.4 Wrong Authors and Title for Wang et al.

**Location:** `guides/algorithms.md`, line 140

The reference list cites:

> Wang, F., Li, T., & Wang, X. (2006). Concept decomposition for large sparse text data using clustering. *ACM SIGKDD Explorations*, 10(2), 42–51.

This paper is about text-data clustering and has nothing to do with LS-NMF. The correct LS-NMF paper—which the `references.bib` file does cite correctly—is Wang, G., Kossenkov, A.V., Bhatt, N.N., & Ochs, M.F. (2006), "LS-NMF: A modified non-negative matrix factorization algorithm utilizing uncertainty estimates," *BMC Bioinformatics*, 7, 175. Wrong authors, wrong title, wrong journal.

**Severity:** High. Readers following this reference will find an unrelated paper.

### 2.5 Langville et al. Date Mismatch

**Location:** `guides/algorithms.md`, line 139; `references.bib`, line 12

The algorithms guide cites "Langville et al. (2014)" in *Journal of Mathematical Modeling and Algorithms*, 5(4), 629–662. The `references.bib` describes it as "arXiv preprint arXiv:1407.7299." Volume 5 of *JMMA* corresponds to approximately 2006, not 2014. The documentation appears to conflate the arXiv preprint date (2014) with the journal publication metadata (which is from an earlier volume).

**Severity:** Low-medium. The paper is locatable, but the mixed provenance is sloppy.

### 2.6 FPEAK Formula Is a Simplification Not Found in Paatero

**Location:** `guides/rotation.md`, lines 19–20

The documentation presents the FPEAK objective as:

$$Q_{\text{FPEAK}} = Q + \text{FPEAK} \cdot \sum_k \left(\sum_i F_{ik}\right)^2$$

This is a pedagogical simplification. Paatero's FPEAK mechanism, as described in the 1997 *CILS* paper and explained in Hopke's EPA guide, operates by forcing additions of one G-vector to another and subtracting the corresponding F-factors—a controlled rotation of the factor space, not a simple additive penalty on profile column sums. The actual rotation mechanism involves the full joint Gauss-Newton system and has never been fully published (as noted by recent literature: "To date, Paatero's exact algorithmic approach to solving [the FPEAK objective] remains unpublished"—Geoscientific Model Development, 2025). Presenting the simplified formula without qualification may mislead users into thinking they can reimplement FPEAK from this equation alone.

**Severity:** Medium. The simplified formula captures the qualitative direction of FPEAK but misrepresents the actual mechanism.

### 2.7 Newton Solver System Size Description

**Location:** `guides/algorithms.md`, lines 56–58

The documentation states the Newton solver "builds and solves the full (mp + np) × (mp + np) normal system at each iteration." Paatero's PMF2 does not build a single monolithic dense system of this size. The 1997 paper describes a Gauss-Newton approach that modifies all elements simultaneously but through structured block updates. The full system description would imply O((mp+np)³) per iteration, which is far more expensive than what PMF2 actually does.

**Severity:** Low-medium. The qualitative point (Newton is expensive and scales poorly) is correct, but the dimensionality claim is inaccurate.

### 2.8 EPA PMF 5.0 Described as "Proprietary"

**Location:** `index.md`, line 22

The text reads: "EPA PMF 5.0 (Paatero's proprietary engine) was the standard." EPA PMF 5.0 was freely distributed by the EPA and is still available for download. What is proprietary and undocumented is the underlying ME-2 (Multilinear Engine v2) algorithm by Paatero. The EPA's own description states the software "is free of charge and does not require a license." Calling the EPA's free tool "proprietary" conflates the application with its internal solver.

**Severity:** Low. The practical point is valid (ME-2 is a black box), but the wording is imprecise.

---

## 3. Conjectures and Assumptions

### 3.1 "Causal Inference About Sources"

**Location:** `index.md`, line 7

> The goal is not just statistical dimension reduction—it is causal inference about sources.

This is a strong philosophical claim. PMF and NMF are correlative decomposition methods. Factor profiles that emerge from the mathematical optimization are interpreted as sources by domain experts, but the factorization itself provides no causal identification in the formal statistical sense (e.g., no intervention, no counterfactual framework, no control for confounders). The receptor modeling literature (e.g., Hopke, 2016) consistently uses the term "source apportionment" rather than "causal inference," and even EPA guidance frames PMF results as requiring external validation against known source profiles. Calling this "causal inference" overstates what the mathematics delivers.

### 3.2 "ACLS" as a Standard Algorithm Name

**Throughout the documentation**

The documentation presents "ACLS" (Alternating Constrained Least Squares) as a recognized algorithm in the PMF/NMF literature. In the broader community, the standard term is ALS (Alternating Least Squares), sometimes qualified as "ANLS" (Alternating Non-negative Least Squares) when non-negativity is enforced. The "C" prefix appears to be package-specific terminology. This is not necessarily wrong—naming conventions vary—but readers should be aware that searching the literature for "ACLS" in the context of NMF will yield few results outside this package.

### 3.3 The "Diagnostic Discrepancy Principle" as Established Methodology

**Location:** `guides/factor_selection.md`

The factor selection guide introduces the "Diagnostic Discrepancy Principle," which prescribes using two independent metrics (a structural elbow test and Q/Qexp) and interpreting their agreement or disagreement. While the individual components are well-established (scree/elbow tests date to Cattell, 1966; Q/Qexp is standard EPA PMF practice), their specific combination under this name and the interpretive framework (the discrepancy table in lines 94–98) appear to be novel to this package. The documentation presents this as established methodology without citing a peer-reviewed source for the combined principle. Users should understand this is a reasonable but author-proposed heuristic, not a validated statistical criterion.

### 3.4 Bayesian λ_G and FPEAK "Encode the Same Rotation Preference"

**Location:** `guides/rotation.md`, lines 138–139

> Bayesian λ_G and ACLS FPEAK encode the same rotation preference, just in different languages.

This equivalence is approximate at best. FPEAK is a deterministic rotation control that modifies the objective surface of a constrained optimization problem. λ_G is a prior hyperparameter in a probabilistic model whose effect is mediated through posterior sampling. While both can qualitatively push solutions toward peaked or diffuse profiles, they operate through fundamentally different mathematical mechanisms. Small λ_G (strong exponential prior) favors sparsity through the prior density, not through the geometric rotation mechanism of FPEAK. The claim of equivalence could mislead users into expecting that tuning λ_G will replicate the behavior of an FPEAK sweep.

### 3.5 Posterior "Cannot Address" Rotational Ambiguity

**Location:** `guides/bayesian.md`, line 175

> Bayesian posterior (joint uncertainty) ... Cannot address: Rotational ambiguity (rotation manifold has measure zero)

The claim that the rotation manifold has measure zero under the posterior is technically correct in a narrow sense (for a fixed rotation matrix T, the set {T} has measure zero), but it understates the practical situation. In Bayesian NMF, label switching and multimodality—which are manifestations of rotational freedom—are well-documented challenges that the posterior *does* partially explore. The exponential prior itself breaks some rotational symmetry, and hierarchical learning of λ implicitly regularizes toward particular rotations. Stating that the posterior "cannot address" rotational ambiguity as a categorical limitation is an oversimplification. A more accurate statement would be that the posterior does not systematically enumerate rotations the way FPEAK sweeps do.

### 3.6 Delta-Method Weight Formula for CLR

**Location:** `guides/compositional.md`, line 87

The weight formula for Aitchison NMF is given as:

$$w_{ij} = \frac{X^2_{ij}}{\sigma^2_{ij}}$$

This is presented without derivation and described as the delta-method propagation of per-element uncertainties into CLR space. While the delta method applied to a log transform does yield a variance proportional to σ²/X² (and hence weights proportional to X²/σ²), the CLR transform involves a *geometric mean* denominator that introduces cross-covariance terms between components. The formula as stated assumes independent measurement errors and ignores the off-diagonal terms in the Jacobian of the CLR transform. For compositions where measurement errors are correlated (common in XRF or ICP-MS due to matrix effects), this approximation may be poor. The documentation should note this is a diagonal approximation to the full delta-method covariance.

### 3.7 LS-NMF Monotone Decrease "Guarantee"

**Location:** `guides/algorithms.md`, lines 36–38

> **Monotone decrease guarantee:** Each iteration provably reduces Q. If you need to document that the solver didn't get stuck in a saddle point, LS-NMF provides formal guarantees.

Multiplicative update rules for NMF (Lee & Seung, 2001) are known to guarantee monotonic non-increase of the objective, but this does not exclude convergence to saddle points. The claim that the guarantee helps "document that the solver didn't get stuck in a saddle point" is misleading: monotone decrease guarantees convergence to a stationary point, which may be a saddle point or local minimum. The NMF landscape is known to be non-convex with many stationary points. The guarantee is about descent direction, not about solution quality.

---

## 4. Minor Issues

**Notation inconsistency:** The index page defines X as "(species × observations)" but the standard PMF convention (Paatero & Tapper, 1994) uses X as (observations × species), i.e., n × m where n = samples and m = species. The original paper defines G as the left (n × p) "scores" matrix and F as the right (p × m) "loadings" matrix. The documentation's transposition could cause confusion when comparing against EPA PMF outputs.

**Geweke z-score mentioned but not explained:** The bayesian guide (line 93) mentions "Geweke z-score" as a convergence diagnostic but never defines or discusses it, despite the section heading "Convergence Diagnostics: How to Read Them." The Gelman-Rubin statistic is discussed instead (Rhat), which is a multi-chain diagnostic, while Geweke is a single-chain diagnostic. These are not interchangeable.

**Bootstrap description omits block structure detail:** The uncertainty guide states bootstrap uses "block resampling to preserve temporal/spatial structure if present" (line 7), but provides no guidance on block size selection—a critical parameter that determines whether the bootstrap captures autocorrelation structure. EPA PMF 5.0 uses a specific block bootstrap protocol; this should be referenced.

---

## 5. Bibliographic Audit Summary

| Citation | Issue | Severity |
|----------|-------|----------|
| Paatero (1993) in factor_selection.md | Wrong year; should be 1997 | Medium |
| Brouwer et al. (2017) in bayesian.md | Wrong title and wrong venue | High |
| Wang et al. (2006) in algorithms.md | Wrong authors, title, and journal | High |
| Langville et al. (2014) in algorithms.md | Year/volume mismatch | Low-medium |
| Egozcue et al. (2003) in compositional.md | Correct | — |
| Paatero & Tapper (1994) in references.bib | Correct | — |
| Schmidt et al. (2009) in references.bib | Correct | — |
| Wang et al. (2006) in references.bib | Correct (different from algorithms.md inline) | — |
| Norris et al. (2014) in references.bib | Correct | — |
| Paatero (1997) in rotation.md | Correct date, but page range 23–35 should be verified against actual pagination | Low |

---

## 6. Overall Assessment

The documentation is conceptually sophisticated and reflects genuine expertise in environmental source apportionment. The treatment of error-weighting philosophy, the honest-limitations discussion of Bayesian inference, and the separation of structural vs. noise-model diagnostics are commendable. However, the work is undermined by a pattern of bibliographic errors that suggest the reference lists were not systematically verified against the primary sources. Two of the inline citations (Brouwer, Wang) are wholly incorrect, and the CLR bijection error is a mathematical mistake with practical consequences.

The documentation also introduces several novel frameworks (Diagnostic Discrepancy Principle, FPEAK–λ_G equivalence) without clearly distinguishing them from established methodology. These frameworks are reasonable but should be explicitly flagged as package-specific proposals rather than community consensus.

**Recommended actions:**

1. Correct all bibliographic errors identified in Section 5.
2. Fix the CLR description: it maps to a (D−1)-dimensional hyperplane in R^D, not a bijection to R^{D−1}. Consider recommending ILR where full-rank representations are needed.
3. Qualify the FPEAK formula as a pedagogical simplification.
4. Flag the Diagnostic Discrepancy Principle, ACLS nomenclature, and FPEAK–λ_G equivalence as package-specific constructs.
5. Replace "causal inference" with "source apportionment" in the introductory framing.
6. Add a note on the diagonal approximation in the delta-method weight derivation for CLR.

---

*Review prepared by cross-referencing the documentation against: Paatero & Tapper (1994, Environmetrics); Paatero (1997, CILS); Hopke (2000, EPA Guide to PMF); Norris et al. (2014, EPA PMF 5.0 User Guide); Brown et al. (2015, Sci. Total Environ.); Wang et al. (2006, BMC Bioinformatics); Schmidt et al. (2009, ICA/SSS); Brouwer et al. (2017, ECML PKDD); Egozcue et al. (2003, Math. Geol.); Aitchison (1986, Chapman & Hall); the R `compositions` package documentation; the ESAT GitHub repository (quanted/esat); and Geoscientific Model Development (2025).*
