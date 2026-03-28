# Adversarial Review: `pmf-acls-docs` Documentation

**Repository:** `github.com/collord/pmf-acls-docs` (branch: `main/docs/`)
**Scope:** Factual accuracy, underpinning conjectures, bibliographic integrity. Source-code alignment excluded per request.
**Date:** 28 March 2026

---

## Preamble: Context and Prior Review

This repository already contains a review (`pmf-acls-docs_review.md`) and a detailed author response (`REVIEW_RESPONSE.md`), both dated 26 March 2026. The prior review identified seven factual/bibliographic errors and seven conjectures. The author accepted most corrections and committed to specific fixes. This second-pass review (a) independently verifies the prior review's claims against primary sources, (b) identifies issues the prior review missed, (c) evaluates whether the author's proposed corrections are sufficient, and (d) provides additional adversarial pressure on the documentation's theoretical claims.

Note: as of this review date, neither the prior review's corrections nor the author's proposed fixes appear to have been applied to the actual documentation files. The docs themselves remain in their original state.

---

## Part 1: Verification of Prior Review Findings

### 1.1 CLR Bijection Claim — CONFIRMED ERROR

The prior review correctly identifies that CLR maps a D-part composition to a D-dimensional vector whose components sum to zero, producing a (D−1)-dimensional hyperplane in ℝ^D. The CLR is not a bijection to ℝ^{D−1}. The `compositions` R package documentation and Egozcue et al. (2003) are clear on this. The ILR is the bijection.

**Author's proposed fix** (adding a note about ILR) is adequate but could go further. The documentation should explicitly warn that CLR covariance matrices are *necessarily singular*, which matters when users try to compute Mahalanobis distances or run multivariate tests on CLR-transformed data. The current documentation's `aitchison_nmf()` function operates in CLR space — the singularity is not a problem for factorization (which works on the hyperplane), but the docs should explain *why* it's not a problem, rather than leaving users to discover it.

### 1.2 Paatero (1993) vs. (1997) — CONFIRMED ERROR

Independently verified. Volume 37 of *Chemometrics and Intelligent Laboratory Systems* was published in 1997. The paper "Least squares formulation of robust non-negative factor analysis" (Paatero, 1997) is correctly dated in `rotation.md` but incorrectly as 1993 in `factor_selection.md`. The 1993 Paatero & Tapper paper is "Analysis of different modes of factor analysis as least squares fit problems" in *CILS* 18, a different work entirely.

### 1.3 Brouwer et al. Citation — CONFIRMED FABRICATION

The title "Variational auto-encoded deep Gaussian processes" and venue "ICLR" are both wrong. The correct paper is "Comparative study of inference methods for Bayesian nonnegative matrix factorisation" at ECML PKDD 2017 (LNCS vol. 10534). No paper with the cited title appears in any ICLR proceedings. This is a hallucinated citation.

### 1.4 Wang et al. Citation — CONFIRMED ERROR

The inline citation in `algorithms.md` points to a text-clustering paper (Wang, Li & Wang, 2006, *ACM SIGKDD*), not the LS-NMF paper (Wang, Kossenkov, Bhatt & Ochs, 2006, *BMC Bioinformatics*). The `references.bib` has the correct entry. Two different "Wang et al. (2006)" papers were confused.

### 1.5 Langville et al. Date — CONFIRMED AMBIGUITY

The `references.bib` calls it "arXiv preprint arXiv:1407.7299" (2014), while the algorithms guide suggests a journal publication. The arXiv preprint is from 2014; the published version appeared in *KDD Explorations* (2006). The documentation conflates these.

### 1.6 FPEAK Formula Simplification — CONFIRMED, WITH ADDITIONAL CONCERNS (see §2.1)

### 1.7 EPA PMF 5.0 "Proprietary" — CONFIRMED IMPRECISION

EPA PMF 5.0 is explicitly described by the EPA as "free of charge and does not require a license." What is proprietary and undocumented is the ME-2 solver engine. The EPA's own Science Inventory page states ESAT was developed as a replacement precisely because PMF5 "relies on the proprietary Multilinear Engine v2 (ME2) that lacks documentation."

---

## Part 2: New Factual Issues Not Identified in Prior Review

### 2.1 FPEAK Formula Is Not Just Simplified — It Describes a Different Mechanism

**Location:** `guides/rotation.md`, lines 19–25

The prior review flagged this as a "pedagogical simplification." This understates the problem. The formula presented:

$$Q_{\text{FPEAK}} = Q + \text{FPEAK} \cdot \sum_k \left(\sum_i F_{ik}\right)^2$$

describes an L2 penalty on factor profile column sums. This is not what FPEAK does. Hopke's EPA guide (2000) states that FPEAK works by "forc[ing] additions of one G vector to another and subtract[ing] the corresponding F factors from each other." The 2025 *Geoscientific Model Development* paper (Massoli et al. dataset analysis) explicitly states: "To date, Paatero's exact algorithmic approach to solving [the FPEAK objective] remains unpublished." Paatero (1997) describes FPEAK as part of a controlled rotation within the Gauss-Newton iteration, not as an additive penalty term.

The formula in the docs describes something more like Tikhonov regularization on profile norms. A user who implements this formula will get regularized NMF with a sparsity preference — *not* the rotation exploration that FPEAK is designed to perform. This is not a simplification; it is a misdescription of the mechanism.

**Severity:** High. Users interpreting FPEAK sweep results through the lens of this formula will draw incorrect conclusions about what the parameter is doing.

**Recommended fix:** State explicitly that the formula is an *analogy* for FPEAK's qualitative direction, not a description of its mechanism. Add: "The actual FPEAK mechanism operates within the Gauss-Newton iteration by controlled additions and subtractions of factor vectors, a process that has never been fully published."

### 2.2 ESAT Uses LS-NMF, Not "LS-PMF"

**Location:** `guides/algorithms.md`, line 32; `index.md`, line 22

The documentation refers to the ESAT algorithm as "LS-PMF" in multiple places. ESAT's own GitHub repository (`quanted/esat`) describes its algorithms as "LS-NMF: Least-squares NMF" and "WS-NMF: Weight-Semi NMF." The term "LS-PMF" does not appear in the ESAT documentation or codebase. The original algorithm paper (Wang et al., 2006, *BMC Bioinformatics*) titles it "LS-NMF." The documentation's use of "LS-PMF" appears to be a package-specific renaming that could confuse users searching for the algorithm in the literature or ESAT's own docs.

**Severity:** Low-medium. The algorithm is the same; the name is non-standard.

### 2.3 Monotone Decrease and Convergence: The Convergence Gap Is Deeper Than Stated

**Location:** `guides/algorithms.md`, lines 36–38

The prior review correctly flagged that monotone decrease doesn't prevent saddle-point convergence. But the issue is deeper. Lee & Seung (2001, NIPS) proved only that the objective is *non-increasing* — not that it strictly decreases. Lin (2007, *IEEE Trans. Neural Networks*) demonstrated that the original Lee-Seung proof is *incomplete* regarding convergence to stationary points. Gonzalez & Zhang (2005) showed numerical examples where the algorithm fails to approach a stationary point. Lin proposed *modified* multiplicative updates that do provably converge to stationary points, but only with additional safeguards (small ε to prevent elements from reaching zero).

The documentation states the algorithm has a "monotone decrease guarantee" without specifying which variant is implemented. If the package uses the original Lee-Seung updates, the guarantee is weaker than stated (non-increase, not strict decrease, and convergence to a stationary point is not guaranteed). If it uses Lin's modified updates, that should be stated.

**Severity:** Medium. The theoretical claim is stronger than what the cited literature supports for the unmodified algorithm.

### 2.4 Newton Solver: "Closest to EPA PMF's Internals" Is Misleading

**Location:** `guides/algorithms.md`, lines 50–55

The documentation states the Newton solver is the "closest to EPA PMF's internals" and has "lineage" to PMF2/ME-2. However, EPA PMF 5.0 uses ME-2, not PMF2. Hopke's guide explains that PMF2 and ME-2 have "differences in the computational approach." ME-2 is a more general multilinear engine that solves the PMF problem as a special case. Claiming that a Gauss-Newton solver in this package reproduces "PMF2/ME-2 behavior" conflates two different solvers with different algorithmic characteristics.

Furthermore, the documentation states Newton is "most mathematically principled" — but principled relative to what criterion? ANLS (alternating non-negative least squares) with exact NNLS subproblem solvers (e.g., Kim & Park, 2008) has stronger convergence guarantees than Gauss-Newton approaches to bilinear problems, because each subproblem is convex.

**Severity:** Low-medium. The qualitative point (Newton is expensive) is correct, but the lineage and "principled" claims overstate the connection.

### 2.5 Exponential Prior ≠ Non-Negativity in the Bayesian Model

**Location:** `guides/bayesian.md`, lines 53–55

The documentation states:

> Draws samples from the posterior distribution P(F, G | X, σ) using exponential priors (non-negativity)

and later:

> The priors are truncated exponential (equivalent to imposing non-negativity)

An exponential prior on F_{ik} is *not* equivalent to imposing non-negativity. An exponential distribution is naturally supported on [0, ∞), so it does enforce non-negativity. But it also introduces a *specific shape*: an exponential prior with rate λ places most mass near zero, creating a strong sparsity-inducing effect. A non-informative non-negativity prior would be a uniform distribution on [0, ∞) (an improper prior) or a half-normal/half-Cauchy with wide scale. The exponential prior encodes a specific belief that factor elements should be small and sparse — this is a substantive modeling choice, not just "imposing non-negativity."

Schmidt et al. (2009) — the paper the docs cite — explicitly discuss this: the exponential prior serves as a sparsity prior, and its rate parameter controls the degree of sparsity. This is relevant because users who think they're just "imposing non-negativity" may not realize they're also imposing sparsity preferences that influence factor structure.

**Severity:** Medium. Mischaracterizing the prior's role could lead users to misinterpret the effect of λ_F and λ_G on their results.

### 2.6 Q/Qexp DOF Formula: Both Variants Are Presented Without Noting Their Disagreement

**Location:** `guides/factor_selection.md`, lines 17–20

The documentation presents two DOF formulas:

- Paatero's: ν = nm − p(n + m − p)
- "EPA simplification": nm − p(n + m)

and notes the latter is "slightly different and less accurate when constraints are active." But the disagreement between these formulas is not slight — it's p² parameters' worth of difference, which matters for small problems. More importantly, *neither formula accounts for active non-negativity constraints*, which the documentation itself acknowledges (point 2 in the "why theory breaks down" section). The docs present the Paatero formula as more accurate without noting that both formulas are theoretical approximations that assume unconstrained optimization.

For a problem with significant active constraints (many zero-valued elements in F or G), the effective DOF could be substantially different from either formula. This is well-recognized in the PMF literature (Brown et al., 2015, *Science of the Total Environment*).

**Severity:** Low-medium. The conceptual point (Q/Qexp is approximate) is correct, but presenting one formula as "more accurate" without qualification is misleading when both are wrong for constrained problems.

### 2.7 ARD "Learns Which Factors Are Active" — Overstated for Non-Convex Models

**Location:** `guides/bayesian.md`, lines 125–130

The documentation states ARD "learns a per-factor precision hyperparameter α_k" and provides "a data-driven factor count, not analyst judgment." This framing suggests ARD reliably discovers the true number of sources. However, for non-convex models like NMF:

- ARD's pruning behavior depends on initialization and sampler mixing. If the Gibbs sampler hasn't mixed well (a common problem in Bayesian NMF due to label switching and multimodality), ARD may prune factors that the sampler hasn't adequately explored.
- The exponential prior already induces sparsity. ARD adds a second layer of sparsity that can interact with the prior in unpredictable ways for multi-modal posteriors.
- Brouwer et al. (2017, ECML PKDD — the correctly cited version) found that different inference methods for Bayesian NMF can yield substantially different factor estimates, suggesting the posterior is sensitive to algorithmic choices.

The documentation does note that ARD "can spuriously prune factors if the data are sparse," but frames ARD factor count as "data-driven" rather than "data-and-model-and-algorithm-driven."

**Severity:** Medium. The honest-limitations section partially mitigates this, but the framing in the main description is too confident.

---

## Part 3: Conjectures and Assumptions — Additional Analysis

### 3.1 The "Knowledge-Level Spectrum" Is an Author Construct, Not Established Practice

**Location:** `guides/why_error_weighting.md`, lines 12–25

The five-level knowledge spectrum (Level 1 through Level 5) is presented with a professional table format that suggests established methodology. No citation is provided. This framework does not appear in the EPA PMF user guide (Norris et al., 2014), Paatero's publications, or the broader receptor modeling literature. The framework is reasonable — it synthesizes genuine practical considerations — but it is an author contribution presented as taxonomy.

The author's response to the prior review addressed the "Diagnostic Discrepancy Principle" as needing a "package-specific" label. The same treatment should apply to the Knowledge-Level Spectrum.

### 3.2 "Level 2 Is Often the Sweet Spot" — An Opinion Presented as Finding

**Location:** `guides/why_error_weighting.md`, lines 37–40

The claim that Level 2 (coarse weight classes) is "often the sweet spot" when you lack precise QA characterization is a value judgment masquerading as a technical finding. Whether two weight classes are sufficient depends entirely on the data and the signal-to-noise ratio of the factors of interest. For datasets where multiple sources contribute at intermediate concentrations (neither near-MDL nor well-above-MDL), two weight classes could miss important structure.

This claim should be framed as the author's recommendation, not as a general finding.

### 3.3 The "Source Apportionment" vs. "Dimension Reduction" Distinction Is Overdrawn

**Location:** `index.md`, line 7

The prior review correctly flagged "causal inference" as overstated, and the author agreed to change it to "source apportionment." However, the broader framing — that PMF's "goal is not just statistical dimension reduction" — also deserves scrutiny.

Mathematically, PMF *is* a dimension-reduction technique with non-negativity constraints and heteroscedastic weighting. The interpretation of factors as sources is imposed by the analyst, not discovered by the algorithm. The same factorization applied to gene expression data would yield "gene programs," not "pollution sources." The algorithm is agnostic to the physical interpretation.

The distinction matters because it sets user expectations. Calling the goal "source apportionment" (rather than "dimension reduction that can be interpreted as source apportionment") may lead users to expect that physically meaningful sources will automatically emerge from the factorization — when in reality, physically meaningful interpretation requires external validation, domain expertise, and often constrained rotation.

### 3.4 Gelman-Rubin Threshold of 1.05 — More Conservative Than Standard Practice

**Location:** `guides/bayesian.md`, lines 85–89

The documentation recommends R̂ < 1.05 for convergence. The original Gelman & Rubin (1992) recommendation was R̂ < 1.2. The widely used threshold of 1.1 comes from Brooks & Gelman (1998). The stricter 1.05 threshold appears in some recent literature (e.g., Vehtari et al., 2021, *Bayesian Analysis*, recommending R̂ < 1.01 for the rank-normalized R̂), but applying the 1.05 threshold to the *original* R̂ statistic (not the rank-normalized version) is non-standard and not attributed. This is a reasonable choice, but it should be cited or flagged as a package convention.

### 3.5 Bootstrap "Model-Light" Framing Understates Bootstrap's Own Assumptions

**Location:** `guides/uncertainty.md`, lines 5–10

The documentation describes bootstrap as "model-light — it doesn't assume a specific error distribution, just resamples the observed data." This is partially true for residual bootstrap but misleading in the PMF context.

The block bootstrap used in EPA PMF *does* depend on the model: it resamples observation blocks and *re-runs the full PMF factorization* on each bootstrapped sample. The resolved factors are then matched to the base solution via alignment algorithms. This process assumes that:

1. The base-solution factor count is correct (bootstrap doesn't test this).
2. Factor matching between bootstrap samples and the base solution is unambiguous (it often isn't for overlapping factors).
3. Block structure captures the relevant autocorrelation (block size selection is acknowledged as important but no guidance is given).

Calling this "model-light" understates the degree to which bootstrap results depend on the base solution and matching algorithm.

---

## Part 4: Assessment of the Author's Proposed Corrections

The `REVIEW_RESPONSE.md` demonstrates genuine engagement with the prior review. Most proposed fixes are appropriate. However, several responses warrant additional scrutiny:

### 4.1 "Items I'm Confident About" — FPEAK Pedagogical Simplification

The author writes: "The simplified formula is pedagogically useful for understanding *intent*... Users looking to reimplement will quickly discover the documentation gap."

This is insufficiently cautious. The formula is not merely simplified — it describes a different operation (L2 penalty on column sums vs. controlled rotation via vector addition/subtraction). Users who *don't* try to reimplement but instead use the formula to interpret their FPEAK sweep results will be misled. The fix should go beyond qualification to explicitly state that the formula does not describe FPEAK's mechanism.

### 4.2 REVIEW_RESPONSE Prepared by "Claude Haiku 4.5"

The author response is attributed to "Claude Haiku 4.5." If the review response was AI-generated, this should be noted as context: an AI model responding to an AI-generated review may exhibit systematic agreement patterns that wouldn't be present in a genuinely adversarial human review. Several of the "I agree with the reviewer" responses accept claims without independent verification (e.g., the Newton solver system-size claim, which the author marks as "likely correct" without checking Paatero 1997).

---

## Part 5: Bibliographic Audit — Updated Summary

| Citation | Status | Detail |
|----------|--------|--------|
| Paatero (1993) in factor_selection.md | **Wrong year** | Should be 1997; CILS vol. 37 |
| Brouwer et al. (2017) in bayesian.md | **Fabricated title/venue** | Correct: ECML PKDD 2017, LNCS 10534 |
| Wang et al. (2006) in algorithms.md | **Wrong paper** | Text clustering, not LS-NMF (BMC Bioinformatics) |
| Langville et al. (2014) in algorithms.md | **Ambiguous** | Conflates arXiv date with journal volume |
| Paatero & Tapper (1994) in references.bib | Correct | Environmetrics 5(2), 111–126 |
| Schmidt et al. (2009) in references.bib | Correct | ICA/BSS 2009, pp. 540–547 |
| Egozcue et al. (2003) in compositional.md | Correct | Math. Geology 35(3), 279–300 |
| Aitchison (1986) in compositional.md | Correct | Chapman and Hall |
| Norris et al. (2014) in references.bib | Correct | EPA/600/R-14/108 |
| Polissar et al. (1998) in why_error_weighting.md | Correct | JGR 103(D3), 3601–3609 |
| Paatero et al. (2014) in uncertainty.md | Correct | AMT 7(3), 781–797 |
| Brown et al. (2015) in uncertainty.md | Correct | Sci. Total Environ. 518–519, 626–635 |
| Gillis & Vavasis (2015) in compositional.md | Correct | IEEE TPAMI 37(11), 2275–2287 |

**Three of thirteen checked citations are wrong.** A ~23% error rate in the bibliography is high for documentation that positions itself as a complement to EPA PMF 5.0 and invites scientific use.

---

## Part 6: Overall Assessment

### What the Documentation Does Well

The documentation is unusually thoughtful for a package of this kind. Several aspects are commendable:

- **Error-weighting philosophy** (`why_error_weighting.md`): The treatment of "when is PMF appropriate vs. plain NMF" is more honest than most PMF tutorials, which assume observation-level uncertainties are always available and well-characterized.
- **Honest limitations of Bayesian inference** (`bayesian.md`): The acknowledgment that "the posterior is tight and wrong" under model misspecification, and that sigma learning can mask model inadequacy, is rare in package documentation.
- **Complementarity table** (`bayesian.md`): The table separating what DISP, bootstrap, Bayesian posterior, and Q/Qexp each address vs. cannot address is pedagogically effective.
- **Compositional data chapter** (`compositional.md`): The distinction between the air-quality PMF tradition and the sediment/geochemistry unmixing tradition reflects genuine cross-disciplinary awareness.

### What the Documentation Gets Wrong

The errors cluster into three categories:

1. **Bibliographic negligence** (3 wrong citations, 1 ambiguous): These appear to result from citation-by-memory or AI-assisted writing without verification. They are easily fixed but damage credibility.

2. **Mechanism misdescription** (FPEAK formula, LS-PMF naming, exponential-prior-as-non-negativity): These are more concerning because they affect how users understand what the algorithms do. The FPEAK formula issue is the most serious: it presents a regularization penalty as a rotation mechanism.

3. **Author constructs presented as established practice** (Knowledge-Level Spectrum, Diagnostic Discrepancy Principle, "Level 2 sweet spot," ACLS nomenclature): The author's proposed fixes for the Diagnostic Discrepancy Principle and ACLS naming are adequate. The Knowledge-Level Spectrum and "sweet spot" claim still need similar treatment.

### Recommendations

1. **Fix all bibliographic errors immediately.** This is non-negotiable for documentation that aspires to scientific credibility.
2. **Rewrite the FPEAK formula section** to clearly state it is an analogy, not a description. Better yet, present the Hopke (2000) verbal description ("additions of one G vector to another, subtracting corresponding F factors") and cite the GMD 2025 paper's statement that "Paatero's exact algorithmic approach remains unpublished."
3. **Clarify the exponential prior's role** as a sparsity-inducing prior, not just a non-negativity constraint.
4. **Label all author-originated frameworks** (Knowledge-Level Spectrum, Diagnostic Discrepancy Principle, ACLS nomenclature) as package-specific constructs on first mention.
5. **Add convergence-variant specificity** to the LS-PMF/LS-NMF section: which variant of multiplicative updates is implemented, and does it have stationary-point convergence guarantees?
6. **Use the literature-standard name "LS-NMF"** rather than "LS-PMF" to match ESAT's own documentation and Wang et al. (2006).

---

*Sources consulted: Paatero & Tapper (1994, Environmetrics); Paatero (1997, CILS 37); Hopke (2000, EPA Guide to PMF, available at people.clarkson.edu/~phopke/PMF-Guidance.htm); Norris et al. (2014, EPA PMF 5.0 User Guide, EPA/600/R-14/108); EPA PMF landing page (epa.gov/air-research/positive-matrix-factorization-model-environmental-data-analyses); Brown et al. (2015, Sci. Total Environ.); Wang et al. (2006, BMC Bioinformatics 7, 175); Schmidt et al. (2009, ICA/BSS); Brouwer et al. (2017, ECML PKDD, LNCS 10534); Egozcue et al. (2003, Math. Geology); Lee & Seung (2001, NIPS); Lin (2007, IEEE Trans. Neural Networks 18(6)); Gonzalez & Zhang (2005, Rice University Tech. Report); GMD 2025 (gmd.copernicus.org/articles/18/2891/2025/); ESAT repository (github.com/quanted/esat); EPA Science Inventory ESAT entry (cfpub.epa.gov); R `compositions` package documentation; Gelman & Rubin (1992); Brooks & Gelman (1998).*
