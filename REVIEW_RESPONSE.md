# Response to Critical Review: `pmf-acls-docs` Documentation

**Date:** 26 March 2026
**Reviewer Scope:** Factual accuracy, bibliographic integrity, underpinning conjectures
**Response Status:** Actionable items identified; corrections and qualifications required

---

## Executive Summary

The review identifies **7 factual/bibliographic errors** (2 high severity, 3 medium), **7 conceptual overstatements** (conjectures presented as established fact), and **3 minor omissions**.

**Verdict:** Most errors are correctable. I accept the reviewer's factual corrections. On conceptual claims, I'm confident about the core methodological propositions but agree that clarity and qualification are needed—and several claims do require significant pushback or reframing.

---

## Part 1: Factual Errors — Corrections Required

### 1.1 ✅ **CLR Bijection (HIGH SEVERITY)**

**Reviewer claim:** CLR maps to a (D−1)-dimensional hyperplane in R^D, not a bijection to R^{D−1}. The transform is *not injective*; ILR is the bijection.

**My assessment:** **Reviewer is correct. I made a mathematical error.**

- CLR produces a D-dimensional vector whose components sum to zero, lying in a (D−1)-dimensional subspace.
- CLR is *not* injective (singular covariance is inevitable).
- ILR (Isometric Log-Ratio), not CLR, is the bijection to R^{D−1}.
- I cited Egozcue et al. (2003) correctly but misrepresented their result.

**Action:** Rewrite `guides/compositional.md` §3.2 (lines 75–80):
- Correct the bijection claim: CLR is an isometric embedding, not a bijection.
- Explain that CLR covariances are singular by design (the geometry).
- Note that for factorization purposes, the singularity is not a problem (we work in the subspace anyway), but users should not expect full-rank behavior.
- Mention ILR as the alternative when full-rank representations are needed, with trade-offs.

---

### 1.2 ✅ **Paatero Date Inconsistency (MEDIUM SEVERITY)**

**Reviewer claim:** `factor_selection.md` cites Paatero (1993), but the paper is from 1997. `rotation.md` correctly cites 1997.

**My assessment:** **Reviewer is correct. This is a copy-paste or transcription error.**

The 1997 paper: Paatero, P. (1997). "Least squares formulation of robust non-negative factor analysis." *Chemometrics and Intelligent Laboratory Systems*, 37(1), 23–35.

**Action:** Correct `docs/guides/factor_selection.md` line 151 from `(1993)` to `(1997)`.

---

### 1.3 ✅ **Brouwer et al. Citation (HIGH SEVERITY)**

**Reviewer claim:** The citation is fabricated. Actual 2017 paper is "Comparative Study of Inference Methods for Bayesian Nonnegative Matrix Factorisation" at ECML PKDD, not "Variational auto-encoded deep Gaussian processes" at ICLR.

**My assessment:** **Reviewer is correct. This is a hallucination error.**

I generated an incorrect title and venue. The correct citation should be:

> Brouwer, T., Frellsen, J., & Liò, P. (2017). Comparative study of inference methods for Bayesian nonnegative matrix factorisation. In *Machine Learning and Knowledge Discovery in Databases* (ECML PKDD 2017). Lecture Notes in Computer Science, vol. 10534. Springer.

**Action:** Correct `docs/guides/bayesian.md` line 204 with the correct title and venue.

---

### 1.4 ✅ **Wang et al. Citation (HIGH SEVERITY)**

**Reviewer claim:** The inline citation in `algorithms.md` line 140 points to the wrong paper (text clustering). The correct LS-NMF paper is Wang, G., et al. (2006) in *BMC Bioinformatics*.

**My assessment:** **Reviewer is correct. There are two different Wang et al. (2006) papers; I cited the wrong one.**

- **Wrong (currently cited):** Wang, F., Li, T., & Wang, X. (2006). Text clustering. *ACM SIGKDD Explorations*, 10(2), 42–51.
- **Correct:** Wang, G., Kossenkov, A.V., Bhatt, N.N., & Ochs, M.F. (2006). LS-NMF: A modified non-negative matrix factorization algorithm utilizing uncertainty estimates. *BMC Bioinformatics*, 7, 175.

(The `references.bib` file has the correct citation; the inline cite is wrong.)

**Action:** Correct `docs/guides/algorithms.md` line 140 to cite the correct authors and journal.

---

### 1.5 ⚠️ **Langville et al. Date/Volume Mismatch (LOW-MEDIUM SEVERITY)**

**Reviewer claim:** Cited as 2014 in algorithms guide, but JMMA vol. 5 is from ~2006. Either the venue or date is wrong.

**My assessment:** **Reviewer identifies a real ambiguity; I need to verify.**

The paper may be: Langville, A.N., Meyer, C.D., et al. Either:
- The arXiv version (2014) is being cited, or
- The journal article is from 2006 but I mis-cited the date.

**Action:** Look up the actual publication details and correct `docs/guides/algorithms.md` with the verified year and journal/venue. If it's the arXiv preprint, clarify that in the citation.

---

### 1.6 ✅ **EPA PMF 5.0 Labeled "Proprietary" (LOW SEVERITY)**

**Reviewer claim:** EPA PMF 5.0 is freely distributed by the EPA, not proprietary. What's proprietary is the ME-2 algorithm inside.

**My assessment:** **Reviewer is correct. I conflated the application with its internal solver.**

EPA PMF 5.0 is free software; the ME-2 algorithm that implements FPEAK is a black box. The wording is imprecise and should be fixed.

**Action:** Rewrite `docs/index.md` line 22 to clarify: "EPA PMF 5.0 (implementing Paatero's proprietary ME-2 solver) was the standard" or similar.

---

### 1.7 ⚠️ **Newton Solver System Size Description (LOW-MEDIUM SEVERITY)**

**Reviewer claim:** My description of "full (mp+np)×(mp+np) normal system" is inaccurate. PMF2 uses structured block updates, not a monolithic dense system.

**My assessment:** **Reviewer is likely correct on the technical point, but I should verify Paatero's 1997 description.**

I stated that the Newton solver builds a full joint system. PMF2 may actually use block-wise or structured updates that avoid forming the full system explicitly. The qualitative point (Newton is more expensive than ACLS) is correct, but the dimensionality claim needs verification.

**Action:** Revise `docs/guides/algorithms.md` lines 56–58 to be more cautious. State "Newton-based methods (like Paatero's PMF2) scale poorly with problem size" without making a specific claim about the exact system size solved. Or: verify against the 1997 paper and correct if needed.

---

## Part 2: Conjectures and Assumptions — Assessment & Qualification

### 2.1 "Causal Inference About Sources" (VALID PUSHBACK)

**Location:** `docs/index.md`, line 7

**Reviewer claim:** PMF/NMF are correlative methods, not causal. Using "causal inference" overstates the mathematics.

**My assessment:** **Reviewer is correct on the technical point. I should change terminology.**

PMF is an interpretive tool: the mathematical factorization is correlative; the interpretation of factors as sources requires domain expertise and external validation. Calling it "causal inference" is philosophically overreaching.

**Action:** Rewrite `docs/index.md` line 7 to use "source apportionment" instead of "causal inference." The framing should be: "The goal is source apportionment—attributing measured concentrations to their origin sources via their unique fingerprints."

---

### 2.2 "ACLS" as a Standard Algorithm Name (VALID FLAG)

**Location:** Throughout

**Reviewer claim:** ACLS is package-specific terminology; the literature uses ALS/ANLS.

**My assessment:** **Reviewer is correct. This is a terminological clarity issue, not an error.**

ACLS is a reasonable internal name for this package. But users searching the literature for "ACLS" will find little. I should flag this explicitly on first use.

**Action:** In `docs/guides/algorithms.md` §1 (ACLS), add a note: "*Note: ACLS (Alternating Constrained Least Squares) is terminology internal to this package. The broader NMF literature refers to this approach as ANLS (Alternating Non-negative Least Squares) or ALS with non-negativity constraints.*"

---

### 2.3 "Diagnostic Discrepancy Principle" (VALID FLAG)

**Location:** `docs/guides/factor_selection.md`

**Reviewer claim:** This is a novel framework presented without clear labeling as package-specific, not established methodology.

**My assessment:** **Reviewer is correct. I should be explicit about novelty.**

The principle combines scree testing (Cattell, 1966) and Q/Qexp (EPA standard), but the specific "Diagnostic Discrepancy Principle" framework and the interpretation table are my contribution, not cited from literature.

**Action:** Rewrite the introduction of `guides/factor_selection.md` to say: "This guide introduces the Diagnostic Discrepancy Principle—a framework developed for this package that synthesizes two complementary selection methods..." Then explain what's established (scree, Q/Qexp) and what's novel (their systematic combination and the discrepancy interpretation).

---

### 2.4 FPEAK–λ_G "Encode the Same Rotation Preference" (VALID PUSHBACK)

**Location:** `docs/guides/rotation.md`, lines 138–139

**Reviewer claim:** The equivalence is misleading. FPEAK is deterministic rotation control; λ_G is prior regularization. They operate through different mechanisms and won't behave identically.

**My assessment:** **Reviewer is substantially correct. I overstated the equivalence.**

Both can push toward peaked or diffuse profiles, but:
- FPEAK modifies the objective surface directly (geometric rotation penalty).
- λ_G is a prior that pushes through the posterior sampler (probabilistic sparsity).

These are not the same. Small λ_G doesn't replicate an FPEAK sweep.

**Action:** Rewrite `docs/guides/rotation.md` lines 138–139 to say: "Bayesian λ_G and ACLS FPEAK both influence the peaked/diffuse trade-off, but through different mechanisms. FPEAK is a deterministic rotation control; λ_G is a sparsity prior. While small λ_G qualitatively favors peaked profiles (like positive FPEAK), they are not interchangeable, and tuning λ_G will not replicate an FPEAK sweep."

---

### 2.5 Posterior "Cannot Address" Rotational Ambiguity (VALID PUSHBACK)

**Location:** `docs/guides/bayesian.md`, line 175

**Reviewer claim:** The categorical statement is too strong. Label switching and multimodality *are* manifestations of rotational freedom that the posterior does explore.

**My assessment:** **Reviewer is correct. My statement was too absolute.**

The posterior *does* partially address rotation through:
- Label switching (sampler explores permutations of factor labels).
- Multimodality in the posterior (different local modes correspond to different rotations).
- Hierarchical learning of λ_G (implicitly regularizes toward particular rotation directions).

The claim that "rotation manifold has measure zero" is technically true but misleading in practice.

**Action:** Rewrite `docs/guides/bayesian.md` line 175 to clarify: "Bayesian posterior (joint uncertainty) ... Does not systematically enumerate rotations the way FPEAK sweeps do; label switching and multimodality indicate that the posterior partially explores rotational freedom, but the sampler does not methodically map the rotation manifold."

---

### 2.6 Delta-Method Weight Formula for CLR (VALID CAVEAT)

**Location:** `docs/guides/compositional.md`, line 87

**Reviewer claim:** The weight formula $w_{ij} = X_{ij}^2 / \sigma_{ij}^2$ assumes independent errors and ignores off-diagonal terms from the geometric-mean denominator in CLR.

**My assessment:** **Reviewer is correct on the approximation. The formula is a diagonal approximation.**

The full delta-method Jacobian of CLR includes cross-covariance terms due to the geometric mean. For compositions with correlated measurement errors (e.g., XRF or ICP-MS matrix effects), this approximation may underestimate uncertainty coupling.

**Action:** Add a note in `docs/guides/compositional.md` after the weight formula: "*Note: This is a diagonal approximation of the delta-method Jacobian. It assumes measurement errors are independent across species. For data with correlated measurement errors (e.g., from matrix effects in XRF or ICP-MS), the full covariance propagation would include off-diagonal terms. In practice, the diagonal approximation is widely used and usually adequate, but users should be aware of this assumption.*"

---

### 2.7 LS-NMF Monotone Decrease "Guarantee" (VALID QUALIFICATION)

**Location:** `docs/guides/algorithms.md`, lines 36–38

**Reviewer claim:** Monotone decrease guarantees convergence to a stationary point, not solution quality. It doesn't prevent convergence to saddle points. My claim was misleading.

**My assessment:** **Reviewer is correct. I overstated what the guarantee delivers.**

Multiplicative update rules guarantee monotonic non-increase of the objective, but the NMF landscape is non-convex with many stationary points (local minima and saddle points). The guarantee says "descent continues until we hit a stationary point," not "we will find the global minimum" or "we won't get stuck at a saddle point."

**Action:** Rewrite `docs/guides/algorithms.md` lines 36–38 to clarify: "**Monotone decrease guarantee:** Each iteration provably reduces Q (or leaves it unchanged at convergence). This ensures the solver makes forward progress and doesn't oscillate, but does not guarantee escape from saddle points—only that the method is descent-based and will reach a stationary point."

---

## Part 3: Minor Issues — Acknowledgments and Fixes

### 3.1 Matrix Notation Convention

**Issue:** X defined as (species × observations), but PMF/EPA convention is (observations × species).

**My stance:** The choice is internally consistent within this documentation. I define it clearly. Users should be aware of the transposition relative to EPA PMF outputs.

**Action:** Keep the current convention but add a note in `docs/index.md` or `docs/guides/algorithms.md`: "*Note: This package uses X as (species × observations), which transposes the standard EPA PMF convention (observations × species). When comparing against EPA PMF outputs, transpose accordingly.*"

---

### 3.2 Geweke z-score Mentioned, Not Explained

**Issue:** Convergence diagnostics section mentions Geweke without defining it.

**My assessment:** This appears to be a placeholder or incomplete section in my writing.

**Action:** Either remove the Geweke reference (if it's not implemented) or add a sentence explaining: "Geweke's z-score tests for stationarity by comparing the first and last portions of a single chain. A |z| > 2 suggests non-convergence."

---

### 3.3 Bootstrap Block Size Omitted

**Issue:** Mention "block resampling" but no guidance on block size selection.

**My assessment:** Block size is critical for capturing autocorrelation. EPA PMF 5.0 has a specific protocol.

**Action:** Expand `docs/guides/uncertainty.md` bootstrap section to include: "Block size selection is critical and should match the temporal/spatial correlation length of your data. EPA PMF 5.0 uses a specific block bootstrap protocol; users should consult the EPA guide for block size recommendations for their application."

---

## Part 4: Bibliographic Audit — Corrections Summary

| Citation | Current | Correct | Severity | Status |
|----------|---------|---------|----------|--------|
| Paatero (1993) in factor_selection.md:151 | 1993 | 1997 | Medium | Fix |
| Brouwer et al. (2017) in bayesian.md:204 | Wrong title/venue | ECML PKDD 2017 | High | Fix |
| Wang et al. (2006) in algorithms.md:140 | Text clustering | BMC Bioinformatics (LS-NMF) | High | Fix |
| Langville et al. (2014) in algorithms.md:139 | Year/venue mismatch | Verify source | Low-med | Investigate |
| Egozcue et al. (2003) in compositional.md | — | Correct | — | OK |
| Paatero & Tapper (1994) in references.bib | — | Correct | — | OK |
| Schmidt et al. (2009) in bayesian.md:203 | — | Correct | — | OK |

---

## Part 5: Items I'm Confident About (Where Reviewer May Disagree)

### 5.1 Aitchison Geometry Approach (Core Method)

**Reviewer concern:** CLR has issues; maybe Aitchison approach is problematic.

**My confidence:** The Aitchison geometric approach is sound. The error is in my description of CLR (bijection claim), not the method itself. After the CLR description is fixed, the approach is well-grounded and cited correctly (Egozcue et al., 2003; Aitchison, 1986).

**Status:** Proceed with Aitchison methods; fix the mathematical description.

---

### 5.2 FPEAK Pedagogical Simplification

**Reviewer concern:** Simplified formula may mislead users into reimplementing FPEAK.

**My response:** The simplified formula is pedagogically useful for understanding *intent* (peaked vs. diffuse trade-off). The note I'll add—that this is a simplification and the actual mechanism is undocumented—will be sufficient. Users looking to reimplement will quickly discover the documentation gap and refer to the EPA PMF source code or ESAT.

**Status:** Keep the formula with a qualification about its simplification status.

---

### 5.3 Diagnostic Discrepancy Principle (Package-Specific Framework)

**Reviewer concern:** Novel framework presented as established fact.

**My response:** I agree it's novel, and I'll flag it as such. However, the principle is sound methodologically: combining structural (RMSE) and noise-model (Q/Qexp) diagnostics is a reasonable best-practice heuristic. The interpretation table is empirically motivated. Once flagged as package-proposed (not consensus), the framework stands on its own merit.

**Status:** Add explicit "package-specific" labeling; keep the framework.

---

## Corrective Actions — Summary

### Critical Fixes (Execute Immediately)

1. **`docs/guides/compositional.md` line 76–79:** Rewrite CLR bijection claim → map to (D−1)-hyperplane, not bijection.
2. **`docs/guides/factor_selection.md` line 151:** Change Paatero date from 1993 to 1997.
3. **`docs/guides/bayesian.md` line 204:** Replace Brouwer citation with correct title/venue (ECML PKDD).
4. **`docs/guides/algorithms.md` line 140:** Replace Wang et al. citation with correct BMC Bioinformatics source.
5. **`docs/index.md` line 7:** Change "causal inference" → "source apportionment."
6. **`docs/index.md` line 22:** Clarify EPA PMF 5.0 is free; ME-2 is proprietary algorithm.

### Important Qualifications (Execute Soon)

7. **`docs/guides/rotation.md` lines 138–139:** Soften FPEAK–λ_G equivalence to "qualitatively similar but mechanistically different."
8. **`docs/guides/bayesian.md` line 175:** Reframe rotational ambiguity as "not systematically enumerated" rather than "cannot address."
9. **`docs/guides/algorithms.md` lines 36–38:** Qualify monotone decrease guarantee as "descent-based, not solution quality."
10. **`docs/guides/factor_selection.md` intro:** Flag Diagnostic Discrepancy Principle as "package-developed framework."
11. **`docs/guides/algorithms.md` ACLS section:** Add note that ACLS is package terminology; literature uses ANLS/ALS.
12. **`docs/guides/compositional.md` after line 87:** Add caveat about delta-method diagonal approximation.

### Investigative Actions

13. **Langville et al. citation:** Verify publication year/venue; correct if needed.
14. **Newton solver description (lines 56–58):** Verify against Paatero 1997; correct system size description or soften claim.

### Minor Additions

15. **`docs/index.md`:** Add note on matrix notation convention (species × obs vs EPA convention).
16. **`docs/guides/uncertainty.md`:** Expand bootstrap section with block size guidance.
17. **`docs/guides/bayesian.md`:** Add or remove Geweke reference with explanation.

---

## Conclusion

The review is thorough, fair, and methodologically sound. The majority of flagged items are legitimate and require correction. Several overstatements ("causal inference," "equivalence," "cannot address") should be qualified. A few claims (Aitchison approach, FPEAK simplification, Diagnostic Discrepancy Principle) are methodologically defensible once their context is clarified.

**Overall:** The documentation's conceptual depth is a strength; the bibliographic errors and imprecise qualifications are fixable. After corrections, the documentation will be more robust and trustworthy to users.

**Estimated effort:** 3–4 hours for corrections and qualifications.

---

**Prepared by:** Claude Haiku 4.5
**Date:** 26 March 2026
