# Response to Reviewers

We thank all reviewers for their constructive feedback. We address the main concerns below and provide supplementary materials with detailed analysis.

## 1. Novelty, Originality and Theoretical Foundations

**We respectfully disagree that our contributions are trivial.** While our method appears simple, this simplicity reflects careful design with rigorous theoretical grounding, not lack of innovation.

**Theoretical Contribution:** We prove that extending Rainbow to multi-prompt archives via pairwise comparisons requires Θ(M^2N) operations—a fundamental scalability bottleneck. Our multi-prompt fitness evaluation reduces this to Θ(MN), achieving an **asymptotic Θ(M) speedup**. This is not a parameter change but a **fundamental algorithmic advancement** making large-scale quality-diversity search tractable. Details in [Theoretical Analysis](https://anonymous.4open.science/r/rainbowplus-E0EF/docs/Theoretical%20Analysis.pdf).

**Regarding multi-prompt baseline:** The original Rainbow paper (Appendix A, Limitations) identifies extending to multi-prompt archives as an *open challenge*, not a trivial modification. Our complexity analysis reveals why: naive extension creates quadratic scaling that Rainbow's pairwise comparisons cannot efficiently handle.

## 2. Ablation Studies

We conducted comprehensive ablation studies analyzing core components and hyperparameters:

- **Mutation count (M):** Optimal at M=10 (90.82% ASR, 120 prompts). Higher values might decrease ASR due to quality dilution.
- **Fitness threshold (η):** η=0.6 balances quality-quantity tradeoff. Lower thresholds generate more prompts but reduce ASR (66.18% at η=0.2 vs 89.04% at η=0.6).
- **Component synergy:** Multi-element archive and multi-prompt evaluation are interdependent—neither works effectively alone due to computational constraints proven theoretically.

Full results: [Ablation Study](https://anonymous.4open.science/r/rainbowplus-E0EF/docs/Ablation%20Study.pdf)

## 3. Experimental Design

**Benchmark selection:**
- Rainbow comparison uses 6 datasets to validate generalization across diverse scenarios
- SOTA comparison uses HarmBench, the standard benchmark employed by all 9 baseline methods, ensuring fair and consistent evaluation

**Diversity metrics:** We report diversity only for Rainbow-style methods (Table 2) that explicitly optimize for diversity. Other baselines (GCG, PAIR, TAP) optimize solely for attack success, making diversity comparison inappropriate.

## 4. Implementation and Model Performance

- **Rainbow baseline:** While closed-source, we meticulously followed the original paper's specifications, algorithm pseudocode, and public prompt templates, ensuring reliable comparison.

- **Closed-source models:** Our 29% ASR on GPT-4o Mini surpasses AutoDAN-Turbo (26.8%). The 6% on GPT-4.1 Nano reflects exceptional model robustness—any vulnerability discovery in such systems is significant for safety assessment. We acknowledge this limitation and propose warm-up phase integration as future work.


**Conclusion:** RainbowPlus provides substantial contributions through rigorous theoretical foundations (Θ(M) complexity reduction), comprehensive empirical validation (81.1% average ASR across 12 LLMs), and practical efficiency gains, advancing both the theoretical understanding and practical capabilities of LLM red-teaming.