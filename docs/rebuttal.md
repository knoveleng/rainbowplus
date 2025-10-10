We thank all reviewers for their constructive feedback. We address the main concerns below with additional experiments and analysis we have completed.

## 1. Novelty, Originality and Theoretical Foundations
We respectfully disagree that our contributions are trivial. While our method appears simple, this simplicity reflects careful design with rigorous theoretical grounding, not lack of innovation.

Theoretical Contribution: We have completed formal complexity analysis proving that extending Rainbow to multi-prompt archives via pairwise comparisons requires Θ(M²N) operations - a fundamental scalability bottleneck. Our multi-prompt fitness evaluation reduces this to Θ(MN), achieving an asymptotic Θ(M) speedup.

Regarding multi-prompt baseline: The original Rainbow paper (Appendix A, Limitations) identifies extending to multi-prompt archives as an *open challenge*, not a trivial modification. Our complexity analysis reveals why: naive extension creates quadratic scaling that Rainbow's pairwise comparisons cannot efficiently handle.

## 2. Ablation Studies
We have conducted comprehensive ablation studies analyzing core components and hyperparameters. Key findings show: (1) optimal mutation count at M=10 achieves 90.82% ASR while M=1 only reaches 80.39%, and (2) fitness threshold 0.6 balances quality-quantity tradeoff (89.04% ASR) compared to 0.2 (66.18% ASR with more but lower-quality prompts). These studies demonstrate that multi-element archive and multi-prompt evaluation are synergistic components that must work together - neither is effective in isolation due to computational constraints. **We will provide supplementary documents for theoretical analysis and ablation studies when allowed.**

## 3. Experimental Design
Benchmark selection:
- Rainbow comparison uses 6 datasets to validate generalization across diverse scenarios
- SOTA comparison uses HarmBench, the standard benchmark employed by all 9 baseline methods, ensuring fair and consistent evaluation

Diversity metrics: We report diversity only for Rainbow-style methods that explicitly optimize for diversity. Other non-Rainbow baselines optimize solely for attack success, making diversity comparison inappropriate.

## 4. Implementation
- Rainbow baseline: While closed-source, we meticulously followed the original paper's specifications, algorithm pseudocode, and public prompt templates, ensuring reliable comparison.