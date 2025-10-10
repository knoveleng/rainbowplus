We thank all reviewers for their constructive feedback. We address the main concerns below with additional experiments and analysis we have completed.

## 1. Novelty, Originality and Theoretical Foundations
Our contribution goes beyond an incremental extension of Rainbow. RAINBOWPLUS is the **first LLM-based Multi-Objective Evolutionary Algorithm (LLM-MOEA)**, where large language models serve as evolutionary operators. (1) The Mutator LLM performs adaptive mutation, (2) the Judge LLM enables multi-prompt probabilistic fitness evaluation instead of pairwise comparison, and (3) the multi-element archive generalizes MAP-Elites into a population-based structure, **reducing complexity from Θ(M²N) to Θ(MN)**. This integration forms a **novel, LLM-driven evolutionary optimization framework** that resolves Rainbow's open challenge of multi-prompt scalability both theoretically and empirically.

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