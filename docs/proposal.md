# Decodable vs. Used: Tracking the Emergence and Causal Role of Internal Representations

2026-10-14 · Angelina Chen

## Summary

**The project asks when a transformer starts *using* a linguistic feature, and how that compares to when the feature first becomes *decodable*.** Small autoregressive language models are trained from scratch, dense checkpoints are saved, and three curves are measured per feature: behavior, probe decodability, and causal importance. The headline quantity is the **lag** between decodability onset and causal onset, with confidence intervals across seeds.

- **Gap:** Checkpoint-level work (TRACE, Crosscoding Through Time) tracks when representations appear but does not test causal use. Intervention work (Hao 2023, Jin et al.) tests causal use but only on a finished model.
- **Contribution:** A quantitative, multi-seed estimate of the decodability-to-use lag, validated on features where a representation is present but should not be used.
- **Scope:** Three features studied carefully (subject number for agreement, part of speech, one longer-range dependency), not a broad survey.

## Background and gap

Decodable information is not necessarily information the model uses, and the two literatures that would answer this each cover only half of it.

- **Over training, no causal test.** TRACE and Crosscoding Through Time track when linguistic representations emerge across checkpoints. Neither tests whether those representations drive predictions.
- **Causal test, no training axis.** Hao (2023) shows verb conjugation depends on a linear encoding of subject number. Jin et al. (2023) use interventions to check that probed representations are really encoded. Both analyze a finished model.
- **Methods built on.** Diagnostic classifiers for agreement (Giulianelli et al., 2018); probe controls that separate encoding from probe capacity (Hewitt and Liang, 2019); amnesic probing (Elazar et al., 2021); linear concept erasure (LEACE, Belrose et al., 2023); activation patching and causal mediation (Vig et al., 2020; Finlayson et al., 2021); emergence of induction heads during training (Olsson et al., 2022).

**Open gap:** no work found so far estimates when a feature goes from decodable to causally used, and how consistent that timing is across seeds. A literature search is still needed to confirm this.

## Hypotheses

Each hypothesis states a prediction that the data can contradict.

1. **H1, decodability leads use.** For each feature, probe-selectivity onset precedes causal onset by a positive lag, and the lag is larger than seed-to-seed variation in onset time.
2. **H2, use implies decodability.** No feature shows causal onset before decodability onset. If one does, the probes miss the representation (for example, it is nonlinear).
3. **H3, decodable but unused exists.** A property that is predictable from context but irrelevant to the next-token loss becomes decodable without ever becoming causally important. This checks that the causal test can say "no."
4. **H4, abrupt onset.** Causal onset coincides with a sharp behavioral change (a loss or agreement-accuracy jump) more tightly than decodability onset does.

**Null results count.** If lag is zero within noise, the finding is that decodability is a good proxy for use in this setting, which is useful to know.

## Methods

### Models and data

- **Model:** Decoder-only transformer in PyTorch, starting at about 6 layers, 8 heads, d_model 256 (roughly 10M parameters), with a second size later to test whether the lag scales.
- **Corpus:** WikiText-103 as the main corpus. WikiText-2 (about 2M tokens) is too small for rare agreement-with-attractor constructions and invites overfitting. WikiText-2 stays as a fast debugging set.
- **Tokenization:** Fixed BPE tokenizer, trained once and shared by all runs. Labels (number, POS, dependencies) come from spaCy and are aligned to subword tokens, using the final subword of each word.
- **Checkpoints:** Dense early and around loss drops, sparse late (log-spaced, with extra checkpoints triggered when the loss moves sharply), over at least 3 seeds.

### Behavioral evaluation

- Held-out perplexity.
- **Agreement minimal pairs** such as "The key to the cabinets is / are", with attractor nouns, varied distance, and relative-clause variants. The score is the probability of the correct verb form minus the incorrect one.
- Matching minimal-pair sets for the other chosen features. Templates are filled from a vocabulary held out from training sentences where possible.

### Decodability

- Linear probes per layer, per checkpoint, per feature, on the residual stream at the relevant token.
- **Controls:** random-label control tasks to report selectivity (Hewitt and Liang), an untrained-model baseline with the same architecture, and a shuffled-checkpoint baseline.
- Probe strength is fixed in advance (regularized logistic regression, same hyperparameters everywhere) so curves are comparable across checkpoints.

### Causal importance

Three intervention types are used, each compared against matched random controls, so the claim does not rest on ablation alone:

1. **Component ablation:** zero or mean-ablate heads and MLPs, versus random components of the same size.
2. **Activation patching on minimal pairs:** patch activations from the singular sentence into the plural one (and back), and measure the shift in verb-form preference. This is robust to redundancy that fools ablation.
3. **Direction-level intervention:** remove the feature subspace with LEACE (necessity), and add or swap it (sufficiency), versus random subspaces of equal rank.

Each intervention is reported on the targeted metric (agreement accuracy or logit difference) and a side-effect metric (perplexity), so "specific" is measured, not assumed.

## Metrics and onset definitions

The lag is only meaningful if each onset is defined the same way before looking at results. Each onset is the first checkpoint where a curve reaches 50% of its final-minus-initial gain and stays there for the following three checkpoints.

| Curve | Quantity | Onset time |
| --- | --- | --- |
| Behavior | Agreement logit difference (per feature); perplexity | t_beh |
| Decodability | Best-layer probe selectivity (accuracy minus control-task accuracy) | t_dec |
| Causal use | Drop in targeted metric under patching or LEACE, minus the drop under random controls | t_use |

```latex
\text{lag} = t_{\text{use}} - t_{\text{dec}}
```

- **Seeds:** Report lag per seed and its bootstrap 95% interval across checkpoints and seeds. With 3 seeds, all points are shown, not only means.
- **Time axis:** Training steps, with token count and loss as alternative axes, since onset ordering can change with the choice.
- **Specificity:** An intervention counts as causal only if it moves the targeted metric more than its random control and changes perplexity by less than a preset margin.

## Experiment matrix

Three features plus one dissociation control, chosen so that the control should come out decodable but unused and one feature is a pipeline check.

| Feature | Why it is here | Expected result |
| --- | --- | --- |
| Subject number (agreement) | Best-studied case; Hao (2023) gives a known causal story | Decodable, then used (H1) |
| Part of speech | Easy, early, probably decodable and used almost at once | Lag near zero; sanity check |
| Subject number at a non-agreeing position (for example, the subject noun in a sentence with no later verb in the window) | Decodable from context, irrelevant to the loss | Decodable but unused (H3) |
| Longer-range dependency (for example, quote or bracket closing) | Tests whether results extend past agreement | Later onset, possibly larger lag |

**Run plan:** 3 seeds at the base size for all features (core result), 1 additional model size and 1 additional data fraction for agreement only (does the lag scale), and a synthetic-grammar corpus where the correct mechanism is known, used to validate that the probes and interventions recover it before they are trusted on WikiText.

The feature table is a draft: the final choice is made in the first week after checking the model learns each phenomenon at all.

## Risks and mitigations

- **Model never learns agreement.** A 10M-parameter model on WikiText may not reach it. Mitigation: check at the end of week 1; fall back to a larger corpus, a longer run, or a templated agreement corpus mixed into training.
- **Onset is noisy or threshold-dependent.** Mitigation: fix definitions now, report a sweep of thresholds, and use 3 or more seeds.
- **Probes miss a nonlinear representation, so t_dec is late.** This would look like "use before decodability." Mitigation: add a small MLP probe as a robustness check and report both.
- **Ablation shows no effect because of redundancy.** Mitigation: patching and direction-level interventions, not ablation alone.
- **Interventions damage the model generally.** Mitigation: always report perplexity, and compare against matched random controls.
- **Compute.** Dense checkpoints times seeds times sizes adds up. Mitigation: store residual activations only for the eval sets, not the full corpus, and prioritize the base-size runs.
- **Schedule.** Mitigation: the core result (agreement, 3 seeds) is defined first; other features and the second size are stretch goals.

## References

- Olsson, C., et al. (2022). In-context Learning and Induction Heads. [arXiv:2209.11895](https://arxiv.org/abs/2209.11895)
- Giulianelli, M., Harding, J., Mohnert, F., Hupkes, D., and Zuidema, W. (2018). Under the Hood: Using Diagnostic Classifiers to Investigate and Improve how Language Models Track Agreement Information. [arXiv:1808.08079](https://arxiv.org/abs/1808.08079)
- Hao, Y. (2023). Verb Conjugation in Transformers Is Determined by Linear Encodings of Subject Number. [arXiv:2310.15151](https://arxiv.org/abs/2310.15151)
- Jin, C., et al. (2023). Emergent Representations of Program Semantics in Language Models Trained on Programs. [arXiv:2305.11169](https://arxiv.org/abs/2305.11169)
- TRACE: Training and Inference-Time Interpretability Analysis for Language Models (2025). [arXiv:2507.03668](https://arxiv.org/abs/2507.03668)
- Crosscoding Through Time: Tracking Emergence and Consolidation of Linguistic Representations Throughout LLM Pretraining (2025).
- To add after checking each source: Hewitt and Liang (2019), control tasks; Elazar et al. (2021), amnesic probing; Belrose et al. (2023), LEACE; Vig et al. (2020) and Finlayson et al. (2021), causal mediation.

Claude and ChatGPT were used as learning resources while writing the original proposal; Claude was used to revise this version.
