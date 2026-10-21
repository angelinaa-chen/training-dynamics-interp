# training-dynamics-interp

**Decodable vs. used.** We train small autoregressive transformers from scratch, save dense
checkpoints, and measure for each linguistic feature (subject number, part of speech, longer-range
dependencies) three things over training: behavior, probe decodability, and causal importance. The
headline quantity is the lag between when a feature becomes decodable and when the model starts
relying on it.

Team: Falak Raheja, Angelina Chen, Amy Wang.

Full proposal: [docs/proposal.md](docs/proposal.md).

## Layout

```
configs/                 YAML configs (base = WikiText-103; synthetic_* = templated agreement grammar)
src/tdi/
  config.py              dataclass configs
  model.py               GPT with head masks, residual hooks, residual-stream readout
  synthetic.py           agreement grammar: corpus generator + minimal-pair evals
  data.py                tokenizer, corpus prep, batching
  checkpoints.py         checkpoint schedule, save / load
  evals/agreement.py     subject-verb agreement minimal pairs
  train.py               training loop (writes runs/<name>_seed<k>/)
  prepare_data.py        python -m tdi.prepare_data
tests/
```

```
  encoding.py            tokenization with word-to-token alignment, counterfactual minimal pairs
  probing/               activations, feature definitions, probes + control tasks, per-checkpoint runner
  interventions/         head ablation, LEACE erasure (global and position-local), activation patching,
                         per-checkpoint runner with matched random controls
  analysis/              onset detection, lag estimation, bootstrap intervals, curve plots
scripts/run_seed.sh      train -> probe -> intervene for one seed
```

## Quick start

```bash
pip install -e ".[dev]"
pytest

# pipeline check on the synthetic grammar (about a minute)
python -m tdi.prepare_data --config configs/synthetic_debug.yaml
python -m tdi.train --config configs/synthetic_debug.yaml

# real run (downloads WikiText-103 on first use)
python -m tdi.prepare_data --config configs/base.yaml
python -m tdi.train --config configs/base.yaml --seed 0
```

One seed end to end, then the analysis across seeds:

```bash
bash scripts/run_seed.sh configs/synthetic_debug.yaml 0
bash scripts/run_seed.sh configs/synthetic_debug.yaml 1
python -m tdi.analysis.analyze --runs runs/synthetic_debug_seed0 runs/synthetic_debug_seed1
```

Each run writes `runs/<name>_seed<k>/` with `config.yaml`, `metrics.jsonl` (one row per
checkpoint: train/val loss, perplexity, agreement accuracy and logit difference per condition) and
`checkpoints/step_*.pt` (weights only, roughly 30 MB each for the base model), plus
`probe_results.csv` and `causal_results.csv` after probing and intervening. Checkpoints are dense early,
log-spaced later, and a loss trigger with a rolling weight buffer adds checkpoints on both sides of
sharp loss changes.

## Design notes

- **One tokenizer per corpus**, trained once and shared by all seeds, so checkpoints are comparable.
- **`<eos>` (id 0) delimits lines** in training data, and the agreement evals prompt the model as
  `<eos> prefix`, matching the training distribution.
- **Agreement evals** compare log P(correct verb) and log P(wrong verb) after a prefix. Pairs whose
  verb forms are not single tokens in the tokenizer are dropped. Conditions: `simple`, `adverb`, `pp_match`, `pp_mismatch`, `pp2_mismatch`, `rel_mismatch`.
- **Interventions are built into the model**: `head_mask` (layers x heads), `resid_hooks`, and
  `return_resid` for probing.
- Validation batches are fixed across checkpoints and seeds, so loss curves are directly comparable.
- **Probes** split train and test by group (noun lexeme for number features), so a probe must
  generalize to unseen nouns. Every probe is paired with a control-task probe (Hewitt and Liang);
  selectivity = accuracy minus control accuracy. Step 0 is the untrained baseline.
- **Erasure** comes in two forms. *Global* erasure applies LEACE at every token position and reports the
  language-modeling side effect; it is not specific, because number is one shared subspace. *Local*
  erasure applies it only at the position where the feature is read, which is the test used for
  causal onset. Both are compared with random same-rank subspaces.
- **Onsets** are the first step where a curve covers 50% of its total change and holds for three more
  checkpoints. Causal curves need a total effect of at least 1 nat to count as "used".
