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

Planned next: `probing/` (activation extraction, probes with control tasks, onset detection) and
`interventions/` (head ablation, activation patching, LEACE), then `analysis/` for the lag plots.

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

Each run writes `runs/<name>_seed<k>/` with `config.yaml`, `metrics.jsonl` (one row per
checkpoint: train/val loss, perplexity, agreement accuracy and logit difference per condition) and
`checkpoints/step_*.pt` (weights only, roughly 30 MB each for the base model).

## Design notes

- **One tokenizer per corpus**, trained once and shared by all seeds, so checkpoints are comparable.
- **`<eos>` (id 0) delimits lines** in training data, and the agreement evals prompt the model as
  `<eos> prefix`, matching the training distribution.
- **Agreement evals** compare log P(correct verb) and log P(wrong verb) after a prefix. Pairs whose
  verb forms are not single tokens in the tokenizer are dropped. Conditions: `simple`, `adverb`, `pp_match`, `pp_mismatch`, `pp2_mismatch`, `rel_mismatch`.
- **Interventions are built into the model**: `head_mask` (layers x heads), `resid_hooks`, and
  `return_resid` for probing.
- Validation batches are fixed across checkpoints and seeds, so loss curves are directly comparable.
