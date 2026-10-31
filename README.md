# training-dynamics-interp

**Does a language model use a piece of information as soon as it can represent it?**

We train small transformer language models from scratch and track, checkpoint by checkpoint, when
a grammatical feature (starting with the singular/plural number of a sentence's subject) first
becomes *readable* from the model's internal values, and when the model first *relies on it* to
predict the right verb. The headline measurement is the **lag** between those two moments.

Team: Falak Raheja, Angelina Chen, Amy Wang. Full proposal: [docs/proposal.md](docs/proposal.md).

## The idea in three measurements

For every saved checkpoint we measure:

| | Question | Method |
|---|---|---|
| **Behavior** | Does the model pick the right verb? ("The keys near the door **are**", not "is") | Compare log P(correct verb) and log P(wrong verb) on minimal pairs in six sentence types, with and without confusing nearby nouns |
| **Decodability** | Can the subject's number be read from the hidden states? | Linear probes at every layer, tested on nouns they never saw, compared with the untrained model |
| **Causal use** | Does the model actually depend on that information? | Remove or swap it (head ablation, LEACE erasure, activation patching) and check whether verb choice gets worse, always against random controls of the same size |

The **onset** of each curve is the training step where it covers half its total change. The
**lag** is the causal onset minus the decodability onset. A feature that is decodable but never
used (distractor-noun number) should show up as readable with no causal effect.

## Results so far (preliminary)

| Setting | Finding |
|---|---|
| **Synthetic grammar, 3 seeds** (tiny model, templated sentences) | Subject number becomes decodable 77-214 steps before the model uses it. Distractor-noun number becomes decodable but erasing it never changes verb choice by more than 1 nat (the threshold for counting as used). |
| **WikiText-103, 1 seed** (20,000 steps, about 7M parameters) | Validation perplexity falls from 8,537 to 28.4; agreement accuracy rises slowly from chance (0.49) to 0.75, but stays weak when a nearby noun has the opposite number (0.52-0.67). |
| | Subject number is decodable by about step 175-290, but erasing it only starts to hurt around step 2,400-3,300, close to the behavioral onset (about 3,100). |
| | Erasing subject number at the subject or last token removes essentially all of the model's agreement margin (about 1.5 nats; random control about 0.01). Ablating the top 2 attention heads costs 0.95 nats (random heads 0.02). |
| | Patching shows the number is read at the subject token in early layers and carried to the last token before the verb in later layers. |

Caveats: small models, one WikiText seed, three synthetic seeds. Early decodability may partly
reflect shallow cues such as plural word endings; the causal tests are what separate readable from
used. Treat the timings as preliminary.

## Quick start

```bash
pip install -e ".[dev]"
pytest

# Fast end-to-end check on the synthetic grammar (about 15 minutes per seed, start to finish)
python -m tdi.prepare_data --config configs/synthetic_debug.yaml
bash scripts/run_seed.sh configs/synthetic_debug.yaml 0 --max-steps 3000

# Real run on WikiText-103 (downloads the corpus on first use; about 4.5 hours on a laptop GPU)
python -m tdi.prepare_data --config configs/base.yaml
python -m tdi.train --config configs/base.yaml --seed 0 --out-dir <folder for checkpoints>
python -m tdi.probing.run --run <run folder> --n-examples 4000
python -m tdi.interventions.run --run <run folder> --device mps --every 3 --n-examples 2000 --n-random 3

# Lags and plots (accepts several runs, i.e. seeds)
python -m tdi.analysis.analyze --runs <run folder> [<run folder> ...] --out analysis_out
```

`scripts/run_seed.sh` chains training, probing and interventions for one seed. Keep checkpoints
(a few GB per WikiText run) in a folder that is not cloud-synced.

## What a run produces

`<out_dir>/<name>_seed<k>/` contains `config.yaml`, `metrics.jsonl` (loss, perplexity and agreement
accuracy per checkpoint), `checkpoints/step_*.pt` (weights only), `probe_results.csv` and
`causal_results.csv`. The analysis step writes per-run onset tables, a cross-seed lag summary with
bootstrap intervals, and curve plots.

## Repository layout

```
configs/                  base.yaml (WikiText-103), synthetic_debug.yaml (templated grammar)
scripts/run_seed.sh       train -> probe -> intervene for one seed
src/tdi/
  model.py                small GPT; heads can be switched off, hidden states read or edited per layer
  train.py, checkpoints.py  training loop; checkpoints dense early, log-spaced later, and added
                          around sharp loss changes (with a rolling buffer of recent weights)
  data.py, prepare_data.py  tokenizer (one per corpus, shared by all seeds) and corpus preparation
  synthetic.py, encoding.py  templated agreement grammar, minimal pairs, subject-number counterfactuals
  evals/agreement.py      agreement minimal-pair evaluation
  probing/                activations, features, probes with control tasks, per-checkpoint runner
  interventions/          head ablation, LEACE erasure, activation patching, per-checkpoint runner
  analysis/               onset detection, lag estimates, bootstrap intervals, plots
tests/                    26 tests (pytest)
docs/proposal.md          project proposal
```

## Design choices worth knowing

- **Probes must generalize.** Train and test nouns are disjoint, so a probe cannot just memorize words.
  The baseline is the untrained model (step 0). Control-task probes (random labels per word) are
  computed too, but on WikiText they were as easy as the real task for some features, so decodability
  onsets use probe accuracy against the untrained baseline.
- **Erase at one position, not everywhere.** Erasing the number direction at every position hurt
  agreement whichever noun it was fit on (and raised general loss by 0.3-1.2 nats), so it could not
  show what the model relies on. Interventions now act at the position being tested: the subject, the
  last token before the verb, or a distractor noun.
- **A causal effect must be big enough to count.** It has to move the verb preference by at least 1
  nat (and a decodability gain must be at least 0.1 in accuracy) before an onset is reported.
- **Reproducible data.** WikiText-103 is read from a pinned version of the dataset repository, and the
  row counts are checked after loading.
- **Validation batches are fixed** across checkpoints and seeds, so loss curves are directly comparable.

## Status and next steps

Done: training pipeline, checkpointing, agreement evaluation, probing, interventions, lag analysis,
3 synthetic seeds, 1 WikiText-103 seed with full probing and causal analysis.

Next: more WikiText seeds (to put intervals on the lag), part-of-speech probes with spaCy labels (the
current part-of-speech probe is uninformative: it scores 100% even on the untrained model), a
longer-range dependency feature, and the final plots and report.
