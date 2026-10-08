# haiku-decides

Claude Haiku 5.5 as a decision model, benchmarked head-to-head against Jev and
the OpenAI Decisions API.

Decision models answer a closed question about some input and return a typed
answer with a probability for every option. TypeSafe AI's Jev does this
natively, and so does OpenAI's Decisions API. Anthropic has no decision API.
`haiku-decides` emulates the same interface on top of Claude Haiku 5.5 in three
ways and measures how close each one gets.

## Results

The first full run is in progress. The scorecard and chart will appear here when
it finishes.

It covers 500 items per dataset for accuracy and calibration. To keep API spend
small, the order-sensitivity and latency passes use fewer items than the
defaults; the exact counts will be listed with the results.

## How it works

All three systems receive the same question in the same shape: a `state` to
judge and a question of type `choice`, `noul` (yes/no), or `score`.

Jev and the OpenAI Decisions API are called as they are. Claude Haiku 5.5 runs
with thinking disabled and an enum-constrained structured output, in three
modes:

| Mode | How it runs | What it shows |
|---|---|---|
| `single` | One call | What most people do today. No probabilities. |
| `sampled` | The same request N times in parallel; vote shares become probabilities | Whether you can get probabilities without logprobs |
| `shuffled` | N calls, each listing the options in a different order | What changes once order bias is averaged out |

Claude Haiku 5.5 only accepts a temperature of 1, so N samples are honest draws
from the model's own answer distribution.

Four things are measured:

- **Accuracy** against the dataset's label.
- **Calibration**: when a system says 80%, is it right 80% of the time?
  Reported as expected calibration error (10 bins) and Brier score.
- **Order sensitivity**: how often the answer flips, and how far the
  probabilities move, across five different orderings of the same options. A
  "repeat flip" column shows how often the answer changes between two runs with
  the same order, so you can see how much of a flip rate is plain sampling noise.
- **Latency and cost**: p50 and p95 per decision, and dollars per 1,000
  decisions.

Accuracy and calibration come with 95% bootstrap intervals. Accuracy verdicts
test the paired difference on the items every system answered, and say "not
distinguishable" when that difference could be zero. Brier is the multiclass
form: 0 is perfect, 2 is worst.

## Reproduce

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env
```

Fill in `.env` with your keys, then:

```bash
.venv/bin/haiku-decides select-ids
.venv/bin/haiku-decides run
.venv/bin/haiku-decides report --location "your city"
```

`run` appends one JSON line per decision under `results/` and resumes where it
stopped if you interrupt it. To try it on a few items first, add
`--limit 5 --results results-dry`.

The tests need no keys and no network:

```bash
.venv/bin/python -m pytest -q
```

## Datasets

500 items from each, chosen with a fixed seed. The chosen row indices are in
`data/ids/`.

| Dataset | Hugging Face id | Question type | Declared license |
|---|---|---|---|
| Banking77 | `legacy-datasets/banking77` | choice, 77 options | cc-by-4.0 |
| AG News | `fancyzhx/ag_news` | choice, 4 options | unknown |
| BoolQ | `google/boolq` | yes/no | cc-by-sa-3.0 |
| SMS Spam | `ucirvine/sms_spam` | yes/no | unknown |
| SST-5 | `SetFit/sst5` | score, 5 levels | not stated |

This repository ships row indices and model outputs only, never dataset text.
The text is downloaded from Hugging Face when you run the benchmark.

## Limitations

- These datasets are very likely in the training data of every model tested.
- With N=10, probabilities come in steps of 0.1, so the calibration numbers for
  the sampled modes are coarse.
- Latency is measured from one location.
- Claude Haiku 5.5 is not deterministic: the same input can produce a different
  decision.
- Only the option list in the prompt is reordered. The enum in the output schema
  stays sorted, so order sensitivity for Claude Haiku 5.5 measures the prompt
  alone.
- The OpenAI Decisions API is in public beta and may change.
- The latency pass re-sends questions the main pass already asked. If a provider
  caches responses, its latency here is flattering.
- Cost is per answered decision at list prices, with whatever prompt caching
  each run happened to get.
- `report` reads each dataset from the local Hugging Face cache to get the
  labels, so it needs the datasets to have been downloaded once (`select-ids`
  does that). Dataset revisions are pinned.

## License

MIT. See [LICENSE](LICENSE).

Not affiliated with Anthropic, OpenAI or TypeSafe AI.
