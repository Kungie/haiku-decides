# haiku-decides

Claude Haiku 5.5 as a decision model, benchmarked head-to-head against Jev.

Anthropic has no decision API. `haiku-decides` emulates the decision-model
interface (choice, yes/no, score, with per-option probabilities) on top of
Claude Haiku 5.5 and measures how close it gets: accuracy, calibration, order
sensitivity, latency, and cost.

**Status:** design stage, no code yet. See the
[design spec](docs/superpowers/specs/2026-10-08-haiku-decides-design.md)
(in Turkish).

Not affiliated with Anthropic or TypeSafe AI.
