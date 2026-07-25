# Cross-Model Adversarial Code Review

Does a reviewer from a *different* model family catch defects that same-model
self-review misses?

AI-assisted pipelines increasingly close the loop with one model on both ends:
a model generates code, then reviews or repairs it. This harness measures
whether that is a mistake, and if so, by how much and in which defect
categories.

## Hypothesis

A model's blind spots are correlated with its own output. If a model failed to
consider an edge case while generating, the same latent gap likely persists
while reviewing that generation. Routing review across model families should
decorrelate those errors.

Three lines of prior work motivate this:

- **Self-preference bias** — LLM evaluators recognize their own generations and
  score them above what human annotators give them, with the bias linearly
  correlated to self-recognition ability (Panickssery et al., NeurIPS 2024,
  [2404.13076](https://arxiv.org/abs/2404.13076)).
- **Weak self-repair** — self-repair gains are modest, highly variable, and
  sometimes absent once the cost of the repair pass is counted
  (Olausson et al., [2306.09896](https://arxiv.org/abs/2306.09896)).
- **The popularity trap** — models trained on similar distributions converge on
  the same plausible-but-wrong answers, so consensus filtering amplifies shared
  errors; diversity-based selection recovered up to 95% of an ideal independent
  ensemble's gain (Vallecillos-Ruiz et al., [2510.21513](https://arxiv.org/abs/2510.21513)).

**The counterweight, which the design takes seriously:** heterogeneous agent
teams have been found to underperform their best individual member — losses up
to 37.6% — when they deliberate toward consensus instead of deferring to
expertise (Pappu et al., [2602.01011](https://arxiv.org/abs/2602.01011)). So
reviewers here run **independently and never see each other's findings**. This
is inspection, not committee.

## The design decision that matters most

The experiment runs **A→B and B→A**.

If cross-review beats self-review in only one direction, the effect is
*capability* — one model is simply the better reviewer — and the diversity
claim is unsupported. Only if **both** directions improve over their respective
self-review arms is the effect attributable to model diversity.

`src/score.py:symmetry_check` reports one of three verdicts: `diversity`,
`capability-confound`, or `no-effect`. A naive comparison that ran only one
direction would report the confounded case as a win.

## Other design constraints

| Constraint | Why |
|---|---|
| Prompts byte-identical across all four configurations | Otherwise prompt differences masquerade as review-quality differences |
| Reviewers not told who wrote the code | Tests blind-spot correlation, not explicit self-preference; also matches production, where review agents aren't told either |
| Full reviewer × generation cross product in one run | Self and cross arms produced under identical conditions — no batch drift between arms |
| Scorer never sees the arm | Arm is derived after matching, so no matching decision can be arm-influenced |
| Each defect claimable once per run | A reviewer that carpet-bombs a file can't inflate recall |
| Temperature 0 everywhere | Rerun variance would otherwise confound the measured effect |
| Ground truth requires human confirmation | An LLM-defined ledger makes the experiment circular — see below |

## The ledger is not automatable

`run.py audit` proposes candidate defects, then stops. Every candidate is
written with `confirmed: false`, and `run.py score` refuses to run until each
one is resolved by hand.

This is not caution for its own sake. If an LLM defines ground truth for an
experiment about LLM review quality, the experiment measures inter-model
agreement rather than defect detection.

**Two auditors, unioned.** Ideally the auditor would be a third model family, so
the ledger is not biased toward defects either generator's family can already
see. With only two providers available, the harness runs *both* families as
auditors and unions their candidates — no single family's blind spots then
determine what counts as ground truth. Each merged entry records which auditors
proposed it in `proposed_by`.

That field is where to spend your confirmation effort. A candidate both
families flagged is very likely real. A candidate only one family flagged is
either a false positive **or a defect the other family cannot see** — and the
second case is the phenomenon under study, so those entries deserve the most
careful attention rather than a quick reject. `audit` prints the agreement
breakdown before you start.

## Model configuration

Set both generators to comparable tiers in `config.yaml`. A frontier model
against a cheap one turns the symmetry check into a capability measurement.
Mid-tier is also better for a second reason: frontier models produce fewer
natural defects, and a ledger with eight entries has nothing to measure. You
want code that is good but not flawless.

Reviewers **must** be the same two models as the generators — the self-review
arm is only "self" if reviewer and generator match. Auditors should be the
strongest model from each family, since the auditor's job is recall and the
human pass filters its false positives.

Fill in OpenAI pricing before publishing; the shipped values are placeholders
and a stale number quietly makes the cost column fiction.

## Metrics

- **Recall** per arm and per defect category, with Wilson score intervals
  (per-category cells are small; the normal approximation misbehaves near 0
  and 1, which is where several categories will land)
- **Precision**, weighted equally with recall by design — a reviewer with a
  high false-positive rate gets muted by engineers within a week, and a muted
  reviewer has zero value regardless of what it catches
- **FPR**, reported as the share of findings mapping to no ground-truth defect.
  This is `1 - precision`, not a classical FPR — "true negatives" aren't well
  defined when the candidate space is every line of code. It's reported because
  it's the number engineers actually feel.
- **Cost per true defect caught**, in tokens and dollars
- **Loose vs strict matching** — loose credits a finding on the right line with
  a mislabelled category; strict doesn't. The gap between them measures how well
  reviewers *characterize* what they find, not just whether they locate it.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY and OPENAI_API_KEY

python run.py doctor          # one cheap call per model — run this first
python run.py generate        # both models implement every task
python run.py audit           # propose candidates  ->  CONFIRM BY HAND
python run.py review          # 4-way matrix: self + cross, both directions
python run.py score           # match, compute metrics, render report
```

Offline dry run, no keys and no spend:

```bash
python run.py generate --mock && python run.py audit --mock \
  && python run.py review --mock && python run.py score --allow-unconfirmed
```

`--allow-unconfirmed` exists for smoke tests only. Never use it for numbers you
intend to publish.

### Troubleshooting

`python run.py doctor` makes one 32-token call per configured model and prints
the provider's own error text on failure. Run it before any paid stage.

HTTP 400 from Anthropic is most often the explicit `temperature`: models with
thinking enabled reject it. Set `temperature: null` in `config.yaml` to omit the
parameter. Note this weakens determinism slightly — already listed under threats
to validity, since neither family is fully deterministic at temperature 0 anyway.

### Failed runs are not zero-finding runs

Three failure modes are recorded as **errors**, never as clean reviews:
API failures, output truncation (`stop_reason: max_tokens` / `finish_reason:
length` — on reasoning models the whole budget can go to reasoning, leaving
empty content), and unparsable output (non-empty response, no JSON).

`score` refuses to run while errored runs are present, because a run missing
from one arm biases the self-vs-cross comparison — the affected model's recall
would be deflated by its formatting or verbosity rather than its review
ability. Fix the cause, re-run `review`, then score.

```bash
python -m pytest tests/ -q
```

## Layout

```
config.yaml           models, pricing, paths
tasks/tasks.yaml      task corpus + repo conventions
prompts/              generation and review prompts (identical across arms)
src/schema.py         taxonomy, Finding/Defect, arm derivation
src/clients.py        symmetric Anthropic/OpenAI clients + offline mock
src/pipeline.py       generation and the reviewer × generation matrix
src/ledger.py         ground truth: multi-auditor union + confirmation gate
src/normalize.py      model output -> canonical Findings, dedupe
src/score.py          blind matching, Wilson intervals, symmetry check
src/report.py         markdown tables for the write-up
tests/                43 tests over matcher, dedupe, parsing, statistics, union,
                      truncation, parse-failure, and category attribution
```

## Threats to validity

- **Sample size.** Ten tasks yields per-category cells that are thin. Intervals
  are reported; single-category claims should be made cautiously. Expand
  `tasks/tasks.yaml` before drawing category-level conclusions.
- **Capability confound.** Mitigated by the symmetric design, not eliminated —
  if the two models differ substantially in raw review ability, a genuine
  diversity effect is attenuated rather than cleanly isolated.
- **Task representativeness.** These are synthetic tasks, not real PRs against a
  mature codebase with years of accumulated context. The direction of the effect
  should transfer; the magnitude probably won't.
- **Prompt sensitivity.** Review quality is prompt-dependent. Prompts are held
  identical across arms, which controls the comparison but makes the absolute
  numbers a floor rather than a ceiling.
- **Imperfect determinism.** Temperature is 0 everywhere, but models with
  adaptive thinking or reasoning modes are not fully deterministic even at 0.
  `reasoning_effort` is set explicitly and recorded rather than left to a
  provider default, but run-to-run variance is not zero.
- **Auditor family overlap.** Both auditors share a family with one of the two
  generators. The union mitigates this; it does not eliminate it. A genuine
  third-family auditor would be better.
- **Contamination.** Task specs are original, but idioms in generated solutions
  may still echo training data.

## What's next

Route review by defect category rather than uniformly — if the per-category
table shows each family has distinct strengths, static cross-routing leaves
value on the table. Add a third family to test whether the effect scales with
ensemble diversity or saturates at two. And measure the thing that actually
matters in production: not benchmark detection rate, but whether engineers
*act* on the findings.
