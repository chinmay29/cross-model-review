## Results

**Overall detection, by arm**

| Arm | Reviewer → Target | Recall | 95% CI | Precision | FPR | Cost / catch |
|---|---|---|---|---|---|---|
| 1 — No review | — | 0.0% | — | — | — | — |
| 3 — Cross | claude → gpt | 29.8% | [19–44] | 100.0% | 0.0% | $0.055 |
| 3 — Cross | gpt → claude | 35.4% | [25–48] | 95.8% | 4.2% | $0.011 |
| 2 — Self | claude → claude | 24.6% | [16–36] | 100.0% | 0.0% | $0.031 |
| 2 — Self | gpt → gpt | 21.3% | [12–35] | 90.9% | 9.1% | $0.017 |

_Ground-truth defects in ledger: 112._

**Recall by defect category — self vs cross**

| Category | Self (claude→claude) | Cross (gpt→claude) | Δ | Self (gpt→gpt) | Cross (claude→gpt) | Δ |
|---|---|---|---|---|---|---|
| `logic-error` | 50.0% (6/12) | 50.0% (6/12) | +0.0pp | 44.4% (4/9) | 55.6% (5/9) | +11.1pp |
| `missing-edge-case` | 16.7% (3/18) | 16.7% (3/18) | +0.0pp | 13.3% (2/15) | 6.7% (1/15) | -6.7pp |
| `null-boundary` | 50.0% (1/2) | 0.0% (0/2) | -50.0pp | 0.0% (0/1) | 100.0% (1/1) | +100.0pp |
| `concurrency` | 16.7% (1/6) | 83.3% (5/6) | +66.7pp | 33.3% (1/3) | 66.7% (2/3) | +33.3pp |
| `resource-leak` | 25.0% (1/4) | 25.0% (1/4) | +0.0pp | 0.0% (0/2) | 50.0% (1/2) | +50.0pp |
| `convention-violation` | 15.8% (3/19) | 42.1% (8/19) | +26.3pp | 23.1% (3/13) | 23.1% (3/13) | +0.0pp |
| `untested-branch` | 100.0% (1/1) | 0.0% (0/1) | -100.0pp | 0.0% (0/3) | 33.3% (1/3) | +33.3pp |
| `security-relevant` | 0.0% (0/3) | 0.0% (0/3) | +0.0pp | 0.0% (0/1) | 0.0% (0/1) | +0.0pp |

_Cells marked n/a had no ground-truth defects in that category. Small cells are reported as caught/total so the reader can judge weight._

**Symmetry check — the central control**

- Cross − self on **claude**'s code: +10.8pp
- Cross − self on **gpt**'s code: +8.5pp

**Verdict: `diversity`.** Cross-review beat self-review in both directions. Supports the decorrelation hypothesis.

**Strict matching (category must also match)**

**Overall detection, by arm**

| Arm | Reviewer → Target | Recall | 95% CI | Precision | FPR | Cost / catch |
|---|---|---|---|---|---|---|
| 1 — No review | — | 0.0% | — | — | — | — |
| 3 — Cross | claude → gpt | 21.3% | [12–35] | 71.4% | 28.6% | $0.078 |
| 3 — Cross | gpt → claude | 30.8% | [21–43] | 83.3% | 16.7% | $0.013 |
| 2 — Self | claude → claude | 15.4% | [9–26] | 62.5% | 37.5% | $0.049 |
| 2 — Self | gpt → gpt | 14.9% | [7–28] | 63.6% | 36.4% | $0.024 |

_Ground-truth defects in ledger: 112._

_Loose matching credits a finding on the right line with a mislabelled category; strict does not. Both are reported because the gap between them says how well reviewers characterize what they find, not just whether they locate it._
