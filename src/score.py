"""Scoring harness.

Matches normalized Findings against the ground-truth defect ledger and computes
per-arm, per-category metrics.

Two properties this module is built to preserve:

1. **Blindness.** `match_findings` receives findings and defects. It does not
   receive the arm. Arm is derived afterwards, from reviewer vs target, so no
   matching decision can be influenced by which arm is being scored.

2. **No double counting.** Each ground-truth defect can be claimed at most once
   per review run, via greedy assignment ordered by match quality. A reviewer
   that carpet-bombs a file cannot inflate recall.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .schema import Arm, Category, Defect, Finding

from .score_constants import LINE_TOLERANCE


@dataclass
class MatchResult:
    matched_defect_ids: set[str] = field(default_factory=set)
    true_positives: list[Finding] = field(default_factory=list)
    false_positives: list[Finding] = field(default_factory=list)


def _match_quality(finding: Finding, defect: Defect) -> Optional[tuple[int, int]]:
    """Lower is better. None means no match.

    Sort key is (category_mismatch, line_distance): an exact-category match at
    distance 3 beats a category mismatch at distance 0.
    """
    if finding.file != defect.file:
        return None

    if defect.line_start <= finding.line <= defect.line_end:
        distance = 0
    else:
        distance = min(
            abs(finding.line - defect.line_start),
            abs(finding.line - defect.line_end),
        )
    if distance > LINE_TOLERANCE:
        return None

    category_mismatch = 0 if finding.category == defect.category else 1
    return (category_mismatch, distance)


def match_findings(
    findings: Iterable[Finding],
    defects: Iterable[Defect],
    require_category_match: bool = False,
) -> MatchResult:
    """Greedy best-first assignment of findings to defects.

    `require_category_match=True` is the strict variant: a finding on the right
    line but with the wrong category does not count. The default (loose) credits
    it, on the grounds that a reviewer flagging the right line for a slightly
    mislabelled reason has still done its job. Both are reported.
    """
    findings = list(findings)
    defects = list(defects)

    candidates = []
    for finding in findings:
        for defect in defects:
            quality = _match_quality(finding, defect)
            if quality is None:
                continue
            if require_category_match and quality[0] != 0:
                continue
            candidates.append((quality, finding, defect))

    candidates.sort(key=lambda c: c[0])

    result = MatchResult()
    claimed_findings: set[str] = set()

    for _, finding, defect in candidates:
        if finding.finding_id in claimed_findings:
            continue
        if defect.defect_id in result.matched_defect_ids:
            continue
        claimed_findings.add(finding.finding_id)
        result.matched_defect_ids.add(defect.defect_id)
        finding.matched_defect_id = defect.defect_id
        result.true_positives.append(finding)

    result.false_positives = [f for f in findings if f.finding_id not in claimed_findings]
    return result


# --- Statistics ------------------------------------------------------------

def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Used instead of the normal approximation because per-category cells are
    small and the normal interval misbehaves badly near 0 and 1 — which is
    exactly where several categories will land.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass
class ArmMetrics:
    arm: str
    reviewer: str
    target_generator: str
    defects_total: int = 0
    defects_caught: int = 0
    findings_total: int = 0
    findings_true: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def recall(self) -> float:
        return self.defects_caught / self.defects_total if self.defects_total else 0.0

    @property
    def precision(self) -> float:
        return self.findings_true / self.findings_total if self.findings_total else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Fraction of findings that map to no ground-truth defect.

        Note this is 1 - precision, not a classical FPR: "true negatives" are
        not well defined when the candidate space is every line of code. It is
        reported because it is the number engineers actually feel — the share of
        review comments that waste their time.
        """
        return 1.0 - self.precision if self.findings_total else 0.0

    @property
    def cost_per_catch(self) -> Optional[float]:
        return self.cost_usd / self.defects_caught if self.defects_caught else None

    def recall_ci(self) -> tuple[float, float]:
        return wilson_interval(self.defects_caught, self.defects_total)

    def precision_ci(self) -> tuple[float, float]:
        return wilson_interval(self.findings_true, self.findings_total)


def score_runs(
    findings_by_run: dict[tuple[str, str, str], list[Finding]],
    defects: list[Defect],
    costs: Optional[dict[tuple[str, str, str], tuple[int, int, float]]] = None,
    require_category_match: bool = False,
) -> dict[tuple[str, str], ArmMetrics]:
    """Score every (reviewer, target_generator) pair across all tasks.

    `findings_by_run` is keyed (task_id, reviewer, target_generator).
    Returns metrics keyed (reviewer, target_generator).
    """
    defects_by_gen: dict[tuple[str, str], list[Defect]] = defaultdict(list)
    for defect in defects:
        defects_by_gen[(defect.task_id, defect.origin)].append(defect)

    metrics: dict[tuple[str, str], ArmMetrics] = {}

    for (task_id, reviewer, target), findings in findings_by_run.items():
        key = (reviewer, target)
        if key not in metrics:
            arm = Arm.SELF if reviewer == target else Arm.CROSS
            metrics[key] = ArmMetrics(arm=arm.value, reviewer=reviewer, target_generator=target)

        task_defects = defects_by_gen.get((task_id, target), [])
        result = match_findings(findings, task_defects, require_category_match)

        m = metrics[key]
        m.defects_total += len(task_defects)
        m.defects_caught += len(result.matched_defect_ids)
        m.findings_total += len(findings)
        m.findings_true += len(result.true_positives)

        if costs and (task_id, reviewer, target) in costs:
            tin, tout, usd = costs[(task_id, reviewer, target)]
            m.input_tokens += tin
            m.output_tokens += tout
            m.cost_usd += usd

    return metrics


def score_by_category(
    findings_by_run: dict[tuple[str, str, str], list[Finding]],
    defects: list[Defect],
    require_category_match: bool = False,
) -> dict[tuple[str, str, str], ArmMetrics]:
    """Same as score_runs, split per defect category.

    Keyed (reviewer, target_generator, category). This is the table that
    actually tests the hypothesis — aggregate recall hides which categories
    the cross-model effect lives in.
    """
    out: dict[tuple[str, str, str], ArmMetrics] = {}

    defects_by_gen: dict[tuple[str, str], list[Defect]] = defaultdict(list)
    for defect in defects:
        defects_by_gen[(defect.task_id, defect.origin)].append(defect)

    for (task_id, reviewer, target), findings in findings_by_run.items():
        task_defects = defects_by_gen.get((task_id, target), [])
        # Match once over the full set, then attribute by the defect's category
        # rather than the finding's — the ledger defines ground truth.
        result = match_findings(list(findings), task_defects, require_category_match)
        caught_ids = result.matched_defect_ids

        # A true positive is attributed to its MATCHED DEFECT's category, not
        # the category the reviewer wrote. Under loose matching a finding can
        # match a defect of a different category; attributing by the finding's
        # own label would put the catch in one category's recall and a
        # different category's precision, making the two tables inconsistent.
        defect_category = {d.defect_id: d.category for d in task_defects}

        for category in Category:
            key = (reviewer, target, category.value)
            if key not in out:
                arm = Arm.SELF if reviewer == target else Arm.CROSS
                out[key] = ArmMetrics(arm=arm.value, reviewer=reviewer, target_generator=target)

            cat_defects = [d for d in task_defects if d.category == category]
            out[key].defects_total += len(cat_defects)
            out[key].defects_caught += sum(1 for d in cat_defects if d.defect_id in caught_ids)

            tp_in_cat = sum(
                1 for f in result.true_positives
                if f.matched_defect_id and defect_category.get(f.matched_defect_id) == category
            )
            fp_in_cat = sum(
                1 for f in result.false_positives if f.category == category
            )
            out[key].findings_true += tp_in_cat
            out[key].findings_total += tp_in_cat + fp_in_cat

    return out


def symmetry_check(metrics: dict[tuple[str, str], ArmMetrics]) -> dict:
    """The experiment's central control.

    If cross-review beats self-review in BOTH directions, the effect is model
    diversity. If it beats self-review in only one direction, the effect is
    capability — one model is simply the better reviewer — and the diversity
    claim is not supported.

    Returns the verdict plus the per-direction deltas behind it.
    """
    generators = sorted({t for (_, t) in metrics})
    if len(generators) != 2:
        return {"verdict": "inconclusive", "reason": "symmetry check requires exactly two generators"}

    a, b = generators
    needed = [(a, a), (b, b), (b, a), (a, b)]
    missing = [k for k in needed if k not in metrics]
    if missing:
        return {"verdict": "inconclusive", "reason": f"missing runs: {missing}"}

    delta_a = metrics[(b, a)].recall - metrics[(a, a)].recall  # cross vs self on A's code
    delta_b = metrics[(a, b)].recall - metrics[(b, b)].recall  # cross vs self on B's code

    if delta_a > 0 and delta_b > 0:
        verdict = "diversity"
    elif delta_a <= 0 and delta_b <= 0:
        verdict = "no-effect"
    else:
        verdict = "capability-confound"

    return {
        "verdict": verdict,
        f"delta_on_{a}_code": round(delta_a, 4),
        f"delta_on_{b}_code": round(delta_b, 4),
        "interpretation": {
            "diversity": "Cross-review beat self-review in both directions. Supports the decorrelation hypothesis.",
            "no-effect": "Cross-review did not beat self-review in either direction.",
            "capability-confound": "Cross-review won in only one direction — consistent with one model simply being a better reviewer, not with diversity.",
        }[verdict],
    }
