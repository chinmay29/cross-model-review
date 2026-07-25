"""Render results as markdown tables matching the write-up's structure.

Output drops straight into Cross_Model_Code_Review.md, so there is no manual
transcription step between the harness and the document — which is one fewer
place for numbers to drift from what was actually measured.
"""

from __future__ import annotations

from .schema import Category
from .score import ArmMetrics, symmetry_check


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _ci(lo: float, hi: float) -> str:
    return f"[{100 * lo:.0f}–{100 * hi:.0f}]"


def overall_table(metrics: dict[tuple[str, str], ArmMetrics], base_defect_count: int) -> str:
    lines = [
        "**Overall detection, by arm**",
        "",
        "| Arm | Reviewer → Target | Recall | 95% CI | Precision | FPR | Cost / catch |",
        "|---|---|---|---|---|---|---|",
        f"| 1 — No review | — | 0.0% | — | — | — | — |",
    ]
    ordered = sorted(metrics.items(), key=lambda kv: (kv[1].arm, kv[0]))
    for (reviewer, target), m in ordered:
        lo, hi = m.recall_ci()
        label = "2 — Self" if m.arm == "self-review" else "3 — Cross"
        cost = f"${m.cost_per_catch:.3f}" if m.cost_per_catch is not None else "—"
        lines.append(
            f"| {label} | {reviewer} → {target} | {_pct(m.recall)} | {_ci(lo, hi)} "
            f"| {_pct(m.precision)} | {_pct(m.false_positive_rate)} | {cost} |"
        )
    lines.append("")
    lines.append(f"_Ground-truth defects in ledger: {base_defect_count}._")
    return "\n".join(lines)


def category_table(
    by_cat: dict[tuple[str, str, str], ArmMetrics],
    generators: list[str],
) -> str:
    if len(generators) != 2:
        return "_Category table requires exactly two generators._"
    a, b = generators

    lines = [
        "**Recall by defect category — self vs cross**",
        "",
        f"| Category | Self ({a}→{a}) | Cross ({b}→{a}) | Δ | Self ({b}→{b}) | Cross ({a}→{b}) | Δ |",
        "|---|---|---|---|---|---|---|",
    ]

    for category in Category:
        c = category.value
        self_a = by_cat.get((a, a, c))
        cross_a = by_cat.get((b, a, c))
        self_b = by_cat.get((b, b, c))
        cross_b = by_cat.get((a, b, c))

        def cell(m: ArmMetrics | None) -> str:
            if m is None or m.defects_total == 0:
                return "n/a"
            return f"{_pct(m.recall)} ({m.defects_caught}/{m.defects_total})"

        def delta(s: ArmMetrics | None, x: ArmMetrics | None) -> str:
            if not s or not x or s.defects_total == 0 or x.defects_total == 0:
                return "—"
            d = x.recall - s.recall
            return f"{'+' if d >= 0 else ''}{100 * d:.1f}pp"

        lines.append(
            f"| `{c}` | {cell(self_a)} | {cell(cross_a)} | {delta(self_a, cross_a)} "
            f"| {cell(self_b)} | {cell(cross_b)} | {delta(self_b, cross_b)} |"
        )

    lines.append("")
    lines.append(
        "_Cells marked n/a had no ground-truth defects in that category. "
        "Small cells are reported as caught/total so the reader can judge weight._"
    )
    return "\n".join(lines)


def symmetry_section(metrics: dict[tuple[str, str], ArmMetrics]) -> str:
    check = symmetry_check(metrics)
    lines = ["**Symmetry check — the central control**", ""]
    if check["verdict"] == "inconclusive":
        lines.append(f"Inconclusive: {check.get('reason', 'unknown')}")
        return "\n".join(lines)

    for key, value in check.items():
        if key.startswith("delta_on_"):
            gen = key.replace("delta_on_", "").replace("_code", "")
            lines.append(f"- Cross − self on **{gen}**'s code: {'+' if value >= 0 else ''}{100 * value:.1f}pp")
    lines.append("")
    lines.append(f"**Verdict: `{check['verdict']}`.** {check['interpretation']}")
    return "\n".join(lines)


def full_report(
    metrics: dict[tuple[str, str], ArmMetrics],
    by_cat: dict[tuple[str, str, str], ArmMetrics],
    generators: list[str],
    base_defect_count: int,
    strict_metrics: dict[tuple[str, str], ArmMetrics] | None = None,
) -> str:
    parts = [
        "## Results",
        "",
        overall_table(metrics, base_defect_count),
        "",
        category_table(by_cat, generators),
        "",
        symmetry_section(metrics),
    ]
    if strict_metrics:
        parts += [
            "",
            "**Strict matching (category must also match)**",
            "",
            overall_table(strict_metrics, base_defect_count),
            "",
            "_Loose matching credits a finding on the right line with a mislabelled "
            "category; strict does not. Both are reported because the gap between "
            "them says how well reviewers characterize what they find, not just "
            "whether they locate it._",
        ]
    return "\n".join(parts)
