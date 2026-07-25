"""Finding normalizer.

Models emit JSON with varying fidelity: fenced blocks, trailing prose,
categories outside the taxonomy, line numbers as strings. This module absorbs
all of that so the scorer receives one canonical shape.

Normalization happens before scoring and is identical for every arm. If it
were not, differences in output formatting between model families would leak
into the results as if they were differences in review quality.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from .schema import Category, Finding, Severity, stable_id

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a model response.

    Tries, in order: the whole string, a fenced block, then the widest
    brace-balanced span. Returns None if nothing parses.
    """
    if not text:
        return None

    candidates: list[str] = [text.strip()]

    for match in _FENCE.finditer(text):
        candidates.append(match.group(1).strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"findings": parsed}
        except json.JSONDecodeError:
            continue
    return None


def extract_code(text: str) -> str:
    """Pull a code block out of a generation response."""
    if not text:
        return ""
    fences = re.findall(r"```(?:\w+)?\s*(.*?)```", text, re.DOTALL)
    if fences:
        return fences[0].strip()
    return text.strip()


def _coerce_line(raw: Any) -> int:
    if isinstance(raw, int):
        return max(raw, 0)
    if isinstance(raw, str):
        digits = re.search(r"\d+", raw)
        if digits:
            return int(digits.group())
    if isinstance(raw, (list, tuple)) and raw:
        return _coerce_line(raw[0])
    return 0


class ParseFailure(ValueError):
    """The response was non-empty but yielded no parsable findings payload.

    Kept distinct from an empty findings list on purpose: 'the model reviewed
    the code and found nothing' is a data point, 'the model's output could not
    be read' is a broken run. Conflating them silently deflates recall for
    whichever model formats worse."""


def normalize_findings(
    raw_text: str,
    task_id: str,
    reviewer: str,
    target_generator: str,
    default_file: str,
) -> list[Finding]:
    """Turn one review response into canonical Findings.

    Raises ParseFailure when the response is non-empty but unparsable, so the
    caller records a broken run rather than a clean zero.
    """
    payload = extract_json(raw_text)
    if not payload:
        if raw_text and raw_text.strip():
            raise ParseFailure(
                f"non-empty response with no parsable JSON (first 200 chars): "
                f"{raw_text.strip()[:200]!r}"
            )
        return []

    raw_findings = payload.get("findings")
    if raw_findings is None:
        # Some responses return a bare list or use a different key.
        for key in ("issues", "defects", "results"):
            if key in payload:
                raw_findings = payload[key]
                break
    if not isinstance(raw_findings, list):
        return []

    findings: list[Finding] = []
    for item in raw_findings:
        if not isinstance(item, dict):
            continue
        category = Category.coerce(str(item.get("category", "")))
        line = _coerce_line(item.get("line"))
        file = str(item.get("file") or default_file).strip() or default_file
        rationale = str(item.get("rationale") or item.get("description") or "").strip()

        # A finding with no rationale is not actionable and is dropped. This
        # rule applies to every arm equally.
        if not rationale:
            continue

        findings.append(Finding(
            finding_id=stable_id(task_id, reviewer, target_generator, file, str(line), rationale[:80]),
            task_id=task_id,
            reviewer=reviewer,
            target_generator=target_generator,
            category=category,
            file=file,
            line=line,
            rationale=rationale,
            severity=Severity.coerce(str(item.get("severity", ""))),
        ))

    return dedupe(findings)


def dedupe(findings: list[Finding]) -> list[Finding]:
    """Collapse findings that point at the same defect.

    Two findings from the same reviewer collide when they share a category and
    sit within DEDUPE_TOLERANCE lines of each other. Without this, a reviewer
    that reports the same issue on three adjacent lines would score three true
    positives for one real defect.

    The tolerance deliberately equals the scorer's LINE_TOLERANCE: dedupe
    collapses anything the matcher would treat as pointing at one defect. A
    looser matcher than deduper would let near-duplicate findings each claim
    separate nearby defects.
    """
    from .score_constants import LINE_TOLERANCE as DEDUPE_TOLERANCE

    kept: list[Finding] = []
    for finding in sorted(findings, key=lambda f: (f.file, f.category.value, f.line)):
        collision = next(
            (
                k for k in kept
                if k.file == finding.file
                and k.category == finding.category
                and abs(k.line - finding.line) <= DEDUPE_TOLERANCE
            ),
            None,
        )
        if collision is None:
            kept.append(finding)
    return kept


def number_code(code: str) -> str:
    """Prefix each line with its number, so findings can cite lines."""
    lines = code.splitlines()
    width = len(str(len(lines))) if lines else 1
    return "\n".join(f"{str(i + 1).rjust(width)}| {line}" for i, line in enumerate(lines))
