"""Ground-truth defect ledger.

This is the part of the experiment that cannot be fully automated, and the
reason is not effort — it is circularity. If an LLM defines the ground truth
for an experiment about LLM review quality, the experiment measures agreement
between models rather than defect detection. Any candidate proposed here is
UNCONFIRMED until a human marks it otherwise, and `load_ledger` refuses to
return unconfirmed entries unless explicitly asked.

Two sources of ground truth, both supported:

  source: "audit"     A real defect found in generated code (test failure or
                      human review). Attributed to the generator that wrote it.
                      These are the ones that test the hypothesis.

  source: "injected"  A defect deliberately introduced into known-good code, to
                      cover categories that generated code rarely produces
                      naturally (races, resource leaks). These fill out the
                      per-category table but do not carry generator attribution
                      in the same meaningful way — flagged so they can be
                      excluded from the headline result.
"""

from __future__ import annotations

import os
from typing import Optional

import yaml

from .clients import ModelClient
from .normalize import extract_json, number_code
from .schema import Category, Defect, Severity, stable_id


PROPOSAL_SYSTEM = """You are auditing code to build a defect ledger for a research benchmark.

List every defect you are confident is real. For each, give the exact line span.
Be exhaustive about correctness and resource handling. Ignore pure style.

Respond with a single JSON object, no prose, no fences:
{"defects": [{"category": "...", "line_start": N, "line_end": N, "severity": "...", "description": "..."}]}

Categories: logic-error, missing-edge-case, null-boundary, concurrency,
resource-leak, convention-violation, untested-branch, security-relevant
"""


def propose_candidates(
    auditor: ModelClient,
    task_id: str,
    origin: str,
    code: str,
    filename: str,
    task_spec: str,
    conventions: str,
) -> list[dict]:
    """Propose candidate defects for human confirmation.

    The auditor SHOULD be a third model family, distinct from both generators.
    Using either generator as auditor would bias the ledger toward defects that
    that model is capable of seeing — precisely the variable under test.

    Returns dicts with `confirmed: false`. Nothing here counts until a human
    flips that flag.
    """
    user = (
        f"### Task\n{task_spec}\n\n"
        f"### Conventions\n{conventions}\n\n"
        f"### Code (`{filename}`)\n```\n{number_code(code)}\n```"
    )
    response = auditor.complete(PROPOSAL_SYSTEM, user)
    payload = extract_json(response.text) or {}

    candidates = []
    for item in payload.get("defects", []):
        if not isinstance(item, dict):
            continue
        line_start = int(item.get("line_start", 0) or 0)
        line_end = int(item.get("line_end", line_start) or line_start)
        description = str(item.get("description", "")).strip()
        if not description:
            continue
        candidates.append({
            "defect_id": stable_id(task_id, origin, str(line_start), description[:60]),
            "task_id": task_id,
            "origin": origin,
            "category": Category.coerce(str(item.get("category", ""))).value,
            "file": filename,
            "line_start": line_start,
            "line_end": max(line_end, line_start),
            "description": description,
            "severity": Severity.coerce(str(item.get("severity", ""))).value,
            "source": "audit",
            "confirmed": False,   # <-- human must flip this
        })
    return candidates


def merge_candidates(
    candidate_sets: dict[str, list[dict]],
    line_tolerance: int | None = None,
) -> list[dict]:
    """Union candidates from several auditors, merging duplicates.

    Two candidates describe the same defect if they share task, origin, file and
    category, and their line spans sit within `line_tolerance` of each other.
    Merged entries record every auditor that proposed them in `proposed_by`.

    That field is the useful part of running more than one auditor. A candidate
    both families flagged is very likely real. A candidate only one family
    flagged is either a false positive or a defect the other family cannot see —
    and the second case is exactly the phenomenon under study, so those entries
    deserve the most careful human attention rather than a quick reject.
    """
    if line_tolerance is None:
        from .score_constants import LINE_TOLERANCE
        line_tolerance = LINE_TOLERANCE

    merged: list[dict] = []

    for auditor_key, candidates in candidate_sets.items():
        for cand in candidates:
            match = None
            for existing in merged:
                if (
                    existing["task_id"] == cand["task_id"]
                    and existing["origin"] == cand["origin"]
                    and existing["file"] == cand["file"]
                    and existing["category"] == cand["category"]
                    and abs(existing["line_start"] - cand["line_start"]) <= line_tolerance
                ):
                    match = existing
                    break

            if match is None:
                entry = dict(cand)
                entry["proposed_by"] = [auditor_key]
                merged.append(entry)
            else:
                if auditor_key not in match["proposed_by"]:
                    match["proposed_by"].append(auditor_key)
                # Widen the span to the union, so a finding anywhere in the
                # region either auditor described still matches at scoring time.
                match["line_start"] = min(match["line_start"], cand["line_start"])
                match["line_end"] = max(match["line_end"], cand["line_end"])

    for entry in merged:
        entry["proposed_by"] = sorted(entry["proposed_by"])

    return merged


def auditor_agreement(entries: list[dict], n_auditors: int) -> dict:
    """How much the auditors agreed. Reported before the human pass begins.

    Low agreement is not a problem with the harness — it is a preview of the
    result. If two families propose largely disjoint defect sets on the same
    code, their blind spots differ, which is the premise the experiment tests.

    `n_auditors` must be the number of auditors that actually ran; inferring it
    from the entries would report "all agreed" whenever, say, only one auditor
    proposed anything at all.
    """
    total = len(entries)
    by_count: dict[int, int] = {}
    for entry in entries:
        n = len(entry.get("proposed_by", []))
        by_count[n] = by_count.get(n, 0) + 1
    return {
        "total_candidates": total,
        "proposed_by_all_auditors": by_count.get(n_auditors, 0),
        "single_auditor_only": by_count.get(1, 0) if n_auditors > 1 else 0,
        "by_auditor_count": dict(sorted(by_count.items())),
    }


def save_ledger(path: str, entries: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump({"defects": entries}, fh, sort_keys=False, allow_unicode=True)


def load_ledger(path: str, allow_unconfirmed: bool = False) -> list[Defect]:
    """Load confirmed defects.

    Raises if the ledger contains unconfirmed entries and `allow_unconfirmed`
    is False — a loud failure is correct here, because silently scoring against
    a half-confirmed ledger produces numbers that look fine and mean nothing.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    entries = raw.get("defects", []) or []
    unconfirmed = [e for e in entries if not e.get("confirmed", False)]

    if unconfirmed and not allow_unconfirmed:
        raise ValueError(
            f"{len(unconfirmed)} of {len(entries)} ledger entries are unconfirmed.\n"
            f"Review them in {path} and set `confirmed: true` on each one you accept, "
            f"or delete the ones you reject. Scoring against unconfirmed ground truth "
            f"makes the results meaningless."
        )

    usable = entries if allow_unconfirmed else [e for e in entries if e.get("confirmed")]

    return [
        Defect(
            defect_id=e["defect_id"],
            task_id=e["task_id"],
            origin=e["origin"],
            category=Category.coerce(e["category"]),
            file=e["file"],
            line_start=int(e["line_start"]),
            line_end=int(e["line_end"]),
            description=e["description"],
            severity=Severity.coerce(e.get("severity", "major")),
            source=e.get("source", "audit"),
        )
        for e in usable
    ]


def ledger_summary(defects: list[Defect]) -> dict:
    """Sanity numbers to check before spending money on review runs."""
    by_origin: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for d in defects:
        by_origin[d.origin] = by_origin.get(d.origin, 0) + 1
        by_category[d.category.value] = by_category.get(d.category.value, 0) + 1
        by_source[d.source] = by_source.get(d.source, 0) + 1
    return {
        "total": len(defects),
        "by_origin": by_origin,
        "by_category": by_category,
        "by_source": by_source,
    }
