"""Tests for the parts where a silent bug would quietly corrupt the results.

The matcher and the symmetry check are the two places where a wrong answer
still looks like a plausible number, so they get the most coverage.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.normalize import dedupe, extract_json, normalize_findings  # noqa: E402
from src.schema import Category, Defect, Finding, Severity  # noqa: E402
from src.score import ArmMetrics, match_findings, symmetry_check, wilson_interval  # noqa: E402


def mk_finding(line, category=Category.LOGIC_ERROR, reviewer="a", target="a", fid=None):
    return Finding(
        finding_id=fid or f"f{line}{category.value}{reviewer}",
        task_id="t1", reviewer=reviewer, target_generator=target,
        category=category, file="x.py", line=line, rationale="r",
    )


def mk_defect(start, end=None, category=Category.LOGIC_ERROR, did="d1", origin="a"):
    return Defect(
        defect_id=did, task_id="t1", origin=origin, category=category,
        file="x.py", line_start=start, line_end=end or start, description="d",
    )


# --- matching --------------------------------------------------------------

def test_exact_line_and_category_matches():
    res = match_findings([mk_finding(10)], [mk_defect(10)])
    assert len(res.true_positives) == 1
    assert not res.false_positives


def test_within_tolerance_matches():
    res = match_findings([mk_finding(12)], [mk_defect(10)])
    assert len(res.true_positives) == 1


def test_outside_tolerance_does_not_match():
    res = match_findings([mk_finding(20)], [mk_defect(10)])
    assert not res.true_positives
    assert len(res.false_positives) == 1


def test_different_file_does_not_match():
    f = mk_finding(10)
    f.file = "other.py"
    res = match_findings([f], [mk_defect(10)])
    assert not res.true_positives


def test_one_defect_claimed_only_once():
    """Carpet-bombing a region must not inflate recall."""
    findings = [
        mk_finding(10, fid="f1"),
        mk_finding(11, fid="f2"),
        mk_finding(12, fid="f3"),
    ]
    res = match_findings(findings, [mk_defect(10)])
    assert len(res.matched_defect_ids) == 1
    assert len(res.true_positives) == 1
    assert len(res.false_positives) == 2


def test_one_finding_claims_only_one_defect():
    findings = [mk_finding(10, fid="f1")]
    defects = [mk_defect(10, did="d1"), mk_defect(11, did="d2")]
    res = match_findings(findings, defects)
    assert len(res.true_positives) == 1
    assert len(res.matched_defect_ids) == 1


def test_category_match_preferred_over_proximity():
    """An exact-category match further away beats a wrong-category match closer in."""
    findings = [
        mk_finding(10, category=Category.CONCURRENCY, fid="wrong-cat-close"),
        mk_finding(13, category=Category.LOGIC_ERROR, fid="right-cat-far"),
    ]
    res = match_findings(findings, [mk_defect(10, category=Category.LOGIC_ERROR)])
    assert res.true_positives[0].finding_id == "right-cat-far"


def test_strict_mode_rejects_category_mismatch():
    res = match_findings(
        [mk_finding(10, category=Category.CONCURRENCY)],
        [mk_defect(10, category=Category.LOGIC_ERROR)],
        require_category_match=True,
    )
    assert not res.true_positives


def test_loose_mode_credits_category_mismatch():
    res = match_findings(
        [mk_finding(10, category=Category.CONCURRENCY)],
        [mk_defect(10, category=Category.LOGIC_ERROR)],
        require_category_match=False,
    )
    assert len(res.true_positives) == 1


# --- dedupe ----------------------------------------------------------------

def test_dedupe_collapses_adjacent_same_category():
    findings = [mk_finding(10, fid="a"), mk_finding(11, fid="b"), mk_finding(12, fid="c")]
    assert len(dedupe(findings)) == 1


def test_dedupe_keeps_distinct_categories():
    findings = [
        mk_finding(10, category=Category.LOGIC_ERROR, fid="a"),
        mk_finding(10, category=Category.CONCURRENCY, fid="b"),
    ]
    assert len(dedupe(findings)) == 2


def test_dedupe_keeps_distant_same_category():
    findings = [mk_finding(10, fid="a"), mk_finding(50, fid="b")]
    assert len(dedupe(findings)) == 2


# --- parsing ---------------------------------------------------------------

def test_extract_json_bare():
    assert extract_json('{"findings": []}') == {"findings": []}


def test_extract_json_fenced():
    assert extract_json('```json\n{"findings": []}\n```') == {"findings": []}


def test_extract_json_with_prose():
    got = extract_json('Here you go:\n{"findings": [{"a": 1}]}\nHope that helps!')
    assert got["findings"][0]["a"] == 1


def test_extract_json_bare_list_wrapped():
    assert extract_json('[{"a": 1}]') == {"findings": [{"a": 1}]}


def test_extract_json_garbage_returns_none():
    assert extract_json("no json at all") is None


def test_normalize_drops_findings_without_rationale():
    raw = '{"findings": [{"category": "logic-error", "line": 3, "rationale": ""}]}'
    assert normalize_findings(raw, "t1", "a", "b", "x.py") == []


def test_normalize_coerces_string_line_numbers():
    raw = '{"findings": [{"category": "logic-error", "line": "line 42", "rationale": "x"}]}'
    out = normalize_findings(raw, "t1", "a", "b", "x.py")
    assert out[0].line == 42


def test_category_coercion_handles_aliases():
    assert Category.coerce("race condition") == Category.CONCURRENCY
    assert Category.coerce("EDGE_CASE") == Category.MISSING_EDGE_CASE
    assert Category.coerce("nonsense") == Category.LOGIC_ERROR


# --- statistics ------------------------------------------------------------

def test_wilson_zero_total():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_brackets_point_estimate():
    lo, hi = wilson_interval(5, 10)
    assert lo < 0.5 < hi


def test_wilson_stays_in_bounds_at_extremes():
    lo, hi = wilson_interval(0, 8)
    assert lo >= 0.0
    lo, hi = wilson_interval(8, 8)
    assert hi <= 1.0


def test_wilson_narrows_with_more_data():
    small_lo, small_hi = wilson_interval(5, 10)
    big_lo, big_hi = wilson_interval(500, 1000)
    assert (big_hi - big_lo) < (small_hi - small_lo)


# --- symmetry check --------------------------------------------------------

def _metrics(recalls: dict[tuple[str, str], tuple[int, int]]):
    out = {}
    for (reviewer, target), (caught, total) in recalls.items():
        m = ArmMetrics(
            arm="self-review" if reviewer == target else "cross-review",
            reviewer=reviewer, target_generator=target,
        )
        m.defects_caught, m.defects_total = caught, total
        out[(reviewer, target)] = m
    return out


def test_symmetry_diversity_when_both_directions_improve():
    verdict = symmetry_check(_metrics({
        ("a", "a"): (4, 10), ("b", "a"): (7, 10),
        ("b", "b"): (5, 10), ("a", "b"): (8, 10),
    }))
    assert verdict["verdict"] == "diversity"


def test_symmetry_flags_capability_confound_when_only_one_direction_improves():
    """This is the case a naive comparison would report as a win."""
    verdict = symmetry_check(_metrics({
        ("a", "a"): (4, 10), ("b", "a"): (9, 10),   # cross much better
        ("b", "b"): (8, 10), ("a", "b"): (5, 10),   # cross worse
    }))
    assert verdict["verdict"] == "capability-confound"


def test_symmetry_no_effect_when_neither_improves():
    verdict = symmetry_check(_metrics({
        ("a", "a"): (6, 10), ("b", "a"): (5, 10),
        ("b", "b"): (6, 10), ("a", "b"): (4, 10),
    }))
    assert verdict["verdict"] == "no-effect"


def test_symmetry_inconclusive_with_missing_runs():
    verdict = symmetry_check(_metrics({("a", "a"): (4, 10), ("b", "b"): (5, 10)}))
    assert verdict["verdict"] == "inconclusive"


# --- auditor union ---------------------------------------------------------

from src.ledger import auditor_agreement, merge_candidates  # noqa: E402


def mk_cand(line, category="logic-error", task="t1", origin="a", file="x.py"):
    return {
        "defect_id": f"d{line}{category}", "task_id": task, "origin": origin,
        "category": category, "file": file, "line_start": line, "line_end": line,
        "description": "d", "severity": "major", "source": "audit", "confirmed": False,
    }


def test_merge_unions_disjoint_candidates():
    merged = merge_candidates({"a": [mk_cand(10)], "b": [mk_cand(50)]})
    assert len(merged) == 2


def test_merge_collapses_agreed_candidates_and_records_both():
    merged = merge_candidates({"a": [mk_cand(10)], "b": [mk_cand(11)]})
    assert len(merged) == 1
    assert merged[0]["proposed_by"] == ["a", "b"]


def test_merge_keeps_different_categories_apart():
    merged = merge_candidates({
        "a": [mk_cand(10, "logic-error")],
        "b": [mk_cand(10, "concurrency")],
    })
    assert len(merged) == 2


def test_merge_keeps_different_origins_apart():
    """The same defect in two models' code is two ground-truth entries."""
    merged = merge_candidates({
        "a": [mk_cand(10, origin="claude")],
        "b": [mk_cand(10, origin="gpt")],
    })
    assert len(merged) == 2


def test_merge_widens_span_to_union():
    c1 = mk_cand(10); c1["line_end"] = 12
    c2 = mk_cand(11); c2["line_start"] = 9; c2["line_end"] = 11
    merged = merge_candidates({"a": [c1], "b": [c2]})
    assert merged[0]["line_start"] == 9
    assert merged[0]["line_end"] == 12


def test_merge_does_not_duplicate_same_auditor():
    merged = merge_candidates({"a": [mk_cand(10), mk_cand(11)]})
    assert merged[0]["proposed_by"] == ["a"]


def test_agreement_counts_single_auditor_candidates():
    merged = merge_candidates({"a": [mk_cand(10), mk_cand(50)], "b": [mk_cand(10)]})
    stats = auditor_agreement(merged, n_auditors=2)
    assert stats["total_candidates"] == 2
    assert stats["by_auditor_count"] == {1: 1, 2: 1}
    assert stats["proposed_by_all_auditors"] == 1
    assert stats["single_auditor_only"] == 1


def test_agreement_not_inflated_when_only_one_auditor_proposed():
    """The bug this replaces: with only single-auditor candidates present,
    max(by_count) == 1 and every entry was reported as full agreement."""
    merged = merge_candidates({"a": [mk_cand(10), mk_cand(50)], "b": []})
    stats = auditor_agreement(merged, n_auditors=2)
    assert stats["proposed_by_all_auditors"] == 0
    assert stats["single_auditor_only"] == 2


# --- parse failure vs zero findings ----------------------------------------

import pytest  # noqa: E402
from src.normalize import ParseFailure  # noqa: E402


def test_unparsable_nonempty_response_raises():
    with pytest.raises(ParseFailure):
        normalize_findings("I could not produce JSON, sorry!", "t1", "a", "b", "x.py")


def test_empty_response_is_zero_findings_not_failure():
    assert normalize_findings("", "t1", "a", "b", "x.py") == []


def test_valid_empty_findings_is_zero_not_failure():
    assert normalize_findings('{"findings": []}', "t1", "a", "b", "x.py") == []


# --- truncation detection ---------------------------------------------------

from src.clients import AnthropicClient, OpenAIClient, TruncatedResponse  # noqa: E402


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = ""
    def json(self):
        return self._payload


def test_anthropic_truncation_raises(monkeypatch):
    client = AnthropicClient(model="m", api_key="k")
    monkeypatch.setattr("src.clients.requests.post", lambda *a, **kw: _FakeResp({
        "stop_reason": "max_tokens", "content": [], "usage": {},
    }))
    with pytest.raises(TruncatedResponse):
        client.complete("s", "u")


def test_openai_truncation_raises(monkeypatch):
    client = OpenAIClient(model="m", api_key="k")
    monkeypatch.setattr("src.clients.requests.post", lambda *a, **kw: _FakeResp({
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        "usage": {},
    }))
    with pytest.raises(TruncatedResponse):
        client.complete("s", "u")


def test_truncation_is_not_retried(monkeypatch):
    calls = {"n": 0}
    def once(*a, **kw):
        calls["n"] += 1
        return _FakeResp({"stop_reason": "max_tokens", "content": [], "usage": {}})
    client = AnthropicClient(model="m", api_key="k")
    monkeypatch.setattr("src.clients.requests.post", once)
    with pytest.raises(TruncatedResponse):
        client.complete("s", "u")
    assert calls["n"] == 1


# --- per-category attribution consistency -----------------------------------

from src.score import score_by_category  # noqa: E402


def test_category_metrics_attribute_tp_by_defect_category():
    """A loose match with a mislabelled category must land in the DEFECT's
    category for both recall and precision, keeping the two tables consistent."""
    finding = mk_finding(10, category=Category.CONCURRENCY, reviewer="a", target="a")
    defect = mk_defect(10, category=Category.LOGIC_ERROR)
    by_cat = score_by_category({("t1", "a", "a"): [finding]}, [defect])

    logic = by_cat[("a", "a", Category.LOGIC_ERROR.value)]
    conc = by_cat[("a", "a", Category.CONCURRENCY.value)]

    assert logic.defects_caught == 1
    assert logic.findings_true == 1        # TP attributed to matched defect's category
    assert conc.findings_true == 0
    assert conc.findings_total == 0        # not counted as a concurrency FP either
