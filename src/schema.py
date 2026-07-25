"""Core data structures for the cross-model review experiment.

Everything that crosses a module boundary is one of these types. The scoring
harness never sees raw model output — only normalized Findings — which is what
keeps it blind to which arm produced a given finding.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


# --- Defect taxonomy -------------------------------------------------------
# Results are reported per category. Aggregate numbers hide where the effect
# lives, and the hypothesis makes different predictions per category:
# reasoning-gap categories should show a cross-model advantage, while
# context-retrieval categories (conventions) should not, since every arm
# shares the same context layer.

class Category(str, Enum):
    LOGIC_ERROR = "logic-error"
    MISSING_EDGE_CASE = "missing-edge-case"
    NULL_BOUNDARY = "null-boundary"
    CONCURRENCY = "concurrency"
    RESOURCE_LEAK = "resource-leak"
    CONVENTION_VIOLATION = "convention-violation"
    UNTESTED_BRANCH = "untested-branch"
    SECURITY_RELEVANT = "security-relevant"

    @classmethod
    def coerce(cls, raw: str) -> "Category":
        """Map loose model output onto the taxonomy.

        Models will not reliably emit our exact enum strings. Normalizing here
        (rather than at scoring time) keeps the scorer deterministic.
        """
        if not raw:
            return cls.LOGIC_ERROR
        key = raw.strip().lower().replace("_", "-").replace(" ", "-")
        for member in cls:
            if member.value == key:
                return member
        aliases = {
            "logic": cls.LOGIC_ERROR,
            "correctness": cls.LOGIC_ERROR,
            "bug": cls.LOGIC_ERROR,
            "edge-case": cls.MISSING_EDGE_CASE,
            "edgecase": cls.MISSING_EDGE_CASE,
            "missing-edge-cases": cls.MISSING_EDGE_CASE,
            "null": cls.NULL_BOUNDARY,
            "null-check": cls.NULL_BOUNDARY,
            "boundary": cls.NULL_BOUNDARY,
            "off-by-one": cls.NULL_BOUNDARY,
            "index-error": cls.NULL_BOUNDARY,
            "race": cls.CONCURRENCY,
            "race-condition": cls.CONCURRENCY,
            "thread-safety": cls.CONCURRENCY,
            "deadlock": cls.CONCURRENCY,
            "leak": cls.RESOURCE_LEAK,
            "resource": cls.RESOURCE_LEAK,
            "unclosed-resource": cls.RESOURCE_LEAK,
            "style": cls.CONVENTION_VIOLATION,
            "convention": cls.CONVENTION_VIOLATION,
            "conventions": cls.CONVENTION_VIOLATION,
            "naming": cls.CONVENTION_VIOLATION,
            "testing": cls.UNTESTED_BRANCH,
            "test-coverage": cls.UNTESTED_BRANCH,
            "missing-test": cls.UNTESTED_BRANCH,
            "security": cls.SECURITY_RELEVANT,
            "injection": cls.SECURITY_RELEVANT,
            "vulnerability": cls.SECURITY_RELEVANT,
        }
        return aliases.get(key, cls.LOGIC_ERROR)


class Severity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"

    @classmethod
    def coerce(cls, raw: str) -> "Severity":
        key = (raw or "").strip().lower()
        if key in ("critical", "blocker", "high"):
            return cls.CRITICAL
        if key in ("major", "medium", "moderate"):
            return cls.MAJOR
        return cls.MINOR


# --- Arms ------------------------------------------------------------------

class Arm(str, Enum):
    """The three experimental arms.

    NO_REVIEW is not a review configuration — it is the base defect rate in
    generated code, i.e. the floor that arms 2 and 3 are measured against.
    """
    NO_REVIEW = "no-review"
    SELF = "self-review"
    CROSS = "cross-review"


# --- Records ---------------------------------------------------------------

@dataclass
class Generation:
    """Code produced by one generator for one task."""
    task_id: str
    generator: str          # model key, e.g. "claude" / "gpt"
    code: str
    language: str = "python"
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def gen_id(self) -> str:
        return f"{self.task_id}::{self.generator}"


@dataclass
class Defect:
    """A ground-truth defect, attributed to the generator that produced it.

    `origin` is the load-bearing field: the self-vs-cross comparison is only
    meaningful if we know which model authored the code the defect lives in.
    """
    defect_id: str
    task_id: str
    origin: str             # generator whose code contains this defect
    category: Category
    file: str
    line_start: int
    line_end: int
    description: str
    severity: Severity = Severity.MAJOR
    source: str = "audit"   # "audit" (found in generated code) | "injected"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        d["severity"] = self.severity.value
        return d


@dataclass
class Finding:
    """A normalized review finding. The scorer only ever sees these."""
    finding_id: str
    task_id: str
    reviewer: str           # model key that produced the finding
    target_generator: str   # whose code was reviewed
    category: Category
    file: str
    line: int
    rationale: str
    severity: Severity = Severity.MAJOR

    # Populated by the scorer, not the reviewer.
    matched_defect_id: Optional[str] = None

    @property
    def arm(self) -> Arm:
        return Arm.SELF if self.reviewer == self.target_generator else Arm.CROSS

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        d["severity"] = self.severity.value
        d["arm"] = self.arm.value
        return d


@dataclass
class ReviewRun:
    """One reviewer's full pass over one generation."""
    task_id: str
    reviewer: str
    target_generator: str
    findings: list[Finding] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None

    @property
    def arm(self) -> Arm:
        return Arm.SELF if self.reviewer == self.target_generator else Arm.CROSS


def stable_id(*parts: str) -> str:
    """Deterministic short id, so reruns produce diffable output."""
    joined = "||".join(str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


def dump_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
