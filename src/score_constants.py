"""Shared constants.

LINE_TOLERANCE is used in three places — finding-to-defect matching, same-run
finding dedupe, and auditor candidate merging — and they must agree. If the
matcher is looser than the deduper, near-duplicate findings can each claim
separate nearby defects; if the merger disagrees with the matcher, a ledger
entry can be unreachable by a correct finding. One constant, imported
everywhere, so the three cannot drift.

Why 3: generated code shifts by a line or two between the audit render and the
review render (blank lines, docstring formatting). Tighter punishes correct
findings for formatting noise; looser starts crediting findings that point at
neighbouring code.
"""

LINE_TOLERANCE = 3
