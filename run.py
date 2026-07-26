#!/usr/bin/env python3
"""Cross-model adversarial code review — experiment runner.

Staged deliberately, because stage 2 requires human judgement and the whole
result is worthless if that stage is skipped:

    python run.py generate    # both models implement every task
    python run.py audit       # propose candidate defects  -> HUMAN CONFIRMS
    python run.py review      # 4-way review matrix (self + cross, both ways)
    python run.py score       # match, compute metrics, render report

Use --mock to exercise the full pipeline with no API keys and no spend.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.clients import build_client  # noqa: E402
from src.ledger import (  # noqa: E402
    auditor_agreement,
    ledger_summary,
    merge_candidates,
    load_ledger,
    propose_candidates,
    save_ledger,
)
from src.pipeline import Task, generate  # noqa: E402
from src.report import full_report  # noqa: E402
from src.schema import Finding, Generation, Category, Severity, dump_jsonl, load_jsonl  # noqa: E402
from src.score import score_by_category, score_runs  # noqa: E402


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_tasks(path: str) -> tuple[dict[str, Task], str]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    conventions = raw.get("conventions", "")
    tasks = {
        t["id"]: Task(
            id=t["id"],
            spec=t["spec"],
            signature=t.get("signature", ""),
            language=t.get("language", "python"),
        )
        for t in raw.get("tasks", [])
    }
    return tasks, conventions


def build_clients(cfg: dict, section: str, mock: bool) -> dict:
    out = {}
    for key, spec in cfg[section].items():
        provider = "mock" if mock else spec["provider"]
        out[key] = build_client(
            provider=provider,
            model=spec["model"],
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 4096),
            reasoning_effort=cfg.get("reasoning_effort", "medium"),
        )
    return out


def cost_for(cfg: dict, model_key: str, tin: int, tout: int) -> float:
    pricing = cfg.get("pricing", {}).get(model_key)
    if not pricing:
        return 0.0
    return (
        tin / 1_000_000 * pricing.get("input_per_mtok", 0.0)
        + tout / 1_000_000 * pricing.get("output_per_mtok", 0.0)
    )


# --- Stages ----------------------------------------------------------------

def cmd_generate(cfg: dict, args) -> None:
    tasks, conventions = load_tasks(cfg["paths"]["tasks"])
    clients = build_clients(cfg, "generators", args.mock)

    rows = []
    for task in tasks.values():
        for key, client in clients.items():
            print(f"  generating {task.id} with {key} ...", flush=True)
            gen = generate(client, key, task, conventions, cfg["paths"]["generate_prompt"])
            rows.append({
                "task_id": gen.task_id,
                "generator": gen.generator,
                "code": gen.code,
                "language": gen.language,
                "input_tokens": gen.input_tokens,
                "output_tokens": gen.output_tokens,
            })

    os.makedirs("results", exist_ok=True)
    dump_jsonl(cfg["paths"]["generations"], rows)
    print(f"\nWrote {len(rows)} generations -> {cfg['paths']['generations']}")


def cmd_audit(cfg: dict, args) -> None:
    tasks, conventions = load_tasks(cfg["paths"]["tasks"])
    gens = [Generation(**r) for r in load_jsonl(cfg["paths"]["generations"])]

    # Run every configured auditor over every generation, then union.
    # With no third model family available, running both families and taking
    # the union keeps either one's blind spots from defining ground truth.
    auditors = {}
    for key, spec in cfg["auditors"].items():
        auditors[key] = build_client(
            provider="mock" if args.mock else spec["provider"],
            model=spec["model"],
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 4096),
            reasoning_effort=cfg.get("reasoning_effort", "medium"),
        )

    candidate_sets: dict[str, list[dict]] = {}
    for auditor_key, auditor in auditors.items():
        found: list[dict] = []
        for gen in gens:
            task = tasks[gen.task_id]
            print(f"  [{auditor_key}] auditing {gen.task_id} ({gen.generator}) ...", flush=True)
            found.extend(propose_candidates(
                auditor=auditor,
                task_id=gen.task_id,
                origin=gen.generator,
                code=gen.code,
                filename=task.filename,
                task_spec=task.spec,
                conventions=conventions,
            ))
        candidate_sets[auditor_key] = found
        print(f"  [{auditor_key}] proposed {len(found)} candidates")

    entries = merge_candidates(candidate_sets)
    save_ledger(cfg["paths"]["ledger"], entries)

    agreement = auditor_agreement(entries, n_auditors=len(auditors))
    print(f"\nUnioned to {len(entries)} candidates -> {cfg['paths']['ledger']}")
    print(f"  auditor agreement: {json.dumps(agreement)}")
    print(
        "\n  NEXT STEP IS MANUAL AND NOT OPTIONAL.\n"
        "  Open the ledger, delete candidates you reject, and set `confirmed: true`\n"
        "  on the ones you accept. Scoring refuses to run until every entry is\n"
        "  resolved — ground truth defined by a model makes this experiment\n"
        "  circular and the numbers meaningless.\n\n"
        "  Spend your attention on entries with a single name in `proposed_by`.\n"
        "  Those are either false positives or defects the other family cannot\n"
        "  see — and the second case is the phenomenon under study."
    )


def _run_key(row: dict) -> tuple[str, str, str]:
    return (row["task_id"], row["reviewer"], row["target_generator"])


def cmd_review(cfg: dict, args) -> None:
    """Run the reviewer × generation matrix, resumably.

    Rows are appended to findings.jsonl as each run completes, so a sleep,
    crash, or Ctrl-C loses at most the single in-flight call. On restart,
    successfully completed runs are skipped; runs that previously ERRORED are
    retried (their stale error rows are dropped first). --fresh discards
    everything and starts over.

    Resume note for the methodology: a resumed batch spans more wall-clock time
    than a single pass, but every run still executes with identical prompts,
    config, and code inputs, which is the property the comparison needs.
    """
    from src.pipeline import review as review_one  # local import to keep top tidy

    tasks, conventions = load_tasks(cfg["paths"]["tasks"])
    gens = [Generation(**r) for r in load_jsonl(cfg["paths"]["generations"])]
    clients = build_clients(cfg, "reviewers", args.mock)

    findings_path = cfg["paths"]["findings"]
    os.makedirs(os.path.dirname(findings_path) or ".", exist_ok=True)

    # --- Load prior progress ------------------------------------------------
    prior_rows: list[dict] = []
    if os.path.exists(findings_path) and not args.fresh:
        prior_rows = load_jsonl(findings_path)

    completed: set[tuple[str, str, str]] = set()
    retry: set[tuple[str, str, str]] = set()
    kept_rows: list[dict] = []
    for row in prior_rows:
        key = _run_key(row)
        if row.get("_run_error"):
            retry.add(key)          # stale error row: drop it, redo the run
            continue
        completed.add(key)
        kept_rows.append(row)
    retry -= completed

    # Rewrite the file without stale error rows, then append from there.
    dump_jsonl(findings_path, kept_rows)

    total = len(gens) * len(clients)
    if completed:
        print(f"  resuming: {len(completed)} of {total} runs already complete"
              + (f", retrying {len(retry)} previously errored" if retry else ""))

    # --- Run what remains, appending per run --------------------------------
    done = len(completed)
    errored = 0
    with open(findings_path, "a", encoding="utf-8") as fh:
        for generation in gens:
            task = tasks[generation.task_id]
            for reviewer_key, client in clients.items():
                key = (task.id, reviewer_key, generation.generator)
                if key in completed:
                    continue
                done += 1
                arm = "self " if reviewer_key == generation.generator else "cross"
                print(f"  [{done:>2}/{total}] {arm}  {reviewer_key} -> "
                      f"{generation.generator}'s {task.id} ...", flush=True)

                run = review_one(
                    client=client,
                    reviewer_key=reviewer_key,
                    generation=generation,
                    task=task,
                    conventions=conventions,
                    prompt_path=cfg["paths"]["review_prompt"],
                )

                rows_out: list[dict] = []
                if run.error:
                    errored += 1
                    print(f"      !! {run.error}")
                    rows_out.append({
                        "finding_id": None,
                        "task_id": run.task_id,
                        "reviewer": run.reviewer,
                        "target_generator": run.target_generator,
                        "input_tokens": run.input_tokens,
                        "output_tokens": run.output_tokens,
                        "_run_error": run.error,
                    })
                elif run.findings:
                    for f in run.findings:
                        row = f.to_dict()
                        row["input_tokens"] = run.input_tokens
                        row["output_tokens"] = run.output_tokens
                        rows_out.append(row)
                else:
                    rows_out.append({
                        "finding_id": None,
                        "task_id": run.task_id,
                        "reviewer": run.reviewer,
                        "target_generator": run.target_generator,
                        "input_tokens": run.input_tokens,
                        "output_tokens": run.output_tokens,
                        "_empty_run": True,
                    })

                for row in rows_out:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()  # survives sleep/crash after every run

    real = done - errored
    print(f"\n{real} of {total} runs complete -> {findings_path}")
    if errored:
        print(
            f"\n  {errored} run(s) FAILED and are recorded as errors, not as "
            f"zero-finding reviews.\n  Fix the cause and re-run "
            f"`python run.py review` — only the failed runs will be retried.\n"
            f"  Scoring refuses to run while errored runs are present."
        )


def cmd_score(cfg: dict, args) -> None:
    defects = load_ledger(cfg["paths"]["ledger"], allow_unconfirmed=args.allow_unconfirmed)
    summary = ledger_summary(defects)
    print(f"  ledger: {json.dumps(summary, indent=2)}")
    if summary["total"] == 0:
        print("\n  Ledger is empty — nothing to score. Run `audit` and confirm entries first.")
        return

    rows = load_jsonl(cfg["paths"]["findings"])

    errored_runs = [r for r in rows if r.get("_run_error")]
    if errored_runs and not args.allow_errored_runs:
        detail = "\n".join(
            f"  {r['task_id']} {r['reviewer']}->{r['target_generator']}: {r['_run_error'][:100]}"
            for r in errored_runs[:10]
        )
        raise ValueError(
            f"{len(errored_runs)} review run(s) failed and were never completed:\n{detail}\n"
            f"Fix the cause and re-run `python run.py review`. A run missing from one arm\n"
            f"biases the self-vs-cross comparison, so scoring refuses to proceed.\n"
            f"(--allow-errored-runs overrides for debugging only.)"
        )

    findings_by_run: dict[tuple[str, str, str], list[Finding]] = {}
    costs: dict[tuple[str, str, str], tuple[int, int, float]] = {}

    for row in rows:
        key = (row["task_id"], row["reviewer"], row["target_generator"])
        findings_by_run.setdefault(key, [])
        if key not in costs:
            costs[key] = (
                row.get("input_tokens", 0),
                row.get("output_tokens", 0),
                cost_for(cfg, row["reviewer"], row.get("input_tokens", 0), row.get("output_tokens", 0)),
            )
        if row.get("_empty_run") or row.get("_run_error"):
            continue
        findings_by_run[key].append(Finding(
            finding_id=row["finding_id"],
            task_id=row["task_id"],
            reviewer=row["reviewer"],
            target_generator=row["target_generator"],
            category=Category.coerce(row["category"]),
            file=row["file"],
            line=int(row["line"]),
            rationale=row["rationale"],
            severity=Severity.coerce(row.get("severity", "major")),
        ))

    loose = score_runs(findings_by_run, defects, costs, require_category_match=False)
    strict = score_runs(findings_by_run, defects, costs, require_category_match=True)
    by_cat = score_by_category(findings_by_run, defects, require_category_match=False)

    generators = sorted({t for (_, t) in loose})
    report = full_report(loose, by_cat, generators, summary["total"], strict_metrics=strict)

    with open(cfg["paths"]["report"], "w", encoding="utf-8") as fh:
        fh.write(report + "\n")

    print("\n" + report)
    print(f"\nWrote report -> {cfg['paths']['report']}")


def cmd_doctor(cfg: dict, args) -> None:
    """One minimal call per configured model, so config problems surface in
    seconds instead of partway through a paid run."""
    sections = {"generators": cfg["generators"], "reviewers": cfg["reviewers"], "auditors": cfg["auditors"]}
    seen: set[tuple[str, str]] = set()
    failures = 0

    for section, entries in sections.items():
        for key, spec in entries.items():
            ident = (spec["provider"], spec["model"])
            if ident in seen:
                continue
            seen.add(ident)
            label = f"{spec['provider']}/{spec['model']}"
            try:
                client = build_client(
                    provider=spec["provider"],
                    model=spec["model"],
                    temperature=cfg.get("temperature", 0.0),
                    # Generous cap: reasoning tokens count against it, and an
                    # empty reply below must mean a real problem, not a tight cap.
                    max_tokens=2048,
                    reasoning_effort=cfg.get("reasoning_effort", "medium"),
                )
                resp = client.complete("Reply with the single word: ok", "ping")
                if not resp.text.strip():
                    raise RuntimeError(
                        "reachable but returned empty text — output cap likely "
                        "consumed by reasoning tokens"
                    )
                print(f"  OK    {label}  -> {resp.text.strip()[:40]!r}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL  {label}\n{exc}\n")

    if failures:
        print(f"\n{failures} model(s) unreachable. Fix config.yaml before running paid stages.")
        sys.exit(1)
    print("\nAll configured models reachable.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["doctor", "generate", "audit", "review", "score"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mock", action="store_true", help="run offline with a deterministic fake client")
    parser.add_argument("--fresh", action="store_true",
                        help="discard prior review progress and rerun all runs from scratch")
    parser.add_argument("--allow-errored-runs", action="store_true",
                        help="score despite failed review runs (debugging only — biases the comparison)")
    parser.add_argument("--allow-unconfirmed", action="store_true",
                        help="score against unconfirmed ledger entries (smoke tests only — never for published numbers)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stages = {"doctor": cmd_doctor, "generate": cmd_generate, "audit": cmd_audit,
              "review": cmd_review, "score": cmd_score}
    try:
        stages[args.stage](cfg, args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        # Expected failure modes — missing keys, unconfirmed ledger, missing
        # upstream stage. A stack trace here is noise, not information.
        print(f"\n{exc}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
