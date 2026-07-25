"""Generation and review stages.

The review stage runs the full cross product of {reviewer} x {generation}, which
gives all four configurations the experiment needs:

    A reviews A's code   (self)
    B reviews B's code   (self)
    B reviews A's code   (cross)
    A reviews B's code   (cross)

Reviewers run independently and never see each other's findings. This is
deliberate: heterogeneous agent teams that deliberate toward consensus have been
found to underperform their best individual member, so the design keeps the
reviewers as independent inspectors rather than a committee.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .clients import ModelClient
from .normalize import ParseFailure, extract_code, normalize_findings, number_code
from .schema import Generation, ReviewRun


def _split_prompt(markdown: str) -> tuple[str, str]:
    """Pull SYSTEM and USER sections out of a prompt file."""
    sys_match = re.search(r"^## SYSTEM\s*$(.*?)^## USER\s*$", markdown, re.DOTALL | re.MULTILINE)
    user_match = re.search(r"^## USER\s*$(.*)", markdown, re.DOTALL | re.MULTILINE)
    if not sys_match or not user_match:
        raise ValueError("prompt file must contain '## SYSTEM' and '## USER' sections")
    return sys_match.group(1).strip(), user_match.group(1).strip()


def load_prompt(path: str) -> tuple[str, str]:
    with open(path, "r", encoding="utf-8") as fh:
        return _split_prompt(fh.read())


@dataclass
class Task:
    id: str
    spec: str
    signature: str
    language: str = "python"

    @property
    def filename(self) -> str:
        return f"{self.id}.{'py' if self.language == 'python' else self.language}"


def generate(
    client: ModelClient,
    generator_key: str,
    task: Task,
    conventions: str,
    prompt_path: str,
) -> Generation:
    system, user_tpl = load_prompt(prompt_path)
    user = (
        user_tpl
        .replace("{task_spec}", task.spec)
        .replace("{conventions}", conventions)
        .replace("{signature}", task.signature)
        .replace("{language}", task.language)
    )
    response = client.complete(system, user)
    return Generation(
        task_id=task.id,
        generator=generator_key,
        code=extract_code(response.text),
        language=task.language,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


def review(
    client: ModelClient,
    reviewer_key: str,
    generation: Generation,
    task: Task,
    conventions: str,
    prompt_path: str,
    context: str = "",
) -> ReviewRun:
    """One reviewer's pass over one generation.

    Note what is NOT passed: the identity of the generator. Reviewers are not
    told whose code they are looking at. The hypothesis under test is about
    correlated blind spots, not about explicit self-preference, and production
    review agents are not told either.
    """
    system, user_tpl = load_prompt(prompt_path)
    user = (
        user_tpl
        .replace("{task_spec}", task.spec)
        .replace("{conventions}", conventions)
        .replace("{context}", context or "(none)")
        .replace("{filename}", task.filename)
        .replace("{language}", task.language)
        .replace("{numbered_code}", number_code(generation.code))
    )

    run = ReviewRun(
        task_id=task.id,
        reviewer=reviewer_key,
        target_generator=generation.generator,
    )
    try:
        response = client.complete(system, user)
    except Exception as exc:  # noqa: BLE001
        run.error = str(exc)
        return run

    run.input_tokens = response.input_tokens
    run.output_tokens = response.output_tokens
    try:
        run.findings = normalize_findings(
            raw_text=response.text,
            task_id=task.id,
            reviewer=reviewer_key,
            target_generator=generation.generator,
            default_file=task.filename,
        )
    except ParseFailure as exc:
        # A broken run, not a clean zero. Recorded so scoring can exclude it —
        # and so a systematic formatting problem on one model is visible
        # instead of masquerading as low recall.
        run.error = f"parse_failure: {exc}"
    return run


def review_matrix(
    clients: dict[str, ModelClient],
    generations: list[Generation],
    tasks: dict[str, Task],
    conventions: str,
    prompt_path: str,
    context_by_task: Optional[dict[str, str]] = None,
) -> list[ReviewRun]:
    """Full cross product of reviewers x generations.

    Every reviewer sees every generation, so self and cross arms are produced
    under identical conditions in the same run — no batch-effect drift between
    arms from running them at different times.
    """
    context_by_task = context_by_task or {}
    runs: list[ReviewRun] = []
    for generation in generations:
        task = tasks[generation.task_id]
        for reviewer_key, client in clients.items():
            runs.append(review(
                client=client,
                reviewer_key=reviewer_key,
                generation=generation,
                task=task,
                conventions=conventions,
                prompt_path=prompt_path,
                context=context_by_task.get(task.id, ""),
            ))
    return runs
