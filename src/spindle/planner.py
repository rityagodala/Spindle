"""LLM-driven orthogonal approach planning for parallel branches."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from spindle.agent import LLMClient
from spindle.context import RepoMap

PLANNER_SYSTEM = """You are a planning module for a parallel coding-agent runtime.

Given a task and a compact repo map, propose exactly N distinct implementation \
approaches. Each approach must be ONE sentence (no semicolons chaining two plans).

Orthogonality rule: every pair of approaches must differ meaningfully along at \
least one of these axes (cover the set across the list, not each pair in isolation):
- diff size (minimal vs broader touch)
- abstraction level (concrete patch vs new helper layer)
- where in the codebase to intervene (different subsystems/files implied in text)
- test-first vs implementation-first
- reuse existing patterns vs greenfield structure

Respond with ONLY valid JSON on a single line or in a fenced block, using this shape:
{"approaches": ["sentence1", "sentence2", ...]}

There must be exactly N strings in "approaches", all non-empty, no duplicates."""


def default_approaches_for_task(task: str, n: int) -> list[str]:
    """Deterministic baseline approaches (orthogonal axes)."""
    seeds = [
        f"Minimal: implement {task} with the smallest possible diff.",
        f"Refactor-first: clean up adjacent code, then implement {task}.",
        f"Abstraction: introduce a helper / class for {task} and use it.",
        f"Test-first: write tests for {task}, then implement to pass them.",
        f"Reuse: implement {task} reusing existing utilities in the repo.",
        f"Conservative: implement {task} behind a feature-flag / opt-in.",
    ]
    return seeds[:n]


def _normalize_sentence(s: str) -> str:
    return " ".join(s.strip().split())


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        key = x.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def _is_single_sentence(s: str) -> bool:
    t = s.strip()
    return bool(t) and ";" not in t


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


@dataclass
class ApproachPlanner:
    """Plans N orthogonal exploration approaches using the shared LLM client."""

    llm: LLMClient
    model: str

    async def generate(self, task: str, n: int, repo_map: RepoMap) -> list[str]:
        """Return exactly `n` one-sentence approaches, deduped; fallback on failure."""
        try:
            repo_digest = repo_map.render()
            max_map = 14_000
            if len(repo_digest) > max_map:
                repo_digest = repo_digest[:max_map] + "\n... (truncated)"

            user = (
                f"N = {n}\n\n# Task\n{task}\n\n"
                f"# Repo map\n```\n{repo_digest}\n```\n"
            )
            text, _, _, _ = await self.llm.complete(
                model=self.model,
                system=PLANNER_SYSTEM,
                messages=[{"role": "user", "content": user}],
                max_tokens=1024,
            )
            parsed = self._parse_json_approaches(text)
            cleaned = [_normalize_sentence(a) for a in parsed if _is_single_sentence(a)]
            cleaned = _dedupe_preserve_order(cleaned)
            if len(cleaned) < n:
                return default_approaches_for_task(task, n)
            return cleaned[:n]
        except Exception:
            return default_approaches_for_task(task, n)

    def _parse_json_approaches(self, text: str) -> list[str]:
        raw = text.strip()
        m = _JSON_BLOCK.search(raw)
        if m:
            raw = m.group(0)
        data: Any = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("approaches"), list):
            return [str(x) for x in data["approaches"]]
        if isinstance(data, list):
            return [str(x) for x in data]
        raise ValueError("unexpected planner JSON shape")
