"""Tests for LLM approach planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from spindle.agent import MockLLMClient
from spindle.context import RepoMap
from spindle.planner import ApproachPlanner, default_approaches_for_task


@pytest.mark.asyncio
async def test_planner_valid_json_returns_n_approaches(sample_repo: Path) -> None:
    payload = (
        '{"approaches": ['
        '"Minimal change: add a single guard clause near the call site.", '
        '"Refactor-first: extract parsing into a helper then extend it.", '
        '"Test-first: add failing cases for the new flag then implement.", '
        '"Reuse: extend the existing ArgParser class with a json option."'
        "]}"
    )
    llm = MockLLMClient([payload])
    planner = ApproachPlanner(llm, model="mock")
    repo_map = RepoMap.build(sample_repo)
    out = await planner.generate("Add --json to the CLI", n=4, repo_map=repo_map)
    assert len(out) == 4
    assert len({x.lower() for x in out}) == 4
    assert all(";" not in x for x in out)


@pytest.mark.asyncio
async def test_planner_bad_json_falls_back(sample_repo: Path) -> None:
    llm = MockLLMClient(['{"approaches": ["only one"]}'])
    planner = ApproachPlanner(llm, model="mock")
    repo_map = RepoMap.build(sample_repo)
    out = await planner.generate("Do something", n=3, repo_map=repo_map)
    assert out == default_approaches_for_task("Do something", 3)


@pytest.mark.asyncio
async def test_planner_exception_falls_back(sample_repo: Path) -> None:
    class FailLLM:
        async def complete(
            self,
            model: str,
            system: str,
            messages: list[dict[str, Any]],
            max_tokens: int = 2048,
        ) -> tuple[str, int, int, float]:
            raise RuntimeError("network")

    planner = ApproachPlanner(FailLLM(), model="mock")
    repo_map = RepoMap.build(sample_repo)
    out = await planner.generate("Task text", n=2, repo_map=repo_map)
    assert out == default_approaches_for_task("Task text", 2)
