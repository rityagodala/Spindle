"""End-to-end runtime test with a deterministic MockLLMClient."""

from __future__ import annotations

from pathlib import Path

import pytest

from spindle.agent import MockLLMClient
from spindle.learning import LearnedRouter
from spindle.ledger import Ledger
from spindle.runtime import Runtime, RuntimeConfig


def _script(patch_ok: bool = True) -> list[str]:
    """A scripted agent run: read file, write patch, done."""
    diff = (
        "--- a/src/cli.py\n"
        "+++ b/src/cli.py\n"
        "@@ -1,3 +1,4 @@\n"
        "+# new comment\n"
        " '''CLI entry point.'''\n"
        " import sys\n"
    ) if patch_ok else "not a diff"
    return [
        '<tool>{"tool": "read_file", "path": "src/cli.py"}</tool>',
        '<tool>{"tool": "write_patch", "diff": ' + repr(diff).replace("'", '"') + '}</tool>',
        '<tool>{"tool": "done", "summary": "added a comment"}</tool>',
    ]


@pytest.mark.asyncio
async def test_runtime_end_to_end_mock(sample_repo: Path, tmp_path: Path) -> None:
    llm = MockLLMClient(script=_script(patch_ok=True) * 4)
    cfg = RuntimeConfig(
        n_branches=2,
        max_steps_per_branch=4,
        checkpoint_every_n_steps=999,  # disable mid-run checkpoints in this test
        use_sandbox=False,
        use_learned_router=True,
        record_outcomes=True,
        use_llm_planner=False,
    )
    ledger = Ledger(path=tmp_path / "ledger.db")
    router = LearnedRouter(path=tmp_path / "outcomes.db")
    rt = Runtime(repo_root=sample_repo, config=cfg, llm=llm, ledger=ledger, router=router)
    result = await rt.run("Add a --json flag to the CLI")

    assert len(result.branches) == 2
    # At least one branch should have completed.
    assert any(b.state.status.value == "completed" for b in result.branches)
    # Router should have learned something.
    stats = router.stats(str(sample_repo.resolve()))
    assert stats["n_outcomes"] >= 1


@pytest.mark.asyncio
async def test_runtime_records_to_ledger(sample_repo: Path, tmp_path: Path) -> None:
    llm = MockLLMClient(script=_script() * 4)
    cfg = RuntimeConfig(
        n_branches=2,
        max_steps_per_branch=4,
        checkpoint_every_n_steps=999,
        use_sandbox=False,
        use_llm_planner=False,
    )
    ledger = Ledger(path=tmp_path / "ledger.db")
    rt = Runtime(repo_root=sample_repo, config=cfg, llm=llm, ledger=ledger)
    result = await rt.run("A task")
    run = ledger.get_run(result.run_id)
    assert run is not None
    branches = ledger.list_branches(result.run_id)
    assert len(branches) == 2


@pytest.mark.asyncio
async def test_runtime_no_router_disables_learning(sample_repo: Path, tmp_path: Path) -> None:
    """--no-router should skip routing entirely; outcomes still recorded if requested."""
    llm = MockLLMClient(script=_script() * 4)
    cfg = RuntimeConfig(
        n_branches=2,
        max_steps_per_branch=4,
        checkpoint_every_n_steps=999,
        use_sandbox=False,
        use_learned_router=False,
        use_llm_planner=False,
    )
    ledger = Ledger(path=tmp_path / "ledger.db")
    rt = Runtime(repo_root=sample_repo, config=cfg, llm=llm, ledger=ledger)
    result = await rt.run("A task")
    assert rt.router is None
    assert len(result.branches) == 2
