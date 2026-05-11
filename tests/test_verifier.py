"""Verifier: scoring + judgment + checkpoint logic."""

from __future__ import annotations

import pytest

from spindle.branch import Branch, BranchStatus
from spindle.verifier import VerdictKind, Verifier


def test_score_increases_with_test_pass_rate() -> None:
    v = Verifier()
    b = Branch.new("a")
    b.state.patch = "--- a\n+++ b\n@@ -1 +1 @@\n-x\n+y\n"
    low = v.score_branch(b, test_pass_rate=0.0)
    high = v.score_branch(b, test_pass_rate=1.0)
    assert high > low


def test_score_zero_without_patch_or_tests() -> None:
    v = Verifier()
    b = Branch.new("a")
    assert v.score_branch(b) == 0.0


def test_judge_picks_winner_with_full_pass() -> None:
    v = Verifier()
    branches = []
    for i in range(3):
        b = Branch.new(f"a{i}")
        b.state.status = BranchStatus.COMPLETED
        b.state.patch = "--- a\n+++ b\n@@ -1 +1 @@\n-x\n+y\n"
        b.state.tokens_in = 100 * (i + 1)
        branches.append(b)
    # branch 0 passes, the others don't
    results = {branches[0].state.branch_id: 1.0,
               branches[1].state.branch_id: 0.5,
               branches[2].state.branch_id: 0.0}
    verdicts = v.judge(branches, test_results=results)
    winners = [x for x in verdicts if x.kind == VerdictKind.WINNER]
    assert len(winners) == 1
    assert winners[0].branch_id == branches[0].state.branch_id


@pytest.mark.asyncio
async def test_checkpoint_no_sandbox_keeps_branches_alive() -> None:
    """With no sandbox, checkpoints can only use cheap signals, so we
    should not kill below the min_branches_alive floor."""
    v = Verifier(min_branches_alive=2)
    branches = [Branch.new(f"a{i}") for i in range(3)]
    for b in branches:
        b.mark_started()
        b.state.patch = "--- a\n+++ b\n@@ -1 +1 @@\n-x\n+y\n"
    report = await v.judge_checkpoint(branches, sandbox=None, step=4)
    alive = [v_ for v_ in report.verdicts if v_.kind == VerdictKind.CONTINUE]
    assert len(alive) >= 2  # floor respected


def test_select_winner_prefers_higher_score_then_lower_tokens() -> None:
    v = Verifier()
    branches = []
    for i in range(3):
        b = Branch.new(f"a{i}")
        b.state.status = BranchStatus.COMPLETED
        b.state.patch = "x"
        b.state.score = 0.5
        b.state.tokens_in = 100 * (i + 1)
        branches.append(b)
    branches[2].state.score = 0.9  # highest
    branches[1].state.score = 0.9  # tied with [2] but cheaper
    winner = v.select_winner(branches)
    assert winner is branches[1]  # tied score → cheaper tokens
