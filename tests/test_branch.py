"""Branch state + lifecycle."""

from __future__ import annotations

from spindle.branch import Branch, BranchStatus


def test_branch_fork_makes_fresh_id() -> None:
    b = Branch.new(approach="A", scoped_files=["a.py"])
    forked = b.state.fork("B")
    assert forked.branch_id != b.state.branch_id
    assert forked.approach == "B"
    assert forked.status == BranchStatus.PENDING
    assert forked.scoped_files == b.state.scoped_files


def test_branch_lifecycle_transitions() -> None:
    b = Branch.new(approach="A")
    assert b.state.status == BranchStatus.PENDING
    b.mark_started()
    assert b.state.status == BranchStatus.RUNNING
    assert b.state.started_at > 0
    b.mark_completed()
    assert b.state.status == BranchStatus.COMPLETED
    assert b.state.finished_at >= b.state.started_at


def test_branch_failed_records_error() -> None:
    b = Branch.new(approach="A")
    b.mark_started()
    b.mark_failed(ValueError("boom"))
    assert b.state.status == BranchStatus.FAILED
    assert "ValueError" in (b.state.error or "")
    assert "boom" in (b.state.error or "")


def test_total_tokens_sums_in_and_out() -> None:
    b = Branch.new(approach="A")
    b.state.tokens_in = 100
    b.state.tokens_out = 50
    assert b.total_tokens == 150
