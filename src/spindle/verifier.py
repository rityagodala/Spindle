"""Verifier: score branches at mid-run checkpoints and decide who lives.

This is one half of Spindle's edge. Other parallel-agent systems either:
  (a) run every branch to completion then pick (best-of-N — wasteful), or
  (b) prune based on another LLM's opinion (unreliable).

Spindle prunes based on *real test execution* at mid-run checkpoints. A
branch that has produced a partial patch is dropped into the sandbox, tests
are run, and the verdict comes from the exit code — not vibes.

Two scoring paths:
  - `score_branch`: cheap signals only (patch shape, tool discipline).
  - `score_at_checkpoint`: runs the partial patch through the sandbox,
    combines test pass rate with cheap signals. Called by the runtime
    supervisor every K steps.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum

from spindle.branch import Branch, BranchStatus
from spindle.sandbox import Sandbox


class VerdictKind(StrEnum):
    CONTINUE = "continue"
    KILL = "kill"
    WINNER = "winner"


@dataclass
class Verdict:
    branch_id: str
    kind: VerdictKind
    score: float
    reason: str
    test_pass_rate: float | None = None


@dataclass
class CheckpointReport:
    """One round of mid-run judgment across all live branches."""

    step: int
    verdicts: list[Verdict] = field(default_factory=list)
    killed: list[str] = field(default_factory=list)

    def killed_count(self) -> int:
        return len(self.killed)


@dataclass
class Verifier:
    """Scores branches and selects winners.

    Knobs:
      kill_below: branches scoring under this at a checkpoint get killed.
      keep_top_k: at any checkpoint, keep at most this many branches alive.
      min_branches_alive: never kill below this many, even if all are bad.
    """

    kill_below: float = 0.15
    keep_top_k: int = 3
    min_branches_alive: int = 2

    # ---- cheap scoring (no test execution) ---------------------------------

    def score_branch(self, branch: Branch, test_pass_rate: float | None = None) -> float:
        """Combine cheap signals into a [0, 1] score."""
        s = branch.state
        score = 0.0

        if test_pass_rate is not None:
            score += 0.7 * test_pass_rate

        if s.patch:
            score += 0.1
            if 200 <= len(s.patch) < 5000:
                score += 0.05
            elif len(s.patch) >= 5000:
                score -= 0.05

        n_assistant = sum(1 for m in s.messages if m.get("role") == "assistant")
        n_tool = sum(
            1
            for m in s.messages
            if m.get("role") == "user" and "<tool_result" in str(m.get("content", ""))
        )
        if n_assistant > 0:
            ratio = n_tool / n_assistant
            score += 0.15 * min(ratio, 1.0)

        score = max(0.0, min(1.0, score))
        branch.state.score = score
        return score

    # ---- expensive scoring (runs the sandbox) ------------------------------

    async def score_at_checkpoint(
        self,
        branch: Branch,
        sandbox: Sandbox | None,
        timeout_s: float = 60.0,
    ) -> tuple[float, float | None]:
        """Score a branch using its current partial patch + real tests."""
        pass_rate: float | None = None
        if sandbox is not None and branch.state.patch:
            try:
                tr = await asyncio.wait_for(
                    sandbox.apply_and_test(branch.state.patch),
                    timeout=timeout_s,
                )
                pass_rate = tr.pass_rate
            except TimeoutError:
                pass_rate = 0.0
            except Exception:
                pass_rate = None
        score = self.score_branch(branch, test_pass_rate=pass_rate)
        return score, pass_rate

    # ---- batch judgment ----------------------------------------------------

    async def judge_checkpoint(
        self,
        branches: list[Branch],
        sandbox: Sandbox | None,
        step: int,
    ) -> CheckpointReport:
        """Run a mid-run checkpoint across all live branches.

        Tests are run *in parallel* across branches so the checkpoint itself
        doesn't dominate wall time.
        """
        live = [
            b for b in branches
            if b.state.status == BranchStatus.RUNNING and not b.cancel_requested
        ]
        if not live:
            return CheckpointReport(step=step)

        async def _one(b: Branch) -> tuple[Branch, float, float | None]:
            score, rate = await self.score_at_checkpoint(b, sandbox)
            return b, score, rate

        results = await asyncio.gather(*[_one(b) for b in live])

        ranked = sorted(results, key=lambda t: -t[1])
        keep_ids = {b.state.branch_id for b, _, _ in ranked[: self.keep_top_k]}

        report = CheckpointReport(step=step)
        n_alive = len(live)
        for b, score, rate in ranked:
            if n_alive <= self.min_branches_alive:
                report.verdicts.append(Verdict(
                    b.state.branch_id, VerdictKind.CONTINUE, score,
                    f"floor: only {n_alive} alive", rate,
                ))
                continue
            if score < self.kill_below:
                report.verdicts.append(Verdict(
                    b.state.branch_id, VerdictKind.KILL, score,
                    f"checkpoint score {score:.2f} < {self.kill_below}", rate,
                ))
                report.killed.append(b.state.branch_id)
                b.cancel_requested = True
                n_alive -= 1
            elif b.state.branch_id not in keep_ids:
                report.verdicts.append(Verdict(
                    b.state.branch_id, VerdictKind.KILL, score,
                    f"not in top {self.keep_top_k}", rate,
                ))
                report.killed.append(b.state.branch_id)
                b.cancel_requested = True
                n_alive -= 1
            else:
                report.verdicts.append(Verdict(
                    b.state.branch_id, VerdictKind.CONTINUE, score,
                    "kept", rate,
                ))
        return report

    # ---- final judgment ----------------------------------------------------

    def judge(
        self, branches: list[Branch], test_results: dict[str, float] | None = None
    ) -> list[Verdict]:
        """End-of-run judgment with optional final test results."""
        test_results = test_results or {}
        verdicts: list[Verdict] = []
        considered = [
            b for b in branches
            if b.state.status in {
                BranchStatus.RUNNING, BranchStatus.CHECKPOINT, BranchStatus.COMPLETED,
            }
        ]
        for b in considered:
            self.score_branch(b, test_results.get(b.state.branch_id))

        winners = [
            b for b in considered if test_results.get(b.state.branch_id, 0.0) >= 0.999
        ]
        if winners:
            winner = min(winners, key=lambda x: x.total_tokens)
            for b in considered:
                rate = test_results.get(b.state.branch_id)
                if b.state.branch_id == winner.state.branch_id:
                    verdicts.append(Verdict(
                        b.state.branch_id, VerdictKind.WINNER, b.state.score,
                        "all tests pass, cheapest", rate,
                    ))
                else:
                    verdicts.append(Verdict(
                        b.state.branch_id, VerdictKind.KILL, b.state.score,
                        "another branch already won", rate,
                    ))
            return verdicts

        ranked = sorted(considered, key=lambda x: -x.state.score)
        keep_ids = {b.state.branch_id for b in ranked[: self.keep_top_k]}
        for b in considered:
            rate = test_results.get(b.state.branch_id)
            if b.state.score < self.kill_below:
                verdicts.append(Verdict(
                    b.state.branch_id, VerdictKind.KILL, b.state.score,
                    f"final score {b.state.score:.2f} < {self.kill_below}", rate,
                ))
            elif b.state.branch_id not in keep_ids:
                verdicts.append(Verdict(
                    b.state.branch_id, VerdictKind.KILL, b.state.score,
                    f"not in top {self.keep_top_k}", rate,
                ))
            else:
                verdicts.append(Verdict(
                    b.state.branch_id, VerdictKind.CONTINUE, b.state.score,
                    "kept alive", rate,
                ))
        return verdicts

    def select_winner(self, branches: list[Branch]) -> Branch | None:
        candidates = [
            b for b in branches
            if b.state.status == BranchStatus.COMPLETED and b.state.patch
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda x: (-x.state.score, x.total_tokens))
        return candidates[0]
