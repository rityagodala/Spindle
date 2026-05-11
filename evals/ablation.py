"""Ablation: does the LearnedRouter actually help?

This is the experiment that justifies Spindle's headline claim — that runs
get cheaper and smarter over time on a given repo. Without this number,
Edge C is vapor.

Design:
  Take a set of N issues for a single repo. Split into two passes:
    Pass A: --no-router (fresh router, never reads, only writes)
    Pass B: full router (reads accumulated outcomes, writes new ones)
  We compare:
    - mean tokens per issue (lower is better)
    - mean wall time per issue (lower is better)
    - win rate (higher is better)
  We also plot tokens-per-issue over the issue index in pass B — if Edge C
  works, the line slopes DOWN over the sequence.

How to read the result:
  If pass B is ≥10% cheaper than pass A at the same win rate, Edge C is
  real and you have a story to tell. If not, the router isn't pulling its
  weight and you need a better signal (embeddings, learned scoring, etc).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from spindle.learning import LearnedRouter
from spindle.ledger import Ledger
from spindle.runtime import Runtime, RuntimeConfig


@dataclass
class IssueResult:
    issue_idx: int
    won: bool
    tokens: int
    cost: float
    wall_s: float


async def _run_one(
    repo: Path,
    issue: str,
    use_router: bool,
    ledger: Ledger,
    router: LearnedRouter | None,
) -> IssueResult:
    cfg = RuntimeConfig(
        n_branches=4,
        checkpoint_every_n_steps=999,  # checkpoints off for cleaner ablation
        use_sandbox=False,
        use_learned_router=use_router,
        record_outcomes=True,
    )
    rt = Runtime(repo_root=repo, config=cfg, ledger=ledger, router=router)
    r = await rt.run(issue)
    won = r.winner is not None and bool(r.winner.state.patch)
    return IssueResult(0, won, r.total_tokens, r.total_cost, r.wall_time_s)


async def ablation(
    repo: Path, issues: list[str], output: Path
) -> dict[str, object]:
    """Run two passes (no-router vs router) and write a report."""
    output.parent.mkdir(parents=True, exist_ok=True)

    # Pass A — no router. Fresh ledger + fresh outcomes store so nothing
    # leaks between conditions.
    ledger_a = Ledger(path=output.parent / "ledger_a.db")
    pass_a: list[IssueResult] = []
    for i, issue in enumerate(issues):
        r = await _run_one(repo, issue, use_router=False, ledger=ledger_a, router=None)
        r.issue_idx = i
        pass_a.append(r)

    # Pass B — full router. Starts cold; learns issue by issue.
    ledger_b = Ledger(path=output.parent / "ledger_b.db")
    router_b = LearnedRouter(path=output.parent / "outcomes_b.db")
    pass_b: list[IssueResult] = []
    for i, issue in enumerate(issues):
        r = await _run_one(repo, issue, use_router=True, ledger=ledger_b, router=router_b)
        r.issue_idx = i
        pass_b.append(r)

    def _agg(rs: list[IssueResult]) -> dict[str, float]:
        return {
            "n": len(rs),
            "win_rate": sum(1 for r in rs if r.won) / max(1, len(rs)),
            "mean_tokens": statistics.mean(r.tokens for r in rs) if rs else 0,
            "mean_cost": statistics.mean(r.cost for r in rs) if rs else 0,
            "mean_wall_s": statistics.mean(r.wall_s for r in rs) if rs else 0,
        }

    # Trend: tokens in first half vs second half of pass B. If Edge C works
    # the second half should be cheaper.
    half = max(1, len(pass_b) // 2)
    trend = {
        "first_half_mean_tokens": (
            statistics.mean(r.tokens for r in pass_b[:half]) if pass_b else 0
        ),
        "second_half_mean_tokens": (
            statistics.mean(r.tokens for r in pass_b[half:])
            if pass_b[half:] else 0
        ),
    }

    report = {
        "pass_a_no_router": _agg(pass_a),
        "pass_b_with_router": _agg(pass_b),
        "trend_within_pass_b": trend,
        "verdict": _verdict(_agg(pass_a), _agg(pass_b), trend),
        "raw_pass_a": [asdict(r) for r in pass_a],
        "raw_pass_b": [asdict(r) for r in pass_b],
    }
    output.write_text(json.dumps(report, indent=2))
    return report


def _verdict(a: dict[str, float], b: dict[str, float], trend: dict[str, float]) -> str:
    """Plain-English read on whether Edge C earned its keep."""
    if a["mean_tokens"] == 0:
        return "inconclusive (no pass A data)"
    token_delta = (a["mean_tokens"] - b["mean_tokens"]) / a["mean_tokens"]
    win_delta = b["win_rate"] - a["win_rate"]
    trend_delta = 0.0
    if trend["first_half_mean_tokens"] > 0:
        trend_delta = (
            trend["first_half_mean_tokens"] - trend["second_half_mean_tokens"]
        ) / trend["first_half_mean_tokens"]

    if token_delta >= 0.10 and win_delta >= -0.05:
        return f"Edge C earns its keep: {token_delta:.0%} cheaper, win delta {win_delta:+.0%}"
    if trend_delta >= 0.15:
        return f"Compounding trend visible: 2nd half {trend_delta:.0%} cheaper than 1st"
    return "Edge C inconclusive on this sample — need more issues or a stronger signal"


def main() -> None:  # pragma: no cover
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--issues", required=True, type=Path,
                   help="JSON file: list of issue strings.")
    p.add_argument("--out", default=Path("ablation_report.json"), type=Path)
    args = p.parse_args()
    issues = json.loads(args.issues.read_text())
    report = asyncio.run(ablation(args.repo, issues, args.out))
    print(json.dumps(report["verdict"], indent=2))
    print(f"\nFull report: {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
