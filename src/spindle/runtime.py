"""Runtime: the orchestrator.

Spawns N branches in parallel, each exploring a distinct approach to the
same task. Manages the full lifecycle:

  1. Build repo-map once (shared across branches, read-only).
  2. Scope each branch's context using keyword baseline + the LearnedRouter.
  3. Fan out branches as concurrent tasks.
  4. Every K agent steps, run a *real* checkpoint: drop each branch's partial
     patch into the sandbox, run tests, score, kill losers (Edge A).
  5. At the end: final scoring, winner selection, ledger commit.
  6. Record outcomes back to the LearnedRouter so the next run is smarter (Edge C).

The two edges are not bolted on — they're load-bearing.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

from spindle.agent import Agent, AgentConfig, LLMClient, default_llm
from spindle.branch import Branch, BranchStatus
from spindle.context import RepoMap, scope_for_approach
from spindle.learning import LearnedRouter
from spindle.ledger import Ledger
from spindle.planner import ApproachPlanner, default_approaches_for_task
from spindle.sandbox import Sandbox, make_sandbox
from spindle.verifier import Verifier


@dataclass
class RuntimeConfig:
    n_branches: int = 4
    model: str = "anthropic/claude-haiku-4-5"
    synthesis_model: str = "anthropic/claude-sonnet-4-6"
    max_steps_per_branch: int = 12
    token_budget_per_branch: int = 80_000
    cost_budget_per_branch_usd: float = 0.50
    max_files_in_scope: int = 8
    checkpoint_every_n_steps: int = 4
    checkpoint_test_timeout_s: float = 60.0
    use_sandbox: bool = False
    sandbox_image: str = "python:3.11-slim"
    test_cmd: str = "pytest -x -q"
    approaches: list[str] = field(default_factory=list)
    # Edge C controls
    use_learned_router: bool = True
    record_outcomes: bool = True
    use_llm_planner: bool = True


@dataclass
class RuntimeResult:
    run_id: str
    winner: Branch | None
    branches: list[Branch]
    total_tokens: int
    total_cost: float
    wall_time_s: float
    checkpoints_run: int = 0
    branches_killed_early: int = 0


class Runtime:
    """Spawns and supervises parallel branches."""

    def __init__(
        self,
        repo_root: str | Path,
        config: RuntimeConfig | None = None,
        llm: LLMClient | None = None,
        ledger: Ledger | None = None,
        router: LearnedRouter | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.config = config or RuntimeConfig()
        self.llm = llm or default_llm()
        self.ledger = ledger or Ledger()
        self.router = router or (LearnedRouter() if self.config.use_learned_router else None)
        self.verifier = Verifier()

    async def run(self, task: str) -> RuntimeResult:
        t0 = time.time()
        repo_key = str(self.repo_root)

        # 1. Build repo-map.
        repo_map = RepoMap.build(self.repo_root)

        # 2. Approaches (caller-supplied, LLM planner, or deterministic defaults).
        if self.config.approaches:
            approaches = self.config.approaches
        elif self.config.use_llm_planner:
            planner = ApproachPlanner(self.llm, self.config.model)
            approaches = await planner.generate(task, self.config.n_branches, repo_map)
        else:
            approaches = default_approaches_for_task(task, self.config.n_branches)

        # 3. Scope each branch's context — consults LearnedRouter.
        branches: list[Branch] = []
        for approach in approaches[: self.config.n_branches]:
            scope = scope_for_approach(
                repo_map,
                approach,
                self.config.max_files_in_scope,
                task=task,
                router=self.router,
                repo_key=repo_key,
            )
            b = Branch.new(approach=approach, scoped_files=scope.files)
            b.state.metadata["scope_rationale"] = scope.rationale
            branches.append(b)

        # 4. Start ledger run.
        run_id = self.ledger.start_run(
            repo=repo_key,
            task=task,
            config={
                "n_branches": self.config.n_branches,
                "model": self.config.model,
                "synthesis_model": self.config.synthesis_model,
                "max_steps": self.config.max_steps_per_branch,
                "checkpoint_every_n_steps": self.config.checkpoint_every_n_steps,
                "use_sandbox": self.config.use_sandbox,
                "use_learned_router": self.config.use_learned_router,
            },
        )
        for b in branches:
            self.ledger.upsert_branch(run_id, b)
        if self.router:
            self.ledger.log_event(run_id, "router_stats", self.router.stats(repo_key))

        # 5. Build sandbox once if requested (shared, but each call uses its
        #    own temp dir, so parallel use is safe).
        sandbox: Sandbox | None = None
        if self.config.use_sandbox:
            sandbox = make_sandbox(
                self.repo_root,
                docker=True,
                image=self.config.sandbox_image,
                test_cmd=self.config.test_cmd,
            )

        # 6. Fan out branches.
        agent_cfg = AgentConfig(
            model=self.config.model,
            max_steps=self.config.max_steps_per_branch,
            token_budget=self.config.token_budget_per_branch,
            cost_budget_usd=self.config.cost_budget_per_branch_usd,
        )
        agent = Agent(agent_cfg, self.llm, self.repo_root, repo_map)

        checkpoints_run = 0
        branches_killed_early = 0

        async def _run_branch(b: Branch) -> None:
            b.mark_started()
            self.ledger.upsert_branch(run_id, b)
            try:
                await agent.run(b, task)
            except asyncio.CancelledError:
                b.mark_killed("cancelled")
                raise
            except Exception as e:
                b.mark_failed(e)
            finally:
                self.ledger.upsert_branch(run_id, b)

        async def _supervise() -> None:
            """Run real mid-run checkpoints (Edge A)."""
            nonlocal checkpoints_run, branches_killed_early
            step = 0
            interval_s = 4.0  # poll every 4s; checkpoint when enough steps have passed
            while any(b.state.status == BranchStatus.RUNNING for b in branches):
                await asyncio.sleep(interval_s)
                step += 1
                if step % self.config.checkpoint_every_n_steps != 0:
                    continue

                report = await self.verifier.judge_checkpoint(
                    branches, sandbox, step=step
                )
                checkpoints_run += 1
                branches_killed_early += report.killed_count()
                self.ledger.log_event(
                    run_id, "checkpoint",
                    {
                        "step": step,
                        "killed": report.killed,
                        "verdicts": [v.__dict__ for v in report.verdicts],
                    },
                )
                for b in branches:
                    self.ledger.upsert_branch(run_id, b)

        try:
            async with asyncio.TaskGroup() as tg:
                for b in branches:
                    tg.create_task(_run_branch(b))
                tg.create_task(_supervise())
        except* Exception as eg:
            for err in eg.exceptions:
                self.ledger.log_event(run_id, "branch_error", {"error": str(err)})

        # 7. Final test pass on every branch with a patch (the authoritative
        #    signal for ranking).
        test_results: dict[str, float] = {}
        if sandbox is not None:
            test_results = await self._final_test(sandbox, branches, run_id)

        # 8. Judge + pick winner.
        verdicts = self.verifier.judge(branches, test_results=test_results)
        for v in verdicts:
            self.ledger.log_event(run_id, "verdict", v.__dict__, branch_id=v.branch_id)

        winner = self.verifier.select_winner(branches)
        total_tokens = sum(b.total_tokens for b in branches)
        total_cost = sum(b.state.cost_usd for b in branches)
        wall = time.time() - t0

        # 9. Record outcomes back to the router (Edge C — the flywheel).
        if self.router is not None and self.config.record_outcomes:
            for b in branches:
                self.router.record_outcome(
                    repo=repo_key,
                    task=task,
                    approach=b.state.approach,
                    files=b.state.scoped_files,
                    score=b.state.score,
                    won=(winner is not None and b.state.branch_id == winner.state.branch_id),
                    tokens=b.total_tokens,
                )

        self.ledger.finish_run(
            run_id,
            winner_id=winner.state.branch_id if winner else None,
            total_tokens=total_tokens,
            total_cost=total_cost,
        )
        for b in branches:
            self.ledger.upsert_branch(run_id, b)

        return RuntimeResult(
            run_id=run_id,
            winner=winner,
            branches=branches,
            total_tokens=total_tokens,
            total_cost=total_cost,
            wall_time_s=wall,
            checkpoints_run=checkpoints_run,
            branches_killed_early=branches_killed_early,
        )

    async def _final_test(
        self,
        sb: Sandbox,
        branches: list[Branch],
        run_id: str,
    ) -> dict[str, float]:
        """Run final tests on every branch in parallel."""
        results: dict[str, float] = {}

        async def _test_one(b: Branch) -> None:
            if not b.state.patch:
                results[b.state.branch_id] = 0.0
                return
            tr = await sb.apply_and_test(b.state.patch)
            results[b.state.branch_id] = tr.pass_rate
            self.ledger.log_event(
                run_id, "test_result",
                {
                    "branch_id": b.state.branch_id,
                    "pass_rate": tr.pass_rate,
                    "exit_code": tr.exit_code,
                    "stdout_tail": tr.stdout[-2000:],
                    "stderr_tail": tr.stderr[-2000:],
                },
                branch_id=b.state.branch_id,
            )

        async with asyncio.TaskGroup() as tg:
            for b in branches:
                tg.create_task(_test_one(b))
        return results
