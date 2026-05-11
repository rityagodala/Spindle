# Spindle architecture

This document traces how a single `spindle run` (or `Runtime.run`) moves from a natural-language task to recorded outcomes, and how the supervisor loop ties Edge A (execution-grounded pruning) to Edge C (learned routing).

## ASCII overview

```
                ┌──────────────┐
   issue ──▶    │   Runtime    │   ◀── repo
                └──────┬───────┘
                       │ scope per branch (keyword baseline ⊕ LearnedRouter)
       ┌───────┬───────┼───────┬───────┐
       ▼       ▼       ▼       ▼       ▼
    Branch  Branch  Branch  Branch  Branch
      │       │       │       │       │
      └───────┴─── every K steps ─────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Verifier checkpoint │ ◀── Docker sandbox
            │  (runs real tests)   │     applies partial patch,
            └──────────┬───────────┘     reports pass rate
                       │ kills losers, keeps top-k
                       ▼
              survivors continue
                       │
                       ▼
            final tests → winner patch
                       │
                       ▼
                  LearnedRouter
              (outcomes recorded for
               next run on this repo)
```

## Invocation → repo map

`spindle.cli:run` builds a `RuntimeConfig` and constructs `Runtime` with `repo_root` pointing at your project. When `Runtime.run(task)` begins, the first substantive step is `RepoMap.build(self.repo_root)`, which walks the tree (skipping common noise directories) and, when tree-sitter is available, extracts symbols per file for cheap relevance signals.

## Approaches and scoping

Approaches come from (in order): explicit `RuntimeConfig.approaches`, else `ApproachPlanner.generate` when `use_llm_planner` is true, else `default_approaches_for_task`. For each approach string, `scope_for_approach` selects up to `max_files_in_scope` paths by blending keyword overlap from the repo map with optional boosts from `LearnedRouter.boost_files` when `use_learned_router` is enabled. Each result becomes a `Branch.new(approach=..., scoped_files=...)`.

## Ledger and sandbox

`Ledger.start_run` allocates a `run_id` and stores configuration JSON. If `use_sandbox` is set, `make_sandbox` returns a shared `Sandbox` used for applying partial patches during checkpoints and for optional final verification. Branch rows are upserted as statuses change.

## Supervisor loop (Edge A)

`Runtime.run` launches two async tasks inside an `asyncio.TaskGroup`: one per-branch coroutine `_run_branch` and `_supervise`. `_run_branch` marks the branch running, then hands control to `Agent.run`, which repeatedly calls `LLMClient.complete`, parses `<tool>` JSON, and dispatches to `_execute_tool` (`read_file`, `grep`, `write_patch`, `run_tests`, `done`). `_supervise` polls while any branch is `RUNNING`. On every `checkpoint_every_n_steps` poll cycle, it calls `Verifier.judge_checkpoint`, which may invoke `score_at_checkpoint` / sandbox `apply_and_test` per branch, update `BranchState.score`, and mark weak branches killed. Checkpoint metadata is written with `Ledger.log_event`.

## Final scoring and winner

When all branches settle, `Runtime._final_test` may run full tests in parallel via the sandbox. `Verifier.judge` merges static signals with optional `test_results`, and `Verifier.select_winner` picks the best branch. `Ledger.finish_run` records totals.

## Edge C flywheel

If `record_outcomes` is true and a router is configured, `LearnedRouter.record_outcome` is called per branch with task keywords, scoped files, score, token usage, and whether that branch won. The next `scope_for_approach` call on the same repo can reuse that history without sending everything to the model.

## Evaluation entry points

`evals.ablation` compares runs with and without the router. `evals.swebench_lite` (optional `bench` extra) loads SWE-bench instances and maps Spindle outputs into harness prediction format for external Docker evaluation.
