# Spindle

> Parallel exploration runtime for coding agents. Branches run real tests at checkpoints, and Spindle learns your repo.

Most coding agents grind a single linear path through a feature: read the repo, plan, edit, test, repeat — dragging a fat context window through every step. **Spindle** spawns N lightweight branches in parallel, each with **scoped context** (only the files that branch actually needs), runs real tests at mid-run checkpoints, kills losing branches based on **actual execution signal** (not another LLM's opinion), and **learns your repo over time** — the 100th issue you run uses less context than the first.

## What's different

Parallel agent search isn't a new idea — best-of-N, Tree-of-Thoughts, Reflexion, and Princeton's SWE-agent have all explored it. The OSS landscape (Cursor, Aider, Claude Code subagents, OpenHands) is either single-threaded or human-supervised. Spindle has two specific edges:

**Edge A — Execution-grounded mid-run pruning.** Other systems either run every branch to completion then pick (wasteful) or prune based on an LLM verifier (unreliable). Spindle's verifier drops each branch's partial patch into a Docker sandbox at checkpoints, runs the actual test suite, and kills branches whose pass rate is below threshold. The pruning signal is `exit code`, not vibes.

**Edge C — Repo-specific context router that compounds.** Every Spindle run logs which files each branch read, which approach it took, and how well it scored. The next run consults this history to bias file selection toward files that have historically succeeded on similar tasks. Cold-start falls back to keyword matching; after ~50 runs the router meaningfully beats baseline; after ~200 it knows the shape of your codebase. The flywheel is local to your repo and yours forever.

## Architecture

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

## Modules

| File | Role |
|---|---|
| `runtime.py` | Async `TaskGroup` orchestrator. Spawns branches, runs the supervisor loop, calls the verifier at checkpoints, finalizes winners, records outcomes back to the router. |
| `branch.py` | `BranchState` dataclass + forkable lifecycle (pending → running → completed/killed/failed). |
| `context.py` | Tree-sitter repo-map + `scope_for_approach(...)`. Blends keyword baseline with learned router boosts. |
| `agent.py` | The tool-use loop. Tools: `read_file`, `grep`, `write_patch`, `run_tests`, `done`. Model-agnostic via litellm. |
| `verifier.py` | Cheap scoring + expensive `score_at_checkpoint` that runs the sandbox. `judge_checkpoint` runs in parallel across branches. |
| `sandbox.py` | Either local temp-dir clone + `git apply` + pytest, or Docker isolation. `apply_and_test` returns a structured pass rate. |
| `learning.py` | `LearnedRouter`: SQLite-backed outcome store. `record_outcome`, `boost_files`, `warm_factor`. The moat. |
| `ledger.py` | SQLite trace store: runs, branches, events. Every prompt, response, tool call, token, dollar amount. |
| `cli.py` | `spindle run`, `spindle stats`, `spindle runs`. Rich-rendered tables. |
| `evals/ablation.py` | The experiment that proves Edge C earns its keep: two passes (no-router vs router), token/win-rate deltas. |
| `evals/swebench_lite.py` | SWE-bench Lite harness scaffold. Wire up `swebench` package to make it real. |

## Install

```bash
git clone https://github.com/ritjayg/spindle
cd spindle
uv sync --all-extras
export ANTHROPIC_API_KEY=...
```

## Quickstart

```bash
spindle run \
  --repo ./my-project \
  --issue "Add a --json flag to the CLI that emits structured output" \
  --branches 4 \
  --model anthropic/claude-haiku-4-5 \
  --sandbox
```

Inspect what the router has learned about a repo:

```bash
spindle stats --repo ./my-project
```

List recent runs:

```bash
spindle runs --limit 10
```

## Prove Edge C is real

```bash
# Run the ablation: 20 issues, two passes (no-router vs router).
python -m evals.ablation \
  --repo ./my-project \
  --issues issues.json \
  --out ablation_report.json
```

The report writes per-issue numbers plus a verdict line. If pass B (with router) is ≥10% cheaper than pass A (no router) at the same win rate — or if pass B's second half is ≥15% cheaper than its first half — Edge C is earning its keep. If not, you need a stronger signal (embeddings, learned scoring).

## Status

Alpha. Core runtime is in. SWE-bench harness is a scaffold. Real benchmark numbers are pending — that's the next thing to ship.

## License

MIT.
