# Spindle

[![ci](https://github.com/ritjayg/spindle/actions/workflows/ci.yml/badge.svg)](https://github.com/ritjayg/spindle/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/spindle?label=PyPI&color=lightgrey)](https://pypi.org/project/spindle/)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12-blue)](https://github.com/ritjayg/spindle)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Why Spindle?

| Dimension | [Claude Code](https://github.com/anthropics/claude-code) (subagents) | [Aider](https://github.com/Aider-AI/aider) | [SWE-agent](https://github.com/SWE-agent/SWE-agent) | **Spindle** |
|---|---|:---:|:---:|:---:|
| **Parallel exploration** | Yes — multiple subagents, human-directed | Single-threaded pair-programming loop | Primarily single agent + terminal | **N branches by default**, each with its own scoped context |
| **Mid-run kill signal** | Varies by workflow | Tests after edits; not multi-branch racing | Harness-driven evaluation | **Verifier checkpoints** on partial patches using real `pytest` exit codes |
| **Repo-local learning** | Session-scoped | Repo map + conversation | Benchmark-oriented | **SQLite LearnedRouter** biases future file scope from past wins on *your* repo |

Spindle is complementary, not a drop-in for every workflow: it optimizes for *searching* a fix surface when tests are trustworthy and you want amortized context savings.

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
| `planner.py` | `ApproachPlanner.generate` — LLM JSON plan of orthogonal one-liners, with deterministic fallback. |
| `branch.py` | `BranchState` dataclass + forkable lifecycle (pending → running → completed/killed/failed). |
| `context.py` | Tree-sitter repo-map + `scope_for_approach(...)`. Blends keyword baseline with learned router boosts. |
| `agent.py` | The tool-use loop. Tools: `read_file`, `grep`, `write_patch`, `run_tests`, `done`. Model-agnostic via litellm. |
| `verifier.py` | Cheap scoring + expensive `score_at_checkpoint` that runs the sandbox. `judge_checkpoint` runs in parallel across branches. |
| `sandbox.py` | Either local temp-dir clone + `git apply` + pytest, or Docker isolation. `apply_and_test` returns a structured pass rate. |
| `learning.py` | `LearnedRouter`: SQLite-backed outcome store. `record_outcome`, `boost_files`, `warm_factor`. The moat. |
| `ledger.py` | SQLite trace store: runs, branches, events. Every prompt, response, tool call, token, dollar amount. |
| `cli.py` | `spindle run`, `spindle stats`, `spindle runs`. Rich-rendered tables. |
| `evals/ablation.py` | The experiment that proves Edge C earns its keep: two passes (no-router vs router), token/win-rate deltas. |
| `evals/swebench_lite.py` | Loads SWE-bench (Lite) via `swebench`, exposes `docker_image_key`, maps winner patches to harness predictions. |

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

## Benchmarks

This section is reserved for **measured** SWE-bench Lite numbers produced on your hardware. Do not paste estimates.

```bash
uv sync --extra bench
uv run python -m evals.swebench_lite --limit 5 --split lite
```

After a real harness run, paste a short summary table here (pass@1, mean tokens, wall time, cost).

## Roadmap

- [x] v0.1: Core runtime, mid-run checkpoints, learned router
- [ ] v0.2: LLM planner, real SWE-bench numbers, ablation results
- [ ] v0.3: Embedding-based scoping, learned verifier, parallel sandbox pool

## Citing Spindle

```bibtex
@software{spindle2026,
  title        = {Spindle: Parallel exploration runtime for coding agents},
  author       = {Ritya},
  year         = {2026},
  url          = {https://github.com/ritjayg/spindle},
  note         = {Open source, MIT License}
}
```

## Status

Alpha. Core runtime, planner, and harness scaffolding are in place. Published SWE-bench figures belong in **Benchmarks** only after you run the harness.

## License

MIT.
