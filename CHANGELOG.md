# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-11

### Added

- Parallel `Runtime` orchestration with `asyncio.TaskGroup` and per-branch `Agent` tool loops.
- Mid-run **checkpoints**: `Verifier.judge_checkpoint` runs real tests in a sandbox when enabled (`RuntimeConfig.use_sandbox`).
- **LearnedRouter** (`learning.py`) for repo-specific context biasing and outcome recording.
- Tree-sitter–backed `RepoMap` and `scope_for_approach` in `context.py`.
- SQLite **Ledger** for runs, branches, and events.
- CLI: `spindle run`, `spindle stats`, `spindle runs`.
- Ablation harness in `evals/ablation.py` for Edge C comparisons.
- `ApproachPlanner` for LLM-driven orthogonal approach lists (`planner.py`).

### Changed

- N/A for initial release.

### Fixed

- N/A for initial release.

[0.1.0]: https://github.com/rityagodala/Spindle/releases/tag/v0.1.0
