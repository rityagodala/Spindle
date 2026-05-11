# Contributing to Spindle

Thanks for helping improve Spindle. This document is the single source of truth for local development and how to extend the runtime.

## Development setup

- Install [uv](https://docs.astral.sh/uv/) (recommended) or use another PEP 517–compatible tool.
- Python **3.11** or **3.12** (see `.python-version`; `tree-sitter-languages` does not ship wheels for 3.14 yet).
- From the repo root:

```bash
uv sync --all-extras
```

Optional API keys for real LLM runs: `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY`. Without them, `default_llm()` uses a stub `MockLLMClient`.

## Running tests and checks

```bash
uv run pytest -q
uv run ruff check src tests evals
uv run mypy src
```

CI runs the same three commands on Python 3.11 and 3.12.

## Adding a new approach generator

1. Implement async generation that returns `list[str]` of length `n` (one short sentence per approach), or integrate with `ApproachPlanner` in `src/spindle/planner.py`.
2. Wire it in `Runtime.run` in `src/spindle/runtime.py` via `RuntimeConfig` (e.g. new flag or replace `ApproachPlanner.generate`).
3. Add unit tests with `MockLLMClient` or a small fake implementing `LLMClient` from `src/spindle/agent.py`.

## Adding a new sandbox backend

1. Implement the `Sandbox` protocol from `src/spindle/sandbox.py` (async `apply_and_test` returning `TestResult`).
2. Register it in `make_sandbox` (same file) or pass a factory from tests.
3. Extend `RuntimeConfig` if new options are needed (`sandbox_image`, `test_cmd`, etc.) and document them here.

## Code style

- **Ruff** is the linter/formatter gate (`ruff check` must be clean for `src`, `tests`, and `evals`).
- **mypy** runs in **strict** mode (`disallow_untyped_defs`, `warn_unused_ignores`). Do not add `# type: ignore` unless unavoidable; if you must, include a one-line comment explaining why.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`, etc. Example: `fix(verifier): handle empty patch at checkpoint`.
