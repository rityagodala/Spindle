# Run: uv run python -m evals.swebench_lite --limit 5 --split lite. Then paste real
# numbers into README under ## Benchmarks. No fake numbers.

"""SWE-bench Lite harness for Spindle.

Requires optional extras: ``uv sync --extra bench`` (installs ``swebench`` and
its transitive dependencies).

Loads instances from Hugging Face via ``swebench.harness.utils.load_swebench_dataset``,
builds per-instance ``TestSpec`` values for Docker image keys, runs ``Runtime`` when
given a checked-out repo path, and maps ``result.winner.state.patch`` into the
prediction dict expected by ``swebench.harness.run_evaluation.run_instance``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from spindle.runtime import Runtime, RuntimeConfig


@dataclass
class SwebenchTask:
    """One SWE-bench instance prepared for evaluation."""

    instance_id: str
    issue_text: str
    docker_image_key: str
    test_spec: Any


def load_instances_from_swebench(split: str, limit: int) -> list[SwebenchTask]:
    """Load SWE-bench instances from Hugging Face via the ``swebench`` package.

    ``split`` selects the dataset variant:
      - ``lite`` → ``princeton-nlp/SWE-bench_Lite`` (HF split ``test``)
      - any other string → HF split name on the full ``princeton-nlp/SWE-bench`` dataset
    """
    from swebench.harness.constants import KEY_INSTANCE_ID
    from swebench.harness.test_spec.test_spec import make_test_spec
    from swebench.harness.utils import load_swebench_dataset

    if split.lower() == "lite":
        raw = load_swebench_dataset("lite", split="test")
    else:
        raw = load_swebench_dataset("princeton-nlp/SWE-bench", split=split)
    out: list[SwebenchTask] = []
    for row in raw[:limit]:
        spec = make_test_spec(row)
        out.append(
            SwebenchTask(
                instance_id=row[KEY_INSTANCE_ID],
                issue_text=row.get("problem_statement") or "",
                docker_image_key=spec.instance_image_key,
                test_spec=spec,
            )
        )
    return out


def spindle_prediction(instance_id: str, patch: str | None) -> dict[str, str]:
    """Map a Spindle winner patch to SWE-bench's prediction dict."""
    from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION

    return {
        KEY_INSTANCE_ID: instance_id,
        KEY_MODEL: "spindle",
        KEY_PREDICTION: patch or "",
    }


@dataclass
class EvalResult:
    instance_id: str
    won: bool
    pass_rate: float
    tokens: int
    cost_usd: float
    wall_s: float
    docker_image_key: str
    swebench_prediction: dict[str, str]


async def evaluate_instance(
    task: SwebenchTask,
    repo_path: Path,
    config: RuntimeConfig | None = None,
) -> EvalResult:
    cfg = config or RuntimeConfig()
    rt = Runtime(repo_root=repo_path, config=cfg)
    result = await rt.run(task.issue_text)
    patch = result.winner.state.patch if result.winner else None
    pred = spindle_prediction(task.instance_id, patch)
    won = result.winner is not None and bool(patch)
    pass_rate = result.winner.state.score if result.winner else 0.0
    return EvalResult(
        instance_id=task.instance_id,
        won=won,
        pass_rate=pass_rate,
        tokens=result.total_tokens,
        cost_usd=result.total_cost,
        wall_s=result.wall_time_s,
        docker_image_key=task.docker_image_key,
        swebench_prediction=pred,
    )


async def run_eval(
    jobs: list[tuple[SwebenchTask, Path]],
    config: RuntimeConfig | None,
    output: Path,
) -> dict[str, float]:
    """Run Spindle on each (task, repo_path) pair; write cumulative JSON results."""
    cfg = config or RuntimeConfig()
    results: list[EvalResult] = []
    for task, repo_path in jobs:
        r = await evaluate_instance(task, repo_path, cfg)
        results.append(r)
        output.write_text(json.dumps([asdict(x) for x in results], indent=2))

    summary = {
        "n": len(results),
        "pass_at_1": sum(1 for r in results if r.won) / max(1, len(results)),
        "mean_tokens": statistics.mean(r.tokens for r in results) if results else 0,
        "mean_cost": statistics.mean(r.cost_usd for r in results) if results else 0,
        "mean_wall_s": statistics.mean(r.wall_s for r in results) if results else 0,
    }
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SWE-bench Lite harness for Spindle")
    p.add_argument("--split", default="lite", help="lite | HF split name on full SWE-bench")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--output", type=Path, default=Path("results.json"))
    return p.parse_args()


if __name__ == "__main__":  # pragma: no cover

    async def _main() -> None:
        args = _parse_args()
        tasks = load_instances_from_swebench(args.split, args.limit)
        preview = [
            {"instance_id": t.instance_id, "docker_image_key": t.docker_image_key}
            for t in tasks
        ]
        args.output.write_text(json.dumps(preview, indent=2))
        print(json.dumps({"wrote": str(args.output), "n": len(preview)}, indent=2))

    asyncio.run(_main())
