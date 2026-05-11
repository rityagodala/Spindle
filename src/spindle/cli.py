"""`spindle` command-line interface."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from spindle.learning import LearnedRouter
from spindle.ledger import Ledger
from spindle.runtime import Runtime, RuntimeConfig, RuntimeResult

app = typer.Typer(
    help="Spindle: parallel exploration runtime for coding agents.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    issue: str = typer.Option(..., "--issue", "-i", help="Task / issue description."),
    repo: Path = typer.Option(Path("."), "--repo", "-r", help="Path to the repo."),
    branches: int = typer.Option(4, "--branches", "-n", min=1, max=16),
    model: str = typer.Option("anthropic/claude-haiku-4-5", "--model", "-m"),
    synthesis_model: str = typer.Option(
        "anthropic/claude-sonnet-4-6", "--synthesis-model"
    ),
    max_steps: int = typer.Option(12, "--max-steps"),
    token_budget: int = typer.Option(80_000, "--token-budget"),
    cost_budget: float = typer.Option(0.50, "--cost-budget"),
    max_files: int = typer.Option(8, "--max-files"),
    checkpoint_every: int = typer.Option(4, "--checkpoint-every"),
    sandbox: bool = typer.Option(False, "--sandbox", help="Run tests in Docker."),
    sandbox_image: str = typer.Option("python:3.11-slim", "--sandbox-image"),
    test_cmd: str = typer.Option("pytest -x -q", "--test-cmd"),
    no_router: bool = typer.Option(
        False, "--no-router", help="Disable the LearnedRouter (Edge C)."
    ),
    no_record: bool = typer.Option(
        False, "--no-record", help="Don't write outcomes to the router."
    ),
    approaches_file: Path | None = typer.Option(
        None, "--approaches", help="JSON list of approach strings."
    ),
) -> None:
    """Run a parallel exploration on a repo."""
    approaches: list[str] = []
    if approaches_file is not None:
        approaches = json.loads(approaches_file.read_text())
        if not (isinstance(approaches, list) and all(isinstance(x, str) for x in approaches)):
            console.print("[red]--approaches file must be a JSON list of strings.[/red]")
            raise typer.Exit(2)

    cfg = RuntimeConfig(
        n_branches=branches,
        model=model,
        synthesis_model=synthesis_model,
        max_steps_per_branch=max_steps,
        token_budget_per_branch=token_budget,
        cost_budget_per_branch_usd=cost_budget,
        max_files_in_scope=max_files,
        checkpoint_every_n_steps=checkpoint_every,
        use_sandbox=sandbox,
        sandbox_image=sandbox_image,
        test_cmd=test_cmd,
        approaches=approaches,
        use_learned_router=not no_router,
        record_outcomes=not no_record,
    )
    rt = Runtime(repo_root=repo, config=cfg)

    console.print(f"[bold]Spindle[/bold] · {branches} branches · {model}")
    console.print(f"  repo: {repo.resolve()}")
    console.print(f"  task: {issue}")
    console.print()

    result = asyncio.run(rt.run(issue))

    _render_result(result)


@app.command()
def stats(
    repo: Path = typer.Option(Path("."), "--repo", "-r"),
) -> None:
    """Show what the LearnedRouter knows about this repo (Edge C)."""
    router = LearnedRouter()
    s = router.stats(str(repo.resolve()))
    table = Table(title=f"Router state · {repo}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in s.items():
        table.add_row(k, str(v))
    console.print(table)


@app.command()
def runs(
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """List recent runs from the ledger."""
    ledger = Ledger()
    rows = ledger.list_recent_runs(limit)
    if not rows:
        console.print("[dim]No runs yet.[/dim]")
        return
    table = Table(title=f"Last {len(rows)} runs")
    for col in ("run", "task", "winner", "tokens", "cost", "elapsed"):
        table.add_column(col)
    for row in rows:
        run_id = row["run_id"]
        started = row["started_at"]
        finished = row["finished_at"]
        task = row["task"]
        winner_id = row["winner_id"]
        tokens = row["total_tokens"]
        cost = row["total_cost"]
        elapsed = f"{(finished or started) - started:.1f}s" if finished else "—"
        table.add_row(
            run_id[:8],
            (task or "")[:60],
            (winner_id or "—")[:8],
            str(tokens or 0),
            f"${cost or 0:.3f}",
            elapsed,
        )
    console.print(table)


def _render_result(result: RuntimeResult) -> None:
    table = Table(title=f"Run {result.run_id}")
    for col in ("branch", "approach", "status", "score", "tokens", "cost", "patch?"):
        table.add_column(col)
    for b in result.branches:
        s = b.state
        is_winner = result.winner and result.winner.state.branch_id == s.branch_id
        marker = "★ " if is_winner else "  "
        table.add_row(
            marker + s.branch_id,
            s.approach[:50],
            s.status.value,
            f"{s.score:.2f}",
            str(b.total_tokens),
            f"${s.cost_usd:.3f}",
            "yes" if s.patch else "no",
        )
    console.print(table)
    console.print()
    console.print(
        f"  [bold]wall:[/bold] {result.wall_time_s:.1f}s  "
        f"[bold]tokens:[/bold] {result.total_tokens}  "
        f"[bold]cost:[/bold] ${result.total_cost:.3f}  "
        f"[bold]checkpoints:[/bold] {result.checkpoints_run}  "
        f"[bold]killed early:[/bold] {result.branches_killed_early}"
    )
    if result.winner:
        console.print(f"[green]winner: {result.winner.state.branch_id}[/green]")
    else:
        console.print("[yellow]no winner[/yellow]")


if __name__ == "__main__":
    app()
