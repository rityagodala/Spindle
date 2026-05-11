"""Repo-map + scoped context selection."""

from __future__ import annotations

from pathlib import Path

from spindle.context import RepoMap, scope_for_approach


def test_repo_map_finds_python_symbols(sample_repo: Path) -> None:
    rmap = RepoMap.build(sample_repo)
    paths = {f.path for f in rmap.files}
    assert "src/cli.py" in paths
    assert "src/parser.py" in paths
    cli = next(f for f in rmap.files if f.path == "src/cli.py")
    assert "parse_args" in cli.functions
    assert "main" in cli.functions
    parser = next(f for f in rmap.files if f.path == "src/parser.py")
    assert "ArgParser" in parser.classes


def test_repo_map_skips_node_modules(tmp_path: Path) -> None:
    (tmp_path / "node_modules" / "junk").mkdir(parents=True)
    (tmp_path / "node_modules" / "junk" / "a.py").write_text("def x(): pass")
    (tmp_path / "src.py").write_text("def real(): pass")
    rmap = RepoMap.build(tmp_path)
    assert not any("node_modules" in f.path for f in rmap.files)


def test_scope_for_approach_picks_relevant_files(sample_repo: Path) -> None:
    rmap = RepoMap.build(sample_repo)
    scope = scope_for_approach(
        rmap, "Add a --json flag to the CLI parser", max_files=3
    )
    # We expect the cli or parser files to score high; not utils.
    assert any("cli" in f or "parser" in f for f in scope.files)


def test_scope_for_approach_handles_no_matches(sample_repo: Path) -> None:
    rmap = RepoMap.build(sample_repo)
    scope = scope_for_approach(rmap, "xyzzy nothing matches", max_files=3)
    # Falls back to top-level files instead of returning empty.
    assert len(scope.files) > 0


def test_scope_uses_router_boost(sample_repo: Path, tmp_path: Path) -> None:
    """When the router has data, boosted files should rise in the ranking."""
    from spindle.learning import LearnedRouter

    router = LearnedRouter(path=tmp_path / "test_outcomes.db")
    # Pretend src/utils.py has been a winner for 'flag' tasks 10x in a row,
    # even though keyword overlap is low.
    for _ in range(10):
        router.record_outcome(
            repo=str(sample_repo),
            task="add a flag to something",
            approach="minimal",
            files=["src/utils.py"],
            score=0.9,
            won=True,
            tokens=1000,
        )
    # Inject many more outcomes so warm_factor rises meaningfully.
    for _ in range(60):
        router.record_outcome(
            repo=str(sample_repo),
            task="add a flag",
            approach="minimal",
            files=["src/utils.py"],
            score=0.9,
            won=True,
            tokens=1000,
        )

    rmap = RepoMap.build(sample_repo)
    scope = scope_for_approach(
        rmap,
        approach="Minimal change",
        max_files=3,
        task="add a flag",
        router=router,
        repo_key=str(sample_repo),
    )
    assert "src/utils.py" in scope.files
