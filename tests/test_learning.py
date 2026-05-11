"""LearnedRouter: cold start, warm-up, and boost direction."""

from __future__ import annotations

from pathlib import Path

from spindle.learning import LearnedRouter


def test_cold_start_returns_neutral_boosts(tmp_path: Path) -> None:
    r = LearnedRouter(path=tmp_path / "o.db")
    boosts = r.boost_files("/repo", "add a json flag", ["a.py", "b.py"])
    assert boosts == {"a.py": 0.0, "b.py": 0.0}


def test_warm_factor_grows_with_runs(tmp_path: Path) -> None:
    r = LearnedRouter(path=tmp_path / "o.db")
    repo = "/repo"
    assert r.warm_factor(repo) == 0.0
    for _ in range(50):
        r.record_outcome(repo, "task", "a", ["a.py"], score=0.5, won=False, tokens=100)
    w = r.warm_factor(repo)
    assert 0.4 < w < 0.6  # 50/(50+50) = 0.5


def test_record_outcome_boosts_winning_files(tmp_path: Path) -> None:
    r = LearnedRouter(path=tmp_path / "o.db")
    repo = "/repo"
    # winners
    for _ in range(8):
        r.record_outcome(repo, "add json flag", "minimal", ["cli.py"],
                         score=0.9, won=True, tokens=1000)
    # losers
    for _ in range(8):
        r.record_outcome(repo, "add json flag", "minimal", ["utils.py"],
                         score=0.1, won=False, tokens=1000)
    boosts = r.boost_files(repo, "add json flag", ["cli.py", "utils.py"])
    assert boosts["cli.py"] > boosts["utils.py"]
    assert boosts["cli.py"] > 0
    assert boosts["utils.py"] < 0


def test_stats_reports_correct_counts(tmp_path: Path) -> None:
    r = LearnedRouter(path=tmp_path / "o.db")
    repo = "/repo"
    for i in range(5):
        r.record_outcome(repo, f"task {i}", "approach", ["a.py"],
                         score=0.8 if i < 3 else 0.2,
                         won=(i < 3),
                         tokens=100)
    s = r.stats(repo)
    assert s["n_outcomes"] == 5
    assert s["n_wins"] == 3
    assert s["n_files_tracked"] == 1
