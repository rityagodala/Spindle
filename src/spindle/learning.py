"""Learning: Spindle's repo-specific context router.

The flywheel: every Spindle run logs (task_keywords, approach, scoped_files,
outcome_score) tuples to a local SQLite store. Future runs query this store
to bias file selection toward files that have historically succeeded on
similar tasks.

  Cold start (0 runs):      falls back to the keyword baseline in context.py
  Warm   (~20 runs):        router meaningfully beats baseline on this repo
  Hot    (~200 runs):       router knows your repo's "shape"

This is Spindle's compounding advantage. Cursor/Devin/Claude-Code learn at
the model layer (which you don't control) or not at all. Spindle learns at
the *context-routing* layer, which is repo-specific and yours forever.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OUTCOMES_SCHEMA = """
CREATE TABLE IF NOT EXISTS outcomes (
    outcome_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL NOT NULL,
    repo          TEXT NOT NULL,
    task          TEXT NOT NULL,
    task_keywords TEXT NOT NULL,  -- JSON list of normalized tokens
    approach      TEXT NOT NULL,
    files         TEXT NOT NULL,  -- JSON list of file paths
    score         REAL NOT NULL,  -- final branch score, [0, 1]
    won           INTEGER NOT NULL DEFAULT 0,  -- did this branch win its run?
    tokens        INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_outcomes_repo ON outcomes(repo);
CREATE INDEX IF NOT EXISTS idx_outcomes_score ON outcomes(score);

-- File-level success/failure tallies, materialized for fast lookup.
CREATE TABLE IF NOT EXISTS file_stats (
    repo          TEXT NOT NULL,
    file          TEXT NOT NULL,
    keyword       TEXT NOT NULL,
    wins          INTEGER NOT NULL DEFAULT 0,
    losses        INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (repo, file, keyword)
);
"""


@dataclass
class FileScore:
    """How strongly a file is associated with success on a given keyword."""

    file: str
    wins: int
    losses: int

    @property
    def strength(self) -> float:
        """Wilson-ish lower bound: rewards wins, penalizes small samples."""
        n = self.wins + self.losses
        if n == 0:
            return 0.0
        p = self.wins / n
        # Confidence shrinkage toward 0.5 with small n.
        return p - 0.5 * (1.0 / (1.0 + n))


class LearnedRouter:
    """Persistent store of past outcomes + a query interface for routing.

    Two query patterns:
      1. `boost_files(task, candidates)`: re-rank candidate files using past
         wins. Returns (file, boost) pairs in [-1, +1].
      2. `record_outcome(...)`: log a finished branch's outcome.
    """

    def __init__(self, path: str | Path = ".spindle/outcomes.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(OUTCOMES_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
        finally:
            conn.close()

    # ---- recording ---------------------------------------------------------

    def record_outcome(
        self,
        repo: str,
        task: str,
        approach: str,
        files: list[str],
        score: float,
        won: bool,
        tokens: int,
    ) -> None:
        """Log a finished branch's outcome and update file_stats."""
        kw = sorted(_tokenize(task))
        with self._conn() as c:
            c.execute(
                "INSERT INTO outcomes(ts, repo, task, task_keywords, approach, "
                "files, score, won, tokens) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    time.time(), repo, task, json.dumps(kw), approach,
                    json.dumps(files), float(score), 1 if won else 0, tokens,
                ),
            )
            # Update file_stats. A file "wins" for a keyword if its branch
            # scored above 0.5; otherwise it "loses". (Branches that scored
            # in the middle update both columns by 0.5, encoded as a fraction.)
            success = score >= 0.5
            for f in files:
                for k in kw:
                    if success:
                        c.execute(
                            "INSERT INTO file_stats(repo, file, keyword, wins, losses) "
                            "VALUES(?,?,?,1,0) ON CONFLICT DO UPDATE SET wins = wins + 1",
                            (repo, f, k),
                        )
                    else:
                        c.execute(
                            "INSERT INTO file_stats(repo, file, keyword, wins, losses) "
                            "VALUES(?,?,?,0,1) ON CONFLICT DO UPDATE SET losses = losses + 1",
                            (repo, f, k),
                        )

    # ---- querying ----------------------------------------------------------

    def boost_files(
        self, repo: str, task: str, candidates: list[str]
    ) -> dict[str, float]:
        """Return {file: boost} in roughly [-1, 1] for candidate files.

        Boost = mean(strength of (file, keyword)) over the task's keywords,
        where strength is a confidence-shrunk win rate. Files with no history
        get 0 (neutral).
        """
        kw = list(_tokenize(task))
        if not kw or not candidates:
            return {f: 0.0 for f in candidates}

        # Pull all relevant rows in one query.
        placeholders_f = ",".join("?" * len(candidates))
        placeholders_k = ",".join("?" * len(kw))
        sql = (
            f"SELECT file, keyword, wins, losses FROM file_stats "
            f"WHERE repo = ? AND file IN ({placeholders_f}) "
            f"AND keyword IN ({placeholders_k})"
        )
        with self._conn() as c:
            rows = c.execute(sql, [repo, *candidates, *kw]).fetchall()

        # Aggregate per file.
        per_file: dict[str, list[FileScore]] = {f: [] for f in candidates}
        for f, _k, wins, losses in rows:
            per_file[f].append(FileScore(file=f, wins=wins, losses=losses))

        out: dict[str, float] = {}
        for f in candidates:
            scores = per_file[f]
            if not scores:
                out[f] = 0.0
            else:
                out[f] = sum(s.strength for s in scores) / len(scores)
        return out

    def warm_factor(self, repo: str) -> float:
        """Return [0, 1] indicating how much we trust the router for this repo.

        0 = cold start, 1 = lots of data. Used to blend learned signal with
        the keyword baseline so cold repos aren't dominated by noise.
        """
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) FROM outcomes WHERE repo = ?", (repo,)
            ).fetchone()
        n = (row[0] if row else 0) or 0
        # Heuristic: 0 runs → 0, 50 runs → 0.5, 200 runs → ~0.8.
        return n / (n + 50.0)

    # ---- introspection (useful for evals + debugging) ----------------------

    def stats(self, repo: str) -> dict[str, Any]:
        """Return counters about what we know for this repo."""
        with self._conn() as c:
            n_outcomes = c.execute(
                "SELECT COUNT(*) FROM outcomes WHERE repo = ?", (repo,)
            ).fetchone()[0]
            n_wins = c.execute(
                "SELECT COUNT(*) FROM outcomes WHERE repo = ? AND won = 1", (repo,)
            ).fetchone()[0]
            n_files = c.execute(
                "SELECT COUNT(DISTINCT file) FROM file_stats WHERE repo = ?", (repo,)
            ).fetchone()[0]
        return {
            "n_outcomes": n_outcomes,
            "n_wins": n_wins,
            "n_files_tracked": n_files,
            "warm_factor": self.warm_factor(repo),
        }


def _tokenize(text: str) -> set[str]:
    """Same tokenizer as context.py — keep them in sync."""
    return {t.lower() for t in re.split(r"\W+", text) if len(t) >= 3}
