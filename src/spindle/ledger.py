"""Ledger: persistent SQLite store for runs, branches, and traces.

Every prompt, response, tool call, token count, and dollar amount goes here.
Without this, debugging a parallel agent run is impossible — you can't see
what happened. The ledger is also the dataset for training a v2 verifier.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from spindle.branch import Branch

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    started_at    REAL NOT NULL,
    finished_at   REAL,
    repo          TEXT NOT NULL,
    task          TEXT NOT NULL,
    config_json   TEXT NOT NULL,
    winner_id     TEXT,
    total_tokens  INTEGER DEFAULT 0,
    total_cost    REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS branches (
    branch_id     TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    approach      TEXT NOT NULL,
    scoped_files  TEXT NOT NULL,
    status        TEXT NOT NULL,
    score         REAL DEFAULT 0.0,
    tokens_in     INTEGER DEFAULT 0,
    tokens_out    INTEGER DEFAULT 0,
    cost_usd      REAL DEFAULT 0.0,
    patch         TEXT,
    error         TEXT,
    started_at    REAL,
    finished_at   REAL,
    metadata_json TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS events (
    event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    branch_id     TEXT,
    ts            REAL NOT NULL,
    kind          TEXT NOT NULL,
    payload_json  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_branches_run ON branches(run_id);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_branch ON events(branch_id);
"""


class Ledger:
    """SQLite-backed run/branch/event store. Thread-safe via per-call connect."""

    def __init__(self, path: str | Path = ".spindle/ledger.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
        finally:
            conn.close()

    # ---- runs --------------------------------------------------------------

    def start_run(self, repo: str, task: str, config: dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute(
                "INSERT INTO runs(run_id, started_at, repo, task, config_json) "
                "VALUES(?,?,?,?,?)",
                (run_id, time.time(), repo, task, json.dumps(config)),
            )
        return run_id

    def finish_run(
        self,
        run_id: str,
        winner_id: str | None,
        total_tokens: int,
        total_cost: float,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET finished_at=?, winner_id=?, total_tokens=?, total_cost=? "
                "WHERE run_id=?",
                (time.time(), winner_id, total_tokens, total_cost, run_id),
            )

    # ---- branches ----------------------------------------------------------

    def upsert_branch(self, run_id: str, branch: Branch) -> None:
        s = branch.state
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO branches(
                    branch_id, run_id, approach, scoped_files, status, score,
                    tokens_in, tokens_out, cost_usd, patch, error,
                    started_at, finished_at, metadata_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(branch_id) DO UPDATE SET
                    status=excluded.status,
                    score=excluded.score,
                    tokens_in=excluded.tokens_in,
                    tokens_out=excluded.tokens_out,
                    cost_usd=excluded.cost_usd,
                    patch=excluded.patch,
                    error=excluded.error,
                    started_at=excluded.started_at,
                    finished_at=excluded.finished_at,
                    metadata_json=excluded.metadata_json
                """,
                (
                    s.branch_id, run_id, s.approach, json.dumps(s.scoped_files),
                    s.status.value, s.score, s.tokens_in, s.tokens_out, s.cost_usd,
                    s.patch, s.error, s.started_at, s.finished_at,
                    json.dumps(s.metadata),
                ),
            )

    # ---- events ------------------------------------------------------------

    def log_event(
        self,
        run_id: str,
        kind: str,
        payload: dict[str, Any],
        branch_id: str | None = None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO events(run_id, branch_id, ts, kind, payload_json) "
                "VALUES(?,?,?,?,?)",
                (run_id, branch_id, time.time(), kind, json.dumps(payload, default=str)),
            )

    # ---- queries -----------------------------------------------------------

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if not row:
                return None
            cols = [d[0] for d in c.execute("SELECT * FROM runs LIMIT 0").description]
            return dict(zip(cols, row, strict=True))

    def list_branches(self, run_id: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM branches WHERE run_id=? ORDER BY started_at",
                (run_id,),
            ).fetchall()
            cols = [d[0] for d in c.execute("SELECT * FROM branches LIMIT 0").description]
            return [dict(zip(cols, r, strict=True)) for r in rows]

    def list_recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent runs as row dicts (public API for CLI / tools)."""
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            cur = c.execute(
                "SELECT run_id, started_at, finished_at, repo, task, winner_id, "
                "total_tokens, total_cost FROM runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]


def serialize_branch(branch: Branch) -> dict[str, Any]:
    """Convert a Branch to a JSON-safe dict (for log_event payloads)."""
    return asdict(branch.state)
