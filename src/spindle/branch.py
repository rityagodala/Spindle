"""Branch: a single parallel exploration path.

A branch is an isolated agent run with its own scoped context, message history,
patch under construction, and token budget. Branches are forkable (deep-copied
from a root state) and cancellable.
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class BranchStatus(StrEnum):
    """Lifecycle states for a branch."""

    PENDING = "pending"
    RUNNING = "running"
    CHECKPOINT = "checkpoint"  # paused at a verifier checkpoint
    KILLED = "killed"  # pruned by verifier
    FAILED = "failed"  # raised an error
    COMPLETED = "completed"  # produced a patch and finished


@dataclass
class BranchState:
    """Mutable per-branch state. Serializable for snapshotting."""

    branch_id: str
    approach: str  # short human description of the approach this branch is exploring
    scoped_files: list[str] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    patch: str = ""  # unified diff, accumulated across tool calls
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    score: float = 0.0  # verifier's running score, [0, 1]
    status: BranchStatus = BranchStatus.PENDING
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def fork(self, new_approach: str) -> BranchState:
        """Return a deep copy with a fresh id and the given approach."""
        clone = copy.deepcopy(self)
        clone.branch_id = uuid.uuid4().hex[:8]
        clone.approach = new_approach
        clone.status = BranchStatus.PENDING
        clone.started_at = 0.0
        clone.finished_at = 0.0
        clone.score = 0.0
        return clone


@dataclass
class Branch:
    """A live branch handle. Wraps state + control."""

    state: BranchState
    cancel_requested: bool = False

    @classmethod
    def new(cls, approach: str, scoped_files: list[str] | None = None) -> Branch:
        return cls(
            state=BranchState(
                branch_id=uuid.uuid4().hex[:8],
                approach=approach,
                scoped_files=scoped_files or [],
            )
        )

    def mark_started(self) -> None:
        self.state.status = BranchStatus.RUNNING
        self.state.started_at = time.time()

    def mark_completed(self) -> None:
        self.state.status = BranchStatus.COMPLETED
        self.state.finished_at = time.time()

    def mark_killed(self, reason: str = "") -> None:
        self.state.status = BranchStatus.KILLED
        self.state.finished_at = time.time()
        if reason:
            self.state.metadata["kill_reason"] = reason

    def mark_failed(self, err: BaseException) -> None:
        self.state.status = BranchStatus.FAILED
        self.state.error = f"{type(err).__name__}: {err}"
        self.state.finished_at = time.time()

    @property
    def total_tokens(self) -> int:
        return self.state.tokens_in + self.state.tokens_out

    @property
    def elapsed(self) -> float:
        if self.state.started_at == 0.0:
            return 0.0
        end = self.state.finished_at or time.time()
        return end - self.state.started_at

    def __repr__(self) -> str:
        s = self.state
        return (
            f"Branch({s.branch_id} {s.status.value} "
            f"score={s.score:.2f} tokens={self.total_tokens} "
            f"approach={s.approach!r})"
        )
