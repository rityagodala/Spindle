"""Spindle: parallel exploration runtime for coding agents."""

from spindle.agent import Agent, AgentConfig, LiteLLMClient, MockLLMClient, default_llm
from spindle.branch import Branch, BranchState, BranchStatus
from spindle.context import RepoMap, ScopedContext, scope_for_approach
from spindle.learning import LearnedRouter
from spindle.ledger import Ledger
from spindle.runtime import Runtime, RuntimeConfig, RuntimeResult
from spindle.sandbox import Sandbox, make_sandbox
from spindle.verifier import CheckpointReport, Verdict, VerdictKind, Verifier

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "Branch",
    "BranchState",
    "BranchStatus",
    "CheckpointReport",
    "LearnedRouter",
    "Ledger",
    "LiteLLMClient",
    "MockLLMClient",
    "RepoMap",
    "Runtime",
    "RuntimeConfig",
    "RuntimeResult",
    "Sandbox",
    "ScopedContext",
    "Verdict",
    "VerdictKind",
    "Verifier",
    "default_llm",
    "make_sandbox",
    "scope_for_approach",
]
