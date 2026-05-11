"""The agent loop.

Runs one branch's tool-use loop against an LLM. Deliberately simple:
- system prompt + scoped context + task
- LLM emits a tool call or a final patch
- We execute the tool, append result, loop
- Stop conditions: max_steps, "DONE" sentinel, or external cancel

Why we don't use LangChain / LlamaIndex here: the whole project is *about*
controlling what goes into the prompt and how state is forked. Hiding that
behind a framework defeats the experiment.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from spindle.branch import Branch
from spindle.context import RepoMap


class LLMClient(Protocol):
    """Anything that can take messages + return (text, tokens_in, tokens_out, cost)."""

    async def complete(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 2048,
    ) -> tuple[str, int, int, float]: ...


@dataclass
class AgentConfig:
    model: str = "anthropic/claude-haiku-4-5"
    max_steps: int = 12
    max_tokens_per_step: int = 2048
    token_budget: int = 80_000  # hard ceiling per branch
    cost_budget_usd: float = 0.50
    system_prompt_extra: str = ""


SYSTEM_PROMPT = """You are a coding agent exploring one specific implementation approach.

You have a scoped view of the repository — only files relevant to your approach. \
You do NOT have access to the rest of the repo. Work with what you've been given.

You can call tools by emitting a single JSON block on its own line, wrapped in \
<tool> ... </tool>. Available tools:

  {{"tool": "read_file",  "path": "relative/path.py"}}
  {{"tool": "grep",       "pattern": "regex", "path": "optional/dir"}}
  {{"tool": "write_patch","diff": "unified diff text"}}
  {{"tool": "run_tests",  "cmd": "pytest tests/ -x -q"}}
  {{"tool": "done",       "summary": "what you did and why"}}

Rules:
- Emit ONE tool call per turn. Wait for the result before the next call.
- Prefer reading before writing. Read at least one file before you patch.
- When you patch, emit a *real* unified diff (--- / +++ / @@). No prose inside <tool>.
- Call "done" when your patch is complete and tests pass (or you've run out of moves).

Your approach: {approach}
"""


@dataclass
class ToolResult:
    ok: bool
    output: str
    meta: dict[str, Any] = field(default_factory=dict)


class Agent:
    """Runs the tool-use loop for one branch."""

    def __init__(
        self,
        config: AgentConfig,
        llm: LLMClient,
        repo_root: Path,
        repo_map: RepoMap,
    ) -> None:
        self.config = config
        self.llm = llm
        self.repo_root = Path(repo_root)
        self.repo_map = repo_map

    async def run(self, branch: Branch, task: str) -> None:
        """Drive the branch to completion (or budget exhaustion)."""
        system = SYSTEM_PROMPT.format(approach=branch.state.approach)
        if self.config.system_prompt_extra:
            system += "\n\n" + self.config.system_prompt_extra

        # Seed the conversation with the scoped repo-map + task.
        scoped_map = self.repo_map.render(files=branch.state.scoped_files)
        user_seed = (
            f"# Task\n{task}\n\n"
            f"# Scoped repo map ({len(branch.state.scoped_files)} files)\n"
            f"```\n{scoped_map}\n```\n\n"
            "Start by reading the file most likely to need changes."
        )
        branch.state.messages.append({"role": "user", "content": user_seed})

        for _step in range(self.config.max_steps):
            if branch.cancel_requested:
                branch.mark_killed("cancel_requested")
                return
            if branch.total_tokens > self.config.token_budget:
                branch.mark_killed("token_budget")
                return
            if branch.state.cost_usd > self.config.cost_budget_usd:
                branch.mark_killed("cost_budget")
                return

            text, tin, tout, cost = await self.llm.complete(
                model=self.config.model,
                system=system,
                messages=branch.state.messages,
                max_tokens=self.config.max_tokens_per_step,
            )
            branch.state.tokens_in += tin
            branch.state.tokens_out += tout
            branch.state.cost_usd += cost
            branch.state.messages.append({"role": "assistant", "content": text})

            call = _extract_tool_call(text)
            if call is None:
                # Model produced prose without a tool call. Nudge it.
                branch.state.messages.append({
                    "role": "user",
                    "content": "No <tool> block found. Emit exactly one tool call.",
                })
                continue

            if call.get("tool") == "done":
                branch.state.metadata["done_summary"] = call.get("summary", "")
                branch.mark_completed()
                return

            result = await self._execute_tool(call, branch)
            tag = "OK" if result.ok else "ERR"
            branch.state.messages.append({
                "role": "user",
                "content": f"<tool_result status=\"{tag}\">\n{result.output}\n</tool_result>",
            })

        # Out of steps without "done".
        branch.state.metadata["stopped"] = "max_steps"
        branch.mark_completed()

    async def _execute_tool(self, call: dict[str, Any], branch: Branch) -> ToolResult:
        tool = call.get("tool", "")
        try:
            if tool == "read_file":
                return self._tool_read(call["path"])
            if tool == "grep":
                return self._tool_grep(call["pattern"], call.get("path", ""))
            if tool == "write_patch":
                return self._tool_write_patch(call["diff"], branch)
            if tool == "run_tests":
                # Real test execution is delegated to sandbox.py at the runtime
                # layer. The agent gets a stubbed response here; the runtime
                # overrides this for sandbox-enabled runs.
                return ToolResult(ok=True, output="(test execution deferred to sandbox)")
            return ToolResult(ok=False, output=f"Unknown tool: {tool!r}")
        except KeyError as e:
            return ToolResult(ok=False, output=f"Missing arg: {e}")
        except Exception as e:
            return ToolResult(ok=False, output=f"{type(e).__name__}: {e}")

    def _tool_read(self, rel: str) -> ToolResult:
        path = (self.repo_root / rel).resolve()
        if not str(path).startswith(str(self.repo_root.resolve())):
            return ToolResult(ok=False, output="path escapes repo root")
        if not path.exists():
            return ToolResult(ok=False, output=f"no such file: {rel}")
        text = path.read_text(encoding="utf-8", errors="replace")
        # truncate aggressively — branches that need the whole file are doing
        # it wrong
        MAX = 8000
        if len(text) > MAX:
            text = text[:MAX] + f"\n... (truncated; file is {len(text)} chars)"
        return ToolResult(ok=True, output=text, meta={"path": rel})

    def _tool_grep(self, pattern: str, rel: str) -> ToolResult:
        base = (self.repo_root / rel).resolve() if rel else self.repo_root.resolve()
        if not str(base).startswith(str(self.repo_root.resolve())):
            return ToolResult(ok=False, output="path escapes repo root")
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return ToolResult(ok=False, output=f"bad regex: {e}")
        hits: list[str] = []
        for fs in self.repo_map.files:
            full = self.repo_root / fs.path
            if not str(full.resolve()).startswith(str(base)):
                continue
            try:
                for i, line in enumerate(
                    full.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
                    if rx.search(line):
                        hits.append(f"{fs.path}:{i}: {line[:200]}")
                        if len(hits) >= 60:
                            break
            except Exception:
                continue
            if len(hits) >= 60:
                break
        return ToolResult(ok=True, output="\n".join(hits) or "(no matches)")

    def _tool_write_patch(self, diff: str, branch: Branch) -> ToolResult:
        if not diff.strip():
            return ToolResult(ok=False, output="empty diff")
        # Cheap sanity check — does it look like a unified diff?
        if "@@" not in diff or ("---" not in diff and "+++" not in diff):
            return ToolResult(
                ok=False,
                output="diff does not look like unified format (need ---/+++/@@)",
            )
        # Accumulate patches (branches may emit multiple).
        if branch.state.patch:
            branch.state.patch += "\n" + diff
        else:
            branch.state.patch = diff
        return ToolResult(
            ok=True,
            output=f"patch recorded ({len(diff)} chars). Call run_tests next.",
        )


# ---------- tool-call parsing -----------------------------------------------

_TOOL_RX = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.DOTALL)


def _extract_tool_call(text: str) -> dict[str, Any] | None:
    m = _TOOL_RX.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        # Try to repair common issues: trailing commas, single quotes
        repaired = m.group(1).replace("'", '"')
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        try:
            return json.loads(repaired)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            return None


# ---------- default LLM client (litellm) ------------------------------------


class LiteLLMClient:
    """Thin wrapper around litellm.acompletion with cost tracking."""

    def __init__(self) -> None:
        # Lazy import so the package works without litellm in dev/test.
        import litellm

        self._litellm = litellm
        litellm.drop_params = True  # ignore unsupported params silently

    async def complete(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 2048,
    ) -> tuple[str, int, int, float]:
        msgs = [{"role": "system", "content": system}, *messages]
        resp = await self._litellm.acompletion(
            model=model, messages=msgs, max_tokens=max_tokens
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        tin = getattr(usage, "prompt_tokens", 0) if usage else 0
        tout = getattr(usage, "completion_tokens", 0) if usage else 0
        cost = 0.0
        try:
            cost = float(self._litellm.completion_cost(completion_response=resp) or 0.0)
        except Exception:
            cost = 0.0
        return text, tin, tout, cost


class MockLLMClient:
    """Deterministic mock for tests. Reads a script of canned responses."""

    def __init__(self, script: list[str]) -> None:
        self._script = list(script)
        self._i = 0

    async def complete(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 2048,
    ) -> tuple[str, int, int, float]:
        if self._i >= len(self._script):
            return "<tool>{\"tool\": \"done\", \"summary\": \"script exhausted\"}</tool>", 1, 1, 0.0
        text = self._script[self._i]
        self._i += 1
        # rough token estimate
        tin = sum(len(m.get("content", "")) for m in messages) // 4
        tout = len(text) // 4
        return text, tin, tout, 0.0


def default_llm() -> LLMClient:
    """Return a LiteLLM client if keys are present, else a stub."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return LiteLLMClient()
    return MockLLMClient([
        '<tool>{"tool": "done", "summary": "no API key configured"}</tool>'
    ])
