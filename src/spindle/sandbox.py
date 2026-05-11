"""Sandbox: isolated test execution per branch.

We can't have 4 parallel branches stomping on the same filesystem. Each branch
gets either:
  1. A Docker container (if Docker is available + opted-in via `--sandbox`)
  2. A temp directory clone of the repo (fallback)

The sandbox applies the branch's patch, runs the test command, and reports
back a pass rate in [0, 1].
"""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestResult:
    ok: bool
    pass_rate: float  # 0..1
    stdout: str
    stderr: str
    exit_code: int


class Sandbox:
    """Base sandbox: runs in a temp-dir clone of the repo."""

    def __init__(self, repo_root: Path, test_cmd: str = "pytest -x -q") -> None:
        self.repo_root = Path(repo_root)
        self.test_cmd = test_cmd

    async def apply_and_test(
        self, patch: str, timeout_s: float = 120.0
    ) -> TestResult:
        """Clone the repo to a temp dir, apply the patch, run tests."""
        if not patch.strip():
            return TestResult(False, 0.0, "", "no patch to apply", 1)

        with tempfile.TemporaryDirectory(prefix="spindle-sb-") as tmp:
            workdir = Path(tmp) / "repo"
            shutil.copytree(self.repo_root, workdir, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(
                                ".git", "__pycache__", ".venv", "node_modules",
                                ".mypy_cache", ".pytest_cache", ".ruff_cache",
                            ))
            apply_result = await _run(["git", "init", "-q"], cwd=workdir)
            if apply_result.exit_code != 0:
                return apply_result
            # Apply the patch via `git apply`. We use `--unsafe-paths` off and
            # `-p0`/`-p1` autodetection by trying `-p1` first.
            patch_file = workdir / ".spindle.patch"
            patch_file.write_text(patch)
            for strip in (1, 0):
                r = await _run(
                    ["git", "apply", f"-p{strip}", "--whitespace=nowarn", str(patch_file)],
                    cwd=workdir,
                )
                if r.exit_code == 0:
                    break
            else:
                return TestResult(False, 0.0, r.stdout, r.stderr or "patch failed", r.exit_code)

            # Run tests.
            try:
                test_r = await asyncio.wait_for(
                    _run(self.test_cmd.split(), cwd=workdir),
                    timeout=timeout_s,
                )
            except TimeoutError:
                return TestResult(False, 0.0, "", "test timeout", 124)

            pass_rate = _parse_pass_rate(test_r.stdout + "\n" + test_r.stderr)
            ok = test_r.exit_code == 0
            return TestResult(ok, pass_rate if ok else min(pass_rate, 0.5),
                              test_r.stdout, test_r.stderr, test_r.exit_code)


class DockerSandbox(Sandbox):
    """Run tests inside a Docker container. Requires `docker` and an image."""

    def __init__(
        self,
        repo_root: Path,
        image: str = "python:3.11-slim",
        test_cmd: str = "pytest -x -q",
    ) -> None:
        super().__init__(repo_root, test_cmd)
        self.image = image

    async def apply_and_test(
        self, patch: str, timeout_s: float = 120.0
    ) -> TestResult:
        if not patch.strip():
            return TestResult(False, 0.0, "", "no patch to apply", 1)

        with tempfile.TemporaryDirectory(prefix="spindle-docker-") as tmp:
            workdir = Path(tmp) / "repo"
            shutil.copytree(self.repo_root, workdir, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(
                                ".git", "__pycache__", ".venv", "node_modules",
                            ))
            (workdir / ".spindle.patch").write_text(patch)
            cmd = [
                "docker", "run", "--rm",
                "--network", "none",
                "-v", f"{workdir}:/work",
                "-w", "/work",
                self.image,
                "sh", "-c",
                "git init -q && git apply -p1 --whitespace=nowarn .spindle.patch "
                f"&& {self.test_cmd}",
            ]
            try:
                r = await asyncio.wait_for(_run(cmd), timeout=timeout_s)
            except TimeoutError:
                return TestResult(False, 0.0, "", "test timeout", 124)
            pass_rate = _parse_pass_rate(r.stdout + "\n" + r.stderr)
            return TestResult(r.exit_code == 0, pass_rate, r.stdout, r.stderr, r.exit_code)


@dataclass
class _RunResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def pass_rate(self) -> float:
        return _parse_pass_rate(self.stdout + "\n" + self.stderr)


async def _run(cmd: list[str], cwd: Path | None = None) -> TestResult:
    """Run a subprocess, return its exit info wrapped as a TestResult."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
    return TestResult(
        ok=proc.returncode == 0,
        pass_rate=0.0,
        stdout=stdout,
        stderr=stderr,
        exit_code=proc.returncode or 0,
    )


_PYTEST_SUMMARY = re.compile(
    r"=+\s*(?:(?P<passed>\d+) passed)?[,\s]*"
    r"(?:(?P<failed>\d+) failed)?[,\s]*"
    r"(?:(?P<errors>\d+) errors?)?",
)


def _parse_pass_rate(output: str) -> float:
    """Parse pytest-style output to a pass rate in [0, 1]."""
    m = None
    # take the LAST summary line (pytest can print intermediate ones)
    for match in _PYTEST_SUMMARY.finditer(output):
        if match.group("passed") or match.group("failed") or match.group("errors"):
            m = match
    if m is None:
        return 0.0
    p = int(m.group("passed") or 0)
    f = int(m.group("failed") or 0)
    e = int(m.group("errors") or 0)
    total = p + f + e
    if total == 0:
        return 0.0
    return p / total


def make_sandbox(repo_root: Path, *, docker: bool = False, image: str = "python:3.11-slim",
                  test_cmd: str = "pytest -x -q") -> Sandbox:
    """Factory: pick DockerSandbox if requested + available, else local Sandbox."""
    if docker:
        try:
            subprocess.run(["docker", "version"], check=True, capture_output=True, timeout=5)
            return DockerSandbox(repo_root, image=image, test_cmd=test_cmd)
        except (subprocess.SubprocessError, FileNotFoundError):
            pass  # fall through
    return Sandbox(repo_root, test_cmd=test_cmd)
