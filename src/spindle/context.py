"""Repo-map and scoped context selection.

Inspired by Aider's repo-map idea: parse every source file with tree-sitter,
extract top-level symbols (functions, classes), and produce a compact
"map" of the repo. Per-branch context scoping picks a small subset of files
based on the branch's approach + a cheap keyword match.

The point: don't dump the whole repo into every branch's context window.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# tree-sitter is optional at runtime — we degrade to a regex fallback if it
# isn't installed, so tests can run in CI without compiling grammars.
try:
    from tree_sitter_languages import get_parser  # type: ignore[import-untyped]

    _TREE_SITTER_OK = True
except Exception:  # pragma: no cover - import-time guard
    _TREE_SITTER_OK = False


SOURCE_EXTS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
}

# Files we never want in a repo map, even if they match an ext.
IGNORE_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build",
    ".mypy_cache", ".ruff_cache", ".pytest_cache", "target", ".next",
}


@dataclass
class FileSymbols:
    """Symbols extracted from one source file."""

    path: str  # relative to repo root
    language: str
    functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    loc: int = 0

    def keywords(self) -> set[str]:
        """Bag of words for cheap relevance matching."""
        bag: set[str] = set()
        for name in self.functions + self.classes:
            # camelCase + snake_case split
            for tok in re.split(r"[_\W]+|(?=[A-Z])", name):
                if len(tok) >= 3:
                    bag.add(tok.lower())
        # also include path components
        for part in Path(self.path).parts:
            stem = Path(part).stem.lower()
            if len(stem) >= 3:
                bag.add(stem)
        return bag


@dataclass
class RepoMap:
    """A compact representation of a repository."""

    root: Path
    files: list[FileSymbols] = field(default_factory=list)

    @classmethod
    def build(cls, root: str | Path, max_files: int = 2000) -> RepoMap:
        """Walk the repo, parse source files, return a RepoMap."""
        root_path = Path(root).resolve()
        rmap = cls(root=root_path)
        count = 0
        for dirpath, dirnames, filenames in os.walk(root_path):
            # in-place prune
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
            for fn in filenames:
                ext = Path(fn).suffix.lower()
                if ext not in SOURCE_EXTS:
                    continue
                full = Path(dirpath) / fn
                rel = full.relative_to(root_path).as_posix()
                try:
                    syms = _extract_symbols(full, SOURCE_EXTS[ext])
                except Exception:
                    # never let a bad parse take down the build
                    syms = FileSymbols(path=rel, language=SOURCE_EXTS[ext])
                syms.path = rel
                rmap.files.append(syms)
                count += 1
                if count >= max_files:
                    return rmap
        return rmap

    def render(self, files: list[str] | None = None) -> str:
        """Render the repo-map as text, optionally filtered to `files`."""
        chosen = self.files if files is None else [f for f in self.files if f.path in set(files)]
        lines: list[str] = []
        for fs in sorted(chosen, key=lambda x: x.path):
            head = f"{fs.path}  ({fs.language}, {fs.loc} loc)"
            lines.append(head)
            for cls_ in fs.classes:
                lines.append(f"  class {cls_}")
            for fn in fs.functions:
                lines.append(f"  def {fn}")
        return "\n".join(lines)


@dataclass
class ScopedContext:
    """A subset of the repo, chosen for a single branch's approach."""

    files: list[str]
    rationale: str
    token_estimate: int = 0


def scope_for_approach(
    rmap: RepoMap,
    approach: str,
    max_files: int = 8,
    *,
    task: str | None = None,
    router: object | None = None,  # spindle.learning.LearnedRouter; typed loosely to avoid import cycle
    repo_key: str | None = None,
) -> ScopedContext:
    """Pick `max_files` files most relevant to the approach.

    Baseline signal: keyword overlap between approach + file symbols.
    Learned signal (if `router` provided): boost files that have historically
    succeeded on similar tasks for this repo. Blended by the router's warm
    factor so cold repos lean on the baseline.

    This is where Spindle's compounding edge lives.
    """
    approach_kw = _tokenize(approach)
    if task:
        approach_kw = approach_kw | _tokenize(task)

    # 1. Baseline keyword overlap score for every file.
    baseline: dict[str, float] = {}
    for fs in rmap.files:
        kw = fs.keywords()
        overlap = len(approach_kw & kw)
        bonus = -0.1 if "test" in fs.path.lower() else 0.0
        baseline[fs.path] = overlap + bonus + (fs.loc / 10000.0)

    # 2. Learned router boost (if available, and we have a task).
    blended: dict[str, float] = dict(baseline)
    blend_alpha = 0.0
    if router is not None and task and repo_key:
        candidates = list(baseline.keys())
        try:
            boosts = router.boost_files(repo_key, task, candidates)  # type: ignore[attr-defined]
            blend_alpha = router.warm_factor(repo_key)  # type: ignore[attr-defined]
        except Exception:
            boosts, blend_alpha = {}, 0.0
        # Blend: (1 - alpha) * baseline + alpha * (baseline + 2 * boost).
        # The factor 2 makes a maxed-out boost dominate when alpha → 1.
        for f, base in baseline.items():
            b = boosts.get(f, 0.0)
            blended[f] = base + (blend_alpha * 2.0 * b)

    # 3. Rank and pick top-k from positively-scored files.
    scored = [(s, f) for f, s in blended.items() if s > 0]
    scored.sort(key=lambda t: -t[0])
    chosen = [f for _, f in scored[:max_files]]

    if not chosen:
        # Cold-start fallback: README / top-level files.
        fallback = [
            fs.path for fs in rmap.files
            if Path(fs.path).parts and len(Path(fs.path).parts) <= 2
        ][:max_files]
        chosen = fallback

    rationale = (
        f"keywords={sorted(approach_kw)[:6]}, learned_blend={blend_alpha:.2f}"
    )
    return ScopedContext(files=chosen, rationale=rationale)


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in re.split(r"\W+", text) if len(t) >= 3}


def _extract_symbols(path: Path, language: str) -> FileSymbols:
    """Pull function/class names from a source file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return FileSymbols(path=str(path), language=language)

    loc = text.count("\n") + 1
    fs = FileSymbols(path=str(path), language=language, loc=loc)

    if _TREE_SITTER_OK:
        try:
            parser = get_parser(language)
            tree = parser.parse(text.encode("utf-8"))
            _walk_ts(tree.root_node, text.encode("utf-8"), fs)
            if fs.functions or fs.classes:
                return fs
        except Exception:
            pass  # fall through to regex

    # Regex fallback — coarse but works for the common case.
    if language == "python":
        fs.functions = re.findall(r"^\s*def\s+([a-zA-Z_]\w*)", text, re.M)
        fs.classes = re.findall(r"^\s*class\s+([a-zA-Z_]\w*)", text, re.M)
    elif language in {"javascript", "typescript", "tsx"}:
        fs.functions = re.findall(r"function\s+([a-zA-Z_]\w*)", text)
        fs.classes = re.findall(r"class\s+([a-zA-Z_]\w*)", text)
    elif language == "go":
        fs.functions = re.findall(r"^func\s+(?:\([^)]*\)\s+)?([A-Za-z_]\w*)", text, re.M)
        fs.classes = re.findall(r"^type\s+([A-Za-z_]\w*)\s+struct", text, re.M)
    elif language == "rust":
        fs.functions = re.findall(r"^\s*(?:pub\s+)?fn\s+([a-zA-Z_]\w*)", text, re.M)
        fs.classes = re.findall(r"^\s*(?:pub\s+)?struct\s+([A-Za-z_]\w*)", text, re.M)

    return fs


def _walk_ts(node, src: bytes, fs: FileSymbols) -> None:  # type: ignore[no-untyped-def]
    """Walk a tree-sitter AST collecting function/class names."""
    # Generic across languages: we look for any node whose type contains
    # 'function' or 'class' / 'struct' and grab its first 'identifier' child.
    type_ = node.type
    if "function" in type_ or type_ in {"method_definition", "method_declaration"}:
        ident = _first_child(node, "identifier") or _first_child(node, "property_identifier")
        if ident is not None:
            fs.functions.append(_text(ident, src))
    elif type_ in {"class_declaration", "class_definition", "struct_item", "type_declaration"}:
        ident = _first_child(node, "identifier") or _first_child(node, "type_identifier")
        if ident is not None:
            fs.classes.append(_text(ident, src))

    for child in node.children:
        _walk_ts(child, src, fs)


def _first_child(node, type_: str):  # type: ignore[no-untyped-def]
    for c in node.children:
        if c.type == type_:
            return c
    return None


def _text(node, src: bytes) -> str:  # type: ignore[no-untyped-def]
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
