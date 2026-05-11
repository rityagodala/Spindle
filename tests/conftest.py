"""Shared fixtures."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """A tiny fake Python repo with a CLI, a parser, and tests."""
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "cli.py").write_text(textwrap.dedent("""
        '''CLI entry point.'''
        import sys

        def parse_args(argv):
            flags = {'verbose': False}
            for a in argv:
                if a == '--verbose':
                    flags['verbose'] = True
            return flags

        def main():
            flags = parse_args(sys.argv[1:])
            print('hello', flags)

        if __name__ == '__main__':
            main()
    """).strip())
    (tmp_path / "src" / "parser.py").write_text(textwrap.dedent("""
        '''Args parser helpers.'''
        class ArgParser:
            def __init__(self):
                self.flags = {}
            def add_flag(self, name):
                self.flags[name] = False
    """).strip())
    (tmp_path / "src" / "utils.py").write_text("def helper():\n    return 1\n")
    (tmp_path / "tests" / "test_cli.py").write_text(textwrap.dedent("""
        from src.cli import parse_args
        def test_verbose():
            assert parse_args(['--verbose'])['verbose'] is True
        def test_default():
            assert parse_args([])['verbose'] is False
    """).strip())
    (tmp_path / "README.md").write_text("# sample repo\n")
    return tmp_path
