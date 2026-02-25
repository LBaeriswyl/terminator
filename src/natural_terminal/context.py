"""Context management — static context, conversation history, directory tree."""

from __future__ import annotations

import getpass
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StaticContext:
    """Gathered once at startup."""

    os_type: str = ""
    shell_type: str = ""
    username: str = ""
    home_dir: str = ""

    @classmethod
    def gather(cls) -> StaticContext:
        return cls(
            os_type=platform.system(),
            shell_type=os.environ.get("SHELL", "/bin/sh").split("/")[-1],
            username=getpass.getuser(),
            home_dir=str(Path.home()),
        )


@dataclass
class ExchangeRecord:
    """A single user↔command exchange."""

    user_input: str
    generated_command: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache", ".ruff_cache", ".pytest_cache"}


class DirectoryTree:
    """Generate a text representation of directory contents."""

    def __init__(self, max_depth: int = 3, max_entries: int = 100, timeout_ms: int = 500):
        self.max_depth = max_depth
        self.max_entries = max_entries
        self.timeout_ms = timeout_ms
        self._cache: dict[str, tuple[str, float]] = {}

    def get(self, cwd: str) -> str:
        # Check cache (invalidate after 30 seconds)
        if cwd in self._cache:
            cached_tree, cached_time = self._cache[cwd]
            if time.time() - cached_time < 30:
                return cached_tree

        tree = self._build(cwd)
        self._cache[cwd] = (tree, time.time())
        return tree

    def invalidate(self, cwd: str) -> None:
        self._cache.pop(cwd, None)

    def _build(self, root: str) -> str:
        entries: list[str] = []
        deadline = time.time() + self.timeout_ms / 1000
        truncated = False

        try:
            self._walk(Path(root), "", 0, entries, deadline)
        except _Timeout:
            truncated = True

        if not entries:
            return "  (empty or inaccessible)"

        result = "\n".join(entries)
        if truncated:
            result += "\n  ... (truncated due to timeout)"
        return result

    def _walk(
        self,
        path: Path,
        prefix: str,
        depth: int,
        entries: list[str],
        deadline: float,
    ) -> None:
        if depth > self.max_depth:
            return
        if len(entries) >= self.max_entries:
            entries.append(f"{prefix}... and more")
            raise _Timeout

        if time.time() > deadline:
            raise _Timeout

        try:
            items = sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return

        for item in items:
            if len(entries) >= self.max_entries:
                entries.append(f"{prefix}... and more")
                raise _Timeout

            if time.time() > deadline:
                raise _Timeout

            if item.name in SKIP_DIRS:
                continue

            if item.is_dir(follow_symlinks=False):
                entries.append(f"{prefix}{item.name}/")
                self._walk(Path(item.path), prefix + "  ", depth + 1, entries, deadline)
            else:
                entries.append(f"{prefix}{item.name}")


class _Timeout(Exception):
    pass


class ConversationHistory:
    """Rolling window of exchange records."""

    def __init__(self, max_length: int = 10, truncate_lines: int = 60):
        self.max_length = max_length
        self.truncate_lines = truncate_lines
        self._records: list[ExchangeRecord] = []

    def add(self, record: ExchangeRecord) -> None:
        self._records.append(record)
        if len(self._records) > self.max_length:
            self._records = self._records[-self.max_length:]

    def clear(self) -> None:
        self._records.clear()

    @property
    def records(self) -> list[ExchangeRecord]:
        return list(self._records)

    def to_messages(self) -> list[dict[str, str]]:
        """Convert history to chat API message format."""
        messages: list[dict[str, str]] = []
        for rec in self._records:
            messages.append({"role": "user", "content": rec.user_input})

            parts = [f'{{"type": "command", "command": "{rec.generated_command}"}}']
            if rec.stdout or rec.stderr:
                output = self._truncate_output(rec.stdout)
                if rec.stderr:
                    output += f"\nSTDERR: {self._truncate_output(rec.stderr)}"
                parts.append(f"\n[Output (exit {rec.exit_code}):\n{output}\n]")
            messages.append({"role": "assistant", "content": "".join(parts)})
        return messages

    def _truncate_output(self, text: str) -> str:
        lines = text.splitlines()
        if len(lines) <= self.truncate_lines:
            return text
        keep = self.truncate_lines // 2
        return "\n".join(
            lines[:keep]
            + [f"... ({len(lines) - 2 * keep} lines omitted) ..."]
            + lines[-keep:]
        )


class ContextManager:
    """Orchestrator holding static context, history, and cwd."""

    def __init__(
        self,
        static: StaticContext | None = None,
        history_length: int = 10,
        truncate_lines: int = 60,
        tree_depth: int = 3,
        tree_max_entries: int = 100,
    ):
        self.static = static or StaticContext.gather()
        self.history = ConversationHistory(history_length, truncate_lines)
        self.tree = DirectoryTree(tree_depth, tree_max_entries)
        self._cwd = os.getcwd()

    @property
    def cwd(self) -> str:
        return self._cwd

    def update_cwd(self, new_cwd: str) -> None:
        resolved = str(Path(new_cwd).expanduser().resolve())
        if os.path.isdir(resolved):
            self._cwd = resolved
            os.chdir(resolved)
            self.tree.invalidate(resolved)

    def build_prompt_context(self) -> dict[str, str]:
        return {
            "os_type": self.static.os_type,
            "shell_type": self.static.shell_type,
            "cwd": self._cwd,
            "dir_tree": self.tree.get(self._cwd),
            "username": self.static.username,
        }

    def get_chat_messages(self, user_input: str) -> list[dict[str, str]]:
        messages = self.history.to_messages()
        messages.append({"role": "user", "content": user_input})
        return messages

    def record_exchange(
        self,
        user_input: str,
        command: str,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = 0,
    ) -> None:
        self.history.add(ExchangeRecord(
            user_input=user_input,
            generated_command=command,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
        ))
