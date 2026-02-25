"""Command execution, safety classification, and shell builtin handling."""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from enum import Enum


class SafetyLevel(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


GREEN_COMMANDS = frozenset({
    "ls", "ll", "la", "dir", "cat", "head", "tail", "less", "more",
    "pwd", "echo", "whoami", "hostname", "uname", "date", "cal",
    "wc", "sort", "uniq", "diff", "cmp", "file", "stat", "du", "df",
    "find", "locate", "which", "whereis", "type", "whatis",
    "grep", "egrep", "fgrep", "rg", "ag", "ack",
    "tree", "realpath", "basename", "dirname",
    "env", "printenv", "set", "id", "groups", "uptime",
    "ps", "top", "htop", "free", "vmstat", "iostat", "lsof",
    "ping", "traceroute", "dig", "nslookup", "host", "curl", "wget",
    "man", "help", "info", "history",
    "git", "svn",
    "python3", "python", "node", "ruby", "perl",
    "jq", "yq", "xargs", "tee", "seq", "tr", "cut", "paste", "fold",
    "md5sum", "sha256sum", "sha1sum", "shasum",
    "ip", "ifconfig", "netstat", "ss",
    "vm_stat", "sysctl", "sw_vers", "system_profiler",
})

RED_COMMANDS = frozenset({
    "rm", "rmdir", "kill", "killall", "pkill",
    "dd", "shred", "wipefs",
    "mkfs", "fdisk", "parted", "gdisk",
    "reboot", "shutdown", "halt", "poweroff", "init",
    "iptables", "nft", "ufw",
    "useradd", "userdel", "usermod", "groupadd", "groupdel",
    "crontab",
    "systemctl",
})

INTERACTIVE_COMMANDS = frozenset({
    "vim", "vi", "nvim", "nano", "emacs", "pico", "joe", "micro",
    "top", "htop", "btop", "glances",
    "python", "python3", "ipython", "node", "irb", "ghci", "lua", "erl",
    "ssh", "telnet", "ftp", "sftp",
    "mysql", "psql", "sqlite3", "mongosh", "redis-cli",
    "less", "more", "man",
    "bash", "zsh", "sh", "fish",
    "tmux", "screen",
    "gdb", "lldb",
})

# Prefixes to skip when extracting the "real" command
COMMAND_PREFIXES = frozenset({
    "sudo", "env", "nice", "nohup", "time", "timeout", "strace", "ltrace",
})

SHELL_BUILTINS = frozenset({"cd", "export", "alias", "source", "."})


class SafetyClassifier:
    def __init__(self, blocked_patterns: list[str] | None = None):
        self.blocked_patterns = blocked_patterns or []

    def classify(self, command: str) -> SafetyLevel:
        if self.is_blocked(command):
            return SafetyLevel.RED

        has_sudo = "sudo " in command or command.startswith("sudo")

        # Pipes, chains, semicolons → minimum YELLOW
        has_chain = bool(re.search(r"[|&;]", command))

        # Output redirection → YELLOW
        has_redirect = bool(re.search(r">{1,2}\s*\S", command))

        first_token = self._extract_first_token(command)

        if has_sudo:
            return SafetyLevel.RED

        if first_token in RED_COMMANDS:
            return SafetyLevel.RED

        if first_token in GREEN_COMMANDS and not has_chain and not has_redirect:
            return SafetyLevel.GREEN

        if has_chain or has_redirect:
            return SafetyLevel.YELLOW

        # Default: YELLOW for unknown commands
        return SafetyLevel.YELLOW

    def is_blocked(self, command: str) -> bool:
        for pattern in self.blocked_patterns:
            if pattern in command:
                return True
        return False

    def is_interactive(self, command: str) -> bool:
        first_token = self._extract_first_token(command)
        return first_token in INTERACTIVE_COMMANDS

    def _extract_first_token(self, command: str) -> str:
        try:
            tokens = shlex.split(command)
        except ValueError:
            # Fallback for unbalanced quotes etc.
            tokens = command.split()

        if not tokens:
            return ""

        # Skip prefixes like sudo, env, nice, etc.
        for i, token in enumerate(tokens):
            # Handle env with VAR=val
            if token == "env":
                continue
            if "=" in token and i > 0:
                continue
            if token in COMMAND_PREFIXES:
                continue
            return token

        return tokens[-1] if tokens else ""


class CommandExecutor:
    def __init__(self, default_timeout: int = 30):
        self.default_timeout = default_timeout

    def execute(self, command: str, cwd: str, timeout: int | None = None) -> ExecutionResult:
        actual_timeout = timeout or self.default_timeout

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if sys.platform != "win32" else None,
            )
        except OSError as e:
            return ExecutionResult(stdout="", stderr=str(e), exit_code=1)

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        timed_out = False

        def read_stream(stream, accumulator, print_fn):
            for line in iter(stream.readline, b""):
                decoded = line.decode("utf-8", errors="replace")
                accumulator.append(decoded)
                try:
                    print_fn(decoded, end="")
                except TypeError:
                    print_fn(decoded)
            stream.close()

        stdout_thread = threading.Thread(
            target=read_stream,
            args=(process.stdout, stdout_lines, sys.stdout.write),
        )
        stderr_thread = threading.Thread(
            target=read_stream,
            args=(process.stderr, stderr_lines, sys.stderr.write),
        )

        stdout_thread.start()
        stderr_thread.start()

        try:
            process.wait(timeout=actual_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                process.kill()

        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

        return ExecutionResult(
            stdout="".join(stdout_lines),
            stderr="".join(stderr_lines),
            exit_code=process.returncode if process.returncode is not None else -1,
            timed_out=timed_out,
        )

    def execute_interactive(self, command: str, cwd: str) -> ExecutionResult:
        """Execute an interactive command with terminal inheritance."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
            )
            return ExecutionResult(
                stdout="",
                stderr="",
                exit_code=result.returncode,
            )
        except OSError as e:
            return ExecutionResult(stdout="", stderr=str(e), exit_code=1)

    def handle_builtin(self, command: str, cwd: str) -> tuple[bool, str]:
        """Handle shell builtins. Returns (handled, message)."""
        stripped = command.strip()

        # cd handling
        cd_match = re.match(r"^cd\s*(.*)", stripped)
        if cd_match:
            target = cd_match.group(1).strip()
            if not target or target == "~":
                target = str(os.path.expanduser("~"))
            elif target.startswith("~"):
                target = str(os.path.expanduser(target))
            elif not os.path.isabs(target):
                target = os.path.join(cwd, target)

            target = os.path.realpath(target)
            if os.path.isdir(target):
                os.chdir(target)
                return True, target
            else:
                return True, f"cd: no such directory: {cd_match.group(1).strip()}"

        # export handling
        export_match = re.match(r"^export\s+(\w+)=(.*)", stripped)
        if export_match:
            key = export_match.group(1)
            val = export_match.group(2).strip().strip("'\"")
            os.environ[key] = val
            return True, f"Set {key}={val}"

        # alias — reject
        if stripped.startswith("alias"):
            return True, "alias is not supported (session-specific; use shell config instead)"

        return False, ""

    def detect_cd_in_command(self, command: str) -> str | None:
        """Detect cd targets in chained commands like 'cd /tmp && ls'."""
        # Look for cd invocations
        cd_patterns = re.findall(r"cd\s+([^\s;&|]+)", command)
        if cd_patterns:
            # Return the last cd target
            target = cd_patterns[-1]
            if target.startswith("~"):
                target = os.path.expanduser(target)
            return target
        return None
