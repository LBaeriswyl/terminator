"""Configuration management with defaults, TOML file, and CLI overrides."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

CONFIG_DIR = Path.home() / ".natural-terminal"
CONFIG_FILE = CONFIG_DIR / "config.toml"
HISTORY_FILE = CONFIG_DIR / "history.txt"

DEFAULT_BLOCKED_PATTERNS = [
    "rm -rf /",
    ":(){ :|:& };:",
    "mkfs",
    "dd if=",
    "> /dev/sda",
    "chmod -R 777 /",
]

DEFAULT_CONFIG_TOML = """\
# Natural Terminal configuration

[model]
name = "llama3.1:8b"
server_url = "http://localhost:8080"
timeout = 30  # seconds for LLM response
model_path = ""  # path to GGUF file (required for auto-start)
models_dir = ""  # directory of GGUF files (for /model switching)

[safety]
auto_execute_safe = true  # auto-run green commands without confirmation
blocked_patterns = ["rm -rf /", ":(){ :|:& };:", "mkfs", "dd if="]

[context]
history_length = 10
dir_tree_depth = 3
dir_tree_max_entries = 100
output_truncate_lines = 60
"""


@dataclass
class ModelConfig:
    name: str = "llama3.1:8b"
    server_url: str = "http://localhost:8080"
    timeout: int = 30
    model_path: str = ""
    models_dir: str = ""


@dataclass
class SafetyConfig:
    auto_execute_safe: bool = True
    blocked_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_BLOCKED_PATTERNS))


@dataclass
class ContextConfig:
    history_length: int = 10
    dir_tree_depth: int = 3
    dir_tree_max_entries: int = 100
    output_truncate_lines: int = 60


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    context: ContextConfig = field(default_factory=ContextConfig)

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
        cli_overrides: dict[str, Any] | None = None,
    ) -> AppConfig:
        config = cls()

        # Load from TOML file
        path = config_path or CONFIG_FILE
        if path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)
            config._apply_toml(data)

        # Apply CLI overrides
        if cli_overrides:
            config._apply_overrides(cli_overrides)

        return config

    def _apply_toml(self, data: dict[str, Any]) -> None:
        if "model" in data:
            m = data["model"]
            if "name" in m:
                self.model.name = m["name"]
            if "server_url" in m:
                self.model.server_url = m["server_url"]
            elif "ollama_url" in m:
                print(
                    "Warning: 'ollama_url' in config.toml is deprecated, "
                    "use 'server_url' instead.",
                    file=sys.stderr,
                )
                self.model.server_url = m["ollama_url"]
            if "timeout" in m:
                self.model.timeout = int(m["timeout"])
            if "model_path" in m:
                self.model.model_path = m["model_path"]
            if "models_dir" in m:
                self.model.models_dir = m["models_dir"]

        if "safety" in data:
            s = data["safety"]
            if "auto_execute_safe" in s:
                self.safety.auto_execute_safe = bool(s["auto_execute_safe"])
            if "blocked_patterns" in s:
                self.safety.blocked_patterns = list(s["blocked_patterns"])

        if "context" in data:
            c = data["context"]
            if "history_length" in c:
                self.context.history_length = int(c["history_length"])
            if "dir_tree_depth" in c:
                self.context.dir_tree_depth = int(c["dir_tree_depth"])
            if "dir_tree_max_entries" in c:
                self.context.dir_tree_max_entries = int(c["dir_tree_max_entries"])
            if "output_truncate_lines" in c:
                self.context.output_truncate_lines = int(c["output_truncate_lines"])

    def _apply_overrides(self, overrides: dict[str, Any]) -> None:
        if "model" in overrides:
            self.model.name = overrides["model"]
        if "url" in overrides:
            self.model.server_url = overrides["url"]
        if "timeout" in overrides:
            self.model.timeout = int(overrides["timeout"])
        if "model_path" in overrides:
            self.model.model_path = overrides["model_path"]
        if "models_dir" in overrides:
            self.model.models_dir = overrides["models_dir"]


def ensure_config_dir() -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def write_default_config(path: Path | None = None) -> Path:
    target = path or CONFIG_FILE
    ensure_config_dir()
    target.write_text(DEFAULT_CONFIG_TOML)
    return target
