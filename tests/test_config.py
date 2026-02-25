"""Tests for configuration loading and override precedence."""

import pytest
from pathlib import Path

from natural_terminal.config import AppConfig, ModelConfig, SafetyConfig, ContextConfig, write_default_config


class TestDefaults:
    def test_default_model(self, default_config):
        assert default_config.model.name == "llama3.1:8b"
        assert default_config.model.ollama_url == "http://localhost:11434"
        assert default_config.model.timeout == 30

    def test_default_safety(self, default_config):
        assert default_config.safety.auto_execute_safe is True
        assert len(default_config.safety.blocked_patterns) > 0
        assert "rm -rf /" in default_config.safety.blocked_patterns

    def test_default_context(self, default_config):
        assert default_config.context.history_length == 10
        assert default_config.context.dir_tree_depth == 3
        assert default_config.context.dir_tree_max_entries == 100
        assert default_config.context.output_truncate_lines == 60


class TestTomlLoading:
    def test_load_from_toml(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[model]
name = "mistral"
ollama_url = "http://remote:11434"
timeout = 60

[safety]
auto_execute_safe = false
blocked_patterns = ["rm -rf /"]

[context]
history_length = 5
dir_tree_depth = 2
""")
        config = AppConfig.load(config_path=config_file)
        assert config.model.name == "mistral"
        assert config.model.ollama_url == "http://remote:11434"
        assert config.model.timeout == 60
        assert config.safety.auto_execute_safe is False
        assert config.safety.blocked_patterns == ["rm -rf /"]
        assert config.context.history_length == 5
        assert config.context.dir_tree_depth == 2
        # Unchanged defaults
        assert config.context.dir_tree_max_entries == 100

    def test_missing_file_uses_defaults(self, tmp_path):
        config = AppConfig.load(config_path=tmp_path / "nonexistent.toml")
        assert config.model.name == "llama3.1:8b"

    def test_partial_toml(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[model]
name = "codellama"
""")
        config = AppConfig.load(config_path=config_file)
        assert config.model.name == "codellama"
        assert config.model.ollama_url == "http://localhost:11434"


class TestOverrides:
    def test_cli_overrides(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[model]
name = "mistral"
""")
        config = AppConfig.load(
            config_path=config_file,
            cli_overrides={"model": "codellama", "url": "http://other:11434"},
        )
        # CLI overrides win
        assert config.model.name == "codellama"
        assert config.model.ollama_url == "http://other:11434"

    def test_overrides_without_toml(self, tmp_path):
        config = AppConfig.load(
            config_path=tmp_path / "nope.toml",
            cli_overrides={"model": "phi3"},
        )
        assert config.model.name == "phi3"


class TestWriteDefault:
    def test_write_default_config(self, tmp_path):
        path = write_default_config(tmp_path / "config.toml")
        assert path.exists()
        content = path.read_text()
        assert "[model]" in content
        assert "[safety]" in content
        assert "[context]" in content
