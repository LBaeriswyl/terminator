"""Tests for per-model prompt resolution."""

from natural_terminal.prompt import (
    DEFAULT_FEW_SHOT_EXAMPLES,
    DEFAULT_SYSTEM_TEMPLATE,
    MODEL_OVERRIDES,
    build_system_prompt,
    get_few_shot_examples,
    get_prompt_config,
)

CUSTOM_TEMPLATE = "Custom template for {os_type} {shell_type} {cwd} {dir_tree} {username}"
CUSTOM_EXAMPLES = [{"role": "user", "content": "custom"}]
FAMILY_TEMPLATE = "Family template for {os_type} {shell_type} {cwd} {dir_tree} {username}"
FAMILY_EXAMPLES = [{"role": "user", "content": "family"}]

CONTEXT = {
    "os_type": "Linux",
    "shell_type": "bash",
    "cwd": "/tmp",
    "dir_tree": ".",
    "username": "test",
}


class TestGetPromptConfig:
    def test_unknown_model_returns_defaults(self, monkeypatch):
        monkeypatch.setattr("natural_terminal.prompt.MODEL_OVERRIDES", {})
        template, examples = get_prompt_config("unknown:latest")
        assert template == DEFAULT_SYSTEM_TEMPLATE
        assert examples == DEFAULT_FEW_SHOT_EXAMPLES

    def test_exact_match(self, monkeypatch):
        monkeypatch.setattr("natural_terminal.prompt.MODEL_OVERRIDES", {
            "mistral:7b": {
                "system_template": CUSTOM_TEMPLATE,
                "few_shot_examples": CUSTOM_EXAMPLES,
            },
        })
        template, examples = get_prompt_config("mistral:7b")
        assert template == CUSTOM_TEMPLATE
        assert examples == CUSTOM_EXAMPLES

    def test_family_match(self, monkeypatch):
        monkeypatch.setattr("natural_terminal.prompt.MODEL_OVERRIDES", {
            "mistral": {
                "system_template": FAMILY_TEMPLATE,
                "few_shot_examples": FAMILY_EXAMPLES,
            },
        })
        template, examples = get_prompt_config("mistral:7b")
        assert template == FAMILY_TEMPLATE
        assert examples == FAMILY_EXAMPLES

    def test_exact_takes_precedence_over_family(self, monkeypatch):
        monkeypatch.setattr("natural_terminal.prompt.MODEL_OVERRIDES", {
            "mistral": {
                "system_template": FAMILY_TEMPLATE,
                "few_shot_examples": FAMILY_EXAMPLES,
            },
            "mistral:7b": {
                "system_template": CUSTOM_TEMPLATE,
                "few_shot_examples": CUSTOM_EXAMPLES,
            },
        })
        template, examples = get_prompt_config("mistral:7b")
        assert template == CUSTOM_TEMPLATE
        assert examples == CUSTOM_EXAMPLES

    def test_partial_override_merges_across_levels(self, monkeypatch):
        """Exact has system_template, family has few_shot_examples."""
        monkeypatch.setattr("natural_terminal.prompt.MODEL_OVERRIDES", {
            "mistral": {
                "few_shot_examples": FAMILY_EXAMPLES,
            },
            "mistral:7b": {
                "system_template": CUSTOM_TEMPLATE,
            },
        })
        template, examples = get_prompt_config("mistral:7b")
        assert template == CUSTOM_TEMPLATE
        assert examples == FAMILY_EXAMPLES

    def test_partial_override_falls_to_default(self, monkeypatch):
        """Only system_template in exact, no family — few_shot_examples falls to default."""
        monkeypatch.setattr("natural_terminal.prompt.MODEL_OVERRIDES", {
            "mistral:7b": {
                "system_template": CUSTOM_TEMPLATE,
            },
        })
        template, examples = get_prompt_config("mistral:7b")
        assert template == CUSTOM_TEMPLATE
        assert examples == DEFAULT_FEW_SHOT_EXAMPLES

    def test_model_without_tag(self, monkeypatch):
        """Model name without colon — no family fallback, just exact or default."""
        monkeypatch.setattr("natural_terminal.prompt.MODEL_OVERRIDES", {
            "mistral": {
                "system_template": CUSTOM_TEMPLATE,
            },
        })
        template, examples = get_prompt_config("mistral")
        assert template == CUSTOM_TEMPLATE
        assert examples == DEFAULT_FEW_SHOT_EXAMPLES


class TestGetFewShotExamples:
    def test_none_returns_default(self):
        assert get_few_shot_examples(None) == DEFAULT_FEW_SHOT_EXAMPLES

    def test_with_model_override(self, monkeypatch):
        monkeypatch.setattr("natural_terminal.prompt.MODEL_OVERRIDES", {
            "mistral": {"few_shot_examples": CUSTOM_EXAMPLES},
        })
        assert get_few_shot_examples("mistral:7b") == CUSTOM_EXAMPLES

    def test_unknown_model_returns_default(self, monkeypatch):
        monkeypatch.setattr("natural_terminal.prompt.MODEL_OVERRIDES", {})
        assert get_few_shot_examples("unknown:latest") == DEFAULT_FEW_SHOT_EXAMPLES


class TestBuildSystemPrompt:
    def test_without_model_uses_default(self):
        result = build_system_prompt(CONTEXT)
        assert "shell command translator" in result
        assert "Linux" in result

    def test_with_model_override(self, monkeypatch):
        monkeypatch.setattr("natural_terminal.prompt.MODEL_OVERRIDES", {
            "mistral": {"system_template": CUSTOM_TEMPLATE},
        })
        result = build_system_prompt(CONTEXT, model="mistral:7b")
        assert result == "Custom template for Linux bash /tmp . test"

    def test_with_unknown_model_uses_default(self, monkeypatch):
        monkeypatch.setattr("natural_terminal.prompt.MODEL_OVERRIDES", {})
        result = build_system_prompt(CONTEXT, model="unknown:latest")
        assert "shell command translator" in result
