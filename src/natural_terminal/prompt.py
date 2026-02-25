"""System prompt templates and few-shot examples for LLM command generation."""

DEFAULT_SYSTEM_TEMPLATE = """\
You are a shell command translator. Your job is to convert natural language \
requests into shell commands.

Environment:
- OS: {os_type}
- Shell: {shell_type}
- Current directory: {cwd}
- Directory contents:
{dir_tree}
- User: {username}

Rules:
1. Respond ONLY in valid JSON. No markdown, no explanation outside the JSON.
2. For commands: {{"type": "command", "command": "<shell command>", "explanation": "<brief one-line explanation>"}}
3. For clarifications: {{"type": "clarify", "question": "<what you need to know>"}}
4. Generate commands for {shell_type}. Use POSIX-compatible syntax when possible.
5. Prefer simple, common commands over complex ones.
6. Never generate commands that you aren't confident about — ask for clarification instead.
7. For file paths, use paths relative to the current directory when possible.
8. When referring to previous commands or outputs, use the conversation history.\
"""

DEFAULT_FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "list all files including hidden ones",
    },
    {
        "role": "assistant",
        "content": '{"type": "command", "command": "ls -la", "explanation": "List all files including hidden ones in long format"}',
    },
    {
        "role": "user",
        "content": "delete something",
    },
    {
        "role": "assistant",
        "content": '{"type": "clarify", "question": "What file or directory would you like to delete?"}',
    },
    {
        "role": "user",
        "content": "find all python files and count them",
    },
    {
        "role": "assistant",
        "content": '{"type": "command", "command": "find . -name \\"*.py\\" | wc -l", "explanation": "Find all .py files recursively and count the total"}',
    },
]

# Backward-compatible aliases
SYSTEM_TEMPLATE = DEFAULT_SYSTEM_TEMPLATE
FEW_SHOT_EXAMPLES = DEFAULT_FEW_SHOT_EXAMPLES

# Per-model prompt overrides.
# Keys: exact model ("llama3.1:8b") or family ("llama3.1", i.e. before the colon).
# Values: dict with optional "system_template" and/or "few_shot_examples".
# Resolution: exact model -> family -> defaults.
#
# Example:
#   MODEL_OVERRIDES = {
#       "mistral": {
#           "system_template": MISTRAL_TEMPLATE,  # all mistral variants
#       },
#       "codellama:13b-instruct": {
#           "few_shot_examples": CODELLAMA_INSTRUCT_EXAMPLES,  # this specific variant
#       },
#   }
MODEL_OVERRIDES: dict[str, dict] = {}


def get_prompt_config(model: str) -> tuple[str, list[dict[str, str]]]:
    """Return (system_template, few_shot_examples) for a model.

    Resolution: exact match -> family match -> defaults.
    For each field independently: if exact match doesn't define it,
    check family, then default.
    """
    family = model.split(":")[0] if ":" in model else None
    exact = MODEL_OVERRIDES.get(model, {})
    fam = MODEL_OVERRIDES.get(family, {}) if family else {}

    template = (
        exact.get("system_template")
        or fam.get("system_template")
        or DEFAULT_SYSTEM_TEMPLATE
    )
    examples = (
        exact.get("few_shot_examples")
        or fam.get("few_shot_examples")
        or DEFAULT_FEW_SHOT_EXAMPLES
    )
    return (template, examples)


def get_few_shot_examples(model: str | None = None) -> list[dict[str, str]]:
    if model:
        _, examples = get_prompt_config(model)
        return examples
    return DEFAULT_FEW_SHOT_EXAMPLES


def build_system_prompt(context: dict, model: str | None = None) -> str:
    if model:
        template, _ = get_prompt_config(model)
    else:
        template = DEFAULT_SYSTEM_TEMPLATE
    return template.format(**context)
