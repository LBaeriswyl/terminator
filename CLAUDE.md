# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (requires Python 3.10+, uses .venv)
source .venv/bin/activate
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test file or test
pytest tests/test_executor.py -v
pytest tests/test_executor.py::TestSafetyClassifier::test_green_commands -v

# Run with coverage
pytest tests/ --cov=natural_terminal

# Run the app
nt                          # default model
nt --model mistral          # different model
nt --url http://remote:11434

# Run eval suite (requires running Ollama instance)
python evals/eval_suite.py --model llama3.1:8b
python evals/eval_suite.py --model llama3.1:8b --save
```

## Architecture

The app translates natural language → shell commands via a local Ollama LLM. The REPL loop lives in `main.py:App.run()` and orchestrates all other modules:

```
User input → LLM request (with spinner) → Parse JSON response →
Safety classify (GREEN/YELLOW/RED) → Confirm → Execute → Record history → Loop
```

**Module dependency graph:**
- `main.py` → imports and wires together all modules below
- `llm.py` → Ollama `/api/chat` client; calls `get_few_shot_examples(model)` from `prompt.py`
- `prompt.py` → System prompt template with `{os_type}`, `{shell_type}`, `{cwd}`, `{dir_tree}`, `{username}` placeholders; per-model overrides via `MODEL_OVERRIDES` dict with 3-level resolution (exact model → family → defaults)
- `executor.py` → Safety classification (curated GREEN/RED/INTERACTIVE command sets) + subprocess execution
- `context.py` → StaticContext, DirectoryTree (cached, 500ms timeout), ConversationHistory (rolling window with output truncation), ContextManager
- `config.py` → Dataclass hierarchy loaded as: defaults → TOML (`~/.natural-terminal/config.toml`) → CLI args
- `ui.py` → Rich panels (border color = safety level) + prompt_toolkit REPL with FileHistory

## Key Design Decisions

**LLM response parsing** uses a 4-tier fallback because small models produce unreliable JSON: direct JSON parse → extract from markdown fences → regex for `{...}` → treat bare single-line output as a command. The `format` parameter on `/api/chat` provides grammar-level JSON enforcement from Ollama's side.

**`cd` requires special handling** — subprocess can't change the parent process's cwd. `executor.handle_builtin()` calls `os.chdir()` directly for `cd`, and `executor.detect_cd_in_command()` catches `cd` inside chained commands (`cd /tmp && ls`). `export` similarly sets `os.environ` directly.

**Safety classification** extracts the "real" command by skipping prefixes (`sudo`, `env`, `nohup`, `time`), then looks up against curated sets. Pipes/chains/redirects force minimum YELLOW regardless of the base command.

**Auto-start Ollama** — `_try_start_ollama()` in `main.py` transparently starts Ollama if it's not running. Only triggers for localhost/127.0.0.1 connections (remote URLs get an error instead). Uses `Popen` with `start_new_session=True` so Ollama outlives the app, then polls `check_health()` with a spinner for up to 15s. Ollama itself handles duplicate instances (exits if port taken).

**Per-model prompt overrides** — `prompt.py` has a `MODEL_OVERRIDES` dict keyed by exact model name (`"llama3.1:8b"`) or family (`"llama3.1"`). Resolution checks exact → family → `DEFAULT_*` for each of `system_template` and `few_shot_examples` independently. `build_system_prompt()` and `get_few_shot_examples()` accept an optional `model` parameter; without it they use defaults (backward compatible).

**Signal handling** is state-dependent: SIGINT during LLM request raises KeyboardInterrupt to cancel httpx; during child execution it's forwarded to the process group; at the REPL prompt it's ignored.

## Testing

Tests use `respx` to mock httpx (no Ollama needed). Test files mirror source modules: `test_llm.py`, `test_executor.py`, `test_context.py`, `test_config.py`, `test_prompt.py`, `test_main.py`. Tests that call `os.chdir()` or `os.environ` restore state in teardown. `test_main.py` uses `unittest.mock.patch` on `shutil.which`, `subprocess.Popen`, `time.sleep`, and `client.check_health` to test auto-start logic. `test_prompt.py` uses `monkeypatch` on `MODEL_OVERRIDES` to test 3-level resolution.

The `evals/` directory is a separate quality benchmark (not CI) — it scores LLM+prompt combinations against 51 cases with exact/partial/fail scoring. Eval output includes `prompt_version` to track which prompt config was used. Run evals after changing `prompt.py`, the `format` parameter, or switching models.
