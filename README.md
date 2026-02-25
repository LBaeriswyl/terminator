# Natural Terminal

Translate natural language into shell commands using a local LLM via [Ollama](https://ollama.com). No external API calls — everything runs on your machine.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed (will be started automatically if not running)
- A pulled model (default: `llama3.1:8b`)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.1:8b
```

> **Note:** If Ollama is installed but not running, `nt` will start it automatically in the background. This only works for local connections — remote Ollama servers (`--url`) must be started separately.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Start the REPL
nt

# Use a different model
nt --model mistral

# Use a remote Ollama server
nt --url http://remote:11434

# Generate default config file
nt --init
```

### In the REPL

Type natural language and get shell commands:

```
nt> list all files including hidden ones
┌── SAFE ──────────────────────────┐
│ $ ls -la                         │
│ List all files in long format    │
└──────────────────────────────────┘

nt> find all python files and count them
┌── CAUTION ───────────────────────────────┐
│ $ find . -name "*.py" | wc -l            │
│ Find .py files recursively and count     │
└──────────────────────────────────────────┘
Execute? [y/N] y
42

nt> delete temp.log
┌── DANGER ────────────────────────┐
│ $ rm temp.log                    │
│ Remove the file temp.log         │
└──────────────────────────────────┘
Type "yes" to execute:
```

### Meta-commands

| Command          | Description                          |
|-----------------|--------------------------------------|
| `/raw <cmd>`    | Execute a shell command directly     |
| `/model <name>` | Switch the LLM model                 |
| `/history`      | Show conversation history            |
| `/context`      | Show current context sent to model   |
| `/clear`        | Clear conversation history           |
| `/help`         | Show available commands              |

### Safety levels

- **Green (SAFE)**: Read-only commands (`ls`, `cat`, `pwd`, etc.) — auto-execute if configured
- **Yellow (CAUTION)**: Modification commands (`cp`, `mv`, `mkdir`, pipes) — requires `y/N` confirmation
- **Red (DANGER)**: Destructive commands (`rm`, `kill`, `sudo`) — requires typing `yes`

Blocked patterns (e.g., `rm -rf /`, fork bombs) are always rejected.

## Configuration

Config file: `~/.natural-terminal/config.toml`

Generate the default config with `nt --init`.

```toml
[model]
name = "llama3.1:8b"
ollama_url = "http://localhost:11434"
timeout = 30

[safety]
auto_execute_safe = true
blocked_patterns = ["rm -rf /", ":(){ :|:& };:", "mkfs", "dd if="]

[context]
history_length = 10
dir_tree_depth = 3
dir_tree_max_entries = 100
output_truncate_lines = 60
```

## Development

```bash
# Run tests
pytest tests/ -v

# Run eval suite (requires Ollama)
python evals/eval_suite.py --model llama3.1:8b
python evals/eval_suite.py --model mistral --save
```

## Architecture

```
User input → REPL → Ollama /api/chat → Parse JSON → Safety classify → Confirm → Execute → History → Loop
```

Key modules:
- `llm.py` — Ollama client with structured JSON output via `format` parameter
- `executor.py` — Safety classification (GREEN/YELLOW/RED) and command execution
- `context.py` — Directory tree, conversation history, environment context
- `prompt.py` — System prompt template with few-shot examples and per-model overrides
- `config.py` — TOML configuration with defaults and CLI overrides
- `ui.py` — Rich panels + prompt_toolkit REPL
