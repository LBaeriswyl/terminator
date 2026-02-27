# Natural Terminal

Translate natural language into shell commands using a local LLM via [llama.cpp](https://github.com/ggerganov/llama.cpp). No external API calls — everything runs on your machine.

## Prerequisites

- Python 3.10+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) installed (`llama-server` binary in PATH)
- A GGUF model file (e.g., from [Hugging Face](https://huggingface.co))

```bash
# Install llama.cpp (macOS)
brew install llama.cpp

# Download a model (example)
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir ~/models
```

> **Note:** If `llama-server` is installed but not running, `nt` will start it automatically using the configured `model_path`. This only works for local connections — remote servers (`--url`) must be started separately.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Start the REPL (requires llama-server running or model_path configured)
nt

# Use a different model name (for prompt overrides)
nt --model mistral

# Use a remote LLM server
nt --url http://remote:8080

# Specify a GGUF model file for auto-start
nt --model-path ~/models/llama3.1-8b-Q4_K_M.gguf

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

| Command          | Description                                    |
|-----------------|------------------------------------------------|
| `/raw <cmd>`    | Execute a shell command directly               |
| `/model [name]` | Show or switch the model (requires models_dir) |
| `/history`      | Show conversation history                      |
| `/context`      | Show current context sent to model             |
| `/clear`        | Clear conversation history                     |
| `/help`         | Show available commands                        |

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
server_url = "http://localhost:8080"
timeout = 30
model_path = ""      # path to GGUF file (required for auto-start)
models_dir = ""      # directory of GGUF files (for /model switching)

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

# Run eval suite (requires running llama-server)
python evals/eval_suite.py --model llama3.1:8b
python evals/eval_suite.py --model mistral --save
```

## Architecture

```
User input → REPL → llama-server /v1/chat/completions → Parse JSON → Safety classify → Confirm → Execute → History → Loop
```

Key modules:
- `llm.py` — llama-server client with structured JSON output via `response_format`
- `executor.py` — Safety classification (GREEN/YELLOW/RED) and command execution
- `context.py` — Directory tree, conversation history, environment context
- `prompt.py` — System prompt template with few-shot examples and per-model overrides
- `config.py` — TOML configuration with defaults and CLI overrides
- `ui.py` — Rich panels + prompt_toolkit REPL
