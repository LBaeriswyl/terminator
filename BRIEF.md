# Natural Language Terminal — Project Brief

## Overview

Build a Python CLI application that translates natural language commands into shell commands and executes them, using a locally-running open-source LLM (via Ollama). No external API calls. The app should feel like a smart terminal where users type things like "list all files" and get `ls -la` executed.

## Architecture

```
User input (natural language)
    → Python REPL loop
    → Ollama REST API (localhost:11434)
    → LLM generates shell command
    → Display command + ask for confirmation
    → Execute via subprocess
    → Capture and display output
    → Store in conversation history
    → Loop
```

## Tech Stack

- **Language:** Python 3.10+
- **LLM Runtime:** Ollama (must be installed separately; app should detect if it's running and give helpful errors if not)
- **Model:** Default to `llama3.1:8b` but make configurable. Other good options: `mistral`, `codellama`, `deepseek-coder`
- **Key libraries:**
  - `httpx` or `requests` — Ollama API calls (`POST http://localhost:11434/api/generate`)
  - `subprocess` — shell command execution
  - `prompt_toolkit` — interactive REPL with history, autocomplete, key bindings
  - `rich` — colored/formatted terminal output

## Core Features

### 1. Interactive REPL Loop
- Custom prompt (e.g., `nl> ` or `🧠> `)
- Input history (up/down arrows) via prompt_toolkit
- Graceful exit on `exit`, `quit`, Ctrl+C, Ctrl+D
- Special meta-commands prefixed with `/`:
  - `/raw <command>` — bypass LLM, execute shell command directly
  - `/model <name>` — switch Ollama model
  - `/history` — show recent conversation history
  - `/context` — show current context being sent to model
  - `/clear` — clear conversation history
  - `/help` — list available meta-commands

### 2. LLM Command Generation
- Send natural language to Ollama with a carefully crafted system prompt
- Parse the response to extract the shell command
- Support two response types from the model:
  - **Command:** a shell command to execute
  - **Clarification:** the model asks for more info instead of guessing
- Use structured output format (see System Prompt section below)

### 3. Safety & Confirmation
- **Always** display the generated command before execution
- Color-code by risk level:
  - 🟢 Green: read-only commands (`ls`, `cat`, `pwd`, `echo`, `find`, `grep`, etc.)
  - 🟡 Yellow: modification commands (`cp`, `mv`, `mkdir`, `touch`, `chmod`, etc.)
  - 🔴 Red: destructive commands (`rm`, `rmdir`, `kill`, `dd`, `mkfs`, `> file`, etc.)
- Green commands can auto-execute (configurable)
- Yellow commands require `y/n` confirmation
- Red commands require typing `yes` in full
- Maintain a configurable blocklist of patterns that are always blocked or always require confirmation
- **Never** auto-execute piped commands or command chains (`|`, `&&`, `||`, `;`) without confirmation

### 4. Context Management

#### Static Context (generated at startup, refreshed periodically)
- Current working directory
- OS and shell type (`bash`, `zsh`, etc.)
- Directory tree of cwd (max depth 3, max ~100 entries, truncate with "... and N more")
- Username and home directory

#### Conversation History (rolling window)
- Store last 10 exchanges as `{ user_input, generated_command, stdout, stderr, exit_code }`
- Truncate command outputs to first/last 30 lines if they exceed 60 lines total
- This enables self-referential queries like "now do the same for the next file" or "filter that output"

#### Dynamic Context
- When cwd changes (user does `cd`), refresh the directory tree context
- Track cwd changes from executed commands

### 5. Output Handling
- Stream command stdout/stderr to terminal in real-time
- Capture output for conversation history
- Handle long-running commands (show a spinner or elapsed time)
- Handle interactive commands gracefully (or detect and warn that interactive commands aren't supported)

## System Prompt Design

This is critical. The system prompt sent to Ollama should:

```
You are a shell command translator. Your job is to convert natural language requests into shell commands.

Environment:
- OS: {os_type}
- Shell: {shell_type}
- Current directory: {cwd}
- Directory contents: {dir_tree}
- User: {username}

Rules:
1. Respond ONLY in the following JSON format, nothing else:
   - For commands: {"type": "command", "command": "<shell command>", "explanation": "<brief one-line explanation>"}
   - For clarifications: {"type": "clarify", "question": "<what you need to know>"}
2. Generate commands for {shell_type}. Use POSIX-compatible syntax when possible.
3. Prefer simple, common commands over complex ones.
4. Never generate commands that you aren't confident about — ask for clarification instead.
5. For file paths, use paths relative to the current directory when possible.

Recent conversation:
{conversation_history}
```

**Important prompt engineering notes:**
- Small models struggle with JSON output. You may need to fall back to regex parsing if JSON fails.
- Include 2-3 few-shot examples in the system prompt to anchor the output format.
- Keep total prompt under ~2000 tokens to maintain speed and leave room for the response.

## Error Handling & Edge Cases

- **Ollama not running:** Detect on startup, print install/start instructions
- **Model not pulled:** Detect and offer to run `ollama pull <model>`
- **LLM returns unparseable output:** Retry once with a shorter/stricter prompt. If still bad, show raw output and ask user to rephrase.
- **Command times out:** Configurable timeout (default 30s), option to kill or let it continue
- **Interactive commands** (`vim`, `top`, `python`): Detect common interactive commands and warn the user, or pass through to a proper PTY
- **Directory changes:** If the generated command contains `cd`, update the app's internal cwd (subprocess won't persist cwd changes otherwise — this is a known pain point; you need to handle `cd` specially by changing `os.chdir()` in the parent process)
- **Sudo commands:** Flag and require explicit confirmation; consider whether to support at all initially

## Project Structure

```
terminator/
├── pyproject.toml          # Project config, dependencies
├── README.md
├── src/
│   └── terminator/
│       ├── __init__.py
│       ├── main.py          # Entry point, REPL loop
│       ├── llm.py           # Ollama API client, prompt construction, response parsing
│       ├── executor.py      # Command execution, safety classification, confirmation
│       ├── context.py       # Context management (filesystem, history, cwd tracking)
│       ├── prompt.py        # System prompt templates and few-shot examples
│       ├── config.py        # Configuration (model, safety levels, timeouts, etc.)
│       └── ui.py            # Terminal UI (colors, formatting, prompt_toolkit setup)
└── tests/
    ├── test_llm.py          # Test prompt construction and response parsing
    ├── test_executor.py     # Test safety classification
    └── test_context.py      # Test context generation and truncation
```

## Configuration

Support a config file (`~/.terminator/config.toml` or similar):

```toml
[model]
name = "llama3.1:8b"
ollama_url = "http://localhost:11434"
timeout = 30  # seconds for LLM response

[safety]
auto_execute_safe = true  # auto-run green commands without confirmation
blocked_patterns = ["rm -rf /", ":(){ :|:& };:", "mkfs", "dd if="]
require_full_yes = ["rm", "kill", "dd"]

[context]
history_length = 10
dir_tree_depth = 3
dir_tree_max_entries = 100
output_truncate_lines = 60
```

## Getting Started (for development)

1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
2. Pull a model: `ollama pull llama3.1:8b`
3. Set up Python project with dependencies
4. Implement in this order:
   a. Ollama API client + basic prompt → get a command back from the model
   b. REPL loop with prompt_toolkit
   c. Command execution with subprocess
   d. Safety classification and confirmation flow
   e. Context management (cwd, dir tree, conversation history)
   f. Config file support
   g. Polish UI with rich
   h. Error handling and edge cases

## Open Questions to Resolve During Development

- **Streaming vs. waiting:** Should we stream the LLM response token-by-token (feels more responsive) or wait for the full response then parse? Streaming is harder to parse as JSON. Consider waiting for full response but showing a spinner.
- **Multi-command support:** Should "find all .py files and count them" produce `find . -name "*.py" | wc -l`? Piped commands are powerful but riskier. Start by supporting them but always requiring confirmation.
- **Command correction:** If a command fails, should we automatically send the error back to the LLM and ask it to fix the command? This could be a nice UX feature but adds complexity.
- **Shell built-ins:** Things like `cd`, `export`, `alias` don't work in subprocess. These need special handling in the executor.
