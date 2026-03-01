"""Main entry point — REPL loop, argument parsing, and signal handling."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import httpx

from terminator import __version__
from terminator.config import AppConfig, write_default_config, CONFIG_FILE
from terminator.context import ContextManager
from terminator.executor import CommandExecutor, SafetyClassifier, SafetyLevel
from terminator.llm import LlamaCppClient, CommandResponse, ClarifyResponse, ParseError
from terminator.prompt import build_system_prompt
from terminator.ui import TerminalUI


class App:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = LlamaCppClient(
            base_url=config.model.server_url,
            model=config.model.name,
            timeout=config.model.timeout,
        )
        self.context = ContextManager(
            history_length=config.context.history_length,
            truncate_lines=config.context.output_truncate_lines,
            tree_depth=config.context.dir_tree_depth,
            tree_max_entries=config.context.dir_tree_max_entries,
        )
        self.classifier = SafetyClassifier(
            blocked_patterns=config.safety.blocked_patterns,
        )
        self.executor = CommandExecutor(default_timeout=config.model.timeout)
        self.ui = TerminalUI(auto_execute_safe=config.safety.auto_execute_safe)
        self._llm_active = False
        self._child_running = False
        self._server_proc: subprocess.Popen | None = None

    def run(self) -> None:
        # Startup checks
        if not self._startup_checks():
            return

        # Install signal handler
        signal.signal(signal.SIGINT, self._handle_sigint)

        self.ui.show_welcome(self.client.model, self.client.base_url)

        # REPL loop
        while True:
            try:
                user_input = self.ui.get_input()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                break

            if user_input.startswith("/"):
                self._handle_meta_command(user_input)
                continue

            self._handle_natural_input(user_input)

        self.client.close()

    def _startup_checks(self) -> bool:
        if not self.client.check_health():
            if not self._try_start_server():
                return False
        return True

    def _try_start_server(self) -> bool:
        """Try to start llama-server locally. Returns True if the server is now reachable."""
        parsed = urllib.parse.urlparse(self.client.base_url)
        host = parsed.hostname or ""
        port = parsed.port or 8080
        if host not in ("localhost", "127.0.0.1"):
            self.ui.show_error(
                f"Cannot reach LLM server at {self.client.base_url}\n"
                "  Check that the server is running and accessible."
            )
            return False

        server_path = shutil.which("llama-server")
        if not server_path:
            self.ui.show_error(
                "llama-server (llama.cpp) is not installed.\n"
                "  Install with: brew install llama.cpp\n"
                "  More info: https://github.com/ggerganov/llama.cpp"
            )
            return False

        model_path = self.config.model.model_path
        if not model_path:
            self.ui.show_error(
                "No model_path configured. Set it in ~/.terminator/config.toml\n"
                "  or pass --model-path /path/to/model.gguf"
            )
            return False

        if not Path(model_path).is_file():
            self.ui.show_error(
                f"Model file not found: {model_path}\n"
                "  Download a GGUF model from https://huggingface.co"
            )
            return False

        try:
            proc = subprocess.Popen(
                [server_path, "-m", model_path, "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as e:
            self.ui.show_error(f"Failed to start llama-server: {e}")
            return False

        # Poll for readiness (up to 30s — GGUF loading can be slow)
        with self.ui.show_spinner("Starting llama-server..."):
            for _ in range(100):  # 100 * 0.3s = 30s
                if proc.poll() is not None:
                    self.ui.show_error(
                        "llama-server process exited immediately.\n"
                        "  Try running 'llama-server -m <model.gguf>' manually to see errors."
                    )
                    return False
                if self.client.check_health():
                    self._server_proc = proc
                    self.ui.console.print("[green]llama-server started.[/green]")
                    return True
                time.sleep(0.3)

        self.ui.show_error(
            "Timed out waiting for llama-server to start.\n"
            "  Try running 'llama-server -m <model.gguf>' manually."
        )
        return False

    def _handle_sigint(self, signum, frame):
        if self._llm_active:
            # Cancel LLM request — raise to break httpx
            raise KeyboardInterrupt
        elif self._child_running:
            # Forward to child — do nothing, let the process group handle it
            pass
        else:
            # At REPL prompt — ignore
            print()

    def _handle_meta_command(self, user_input: str) -> None:
        parts = user_input.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/raw":
            if not arg:
                self.ui.show_error("Usage: /raw <command>")
                return
            self._execute_raw(arg)
        elif cmd == "/model":
            self._handle_model_command(arg)
        elif cmd == "/history":
            self.ui.show_history(self.context.history.records)
        elif cmd == "/context":
            self.ui.show_context(self.context.build_prompt_context())
        elif cmd == "/clear":
            self.context.history.clear()
            self.ui.console.print("[dim]History cleared.[/dim]")
        elif cmd == "/help":
            self.ui.show_help()
        else:
            self.ui.show_error(f"Unknown command: {cmd}. Type /help for available commands.")

    def _handle_model_command(self, arg: str) -> None:
        if not arg:
            self.ui.console.print(f"Current model: [cyan]{self.client.model}[/cyan]")
            models_dir = self.config.model.models_dir
            if models_dir and Path(models_dir).is_dir():
                gguf_files = sorted(Path(models_dir).glob("*.gguf"))
                if gguf_files:
                    self.ui.console.print("Available models:")
                    for f in gguf_files:
                        self.ui.console.print(f"  {f.stem}")
                else:
                    self.ui.console.print(f"[dim]No .gguf files in {models_dir}[/dim]")
            elif models_dir:
                self.ui.console.print(f"[dim]models_dir not found: {models_dir}[/dim]")
            return

        # Find the matching GGUF file
        models_dir = self.config.model.models_dir
        if not models_dir or not Path(models_dir).is_dir():
            self.ui.show_error(
                "No models_dir configured. Set it in ~/.terminator/config.toml\n"
                "  to enable model switching."
            )
            return

        gguf_path = self._find_gguf(arg, Path(models_dir))
        if not gguf_path:
            gguf_files = sorted(Path(models_dir).glob("*.gguf"))
            self.ui.show_error(f"No GGUF file matching '{arg}' in {models_dir}")
            if gguf_files:
                self.ui.console.print("Available:")
                for f in gguf_files:
                    self.ui.console.print(f"  {f.stem}")
            return

        if not self._server_proc:
            self.ui.show_error(
                "Server was not auto-started — cannot restart it.\n"
                "  Restart the server manually with the new model."
            )
            return

        # Kill and restart with new model
        self.ui.console.print(f"[cyan]Switching to {gguf_path.stem}...[/cyan]")
        self._server_proc.terminate()
        try:
            self._server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._server_proc.kill()

        self.config.model.model_path = str(gguf_path)
        self._server_proc = None
        self.client.close()

        # Extract model name for prompt overrides
        self.client = LlamaCppClient(
            base_url=self.config.model.server_url,
            model=gguf_path.stem,
            timeout=self.config.model.timeout,
        )

        if self._try_start_server():
            self.ui.console.print(f"Switched to model: [cyan]{gguf_path.stem}[/cyan]")
        else:
            self.ui.show_error("Failed to restart server with new model.")

    def _find_gguf(self, name: str, models_dir: Path) -> Path | None:
        """Find a GGUF file matching the given name."""
        # Exact filename match (with or without .gguf extension)
        exact = models_dir / name
        if exact.is_file():
            return exact
        exact_gguf = models_dir / f"{name}.gguf"
        if exact_gguf.is_file():
            return exact_gguf

        # Normalize: replace : with -, lowercase, and search
        normalized = name.lower().replace(":", "-")
        for gguf_file in models_dir.glob("*.gguf"):
            if gguf_file.stem.lower().startswith(normalized):
                return gguf_file

        return None

    def _execute_raw(self, command: str) -> None:
        """Execute a command directly, bypassing the LLM."""
        # Check for builtins
        handled, message = self.executor.handle_builtin(command, self.context.cwd)
        if handled:
            if message and not message.startswith("cd:") and not message.startswith("alias"):
                self.context.update_cwd(message)
            if message:
                print(message)
            return

        self._child_running = True
        try:
            if self.classifier.is_interactive(command):
                result = self.executor.execute_interactive(command, self.context.cwd)
            else:
                result = self.executor.execute(command, self.context.cwd)
            self.ui.show_result(result.exit_code, result.timed_out)
        finally:
            self._child_running = False

        # Detect cd in chained commands
        cd_target = self.executor.detect_cd_in_command(command)
        if cd_target:
            self.context.update_cwd(
                cd_target if Path(cd_target).is_absolute() else str(Path(self.context.cwd) / cd_target)
            )

    def _handle_natural_input(self, user_input: str) -> None:
        # Build prompt and messages
        prompt_ctx = self.context.build_prompt_context()
        system_prompt = build_system_prompt(prompt_ctx, model=self.client.model)
        messages = self.context.get_chat_messages(user_input)

        # Call LLM
        self._llm_active = True
        try:
            with self.ui.show_spinner():
                response = self.client.chat(messages, system_prompt)
        except KeyboardInterrupt:
            self.ui.console.print("\n[dim]Cancelled.[/dim]")
            return
        except httpx.ConnectError:
            self.ui.show_error("Lost connection to LLM server. Is it still running?")
            return
        except httpx.TimeoutException:
            self.ui.show_error("LLM request timed out. Try a simpler query or increase timeout.")
            return
        except ParseError as e:
            self.ui.show_error(f"Could not parse LLM response.\nRaw output: {e.raw_output}")
            return
        except httpx.HTTPStatusError as e:
            self.ui.show_error(f"LLM server returned an error: {e.response.status_code}")
            return
        finally:
            self._llm_active = False

        # Handle response
        if isinstance(response, ClarifyResponse):
            self.ui.show_clarification(response.question)
            return

        assert isinstance(response, CommandResponse)
        command = response.command

        # Check for builtins
        handled, message = self.executor.handle_builtin(command, self.context.cwd)
        if handled:
            self.ui.show_command(command, response.explanation, SafetyLevel.GREEN)
            if message and not message.startswith("cd:") and not message.startswith("alias"):
                self.context.update_cwd(message)
            if message:
                print(message)
            self.context.record_exchange(user_input, command, stdout=message)
            return

        # Classify safety
        level = self.classifier.classify(command)

        # Check if blocked
        if self.classifier.is_blocked(command):
            self.ui.show_blocked(command)
            return

        # Long command warning
        if len(command) > 500:
            self.ui.show_long_command_warning()
            level = SafetyLevel.RED  # Force review

        # Show command
        self.ui.show_command(command, response.explanation, level)

        # Confirm
        if not self.ui.confirm_execution(level):
            self.ui.console.print("[dim]Skipped.[/dim]")
            return

        # Execute
        self._child_running = True
        try:
            if self.classifier.is_interactive(command):
                result = self.executor.execute_interactive(command, self.context.cwd)
            else:
                result = self.executor.execute(command, self.context.cwd)
        finally:
            self._child_running = False

        self.ui.show_result(result.exit_code, result.timed_out)

        # Record in history
        self.context.record_exchange(
            user_input, command,
            stdout=result.stdout, stderr=result.stderr,
            exit_code=result.exit_code,
        )

        # Detect cd in executed commands
        cd_target = self.executor.detect_cd_in_command(command)
        if cd_target:
            resolved = cd_target if Path(cd_target).is_absolute() else str(Path(self.context.cwd) / cd_target)
            self.context.update_cwd(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="terminator",
        description="Terminator — translate natural language to shell commands",
    )
    parser.add_argument("--model", help="Model name (used for prompt overrides)")
    parser.add_argument("--url", help="LLM server URL (default: http://localhost:8080)")
    parser.add_argument("--model-path", help="Path to GGUF model file for auto-start")
    parser.add_argument("--config", type=Path, help="Path to config file")
    parser.add_argument("--init", action="store_true", help="Generate default config file")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.init:
        path = write_default_config()
        print(f"Config written to {path}")
        return

    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.url:
        overrides["url"] = args.url
    if args.model_path:
        overrides["model_path"] = args.model_path

    config = AppConfig.load(
        config_path=args.config,
        cli_overrides=overrides if overrides else None,
    )

    app = App(config)
    app.run()


if __name__ == "__main__":
    main()
