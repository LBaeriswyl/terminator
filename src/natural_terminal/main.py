"""Main entry point — REPL loop, argument parsing, and signal handling."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path

import httpx

from natural_terminal import __version__
from natural_terminal.config import AppConfig, write_default_config, CONFIG_FILE
from natural_terminal.context import ContextManager
from natural_terminal.executor import CommandExecutor, SafetyClassifier, SafetyLevel
from natural_terminal.llm import OllamaClient, CommandResponse, ClarifyResponse, ParseError
from natural_terminal.prompt import build_system_prompt
from natural_terminal.ui import TerminalUI


class App:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = OllamaClient(
            base_url=config.model.ollama_url,
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

    def run(self) -> None:
        # Startup checks
        if not self._startup_checks():
            return

        # Background warmup
        warmup_thread = threading.Thread(target=self.client.warmup, daemon=True)
        warmup_thread.start()

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
            self.ui.show_error(
                "Cannot connect to Ollama.\n"
                f"  URL: {self.client.base_url}\n"
                "  Is Ollama running? Try: ollama serve"
            )
            return False

        if not self.client.check_model():
            self.ui.console.print(
                f"[yellow]Model '{self.client.model}' not found.[/yellow]"
            )
            available = self.client.list_models()
            if available:
                self.ui.console.print(f"Available models: {', '.join(available)}")

            try:
                answer = input(f"Pull '{self.client.model}'? [y/N] ")
            except (EOFError, KeyboardInterrupt):
                return False

            if answer.strip().lower() in ("y", "yes"):
                self._pull_model(self.client.model)
            else:
                return False

        return True

    def _pull_model(self, model: str) -> None:
        self.ui.console.print(f"[cyan]Pulling {model}...[/cyan]")
        try:
            with self.client.pull_model(model) as resp:
                for line in resp.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            status = data.get("status", "")
                            if "total" in data and "completed" in data:
                                pct = int(data["completed"] / data["total"] * 100)
                                print(f"\r  {status}: {pct}%", end="", flush=True)
                            else:
                                print(f"\r  {status}", end="", flush=True)
                        except json.JSONDecodeError:
                            pass
                print()
            self.ui.console.print(f"[green]Model {model} pulled successfully.[/green]")
        except httpx.HTTPError as e:
            self.ui.show_error(f"Failed to pull model: {e}")

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
            if not arg:
                self.ui.console.print(f"Current model: [cyan]{self.client.model}[/cyan]")
                available = self.client.list_models()
                if available:
                    self.ui.console.print(f"Available: {', '.join(available)}")
                return
            self.client.model = arg
            self.ui.console.print(f"Switched to model: [cyan]{arg}[/cyan]")
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
        system_prompt = build_system_prompt(prompt_ctx)
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
            self.ui.show_error("Lost connection to Ollama. Is it still running?")
            return
        except httpx.TimeoutException:
            self.ui.show_error("LLM request timed out. Try a simpler query or increase timeout.")
            return
        except ParseError as e:
            self.ui.show_error(f"Could not parse LLM response.\nRaw output: {e.raw_output}")
            return
        except httpx.HTTPStatusError as e:
            self.ui.show_error(f"Ollama returned an error: {e.response.status_code}")
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
        prog="nt",
        description="Natural Language Terminal — translate natural language to shell commands",
    )
    parser.add_argument("--model", help="Ollama model to use")
    parser.add_argument("--url", help="Ollama server URL")
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

    config = AppConfig.load(
        config_path=args.config,
        cli_overrides=overrides if overrides else None,
    )

    app = App(config)
    app.run()


if __name__ == "__main__":
    main()
