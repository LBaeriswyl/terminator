"""Terminal UI — Rich console + prompt_toolkit integration."""

from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.text import Text

from terminator.config import HISTORY_FILE, ensure_config_dir
from terminator.executor import SafetyLevel

META_COMMANDS = ["/raw", "/model", "/history", "/context", "/clear", "/help"]

SAFETY_COLORS = {
    SafetyLevel.GREEN: "green",
    SafetyLevel.YELLOW: "yellow",
    SafetyLevel.RED: "red",
}

SAFETY_LABELS = {
    SafetyLevel.GREEN: "SAFE",
    SafetyLevel.YELLOW: "CAUTION",
    SafetyLevel.RED: "DANGER",
}


class TerminalUI:
    def __init__(self, auto_execute_safe: bool = True):
        self.console = Console()
        self.auto_execute_safe = auto_execute_safe
        ensure_config_dir()
        self._session = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            completer=WordCompleter(META_COMMANDS, sentence=True),
        )

    def get_input(self) -> str:
        return self._session.prompt("terminator> ")

    def show_command(self, command: str, explanation: str, level: SafetyLevel) -> None:
        color = SAFETY_COLORS[level]
        label = SAFETY_LABELS[level]

        content = Text()
        content.append(f"$ {command}", style=f"bold {color}")
        if explanation:
            content.append(f"\n{explanation}", style="dim")

        self.console.print(Panel(
            content,
            title=f"[{color}]{label}[/{color}]",
            border_style=color,
            padding=(0, 1),
        ))

    def confirm_execution(self, level: SafetyLevel) -> bool:
        if level == SafetyLevel.GREEN and self.auto_execute_safe:
            return True

        if level == SafetyLevel.RED:
            try:
                answer = input('Type "yes" to execute: ')
                return answer.strip() == "yes"
            except (EOFError, KeyboardInterrupt):
                return False

        # YELLOW
        try:
            answer = input("Execute? [y/N] ")
            return answer.strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def show_blocked(self, command: str) -> None:
        self.console.print(
            Panel(
                f"[red]BLOCKED:[/red] {command}\nThis command matches a blocked pattern.",
                border_style="red",
            )
        )

    def show_clarification(self, question: str) -> None:
        self.console.print(f"[cyan]?[/cyan] {question}")

    def show_error(self, message: str) -> None:
        self.console.print(f"[red]Error:[/red] {message}")

    def show_result(self, exit_code: int, timed_out: bool = False) -> None:
        if timed_out:
            self.console.print("[yellow]Command timed out[/yellow]")
        elif exit_code != 0:
            self.console.print(f"[red]Exit code: {exit_code}[/red]")

    def show_spinner(self, message: str = "Thinking...") -> Status:
        return self.console.status(f"[cyan]{message}[/cyan]", spinner="dots")

    def show_welcome(self, model: str, url: str) -> None:
        self.console.print(
            Panel(
                f"[bold]Terminator[/bold]\n"
                f"Model: [cyan]{model}[/cyan] | Server: [cyan]{url}[/cyan]\n"
                f'Type naturally or [dim]/help[/dim] for commands. [dim]Ctrl+D[/dim] to exit.',
                border_style="blue",
                padding=(0, 1),
            )
        )

    def show_help(self) -> None:
        help_text = (
            "[bold]Meta-commands:[/bold]\n"
            "  /raw <cmd>    Execute a shell command directly\n"
            "  /model [name] Show or switch the model (requires models_dir config)\n"
            "  /history      Show conversation history\n"
            "  /context      Show current context sent to model\n"
            "  /clear        Clear conversation history\n"
            "  /help         Show this help message\n"
            "\n"
            "[bold]Controls:[/bold]\n"
            "  Ctrl+C        Cancel current operation\n"
            "  Ctrl+D        Exit"
        )
        self.console.print(Panel(help_text, title="Help", border_style="blue", padding=(0, 1)))

    def show_history(self, records) -> None:
        if not records:
            self.console.print("[dim]No history yet.[/dim]")
            return
        for i, rec in enumerate(records, 1):
            self.console.print(f"[dim]{i}.[/dim] [cyan]{rec.user_input}[/cyan]")
            self.console.print(f"   $ {rec.generated_command} [dim](exit {rec.exit_code})[/dim]")

    def show_context(self, context: dict) -> None:
        self.console.print(Panel(
            f"OS: {context['os_type']}\n"
            f"Shell: {context['shell_type']}\n"
            f"CWD: {context['cwd']}\n"
            f"User: {context['username']}\n"
            f"Directory tree:\n{context['dir_tree']}",
            title="Current Context",
            border_style="blue",
            padding=(0, 1),
        ))

    def show_long_command_warning(self) -> None:
        self.console.print("[yellow]Warning: This is a long command. Please review carefully.[/yellow]")
