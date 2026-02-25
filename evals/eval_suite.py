#!/usr/bin/env python3
"""LLM evaluation suite — measures how well a model translates natural language to shell commands."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add project root to path so we can import natural_terminal
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from natural_terminal.llm import OllamaClient, CommandResponse, ClarifyResponse, ParseError
from natural_terminal.prompt import build_system_prompt, MODEL_OVERRIDES
from natural_terminal.context import ExchangeRecord, ConversationHistory

CASES_FILE = Path(__file__).parent / "cases.json"
RESULTS_DIR = Path(__file__).parent / "results"

# Fixed synthetic context for reproducibility
SYNTHETIC_CONTEXT = {
    "os_type": "Darwin",
    "shell_type": "zsh",
    "cwd": "/home/user/projects/myapp",
    "dir_tree": (
        "src/\n"
        "  main.py\n"
        "  utils.py\n"
        "  config.py\n"
        "tests/\n"
        "  test_main.py\n"
        "  test_utils.py\n"
        "README.md\n"
        "pyproject.toml\n"
        "app.log\n"
        "data.csv\n"
        "data.txt\n"
        "notes.md\n"
        "script.sh\n"
        "foo.txt"
    ),
    "username": "user",
}


@dataclass
class CaseResult:
    case_id: str
    category: str
    difficulty: str
    input_text: str
    expected_commands: list[str]
    expected_type: str
    actual_command: str | None = None
    actual_type: str | None = None
    score: float = 0.0
    error: str | None = None


@dataclass
class EvalReport:
    model: str
    prompt_version: str
    timestamp: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def overall_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results) * 100

    def by_category(self) -> dict[str, list[CaseResult]]:
        categories: dict[str, list[CaseResult]] = {}
        for r in self.results:
            categories.setdefault(r.category, []).append(r)
        return categories

    def format(self) -> str:
        lines = [
            f"Model: {self.model} | Prompt: {self.prompt_version} | Date: {self.timestamp}",
            "=" * 60,
            f"{'Category':<25} {'Pass':>5} {'Part':>5} {'Fail':>5} {'Score':>7}",
            "-" * 60,
        ]

        for cat, results in sorted(self.by_category().items()):
            n = len(results)
            passed = sum(1 for r in results if r.score == 1.0)
            partial = sum(1 for r in results if 0 < r.score < 1.0)
            failed = sum(1 for r in results if r.score == 0.0)
            score = sum(r.score for r in results) / n * 100
            lines.append(f"{cat:<25} {passed:>4}/{n} {partial:>4}/{n} {failed:>4}/{n} {score:>6.1f}%")

        lines.append("-" * 60)

        passed_total = sum(1 for r in self.results if r.score == 1.0)
        partial_total = sum(1 for r in self.results if 0 < r.score < 1.0)
        failed_total = sum(1 for r in self.results if r.score == 0.0)
        lines.append(
            f"{'Overall':<25} {passed_total:>4}/{self.total} "
            f"{partial_total:>4}/{self.total} {failed_total:>4}/{self.total} "
            f"{self.overall_score:>6.1f}%"
        )

        # Failed cases
        failures = [r for r in self.results if r.score < 1.0]
        if failures:
            lines.append("")
            lines.append("Issues:")
            for r in failures:
                expected = r.expected_commands[0] if r.expected_commands else f"(type: {r.expected_type})"
                actual = r.actual_command or r.error or "(no output)"
                label = "partial" if r.score > 0 else "FAIL"
                lines.append(f"  [{label}] {r.case_id}: expected \"{expected}\", got \"{actual}\"")

        return "\n".join(lines)


def normalize_command(cmd: str) -> tuple[str, set[str], list[str]]:
    """Split command into (binary, flags, arguments) for comparison."""
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = cmd.split()

    if not tokens:
        return ("", set(), [])

    binary = tokens[0]
    flags: set[str] = set()
    args: list[str] = []

    for token in tokens[1:]:
        if token.startswith("-"):
            # Expand combined short flags like -la -> {-l, -a}
            if len(token) > 2 and not token.startswith("--") and not any(c.isdigit() for c in token[1:]):
                for ch in token[1:]:
                    flags.add(f"-{ch}")
            else:
                flags.add(token)
        else:
            args.append(token)

    return (binary, flags, args)


def score_case(expected_commands: list[str], expected_type: str, response) -> tuple[float, str | None, str | None]:
    """Score a response against expectations. Returns (score, actual_command, actual_type)."""
    if isinstance(response, ParseError):
        return (0.0, None, "parse_error")

    if isinstance(response, ClarifyResponse):
        if expected_type == "clarify":
            return (1.0, response.question, "clarify")
        return (0.0, response.question, "clarify")

    if isinstance(response, CommandResponse):
        if expected_type == "clarify":
            return (0.0, response.command, "command")

        actual = response.command

        # Exact match
        if actual in expected_commands:
            return (1.0, actual, "command")

        # Normalized match (flag reordering)
        actual_norm = normalize_command(actual)
        for exp in expected_commands:
            exp_norm = normalize_command(exp)
            if actual_norm == exp_norm:
                return (1.0, actual, "command")

        # Semantic match — same binary and flags but in different order
        for exp in expected_commands:
            exp_norm = normalize_command(exp)
            if actual_norm[0] == exp_norm[0] and actual_norm[1] == exp_norm[1]:
                # Same binary and flags, different args — partial
                return (0.5, actual, "command")

        # Partial match — right base command
        for exp in expected_commands:
            exp_binary = normalize_command(exp)[0]
            if actual_norm[0] == exp_binary:
                return (0.5, actual, "command")

        return (0.0, actual, "command")

    return (0.0, None, None)


def run_eval(model: str, cases_file: Path, ollama_url: str = "http://localhost:11434") -> EvalReport:
    """Run all cases against Ollama and return scored report."""
    with open(cases_file) as f:
        cases = json.load(f)

    client = OllamaClient(base_url=ollama_url, model=model, timeout=60)

    # Health check
    if not client.check_health():
        print(f"Error: Cannot connect to Ollama at {ollama_url}", file=sys.stderr)
        sys.exit(1)

    if not client.check_model(model):
        print(f"Error: Model '{model}' not found. Available: {client.list_models()}", file=sys.stderr)
        sys.exit(1)

    system_prompt = build_system_prompt(SYNTHETIC_CONTEXT, model=model)
    report = EvalReport(
        model=model,
        prompt_version="default" if model not in MODEL_OVERRIDES else model,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    print(f"Running {len(cases)} eval cases against {model}...")

    for i, case in enumerate(cases):
        case_id = case["id"]
        print(f"  [{i+1}/{len(cases)}] {case_id}...", end=" ", flush=True)

        # Build messages with optional synthetic history
        history = ConversationHistory()
        if "history" in case:
            for h in case["history"]:
                history.add(ExchangeRecord(
                    user_input=h["user_input"],
                    generated_command=h["generated_command"],
                    stdout=h.get("stdout", ""),
                    stderr=h.get("stderr", ""),
                    exit_code=h.get("exit_code", 0),
                ))

        messages = history.to_messages()
        messages.append({"role": "user", "content": case["input"]})

        try:
            response = client.chat(messages, system_prompt)
            score, actual_cmd, actual_type = score_case(
                case["expected_commands"], case["expected_type"], response
            )
            result = CaseResult(
                case_id=case_id,
                category=case["category"],
                difficulty=case["difficulty"],
                input_text=case["input"],
                expected_commands=case["expected_commands"],
                expected_type=case["expected_type"],
                actual_command=actual_cmd,
                actual_type=actual_type,
                score=score,
            )
            status = "PASS" if score == 1.0 else ("PARTIAL" if score > 0 else "FAIL")
            print(f"{status} ({actual_cmd})")

        except ParseError as e:
            result = CaseResult(
                case_id=case_id,
                category=case["category"],
                difficulty=case["difficulty"],
                input_text=case["input"],
                expected_commands=case["expected_commands"],
                expected_type=case["expected_type"],
                score=0.0,
                error=f"ParseError: {e.raw_output[:100]}",
            )
            print(f"FAIL (parse error)")

        except Exception as e:
            result = CaseResult(
                case_id=case_id,
                category=case["category"],
                difficulty=case["difficulty"],
                input_text=case["input"],
                expected_commands=case["expected_commands"],
                expected_type=case["expected_type"],
                score=0.0,
                error=str(e)[:100],
            )
            print(f"FAIL ({type(e).__name__}: {e})")

        report.results.append(result)

    client.close()
    return report


def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluation suite")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model to evaluate")
    parser.add_argument("--cases", type=Path, default=CASES_FILE, help="Path to cases JSON")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--save", action="store_true", help="Save results to evals/results/")
    args = parser.parse_args()

    report = run_eval(args.model, args.cases, args.url)

    print()
    print(report.format())

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = args.model.replace(":", "_").replace("/", "_")
        result_file = RESULTS_DIR / f"{model_slug}_{timestamp}.json"

        data = {
            "model": report.model,
            "prompt_version": report.prompt_version,
            "timestamp": report.timestamp,
            "overall_score": report.overall_score,
            "results": [
                {
                    "case_id": r.case_id,
                    "category": r.category,
                    "difficulty": r.difficulty,
                    "input": r.input_text,
                    "expected_commands": r.expected_commands,
                    "expected_type": r.expected_type,
                    "actual_command": r.actual_command,
                    "actual_type": r.actual_type,
                    "score": r.score,
                    "error": r.error,
                }
                for r in report.results
            ],
        }

        with open(result_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {result_file}")


if __name__ == "__main__":
    main()
