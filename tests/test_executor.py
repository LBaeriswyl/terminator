"""Tests for safety classification and command execution."""

import os
import pytest

from terminator.executor import (
    SafetyClassifier,
    SafetyLevel,
    CommandExecutor,
)


class TestSafetyClassifier:
    @pytest.fixture
    def classifier(self):
        return SafetyClassifier(blocked_patterns=["rm -rf /", ":(){ :|:& };:"])

    def test_green_commands(self, classifier):
        assert classifier.classify("ls") == SafetyLevel.GREEN
        assert classifier.classify("ls -la") == SafetyLevel.GREEN
        assert classifier.classify("pwd") == SafetyLevel.GREEN
        assert classifier.classify("cat file.txt") == SafetyLevel.GREEN
        assert classifier.classify("echo hello") == SafetyLevel.GREEN
        assert classifier.classify("grep pattern file") == SafetyLevel.GREEN
        assert classifier.classify("find . -name '*.py'") == SafetyLevel.GREEN
        assert classifier.classify("git status") == SafetyLevel.GREEN

    def test_yellow_commands(self, classifier):
        assert classifier.classify("cp a b") == SafetyLevel.YELLOW
        assert classifier.classify("mv a b") == SafetyLevel.YELLOW
        assert classifier.classify("mkdir foo") == SafetyLevel.YELLOW
        assert classifier.classify("touch file") == SafetyLevel.YELLOW
        assert classifier.classify("chmod 755 file") == SafetyLevel.YELLOW

    def test_red_commands(self, classifier):
        assert classifier.classify("rm file.txt") == SafetyLevel.RED
        assert classifier.classify("rm -rf dir/") == SafetyLevel.RED
        assert classifier.classify("kill 1234") == SafetyLevel.RED
        assert classifier.classify("killall python") == SafetyLevel.RED
        assert classifier.classify("dd if=/dev/zero of=/dev/sda") == SafetyLevel.RED
        assert classifier.classify("shutdown -h now") == SafetyLevel.RED

    def test_sudo_is_red(self, classifier):
        assert classifier.classify("sudo ls") == SafetyLevel.RED
        assert classifier.classify("sudo apt install foo") == SafetyLevel.RED

    def test_piped_commands_minimum_yellow(self, classifier):
        assert classifier.classify("ls | grep foo") == SafetyLevel.YELLOW
        assert classifier.classify("cat file | wc -l") == SafetyLevel.YELLOW

    def test_chained_commands_minimum_yellow(self, classifier):
        assert classifier.classify("cd /tmp && ls") == SafetyLevel.YELLOW
        assert classifier.classify("echo a; echo b") == SafetyLevel.YELLOW

    def test_redirect_is_yellow(self, classifier):
        assert classifier.classify("echo hello > file.txt") == SafetyLevel.YELLOW
        assert classifier.classify("cat a >> b") == SafetyLevel.YELLOW

    def test_blocked_patterns(self, classifier):
        assert classifier.is_blocked("rm -rf /")
        assert classifier.is_blocked("sudo rm -rf /home")  # contains "rm -rf /"? No actually
        assert not classifier.is_blocked("rm file.txt")
        assert classifier.is_blocked(":(){ :|:& };:")

    def test_interactive_detection(self, classifier):
        assert classifier.is_interactive("vim file.txt")
        assert classifier.is_interactive("python3")
        assert classifier.is_interactive("ssh user@host")
        assert not classifier.is_interactive("echo hello")
        assert not classifier.is_interactive("ls -la")


class TestFirstTokenExtraction:
    @pytest.fixture
    def classifier(self):
        return SafetyClassifier()

    def test_simple_command(self, classifier):
        assert classifier._extract_first_token("ls -la") == "ls"

    def test_sudo_skipped(self, classifier):
        assert classifier._extract_first_token("sudo rm -rf /") == "rm"

    def test_env_skipped(self, classifier):
        assert classifier._extract_first_token("env FOO=bar python script.py") == "python"

    def test_nohup_skipped(self, classifier):
        assert classifier._extract_first_token("nohup python server.py") == "python"

    def test_time_skipped(self, classifier):
        assert classifier._extract_first_token("time ls -la") == "ls"

    def test_multiple_prefixes(self, classifier):
        assert classifier._extract_first_token("sudo nice python script.py") == "python"


class TestCommandExecutor:
    @pytest.fixture
    def executor(self):
        return CommandExecutor(default_timeout=10)

    def test_execute_echo(self, executor):
        result = executor.execute("echo hello", os.getcwd())
        assert "hello" in result.stdout
        assert result.exit_code == 0
        assert result.timed_out is False

    def test_execute_pwd(self, executor):
        result = executor.execute("pwd", os.getcwd())
        assert result.exit_code == 0
        assert result.stdout.strip() == os.getcwd()

    def test_execute_failure(self, executor):
        result = executor.execute("false", os.getcwd())
        assert result.exit_code != 0

    def test_timeout(self, executor):
        result = executor.execute("sleep 30", os.getcwd(), timeout=1)
        assert result.timed_out is True

    def test_cd_handling(self, executor, tmp_path):
        handled, message = executor.handle_builtin("cd /tmp", str(tmp_path))
        assert handled is True
        assert message == "/private/tmp" or message == "/tmp"
        # Restore cwd
        os.chdir(os.path.dirname(__file__))

    def test_cd_home(self, executor):
        handled, message = executor.handle_builtin("cd ~", "/tmp")
        assert handled is True
        assert message == os.path.expanduser("~")
        os.chdir(os.path.dirname(__file__))

    def test_cd_nonexistent(self, executor):
        handled, message = executor.handle_builtin("cd /nonexistent_dir_xyz", "/tmp")
        assert handled is True
        assert "no such directory" in message

    def test_export_handling(self, executor):
        handled, message = executor.handle_builtin("export MY_TEST_VAR=hello", "/tmp")
        assert handled is True
        assert os.environ.get("MY_TEST_VAR") == "hello"
        # Cleanup
        del os.environ["MY_TEST_VAR"]

    def test_alias_rejected(self, executor):
        handled, message = executor.handle_builtin("alias ll='ls -la'", "/tmp")
        assert handled is True
        assert "not supported" in message

    def test_detect_cd_in_command(self, executor):
        assert executor.detect_cd_in_command("cd /tmp && ls") == "/tmp"
        assert executor.detect_cd_in_command("cd ~/projects && git status") is not None
        assert executor.detect_cd_in_command("ls -la") is None
