"""Tests for App._try_start_ollama()."""

from unittest.mock import MagicMock, patch

import pytest

from natural_terminal.main import App
from natural_terminal.config import AppConfig


def make_app(ollama_url="http://localhost:11434"):
    """Create an App with mocked UI and client."""
    config = AppConfig.load()
    config.model.ollama_url = ollama_url
    app = App(config)
    app.ui = MagicMock()
    app.ui.show_spinner.return_value.__enter__ = MagicMock()
    app.ui.show_spinner.return_value.__exit__ = MagicMock(return_value=False)
    app.client = MagicMock()
    app.client.base_url = ollama_url
    return app


class TestTryStartOllama:
    def test_remote_url_does_not_try_to_start(self):
        app = make_app("http://remote-server:11434")
        result = app._try_start_ollama()
        assert result is False
        app.ui.show_error.assert_called_once()
        assert "remote" in app.ui.show_error.call_args[0][0].lower()

    def test_binary_not_found(self):
        app = make_app()
        with patch("natural_terminal.main.shutil.which", return_value=None):
            result = app._try_start_ollama()
        assert result is False
        app.ui.show_error.assert_called_once()
        assert "not installed" in app.ui.show_error.call_args[0][0].lower()

    def test_successful_start(self):
        app = make_app()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        # Health fails twice, then succeeds
        app.client.check_health.side_effect = [False, False, True]

        with (
            patch("natural_terminal.main.shutil.which", return_value="/usr/bin/ollama"),
            patch("natural_terminal.main.subprocess.Popen", return_value=mock_proc),
            patch("natural_terminal.main.time.sleep"),
        ):
            result = app._try_start_ollama()

        assert result is True
        app.ui.console.print.assert_called_once()
        assert "started" in app.ui.console.print.call_args[0][0].lower()

    def test_start_timeout(self):
        app = make_app()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        app.client.check_health.return_value = False

        with (
            patch("natural_terminal.main.shutil.which", return_value="/usr/bin/ollama"),
            patch("natural_terminal.main.subprocess.Popen", return_value=mock_proc),
            patch("natural_terminal.main.time.sleep"),
        ):
            result = app._try_start_ollama()

        assert result is False
        # Last call should be the timeout error
        assert "timed out" in app.ui.show_error.call_args[0][0].lower()

    def test_process_exits_immediately(self):
        app = make_app()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # exited
        app.client.check_health.return_value = False

        with (
            patch("natural_terminal.main.shutil.which", return_value="/usr/bin/ollama"),
            patch("natural_terminal.main.subprocess.Popen", return_value=mock_proc),
            patch("natural_terminal.main.time.sleep"),
        ):
            result = app._try_start_ollama()

        assert result is False
        assert "exited immediately" in app.ui.show_error.call_args[0][0].lower()

    def test_popen_raises_oserror(self):
        app = make_app()
        with (
            patch("natural_terminal.main.shutil.which", return_value="/usr/bin/ollama"),
            patch("natural_terminal.main.subprocess.Popen", side_effect=OSError("Permission denied")),
        ):
            result = app._try_start_ollama()

        assert result is False
        assert "failed to start" in app.ui.show_error.call_args[0][0].lower()

    def test_localhost_127_is_local(self):
        app = make_app("http://127.0.0.1:11434")
        with patch("natural_terminal.main.shutil.which", return_value=None):
            result = app._try_start_ollama()
        assert result is False
        # Should try to find binary (not reject as remote)
        assert "not installed" in app.ui.show_error.call_args[0][0].lower()
