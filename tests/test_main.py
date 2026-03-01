"""Tests for App._try_start_server() and model switching."""

from unittest.mock import MagicMock, patch

import pytest

from terminator.main import App
from terminator.config import AppConfig


def make_app(server_url="http://localhost:8080", model_path="/tmp/model.gguf"):
    """Create an App with mocked UI and client."""
    config = AppConfig.load()
    config.model.server_url = server_url
    config.model.model_path = model_path
    app = App(config)
    app.ui = MagicMock()
    app.ui.show_spinner.return_value.__enter__ = MagicMock()
    app.ui.show_spinner.return_value.__exit__ = MagicMock(return_value=False)
    app.client = MagicMock()
    app.client.base_url = server_url
    return app


class TestTryStartServer:
    def test_remote_url_does_not_try_to_start(self):
        app = make_app("http://remote-server:8080")
        result = app._try_start_server()
        assert result is False
        app.ui.show_error.assert_called_once()
        assert "remote" not in app.ui.show_error.call_args[0][0].lower() or \
               "cannot reach" in app.ui.show_error.call_args[0][0].lower()

    def test_binary_not_found(self):
        app = make_app()
        with patch("terminator.main.shutil.which", return_value=None):
            result = app._try_start_server()
        assert result is False
        app.ui.show_error.assert_called_once()
        assert "not installed" in app.ui.show_error.call_args[0][0].lower()

    def test_model_path_not_configured(self):
        app = make_app(model_path="")
        with patch("terminator.main.shutil.which", return_value="/usr/bin/llama-server"):
            result = app._try_start_server()
        assert result is False
        app.ui.show_error.assert_called_once()
        assert "model_path" in app.ui.show_error.call_args[0][0].lower()

    def test_model_file_not_found(self):
        app = make_app(model_path="/nonexistent/model.gguf")
        with patch("terminator.main.shutil.which", return_value="/usr/bin/llama-server"):
            result = app._try_start_server()
        assert result is False
        app.ui.show_error.assert_called_once()
        assert "not found" in app.ui.show_error.call_args[0][0].lower()

    def test_successful_start(self, tmp_path):
        model_file = tmp_path / "model.gguf"
        model_file.write_text("fake")
        app = make_app(model_path=str(model_file))
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        # Health fails twice, then succeeds
        app.client.check_health.side_effect = [False, False, True]

        with (
            patch("terminator.main.shutil.which", return_value="/usr/bin/llama-server"),
            patch("terminator.main.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch("terminator.main.time.sleep"),
        ):
            result = app._try_start_server()

        assert result is True
        assert app._server_proc is mock_proc
        # Verify Popen was called with llama-server and -m flag
        popen_args = mock_popen.call_args[0][0]
        assert popen_args[0] == "/usr/bin/llama-server"
        assert "-m" in popen_args
        assert str(model_file) in popen_args
        app.ui.console.print.assert_called_once()
        assert "started" in app.ui.console.print.call_args[0][0].lower()

    def test_start_timeout(self, tmp_path):
        model_file = tmp_path / "model.gguf"
        model_file.write_text("fake")
        app = make_app(model_path=str(model_file))
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        app.client.check_health.return_value = False

        with (
            patch("terminator.main.shutil.which", return_value="/usr/bin/llama-server"),
            patch("terminator.main.subprocess.Popen", return_value=mock_proc),
            patch("terminator.main.time.sleep"),
        ):
            result = app._try_start_server()

        assert result is False
        # Last call should be the timeout error
        assert "timed out" in app.ui.show_error.call_args[0][0].lower()

    def test_process_exits_immediately(self, tmp_path):
        model_file = tmp_path / "model.gguf"
        model_file.write_text("fake")
        app = make_app(model_path=str(model_file))
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # exited
        app.client.check_health.return_value = False

        with (
            patch("terminator.main.shutil.which", return_value="/usr/bin/llama-server"),
            patch("terminator.main.subprocess.Popen", return_value=mock_proc),
            patch("terminator.main.time.sleep"),
        ):
            result = app._try_start_server()

        assert result is False
        assert "exited immediately" in app.ui.show_error.call_args[0][0].lower()

    def test_popen_raises_oserror(self, tmp_path):
        model_file = tmp_path / "model.gguf"
        model_file.write_text("fake")
        app = make_app(model_path=str(model_file))
        with (
            patch("terminator.main.shutil.which", return_value="/usr/bin/llama-server"),
            patch("terminator.main.subprocess.Popen", side_effect=OSError("Permission denied")),
        ):
            result = app._try_start_server()

        assert result is False
        assert "failed to start" in app.ui.show_error.call_args[0][0].lower()

    def test_localhost_127_is_local(self):
        app = make_app("http://127.0.0.1:8080")
        with patch("terminator.main.shutil.which", return_value=None):
            result = app._try_start_server()
        assert result is False
        # Should try to find binary (not reject as remote)
        assert "not installed" in app.ui.show_error.call_args[0][0].lower()
