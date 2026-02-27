"""Tests for llama-server client and response parsing."""

import json
import pytest
import httpx
import respx

from natural_terminal.llm import (
    LlamaCppClient,
    CommandResponse,
    ClarifyResponse,
    ParseError,
)


@pytest.fixture
def client():
    c = LlamaCppClient(base_url="http://test:8080", model="test-model", timeout=5)
    yield c
    c.close()


class TestParseResponse:
    def test_valid_command_json(self, client):
        content = '{"type": "command", "command": "ls -la", "explanation": "list files"}'
        result = client._parse_response(content)
        assert isinstance(result, CommandResponse)
        assert result.command == "ls -la"
        assert result.explanation == "list files"

    def test_valid_clarify_json(self, client):
        content = '{"type": "clarify", "question": "Which file?"}'
        result = client._parse_response(content)
        assert isinstance(result, ClarifyResponse)
        assert result.question == "Which file?"

    def test_code_fenced_json(self, client):
        content = '```json\n{"type": "command", "command": "pwd", "explanation": "print cwd"}\n```'
        result = client._parse_response(content)
        assert isinstance(result, CommandResponse)
        assert result.command == "pwd"

    def test_json_in_text(self, client):
        content = 'Here is the command: {"type": "command", "command": "echo hi", "explanation": "say hi"}'
        result = client._parse_response(content)
        assert isinstance(result, CommandResponse)
        assert result.command == "echo hi"

    def test_bare_command_heuristic(self, client):
        content = "ls -la"
        result = client._parse_response(content)
        assert isinstance(result, CommandResponse)
        assert result.command == "ls -la"

    def test_malformed_output_raises(self, client):
        content = "This is not JSON at all\nAnd has multiple lines\nWith no structure"
        with pytest.raises(ParseError):
            client._parse_response(content)

    def test_empty_content_raises(self, client):
        with pytest.raises(ParseError):
            client._parse_response("")

    def test_json_missing_type_field(self, client):
        content = '{"command": "ls"}'
        # Missing "type" → _try_json_parse returns None, heuristic skips { prefix
        with pytest.raises(ParseError):
            client._parse_response(content)


class TestHealthCheck:
    @respx.mock
    def test_health_ok(self, client):
        respx.get("http://test:8080/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        assert client.check_health() is True

    @respx.mock
    def test_health_loading_model(self, client):
        respx.get("http://test:8080/health").mock(
            return_value=httpx.Response(200, json={"status": "loading model"})
        )
        assert client.check_health() is False

    @respx.mock
    def test_health_down(self, client):
        respx.get("http://test:8080/health").mock(side_effect=httpx.ConnectError("refused"))
        assert client.check_health() is False


class TestChat:
    @respx.mock
    def test_chat_returns_command(self, client):
        response_data = {
            "choices": [{
                "message": {
                    "content": '{"type": "command", "command": "ls", "explanation": "list"}'
                }
            }]
        }
        respx.post("http://test:8080/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=response_data)
        )
        result = client.chat(
            messages=[{"role": "user", "content": "list files"}],
            system_prompt="You are a shell translator.",
        )
        assert isinstance(result, CommandResponse)
        assert result.command == "ls"

    @respx.mock
    def test_chat_returns_clarify(self, client):
        response_data = {
            "choices": [{
                "message": {
                    "content": '{"type": "clarify", "question": "Which dir?"}'
                }
            }]
        }
        respx.post("http://test:8080/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=response_data)
        )
        result = client.chat(
            messages=[{"role": "user", "content": "go there"}],
            system_prompt="You are a shell translator.",
        )
        assert isinstance(result, ClarifyResponse)
        assert result.question == "Which dir?"

    @respx.mock
    def test_chat_payload_shape(self, client):
        """Verify request has no 'model' field and uses response_format."""
        response_data = {
            "choices": [{
                "message": {
                    "content": '{"type": "command", "command": "pwd", "explanation": ""}'
                }
            }]
        }
        route = respx.post("http://test:8080/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=response_data)
        )
        client.chat(
            messages=[{"role": "user", "content": "where am i"}],
            system_prompt="test",
        )
        request = route.calls[0].request
        payload = json.loads(request.content)
        assert "model" not in payload
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["stream"] is False
        assert payload["temperature"] == 0
