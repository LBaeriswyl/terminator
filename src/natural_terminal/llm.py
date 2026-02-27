"""llama-server (llama.cpp) API client with structured response parsing."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from natural_terminal.prompt import get_few_shot_examples


class CommandResponse(BaseModel):
    type: str = "command"
    command: str
    explanation: str = ""


class ClarifyResponse(BaseModel):
    type: str = "clarify"
    question: str


class ParseError(Exception):
    """Raised when LLM output cannot be parsed into a known response type."""

    def __init__(self, raw_output: str, message: str = "Could not parse LLM response"):
        self.raw_output = raw_output
        super().__init__(message)


# JSON schema for structured output via response_format
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["command", "clarify"]},
        "command": {"type": "string"},
        "explanation": {"type": "string"},
        "question": {"type": "string"},
    },
    "required": ["type"],
}


class LlamaCppClient:
    def __init__(self, base_url: str = "http://localhost:8080", model: str = "llama3.1:8b", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def check_health(self) -> bool:
        """Check if the server is healthy and ready (model fully loaded)."""
        try:
            resp = self._client.get("/health")
            if resp.status_code != 200:
                return False
            data = resp.json()
            return data.get("status") == "ok"
        except (httpx.ConnectError, httpx.TimeoutException, ValueError):
            return False

    def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
    ) -> CommandResponse | ClarifyResponse:
        full_messages = [
            {"role": "system", "content": system_prompt},
            *get_few_shot_examples(self.model),
            *messages,
        ]

        payload: dict[str, Any] = {
            "messages": full_messages,
            "stream": False,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": RESPONSE_SCHEMA,
                },
            },
        }

        resp = self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()

        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return self._parse_response(content)

    def _parse_response(self, content: str) -> CommandResponse | ClarifyResponse:
        # Strategy 1: direct JSON parse
        parsed = self._try_json_parse(content)
        if parsed:
            return parsed

        # Strategy 2: extract from markdown code fences
        fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
        if fenced:
            parsed = self._try_json_parse(fenced.group(1))
            if parsed:
                return parsed

        # Strategy 3: heuristic — look for JSON-like structure anywhere in text
        json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if json_match:
            parsed = self._try_json_parse(json_match.group(0))
            if parsed:
                return parsed

        # Strategy 4: heuristic command extraction (bare command in output)
        stripped = content.strip()
        if stripped and "\n" not in stripped and not stripped.startswith("{"):
            return CommandResponse(type="command", command=stripped, explanation="(extracted from raw output)")

        raise ParseError(content)

    def _try_json_parse(self, text: str) -> CommandResponse | ClarifyResponse | None:
        try:
            data = json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            return None

        if not isinstance(data, dict) or "type" not in data:
            return None

        try:
            if data["type"] == "clarify":
                return ClarifyResponse.model_validate(data)
            else:
                return CommandResponse.model_validate(data)
        except ValidationError:
            return None
