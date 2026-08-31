from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.llama_completion_liveness_contract import (
    LlamaSemanticProgressTimeout,
    _ProgressCheckedResponse,
    _SemanticProgressWatchdog,
)
from minecraft_mod_ai.llama_generation_budget import structured_response_token_ceiling


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class _Lines:
    status_code = 200

    def __init__(self, lines):
        self._lines = list(lines)

    def iter_lines(self):
        yield from self._lines


def _identity_schema():
    return {
        "type": "object",
        "properties": {
            "section": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 1},
                    "pitch": {"type": "string", "minLength": 1},
                    "core_loop": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "required": ["title", "pitch", "core_loop"],
                "additionalProperties": False,
            }
        },
        "required": ["section"],
        "additionalProperties": False,
    }


def test_identity_section_budget_is_schema_derived_and_bounded():
    request = SimpleNamespace(response_format="json", response_schema=_identity_schema())
    derived = structured_response_token_ceiling(request)
    assert derived is not None
    ceiling, metrics = derived
    assert 1024 < ceiling < 10000
    assert metrics["schema_scalars"] == 3
    assert metrics["schema_arrays"] == 1
    assert metrics["schema_objects"] == 2
    assert metrics["schema_required"] == 4


def test_broader_schema_receives_more_output_capacity():
    identity = SimpleNamespace(response_format="json", response_schema=_identity_schema())
    broad_schema = _identity_schema()
    broad_schema["properties"]["section"]["properties"]["progression"] = {
        "type": "array",
        "items": {"type": "string"},
    }
    broad_schema["properties"]["section"]["required"].append("progression")
    broader = SimpleNamespace(response_format="json", response_schema=broad_schema)
    identity_ceiling, _ = structured_response_token_ceiling(identity)
    broader_ceiling, _ = structured_response_token_ceiling(broader)
    assert broader_ceiling > identity_ceiling


def test_semantic_watchdog_ignores_ping_and_times_out_without_model_progress():
    clock = _Clock()
    watchdog = _SemanticProgressWatchdog(5.0, clock=clock)
    assert watchdog.observe(": ping") is False
    clock.value = 4.9
    assert watchdog.observe(": ping") is False
    clock.value = 5.0
    with pytest.raises(LlamaSemanticProgressTimeout):
        watchdog.observe(": ping")


def test_semantic_watchdog_resets_on_prompt_progress():
    clock = _Clock()
    watchdog = _SemanticProgressWatchdog(5.0, clock=clock)
    clock.value = 4.0
    assert watchdog.observe('data: {"prompt_progress":{"processed":128}}') is True
    clock.value = 8.9
    assert watchdog.observe(": ping") is False


def test_lifecycle_logs_first_progress_completion_and_finalization(capsys):
    response = _ProgressCheckedResponse(
        _Lines(
            [
                'data: {"prompt_progress":{"processed":32}}',
                'data: {"choices":[{"delta":{"content":"ok"}}]}',
                "data: [DONE]",
            ]
        ),
        120.0,
        request_id="test-request",
        started_at=0.0,
    )
    assert list(response.iter_lines())[-1] == "data: [DONE]"
    output = capsys.readouterr().out
    assert "first semantic progress request_id=test-request" in output
    assert "lifecycle complete request_id=test-request" in output
    assert "lifecycle finalized request_id=test-request" in output
