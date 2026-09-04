from __future__ import annotations

import json
import queue

import pytest

from minecraft_mod_ai.java_lsp import JDTLanguageServerError
from minecraft_mod_ai.java_lsp_trace import _collect_diagnostics_traced
from minecraft_mod_ai.root_cause_trace import bounded_safe, traced_callable


def _trace_payloads(stderr: str) -> list[dict[str, object]]:
    prefix = "ROOT CAUSE TRACE: "
    return [
        json.loads(line[len(prefix):])
        for line in stderr.splitlines()
        if line.startswith(prefix)
    ]


def test_bounded_safe_redacts_secret_fields_recursively() -> None:
    value = bounded_safe(
        {
            "token": "do-not-log",
            "nested": {"authorization": "Bearer hidden", "safe": "visible"},
        }
    )
    assert value["token"] == "<redacted>"
    assert value["nested"]["authorization"] == "<redacted>"
    assert value["nested"]["safe"] == "visible"


class _NoFullIterationList(list[int]):
    def __iter__(self):
        raise AssertionError("bounded tracing must not iterate an entire list")


def test_bounded_safe_does_not_copy_unreported_sequence_tail() -> None:
    value = _NoFullIterationList(range(1_000))
    bounded = bounded_safe(value)

    assert bounded[:3] == [0, 1, 2]
    assert len(bounded) == 65
    assert bounded[-1] == "<truncated:936>"


def test_bounded_safe_keeps_large_set_output_bounded_and_ordered() -> None:
    bounded = bounded_safe({f"item-{index:04d}" for index in range(1_000)})

    assert len(bounded) == 65
    assert bounded[:3] == ["item-0000", "item-0001", "item-0002"]
    assert bounded[-1] == "<truncated:936>"


def test_traced_callable_preserves_original_exception_chain(capsys: pytest.CaptureFixture[str]) -> None:
    def failing(api_key: str) -> None:
        try:
            raise KeyError("first-cause")
        except KeyError as exc:
            raise RuntimeError("wrapper") from exc

    wrapped = traced_callable(failing, stage="quality", operation="failing")
    with pytest.raises(RuntimeError, match="wrapper"):
        wrapped("secret-value")

    captured = capsys.readouterr()
    assert "ROOT CAUSE TRACE:" not in captured.out
    payloads = _trace_payloads(captured.err)
    failure = next(item for item in payloads if item["event"] == "operation_failure")
    assert [item["type"] for item in failure["exception_chain"]] == ["RuntimeError", "KeyError"]
    start = next(item for item in payloads if item["event"] == "operation_start")
    assert start["details"]["arguments"]["api_key"] == "<redacted>"


class _AliveProcess:
    pid = 123

    @staticmethod
    def poll() -> None:
        return None


class _AliveReader:
    @staticmethod
    def is_alive() -> bool:
        return True


class _FakeRpc:
    def __init__(self) -> None:
        self.process = _AliveProcess()
        self.messages: queue.Queue[dict[str, object]] = queue.Queue()
        self.stderr: list[str] = []
        self._reader = _AliveReader()
        self.reader_failure = None
        self.stdout_eof = False
        self.protocol_counts = {"textDocument/publishDiagnostics": 1}

    def send(self, payload: dict[str, object]) -> None:
        raise AssertionError(f"unexpected server request response: {payload}")


def test_jdt_timeout_reports_missing_and_mismatched_uris(capsys: pytest.CaptureFixture[str]) -> None:
    rpc = _FakeRpc()
    expected = "file:///workspace/Expected.java"
    unexpected = "file:///workspace/Other.java"
    rpc.messages.put(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {"uri": unexpected, "diagnostics": []},
        }
    )

    with pytest.raises(JDTLanguageServerError) as exc_info:
        _collect_diagnostics_traced(
            rpc,
            expected_uris={expected},
            timeout_seconds=0.02,
            quiet_seconds=0.0,
            page_index=0,
        )

    message = str(exc_info.value)
    assert expected in message
    assert unexpected in message
    assert "reader_alive=True" in message
    captured = capsys.readouterr()
    assert "ROOT CAUSE TRACE:" not in captured.out
    payloads = _trace_payloads(captured.err)
    timeout = next(item for item in payloads if item["event"] == "jdt_publish_timeout")
    assert timeout["details"]["missing_uris"] == [expected]
    assert timeout["details"]["unexpected_uris"] == [unexpected]
