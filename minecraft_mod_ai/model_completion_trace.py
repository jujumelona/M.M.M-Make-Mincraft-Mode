"""Record exact model requests and streamed output before consumers can discard it."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from .root_cause_trace import durable_trace_path

_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("mmm_completion_trace_context", default={})
_ACTIVE: dict[str, "CompletionTrace"] = {}
_ACTIVE_LOCK = threading.Lock()
_OUTPUT_LOCK = threading.Lock()
_PREFIX = "MODEL COMPLETION TRACE: "


@contextmanager
def completion_trace_context(**fields: Any):
    """Carry host-owned job/action identifiers through nested model calls."""
    token = _CONTEXT.set({**_CONTEXT.get(), **fields})
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def report_active_completions(job_ids: list[str], *, reason: str) -> None:
    """Expose partial output even when the scheduler fails outside the model thread."""
    with _ACTIVE_LOCK:
        active = [trace for trace in _ACTIVE.values() if trace.context.get("job_id") in job_ids]
    for trace in active:
        trace.snapshot(reason=reason)


class CompletionTrace:
    """One full request journal with compact console status.

    Each SSE data line is written and flushed before it reaches the parser/watchdog.
    The journal therefore preserves whitespace, reasoning, malformed JSON and partial
    tool arguments without waiting for a successful response or truncating the output.
    HTTP headers are deliberately outside this interface.
    """

    def __init__(self, request_id: str, payload: Mapping[str, Any], *, started_at: float) -> None:
        self.request_id = request_id
        self.context = dict(_CONTEXT.get())
        self.started_at = started_at
        self._lock = threading.RLock()
        self._finished = False
        self._sequence = 0
        self._choices: dict[int, dict[str, Any]] = {}
        self._usage: Any = None
        self._timings: Any = None
        self._output_events = 0
        self._last_output_elapsed: float | None = None
        self._prompt_progress: Any = None
        grammar = payload.get("grammar")
        self._constraint = (
            {"format": "llguidance" if grammar.lstrip().startswith("%llguidance") else "gbnf",
             "grammar_chars": len(grammar), "grammar_rules": len(grammar.splitlines()),
             "grammar_sha256": hashlib.sha256(grammar.encode("utf-8")).hexdigest()}
            if isinstance(grammar, str) else
            {"format": "json-schema"} if payload.get("json_schema") else {"format": "none"}
        )
        self._http_status: int | None = None
        self._journal: Any = None
        self._recording_error: str | None = None
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", request_id)
        self.path = durable_trace_path().parent / "model_completions" / f"{name}.jsonl"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            self._journal = os.fdopen(fd, "a", encoding="utf-8", errors="backslashreplace")
        except OSError as exc:
            self._recording_error = f"{type(exc).__name__}: {exc}"
        self._emit("request", payload=dict(payload))
        with _ACTIVE_LOCK:
            _ACTIVE[request_id] = self

    def _emit(self, event: str, **fields: Any) -> None:
        self._sequence += 1
        record = {
            "schema_version": "mmm/model-completion-trace-v1",
            "event": event,
            "request_id": self.request_id,
            "sequence": self._sequence,
            "elapsed_seconds": round(max(0.0, time.monotonic() - self.started_at), 3),
            "context": self.context,
            "journal_path": str(self.path),
            **fields,
        }
        if self._recording_error:
            record["recording_error"] = self._recording_error
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str)
        with _OUTPUT_LOCK:
            if self._journal is not None:
                try:
                    self._journal.write(line + "\n")
                    self._journal.flush()
                except OSError as exc:
                    self._recording_error = f"{type(exc).__name__}: {exc}"
                    record["recording_error"] = self._recording_error
                    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str)
            if event in {"request", "response_snapshot", "response_finished"}:
                # Per-token JSON and full prompts belong only in the journal. A
                # notebook stores/renders every console row in the user's browser.
                summary = {
                    "event": event,
                    "request_id": self.request_id,
                    "elapsed_seconds": record["elapsed_seconds"],
                    "context": {key: str(value)[:160] for key, value in self.context.items()},
                    "journal_path": str(self.path),
                }
                if event == "request":
                    payload = fields.get("payload", {})
                    summary["max_tokens"] = payload.get("max_tokens")
                    summary["constraint"] = self._constraint
                if "status" in fields:
                    summary["status"] = fields["status"]
                if "reason" in fields:
                    summary["reason"] = fields["reason"]
                error = fields.get("error")
                if isinstance(error, Mapping):
                    summary["error_type"] = error.get("type")
                    summary["http_status"] = error.get("status_code")
                if event == "response_snapshot" or (event == "response_finished" and
                    (fields.get("partial") or fields.get("status") == "error")):
                    summary["partial_output"] = self._partial_summary()
                    summary["constraint"] = self._constraint
                if self._recording_error:
                    summary["recording_error"] = self._recording_error
                try:
                    print(_PREFIX + json.dumps(summary, ensure_ascii=False), flush=True)
                except (OSError, UnicodeError):
                    # Diagnostic output must not replace the inference failure.
                    pass

    def _capture(self, chunk: Mapping[str, Any]) -> None:
        if isinstance(chunk.get("prompt_progress"), Mapping):
            self._prompt_progress = dict(chunk["prompt_progress"])
        if "usage" in chunk:
            self._usage = chunk["usage"]
        if "timings" in chunk:
            self._timings = chunk["timings"]
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            return
        output_progressed = False
        for position, choice in enumerate(choices):
            if not isinstance(choice, Mapping):
                continue
            index = choice.get("index", position)
            if not isinstance(index, int):
                index = position
            output = self._choices.setdefault(index, {"channels": {}, "tool_call_deltas": []})
            if choice.get("finish_reason") is not None:
                output["finish_reason"] = choice["finish_reason"]
            delta = choice.get("delta")
            is_message = not isinstance(delta, Mapping)
            if is_message:
                delta = choice.get("message")
            if not isinstance(delta, Mapping):
                continue
            for channel in ("content", "reasoning_content", "reasoning"):
                value = delta.get(channel)
                if isinstance(value, str):
                    output_progressed = output_progressed or bool(value)
                    pieces = output["channels"].setdefault(channel, [])
                    if is_message:
                        pieces.clear()
                    pieces.append(value)
            if "tool_calls" in delta:
                output["tool_call_deltas"].append(delta["tool_calls"])
                output_progressed = output_progressed or bool(delta["tool_calls"])
        if output_progressed:
            self._output_events += 1
            self._last_output_elapsed = round(max(0.0, time.monotonic() - self.started_at), 3)

    def _partial_summary(self) -> dict[str, Any]:
        """Bounded failure evidence; the complete response stays in the journal."""
        channels = {}
        preview_remaining = 1024
        for _, output in sorted(self._choices.items()):
            for channel, pieces in output["channels"].items():
                characters = sum(map(len, pieces))
                # Assemble only the tail needed by the console, not a second
                # copy of a potentially large unfinished response.
                suffix = []
                remaining = preview_remaining
                for piece in reversed(pieces):
                    if remaining <= 0:
                        break
                    suffix.append(piece[-remaining:])
                    remaining -= len(suffix[-1])
                tail = "".join(reversed(suffix))
                channels[channel] = {"characters": characters, "tail": tail}
                preview_remaining -= len(tail)
            break
        return {"output_delta_events": self._output_events,
                "last_output_elapsed_seconds": self._last_output_elapsed,
                "last_prompt_progress": (
                    {key: value for key, value in self._prompt_progress.items()
                     if key in {"processed", "n_processed", "total", "n_total", "time_ms", "elapsed_ms"}
                     and isinstance(value, (int, float, bool))}
                    if isinstance(self._prompt_progress, Mapping) else None
                ),
                "channels": channels}

    def observe_line(self, raw_line: Any) -> None:
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
        stripped = line.strip()
        if not stripped.startswith("data:"):
            return
        data = stripped[5:].strip()
        with self._lock:
            if data == "[DONE]":
                self._emit("stream_done", raw_line=line)
                return
            try:
                chunk = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                self._emit("stream_unparsed", raw_line=line)
                return
            if isinstance(chunk, Mapping):
                self._capture(chunk)
            self._emit("response_delta", raw_line=line, chunk=chunk)

    def observe_body(self, body: Any, *, status_code: Any) -> None:
        text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body)
        with self._lock:
            if isinstance(status_code, int):
                self._http_status = status_code
            self._emit("response_body", status_code=status_code, body=text)
            try:
                chunk = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                return
            if isinstance(chunk, Mapping):
                self._capture(chunk)

    def _response(self) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "index": index,
                    "message": {channel: "".join(pieces) for channel, pieces in output["channels"].items()},
                    "tool_call_deltas": output["tool_call_deltas"],
                    "finish_reason": output.get("finish_reason"),
                }
                for index, output in sorted(self._choices.items())
            ],
            "usage": self._usage,
            "timings": self._timings,
        }

    def snapshot(self, *, reason: str) -> None:
        with self._lock:
            if not self._finished:
                self._emit("response_snapshot", reason=reason, partial=True, response=self._response())

    def finish(self, *, saw_done: bool = False, exc: BaseException | None = None) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
            error = {"type": type(exc).__name__, "message": str(exc)} if exc is not None else None
            if error is None and self._http_status is not None and self._http_status >= 400:
                error = {"type": "HTTPError", "status_code": self._http_status}
            self._emit(
                "response_finished",
                status="error" if error is not None else "complete" if saw_done else "incomplete",
                partial=not saw_done,
                error=error,
                response=self._response(),
            )
            if self._journal is not None:
                try:
                    self._journal.close()
                except OSError:
                    pass
        with _ACTIVE_LOCK:
            _ACTIVE.pop(self.request_id, None)
