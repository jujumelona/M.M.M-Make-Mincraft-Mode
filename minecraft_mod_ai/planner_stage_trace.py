from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_TRACE_LOCK = threading.RLock()
_TRACE_SCHEMA = "mmm/planner-stage-trace-v1"


def _enabled() -> bool:
    value = os.environ.get("MMM_PLANNER_TRACE", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _console_enabled() -> bool:
    value = os.environ.get("MMM_PLANNER_TRACE_CONSOLE", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _default_root() -> Path:
    configured = os.environ.get("MMM_PLANNER_TRACE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    colab = Path("/content")
    if colab.is_dir():
        return colab / "mmm_planner_traces"
    return Path.home() / ".cache" / "mmm" / "planner-traces"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
        return value
    except Exception:
        return str(value)


def _emit_console(payload: Mapping[str, Any]) -> None:
    if not _console_enabled():
        return
    print(
        "PLANNER STAGE TRACE: "
        + json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


class PlannerStageTrace:
    """Persist and stream raw structured-planner attempts for debugging and SFT/LoRA.

    Every raw model response is recorded before parsing, together with the host validator
    result and safely extracted candidate. The same diagnostic payload is printed to the
    console by default so a Colab/user log contains the actual failure cause instead of a
    later aggregate counter. Set ``MMM_PLANNER_TRACE_CONSOLE=0`` only when explicitly
    choosing quieter output; durable trace files remain controlled by ``MMM_PLANNER_TRACE``.
    """

    def __init__(
        self,
        *,
        stage: str,
        prompt: str,
        media_paths: Sequence[str | Path] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.enabled = _enabled()
        self.stage = stage
        self.prompt = prompt
        self.prompt_sha256 = _sha256_text(prompt)
        self.run_id = (
            f"{self.prompt_sha256[:16]}-"
            f"{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        self.root = _default_root()
        self.directory = self.root / stage / self.run_id
        self._attempt_index = 0
        request_payload = {
            "event": "planner_stage_start",
            "schema_version": _TRACE_SCHEMA,
            "run_id": self.run_id,
            "stage": stage,
            "prompt": prompt,
            "prompt_sha256": self.prompt_sha256,
            "media_paths": [str(Path(path)) for path in media_paths],
            "metadata": dict(metadata or {}),
            "trace_directory": str(self.directory),
        }
        _emit_console(request_payload)
        if not self.enabled:
            return
        with _TRACE_LOCK:
            self.directory.mkdir(parents=True, exist_ok=False)
            self._write_json(
                self.directory / "request.json",
                {
                    "schema_version": _TRACE_SCHEMA,
                    "run_id": self.run_id,
                    "stage": stage,
                    "prompt": prompt,
                    "prompt_sha256": self.prompt_sha256,
                    "media_paths": [str(Path(path)) for path in media_paths],
                    "metadata": dict(metadata or {}),
                },
            )

    def record_attempt(
        self,
        *,
        raw_output: str,
        validation_error: str | None,
        candidate: Mapping[str, Any] | None = None,
        accepted: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        with _TRACE_LOCK:
            index = self._attempt_index
            self._attempt_index += 1
            payload = {
                "schema_version": _TRACE_SCHEMA,
                "run_id": self.run_id,
                "stage": self.stage,
                "attempt_index": index,
                "prompt_sha256": self.prompt_sha256,
                "raw_output": raw_output,
                "raw_output_sha256": _sha256_text(raw_output),
                "raw_output_chars": len(raw_output),
                "validation_error": validation_error,
                "candidate": _json_safe(dict(candidate)) if candidate is not None else None,
                "accepted": _json_safe(dict(accepted)) if accepted is not None else None,
                "context": dict(context or {}),
                "trace_directory": str(self.directory),
                "attempt_path": str(self.directory / f"attempt-{index:06d}.json"),
            }
            _emit_console({"event": "planner_stage_attempt", **payload})
            if not self.enabled:
                return
            self._write_json(
                self.directory / f"attempt-{index:06d}.json",
                payload,
            )
            with (self.directory / "attempts.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    )
                    + "\n"
                )

    def record_success(self, design: Mapping[str, Any]) -> None:
        payload = {
            "event": "planner_stage_success",
            "schema_version": _TRACE_SCHEMA,
            "run_id": self.run_id,
            "stage": self.stage,
            "prompt_sha256": self.prompt_sha256,
            "accepted": dict(design),
            "trace_directory": str(self.directory),
            "accepted_path": str(self.directory / "accepted.json"),
        }
        _emit_console(payload)
        if not self.enabled:
            return
        with _TRACE_LOCK:
            self._write_json(
                self.directory / "accepted.json",
                {
                    "schema_version": _TRACE_SCHEMA,
                    "run_id": self.run_id,
                    "stage": self.stage,
                    "prompt_sha256": self.prompt_sha256,
                    "accepted": dict(design),
                },
            )

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
        temporary.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
