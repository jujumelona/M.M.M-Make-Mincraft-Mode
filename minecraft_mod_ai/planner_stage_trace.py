from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


_TRACE_LOCK = threading.RLock()
_TRACE_SCHEMA = "mmm/planner-stage-trace-v1"


def _enabled() -> bool:
    value = os.environ.get("MMM_PLANNER_TRACE", "1").strip().lower()
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


class PlannerStageTrace:
    """Persist raw structured-planner attempts for debugging and later SFT/LoRA.

    The trace is deliberately append-only for one logical stage run. Every raw model
    response is recorded before parsing, together with the host validator result and
    any safely extracted candidate. This makes failures reproducible without making
    trace persistence part of the planning correctness contract.
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
        if not self.enabled:
            return
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
                "validation_error": validation_error,
                "candidate": _json_safe(dict(candidate)) if candidate is not None else None,
                "accepted": _json_safe(dict(accepted)) if accepted is not None else None,
                "context": dict(context or {}),
            }
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
