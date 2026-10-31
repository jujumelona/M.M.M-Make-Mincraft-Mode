"""Durable first-pass model transcript for exact class and member decisions.

Only deterministic replay of the *same* prompt/schema/context is permitted.
Changing the task, target, source, or input invalidates the prior transcript.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import RLock
from uuid import uuid4

from .construction_state import PROTOCOL_VERSION
from .model_concurrency import model_execution_publication

_LOCK = RLock()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _canonical_messages(messages):
    """JSON metadata key order is not a change to the model's task.

    Durable responses are key-sorted. Re-embedding one in a later JSON prompt
    must hash identically after reload; Java/text strings inside it stay exact.
    """
    result = []
    for message in messages:
        row = dict(message)
        content = row.get("content")
        if isinstance(content, str):
            try:
                value = json.loads(content)
            except ValueError:
                pass
            else:
                if isinstance(value, (dict, list)):
                    row["content"] = _json(value)
        result.append(row)
    return result


class JournaledConstructionRouter:
    def __init__(self, router, project_root, spec, context):
        from .construction_implementation import construction_implementation_fingerprint
        self.router = router
        self.binding = _sha({
            "schema": "mmm/construction-inference-v1",
            "construction_protocol": PROTOCOL_VERSION,
            "implementation": construction_implementation_fingerprint(),
            "authoring_contract": context.get('authoring_contract', {}),
            "spec": spec,
            "target": context["target"],
            "prerequisites": context["prerequisite_hashes"],
            "source_symbols": sorted(
                (key, row.get("source_hash"))
                for key, row in context.get("project_symbols", {}).items()
            ),
        })
        root = Path(project_root).resolve()
        self.path = root / ".mmm" / "construction_inference" / (self.binding + ".json")
        self.position = 0
        with _LOCK:
            if self.path.exists():
                envelope = json.loads(self.path.read_text(encoding="utf-8"))
                if envelope.get("binding") != self.binding or not isinstance(envelope.get("rows"), list):
                    raise ValueError("CONSTRUCTION_INFERENCE_JOURNAL_INVALID")
                rows = envelope["rows"]
                if envelope.get("digest") != _sha(rows):
                    raise ValueError("CONSTRUCTION_INFERENCE_JOURNAL_CORRUPT")
                self.rows = rows
            else:
                self.rows = []

    def __getattr__(self, name):
        return getattr(self.router, name)

    def generate_structured_data(self, role, messages, *, data_name, parameters, **kwargs):
        identity = _sha({"role": role, "messages": _canonical_messages(messages),
                         "data_name": data_name, "parameters": parameters,
                         "kwargs": kwargs})
        with _LOCK:
            index = self.position
            if index < len(self.rows):
                row = self.rows[index]
                if row.get("input_sha256") != identity:
                    raise ValueError("CONSTRUCTION_INFERENCE_REPLAY_DRIFT: " + str(index))
                result = json.loads(row["response_json"])
                self.position += 1
                return result
            if index != len(self.rows):
                raise ValueError("CONSTRUCTION_INFERENCE_GAP")
            # Reserve the in-flight model action, but undo the reservation if
            # inference or atomic publication fails. A rejected response must
            # not poison the same router's next resume attempt.
            self.position += 1
        try:
            result = self.router.generate_structured_data(
                role, messages, data_name=data_name, parameters=parameters, **kwargs)
            serialized = _json(result)
            with _LOCK:
                if len(self.rows) != index:
                    raise ValueError("CONSTRUCTION_INFERENCE_CONCURRENT_CONFLICT")
                updated = [*self.rows, {"input_sha256": identity,
                                        "response_json": serialized}]
                envelope = {"binding": self.binding, "rows": updated,
                            "digest": _sha(updated)}
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_name("." + self.path.name + "." + uuid4().hex + ".tmp")
                try:
                    tmp.write_text(_json(envelope), encoding="utf-8")
                    with model_execution_publication():
                        tmp.replace(self.path)
                    self.rows = updated
                finally:
                    if tmp.exists():
                        tmp.unlink()
            return result
        except BaseException:
            with _LOCK:
                if self.position == index + 1 and len(self.rows) == index:
                    self.position = index
            raise

    def generate_java_source(self, role, messages, *, data_name, source_frame, **kwargs):
        """Replay a previously validated declaration without consuming another LLM call.

        A source frame, prompt, or project change changes the identity and cannot
        silently reuse a different member. Failed decoding/validation is never saved.
        """
        identity = _sha({"role": role, "messages": _canonical_messages(messages),
                         "data_name": data_name, "source_frame": source_frame,
                         "kwargs": kwargs})
        with _LOCK:
            index = self.position
            if index < len(self.rows):
                row = self.rows[index]
                if row.get("input_sha256") != identity:
                    raise ValueError("CONSTRUCTION_INFERENCE_REPLAY_DRIFT: " + str(index))
                source = json.loads(row["response_json"])
                if not isinstance(source, str):
                    raise ValueError("CONSTRUCTION_INFERENCE_SOURCE_CORRUPT")
                from .construction_source_repair import validate_generated_java_source
                validated = validate_generated_java_source(source, source_frame,
                    source_frame["prefix"], source_frame["suffix"])
                self.position += 1
                return validated
            if index != len(self.rows):
                raise ValueError("CONSTRUCTION_INFERENCE_GAP")
        source = self.router.generate_java_source(role, messages,
            data_name=data_name, source_frame=source_frame, **kwargs)
        if not isinstance(source, str):
            raise ValueError("CONSTRUCTION_INFERENCE_SOURCE_REQUIRED")
        from .construction_source_repair import validate_generated_java_source
        validate_generated_java_source(source, source_frame,
            source_frame["prefix"], source_frame["suffix"])
        with _LOCK:
            if len(self.rows) != index:
                raise ValueError("CONSTRUCTION_INFERENCE_CONCURRENT_CONFLICT")
            updated = [*self.rows, {"input_sha256": identity,
                                    "response_json": _json(source)}]
            envelope = {"binding": self.binding, "rows": updated,
                        "digest": _sha(updated)}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name("." + self.path.name + "." + uuid4().hex + ".tmp")
            try:
                tmp.write_text(_json(envelope), encoding="utf-8")
                with model_execution_publication():
                    tmp.replace(self.path)
                self.rows = updated
                self.position += 1
            finally:
                if tmp.exists():
                    tmp.unlink()
        return source
