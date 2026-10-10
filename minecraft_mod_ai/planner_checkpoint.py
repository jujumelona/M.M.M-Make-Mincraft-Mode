"""Crash-resumable, input-bound checkpoints for expensive authored planning.

Only model outputs already accepted by the existing schema validators are stored.
The authoritative content graph is still compiled and validated on first creation.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

_SCHEMA = "mmm/planner-checkpoint-v1"


def _digest(value: Any) -> str:
    data = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _media_identity(paths: Sequence[str | Path]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve(strict=True)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result.append({"path": str(path), "sha256": digest.hexdigest()})
    return result


class PlannerCheckpoint:
    """Persist completed planner records across Colab cell/kernel reruns.

    A new prompt, media content, model, target or source revision gets a distinct
    namespace. A new catalog gets a distinct record store; no guessed record is
    ever substituted for an uncached result.
    """

    def __init__(
        self,
        *,
        prompt: str,
        media_paths: Sequence[str | Path] = (),
        existing_input_sha256: str = "",
        model_id: str = "",
        target_version: str = "",
        target_loader: str = "",
        source_revision: str = "",
    ) -> None:
        self._lock = threading.RLock()
        self.enabled = os.environ.get(
            "MMM_PLAN_CHECKPOINT_ENABLED", "1"
        ).strip().lower() not in {"0", "false", "off", "no"}
        identity = {
            "schema": _SCHEMA,
            "prompt": prompt,
            "media": _media_identity(media_paths) if self.enabled else [],
            "existing_input_sha256": existing_input_sha256,
            "model_id": model_id,
            "target_version": target_version,
            "target_loader": target_loader,
            "source_revision": source_revision,
        }
        self._key = _digest(identity)
        directory = os.environ.get("MMM_PLAN_CHECKPOINT_DIR", "").strip()
        if directory:
            root = Path(directory).expanduser()
        elif os.environ.get("MMM_COLAB_SETUP_RECEIPT", "").strip():
            root = Path("/content/mmm-output/plan-checkpoints")
        else:
            root = Path.cwd() / ".mmm" / "plan-checkpoints"
        self.path = root / (self._key + ".json")
        self._data: dict[str, Any] = {
            "schema": _SCHEMA,
            "identity_sha256": self._key,
            "structured_sections": None,
            "catalog_sha256": None,
            "content_progress": {},
            "content_design": None,
        }
        if self.enabled:
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError, UnicodeError):
                saved = None
            if (
                isinstance(saved, dict)
                and saved.get("schema") == _SCHEMA
                and saved.get("identity_sha256") == self._key
            ):
                self._data.update(saved)

    def _write(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            "." + self.path.name + "." + uuid.uuid4().hex + ".tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(
                    self._data, handle, sort_keys=True, ensure_ascii=False,
                    separators=(",", ":"), allow_nan=False,
                )
                handle.write("\n")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def load_sections(self) -> dict[str, Any] | None:
        value = self._data.get("structured_sections")
        return deepcopy(value) if isinstance(value, dict) else None

    def save_sections(self, sections: Mapping[str, Any]) -> None:
        with self._lock:
            self._data["structured_sections"] = deepcopy(dict(sections))
            self._write()

    def clear_sections(self) -> None:
        with self._lock:
            self._data["structured_sections"] = None
            self._data["catalog_sha256"] = None
            self._data["content_progress"] = {}
            self._data["content_design"] = None
            self._write()

    def _select_catalog(self, catalog: Mapping[str, Any]) -> None:
        current = _digest(catalog)
        if self._data.get("catalog_sha256") != current:
            self._data["catalog_sha256"] = current
            self._data["content_progress"] = {}
            self._data["content_design"] = None
        if not isinstance(self._data.get("content_progress"), dict):
            self._data["content_progress"] = {}

    def content_progress(self, catalog: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._select_catalog(catalog)
            return deepcopy(self._data["content_progress"])

    def save_content_record(self, binding: str, accepted: Any) -> None:
        with self._lock:
            self._data["content_progress"][binding] = deepcopy(accepted)
            self._write()

    def load_content_design(self, catalog: Mapping[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            self._select_catalog(catalog)
            value = self._data.get("content_design")
            return deepcopy(value) if isinstance(value, dict) else None

    def save_content_design(
        self, catalog: Mapping[str, Any], design: Mapping[str, Any]
    ) -> None:
        with self._lock:
            self._select_catalog(catalog)
            self._data["content_design"] = deepcopy(dict(design))
            self._write()


__all__ = ["PlannerCheckpoint"]
