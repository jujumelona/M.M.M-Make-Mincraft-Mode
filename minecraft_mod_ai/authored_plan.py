"""A player's design document, independent of executable production contracts."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AuthoredPlan:
    requested_prompt: str
    text: str
    existing_input_sha256: str = ""
    media_paths: tuple[str, ...] = ()
    structured_sections: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "mmm/authored-plan-v2"

    def __post_init__(self) -> None:
        from .authored_structured_design import normalize_structured_sections

        object.__setattr__(
            self,
            "structured_sections",
            normalize_structured_sections(self.structured_sections),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requested_prompt": self.requested_prompt,
            "text": self.text,
            "existing_input_sha256": self.existing_input_sha256,
            "media_paths": list(self.media_paths),
            "structured_sections": deepcopy(self.structured_sections),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuthoredPlan:
        return cls(
            requested_prompt=data["requested_prompt"],
            text=data["text"],
            existing_input_sha256=data.get("existing_input_sha256", ""),
            media_paths=tuple(data.get("media_paths", ())),
            structured_sections=deepcopy(data.get("structured_sections") or {}),
            schema_version=str(data.get("schema_version") or "mmm/authored-plan-v1"),
        )

    def calculate_hash(self) -> str:
        encoded = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    def production_prompt(self) -> str:
        return (
            self.requested_prompt
            + "\n\nAuthored game design to implement (preserve its mechanics and details):\n"
            + self.text
        )
