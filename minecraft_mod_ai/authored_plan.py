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
    schema_version: str = "mmm/authored-plan-v2"
    structured_sections: dict[str, Any] = field(default_factory=dict)
    content_design: dict[str, Any] = field(default_factory=dict)
    typed_plan_ir: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != "mmm/authored-plan-v2":
            raise ValueError(
                f"AUTHORED_PLAN_SCHEMA_UNSUPPORTED: {self.schema_version!r}"
            )
        for field_name, value in (
            ("requested_prompt", self.requested_prompt),
            ("text", self.text),
            ("existing_input_sha256", self.existing_input_sha256),
        ):
            if not isinstance(value, str):
                raise TypeError(f"AUTHORED_PLAN_STRING_REQUIRED: {field_name}")
        if not self.requested_prompt.strip():
            raise ValueError("AUTHORED_PLAN_PROMPT_REQUIRED")
        if not self.text.strip():
            raise ValueError("AUTHORED_PLAN_TEXT_REQUIRED")
        if not isinstance(self.media_paths, tuple) or any(
            not isinstance(value, str) for value in self.media_paths
        ):
            raise TypeError("AUTHORED_PLAN_MEDIA_PATHS_CANONICAL_REQUIRED")
        for field_name, value in (
            ("structured_sections", self.structured_sections),
            ("content_design", self.content_design),
            ("typed_plan_ir", self.typed_plan_ir),
        ):
            if not isinstance(value, dict):
                raise TypeError(f"AUTHORED_PLAN_OBJECT_REQUIRED: {field_name}")
            object.__setattr__(self, field_name, deepcopy(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requested_prompt": self.requested_prompt,
            "text": self.text,
            "existing_input_sha256": self.existing_input_sha256,
            "media_paths": list(self.media_paths),
            "structured_sections": deepcopy(self.structured_sections),
            "content_design": deepcopy(getattr(self, "content_design", {})),
            "typed_plan_ir": deepcopy(getattr(self, "typed_plan_ir", {})),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuthoredPlan:
        if not isinstance(data, dict):
            raise TypeError("AUTHORED_PLAN_OBJECT_REQUIRED")
        expected = {
            "schema_version",
            "requested_prompt",
            "text",
            "existing_input_sha256",
            "media_paths",
            "structured_sections",
            "content_design",
            "typed_plan_ir",
        }
        if set(data) != expected:
            raise ValueError(
                "AUTHORED_PLAN_FIELDS_INVALID: "
                f"missing={sorted(expected - set(data))}, "
                f"unknown={sorted(set(data) - expected)}"
            )
        if data["schema_version"] != "mmm/authored-plan-v2":
            raise ValueError(
                f"AUTHORED_PLAN_SCHEMA_UNSUPPORTED: {data['schema_version']!r}"
            )
        if not isinstance(data["media_paths"], list) or any(
            not isinstance(value, str) for value in data["media_paths"]
        ):
            raise TypeError("AUTHORED_PLAN_MEDIA_PATHS_ARRAY_REQUIRED")
        for field_name in ("structured_sections", "content_design", "typed_plan_ir"):
            if not isinstance(data[field_name], dict):
                raise TypeError(f"AUTHORED_PLAN_OBJECT_REQUIRED: {field_name}")
        return cls(
            requested_prompt=data["requested_prompt"],
            text=data["text"],
            existing_input_sha256=data["existing_input_sha256"],
            media_paths=tuple(data["media_paths"]),
            structured_sections=deepcopy(data["structured_sections"]),
            content_design=deepcopy(data["content_design"]),
            typed_plan_ir=deepcopy(data["typed_plan_ir"]),
            schema_version=data["schema_version"],
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
