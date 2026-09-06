from __future__ import annotations

from typing import Any


_SECTION_SPECS: tuple[tuple[str, tuple[str, ...], dict[str, Any]], ...] = (
    (
        "identity_and_loop",
        ("title", "pitch", "core_loop"),
        {
            "title": {"type": "string", "minLength": 1},
            "pitch": {"type": "string", "minLength": 1},
            "core_loop": {"type": "array", "items": {"type": "string", "minLength": 1}},
        },
    ),
    (
        "systems_and_progression",
        ("progression", "combat", "mod_context"),
        {
            "progression": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
            "combat": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
            },
            "mod_context": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
            },
        },
    ),
    (
        "modules_and_assets",
        ("modules", "assets"),
        {
            "modules": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "plugin_id": {"type": "string", "minLength": 1},
                        "status": {"type": "string", "minLength": 1},
                        "reason": {"type": "string", "minLength": 1},
                        "requirement_refs": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "uniqueItems": True,
                        },
                        "implementation_obligations": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "minLength": 1},
                            "uniqueItems": True,
                        },
                    },
                    "required": [
                        "plugin_id",
                        "status",
                        "reason",
                        "requirement_refs",
                        "implementation_obligations",
                    ],
                    "additionalProperties": False,
                },
            },
            "assets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "kind": {"type": "string", "minLength": 1},
                        "brief": {"type": "string", "minLength": 1},
                    },
                    "required": ["id", "kind", "brief"],
                    "additionalProperties": False,
                },
            },
        },
    ),
    (
        "quality_and_art",
        ("acceptance_tests", "art_direction"),
        {
            "acceptance_tests": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
            "art_direction": {"type": "object"},
        },
    ),
)
