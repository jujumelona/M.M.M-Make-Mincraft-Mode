from __future__ import annotations

"""Host-owned implementation facts supplied before the small coder decodes.

This module projects only facts already admitted by the immutable host version catalog.
It never searches, guesses API names, or asks the model to choose a retriever.
"""

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .host_version_catalog import host_target
from .minecraft_template_steps import steps_for_artifact
from .structural_routing_contract import validate_artifact_kinds
from .task_template_catalog import load_template


def _sha(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "sha256:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _module_config(module: Any) -> Mapping[str, Any]:
    value = getattr(module, "config", None)
    return value if isinstance(value, Mapping) else {}


def _evidence_task(module: Any) -> Mapping[str, Any]:
    value = _module_config(module).get("evidence_task")
    return value if isinstance(value, Mapping) else {}


def _explicit_artifact_kind(module: Any) -> str:
    config = _module_config(module)
    task = _evidence_task(module)
    raw = (
        config.get("semantic_kind")
        or config.get("artifact_kind")
        or task.get("artifact_kind")
        or task.get("semantic_kind")
        or ""
    )
    value = str(raw).strip()
    if not value:
        return ""
    return validate_artifact_kinds((value,))[0]


def build_generation_implementation_grounding(
    module: Any,
    *,
    minecraft_version: str,
) -> dict[str, Any] | None:
    """Return exact target API/template facts for an explicitly typed artifact task."""

    kind = _explicit_artifact_kind(module)
    version = str(minecraft_version or "").strip()
    if not kind or not version:
        return None

    target = host_target(version)
    context = target.version_context
    facts: list[dict[str, Any]] = []

    for step in steps_for_artifact(kind):
        try:
            binding = context.require_leaf_binding(step.template_id)
        except Exception:
            continue
        implementation = binding.get("implementation")
        if not isinstance(implementation, Mapping):
            continue

        required = dict(step.host_requirements)
        symbols: dict[str, Any] = {}
        for name in required.get("symbols", ()):
            try:
                symbols[str(name)] = context.require_fact("api_symbols", str(name))
            except Exception:
                continue

        template_id = str(implementation.get("template") or "").strip()
        template_fact: dict[str, Any] | None = None
        if template_id:
            try:
                template = load_template(template_id)
                context.admit_template(template)
            except Exception:
                template = None
            if isinstance(template, Mapping):
                render = template.get("render")
                body = render.get("body") if isinstance(render, Mapping) else None
                template_fact = {
                    "template_id": template_id,
                    "target": template.get("target"),
                    "requires": list(template.get("requires") or ()),
                    "dependencies": list(template.get("dependencies") or ()),
                    "render_body": body if isinstance(body, str) else "",
                }

        facts.append(
            {
                "responsibility": step.template_id,
                "outcome": step.outcome,
                "implementation_id": implementation.get("implementation_id"),
                "executor_type": implementation.get("executor_type"),
                "api_symbols": symbols,
                "template": template_fact,
                "validators": list(step.validators),
                "postconditions": list(step.postconditions),
            }
        )

    if not facts:
        return None

    core = {
        "schema_version": "mmm/generation-implementation-grounding-v1",
        "artifact_kind": kind,
        "minecraft_version": target.minecraft_version,
        "loader": target.loader,
        "naming_regime": (
            "mapped_obfuscated" if target.mappings_applicable else "native_unobfuscated"
        ),
        "context_id": context.context_id,
        "facts": facts,
        "policy": {
            "host_owned": True,
            "model_must_not_substitute_api_names": True,
            "prefer_minimal_required_responsibilities": True,
            "do_not_invent_entrypoints": True,
            "compile_feedback_repairs_same_generation": True,
        },
    }
    return {
        **core,
        "selected_fact_count": len(facts),
        "grounding_sha256": _sha(core),
    }


__all__ = ["build_generation_implementation_grounding"]
