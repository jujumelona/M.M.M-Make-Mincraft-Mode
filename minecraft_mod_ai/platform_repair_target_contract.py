from __future__ import annotations

import json
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any

from .platform_catalog import adapter_from_project

_ACTIVE_REPAIR_TARGET: ContextVar[Any | None] = ContextVar(
    "mmm_repair_platform_target",
    default=None,
)


def install(repair_engine_module: Any) -> None:
    _install_repair_scope(repair_engine_module)
    _install_dynamic_patch_request(repair_engine_module)
    # Gradle Kotlin DSL is project-owned metadata and must be repairable during a
    # version port just like Groovy Gradle files.
    repair_engine_module._ALLOWED_SUFFIXES = set(
        repair_engine_module._ALLOWED_SUFFIXES
    ) | {".kts"}


def _install_repair_scope(module: Any) -> None:
    cls = module.RepairEngine
    original = cls.repair
    if getattr(original, "_mmm_dynamic_repair_target", False):
        return

    @wraps(original)
    def repair(self: Any, project_root: Any, **kwargs: Any):
        root = Path(project_root).expanduser().resolve()
        try:
            adapter = adapter_from_project(root)
        except ValueError as exc:
            raise module.RepairEngineError(
                "Repair target has no valid approved platform lock: " + str(exc)
            ) from exc
        token = _ACTIVE_REPAIR_TARGET.set(adapter)
        try:
            return original(self, root, **kwargs)
        finally:
            _ACTIVE_REPAIR_TARGET.reset(token)

    repair._mmm_dynamic_repair_target = True
    cls.repair = repair


def _install_dynamic_patch_request(module: Any) -> None:
    cls = module.RepairEngine
    if getattr(cls._request_patch, "_mmm_dynamic_repair_target", False):
        return

    def request_patch(
        self: Any,
        evidence: dict[str, Any],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        adapter = _ACTIVE_REPAIR_TARGET.get()
        if adapter is None:
            raise module.RepairEngineError(
                "Repair patch request is not bound to a platform target."
            )
        prompt = {
            "task": (
                f"Repair the Minecraft Java {adapter.minecraft_version} "
                f"{adapter.loader.capitalize()} project using exact minimal patches."
            ),
            "target": {
                "minecraft_version": adapter.minecraft_version,
                "loader": adapter.loader,
                "mappings": adapter.yarn_mappings,
                "java": adapter.java_version,
                "fabric_loader": adapter.fabric_loader,
                "fabric_api": adapter.fabric_api,
                "loom": adapter.fabric_loom,
                "gradle": adapter.gradle,
            },
            "constraints": [
                "Return exactly one JSON object with key operations.",
                "Use only create, replace or edit operations.",
                "Every non-create operation must use the supplied exact SHA-256.",
                "Do not delete requested functionality or mix loaders/versions.",
                "Do not emit shell commands, scripts or markdown.",
                "Use project-index paths; do not assume that omitted content means a file does not exist.",
                "Treat the target object as immutable; repair source and Gradle metadata to it, never alter it.",
            ],
            "evidence": evidence,
            "project_context": context,
        }
        text = self.router.generate_text(
            "coder",
            [
                {
                    "role": "system",
                    "content": (
                        "You are a hash-guarded Fabric source repair agent. Repair only "
                        "for the exact immutable target object supplied by the host."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            response_format="json",
            # Best-of-N candidate generation is a pure proposal phase. The normal
            # generation stage exposes mutating MCP tools such as apply_source_patch;
            # allowing candidates to call them would let a losing candidate change the
            # project before verifier selection. Candidates therefore receive only the
            # exact host-supplied evidence/context and return inert patch JSON. The one
            # selected patch is applied later by RepairEngine.
            enable_tools=False,
        )
        value = module._extract_json(text)
        if set(value) != {"operations"}:
            raise module.RepairEngineError(
                "Coder repair response must contain only operations."
            )
        operations = value["operations"]
        if not isinstance(operations, list) or not operations:
            raise module.RepairEngineError(
                "Coder did not return a non-empty operations list."
            )
        encoded = len(json.dumps(operations, ensure_ascii=False).encode("utf-8"))
        if encoded > self.policy.max_patch_bytes:
            raise module.RepairEngineError(
                "Repair patch exceeds MMM_MAX_PATCH_BYTES; raise the host policy explicitly."
            )

        # Candidate generation must be side-effect free. Best-of-N repair may execute
        # several requests concurrently, so mutating progressive JDT scope here would
        # race between candidates. The agentic repair selector commits
        # _mmm_last_java_paths only after the deterministic winner is selected.
        return operations

    request_patch._mmm_dynamic_repair_target = True
    # Prevent validation_execution_contract from re-introducing per-candidate scope
    # mutation if contracts are composed manually in a different order. The actual
    # progressive scope is committed downstream by the repair winner selector.
    request_patch._mmm_tracks_repair_scope = True
    request_patch._mmm_defers_repair_scope_commit = True
    request_patch._mmm_pure_candidate_generation = True
    cls._request_patch = request_patch
