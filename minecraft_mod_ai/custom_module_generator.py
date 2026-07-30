from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .complete_spec import ProductionModule
from .model_router import ModelRouter
from .project_index import ProjectIndex
from .scale_policy import ScalePolicy
from .source_patch import TransactionalSourcePatcher


class CustomModuleGenerationError(RuntimeError):
    pass


class CustomModuleGenerator:
    """Generate unusual Fabric modules from a whole-project relevance index.

    The previous implementation inspected the first 20 Java and 12 JSON files and
    rejected modules touching more than 40 files. This implementation indexes the
    complete project and paginates model output until the module is complete. Host
    protection is byte-based and configurable; feature/file counts are not capped.
    """

    def __init__(
        self,
        router: ModelRouter,
        *,
        policy: ScalePolicy | None = None,
    ) -> None:
        self.router = router
        self.policy = policy or ScalePolicy.from_environment()
        self.policy.validate()

    def generate(
        self,
        project_root: str | Path,
        *,
        module: ProductionModule,
        minecraft_version: str = "1.20.1",
        loader: str = "fabric",
        mappings: str = "1.20.1+build.1",
    ) -> dict[str, Any]:
        module.validate(policy=self.policy)
        root = Path(project_root).expanduser().resolve()
        if not root.is_dir() or root.is_symlink():
            raise CustomModuleGenerationError("Custom module target must be a regular project directory.")

        index = ProjectIndex(root, policy=self.policy)
        query = json.dumps(
            {
                "module_id": module.module_id,
                "kind": module.kind,
                "config": module.config,
                "depends_on": module.depends_on,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        context = index.select(query=query)
        base_request = {
            "task": "Implement one complete approved Minecraft Fabric module as exact source patches.",
            "target": {
                "minecraft_version": minecraft_version,
                "loader": loader,
                "mappings": mappings,
                "java": "17",
            },
            "module": {
                "module_id": module.module_id,
                "kind": module.kind,
                "config": module.config,
                "depends_on": list(module.depends_on),
                "required_gates": list(module.required_gates),
            },
            "project_manifest": index.manifest(),
            "relevant_context": context,
            "output_contract": {
                "operations": [
                    {
                        "operation": "create|replace|edit",
                        "path": "project-relative UTF-8 text path",
                        "expected_sha256": "required for replace/edit",
                        "content": "required for create/replace",
                        "replacements": "required for edit",
                    }
                ],
                "runtime_tests": ["observable tests"],
                "complete": True,
                "next_cursor": "empty when complete; otherwise stable opaque cursor",
            },
            "forbidden": [
                "shell or scripts",
                "deleting files or requested functionality",
                "mixing loaders, mappings or Minecraft versions",
                "writing outside src, Gradle metadata or .minecraft_ai",
                "claiming success without generated code and resources",
            ],
        }

        operations: list[dict[str, Any]] = []
        runtime_tests: list[str] = []
        seen_cursors: set[str] = set()
        cursor = ""
        while True:
            request = {**base_request, "cursor": cursor}
            text = self.router.generate_text(
                "coder",
                [
                    {
                        "role": "system",
                        "content": (
                            "Return exactly one JSON object. Implement compilable Minecraft Java 1.20.1 "
                            "Fabric code and data. Use project conventions, server authority and persistence. "
                            "When the patch is too large, return a non-empty next_cursor and continue without "
                            "repeating paths on the next page."
                        ),
                    },
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
                ],
                response_format="json",
            )
            payload = _extract_json(text)
            allowed = {"operations", "runtime_tests", "complete", "next_cursor"}
            if set(payload) != allowed:
                raise CustomModuleGenerationError("Custom module response fields are invalid.")
            page_operations = payload["operations"]
            page_tests = payload["runtime_tests"]
            complete = payload["complete"]
            next_cursor = payload["next_cursor"]
            if not isinstance(page_operations, list) or not page_operations:
                raise CustomModuleGenerationError("Custom module page did not return patch operations.")
            if not isinstance(page_tests, list):
                raise CustomModuleGenerationError("Custom module runtime_tests must be a list.")
            if type(complete) is not bool or not isinstance(next_cursor, str):
                raise CustomModuleGenerationError("Custom module pagination contract is invalid.")
            self._validate_operations(page_operations)
            operations.extend(page_operations)
            runtime_tests.extend(str(value) for value in page_tests if str(value).strip())
            self._validate_total_patch_bytes(operations)
            if complete:
                if next_cursor:
                    raise CustomModuleGenerationError("A complete custom module may not return next_cursor.")
                break
            if not next_cursor or next_cursor in seen_cursors:
                raise CustomModuleGenerationError("Custom module pagination did not advance.")
            seen_cursors.add(next_cursor)
            cursor = next_cursor

        paths = [str(item.get("path", "")).replace("\\", "/") for item in operations]
        if len(paths) != len(set(paths)):
            raise CustomModuleGenerationError(
                "Paginated custom module returned the same path more than once; combine edits per path."
            )
        if not runtime_tests:
            raise CustomModuleGenerationError("Custom module must provide runtime tests.")
        receipt = TransactionalSourcePatcher(root).apply(operations)
        ProjectIndex(root, policy=self.policy).write_manifest()
        return {
            "schema_version": "mmm/custom-module-result-v2",
            "module_id": module.module_id,
            "kind": module.kind,
            "status": "SOURCE_GENERATED",
            "patch_receipt": receipt,
            "operation_count": len(operations),
            "runtime_tests": runtime_tests,
            "required_gates": ["JDT", "Gradle", "GameTest", *module.required_gates],
        }

    def _validate_operations(self, operations: list[dict[str, Any]]) -> None:
        for item in operations:
            if not isinstance(item, dict):
                raise CustomModuleGenerationError("Patch operation must be an object.")
            if item.get("operation") not in {"create", "replace", "edit"}:
                raise CustomModuleGenerationError("Custom module may not delete files.")
            path = str(item.get("path", "")).replace("\\", "/")
            allowed = (
                path.startswith("src/main/java/")
                or path.startswith("src/main/resources/")
                or path.startswith("src/test/java/")
                or path.startswith("src/gametest/")
                or path.startswith(".minecraft_ai/")
                or path in {"build.gradle", "gradle.properties", "settings.gradle"}
            )
            if not allowed:
                raise CustomModuleGenerationError(
                    f"Custom module path is outside the allowed scope: {path}"
                )

    def _validate_total_patch_bytes(self, operations: list[dict[str, Any]]) -> None:
        size = len(json.dumps(operations, ensure_ascii=False).encode("utf-8"))
        if size > self.policy.max_patch_bytes:
            raise CustomModuleGenerationError(
                "Custom module patch exceeds MMM_MAX_PATCH_BYTES; raise the explicit host resource policy "
                "or split the feature into dependency-linked modules."
            )


def _extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise CustomModuleGenerationError("Custom module response did not contain JSON.")
