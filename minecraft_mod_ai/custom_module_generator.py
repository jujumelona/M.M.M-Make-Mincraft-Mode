from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .complete_spec import ProductionModule
from .model_router import ModelRouter
from .source_patch import TransactionalSourcePatcher


class CustomModuleGenerationError(RuntimeError):
    pass


class CustomModuleGenerator:
    """Generate unusual Fabric modules through bounded, exact source patches."""

    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    def generate(
        self,
        project_root: str | Path,
        *,
        module: ProductionModule,
        minecraft_version: str = "1.20.1",
        loader: str = "fabric",
        mappings: str = "1.20.1+build.1",
    ) -> dict[str, Any]:
        module.validate()
        root = Path(project_root).expanduser().resolve()
        if not root.is_dir() or root.is_symlink():
            raise CustomModuleGenerationError("Custom module target must be a regular project directory.")
        snapshot_paths = self._snapshot_paths(root)
        snapshot = TransactionalSourcePatcher(root).snapshot(snapshot_paths)
        request = {
            "task": "Implement one complete approved Minecraft Fabric module as the smallest exact source patch.",
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
            "snapshot": snapshot,
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
            },
            "forbidden": [
                "shell or scripts",
                "deleting files or requested functionality",
                "mixing loaders, mappings or Minecraft versions",
                "writing outside src, build.gradle, gradle.properties or .minecraft_ai",
                "claiming success without generated code and resources",
            ],
        }
        text = self.router.generate_text(
            "coder",
            [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one JSON object. Implement compileable Minecraft Java 1.20.1 "
                        "Fabric code and data. Use current project conventions and server authority."
                    ),
                },
                {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
            ],
            response_format="json",
        )
        payload = _extract_json(text)
        if set(payload) != {"operations", "runtime_tests"}:
            raise CustomModuleGenerationError("Custom module response fields are invalid.")
        operations = payload["operations"]
        if not isinstance(operations, list) or not operations:
            raise CustomModuleGenerationError("Custom module did not return patch operations.")
        self._validate_operations(operations)
        receipt = TransactionalSourcePatcher(root).apply(operations)
        runtime_tests = payload["runtime_tests"]
        if not isinstance(runtime_tests, list) or not runtime_tests:
            raise CustomModuleGenerationError("Custom module must provide runtime tests.")
        return {
            "schema_version": "mmm/custom-module-result-v1",
            "module_id": module.module_id,
            "kind": module.kind,
            "status": "SOURCE_GENERATED",
            "patch_receipt": receipt,
            "runtime_tests": [str(value) for value in runtime_tests],
            "required_gates": ["JDT", "Gradle", "GameTest", *module.required_gates],
        }

    @staticmethod
    def _snapshot_paths(root: Path) -> list[str]:
        paths: list[str] = []
        for fixed in (
            "build.gradle",
            "gradle.properties",
            "src/main/resources/fabric.mod.json",
        ):
            if (root / fixed).is_file():
                paths.append(fixed)
        for path in sorted((root / "src/main/java").rglob("*.java"))[:20]:
            if path.is_file() and not path.is_symlink():
                paths.append(path.relative_to(root).as_posix())
        for path in sorted((root / "src/main/resources").rglob("*.json"))[:12]:
            if path.is_file() and not path.is_symlink():
                paths.append(path.relative_to(root).as_posix())
        return paths

    @staticmethod
    def _validate_operations(operations: list[dict[str, Any]]) -> None:
        if len(operations) > 40:
            raise CustomModuleGenerationError("One custom module may touch at most 40 files.")
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
                or path.startswith(".minecraft_ai/")
                or path in {"build.gradle", "gradle.properties", "settings.gradle"}
            )
            if not allowed:
                raise CustomModuleGenerationError(f"Custom module path is outside the allowed scope: {path}")


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
