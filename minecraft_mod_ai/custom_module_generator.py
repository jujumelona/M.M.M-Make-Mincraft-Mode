from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .complete_spec import ProductionModule
from .host_grounding import (
    build_coder_grounding,
    custom_module_path_allowed,
    custom_module_path_protected,
)
from .model_router import ModelRouter
from .platform_catalog import adapter_for_target, adapter_from_project
from .project_index import ProjectIndex
from .research_ledger import select_module_research_context
from .scale_policy import ScalePolicy
from .source_patch import TransactionalSourcePatcher


class CustomModuleGenerationError(RuntimeError):
    pass


_OBSERVATION_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,127}")
_ANCHOR_TERMS = {
    "api",
    "contract",
    "dependency",
    "implements",
    "interface",
    "public",
    "register",
    "required",
    "schema",
}
_FABRIC_ROOT_METADATA_PATH = "fabric.mod.json"
_FABRIC_CANONICAL_METADATA_PATH = "src/main/resources/fabric.mod.json"
_MAX_CUSTOM_MODULE_REPAIR_ATTEMPTS = 2


def _coder_project_context_budget(
    router: ModelRouter,
    policy: ScalePolicy,
    *,
    fast_mode: bool,
) -> int:
    """Size exact-source grounding from the active coder model's input window."""
    hard_cap = max(1024, int(policy.model_context_bytes))
    if fast_mode:
        return min(hard_cap, 4 * 1024)

    fallback = min(hard_cap, 12 * 1024)
    registry = getattr(router, "registry", None)
    resolve_role = getattr(registry, "role", None)
    profile = str(getattr(router, "profile", "") or "").strip()
    if not callable(resolve_role) or not profile:
        return fallback
    try:
        config = resolve_role(profile, "coder")
    except Exception:
        return fallback

    max_context = int(getattr(config, "max_context", 0) or 0)
    max_input = int(getattr(config, "max_input_tokens", 0) or 0)
    max_new = int(getattr(config, "max_new_tokens", 0) or 0)
    input_tokens = max_input if max_input > 0 else max(0, max_context - max_new)
    if input_tokens <= 0:
        return fallback

    reserve_tokens = max(2048, min(8192, max_new // 2 if max_new > 0 else 4096))
    evidence_tokens = max(512, input_tokens - reserve_tokens)
    return min(hard_cap, max(1024, evidence_tokens * 2))


class CustomModuleGenerator:
    """Generate one custom module with host-owned patch/state semantics.

    The model chooses semantic files and writes file text. Everything deterministic is
    owned by the host: canonical paths, create-vs-replace, expected SHA-256, progress,
    completion, validation and transactional application. Model output is received only
    through native forced function-call arguments; free-form JSON repair is not part of
    the production path.
    """

    def __init__(
        self,
        router: ModelRouter,
        *,
        policy: ScalePolicy | None = None,
        fast_mode: bool = False,
        project_index: ProjectIndex | None = None,
    ) -> None:
        self.router = router
        self.policy = policy or ScalePolicy.from_environment()
        self.policy.validate()
        self.fast_mode = fast_mode
        self._cached_index: ProjectIndex | None = project_index
        self._cached_root: Path | None = (
            project_index.root if project_index is not None else None
        )

    def generate(
        self,
        project_root: str | Path,
        *,
        module: ProductionModule,
        research_modules: Iterable[ProductionModule] = (),
        minecraft_version: str | None = None,
        loader: str | None = None,
        mappings: str | None = None,
    ) -> dict[str, Any]:
        module.validate(policy=self.policy)
        root = Path(project_root).expanduser().resolve()
        if not root.is_dir() or root.is_symlink():
            raise CustomModuleGenerationError(
                "Custom module target must be a regular project directory."
            )

        requested = tuple(
            str(value or "").strip()
            for value in (minecraft_version, loader, mappings)
        )
        if any(requested):
            if not all(requested):
                raise CustomModuleGenerationError(
                    "minecraft_version, loader and mappings must be supplied together."
                )
            try:
                adapter = adapter_for_target(requested[0], requested[1])
            except ValueError as exc:
                raise CustomModuleGenerationError(str(exc)) from exc
            if requested[2] != adapter.yarn_mappings:
                raise CustomModuleGenerationError(
                    "Requested mappings disagree with the executable provider target."
                )
        else:
            try:
                adapter = adapter_from_project(root)
            except ValueError as exc:
                raise CustomModuleGenerationError(
                    "Custom generation requires an explicit host target or an unambiguous "
                    "existing project platform lock; historical defaults are disabled."
                ) from exc

        minecraft_version = adapter.minecraft_version
        loader = adapter.loader
        mappings = adapter.yarn_mappings
        java_version = adapter.java_version

        if self._cached_root == root and self._cached_index is not None:
            index = self._cached_index
        else:
            index = ProjectIndex(root, policy=self.policy)
            self._cached_root = root
            self._cached_index = index

        self.router.bind_agent_workspace(root.parent, require_fresh_evidence=True)
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
        project_context_budget = _coder_project_context_budget(
            self.router,
            self.policy,
            fast_mode=self.fast_mode,
        )
        if self.fast_mode:
            print(
                "🚀 [Fast-Path] host exact-source grounding limited to 4 KiB.",
                flush=True,
            )

        observation_ledger: dict[str, Any] | None = None
        last_snapshot_error: ValueError | None = None
        for snapshot_attempt in range(3):
            try:
                observation_ledger = _collect_initial_observations(
                    self.router,
                    index,
                    query=query,
                    byte_budget=project_context_budget,
                )
                break
            except ValueError as exc:
                if not _is_stale_project_index_error(exc):
                    raise
                last_snapshot_error = exc
                index = ProjectIndex(root, policy=self.policy)
                self._cached_index = index
                self._cached_root = root
                print(
                    "↻ [CustomModule] ProjectIndex snapshot changed during parallel generation; "
                    f"refreshed context ({snapshot_attempt + 1}/3).",
                    flush=True,
                )
        if observation_ledger is None:
            raise CustomModuleGenerationError(
                "Project source kept changing while custom-module context was being captured; "
                "refusing to generate from a stale snapshot. "
                f"Last error: {last_snapshot_error}"
            )

        observation_pages = _observation_context_pages(
            observation_ledger,
            query=query,
            byte_budget=project_context_budget,
        )
        research_context = select_module_research_context(
            research_modules,
            query=query,
            byte_budget=8 * 1024,
        )
        host_grounding = build_coder_grounding(
            module_kind=module.kind,
            source_observation_receipt=observation_ledger["receipt"],
            research_context=research_context,
            minecraft_version=minecraft_version,
            loader=loader,
            mappings=mappings,
        )

        plan_request = {
            "phase": "plan_files",
            "task": "Choose only the source/resource files required to implement the approved module.",
            "target": {
                "minecraft_version": minecraft_version,
                "loader": loader,
                "mappings": mappings,
                "java": java_version,
            },
            "module": {
                "module_id": module.module_id,
                "kind": module.kind,
                "config": module.config,
                "depends_on": list(module.depends_on),
                "required_gates": list(module.required_gates),
            },
            "project_manifest": index.manifest_receipt(),
            "source_observation_receipt": observation_ledger["receipt"],
            "planning_context": observation_pages[0],
            "research_context": research_context,
            "host_grounding": host_grounding,
            "host_owned": [
                "create/replace/edit decision",
                "expected_sha256",
                "patch transaction",
                "pagination/cursor/progress/completion",
                "path canonicalization and protection",
            ],
            "rules": [
                "Return semantic file choices only; never return patch operations or SHA values.",
                "Use src/main/resources/fabric.mod.json for Fabric metadata.",
                "Split large implementations into cohesive files instead of one giant source file.",
                "Do not invent shell scripts or delete files.",
            ],
        }
        plan_messages = [
            {
                "role": "system",
                "content": (
                    "You are the Minecraft implementation planner. Decide only which project "
                    "text files are needed and why. Host code owns all patch mechanics, state, "
                    "SHA checks and completion. Use the supplied exact-source/research evidence."
                ),
            },
            {"role": "user", "content": json.dumps(plan_request, ensure_ascii=False)},
        ]
        plan_payload = self.router.generate_tool_decision(
            "coder",
            plan_messages,
            tool_name="return_custom_module_file_plan",
            parameters=_file_plan_schema(),
            description="Return the semantic file plan for one approved Minecraft module.",
        )
        planned_files, runtime_tests = self._validate_file_plan(plan_payload)

        operations: list[dict[str, Any]] = []
        touched_paths: list[str] = []
        for planned in planned_files:
            path = planned["path"]
            purpose = planned["purpose"]
            target = root / Path(path)
            if target.is_symlink():
                raise CustomModuleGenerationError(
                    f"Custom module target may not replace a symlink: {path}"
                )

            existed = target.exists()
            if existed and not target.is_file():
                raise CustomModuleGenerationError(
                    f"Custom module target is not a regular text file: {path}"
                )
            expected_sha256 = ""
            if existed:
                expected_sha256 = "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()

            file_context = _select_file_context(
                observation_ledger,
                path=path,
                purpose=purpose,
                byte_budget=project_context_budget,
            )
            file_request = {
                "phase": "write_file",
                "target": {
                    "minecraft_version": minecraft_version,
                    "loader": loader,
                    "mappings": mappings,
                    "java": java_version,
                },
                "module": {
                    "module_id": module.module_id,
                    "kind": module.kind,
                    "config": module.config,
                    "depends_on": list(module.depends_on),
                },
                "path": path,
                "purpose": purpose,
                "existing_file": existed,
                "exact_source_context": file_context,
                "research_context": research_context,
                "host_grounding": host_grounding,
                "prior_generated_files": _patch_operation_receipt(operations),
                "rules": [
                    "Return the complete final UTF-8 text for this one host-selected file.",
                    "Do not return a path, operation, SHA, cursor, completion flag or patch wrapper.",
                    "For JSON resources, content is still one text string containing the JSON file.",
                    "Use only the selected Minecraft/loader/mappings/Java target.",
                    "Keep behavior server-authoritative and persistent where the module requires it.",
                ],
            }
            file_messages = [
                {
                    "role": "system",
                    "content": (
                        "Write exactly one host-selected project text file. The host owns file "
                        "identity and every patch/state decision. Your only required output is "
                        "the complete file text plus optional observable runtime tests."
                    ),
                },
                {"role": "user", "content": json.dumps(file_request, ensure_ascii=False)},
            ]
            file_payload = self.router.generate_tool_decision(
                "coder",
                file_messages,
                tool_name="return_custom_module_file_content",
                parameters=_file_content_schema(),
                description="Return complete UTF-8 text for one host-selected project file.",
            )
            content, file_tests = _validate_file_content_payload(file_payload)
            runtime_tests.extend(file_tests)

            operation: dict[str, Any]
            if existed:
                operation = {
                    "operation": "replace",
                    "path": path,
                    "expected_sha256": expected_sha256,
                    "content": content,
                }
            else:
                operation = {
                    "operation": "create",
                    "path": path,
                    "content": content,
                }
            self._validate_operations([operation])
            operations.append(operation)
            touched_paths.append(path)

        if not operations:
            raise CustomModuleGenerationError(
                f"Custom module {module.module_id!r} failed to produce any planned source files."
            )

        self._validate_total_patch_bytes(operations)
        if not runtime_tests:
            runtime_tests = ["Verify mod functionality and compilation without crash."]
        runtime_tests = list(dict.fromkeys(test for test in runtime_tests if test.strip()))

        receipt = TransactionalSourcePatcher(root).apply(operations)
        if self._cached_index is not None:
            try:
                self._cached_index.update_files(touched_paths)
            except (OSError, ValueError):
                self._cached_index = ProjectIndex(root, policy=self.policy)
        else:
            self._cached_index = ProjectIndex(root, policy=self.policy)
        self._cached_root = root
        self._cached_index.write_manifest()
        return {
            "schema_version": "mmm/custom-module-result-v3",
            "module_id": module.module_id,
            "kind": module.kind,
            "status": "SOURCE_GENERATED",
            "patch_receipt": receipt,
            "operation_count": len(operations),
            "runtime_tests": runtime_tests,
            "source_observation_receipt": observation_ledger["receipt"],
            "touched_paths": touched_paths,
            "required_gates": ["JDT", "Gradle", "GameTest", *module.required_gates],
        }

    def _validate_file_plan(
        self,
        payload: dict[str, Any],
    ) -> tuple[list[dict[str, str]], list[str]]:
        if not isinstance(payload, dict):
            raise CustomModuleGenerationError("Custom-module file plan must be an object.")
        raw_files = payload.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise CustomModuleGenerationError(
                "Custom-module file plan must contain at least one file."
            )
        planned: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in raw_files:
            if not isinstance(item, dict):
                raise CustomModuleGenerationError("Custom-module planned file must be an object.")
            path = _canonicalize_planned_path(item.get("path"))
            purpose = str(item.get("purpose", "")).strip()
            if not purpose:
                raise CustomModuleGenerationError(
                    f"Custom-module planned file needs a non-empty purpose: {path}"
                )
            if custom_module_path_protected(path):
                raise CustomModuleGenerationError(
                    "Model file planning may not modify the code-owned research/context ledgers."
                )
            if not custom_module_path_allowed(path):
                raise CustomModuleGenerationError(
                    f"Custom module path is outside the allowed scope: {path}"
                )
            if path not in seen:
                seen.add(path)
                planned.append({"path": path, "purpose": purpose})

        tests_value = payload.get("runtime_tests", [])
        if not isinstance(tests_value, list) or any(not isinstance(v, str) for v in tests_value):
            raise CustomModuleGenerationError("File plan runtime_tests must be a list of strings.")
        return planned, [value.strip() for value in tests_value if value.strip()]

    def _validate_operations(self, operations: list[dict[str, Any]]) -> None:
        for item in operations:
            if not isinstance(item, dict):
                raise CustomModuleGenerationError("Patch operation must be an object.")
            if item.get("operation") not in {"create", "replace", "edit"}:
                raise CustomModuleGenerationError("Custom module may not delete files.")
            path = _normalized_operation_path(item)
            if custom_module_path_protected(path):
                raise CustomModuleGenerationError(
                    "Model patches may not modify the code-owned research ledger or "
                    "context-observation ledger."
                )
            if not custom_module_path_allowed(path):
                raise CustomModuleGenerationError(
                    f"Custom module path is outside the allowed scope: {path}"
                )

    def _validate_total_patch_bytes(self, operations: list[dict[str, Any]]) -> None:
        size = len(json.dumps(operations, ensure_ascii=False).encode("utf-8"))
        if size > self.policy.max_patch_bytes:
            raise CustomModuleGenerationError(
                "Custom module patch exceeds MMM_MAX_PATCH_BYTES; raise the explicit host "
                "resource policy or split the feature into dependency-linked modules."
            )


def _file_plan_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["files", "runtime_tests"],
        "properties": {
            "files": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["path", "purpose"],
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "purpose": {"type": "string", "minLength": 1},
                    },
                },
            },
            "runtime_tests": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }


def _file_content_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["content", "runtime_tests"],
        "properties": {
            "content": {"type": "string"},
            "runtime_tests": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }


def _validate_file_content_payload(payload: dict[str, Any]) -> tuple[str, list[str]]:
    if not isinstance(payload, dict):
        raise CustomModuleGenerationError("Custom-module file content must be an object.")
    content = payload.get("content")
    if not isinstance(content, str):
        raise CustomModuleGenerationError("Custom-module file content must be UTF-8 text.")
    tests = payload.get("runtime_tests", [])
    if not isinstance(tests, list) or any(not isinstance(value, str) for value in tests):
        raise CustomModuleGenerationError("File runtime_tests must be a list of strings.")
    return content, [value.strip() for value in tests if value.strip()]


def _canonicalize_planned_path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        raise CustomModuleGenerationError("Custom-module planned path must not be empty.")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts:
        raise CustomModuleGenerationError(
            f"Custom-module planned path must remain project-relative: {raw}"
        )
    path = pure.as_posix()
    if path == _FABRIC_ROOT_METADATA_PATH:
        return _FABRIC_CANONICAL_METADATA_PATH
    return path


def _select_file_context(
    ledger: dict[str, Any],
    *,
    path: str,
    purpose: str,
    byte_budget: int,
) -> dict[str, Any]:
    records = list(ledger.get("records", []))
    query_tokens = {
        token.lower()
        for token in _OBSERVATION_TOKEN.findall(f"{path} {purpose}")
    }
    ranked = sorted(
        records,
        key=lambda record: (
            -_observation_score(record, query_tokens),
            0 if str(record.get("path", "")) == path else 1,
            str(record.get("path", "")),
            int(record.get("content_start_bytes", 0)),
        ),
    )
    selected: list[dict[str, Any]] = []
    base = {
        "schema_version": "mmm/file-source-context-v1",
        "ledger_receipt": ledger.get("receipt", {}),
        "target_path": path,
        "records": selected,
    }
    safe_budget = max(1024, byte_budget - 256)
    for record in ranked:
        selected.append(record)
        if _json_size(base) > safe_budget:
            selected.pop()
            break
    return base


# Legacy parsing/state helpers remain for compatibility tests and old persisted traces.
# Production generation above no longer calls them.
def _canonicalize_generation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CustomModuleGenerationError("Custom-module response must be a JSON object.")
    normalized = dict(payload)
    if "tests" in normalized and "runtime_tests" not in normalized:
        normalized["runtime_tests"] = normalized.pop("tests")
    if "cursor" in normalized and "next_cursor" not in normalized:
        normalized["next_cursor"] = normalized.pop("cursor")
    if "patch_operations" in normalized and "operations" not in normalized:
        normalized["operations"] = normalized.pop("patch_operations")
    if "patches" in normalized and "operations" not in normalized:
        normalized["operations"] = normalized.pop("patches")

    operations_value = normalized.get("operations", [])
    if not isinstance(operations_value, list):
        raise CustomModuleGenerationError("Response 'operations' must be a JSON list.")
    normalized["operations"] = [
        _canonicalize_custom_module_operation(item) for item in operations_value
    ]

    tests_value = normalized.get(
        "runtime_tests", ["Verify mod functionality and compilation without crash."]
    )
    if not isinstance(tests_value, list):
        raise CustomModuleGenerationError("Response 'runtime_tests' must be a JSON list.")
    normalized["runtime_tests"] = tests_value

    complete_value = normalized.get("complete", True)
    if not isinstance(complete_value, bool):
        raise CustomModuleGenerationError("Response 'complete' must be a JSON boolean.")
    normalized["complete"] = complete_value

    next_cursor_value = normalized.get("next_cursor", "")
    if next_cursor_value is None:
        next_cursor_value = ""
    if not isinstance(next_cursor_value, str):
        raise CustomModuleGenerationError("Response 'next_cursor' must be a JSON string.")
    normalized["next_cursor"] = next_cursor_value.strip()

    if "context_page_complete" in normalized:
        page_complete_value = normalized["context_page_complete"]
        if not isinstance(page_complete_value, bool):
            raise CustomModuleGenerationError(
                "Response 'context_page_complete' must be a JSON boolean."
            )
    else:
        page_complete_value = bool(normalized["operations"]) and not normalized["next_cursor"]
    normalized["context_page_complete"] = page_complete_value
    return normalized


def _generation_fragment_action(
    payload: dict[str, Any],
    *,
    is_last_page: bool,
    has_accumulated_operations: bool,
    current_cursor: str,
    seen_cursors: set[str],
) -> str:
    operations = payload["operations"]
    next_cursor = payload["next_cursor"]
    page_complete = payload["context_page_complete"]

    if next_cursor:
        if next_cursor == current_cursor or next_cursor in seen_cursors:
            raise CustomModuleGenerationError(
                "Custom-module response repeated next_cursor without protocol progress."
            )
        if page_complete:
            raise CustomModuleGenerationError(
                "Response cannot set context_page_complete=true while also returning an "
                "advancing next_cursor for the same observation page."
            )
        return "cursor"

    if page_complete:
        if is_last_page and not has_accumulated_operations:
            raise CustomModuleGenerationError(
                "Final observation page completed before any patch operation was accumulated."
            )
        return "page_complete"

    if operations:
        raise CustomModuleGenerationError(
            "Patch operations were returned with context_page_complete=false but without an "
            "advancing next_cursor; the response cannot make further progress."
        )

    keys_str = ", ".join(payload.keys())
    raise CustomModuleGenerationError(
        "Custom-module response fragment made no protocol progress: received object with keys "
        f"[{keys_str}] but no operations, advancing next_cursor, or "
        "context_page_complete=true transition."
    )


def _repair_generation_messages(
    generation_messages: list[dict[str, str]],
    error_reason: str,
) -> list[dict[str, str]]:
    return [
        *generation_messages,
        {
            "role": "user",
            "content": (
                "Execution & Validation Failure: the previous response failed host protocol "
                f"validation with reason: {error_reason}. The invalid assistant payload is "
                "intentionally omitted so you do not copy its shape. Repair only the JSON/patch/"
                "cursor transition for the host-selected target. A valid response must make "
                "progress using patch operations, a new next_cursor, or an explicit "
                "context_page_complete=true transition. Do not emit range-only {start,end} "
                "objects. Fabric metadata belongs at src/main/resources/fabric.mod.json. Do not "
                "retrieve new RAG/MCP evidence and do not change the approved feature scope. "
                "Return exactly one valid JSON object using only the supplied host evidence."
            ),
        },
    ]


def _canonicalize_custom_module_operation(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    canonical = dict(item)
    operation = str(canonical.get("operation", "")).strip().lower()
    path = _normalized_operation_path(canonical)
    if operation == "create" and path == _FABRIC_ROOT_METADATA_PATH:
        canonical["path"] = _FABRIC_CANONICAL_METADATA_PATH
    return canonical


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    if "<|channel|>" in cleaned:
        cleaned = cleaned.split("<|channel|>")[-1]
    cleaned_strip = cleaned.strip()
    if "```" in cleaned_strip:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_strip, re.DOTALL)
        if match:
            cleaned = match.group(1)

    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value

    curly_match = re.search(r"\{.*\}", text, re.DOTALL)
    if curly_match:
        try:
            value = json.loads(curly_match.group(0))
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    raise CustomModuleGenerationError(
        "Model output did not contain one parseable JSON object; refusing a fake complete fallback."
    )


def _is_stale_project_index_error(exc: ValueError) -> bool:
    return str(exc).startswith("Project source changed after its context index was built:")


def _collect_initial_observations(
    router: ModelRouter,
    index: ProjectIndex,
    *,
    query: str,
    byte_budget: int,
    diagnostic_paths: Iterable[str] = (),
) -> dict[str, Any]:
    del router
    records: list[dict[str, Any]] = []
    record_keys: set[tuple[str, int, int]] = set()
    source_page_digest = hashlib.sha256()
    cursor = ""
    seen_cursors: set[str] = set()
    page_count = 0
    project_sha256 = ""
    query_sha256 = ""

    while True:
        page = index.select_page(
            query=query,
            diagnostic_paths=diagnostic_paths,
            byte_budget=byte_budget,
            cursor=cursor,
        )
        if _json_size(page) > byte_budget:
            raise CustomModuleGenerationError("Host project context page exceeded its byte budget.")

        current_project_sha256 = str(page["project_sha256"])
        current_query_sha256 = str(page["query_sha256"])
        if page_count == 0:
            project_sha256 = current_project_sha256
            query_sha256 = current_query_sha256
        elif current_project_sha256 != project_sha256 or current_query_sha256 != query_sha256:
            raise CustomModuleGenerationError("Project context pagination changed its bound identity.")

        page_commitment = {
            "page_index": page["page_index"],
            "project_sha256": current_project_sha256,
            "query_sha256": current_query_sha256,
            "start_position": page["start_position"],
            "start_offset": page["start_offset"],
            "next_cursor": page["next_cursor"],
            "files": [
                {
                    "path": item["path"],
                    "sha256": item["sha256"],
                    "content_start_bytes": item["content_start_bytes"],
                    "content_end_bytes": item["content_end_bytes"],
                }
                for item in page["files"]
            ],
        }
        _update_digest(source_page_digest, page_commitment)

        for item in page.get("files", []):
            if isinstance(item, dict) and "path" in item and ("content" in item or "text" in item):
                content_str = str(item.get("content", item.get("text", "")))
                _append_observation(
                    records,
                    record_keys,
                    _exact_observation(
                        path=str(item["path"]),
                        sha256=str(item.get("sha256", "")),
                        start=int(item.get("content_start_bytes", 0)),
                        content=content_str.encode("utf-8"),
                        source_page=int(page.get("page_index", page_count)),
                    ),
                )

        page_count += 1
        if bool(page.get("complete", False)):
            break
        next_cursor = str(page.get("next_cursor", "")).strip()
        if not next_cursor or next_cursor == cursor or next_cursor in seen_cursors:
            raise CustomModuleGenerationError("Project context pagination made no progress.")
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    observation_digest = hashlib.sha256()
    for record in records:
        _update_digest(observation_digest, record)
    receipt = {
        "schema_version": "mmm/source-observation-receipt-v1",
        "project_sha256": project_sha256,
        "query_sha256": query_sha256,
        "source_page_count": page_count,
        "observation_count": len(records),
        "source_pages_sha256": "sha256:" + source_page_digest.hexdigest(),
        "observations_sha256": "sha256:" + observation_digest.hexdigest(),
        "policy": {"exact_source_quotes": True, "path_sha256_byte_range_bound": True},
    }
    return {
        "schema_version": "mmm/source-observation-ledger-v1",
        "receipt": receipt,
        "records": records,
    }


def _observation_context_pages(
    ledger: dict[str, Any],
    *,
    query: str,
    byte_budget: int,
) -> tuple[dict[str, Any], ...]:
    records = list(ledger["records"])
    query_tokens = {token.lower() for token in _OBSERVATION_TOKEN.findall(query)}
    ranked = sorted(
        records,
        key=lambda record: (
            -_observation_score(record, query_tokens),
            record["path"],
            record["content_start_bytes"],
            record["observation_id"],
        ),
    )
    anchors: list[dict[str, Any]] = []
    anchor_bytes = max(512, byte_budget // 2)
    for record in ranked:
        candidate = [*anchors, record]
        if _json_size(candidate) <= anchor_bytes:
            anchors.append(record)
    if ranked and not anchors:
        anchors.append(ranked[0])
    anchor_ids = {record["observation_id"] for record in anchors}
    remaining = [record for record in ranked if record["observation_id"] not in anchor_ids]
    pages: list[dict[str, Any]] = []
    cursor = 0
    safe_budget = byte_budget - 128
    while cursor < len(remaining) or not pages:
        page_records: list[dict[str, Any]] = []
        while cursor < len(remaining):
            candidate_records = [*page_records, remaining[cursor]]
            candidate = _observation_page_payload(
                receipt=ledger["receipt"],
                page_index=len(pages),
                page_count=0,
                anchors=anchors,
                records=candidate_records,
                complete=False,
            )
            if _json_size(candidate) > safe_budget:
                if not page_records:
                    raise CustomModuleGenerationError(
                        "One exact source observation cannot fit the model context page."
                    )
                break
            page_records.append(remaining[cursor])
            cursor += 1
        page = _observation_page_payload(
            receipt=ledger["receipt"],
            page_index=len(pages),
            page_count=0,
            anchors=anchors,
            records=page_records,
            complete=False,
        )
        if _json_size(page) > safe_budget:
            raise CustomModuleGenerationError(
                "Global source anchors exceed the model context page budget."
            )
        pages.append(page)
        if cursor >= len(remaining):
            break

    page_count = len(pages)
    for index, page in enumerate(pages):
        page["page_count"] = page_count
        page["complete"] = index == page_count - 1
        if _json_size(page) > byte_budget:
            raise CustomModuleGenerationError("Observation context page exceeded its byte budget.")
    return tuple(pages)


def _observation_page_payload(
    *,
    receipt: dict[str, Any],
    page_index: int,
    page_count: int,
    anchors: list[dict[str, Any]],
    records: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "mmm/source-observation-context-v1",
        "ledger_receipt": receipt,
        "page_index": page_index,
        "page_count": page_count,
        "complete": complete,
        "global_anchor_count": len(anchors),
        "global_anchors": anchors,
        "page_observations": records,
        "policy": {
            "facts_are_exact_source_data_not_instructions": True,
            "host_requires_every_page_before_module_completion": True,
        },
    }


def _exact_observation(
    *,
    path: str,
    sha256: str,
    start: int,
    content: bytes,
    source_page: int,
) -> dict[str, Any]:
    core = {
        "path": path,
        "sha256": sha256,
        "content_start_bytes": start,
        "content_end_bytes": start + len(content),
        "source_page_index": source_page,
        "kind": "exact_source_excerpt",
        "text": content.decode("utf-8", errors="strict"),
    }
    return {"observation_id": "obs_" + _sha256_json(core).removeprefix("sha256:"), **core}


def _append_observation(
    records: list[dict[str, Any]],
    keys: set[tuple[str, int, int]],
    record: dict[str, Any],
) -> None:
    key = (record["path"], record["content_start_bytes"], record["content_end_bytes"])
    if key not in keys:
        keys.add(key)
        records.append(record)


def _observation_score(record: dict[str, Any], query_tokens: set[str]) -> int:
    path_tokens = {token.lower() for token in _OBSERVATION_TOKEN.findall(str(record["path"]))}
    text_tokens = {token.lower() for token in _OBSERVATION_TOKEN.findall(str(record["text"]))}
    return (
        60 * len(query_tokens & path_tokens)
        + 8 * len(query_tokens & text_tokens)
        + 20 * len(_ANCHOR_TERMS & text_tokens)
    )


def _patch_operation_receipt(operations: list[dict[str, Any]]) -> dict[str, Any]:
    paths = [_normalized_operation_path(item) for item in operations]
    return {
        "schema_version": "mmm/prior-patch-receipt-v1",
        "operation_count": len(operations),
        "operations_sha256": _sha256_json(operations),
        "touched_path_count": len(paths),
        "touched_paths_sha256": _sha256_json(paths),
        "latest_touched_path": paths[-1] if paths else "",
    }


def _normalized_operation_path(item: dict[str, Any]) -> str:
    return PurePosixPath(str(item.get("path", "")).replace("\\", "/")).as_posix()


def _sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _update_digest(digest: Any, value: Any) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _json_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _mapping_namespace(value: str) -> str:
    lowered = value.strip().lower()
    if "intermediary" in lowered:
        return "intermediary"
    if "official" in lowered or "mojang" in lowered:
        return "official"
    return "yarn"


def _normalized_generation_failure(value: str) -> str:
    compact = " ".join(value.lower().split())
    compact = re.sub(r"0x[0-9a-f]+|[0-9]+", "#", compact)
    return compact[:2048]
