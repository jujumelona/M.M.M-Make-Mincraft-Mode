from __future__ import annotations

"""Direct, whole-file custom-module generation.

Authored designs first compile to a responsibility/dependency implementation graph.
Each admitted source unit owns an exact host-selected file. The coder returns complete
source, and Gradle failures enter compiler repair. Output exhaustion instead returns to
graph decomposition and cannot retry the exhausted task. No patch transport is involved.
"""

import hashlib
import inspect
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .complete_spec import ProductionModule
from .custom_module_errors import CustomModuleGenerationError
from .generation_implementation_grounding import (
    build_generation_implementation_grounding,
    render_generation_implementation_authority_prompt,
)
from .host_grounding import custom_module_path_allowed
from .implementation_ir import OutputBudgetExhausted
from .llama_finish_reason_contract import OUTPUT_EXHAUSTED, completion_boundary_error
from .model_router import ModelRouter
from .platform_catalog import adapter_for_target, adapter_from_project
from .project_write_lock import project_write_lock
from .runner import GradleRunner
from .scale_policy import ScalePolicy
from .target_contract import TargetContractError, validate_target_coordinates

_SOURCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["content", "summary"],
    "properties": {
        "content": {"type": "string", "minLength": 1},
        "summary": {"type": "string"},
    },
}
_LOCATOR = re.compile(r"^(?P<path>[^#]+\.java)#(?P<symbol>[A-Za-z_$][A-Za-z0-9_$]*)$")
_PACKAGE = re.compile(r"(?m)^\s*package\s+([A-Za-z_$][A-Za-z0-9_$.]*)\s*;\s*$")
_PUBLIC_TYPE = re.compile(
    r"\bpublic\s+final\s+class\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\b"
)
_INITIALIZE = re.compile(
    r"\bpublic\s+static\s+void\s+initialize\s*\(\s*\)\s*(?:throws\s+[^{]+)?\{"
)
_SIDE_ONLY = re.compile(r"@Environment\s*\(\s*EnvType\.(?:CLIENT|SERVER)\s*\)")
_FORBIDDEN_ENTRYPOINT = re.compile(
    r"\b(?:implements\s+)?(?:ModInitializer|ClientModInitializer)\b"
)
_BODY_MARKER = "MMM_AUTHORED_FEATURE_BODY"

def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_project_path(value: Any) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    candidate = PurePosixPath(raw)
    if (
        not raw
        or candidate.is_absolute()
        or raw in {".", ".."}
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise CustomModuleGenerationError(
            f"Custom module target must be a safe project-relative path: {value!r}"
        )
    return candidate.as_posix()


def _task_local_module_contract(module: ProductionModule) -> dict[str, Any]:
    config = module.config if isinstance(module.config, dict) else {}
    task = config.get("evidence_task")
    if isinstance(task, Mapping):
        return dict(task)
    obligations = config.get("implementation_obligations")
    if not (
        isinstance(obligations, Sequence)
        and not isinstance(obligations, (str, bytes, bytearray))
    ):
        obligations = [f"Implement {module.module_id}"]
    return {
        "task_id": module.module_id,
        "semantic_outcome": str(config.get("semantic_outcome") or module.module_id),
        "implementation_obligations": list(obligations),
        "required_gates": list(module.required_gates),
    }


def _bounded_execution_feedback(value: Any) -> dict[str, Any] | None:
    """Normalize and deduplicate structured diagnostics without arbitrary truncation."""
    if not isinstance(value, Mapping):
        return None
    diagnostics = value.get("diagnostics")
    if not isinstance(diagnostics, Sequence) or isinstance(
        diagnostics, (str, bytes, bytearray)
    ):
        return None
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in diagnostics:
        if not isinstance(raw, Mapping):
            continue
        row = {
            "path": str(raw.get("path") or "").strip(),
            "code": str(raw.get("code") or "").strip(),
            "message": " ".join(str(raw.get("message") or "").split()),
        }
        key = (row["path"], row["code"], row["message"])
        if any(row.values()) and key not in seen:
            seen.add(key)
            rows.append(row)
    return {"diagnostics": rows} if rows else None


def _exact_target(module: ProductionModule) -> tuple[str, str, dict[str, Any]]:
    task = _task_local_module_contract(module)
    anchors = task.get("owned_anchors")
    candidates: list[tuple[str, str]] = []
    if isinstance(anchors, Sequence) and not isinstance(
        anchors, (str, bytes, bytearray)
    ):
        for anchor in anchors:
            if not isinstance(anchor, Mapping):
                continue
            match = _LOCATOR.fullmatch(str(anchor.get("locator") or "").strip())
            if match:
                candidates.append(
                    (
                        _normalize_project_path(match.group("path")),
                        match.group("symbol"),
                    )
                )
    unique = tuple(dict.fromkeys(candidates))
    if len(unique) != 1:
        raise CustomModuleGenerationError(
            "DIRECT_CODER_EXACT_TARGET_REQUIRED: each custom Java task must own "
            "exactly one host-selected Java file and top-level symbol."
        )
    path, symbol = unique[0]
    if not path.startswith("src/main/java/"):
        raise CustomModuleGenerationError(
            f"DIRECT_CODER_JAVA_TARGET_REQUIRED: {path}"
        )
    return path, symbol, task


def _resolve_generation_adapter(
    root: Path,
    *,
    minecraft_version: str | None,
    loader: str | None,
):
    requested_version = str(minecraft_version or "").strip()
    requested_loader = str(loader or "").strip()
    if requested_version and requested_loader:
        try:
            return adapter_for_target(requested_version, requested_loader)
        except ValueError as exc:
            raise CustomModuleGenerationError(str(exc)) from exc
    try:
        return adapter_from_project(root)
    except ValueError as exc:
        raise CustomModuleGenerationError(
            "Custom generation requires one unambiguous executable platform target."
        ) from exc


def _safe_target(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise CustomModuleGenerationError(
            f"Custom module target escapes the project root: {relative}"
        ) from exc
    if target.is_symlink():
        raise CustomModuleGenerationError(
            f"Custom module target may not be a symbolic link: {relative}"
        )
    return target


def _host_reserved(anchor: Mapping[str, Any]) -> bool:
    return str(anchor.get("status") or "").strip() in {
        "host_reserved",
        "host_owned",
        "new",
    }


def _target_anchor(task: Mapping[str, Any], relative: str, symbol: str) -> Mapping[str, Any] | None:
    anchors = task.get("owned_anchors")
    if not isinstance(anchors, Sequence) or isinstance(
        anchors, (str, bytes, bytearray)
    ):
        return None
    locator = f"{relative}#{symbol}"
    for anchor in anchors:
        if isinstance(anchor, Mapping) and str(anchor.get("locator") or "").strip() == locator:
            return anchor
    return None


def _package_from_java_target(relative: str) -> str:
    prefix = "src/main/java/"
    if not relative.startswith(prefix) or not relative.endswith(".java"):
        raise CustomModuleGenerationError(
            f"DIRECT_CODER_JAVA_TARGET_REQUIRED: {relative}"
        )
    body = relative[len(prefix):]
    parent = PurePosixPath(body).parent
    if str(parent) in {"", "."}:
        return ""
    return ".".join(parent.parts)


def _materialize_host_scaffold(
    target: Path,
    *,
    relative: str,
    symbol: str,
    task: Mapping[str, Any],
) -> str:
    anchor = _target_anchor(task, relative, symbol)
    if anchor is None or not _host_reserved(anchor):
        raise CustomModuleGenerationError(
            f"DIRECT_CODER_HOST_SCAFFOLD_MISSING: {relative}"
        )
    if not custom_module_path_allowed(relative):
        raise CustomModuleGenerationError(
            f"DIRECT_CODER_TARGET_OUTSIDE_WRITE_SCOPE: {relative}"
        )
    package_name = _package_from_java_target(relative)
    package_line = f"package {package_name};\n\n" if package_name else ""
    source = (
        package_line
        + f"public final class {symbol} {{\n"
        + f"    private {symbol}() {{}}\n\n"
        + "    // MMM_AUTHORED_FEATURE_BODY\n"
        + "}\n"
    )
    _atomic_write(target, source)
    return source


def _atomic_write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = content.encode("utf-8") if isinstance(content, str) else content
    temporary = path.with_name(
        f".{path.name}.mmm-{os.getpid()}-"
        f"{hashlib.sha256(raw).hexdigest()}.tmp"
    )
    try:
        temporary.write_bytes(raw)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _project_context(root: Path, target: Path, *, relevance_text: str) -> str:
    """Return only project sources explicitly referenced by the active task/scaffold."""
    java_root = root / "src/main/java"
    if not java_root.is_dir():
        return ""

    evidence = relevance_text
    if target.is_file() and not target.is_symlink():
        try:
            evidence += "\n" + target.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            pass

    candidates = [
        path
        for path in java_root.rglob("*.java")
        if path.is_file() and not path.is_symlink() and path.resolve() != target
    ]
    selected = [
        path
        for path in candidates
        if re.search(rf"\b{re.escape(path.stem)}\b", evidence)
    ]
    selected.sort(key=lambda path: path.relative_to(root).as_posix())

    rendered: list[str] = []
    for path in selected:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        relative = path.relative_to(root).as_posix()
        rendered.append(f"\n--- {relative} ---\n{text}\n")
    return "".join(rendered)


def _source_invariant_errors(
    source: str,
    *,
    symbol: str,
    expected_package: str,
    require_initialize: bool,
) -> tuple[str, ...]:
    errors: list[str] = []
    if "```" in source:
        errors.append("source contains Markdown code fences")
    package_match = _PACKAGE.search(source)
    if expected_package and (
        package_match is None or package_match.group(1) != expected_package
    ):
        errors.append(f"package must remain exactly {expected_package}")
    public = _PUBLIC_TYPE.findall(source)
    if public != [symbol]:
        errors.append(
            f"top-level contract must be exactly `public final class {symbol}`"
        )
    if require_initialize and not _INITIALIZE.search(source):
        errors.append("required `public static void initialize()` is missing")
    if _SIDE_ONLY.search(source):
        errors.append(
            "common authored source may not carry @Environment(CLIENT/SERVER)"
        )
    if _FORBIDDEN_ENTRYPOINT.search(source):
        errors.append(
            "feature source may not implement or declare a Fabric mod entrypoint"
        )
    if _BODY_MARKER in source:
        errors.append("host implementation marker was not replaced")
    return tuple(errors)


def _response_payload(text: str) -> dict[str, str]:
    raw = str(text or "").strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CustomModuleGenerationError(
            "DIRECT_CODER_INVALID_RESPONSE: coder must return one JSON object "
            "with complete `content` and `summary` strings."
        ) from exc
    if not isinstance(value, Mapping):
        raise CustomModuleGenerationError(
            "DIRECT_CODER_INVALID_RESPONSE: coder response is not an object."
        )
    if set(value) != {"content", "summary"}:
        raise CustomModuleGenerationError(
            "DIRECT_CODER_INVALID_RESPONSE: response must contain exactly "
            "`content` and `summary`."
        )
    content = value.get("content")
    summary = value.get("summary")
    if not isinstance(content, str) or not content.strip():
        raise CustomModuleGenerationError(
            "DIRECT_CODER_INVALID_RESPONSE: complete Java `content` is required."
        )
    if not isinstance(summary, str):
        raise CustomModuleGenerationError(
            "DIRECT_CODER_INVALID_RESPONSE: `summary` must be a string."
        )
    return {"content": content, "summary": summary}


def _supports_kwarg(callable_value: Any, name: str) -> bool:
    try:
        signature = inspect.signature(callable_value)
    except (TypeError, ValueError):
        return True
    return name in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _call_coder(
    router: Any,
    messages: Sequence[Mapping[str, str]],
) -> dict[str, str]:
    callback = getattr(router, "generate_text", None)
    if not callable(callback):
        raise CustomModuleGenerationError(
            "DIRECT_CODER_ROUTER_REQUIRED: router has no generate_text()."
        )
    kwargs: dict[str, Any] = {}
    for key, value in (
        ("response_format", "json"),
        ("response_schema", _SOURCE_SCHEMA),
        ("enable_tools", False),
        ("tool_stage", "generation"),
    ):
        if _supports_kwarg(callback, key):
            kwargs[key] = value
    try:
        text = callback("coder", messages, **kwargs)
    except Exception as exc:
        boundary = completion_boundary_error(exc)
        if boundary is not None and boundary.kind == OUTPUT_EXHAUSTED:
            raise OutputBudgetExhausted(
                "OUTPUT_BUDGET_EXHAUSTED: return to implementation decomposition; "
                f"completion_tokens={boundary.completion_tokens}, max_tokens={boundary.max_tokens}"
            ) from exc
        raise
    return _response_payload(text)


def _compile_log(report: Any) -> str:
    """Extract compiler diagnostics by structure instead of truncating raw logs."""
    fallback = str(getattr(report, "error", "") or "")
    commands = tuple(getattr(report, "commands", ()) or ())
    if not commands:
        return fallback
    path = Path(str(getattr(commands[-1], "log_path", "") or ""))
    if not path.is_file() or path.is_symlink():
        return fallback
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return fallback

    lines = text.splitlines()
    header = re.compile(r"(?i)(?:\.java:\d+:\s*(?:error|warning):|^\s*(?:error|failure):)")
    stop = re.compile(r"^\s*(?:> Task |BUILD (?:FAILED|SUCCESSFUL)|FAILURE:)")
    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        if header.search(line):
            if current:
                blocks.append("\n".join(current).strip())
            current = [line]
            continue
        if current:
            if stop.search(line):
                blocks.append("\n".join(current).strip())
                current = []
            elif line.strip():
                current.append(line)
            else:
                blocks.append("\n".join(current).strip())
                current = []
    if current:
        blocks.append("\n".join(current).strip())

    diagnostics = "\n\n".join(block for block in blocks if block)
    return diagnostics or fallback or text


def _compile_failure_measure(log: str) -> tuple[int, int]:
    """Well-founded source-repair measure; smaller is objectively closer to compile."""
    error_lines = {
        line.strip()
        for line in log.splitlines()
        if "error" in line.casefold() and line.strip()
    }
    return (0, len(error_lines) or 1)


class CustomModuleGenerator:
    """One exact task -> one complete source file -> compiler-guided repair loop."""

    def __init__(
        self,
        router: ModelRouter,
        *,
        policy: ScalePolicy | None = None,
        fast_mode: bool = False,
        project_index: Any | None = None,
        checkpoint_root: str | Path | None = None,
    ) -> None:
        self.router = router
        self.policy = policy or ScalePolicy.from_environment()
        self.fast_mode = bool(fast_mode)
        self._cached_index = project_index
        self._cached_root = (
            Path(project_index.root).resolve()
            if project_index is not None and getattr(project_index, "root", None)
            else None
        )
        self._checkpoint_root = (
            Path(checkpoint_root).expanduser().resolve()
            if checkpoint_root is not None
            else None
        )

    def _cache_dir(self, root: Path) -> Path:
        if self._checkpoint_root is not None:
            run_root = self._checkpoint_root.parent.parent
            if run_root != self._checkpoint_root:
                return run_root / ".cache" / "gradle"
        return root / ".minecraft_ai" / "gradle-cache"

    def generate(
        self,
        project_root: str | Path,
        *,
        module: ProductionModule,
        research_modules: Iterable[ProductionModule] = (),
        minecraft_version: str | None = None,
        loader: str | None = None,
        mappings: str | None = None,
        execution_feedback: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if "implementation_graph_request" in module.config:
            from .implementation_graph_execution import execute_implementation_graph

            return execute_implementation_graph(
                self, project_root, module=module, execution_feedback=execution_feedback,
            )
        del research_modules
        module.validate(policy=self.policy)
        root = Path(project_root).expanduser().resolve()
        if not root.is_dir() or root.is_symlink():
            raise CustomModuleGenerationError(
                "Custom module target must be a regular project directory."
            )

        bind_workspace = getattr(self.router, "bind_agent_workspace", None)
        if callable(bind_workspace):
            if _supports_kwarg(bind_workspace, "require_fresh_evidence"):
                bind_workspace(root, require_fresh_evidence=True)
            else:
                bind_workspace(root)

        adapter = _resolve_generation_adapter(
            root,
            minecraft_version=minecraft_version,
            loader=loader,
        )
        requested_mappings = (
            str(getattr(adapter, "yarn_mappings", "") or "")
            if mappings is None
            else str(mappings).strip()
        )
        try:
            coordinates = validate_target_coordinates(
                adapter.minecraft_version,
                adapter.loader,
                requested_mappings,
                declared_mappings_applicable=getattr(
                    adapter, "mappings_applicable", None
                ),
            )
        except TargetContractError as exc:
            raise CustomModuleGenerationError(str(exc)) from exc
        adapter_mapping = str(getattr(adapter, "yarn_mappings", "") or "")
        if (
            coordinates.mappings_applicable
            and adapter_mapping
            and coordinates.mappings != adapter_mapping
        ):
            raise CustomModuleGenerationError(
                "TARGET_MAPPINGS_ALIAS: requested mappings disagree with the "
                "resolved executable platform target."
            )

        relative, symbol, task = _exact_target(module)
        target = _safe_target(root, relative)
        target_existed = target.is_file()
        if target.exists() and not target_existed:
            raise CustomModuleGenerationError(
                f"DIRECT_CODER_TARGET_NOT_REGULAR: {relative}"
            )
        if target_existed:
            try:
                original_bytes = target.read_bytes()
                original = target.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise CustomModuleGenerationError(
                    f"Could not read host-owned source {relative}: {exc}"
                ) from exc
        else:
            original = _materialize_host_scaffold(
                target,
                relative=relative,
                symbol=symbol,
                task=task,
            )

        package_match = _PACKAGE.search(original)
        expected_package = package_match.group(1) if package_match else ""
        require_initialize = bool(_INITIALIZE.search(original))
        grounding = build_generation_implementation_grounding(
            module,
            minecraft_version=adapter.minecraft_version,
        )
        authority_prompt = render_generation_implementation_authority_prompt(
            grounding
        )
        task_text = json.dumps(task, ensure_ascii=False, sort_keys=True)
        grounding_text = json.dumps(
            grounding or {},
            ensure_ascii=False,
            sort_keys=True,
        )
        ir_contract = module.config.get("implementation_ir_node")
        if isinstance(ir_contract, Mapping):
            context = str(module.config.get("implementation_dependency_context") or "")
        else:
            context = _project_context(
                root,
                target,
                relevance_text=task_text + "\n" + grounding_text + "\n" + original,
            )
        bounded_feedback = _bounded_execution_feedback(execution_feedback)
        system = (
            "You implement exactly one host-owned Minecraft Java source file. "
            "Return one JSON object only: {\"content\": "
            "\"<complete Java file>\", \"summary\": "
            "\"<short summary>\"}. Never return a patch or diff. "
            "Do not change the package, public final top-level class name, or "
            "declared public API (including initialize() when required). Do not create "
            "another mod entrypoint. The host will compile the real project and "
            "return the exact compiler failure for repair."
            + ("\n\n" + authority_prompt if authority_prompt else "")
        )
        initial_user = (
            f"Platform: Minecraft {adapter.minecraft_version}; "
            f"loader {adapter.loader}; Java {adapter.java_version}; "
            f"mappings {adapter.yarn_mappings or '<none>'}.\n"
            f"Exact target: {relative}#{symbol}\n\n"
            f"Approved task:\n{task_text}\n\n"
            f"Host implementation grounding:\n{grounding_text}\n\n"
            f"Current host scaffold:\n{original}\n\n"
            f"Relevant existing project source:\n{context or '<none>'}"
        )
        if bounded_feedback:
            initial_user += (
                "\n\nDownstream execution feedback from the previous attempt:\n"
                + json.dumps(
                    bounded_feedback,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

        before_sha = _sha256_text(original)
        current = original
        summary = ""
        last_failure = ""
        compiler = GradleRunner(self._cache_dir(root))
        attempt = 0
        best_failure_measure: tuple[int, int] | None = None
        seen_candidates: set[str] = set()

        with project_write_lock(root):
            while True:
                attempt += 1
                if attempt == 1:
                    messages: list[dict[str, str]] = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": initial_user},
                    ]
                else:
                    messages = [
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": (
                                f"Repair pass {attempt} for "
                                f"{relative}#{symbol}.\n"
                                "Return the complete corrected Java file, "
                                "not a patch.\n\n"
                                f"Approved task:\n{task_text}\n\n"
                                f"Host implementation grounding:\n{grounding_text}\n\n"
                                f"Current complete source:\n{current}\n\n"
                                "Exact validation/compiler failure:\n"
                                f"{last_failure}"
                            ),
                        },
                    ]

                try:
                    payload = _call_coder(self.router, messages)
                except OutputBudgetExhausted:
                    if target_existed:
                        _atomic_write(target, original_bytes)
                    else:
                        target.unlink(missing_ok=True)
                    raise
                except Exception as exc:  # noqa: BLE001 - output exhaustion handled above
                    last_failure = (
                        "DIRECT_CODER_RESPONSE_FAILED: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    break
                candidate = (
                    payload["content"]
                    .replace("\r\n", "\n")
                    .replace("\r", "\n")
                )
                summary = payload["summary"]
                invariant_errors = _source_invariant_errors(
                    candidate,
                    symbol=symbol,
                    expected_package=expected_package,
                    require_initialize=require_initialize,
                )
                if isinstance(ir_contract, Mapping):
                    from .implementation_graph_execution import public_api_errors
                    invariant_errors += public_api_errors(candidate, ir_contract)
                candidate_sha = _sha256_text(candidate)
                if invariant_errors:
                    current = candidate
                    last_failure = "\n".join(
                        f"- {error}" for error in invariant_errors
                    )
                    measure = (1, len(set(invariant_errors)))
                    if (
                        candidate_sha in seen_candidates
                        or (
                            best_failure_measure is not None
                            and measure >= best_failure_measure
                        )
                    ):
                        break
                    seen_candidates.add(candidate_sha)
                    best_failure_measure = measure
                    continue

                _atomic_write(target, candidate)
                current = candidate
                report = compiler.compile_java(root)
                if getattr(report, "status", "") == "PASS":
                    after_sha = _sha256_text(candidate)
                    return {
                        "schema_version": "mmm/custom-module-result-v3",
                        "module_id": module.module_id,
                        "kind": module.kind,
                        "status": "SOURCE_GENERATED",
                        "patch_receipt": {
                            "schema_version": "mmm/direct-source-write-v1",
                            "status": "APPLIED",
                            "operations": [
                                {
                                    "operation": "replace" if target_existed else "create",
                                    "path": relative,
                                    "before_sha256": before_sha if target_existed else "",
                                    "after_sha256": after_sha,
                                }
                            ],
                            "touched_paths": [relative],
                        },
                        "operation_count": 1,
                        "runtime_tests": [
                            "Build the real project and execute the requested GameTest/runtime gates."
                        ],
                        "source_observation_receipt": {
                            "path": relative,
                            "sha256": before_sha,
                        },
                        "touched_paths": [relative],
                        "discarded_out_of_scope_paths": [],
                        "agent_summary": summary.strip(),
                        "generation_verification": {
                            "status": "PASS",
                            "mode": "gradle_compile_java",
                            "target_path": relative,
                            "attempt": attempt,
                        },
                        "output_exhaustion_continuations": 0,
                        "generation_checkpoint_resumed": False,
                        "required_gates": list(module.required_gates),
                    }

                last_failure = _compile_log(report) or str(
                    getattr(report, "error", "")
                    or "Gradle compileJava failed."
                )
                measure = _compile_failure_measure(last_failure)
                if (
                    candidate_sha in seen_candidates
                    or (
                        best_failure_measure is not None
                        and measure >= best_failure_measure
                    )
                ):
                    break
                seen_candidates.add(candidate_sha)
                best_failure_measure = measure

            if target_existed:
                _atomic_write(target, original_bytes)
            else:
                target.unlink(missing_ok=True)
            raise CustomModuleGenerationError(
                "DIRECT_CODER_COMPILE_FAILED: exact whole-file generation "
                f"stopped after repair evidence ceased to improve for {relative}. "
                "Last failure:\n"
                + last_failure
            )

    def ensure_generation_live_commit(
        self,
        result: Any,
        *,
        project_root: str | Path,
    ) -> bool:
        if not isinstance(result, Mapping):
            return False
        paths = result.get("touched_paths")
        receipt = result.get("patch_receipt")
        if (
            not isinstance(paths, Sequence)
            or isinstance(paths, (str, bytes, bytearray))
            or not paths
            or not isinstance(receipt, Mapping)
        ):
            return False
        operations = receipt.get("operations")
        if not isinstance(operations, Sequence) or len(operations) != len(paths):
            return False
        if any(not isinstance(path, str) for path in paths) or len(set(paths)) != len(paths):
            return False
        for path, operation in zip(paths, operations, strict=True):
            if not isinstance(operation, Mapping) or operation.get("path") != path:
                return False
            expected = str(operation.get("after_sha256") or "")
            try:
                target = _safe_target(
                    Path(project_root).expanduser().resolve(), _normalize_project_path(path),
                )
                content = target.read_text(encoding="utf-8")
            except (CustomModuleGenerationError, OSError, UnicodeError):
                return False
            if _sha256_text(content) != expected:
                return False
        return True

    def finalize_committed_generation_checkpoint(
        self,
        result: Any,
        *,
        project_root: str | Path,
    ) -> bool:
        return self.ensure_generation_live_commit(
            result,
            project_root=project_root,
        )

    def acknowledge_generation_checkpoint(self, result: Any) -> bool:
        return isinstance(result, Mapping)

    def release_generation_checkpoint(self, result: Any) -> bool:
        return isinstance(result, Mapping)

    def discard_generation_checkpoint(self, result: Any) -> bool:
        return isinstance(result, Mapping)


def _parse_coder_summary(text: str) -> str:
    return _response_payload(text)["summary"]


def _normalized_operation_path(item: Mapping[str, Any]) -> str:
    return PurePosixPath(
        str(item.get("path", "")).replace("\\", "/")
    ).as_posix()


_agent_mutable_path = custom_module_path_allowed

__all__ = [
    "CustomModuleGenerationError",
    "CustomModuleGenerator",
]
