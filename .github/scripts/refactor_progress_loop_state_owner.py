from __future__ import annotations

import re
from pathlib import Path

CORE = Path("minecraft_mod_ai/progress_aware_tool_loop.py")
INIT = Path("minecraft_mod_ai/__init__.py")
OLD_WRAPPER = Path("minecraft_mod_ai/verifier_fail_closed_completion_installation.py")
OLD_TEST = Path("tests/test_verifier_fail_closed_completion_installation.py")
NEW_TEST = Path("tests/test_progress_aware_verifier_repair_state.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


text = CORE.read_text(encoding="utf-8")

text = replace_once(
    text,
    '    applied_mutations: list[str] = field(default_factory=list)\n'
    '    workspace_changed: bool = False\n'
    '    validation_status: str = "PENDING"\n'
    '    last_failure_digest: str | None = None\n',
    '    applied_mutations: list[str] = field(default_factory=list)\n'
    '    created_paths: set[str] = field(default_factory=set)\n'
    '    workspace_changed: bool = False\n'
    '    validation_status: str = "PENDING"\n'
    '    latest_verifier_tool: str | None = None\n'
    '    latest_verifier_errors: tuple[dict[str, Any], ...] = ()\n'
    '    latest_verifier_fingerprint: str | None = None\n'
    '    repair_guidance_fingerprint: str | None = None\n'
    '    last_failure_digest: str | None = None\n',
    "HostRunState fields",
)

new_methods = """    def record_mutation(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> bool:
        # Mutation, target authority, and verifier invalidation share one owner.
        if not mutation_payload_applied(tool_name, payload):
            return False

        operation = str(arguments.get("operation") or "").strip().casefold()
        path = ""
        for key in _SOURCE_EDIT_PATH_KEYS:
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                path = _canonical_mutation_path(value)
                break

        with self._lock:
            self.applied_mutations.append(tool_name)
            self.workspace_changed = True
            if path and operation in _SOURCE_CREATE_OPERATIONS:
                self.created_paths.add(path)

            context = self.mutation_context
            if context is not None and path == _canonical_mutation_path(context.target_path):
                source_body = context.source_body
                if operation in _SOURCE_CREATE_OPERATIONS:
                    content = arguments.get("content")
                    if isinstance(content, str):
                        source_body = content
                elif operation == "replace_exact" and isinstance(source_body, str):
                    old = arguments.get("old")
                    new = arguments.get("new")
                    if (
                        isinstance(old, str)
                        and isinstance(new, str)
                        and old
                        and source_body.count(old) == 1
                    ):
                        source_body = source_body.replace(old, new, 1)
                self.mutation_context = replace(
                    context,
                    source_body=source_body,
                    is_new_file=False,
                    evidence_source="mutation_receipt",
                )

            self.validation_status = "PENDING"
            self.latest_verifier_tool = None
            self.latest_verifier_errors = ()
            self.latest_verifier_fingerprint = None
            self.repair_guidance_fingerprint = None
        return True

    def record_verification(
        self,
        tool_name: str,
        payload: Mapping[str, Any],
        status: str,
    ) -> bool:
        from .validation_diagnostic_contract import diagnostic_errors

        result = payload.get("result")
        receipt = result if isinstance(result, Mapping) else payload
        errors: list[dict[str, Any]] = []
        if status == "FAIL":
            for item in diagnostic_errors(receipt):
                compact = {
                    key: item.get(key)
                    for key in (
                        "uri",
                        "path",
                        "file",
                        "severity",
                        "code",
                        "source",
                        "message",
                        "range",
                        "line",
                    )
                    if item.get(key) not in (None, "", [], {})
                }
                if compact:
                    errors.append(compact)

        fingerprint = evidence_fingerprint(
            {"tool": tool_name, "status": status, "errors": errors}
        )
        with self._lock:
            changed = (
                status != self.validation_status
                or fingerprint != self.latest_verifier_fingerprint
            )
            self.validation_status = status
            self.latest_verifier_tool = tool_name
            self.latest_verifier_errors = tuple(errors)
            self.latest_verifier_fingerprint = fingerprint
            if status != "FAIL":
                self.repair_guidance_fingerprint = None
            return changed

    def take_verifier_repair_guidance(self) -> str | None:
        with self._lock:
            if (
                self.validation_status != "FAIL"
                or not self.latest_verifier_fingerprint
                or self.latest_verifier_fingerprint == self.repair_guidance_fingerprint
            ):
                return None
            self.repair_guidance_fingerprint = self.latest_verifier_fingerprint
            context = self.mutation_context
            evidence = {
                "verifier": self.latest_verifier_tool,
                "diagnostics": list(self.latest_verifier_errors),
                "target_path": context.target_path if context is not None else None,
                "target_is_new_file": context.is_new_file if context is not None else None,
                "paths_created_in_this_run": sorted(self.created_paths),
            }

        return (
            "MMM_CORE_VERIFIER_REPAIR_V1\\n"
            "The verifier still reports source defects. Repair the existing target; "
            "do not restart generation and do not recreate any path that already has "
            "an APPLIED mutation receipt. Make a material existing-file source edit "
            "that addresses the diagnostics, then allow VERIFY to run again.\\n"
            + json.dumps(evidence, ensure_ascii=False, sort_keys=True, default=str)
        )

"""

pattern = re.compile(
    r"    def record_mutation\(self, tool_name: str, payload: Mapping\[str, Any\]\) -> bool:\n"
    r".*?"
    r"(?=    def record_failure\()",
    re.DOTALL,
)
text, count = pattern.subn(new_methods, text, count=1)
if count != 1:
    raise SystemExit(f"record_mutation method block: expected one match, found {count}")

text = replace_once(
    text,
    '                mutation_applied = state.record_mutation(call.name, payload)\n',
    '                mutation_applied = state.record_mutation(call.name, call.arguments, payload)\n',
    "mutation args",
)

text = replace_once(
    text,
    '                if status != state.validation_status:\n'
    '                    state.validation_status = status\n'
    '                    turn_made_progress = True\n'
    '                if status == "FAIL" and implementation_requires_mutation:\n',
    '                if state.record_verification(call.name, payload, status):\n'
    '                    turn_made_progress = True\n'
    '                if status == "FAIL" and implementation_requires_mutation:\n',
    "verification owner",
)

text = replace_once(
    text,
    '        turn_request = replace(\n'
    '            request,\n'
    '            tools=phase_tools,\n'
    '            tool_choice=tool_choice,\n'
    '            parallel_tool_calls=parallel_tool_calls,\n'
    '        )\n',
    '        if implementation_requires_mutation and state.phase == LoopPhase.ACT:\n'
    '            repair_guidance = state.take_verifier_repair_guidance()\n'
    '            if repair_guidance is not None:\n'
    '                messages.append({"role": "system", "content": repair_guidance})\n'
    '\n'
    '        turn_request = replace(\n'
    '            request,\n'
    '            tools=phase_tools,\n'
    '            tool_choice=tool_choice,\n'
    '            parallel_tool_calls=parallel_tool_calls,\n'
    '        )\n',
    "repair guidance",
)

text = replace_once(
    text,
    '            if implementation_requires_mutation and not state.workspace_changed and not mutation_history_applied(messages):\n'
    '                if state.phase == LoopPhase.OBSERVE and is_mutation_ready(messages, state):\n',
    '            if implementation_requires_mutation and state.validation_status == "FAIL":\n'
    '                state.phase = LoopPhase.ACT\n'
    '                repeated = state.record_no_progress_result(\n'
    '                    {\n'
    '                        "phase": LoopPhase.ACT.value,\n'
    '                        "validation_status": "FAIL",\n'
    '                        "verifier_fingerprint": state.latest_verifier_fingerprint,\n'
    '                        "prose": content,\n'
    '                    }\n'
    '                )\n'
    '                if repeated > 1:\n'
    '                    raise ModelConfigurationError(\n'
    '                        "VERIFICATION_REPAIR_FIXED_POINT: coder repeated the same prose-only "\n'
    '                        "state while trustworthy verifier diagnostics remain unresolved."\n'
    '                    )\n'
    '                messages.extend([\n'
    '                    {"role": "assistant", "content": content},\n'
    '                    {\n'
    '                        "role": "system",\n'
    '                        "content": (\n'
    '                            "Verifier status remains FAIL. Prose cannot complete this implementation. "\n'
    '                            "Use the exposed mutation tool to materially repair the existing target, "\n'
    '                            "then allow host verification to run again."\n'
    '                        ),\n'
    '                    },\n'
    '                ])\n'
    '                continue\n'
    '            if implementation_requires_mutation and not state.workspace_changed and not mutation_history_applied(messages):\n'
    '                if state.phase == LoopPhase.OBSERVE and is_mutation_ready(messages, state):\n',
    "prose fail-close",
)

text = replace_once(
    text,
    '            if implementation_requires_mutation and not state.workspace_changed and not mutation_history_applied(messages):\n'
    '                raise ModelConfigurationError(\n'
    '                    "Writable coder reached the host tool-round limit before a "\n'
    '                    "reviewed source mutation was applied; refusing a prose-only implementation."\n'
    '                )\n'
    '            return _finalize_without_tools(\n',
    '            if implementation_requires_mutation and not state.workspace_changed and not mutation_history_applied(messages):\n'
    '                raise ModelConfigurationError(\n'
    '                    "Writable coder reached the host tool-round limit before a "\n'
    '                    "reviewed source mutation was applied; refusing a prose-only implementation."\n'
    '                )\n'
    '            if implementation_requires_mutation and state.validation_status == "FAIL":\n'
    '                raise ModelConfigurationError(\n'
    '                    "VERIFICATION_REPAIR_FIXED_POINT: host execution boundary reached while "\n'
    '                    "trustworthy verifier evidence still reports source defects."\n'
    '                )\n'
    '            return _finalize_without_tools(\n',
    "round fail-close",
)

text = replace_once(
    text,
    '            if implementation_requires_mutation and not state.workspace_changed and not mutation_history_applied(messages):\n'
    '                raise ModelConfigurationError(\n'
    '                    "Writable coder reached a no-progress boundary before a reviewed source "\n'
    '                    f"mutation was applied; refusing a prose-only implementation.{reason_suffix}\\n"\n'
    '                    f"Execution trajectory:\\n{traj_summary}"\n'
    '                )\n'
    '            return _finalize_without_tools(\n',
    '            if implementation_requires_mutation and not state.workspace_changed and not mutation_history_applied(messages):\n'
    '                raise ModelConfigurationError(\n'
    '                    "Writable coder reached a no-progress boundary before a reviewed source "\n'
    '                    f"mutation was applied; refusing a prose-only implementation.{reason_suffix}\\n"\n'
    '                    f"Execution trajectory:\\n{traj_summary}"\n'
    '                )\n'
    '            if implementation_requires_mutation and state.validation_status == "FAIL":\n'
    '                raise ModelConfigurationError(\n'
    '                    "VERIFICATION_REPAIR_FIXED_POINT: the same source/action/result state "\n'
    '                    "recurred while trustworthy verifier diagnostics remain unresolved."\n'
    '                    f"{reason_suffix}\\nExecution trajectory:\\n{traj_summary}"\n'
    '                )\n'
    '            return _finalize_without_tools(\n',
    "fixed-point fail-close",
)

text = replace_once(
    text,
    '                    "validation_status": state.validation_status,\n'
    '                    "workspace_changed": state.workspace_changed,\n'
    '                    "applied_mutations": tuple(state.applied_mutations),\n'
    '                    "tool_results": _fixed_point_tool_results(executed),\n',
    '                    "validation_status": state.validation_status,\n'
    '                    "verifier_fingerprint": state.latest_verifier_fingerprint,\n'
    '                    "workspace_changed": state.workspace_changed,\n'
    '                    "applied_mutations": tuple(state.applied_mutations),\n'
    '                    "tool_results": _fixed_point_tool_results(executed),\n',
    "fixed-point identity",
)

CORE.write_text(text, encoding="utf-8")

init_text = INIT.read_text(encoding="utf-8")
init_text = replace_once(
    init_text,
    'from .verifier_fail_closed_completion_installation import (\n'
    '    install as install_verifier_fail_closed_completion,\n'
    ')\n',
    "",
    "wrapper import",
)
init_text = replace_once(
    init_text,
    '# Final runtime composition is now known; make completion fail closed without altering\n'
    '# the mutation/verifier dispatch wrappers composed by runtime_finalization.\n'
    'install_verifier_fail_closed_completion(_progress_aware_tool_loop)\n',
    "",
    "wrapper install",
)
INIT.write_text(init_text, encoding="utf-8")

for obsolete in (OLD_WRAPPER, OLD_TEST):
    if not obsolete.exists():
        raise SystemExit(f"expected obsolete wrapper artifact missing: {obsolete}")
    obsolete.unlink()

NEW_TEST.write_text(
    """from __future__ import annotations

from minecraft_mod_ai.progress_aware_tool_loop import (
    HostRunState,
    TargetMutationContext,
    _mutation_target_error,
)

PATH = "src/main/java/dev/mmm/debugfixture/DebugToken.java"


def _applied_receipt() -> dict:
    return {
        "ok": True,
        "result": {
            "schema_version": "mmm/source-patch-receipt-v1",
            "status": "APPLIED",
            "operations": [
                {
                    "path": PATH,
                    "before_sha256": None,
                    "after_sha256": "sha256:created",
                }
            ],
        },
    }


def _failed_diagnostics(message: str) -> dict:
    return {
        "ok": True,
        "result": {
            "status": "FAIL",
            "diagnostics": {
                f"file:///{PATH}": [
                    {
                        "severity": 1,
                        "code": "UndefinedType",
                        "message": message,
                    }
                ]
            },
        },
    }


def test_applied_create_converts_target_to_existing_file_authority():
    state = HostRunState(
        mutation_context=TargetMutationContext(
            target_path=PATH,
            target_symbol="DebugToken",
            is_new_file=True,
            evidence_source="task_capsule",
        )
    )
    args = {
        "operation": "create_file",
        "path": PATH,
        "content": "package dev.mmm.debugfixture; public class DebugToken {}",
    }
    assert state.record_mutation("apply_source_edit", args, _applied_receipt())
    assert state.mutation_context is not None
    assert state.mutation_context.is_new_file is False
    assert PATH in state.created_paths
    error = _mutation_target_error("apply_source_edit", args, state.mutation_context)
    assert error is not None
    assert error.startswith("MUTATION_TARGET_CREATION_CONFLICT")


def test_verifier_fail_and_diagnostics_are_owned_by_host_state():
    state = HostRunState(
        mutation_context=TargetMutationContext(
            target_path=PATH,
            target_symbol="DebugToken",
            source_body="class DebugToken {}",
            is_new_file=False,
            evidence_source="mutation_receipt",
        )
    )
    assert state.record_verification(
        "java_diagnostics",
        _failed_diagnostics("RegistryWrapper cannot be resolved to a type"),
        "FAIL",
    )
    assert state.validation_status == "FAIL"
    assert state.latest_verifier_tool == "java_diagnostics"
    assert state.latest_verifier_fingerprint
    assert state.latest_verifier_errors
    assert "RegistryWrapper cannot be resolved to a type" in str(state.latest_verifier_errors)


def test_repair_guidance_tracks_verifier_fingerprint_not_message_history():
    state = HostRunState(
        mutation_context=TargetMutationContext(
            target_path=PATH,
            target_symbol="DebugToken",
            source_body="class DebugToken {}",
            is_new_file=False,
            evidence_source="mutation_receipt",
        )
    )
    state.record_verification(
        "java_diagnostics",
        _failed_diagnostics("RegistryWrapper cannot be resolved to a type"),
        "FAIL",
    )
    first = state.take_verifier_repair_guidance()
    assert first is not None
    assert PATH in first
    assert "RegistryWrapper cannot be resolved to a type" in first
    assert state.take_verifier_repair_guidance() is None

    state.record_verification(
        "java_diagnostics",
        _failed_diagnostics("Item.Settings cannot be resolved to a type"),
        "FAIL",
    )
    second = state.take_verifier_repair_guidance()
    assert second is not None
    assert second != first
    assert "Item.Settings cannot be resolved to a type" in second


def test_real_edit_invalidates_stale_verifier_fail_and_updates_source_body():
    state = HostRunState(
        mutation_context=TargetMutationContext(
            target_path=PATH,
            target_symbol="DebugToken",
            source_body="RegistryWrapper value;",
            is_new_file=False,
            evidence_source="mutation_receipt",
        )
    )
    state.record_verification(
        "java_diagnostics",
        _failed_diagnostics("RegistryWrapper cannot be resolved to a type"),
        "FAIL",
    )
    args = {
        "operation": "replace_exact",
        "path": PATH,
        "old": "RegistryWrapper",
        "new": "RegistryEntry",
    }
    receipt = {
        "ok": True,
        "result": {
            "schema_version": "mmm/source-patch-receipt-v1",
            "status": "APPLIED",
            "operations": [
                {
                    "path": PATH,
                    "before_sha256": "sha256:old",
                    "after_sha256": "sha256:new",
                }
            ],
        },
    }
    assert state.record_mutation("apply_source_edit", args, receipt)
    assert state.validation_status == "PENDING"
    assert state.latest_verifier_fingerprint is None
    assert state.mutation_context is not None
    assert "RegistryEntry" in (state.mutation_context.source_body or "")
""",
    encoding="utf-8",
)

print("core verifier ownership migration prepared")
