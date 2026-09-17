from __future__ import annotations

"""Install semantic execution-feedback recovery before runtime finalization.

Two host-owned invariants live here:
* validation diagnostics for the deterministic Fabric project skeleton belong to the
  ``prepare-project`` work node when no generation receipt owns the failing path;
* feedback retries terminate on repeated evidence, not on an arbitrary retry count.

Verifier/toolchain infrastructure failures are a third, stricter boundary: they are
never source-repair evidence.  Replaying a successful generation action because JDT is
unavailable cannot improve the verifier and is therefore forbidden.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import wraps
from typing import Any

from .root_cause_trace import emit_root_cause, trace_scope

_BASE_EXACT_SUFFIXES = (
    "build.gradle",
    "build.gradle.kts",
    "gradle.properties",
    "settings.gradle",
    "settings.gradle.kts",
    "src/main/resources/fabric.mod.json",
    "src/main/resources/pack.mcmeta",
)
_VERIFIER_INFRASTRUCTURE_CODES = frozenset(
    {
        "JDT_DIAGNOSTICS_UNAVAILABLE",
        "VERIFIER_UNAVAILABLE",
    }
)
_RELEASE_NOT_FOUND = re.compile(
    r"\brelease\s+(?P<major>\d+)\s+is\s+not\s+found\s+in\s+the\s+system\b",
    re.IGNORECASE,
)
_VERIFIER_UNAVAILABLE_TEXT = re.compile(
    r"\b(?:jdt(?:\s+diagnostics)?|verifier)\b[^\n]{0,160}\bunavailable\b",
    re.IGNORECASE,
)


def _host_base_owned_path(feedback_module: Any, value: Any) -> bool:
    path = feedback_module._norm_path(value).casefold()
    if not path:
        return False
    for suffix in _BASE_EXACT_SUFFIXES:
        folded = suffix.casefold()
        if path == folded or path.endswith("/" + folded):
            return True
    marker = "/src/main/resources/assets/"
    candidate = "/" + path.lstrip("/")
    if marker not in candidate:
        return False
    tail = candidate.split(marker, 1)[1]
    parts = [part for part in tail.split("/") if part]
    return (
        len(parts) == 3
        and parts[1] == "lang"
        and parts[2] in {"en_us.json", "ko_kr.json"}
    )


def _failure_scalars(value: Any, *, depth: int = 0) -> list[tuple[str, str]]:
    """Return bounded key/value text from nested feedback without assuming one schema."""
    if depth > 12:
        return []
    if isinstance(value, Mapping):
        result: list[tuple[str, str]] = []
        for key, child in value.items():
            key_text = str(key)
            if isinstance(child, Mapping) or (
                isinstance(child, Sequence)
                and not isinstance(child, (str, bytes, bytearray))
            ):
                result.extend(_failure_scalars(child, depth=depth + 1))
            elif child is not None:
                result.append((key_text, str(child)))
        return result[:4096]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for child in value:
            result.extend(_failure_scalars(child, depth=depth + 1))
            if len(result) >= 4096:
                break
        return result[:4096]
    if value is None:
        return []
    return [("", str(value))]


def _verifier_infrastructure_failure(feedback: Any) -> dict[str, Any] | None:
    """Classify non-source-repairable verifier failures and extract required Java."""
    scalars = _failure_scalars(feedback)
    matched_code = ""
    required_java: int | None = None
    matched_text = ""

    for key, text in scalars:
        normalized_key = key.casefold().replace("-", "_")
        normalized_text = text.strip()
        upper = normalized_text.upper()
        if (
            normalized_key in {"code", "error_code", "failure_code", "reason_code"}
            and upper in _VERIFIER_INFRASTRUCTURE_CODES
        ):
            matched_code = upper
            matched_text = normalized_text
        release_match = _RELEASE_NOT_FOUND.search(normalized_text)
        if release_match is not None:
            required_java = int(release_match.group("major"))
            matched_text = normalized_text
        if not matched_code and _VERIFIER_UNAVAILABLE_TEXT.search(normalized_text):
            matched_code = "VERIFIER_UNAVAILABLE"
            matched_text = normalized_text

    if not matched_code and required_java is None:
        return None
    if not matched_code:
        matched_code = "JDT_DIAGNOSTICS_UNAVAILABLE"

    canonical = {
        "code": matched_code,
        "required_java": required_java,
        "message": matched_text,
    }
    rendered = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    canonical["fingerprint"] = "sha256:" + hashlib.sha256(
        rendered.encode("utf-8")
    ).hexdigest()
    return canonical


def _install_base_project_owner(feedback_module: Any) -> None:
    current = feedback_module._derive_impacted_seeds
    if getattr(current, "_mmm_host_base_project_owner", False):
        return

    @wraps(current)
    def derive_impacted_seeds(ledger: Any, feedback: Mapping[str, Any]):
        seed_nodes, owner_ids, requirement_ids, matched = current(ledger, feedback)
        seed_nodes = set(seed_nodes)
        owner_ids = set(owner_ids)
        requirement_ids = set(requirement_ids)
        matched = [dict(item) for item in matched]

        already_matched: list[str] = []
        for item in matched:
            paths = item.get("diagnostic_paths")
            if isinstance(paths, Sequence) and not isinstance(
                paths, (str, bytes, bytearray)
            ):
                already_matched.extend(
                    feedback_module._norm_path(path)
                    for path in paths
                    if feedback_module._norm_path(path)
                )

        diagnostics = feedback.get("diagnostics")
        diagnostics = diagnostics if isinstance(diagnostics, Sequence) else ()
        base_paths: set[str] = set()
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, Mapping):
                continue
            path = feedback_module._norm_path(diagnostic.get("path"))
            if not path or not _host_base_owned_path(feedback_module, path):
                continue
            if any(
                feedback_module._path_equivalent(path, observed)
                for observed in already_matched
            ):
                continue
            base_paths.add(path)

        if base_paths:
            seed_nodes.add("prepare-project")
            owner_ids.add("prepare-project")
            matched.append(
                {
                    "node_id": "prepare-project",
                    "owner_ids": ["prepare-project"],
                    "requirement_ids": [],
                    "diagnostic_paths": sorted(base_paths),
                    "match": {
                        "explicit_owner": False,
                        "explicit_requirement": False,
                        "observed_path": False,
                        "host_base_project": True,
                    },
                }
            )
        return seed_nodes, owner_ids, requirement_ids, matched

    derive_impacted_seeds._mmm_host_base_project_owner = True  # type: ignore[attr-defined]
    derive_impacted_seeds.__wrapped__ = current
    feedback_module._derive_impacted_seeds = derive_impacted_seeds



def _run_feedback_loop(
    self: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    current_execute: Any,
    feedback_module: Any,
    orchestrator_module: Any,
) -> Any:
    seen: set[str] = set()
    call_kwargs = dict(kwargs)
    while True:
        try:
            return current_execute(self, *args, **call_kwargs)
        except orchestrator_module.CompleteProductionError as exc:
            emit_root_cause(
                "execution_feedback_failure_observed",
                stage="generation",
                operation="execute_with_feedback",
                gate="adjudication",
                result="FAIL",
                reason=f"{type(exc).__name__}: {exc}",
                details={"seen_fingerprints": sorted(seen)},
                exc=exc,
            )
            ledger = getattr(self, "_mmm_feedback_ledger", None)
            if ledger is None or not hasattr(ledger, "invalidate_execution_feedback"):
                raise
            feedback = feedback_module._latest_failed_feedback(ledger)
            if not isinstance(feedback, Mapping):
                raise

            infrastructure_failure = _verifier_infrastructure_failure(feedback)
            if infrastructure_failure is not None:
                fingerprint = str(infrastructure_failure["fingerprint"])
                seen.add(fingerprint)
                emit_root_cause(
                    "execution_feedback_abort",
                    stage="generation",
                    operation="execute_with_feedback",
                    gate="retry_eligibility",
                    result="FAIL",
                    reason=(
                        "verifier infrastructure failure is not source-repairable; "
                        "generation replay is forbidden"
                    ),
                    details={
                        "feedback": feedback,
                        "infrastructure_failure": infrastructure_failure,
                        "seen_fingerprints": sorted(seen),
                    },
                )
                raise

            receipt = ledger.invalidate_execution_feedback(feedback)
            fingerprint = str(receipt.get("feedback_fingerprint") or "")
            emit_root_cause(
                "execution_feedback_adjudicated",
                stage="generation",
                operation="execute_with_feedback",
                gate="impact_analysis",
                result="PASS",
                details={
                    "feedback": feedback,
                    "invalidation_receipt": receipt,
                    "fingerprint": fingerprint,
                },
            )
            if (
                receipt.get("global_replan_required") is True
                or not receipt.get("impacted_generation_node_ids")
                or not fingerprint
                or fingerprint in seen
            ):
                emit_root_cause(
                    "execution_feedback_abort",
                    stage="generation",
                    operation="execute_with_feedback",
                    gate="retry_eligibility",
                    result="FAIL",
                    reason="feedback cannot produce a novel owner-bound retry",
                    details={"receipt": receipt, "seen_fingerprints": sorted(seen)},
                )
                raise
            seen.add(fingerprint)
            options = call_kwargs.get("options")
            if options is None:
                options = orchestrator_module.CompleteExecutionOptions(resume=True)
            else:
                try:
                    options = replace(options, resume=True)
                except TypeError:
                    raise
            call_kwargs["options"] = options
            emit_root_cause(
                "execution_feedback_retry",
                stage="generation",
                operation="execute_with_feedback",
                gate="retry_eligibility",
                result="START",
                reason="novel impacted nodes invalidated",
                details={
                    "fingerprint": fingerprint,
                    "options": options,
                    "seen_fingerprint_count": len(seen),
                },
            )


def _trace_feedback_execution(
    self: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    execute_feedback_loop: Any,
) -> Any:
    with trace_scope("complete_production"):
        emit_root_cause(
            "pipeline_boundary_start",
            stage="runtime",
            operation="complete_production",
            gate="end_to_end_execution",
            result="START",
            details={"args": args, "kwargs": kwargs, "semantic_convergence": True},
        )
        try:
            result = execute_feedback_loop(self, *args, **kwargs)
        except BaseException as exc:
            emit_root_cause(
                "pipeline_boundary_failure",
                stage="runtime",
                operation="complete_production",
                gate="end_to_end_execution",
                result="FAIL",
                reason=f"{type(exc).__name__}: {exc}",
                exc=exc,
            )
            raise
        emit_root_cause(
            "pipeline_boundary_result",
            stage="runtime",
            operation="complete_production",
            gate="end_to_end_execution",
            result="PASS",
            details={"result": result},
        )
        return result

def _semantic_install_run_context(feedback_module: Any, orchestrator_module: Any) -> None:
    cls = orchestrator_module.CompleteProductionOrchestrator
    current_open = cls._open_run
    if not getattr(current_open, "_mmm_feedback_context", False):

        @wraps(current_open)
        def open_run(self: Any, run_name: str, plan: Any, *, resume: bool):
            root, ledger, resumed = current_open(self, run_name, plan, resume=resume)
            self._mmm_feedback_run_root = root
            self._mmm_feedback_ledger = ledger
            self._mmm_feedback_plan = plan
            return root, ledger, resumed

        open_run._mmm_feedback_context = True  # type: ignore[attr-defined]
        open_run.__wrapped__ = current_open
        cls._open_run = open_run

    current_execute = cls.execute
    if getattr(current_execute, "_mmm_impacted_feedback_loop", False):
        return

    def execute_feedback_loop(self: Any, *args: Any, **kwargs: Any):
        return _run_feedback_loop(
            self, args, kwargs, current_execute, feedback_module, orchestrator_module
        )

    @wraps(current_execute)
    def execute_with_feedback(self: Any, *args: Any, **kwargs: Any):
        return _trace_feedback_execution(self, args, kwargs, execute_feedback_loop)

    execute_with_feedback._mmm_impacted_feedback_loop = True  # type: ignore[attr-defined]
    execute_with_feedback._mmm_semantic_convergence = True  # type: ignore[attr-defined]
    execute_with_feedback.__wrapped__ = current_execute
    cls.execute = execute_with_feedback


def install(feedback_module: Any) -> None:
    """Patch the feedback contract before ``runtime_finalization`` installs it."""
    # java_diagnostics must resolve the target project JDK inside the verifier call;
    # Colab bootstrap may have happened before the target Minecraft version existed.
    from . import production_tools as production_tools_module
    from .project_java_diagnostics_installation import install as install_project_java_diagnostics

    install_project_java_diagnostics(production_tools_module)
    _install_base_project_owner(feedback_module)
    current = feedback_module._install_run_context
    if getattr(current, "_mmm_semantic_convergence_installation", False):
        return

    def install_run_context(orchestrator_module: Any) -> None:
        _semantic_install_run_context(feedback_module, orchestrator_module)

    install_run_context._mmm_semantic_convergence_installation = True  # type: ignore[attr-defined]
    install_run_context.__wrapped__ = current
    feedback_module._install_run_context = install_run_context


__all__ = [
    "_verifier_infrastructure_failure",
    "install",
]
