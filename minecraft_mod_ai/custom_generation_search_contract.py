from __future__ import annotations

import copy
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Sequence


_STRATEGIES = (
    "minimal_surface_area",
    "fabric_api_contract_first",
    "runtime_and_persistence_first",
)


class _HostEvidenceRouter:
    """Keep coder tools optional when the host already supplied fresh exact evidence.

    CustomModuleGenerator builds a current ProjectIndex, indexes project RAG and sends
    bounded source observations plus research_context before asking the coder to emit a
    patch. Requiring a model-driven RAG call after that host work turns every custom
    module into model -> tool -> model even when no new uncertainty exists. This proxy
    preserves the same tool surface but marks the host evidence as satisfying the
    mandatory-freshness precondition; the model may still choose tools normally.
    """

    def __init__(self, router: Any) -> None:
        self._router = router

    def __getattr__(self, name: str) -> Any:
        return getattr(self._router, name)

    def bind_agent_workspace(
        self,
        workspace_root: str | Path,
        *,
        require_fresh_evidence: bool = False,
    ) -> "_HostEvidenceRouter":
        del require_fresh_evidence
        self._router.bind_agent_workspace(
            workspace_root,
            require_fresh_evidence=False,
        )
        return self

    def generate_text(
        self,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> str:
        return self._router.generate_text(role, messages, **kwargs)


def _host_evidence_router(router: Any) -> Any:
    if isinstance(router, _HostEvidenceRouter):
        return router
    return _HostEvidenceRouter(router)


class _StrategyRouter:
    def __init__(self, router: Any, *, strategy: str, candidate_index: int, count: int) -> None:
        self._router = router
        self._strategy = strategy
        self._candidate_index = candidate_index
        self._count = count

    def __getattr__(self, name: str) -> Any:
        return getattr(self._router, name)

    def generate_text(self, role: str, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> str:
        if role != "coder":
            return self._router.generate_text(role, messages, **kwargs)
        augmented = [dict(message) for message in messages]
        augmented.insert(
            1 if augmented and augmented[0].get("role") == "system" else 0,
            {
                "role": "system",
                "content": (
                    "Host candidate-search directive: solve independently using strategy="
                    f"{self._strategy}. This is candidate {self._candidate_index + 1}/"
                    f"{self._count}. Keep the exact JSON/patch contract and requested "
                    "functionality; do not mention candidate search in generated files."
                ),
            },
        )
        return self._router.generate_text(role, augmented, **kwargs)


def _mode() -> str:
    value = os.environ.get("MMM_AGENTIC_SEARCH", "auto").strip().lower()
    return value if value in {"auto", "on", "off"} else "auto"


def _active_native_slots() -> int:
    raw = os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "1").strip()
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 1


def _width(module: Any) -> int:
    mode = _mode()
    if mode == "off":
        return 1
    try:
        configured = int(os.environ.get("MMM_CUSTOM_SEARCH_WIDTH", "2"))
    except ValueError:
        configured = 2
    configured = max(1, min(3, configured))
    if mode == "on":
        return configured

    # Auto search must never multiply a single native decode lane. The old path
    # generated candidate 1 and candidate 2 in a Python for-loop, so a complex custom
    # module paid roughly twice the LLM wall time before verification even started.
    # Explicit MMM_AGENTIC_SEARCH=on remains the opt-in quality-over-latency mode.
    if _active_native_slots() <= 1:
        return 1

    kind = str(getattr(module, "kind", ""))
    config = getattr(module, "config", {})
    config = config if isinstance(config, Mapping) else {}
    depends = tuple(getattr(module, "depends_on", ()) or ())
    gates = tuple(getattr(module, "required_gates", ()) or ())
    risk = 0
    if kind in {
        "custom_java",
        "integration",
        "structure",
        "biome",
        "dimension",
        "world_event",
    }:
        risk += 1
    rendered = json.dumps(config, ensure_ascii=False, sort_keys=True)
    if len(rendered.encode("utf-8")) >= 2048 or len(depends) >= 2 or len(gates) >= 2:
        risk += 1
    lowered = rendered.casefold()
    if any(
        marker in lowered
        for marker in (
            "network",
            "multiplayer",
            "persist",
            "migration",
            "ai_",
            "speech",
            "runtime",
            "dimension",
        )
    ):
        risk += 1
    return min(configured, _active_native_slots()) if risk >= 2 else 1


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def _capture_candidate(
    self: Any,
    original: Any,
    candidate_root: Path,
    *,
    strategy: str,
    candidate_index: int,
    count: int,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from . import performance_final_contract as performance_module

    records: list[dict[str, Any]] = []
    old_records = getattr(performance_module._CAPTURE, "records", None)
    old_staging_root = getattr(performance_module._CAPTURE, "staging_root", None)
    old_router = self.router
    old_index = getattr(self, "_cached_index", None)
    old_root = getattr(self, "_cached_root", None)
    performance_module._CAPTURE.records = records
    performance_module._CAPTURE.staging_root = candidate_root
    self.router = _StrategyRouter(
        old_router,
        strategy=strategy,
        candidate_index=candidate_index,
        count=count,
    )
    self._cached_index = None
    self._cached_root = None
    try:
        result = original(self, candidate_root, *args, **kwargs)
        if not isinstance(result, dict):
            raise RuntimeError("Custom generation candidate returned a non-object receipt.")
        captured = performance_module._select_custom_patch_capture(records, result)
        return result, captured
    finally:
        self.router = old_router
        self._cached_index = old_index
        self._cached_root = old_root
        if old_records is None:
            try:
                delattr(performance_module._CAPTURE, "records")
            except AttributeError:
                pass
        else:
            performance_module._CAPTURE.records = old_records
        if old_staging_root is None:
            try:
                delattr(performance_module._CAPTURE, "staging_root")
            except AttributeError:
                pass
        else:
            performance_module._CAPTURE.staging_root = old_staging_root


def _verify_candidate(candidate_root: Path, result: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    touched = [
        str(value).replace("\\", "/")
        for value in result.get("touched_paths", [])
        if isinstance(value, str)
    ]
    java_paths = tuple(sorted(path for path in touched if path.lower().endswith(".java")))
    operation_count = int(result.get("operation_count", 0) or 0)
    runtime_tests = result.get("runtime_tests", [])
    runtime_tests = runtime_tests if isinstance(runtime_tests, list) else []
    score = 2.0 * len(runtime_tests) - 0.3 * operation_count - 0.05 * len(touched)
    verifier: dict[str, Any] = {
        "operation_count": operation_count,
        "touched_path_count": len(touched),
        "runtime_test_count": len(runtime_tests),
        "jdt_status": "NOT_RUN",
        "jdt_error_count": None,
    }
    if not java_paths:
        return score, verifier
    if os.environ.get("MMM_CUSTOM_CANDIDATE_JDT", "auto").strip().lower() == "off":
        return score, verifier
    try:
        from .java_lsp import JavaLanguageService
        from .repair_diagnostics_contract import diagnostic_errors

        diagnostics = JavaLanguageService().diagnostics(
            candidate_root,
            relative_files=java_paths,
            timeout_seconds=60,
        )
        errors = diagnostic_errors(diagnostics)
        verifier["jdt_status"] = "AVAILABLE"
        verifier["jdt_error_count"] = len(errors)
        score += 1000.0 if not errors else -120.0 * len(errors)
    except Exception as exc:
        verifier["jdt_status"] = "VERIFIER_ERROR"
        verifier["verifier_error"] = f"{type(exc).__name__}: {exc}"[:1000]
        score -= 5.0
    return score, verifier


def install(custom_module_generator_module: Any) -> None:
    """Search complex custom generation without duplicating a single decode lane."""

    from . import performance_final_contract as performance_module
    from . import source_patch as source_patch_module

    performance_module._install_locked_source_patcher(source_patch_module)

    cls = custom_module_generator_module.CustomModuleGenerator
    original_init = cls.__init__
    if not getattr(original_init, "_mmm_host_evidence_router", False):

        @wraps(original_init)
        def init_with_host_evidence(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            self.router = _host_evidence_router(self.router)

        init_with_host_evidence._mmm_host_evidence_router = True  # type: ignore[attr-defined]
        cls.__init__ = init_with_host_evidence

    original = cls.generate
    if getattr(original, "_mmm_custom_verifier_search", False):
        return

    @wraps(original)
    def generate_with_search(self: Any, project_root: str | Path, *args: Any, **kwargs: Any):
        self.router = _host_evidence_router(self.router)
        module = kwargs.get("module")
        count = _width(module)
        if count <= 1:
            return original(self, project_root, *args, **kwargs)

        from . import performance_final_contract as performance_module
        from .source_patch import TransactionalSourcePatcher

        root = Path(project_root).expanduser().resolve()
        candidates: list[tuple[int, Path, dict[str, Any], dict[str, Any]]] = []
        errors: list[BaseException] = []
        try:
            for candidate_index in range(count):
                candidate_root = performance_module._clone_source_snapshot(root)
                strategy = _STRATEGIES[candidate_index % len(_STRATEGIES)]
                try:
                    result, capture = _capture_candidate(
                        self,
                        original,
                        candidate_root,
                        strategy=strategy,
                        candidate_index=candidate_index,
                        count=count,
                        args=args,
                        kwargs=kwargs,
                    )
                except BaseException as exc:
                    errors.append(exc)
                    shutil.rmtree(candidate_root, ignore_errors=True)
                    continue
                candidates.append((candidate_index, candidate_root, result, capture))

            if not candidates:
                if errors:
                    raise errors[-1]
                raise RuntimeError("Custom generation search produced no candidate.")

            if len(candidates) == 1:
                candidate_index, candidate_root, result, capture = candidates[0]
                score, verifier = _verify_candidate(candidate_root, result)
                evaluations = [
                    (score, candidate_index, candidate_root, result, capture, verifier)
                ]
            else:
                with ThreadPoolExecutor(
                    max_workers=min(2, len(candidates)),
                    thread_name_prefix="mmm_custom_verify",
                ) as pool:
                    pending = [
                        (
                            candidate_index,
                            candidate_root,
                            result,
                            capture,
                            pool.submit(_verify_candidate, candidate_root, result),
                        )
                        for candidate_index, candidate_root, result, capture in candidates
                    ]
                    evaluations = []
                    for candidate_index, candidate_root, result, capture, future in pending:
                        score, verifier = future.result()
                        evaluations.append(
                            (score, candidate_index, candidate_root, result, capture, verifier)
                        )

            evaluations.sort(
                key=lambda item: (
                    -item[0],
                    _json_size(item[4].get("operations", [])),
                    item[1],
                )
            )
            score, winner_index, winner_root, result, capture, verifier = evaluations[0]
            operations = [copy.deepcopy(item) for item in capture.get("operations", [])]
            if not operations:
                raise RuntimeError("Winning custom candidate contains no patch operations.")
            commit_receipt = TransactionalSourcePatcher(root).apply(operations)
            rewritten = performance_module._rewrite_root_paths(result, winner_root, root)
            rewritten["patch_receipt"] = commit_receipt
            rewritten["agentic_generation_search"] = {
                "schema_version": "mmm/custom-generation-search-v1",
                "candidate_count": len(evaluations),
                "winner_index": winner_index,
                "winner_score": score,
                "winner_verifier": verifier,
                "candidate_scores": [
                    {
                        "candidate_index": item[1],
                        "score": item[0],
                        "verifier": item[5],
                    }
                    for item in sorted(evaluations, key=lambda item: item[1])
                ],
            }
            print(
                "custom generation search:",
                f"candidates={len(evaluations)}",
                f"winner={winner_index + 1}",
                f"score={score:.3f}",
                flush=True,
            )
            return rewritten
        finally:
            for _candidate_index, candidate_root, _result, _capture in candidates:
                shutil.rmtree(candidate_root, ignore_errors=True)

    generate_with_search._mmm_custom_verifier_search = True  # type: ignore[attr-defined]
    generate_with_search._mmm_host_evidence_router = True  # type: ignore[attr-defined]
    cls.generate = generate_with_search


__all__ = ["_active_native_slots", "_host_evidence_router", "_width", "install"]
