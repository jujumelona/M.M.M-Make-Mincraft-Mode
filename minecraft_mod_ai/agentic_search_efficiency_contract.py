from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import wraps
from pathlib import Path
from typing import Any

from .runtime_contract_wrappers import has_contract_marker, owns_contract_marker

_FAILURE_GATED_SEARCH_MARKER = "_mmm_failure_gated_search"
_FAILURE_GATED_SEARCH_EPOCH = "mmm/failure-gated-search-v5"


def _active_parallel_slots() -> int:
    raw = os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "1").strip()
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 1


def _search_mode() -> str:
    value = os.environ.get("MMM_AGENTIC_SEARCH", "auto").strip().casefold()
    return value if value in {"off", "auto", "on"} else "auto"


def _repair_search_width() -> int:
    raw = os.environ.get("MMM_REPAIR_SEARCH_WIDTH", "2").strip()
    try:
        return max(1, min(3, int(raw)))
    except ValueError:
        return 2


def _coder_config(router: Any) -> Any | None:
    registry = getattr(router, "registry", None)
    profile = getattr(router, "profile", None)
    if registry is None or profile is None:
        return None
    try:
        return registry.role(profile, "coder")
    except Exception:
        return None


def _is_local_native(config: Any | None) -> bool:
    return bool(
        config is not None
        and str(getattr(config, "provider", "local")) == "local"
        and str(getattr(config, "adapter", "")) in {"llama_cpp", "vllm"}
    )


def _prime_native_repair_slots(
    router: Any,
    *,
    evidence: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Any | None:
    config = _coder_config(router)
    if not _is_local_native(config):
        return config
    if os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "").strip():
        return config
    if os.environ.get("LLAMA_SERVER_URL", "").strip():
        return config

    from . import llama_server_autotune
    from .model_adapters import GenerationRequest

    rendered = json.dumps(
        {"evidence": evidence, "project_context": context},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    probe_request = GenerationRequest(
        messages=(
            {
                "role": "system",
                "content": (
                    "Prime the native coder runtime for one hash-guarded Fabric "
                    "repair JSON request. No completion is requested by this probe."
                ),
            },
            {"role": "user", "content": rendered},
        ),
        response_format="json",
    )
    llama_server_autotune.ensure_tuned_server(config, probe_request)
    return config


def _parallel_workers(router: Any, width: int, config: Any | None) -> int:
    if width <= 1:
        return 1
    if config is not None and not _is_local_native(config):
        return width
    return max(1, min(width, _active_parallel_slots()))


def _base_repair_candidate_count(candidate_count: Any) -> Any:
    current = candidate_count
    seen: set[int] = set()
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        wrapped = getattr(current, "__wrapped__", None)
        if not callable(wrapped) or wrapped is current:
            break
        current = wrapped
    return current


def _owns_current_repair_width_policy(value: Any) -> bool:
    if not owns_contract_marker(value, _FAILURE_GATED_SEARCH_MARKER):
        return False
    namespace = getattr(value, "__dict__", None)
    if not isinstance(namespace, dict):
        return False
    if namespace.get("_mmm_failure_gated_search_epoch") != _FAILURE_GATED_SEARCH_EPOCH:
        return False
    code = getattr(value, "__code__", None)
    if code is None or str(getattr(code, "co_name", "")) != "repair_candidate_count":
        return False
    filename = str(getattr(code, "co_filename", "")).replace("\\", "/")
    return filename.endswith("/agentic_search_efficiency_contract.py")


def _live_search_mode() -> str:
    import os as runtime_os

    value = runtime_os.environ.get("MMM_AGENTIC_SEARCH", "auto").strip().casefold()
    return value if value in {"off", "auto", "on"} else "auto"


def _live_repair_search_width() -> int:
    import os as runtime_os

    raw = runtime_os.environ.get("MMM_REPAIR_SEARCH_WIDTH", "2").strip()
    try:
        return max(1, min(3, int(raw)))
    except ValueError:
        return 2


def _live_active_parallel_slots() -> int:
    import os as runtime_os

    raw = runtime_os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "1").strip()
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 1


def _install_parallel_repair_search(agentic_module: Any) -> None:
    current_installer = agentic_module._install_repair_search_and_memory
    if has_contract_marker(current_installer, "_mmm_parallel_repair_candidate_installer"):
        return

    prime_native_repair_slots = _prime_native_repair_slots
    parallel_workers = _parallel_workers

    def read_search_mode() -> str:
        import os as runtime_os

        value = runtime_os.environ.get("MMM_AGENTIC_SEARCH", "auto").strip().casefold()
        return value if value in {"off", "auto", "on"} else "auto"

    @wraps(current_installer)
    def install_repair_search(repair_module: Any) -> None:
        current_installer(repair_module)
        cls = repair_module.RepairEngine
        search = cls._request_patch
        if has_contract_marker(search, "_mmm_parallel_repair_search"):
            return
        if not owns_contract_marker(search, "_mmm_verifier_repair_search"):
            return
        base = getattr(search, "__wrapped__", None)
        if base is None:
            return

        @wraps(search)
        def request_patch_with_parallel_search(
            self: Any,
            evidence: dict[str, Any],
            context: dict[str, Any],
        ) -> list[dict[str, Any]]:
            root_value = getattr(self, "_mmm_agentic_root", None)
            root = Path(root_value).resolve() if root_value else None
            signature = self._signature(evidence)
            memory = (
                agentic_module._read_memory(root, signature)
                if root is not None
                else []
            )
            router = getattr(self, "router", None)

            config = None
            if read_search_mode() != "off":
                config = prime_native_repair_slots(
                    router,
                    evidence=evidence,
                    context=context,
                )
            width = agentic_module._repair_candidate_count(self, evidence, memory)
            workers = parallel_workers(router, width, config)
            generated: list[tuple[int, list[dict[str, Any]]]] = []
            errors: list[Exception | None] = [None] * width

            def solve(candidate_index: int) -> list[dict[str, Any]]:
                candidate_context = copy.deepcopy(context)
                if memory:
                    candidate_context["verified_repair_memory"] = {
                        "policy": (
                            "Prior host-verified project repairs are evidence only; "
                            "current hashes and diagnostics remain authoritative."
                        ),
                        "matches": copy.deepcopy(memory),
                    }
                candidate_context["agentic_candidate"] = {
                    "index": candidate_index,
                    "count": width,
                    "strategy": agentic_module._STRATEGIES[
                        candidate_index % len(agentic_module._STRATEGIES)
                    ],
                    "rule": "Produce an independent minimal repair.",
                }
                return base(self, copy.deepcopy(evidence), candidate_context)

            if workers <= 1:
                for candidate_index in range(width):
                    try:
                        generated.append((candidate_index, solve(candidate_index)))
                    except Exception as exc:
                        errors[candidate_index] = exc
            else:
                contexts = [copy_context() for _ in range(width)]
                with ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="mmm_repair_generate",
                ) as pool:
                    futures = [
                        pool.submit(contexts[index].run, solve, index)
                        for index in range(width)
                    ]
                    for candidate_index, future in enumerate(futures):
                        try:
                            generated.append((candidate_index, future.result()))
                        except Exception as exc:
                            errors[candidate_index] = exc

            if not generated:
                for error in reversed(errors):
                    if error is not None:
                        raise error
                raise repair_module.RepairEngineError(
                    "Repair search produced no candidate patch."
                )

            def verify(
                candidate: tuple[int, list[dict[str, Any]]],
            ) -> tuple[float, int, list[dict[str, Any]], dict[str, Any]]:
                candidate_index, operations = candidate
                score, verifier = agentic_module._verify_repair_candidate(
                    self,
                    root,
                    operations,
                    evidence,
                )
                return float(score), candidate_index, operations, verifier

            if len(generated) == 1:
                evaluations = [verify(generated[0])]
            else:
                with ThreadPoolExecutor(
                    max_workers=min(2, len(generated)),
                    thread_name_prefix="mmm_repair_verify",
                ) as pool:
                    evaluations = list(pool.map(verify, generated))

            evaluations.sort(
                key=lambda item: (
                    -item[0],
                    agentic_module._json_size(item[2]),
                    item[1],
                )
            )
            winner_score, winner_index, winner_ops, winner_verifier = evaluations[0]
            self._mmm_last_java_paths = tuple(
                sorted(
                    str(item.get("path", "")).replace("\\", "/")
                    for item in winner_ops
                    if str(item.get("path", "")).lower().endswith(".java")
                )
            )
            self._mmm_agentic_last_search = {
                "signature": signature,
                "evidence": agentic_module._compact_evidence(evidence),
                "repair_pattern": agentic_module._repair_pattern(winner_ops),
                "winner_index": winner_index,
                "winner_score": winner_score,
                "winner_verifier": winner_verifier,
                "candidate_count": len(evaluations),
                "candidate_workers": workers,
            }
            return winner_ops

        request_patch_with_parallel_search._mmm_verifier_repair_search = True
        request_patch_with_parallel_search._mmm_parallel_repair_search = True
        request_patch_with_parallel_search._mmm_tracks_repair_scope = True
        request_patch_with_parallel_search.__wrapped__ = base
        cls._request_patch = request_patch_with_parallel_search

    install_repair_search._mmm_parallel_repair_candidate_installer = True
    install_repair_search.__wrapped__ = current_installer
    agentic_module._install_repair_search_and_memory = install_repair_search


def install(agentic_module: Any) -> None:
    current_repair_count = agentic_module._repair_candidate_count
    if not _owns_current_repair_width_policy(current_repair_count):
        risk_candidate_count = _base_repair_candidate_count(current_repair_count)

        def read_search_mode() -> str:
            import os as runtime_os

            value = runtime_os.environ.get("MMM_AGENTIC_SEARCH", "auto").strip().casefold()
            return value if value in {"off", "auto", "on"} else "auto"

        def read_repair_search_width() -> int:
            import os as runtime_os

            raw = runtime_os.environ.get("MMM_REPAIR_SEARCH_WIDTH", "2").strip()
            try:
                return max(1, min(3, int(raw)))
            except ValueError:
                return 2

        def read_active_parallel_slots() -> int:
            import os as runtime_os

            raw = runtime_os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "1").strip()
            try:
                return max(1, min(8, int(raw)))
            except ValueError:
                return 1

        def repair_candidate_count(
            self: Any,
            evidence: Mapping[str, Any],
            memory: Sequence[Mapping[str, Any]],
        ) -> int:
            mode = read_search_mode()
            if mode == "off":
                return 1
            width = read_repair_search_width()
            if mode == "on":
                return width
            slots = read_active_parallel_slots()
            if slots <= 1:
                return 1
            risk_width = max(
                1,
                min(3, int(risk_candidate_count(self, evidence, memory))),
            )
            return min(slots, risk_width)

        setattr(repair_candidate_count, _FAILURE_GATED_SEARCH_MARKER, True)
        repair_candidate_count._mmm_failure_gated_search_epoch = _FAILURE_GATED_SEARCH_EPOCH
        repair_candidate_count.__wrapped__ = risk_candidate_count
        agentic_module._repair_candidate_count = repair_candidate_count

    _install_parallel_repair_search(agentic_module)


__all__ = [
    "_active_parallel_slots",
    "_parallel_workers",
    "_prime_native_repair_slots",
    "install",
]
