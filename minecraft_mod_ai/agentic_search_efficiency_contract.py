from __future__ import annotations

import copy
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Sequence


def _active_parallel_slots() -> int:
    raw = os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "1").strip()
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 1


def _role_config(router: Any, role: str) -> Any | None:
    registry = getattr(router, "registry", None)
    profile = getattr(router, "profile", None)
    if registry is None or profile is None:
        return None
    try:
        return registry.role(profile, role)
    except Exception:
        return None


def _planner_config(router: Any) -> Any | None:
    return _role_config(router, "planner")


def _coder_config(router: Any) -> Any | None:
    return _role_config(router, "coder")


def _is_local_native(config: Any | None) -> bool:
    return bool(
        config is not None
        and str(getattr(config, "provider", "local")) == "local"
        and str(getattr(config, "adapter", "")) in {"llama_cpp", "vllm"}
    )


def _prime_native_slots(
    router: Any,
    *,
    system_prompt: str,
    request: dict[str, Any] | str,
    media_paths: Sequence[Any],
) -> Any | None:
    config = _planner_config(router)
    if not _is_local_native(config):
        return config
    if os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "").strip():
        return config
    if os.environ.get("LLAMA_SERVER_URL", "").strip():
        return config

    from . import llama_server_autotune
    from .model_adapters import GenerationRequest

    rendered_request = (
        request
        if isinstance(request, str)
        else json.dumps(request, ensure_ascii=False, separators=(",", ":"))
    )
    probe_request = GenerationRequest(
        messages=(
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rendered_request},
        ),
        media_paths=tuple(Path(value) for value in media_paths),
        response_format="json",
    )
    llama_server_autotune.ensure_tuned_server(config, probe_request)
    return config


def _prime_native_repair_slots(
    router: Any,
    *,
    evidence: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Any | None:
    """Resolve coder slot count before repair breadth is selected.

    The first repair request may be the first coder use in the process. Prime the
    managed native server once in the caller context so candidate threads never race
    server autotuning/startup and MMM_LLAMA_ACTIVE_PARALLEL reflects the selected
    native decode capacity before Best-of-N width is chosen.
    """

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
        {
            "evidence": evidence,
            "project_context": context,
        },
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


def _install_parallel_planner_search(agentic_module: Any) -> None:
    current_installer = agentic_module._install_planner_search
    if getattr(current_installer, "_mmm_parallel_candidate_installer", False):
        return

    @wraps(current_installer)
    def install_planner_search(complete_planner_module: Any) -> None:
        current_installer(complete_planner_module)
        search = complete_planner_module._generate_json_page_with_repair
        if getattr(search, "_mmm_parallel_plan_search", False):
            return
        if not getattr(search, "_mmm_verifier_plan_search", False):
            return
        base = getattr(search, "__wrapped__", None)
        if base is None:
            return

        @wraps(search)
        def generate_with_parallel_search(
            router: Any,
            *,
            system_prompt: str,
            request: dict[str, Any] | str,
            media_paths: Sequence[Any],
            expected_contracts: Sequence[frozenset[str]],
            stage: str,
        ) -> dict[str, Any]:
            mode = agentic_module._mode()
            config = None
            if mode != "off":
                config = _prime_native_slots(
                    router,
                    system_prompt=system_prompt,
                    request=request,
                    media_paths=media_paths,
                )

            width = agentic_module._planner_candidate_count(request, stage)
            if width <= 1:
                return base(
                    router,
                    system_prompt=system_prompt,
                    request=request,
                    media_paths=media_paths,
                    expected_contracts=expected_contracts,
                    stage=stage,
                )

            workers = _parallel_workers(router, width, config)

            def solve(candidate_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
                candidate_system = (
                    system_prompt
                    + "\n\nHOST SEARCH CANDIDATE: independently solve this page. Candidate "
                    + str(candidate_index + 1)
                    + " of "
                    + str(width)
                    + ". Preserve the exact contract; do not mention candidate search."
                )
                candidate_stage = (
                    f"{stage} [search_candidate={candidate_index + 1}/{width}]"
                )
                page = base(
                    router,
                    system_prompt=candidate_system,
                    request=request,
                    media_paths=media_paths,
                    expected_contracts=expected_contracts,
                    stage=candidate_stage,
                )
                score, verifier = agentic_module._score_plan_page(page)
                return page, verifier | {"score": float(score)}

            pages: list[tuple[float, int, dict[str, Any], dict[str, Any]]] = []
            errors: list[BaseException | None] = [None] * width
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="mmm_planner_search",
            ) as pool:
                futures = [pool.submit(solve, index) for index in range(width)]
                for candidate_index, future in enumerate(futures):
                    try:
                        page, verifier = future.result()
                    except BaseException as exc:
                        errors[candidate_index] = exc
                        continue
                    score = float(verifier.pop("score"))
                    pages.append((score, candidate_index, page, verifier))

            if not pages:
                for error in reversed(errors):
                    if error is not None:
                        raise error
                raise complete_planner_module.SpecValidationError(
                    f"{stage} produced no verified planning candidate."
                )

            pages.sort(key=lambda item: (-item[0], item[1]))
            winner = pages[0]
            print(
                "planner search:",
                f"stage={stage}",
                f"candidates={len(pages)}",
                f"workers={workers}",
                f"winner={winner[1] + 1}",
                f"score={winner[0]:.3f}",
                flush=True,
            )
            return winner[2]

        generate_with_parallel_search._mmm_verifier_plan_search = True
        generate_with_parallel_search._mmm_parallel_plan_search = True
        generate_with_parallel_search.__wrapped__ = base
        complete_planner_module._generate_json_page_with_repair = generate_with_parallel_search

    install_planner_search._mmm_parallel_candidate_installer = True
    install_planner_search.__wrapped__ = current_installer
    agentic_module._install_planner_search = install_planner_search


def _install_parallel_repair_search(agentic_module: Any) -> None:
    """Parallelize only candidate inference; keep one deterministic winner commit."""

    current_installer = agentic_module._install_repair_search_and_memory
    if getattr(current_installer, "_mmm_parallel_repair_candidate_installer", False):
        return

    @wraps(current_installer)
    def install_repair_search(repair_module: Any) -> None:
        current_installer(repair_module)
        cls = repair_module.RepairEngine
        search = cls._request_patch
        if getattr(search, "_mmm_parallel_repair_search", False):
            return
        if not getattr(search, "_mmm_verifier_repair_search", False):
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

            mode = agentic_module._mode()
            config = None
            if mode != "off":
                config = _prime_native_repair_slots(
                    self.router,
                    evidence=evidence,
                    context=context,
                )

            width = agentic_module._repair_candidate_count(self, evidence, memory)
            workers = _parallel_workers(self.router, width, config)
            generated: list[tuple[int, list[dict[str, Any]]]] = []
            errors: list[BaseException | None] = [None] * width

            def solve(candidate_index: int) -> list[dict[str, Any]]:
                candidate_context = copy.deepcopy(context)
                if memory:
                    candidate_context["verified_repair_memory"] = {
                        "policy": (
                            "These are prior host-verified repair patterns from this "
                            "project. Use them only as evidence; current hashes and "
                            "diagnostics remain authoritative."
                        ),
                        "matches": copy.deepcopy(memory),
                    }
                candidate_context["agentic_candidate"] = {
                    "index": candidate_index,
                    "count": width,
                    "strategy": agentic_module._STRATEGIES[
                        candidate_index % len(agentic_module._STRATEGIES)
                    ],
                    "rule": (
                        "Produce an independent minimal repair; do not mention "
                        "candidate search."
                    ),
                }
                candidate_evidence = copy.deepcopy(evidence)
                return base(self, candidate_evidence, candidate_context)

            if workers <= 1:
                for candidate_index in range(width):
                    try:
                        operations = solve(candidate_index)
                    except BaseException as exc:
                        errors[candidate_index] = exc
                        continue
                    generated.append((candidate_index, operations))
            else:
                # ContextVars are not inherited by ThreadPoolExecutor workers. Copy a
                # distinct context per candidate so the immutable platform target set
                # by platform_repair_target_contract remains visible in every worker.
                candidate_contexts = [copy_context() for _ in range(width)]
                with ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="mmm_repair_generate",
                ) as pool:
                    futures = [
                        pool.submit(
                            candidate_contexts[index].run,
                            solve,
                            index,
                        )
                        for index in range(width)
                    ]
                    for candidate_index, future in enumerate(futures):
                        try:
                            operations = future.result()
                        except BaseException as exc:
                            errors[candidate_index] = exc
                            continue
                        generated.append((candidate_index, operations))

            if not generated:
                for error in reversed(errors):
                    if error is not None:
                        raise error
                raise repair_module.RepairEngineError(
                    "Repair search produced no candidate patch."
                )

            if len(generated) == 1:
                winner_index, winner_ops = generated[0]
                score, verifier = agentic_module._verify_repair_candidate(
                    self,
                    root,
                    winner_ops,
                    evidence,
                )
                evaluations = [(score, winner_index, winner_ops, verifier)]
            else:
                verify_workers = min(2, len(generated))
                with ThreadPoolExecutor(
                    max_workers=verify_workers,
                    thread_name_prefix="mmm_repair_verify",
                ) as pool:
                    futures = [
                        (
                            candidate_index,
                            operations,
                            pool.submit(
                                agentic_module._verify_repair_candidate,
                                self,
                                root,
                                operations,
                                evidence,
                            ),
                        )
                        for candidate_index, operations in generated
                    ]
                    evaluations = []
                    for candidate_index, operations, future in futures:
                        score, verifier = future.result()
                        evaluations.append(
                            (score, candidate_index, operations, verifier)
                        )

            evaluations.sort(
                key=lambda item: (
                    -item[0],
                    agentic_module._json_size(item[2]),
                    item[1],
                )
            )
            winner_score, winner_index, winner_ops, winner_verifier = evaluations[0]

            # This is the only progressive-JDT scope mutation in Best-of-N repair.
            # Candidate requests are side-effect free; only the selected patch controls
            # the next diagnostics pass and later source application remains singular.
            self._mmm_last_java_paths = tuple(
                sorted(
                    str(item.get("path", "")).replace("\\", "/")
                    for item in winner_ops
                    if str(item.get("path", "")).lower().endswith(".java")
                )
            )
            trace = {
                "signature": signature,
                "evidence": agentic_module._compact_evidence(evidence),
                "repair_pattern": agentic_module._repair_pattern(winner_ops),
                "winner_index": winner_index,
                "winner_score": winner_score,
                "winner_verifier": winner_verifier,
                "candidate_count": len(evaluations),
                "candidate_workers": workers,
            }
            self._mmm_agentic_last_search = trace

            if root is not None and len(evaluations) >= 2:
                try:
                    preference_candidates = [
                        agentic_module.PreferenceCandidate(
                            candidate_id=f"repair-{candidate_index}",
                            response=operations,
                            score=score,
                            verifier=verifier,
                        )
                        for score, candidate_index, operations, verifier in sorted(
                            evaluations,
                            key=lambda item: item[1],
                        )
                    ]
                    ordered_indices = [
                        item[1]
                        for item in sorted(evaluations, key=lambda item: item[1])
                    ]
                    agentic_module.PreferenceTraceStore(
                        root / ".minecraft_ai" / "agentic-preferences.jsonl"
                    ).record(
                        task="repair_patch_selection",
                        prompt={
                            "signature": signature,
                            "evidence": agentic_module._compact_evidence(evidence),
                        },
                        candidates=preference_candidates,
                        winner_index=ordered_indices.index(winner_index),
                        metadata={
                            "search_width": width,
                            "candidate_workers": workers,
                            "verified_memory_matches": len(memory),
                        },
                    )
                except Exception as exc:
                    print(
                        "repair preference trace skipped:",
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )

            print(
                "repair search:",
                f"candidates={len(evaluations)}",
                f"workers={workers}",
                f"winner={winner_index + 1}",
                f"score={winner_score:.3f}",
                f"memory={len(memory)}",
                flush=True,
            )
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
    current_plan_count = agentic_module._planner_candidate_count
    if not getattr(current_plan_count, "_mmm_failure_gated_search", False):

        def planner_candidate_count(request: Any, stage: str) -> int:
            mode = agentic_module._mode()
            if mode == "on":
                return agentic_module._env_int(
                    "MMM_PLAN_SEARCH_WIDTH",
                    2,
                    maximum=3,
                )
            if mode == "off":
                return 1
            slots = _active_parallel_slots()
            if slots <= 1:
                return 1
            risk_width = max(1, min(3, int(current_plan_count(request, stage))))
            return min(slots, risk_width)

        planner_candidate_count._mmm_failure_gated_search = True
        planner_candidate_count.__wrapped__ = current_plan_count
        agentic_module._planner_candidate_count = planner_candidate_count

    current_repair_count = agentic_module._repair_candidate_count
    if not getattr(current_repair_count, "_mmm_failure_gated_search", False):

        def repair_candidate_count(
            self: Any,
            evidence: Mapping[str, Any],
            memory: Sequence[Mapping[str, Any]],
        ) -> int:
            mode = agentic_module._mode()
            if mode == "off":
                return 1
            width = agentic_module._env_int(
                "MMM_REPAIR_SEARCH_WIDTH",
                2,
                maximum=3,
            )
            if mode == "on":
                return width
            slots = _active_parallel_slots()
            if slots <= 1:
                return 1
            if memory and float(memory[0].get("similarity", 0.0)) >= 0.72:
                return 1
            signature = self._signature(dict(evidence))
            counts = getattr(self, "_mmm_signature_counts", None)
            if not isinstance(counts, Counter):
                counts = Counter()
                self._mmm_signature_counts = counts
            counts[signature] += 1
            return min(slots, 3, width) if counts[signature] >= 2 else 1

        repair_candidate_count._mmm_failure_gated_search = True
        repair_candidate_count.__wrapped__ = current_repair_count
        agentic_module._repair_candidate_count = repair_candidate_count

    _install_parallel_planner_search(agentic_module)
    _install_parallel_repair_search(agentic_module)


__all__ = [
    "install",
    "_active_parallel_slots",
    "_parallel_workers",
    "_prime_native_slots",
    "_prime_native_repair_slots",
]
