from __future__ import annotations

import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Sequence


def _active_parallel_slots() -> int:
    raw = os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "1").strip()
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 1


def _planner_config(router: Any) -> Any | None:
    registry = getattr(router, "registry", None)
    profile = getattr(router, "profile", None)
    if registry is None or profile is None:
        return None
    try:
        return registry.role(profile, "planner")
    except Exception:
        return None


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
            if memory and float(memory[0].get("similarity", 0.0)) >= 0.72:
                return 1
            signature = self._signature(dict(evidence))
            counts = getattr(self, "_mmm_signature_counts", None)
            if not isinstance(counts, Counter):
                counts = Counter()
                self._mmm_signature_counts = counts
            counts[signature] += 1
            return min(3, width) if counts[signature] >= 2 else 1

        repair_candidate_count._mmm_failure_gated_search = True
        repair_candidate_count.__wrapped__ = current_repair_count
        agentic_module._repair_candidate_count = repair_candidate_count

    _install_parallel_planner_search(agentic_module)


__all__ = [
    "install",
    "_active_parallel_slots",
    "_parallel_workers",
    "_prime_native_slots",
]
