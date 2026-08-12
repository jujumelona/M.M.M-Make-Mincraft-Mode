from __future__ import annotations

from functools import wraps
from typing import Any


_MAX_SELECTION_REQUEST_CHARS = 8_000
_MAX_SELECTION_DESIGN_CHARS = 8_000


def _bounded_text(value: str, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head] + "\n...[host-bounded platform context]...\n" + text[-tail:]


def install(*, resolver_module: Any, central_contract_module: Any) -> None:
    """Keep target choice deterministic in normal planning and bounded elsewhere.

    ``GameDesignPlanner`` has already paid for the semantic model call.  Explicit
    Minecraft versions, loaders, migration requests and existing-project targets are
    all host-parsable constraints, while an unpinned new project can safely choose the
    newest fully resolved reviewed Fabric profile.  Calling the planner a second time
    solely to choose among those profiles adds latency and can duplicate an arbitrarily
    large user request.

    The resolver still supports an explicit AI-choice caller for experiments/tools, but
    that path receives a bounded request/design view so it can never re-inject a huge
    prompt into one model call.
    """

    original_choose = resolver_module._choose_with_central_ai
    if not getattr(original_choose, "_mmm_bounded_platform_choice", False):

        @wraps(original_choose)
        def bounded_choose(
            router: Any | None,
            *,
            prompt: str,
            design: dict[str, Any] | None,
            candidates: tuple[Any, ...],
        ):
            bounded_prompt = _bounded_text(prompt, _MAX_SELECTION_REQUEST_CHARS)
            bounded_design = design
            if isinstance(design, dict):
                # The resolver serializes this value again.  Preserve the most useful
                # top-level fields without permitting an unbounded nested request copy.
                import json

                encoded = json.dumps(design, ensure_ascii=False, default=str)
                if len(encoded) > _MAX_SELECTION_DESIGN_CHARS:
                    bounded_design = {
                        "title": design.get("title"),
                        "pitch": design.get("pitch"),
                        "compatibility": design.get("compatibility"),
                        "requested_capabilities": design.get("requested_capabilities"),
                        "_bounded": True,
                    }
            return original_choose(
                router,
                prompt=bounded_prompt,
                design=bounded_design,
                candidates=candidates,
            )

        bounded_choose._mmm_bounded_platform_choice = True
        resolver_module._choose_with_central_ai = bounded_choose

    original_resolve = central_contract_module.resolve_platform
    if not getattr(original_resolve, "_mmm_host_deterministic_platform_choice", False):

        @wraps(original_resolve)
        def host_resolve_platform(*args: Any, **kwargs: Any):
            # The normal GameDesignPlanner path must not trigger a second planner
            # generation. The resolver still parses all hard target constraints and
            # selects only from fully resolved reviewed profiles.
            kwargs["router"] = None
            return resolver_module.resolve_platform(*args, **kwargs)

        host_resolve_platform._mmm_host_deterministic_platform_choice = True
        central_contract_module.resolve_platform = host_resolve_platform
