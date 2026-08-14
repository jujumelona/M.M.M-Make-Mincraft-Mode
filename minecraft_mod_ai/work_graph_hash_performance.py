from __future__ import annotations

import sys
from contextvars import ContextVar
from functools import wraps
from typing import Any


_WORK_GRAPH_PROPOSAL_HASH_CACHE: ContextVar[
    dict[int, tuple[Any, str]] | None
] = ContextVar(
    "mmm_work_graph_proposal_hash_cache",
    default=None,
)


def harden(work_graph_module: Any, complete_spec_module: Any) -> None:
    """Hash each proposal once per synchronous work-graph compile.

    CompleteProposal is frozen, but its JSON payload can still be large. The graph
    builder asks for the same canonical proposal digest several times while compiling
    one immutable snapshot. Keep that digest only inside the current compile context;
    a later compile starts with an empty cache and therefore still detects nested
    payload mutation before reusing any validation proof.
    """

    proposal_cls = complete_spec_module.CompleteProposal
    current_hash = proposal_cls.calculate_hash
    if not getattr(current_hash, "_mmm_work_graph_hash_cache", False):

        @wraps(current_hash)
        def calculate_hash(self: Any) -> str:
            cache = _WORK_GRAPH_PROPOSAL_HASH_CACHE.get()
            if cache is None:
                return current_hash(self)
            key = id(self)
            cached = cache.get(key)
            if cached is not None and cached[0] is self:
                return cached[1]
            digest = current_hash(self)
            cache[key] = (self, digest)
            return digest

        calculate_hash._mmm_work_graph_hash_cache = True
        calculate_hash.__wrapped__ = current_hash
        proposal_cls.calculate_hash = calculate_hash

    current_build = work_graph_module.build_production_work_plan
    if getattr(current_build, "_mmm_proposal_hash_scope", False):
        return

    @wraps(current_build)
    def build_production_work_plan(*args: Any, **kwargs: Any):
        token = _WORK_GRAPH_PROPOSAL_HASH_CACHE.set({})
        try:
            return current_build(*args, **kwargs)
        finally:
            _WORK_GRAPH_PROPOSAL_HASH_CACHE.reset(token)

    build_production_work_plan._mmm_proposal_hash_scope = True
    build_production_work_plan.__wrapped__ = current_build
    work_graph_module.build_production_work_plan = build_production_work_plan

    # Retarget only MMM modules that captured the prior build function during
    # bootstrap. Avoid touching third-party lazy modules via getattr side effects.
    for module_name, loaded in tuple(sys.modules.items()):
        if not (
            module_name == "minecraft_mod_ai"
            or module_name.startswith("minecraft_mod_ai.")
        ):
            continue
        if loaded is None:
            continue
        try:
            namespace = vars(loaded)
        except TypeError:
            continue
        if namespace.get("build_production_work_plan") is current_build:
            namespace["build_production_work_plan"] = build_production_work_plan


__all__ = ["harden"]
