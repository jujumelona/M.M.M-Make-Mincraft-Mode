from __future__ import annotations

import hashlib
import sys
from contextvars import ContextVar
from functools import wraps
from typing import Any

from .json_stream import iter_canonical_json


_WORK_GRAPH_PROPOSAL_HASH_CACHE: ContextVar[
    dict[int, tuple[Any, str]] | None
] = ContextVar(
    "mmm_work_graph_proposal_hash_cache",
    default=None,
)
_WORK_GRAPH_VALIDATED_MODULES: ContextVar[
    dict[int, tuple[Any, Any]] | None
] = ContextVar(
    "mmm_work_graph_validated_modules",
    default=None,
)


def _buffered_canonical_json_sha256(value: Any) -> str:
    """Hash canonical JSON with bounded string coalescing.

    ``iter_canonical_json`` intentionally emits punctuation and scalar fragments
    separately so callers never need one project-sized JSON string. Feeding every
    tiny fragment directly to hashlib adds substantial Python call/UTF-8 overhead on
    large proposals. Coalesce at most 16K Unicode characters before encoding; the
    canonical byte stream and digest stay identical while memory remains bounded.
    """

    digest = hashlib.sha256()
    buffer: list[str] = []
    buffered_characters = 0
    for text in iter_canonical_json(value):
        buffer.append(text)
        buffered_characters += len(text)
        if buffered_characters >= 16 * 1024:
            digest.update("".join(buffer).encode("utf-8"))
            buffer.clear()
            buffered_characters = 0
    if buffer:
        digest.update("".join(buffer).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


_buffered_canonical_json_sha256._mmm_buffered_canonical_hash = True


def harden(work_graph_module: Any, complete_spec_module: Any) -> None:
    """Reuse proven proposal work only inside one synchronous graph compile.

    CompleteProposal is frozen, but nested JSON payloads can still be mutated. Every
    graph compile therefore starts with empty invocation-local caches. The proposal
    digest is computed once per compile, and modules are reused only after the exact
    proposal has successfully validated under the exact same ScalePolicy object.
    Separate execution modules continue through their normal validator.
    """

    proposal_cls = complete_spec_module.CompleteProposal
    module_cls = complete_spec_module.ProductionModule

    current_canonical_hash = complete_spec_module.canonical_json_sha256
    if not getattr(
        current_canonical_hash,
        "_mmm_buffered_canonical_hash",
        False,
    ):
        _buffered_canonical_json_sha256.__wrapped__ = current_canonical_hash
        complete_spec_module.canonical_json_sha256 = _buffered_canonical_json_sha256

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

    current_module_validate = module_cls.validate
    if not getattr(
        current_module_validate,
        "_mmm_work_graph_validated_module_cache",
        False,
    ):

        @wraps(current_module_validate)
        def validate_module(self: Any, *, policy: Any = None) -> None:
            cache = _WORK_GRAPH_VALIDATED_MODULES.get()
            if cache is not None and policy is not None:
                cached = cache.get(id(self))
                if (
                    cached is not None
                    and cached[0] is self
                    and cached[1] is policy
                ):
                    return
            current_module_validate(self, policy=policy)
            if cache is not None and policy is not None:
                cache[id(self)] = (self, policy)

        validate_module._mmm_work_graph_validated_module_cache = True
        validate_module.__wrapped__ = current_module_validate
        module_cls.validate = validate_module

    current_proposal_validate = proposal_cls.validate
    if not getattr(
        current_proposal_validate,
        "_mmm_work_graph_validated_module_transfer",
        False,
    ):

        @wraps(current_proposal_validate)
        def validate_proposal(self: Any, *, policy: Any = None) -> None:
            current_proposal_validate(self, policy=policy)
            cache = _WORK_GRAPH_VALIDATED_MODULES.get()
            if cache is None or policy is None:
                return
            # The authoritative proposal validator has either validated this payload
            # now or accepted its current full-payload hash proof. No user code runs
            # between this return and the graph builder's redundant module loop.
            for module in self.modules:
                cache[id(module)] = (module, policy)

        validate_proposal._mmm_work_graph_validated_module_transfer = True
        validate_proposal.__wrapped__ = current_proposal_validate
        proposal_cls.validate = validate_proposal

    current_build = work_graph_module.build_production_work_plan
    if getattr(current_build, "_mmm_proposal_hash_scope", False):
        return

    @wraps(current_build)
    def build_production_work_plan(*args: Any, **kwargs: Any):
        hash_token = _WORK_GRAPH_PROPOSAL_HASH_CACHE.set({})
        validation_token = _WORK_GRAPH_VALIDATED_MODULES.set({})
        try:
            return current_build(*args, **kwargs)
        finally:
            _WORK_GRAPH_VALIDATED_MODULES.reset(validation_token)
            _WORK_GRAPH_PROPOSAL_HASH_CACHE.reset(hash_token)

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
