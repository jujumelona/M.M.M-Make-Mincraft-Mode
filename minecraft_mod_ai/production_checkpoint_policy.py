from __future__ import annotations

"""Implementation fingerprints for resumable production work.

A durable generation receipt is reusable only when the code/configuration that
owns that production stage is unchanged.  Proposal hashes alone are insufficient
during active development because resume=True may otherwise replay outputs made
by an older generator implementation.
"""

import hashlib
from pathlib import Path
from typing import Any


def _file_digest(value: Any) -> str:
    if isinstance(value, Path):
        path = value
    else:
        raw = getattr(value, "__file__", None)
        if not raw:
            return "missing"
        path = Path(str(raw))
    try:
        return hashlib.sha256(path.resolve().read_bytes()).hexdigest()
    except OSError:
        return "missing"


def _production_inputs(stage: str) -> tuple[Any, ...]:
    from . import (
        artifact_graph_executor,
        artifact_materializer,
        canonical_generators,
        extended_content_generator,
        geckolib_generation_contract,
        geckolib_generator,
        host_generator_catalog_migration,
        host_typed_state_access,
        integrity_dispatcher,
        model_registry,
        platform_backend_contract,
        production_routing_contract,
        production_state_compiler,
        project_edit,
        registered_leaf_binding,
        request_runtime_ownership,
        request_runtime_records,
        resource_asset_production,
        resource_contracts,
        resource_image_pipeline,
        resource_prompt_compiler,
        scalable_generator,
        structured_state_runtime,
        system_pack_generator,
        typed_plan_ir,
        typed_plan_java,
        typed_plan_production,
    )
    from .model_adapters import image_diffusion

    common: tuple[Any, ...] = (
        production_routing_contract,
        platform_backend_contract,
    )
    state_compiler_inputs = (
        production_state_compiler, structured_state_runtime, host_typed_state_access,
    )
    if stage == "prepare":
        return (*common, scalable_generator, project_edit, *state_compiler_inputs)
    if stage == "content":
        return (
            *common,
            artifact_graph_executor,
            artifact_materializer,
            canonical_generators,
            integrity_dispatcher,
            request_runtime_ownership,
            request_runtime_records,
            registered_leaf_binding,
            host_generator_catalog_migration,
            extended_content_generator,
        )
    if stage == "system":
        return (*common, system_pack_generator)
    if stage == "entity":
        return (
            *common,
            geckolib_generation_contract,
            geckolib_generator,
        )
    if stage == "host":
        return (
            *common,
            *state_compiler_inputs,
            typed_plan_ir,
            typed_plan_java,
            typed_plan_production,
        )
    if stage == "assets":
        registry_yaml = Path(model_registry.__file__).resolve().parent / "config" / "model_registry.yaml"
        return (
            *common,
            model_registry,
            registry_yaml,
            resource_asset_production,
            resource_contracts,
            resource_image_pipeline,
            resource_prompt_compiler,
            image_diffusion,
        )
    raise ValueError(f"Unsupported production checkpoint stage: {stage}")


def production_implementation_fingerprint(stage: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"mmm/production-implementation-v1\0")
    digest.update(str(stage).encode("utf-8"))
    digest.update(b"\0")
    if stage == 'content':
        from .construction_implementation import construction_implementation_fingerprint
        digest.update(construction_implementation_fingerprint().encode('ascii'))
        digest.update(b'\0')
    for value in _production_inputs(stage):
        name = (
            value.as_posix()
            if isinstance(value, Path)
            else str(getattr(value, "__name__", value))
        )
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_digest(value).encode("ascii"))
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


__all__ = ["production_implementation_fingerprint"]
