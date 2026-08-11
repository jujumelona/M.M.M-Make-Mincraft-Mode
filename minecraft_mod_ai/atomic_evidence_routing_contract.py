from __future__ import annotations

from functools import wraps
from typing import Any, Mapping


_ALLOWED = frozenset(
    {
        "runtime",
        "visual_3d",
        "audio",
        "state_save_migration",
        "multiplayer",
        "performance",
        "accessibility",
        "ai_voice",
        "research",
        "build",
    }
)
_MULTIPLAYER_KINDS = frozenset({"networking", "party", "guild"})
_VISUAL_KINDS = frozenset({"entity_model", "animation", "model", "texture"})
_RESEARCH_TERMS = (
    "license", "licence", "compatible", "compatibility", "fabric", "yarn",
    "dependency", "version", "라이선스", "호환", "의존성", "버전",
)
_BUILD_TERMS = (
    "compile", "build", "gradle", "java 17", "컴파일", "빌드",
)


def _module_kinds(proposal: Any) -> dict[str, str]:
    return {
        f"implementation:module:{item.module_id}": str(item.kind).casefold()
        for item in getattr(proposal, "modules", ())
    }


def _routes_for_atom(
    proposal: Any,
    atom: Mapping[str, Any],
    production_contract_module: Any,
) -> list[str]:
    text = str(atom.get("text", ""))
    lowered = " " + text.casefold() + " "
    refs = [str(value) for value in atom.get("implementation_refs", [])]
    module_kinds = _module_kinds(proposal)
    routes: set[str] = set()

    for dimension in (
        "visual_3d",
        "audio",
        "state_save_migration",
        "multiplayer",
        "performance",
        "accessibility",
        "ai_voice",
    ):
        try:
            if production_contract_module._text_triggers_dimension(text, dimension):
                routes.add(dimension)
        except Exception:
            pass

    has_module = False
    for ref in refs:
        if ref.startswith("implementation:asset:"):
            routes.add("visual_3d")
        elif ref.startswith("implementation:audio:"):
            routes.add("audio")
        elif ref.startswith("implementation:module:"):
            has_module = True
            kind = module_kinds.get(ref, "")
            if kind in _MULTIPLAYER_KINDS:
                routes.add("multiplayer")
            if kind in _VISUAL_KINDS:
                routes.add("visual_3d")
            if kind == "integration":
                module_id = ref.rsplit(":", 1)[-1]
                module = next(
                    (
                        value
                        for value in getattr(proposal, "modules", ())
                        if str(value.module_id) == module_id
                    ),
                    None,
                )
                if module is not None and str(
                    getattr(module, "config", {}).get("integration_type", "")
                ) == "mmm_local_ai_sidecar":
                    routes.add("ai_voice")

    if any(term in lowered for term in _RESEARCH_TERMS):
        routes.add("research")
    if any(term in lowered for term in _BUILD_TERMS):
        routes.add("build")

    infrastructure_only = bool(routes) and routes <= {"research", "build"}
    media_only = bool(routes) and routes <= {"visual_3d", "audio"} and not has_module
    if has_module and not infrastructure_only:
        routes.add("runtime")
    elif not routes:
        routes.add("runtime")
    elif not infrastructure_only and not media_only and not (
        routes <= {"visual_3d", "audio"}
    ):
        routes.add("runtime")

    return [
        value
        for value in (
            "runtime",
            "visual_3d",
            "audio",
            "state_save_migration",
            "multiplayer",
            "performance",
            "accessibility",
            "ai_voice",
            "research",
            "build",
        )
        if value in routes
    ]


def _route_ir(
    proposal: Any,
    ir: dict[str, Any],
    atomic_module: Any,
    production_contract_module: Any,
) -> dict[str, Any]:
    atoms = []
    for raw in ir.get("atoms", []):
        atom = dict(raw)
        atom["evidence_dimensions"] = _routes_for_atom(
            proposal,
            atom,
            production_contract_module,
        )
        atoms.append(atom)
    routed = {**ir, "atoms": atoms, "ir_sha256": ""}
    routed["ir_sha256"] = atomic_module._hash_without(routed, "ir_sha256")
    return routed


def install(
    atomic_module: Any,
    production_contract_module: Any,
) -> None:
    """Attach the narrowest objective verifier route to every requirement atom."""

    current_compile = atomic_module.compile_ir
    if not getattr(current_compile, "_mmm_atomic_evidence_routes", False):
        @wraps(current_compile)
        def compile_ir(proposal: Any):
            return _route_ir(
                proposal,
                current_compile(proposal),
                atomic_module,
                production_contract_module,
            )

        compile_ir._mmm_atomic_evidence_routes = True
        atomic_module.compile_ir = compile_ir

    current_review = atomic_module.semantic_review
    if not getattr(current_review, "_mmm_atomic_evidence_routes", False):
        @wraps(current_review)
        def semantic_review(router: Any, proposal: Any, ir: dict[str, Any]):
            reviewed = current_review(router, proposal, ir)
            return _route_ir(
                proposal,
                reviewed,
                atomic_module,
                production_contract_module,
            )

        semantic_review._mmm_atomic_evidence_routes = True
        atomic_module.semantic_review = semantic_review

    current_validate = atomic_module.validate_ir
    if not getattr(current_validate, "_mmm_atomic_evidence_routes", False):
        @wraps(current_validate)
        def validate_ir(proposal: Any):
            ir = current_validate(proposal)
            for atom in ir.get("atoms", []):
                routes = atom.get("evidence_dimensions")
                if (
                    not isinstance(routes, list)
                    or not routes
                    or len(routes) != len(set(routes))
                    or any(value not in _ALLOWED for value in routes)
                ):
                    raise atomic_module.AtomicRequirementError(
                        "Atomic requirement has an invalid evidence route."
                    )
            return ir

        validate_ir._mmm_atomic_evidence_routes = True
        atomic_module.validate_ir = validate_ir
