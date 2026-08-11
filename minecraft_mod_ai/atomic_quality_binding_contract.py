from __future__ import annotations

from functools import wraps
from typing import Any, Mapping


def install(
    atomic_module: Any,
    quality_evidence_module: Any,
) -> None:
    """Make atomic requirement coverage part of independent correctness evidence."""

    original = quality_evidence_module.compile_quality_evidence
    if getattr(original, "_mmm_atomic_correctness_evidence", False):
        return

    def valid_ir(
        ir: Any,
        contract: Mapping[str, Any],
    ) -> bool:
        if (
            not isinstance(ir, Mapping)
            or ir.get("schema_version") != atomic_module.SCHEMA
            or ir.get("prompt_sha256")
            != atomic_module._sha(str(contract.get("requested_prompt", "")))
            or ir.get("ir_sha256")
            != atomic_module._hash_without(ir, "ir_sha256")
            or ir.get("unresolved_atom_ids") != []
        ):
            return False
        atoms = ir.get("atoms")
        if (
            not isinstance(atoms, list)
            or not atoms
            or ir.get("atom_count") != len(atoms)
        ):
            return False
        implementation_refs = {
            str(item.get("implementation_ref"))
            for item in contract.get("implementation_catalog", [])
            if isinstance(item, Mapping)
        }
        acceptance_refs = {
            f"acceptance:{index:08d}"
            for index, _item in enumerate(
                contract.get("acceptance_catalog", [])
            )
        }
        for atom in atoms:
            if (
                not isinstance(atom, Mapping)
                or atom.get("status") != "COVERED"
                or not atom.get("implementation_refs")
                or not atom.get("acceptance_refs")
                or not set(atom["implementation_refs"])
                <= implementation_refs
                or not set(atom["acceptance_refs"])
                <= acceptance_refs
            ):
                return False
        return True

    @wraps(original)
    def compile_with_atomic_ir(
        contract: Mapping[str, Any],
        proposal_hash: str,
        *args: Any,
        **kwargs: Any,
    ):
        result = original(
            contract,
            proposal_hash,
            *args,
            **kwargs,
        )
        game_design = kwargs.get("game_design")
        ir = (
            game_design.get("_atomic_requirement_ir")
            if isinstance(game_design, Mapping)
            else None
        )
        correctness = result.get("correctness")
        if (
            not isinstance(correctness, Mapping)
            or not valid_ir(ir, contract)
        ):
            result.pop("correctness", None)
            return result

        assert isinstance(ir, Mapping)
        refs = [
            *correctness.get("evidence_refs", []),
            "atomic-requirements:" + str(ir["ir_sha256"]),
        ]
        result["correctness"] = quality_evidence_module._quality_receipt(
            dimension_id="correctness",
            route_ref=str(correctness["route_ref"]),
            proposal_hash=proposal_hash,
            evidence_refs=refs,
            observed_sources=[correctness, ir],
        )
        return result

    compile_with_atomic_ir._mmm_atomic_correctness_evidence = True
    quality_evidence_module.compile_quality_evidence = (
        compile_with_atomic_ir
    )
