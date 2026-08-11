from __future__ import annotations

from functools import wraps
from typing import Any, Mapping


def _matched_acceptance_refs(playtest: Any) -> set[str]:
    if not isinstance(playtest, Mapping) or playtest.get("status") != "PASS":
        return set()
    results = playtest.get("results")
    if not isinstance(results, list):
        return set()
    refs: set[str] = set()
    for item in results:
        if not isinstance(item, Mapping) or item.get("action") != "wait_for":
            continue
        result = item.get("result")
        params = item.get("params")
        if (
            not isinstance(result, Mapping)
            or result.get("matched") is not True
            or not isinstance(params, Mapping)
        ):
            continue
        raw: list[Any] = []
        if "acceptance_ref" in params:
            raw.append(params.get("acceptance_ref"))
        many = params.get("acceptance_refs")
        if isinstance(many, list):
            raw.extend(many)
        for value in raw:
            if isinstance(value, str) and value.startswith("acceptance:"):
                refs.add(value)
    return refs


def install(
    atomic_module: Any,
    quality_evidence_module: Any,
    orchestrator_module: Any | None = None,
) -> None:
    """Bind each atom to the objective verifier appropriate for that requirement."""

    original = quality_evidence_module.compile_quality_evidence
    if getattr(original, "_mmm_atomic_playtest_evidence", False):
        return

    @wraps(original)
    def compile_with_atomic_evidence(
        contract: Mapping[str, Any],
        proposal_hash: str,
        *args: Any,
        **kwargs: Any,
    ):
        result = original(contract, proposal_hash, *args, **kwargs)
        game_design = kwargs.get("game_design")
        ir = (
            game_design.get("_atomic_requirement_ir")
            if isinstance(game_design, Mapping)
            else None
        )
        if ir is None:
            return result
        if (
            not isinstance(ir, Mapping)
            or ir.get("schema_version") != atomic_module.SCHEMA
            or ir.get("unresolved_atom_ids") != []
        ):
            result.pop("correctness", None)
            return result

        matched_runtime = _matched_acceptance_refs(
            kwargs.get("playtest_receipt")
        )
        atoms = ir.get("atoms")
        if not isinstance(atoms, list) or not atoms:
            result.pop("correctness", None)
            return result

        uncovered: list[str] = []
        evidence_refs: set[str] = set()
        for atom in atoms:
            if not isinstance(atom, Mapping):
                uncovered.append("invalid")
                continue
            routes = atom.get("evidence_dimensions")
            if not isinstance(routes, list) or not routes:
                uncovered.append(str(atom.get("atom_id", "invalid")))
                continue
            acceptance_refs = {
                str(value)
                for value in atom.get("acceptance_refs", [])
                if isinstance(value, str)
            }
            atom_ok = True
            for route in routes:
                if route == "runtime":
                    matched = acceptance_refs & matched_runtime
                    if not matched:
                        atom_ok = False
                        break
                    evidence_refs.update(
                        "atomic-runtime:" + value for value in matched
                    )
                else:
                    receipt = result.get(str(route))
                    if not isinstance(receipt, Mapping):
                        atom_ok = False
                        break
                    evidence_refs.add(
                        "atomic-dimension:"
                        + str(atom.get("atom_id", ""))
                        + ":"
                        + str(route)
                    )
            if not atom_ok:
                uncovered.append(str(atom.get("atom_id", "invalid")))

        correctness = result.get("correctness")
        if uncovered or not isinstance(correctness, Mapping):
            result.pop("correctness", None)
            return result

        result["correctness"] = quality_evidence_module._quality_receipt(
            dimension_id="correctness",
            route_ref=str(correctness["route_ref"]),
            proposal_hash=proposal_hash,
            evidence_refs=[
                *correctness.get("evidence_refs", []),
                *sorted(evidence_refs),
            ],
            observed_sources=[
                correctness,
                kwargs.get("playtest_receipt"),
                *(
                    result[route]
                    for route in sorted(
                        {
                            str(route)
                            for atom in atoms
                            if isinstance(atom, Mapping)
                            for route in atom.get("evidence_dimensions", [])
                            if route != "runtime" and route in result
                        }
                    )
                ),
            ],
        )
        return result

    compile_with_atomic_evidence._mmm_atomic_playtest_evidence = True
    compile_with_atomic_evidence._mmm_atomic_routed_evidence = True
    quality_evidence_module.compile_quality_evidence = compile_with_atomic_evidence
    if orchestrator_module is not None:
        orchestrator_module.compile_quality_evidence = compile_with_atomic_evidence
