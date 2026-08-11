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
    """Require every authoritative atom to be backed by a matched runtime assertion."""

    original = quality_evidence_module.compile_quality_evidence
    if getattr(original, "_mmm_atomic_playtest_evidence", False):
        return

    @wraps(original)
    def compile_with_playtest_coverage(
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

        matched = _matched_acceptance_refs(kwargs.get("playtest_receipt"))
        atoms = ir.get("atoms")
        if not isinstance(atoms, list) or not atoms:
            result.pop("correctness", None)
            return result

        uncovered: list[str] = []
        used_refs: set[str] = set()
        for atom in atoms:
            if not isinstance(atom, Mapping):
                uncovered.append("invalid")
                continue
            refs = {
                str(value)
                for value in atom.get("acceptance_refs", [])
                if isinstance(value, str)
            }
            evidence_refs = refs & matched
            if not evidence_refs:
                uncovered.append(str(atom.get("atom_id", "invalid")))
            used_refs.update(evidence_refs)

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
                *(
                    "atomic-playtest:" + value
                    for value in sorted(used_refs)
                ),
            ],
            observed_sources=[correctness, kwargs.get("playtest_receipt")],
        )
        return result

    compile_with_playtest_coverage._mmm_atomic_playtest_evidence = True
    quality_evidence_module.compile_quality_evidence = compile_with_playtest_coverage
    if orchestrator_module is not None:
        orchestrator_module.compile_quality_evidence = compile_with_playtest_coverage
