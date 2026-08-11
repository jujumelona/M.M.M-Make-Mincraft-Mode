from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, is_dataclass, replace
from functools import wraps
from typing import Any, Mapping

SCHEMA = "mmm/atomic-requirement-ir-v1"
_MAX_ATOM_BYTES = 1024
_MAX_IMPL_CANDIDATES = 48
_MAX_ACCEPTANCE_CANDIDATES = 24
_TOKEN = re.compile(r"[A-Za-z0-9_]+|[가-힣]+|[\u3040-\u30ff\u3400-\u9fff]+", re.UNICODE)
_SENTENCE = re.compile(r"[^.!?。！？;\n]+(?:[.!?。！？;]+|$)", re.UNICODE)
_STOP = frozenset({
    "a","an","the","and","or","to","of","for","with","in","on","is","are","be",
    "make","create","minecraft","mod","기능","모드","마인크래프트","만들","만들어",
    "추가","사용","그리고","및","으로","에서","하는","되게","해줘","해주세요",
})


class AtomicRequirementError(ValueError):
    pass


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else _canonical(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _hash_without(value: Mapping[str, Any], field: str) -> str:
    copy = dict(value)
    copy[field] = ""
    return _sha(copy)


def _object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raw = getattr(value, "__dict__", None)
    return dict(raw) if isinstance(raw, dict) else {"value": str(value)}


def _split_range(prompt: str, start: int, end: int) -> list[tuple[int, int]]:
    if len(prompt[start:end].encode("utf-8")) <= _MAX_ATOM_BYTES:
        return [(start, end)]
    result: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        encoded = 0
        limit = cursor
        preferred: int | None = None
        while limit < end:
            size = len(prompt[limit].encode("utf-8"))
            if encoded + size > _MAX_ATOM_BYTES:
                break
            encoded += size
            limit += 1
            if prompt[limit - 1].isspace() or prompt[limit - 1] in {",","，","、",":","："}:
                preferred = limit
        cut = preferred if preferred is not None and limit < end else limit
        if cut <= cursor:
            cut = min(end, cursor + 1)
        left, right = cursor, cut
        while left < right and prompt[left].isspace():
            left += 1
        while right > left and prompt[right - 1].isspace():
            right -= 1
        if left < right:
            result.append((left, right))
        cursor = cut
    return result


def _atom_ranges(prompt: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for match in _SENTENCE.finditer(prompt):
        start, end = match.span()
        while start < end and prompt[start].isspace():
            start += 1
        while end > start and prompt[end - 1].isspace():
            end -= 1
        if start < end:
            result.extend(_split_range(prompt, start, end))
    if not result and prompt.strip():
        start = len(prompt) - len(prompt.lstrip())
        end = len(prompt.rstrip())
        result.extend(_split_range(prompt, start, end))
    return result


def _features(text: str) -> set[str]:
    result: set[str] = set()
    for token in _TOKEN.findall(text.casefold()):
        if len(token) <= 1 or token in _STOP:
            continue
        result.add("w:" + token)
        if any(
            "\u3400" <= ch <= "\u9fff"
            or "\u3040" <= ch <= "\u30ff"
            or "\uac00" <= ch <= "\ud7a3"
            for ch in token
        ):
            result.update("b:" + token[i:i + 2] for i in range(max(0, len(token) - 1)))
            result.update("t:" + token[i:i + 3] for i in range(max(0, len(token) - 2)))
    return result


def _score(left: str, right: str) -> float:
    a, b = _features(left), _features(right)
    if not a or not b:
        return 0.0
    common = a & b
    if not common:
        return 0.0
    weight = sum(3.0 if item.startswith("w:") else 1.5 if item.startswith("t:") else 1.0 for item in common)
    return weight / math.sqrt(max(1.0, float(len(a) * len(b))))


def _implementations(proposal: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in getattr(proposal, "modules", ()):
        result[f"implementation:module:{item.module_id}"] = _canonical(_object(item))
    for item in getattr(proposal, "assets", ()):
        result[f"implementation:asset:{item.asset_id}"] = _canonical(_object(item))
    for item in getattr(proposal, "audio", ()):
        result[f"implementation:audio:{item.sound_id}"] = _canonical(_object(item))
    return result


def _acceptances(proposal: Any) -> dict[str, str]:
    return {
        f"acceptance:{index:08d}": str(text)
        for index, text in enumerate(getattr(proposal, "acceptance_tests", ()))
    }


def _rank(text: str, catalog: Mapping[str, str], limit: int) -> list[tuple[float, str]]:
    ranked = sorted(
        ((_score(text, descriptor), ref) for ref, descriptor in catalog.items()),
        key=lambda item: (-item[0], item[1]),
    )
    return [item for item in ranked[:limit] if item[0] > 0.0]


def compile_ir(proposal: Any) -> dict[str, Any]:
    prompt = str(getattr(proposal, "requested_prompt", ""))
    implementations = _implementations(proposal)
    acceptances = _acceptances(proposal)
    if not prompt.strip() or not implementations or not acceptances:
        raise AtomicRequirementError("Atomic requirement IR needs request, implementations, and acceptance tests.")
    prompt_hash = _sha(prompt)
    atoms: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for index, (start, end) in enumerate(_atom_ranges(prompt)):
        text = prompt[start:end]
        atom_id = "atom:" + hashlib.sha256(
            _canonical({"prompt_sha256": prompt_hash, "index": index, "start": start, "end": end, "text": text}).encode("utf-8")
        ).hexdigest()
        impl = [ref for score, ref in _rank(text, implementations, 8) if score >= 0.08]
        tests = [ref for score, ref in _rank(text, acceptances, 4) if score >= 0.06]
        status = "COVERED" if impl and tests else "REVIEW_REQUIRED"
        if status != "COVERED":
            unresolved.append(atom_id)
        atoms.append({
            "atom_id": atom_id,
            "index": index,
            "char_start": start,
            "char_end": end,
            "text": text,
            "text_sha256": _sha(text),
            "implementation_refs": impl,
            "acceptance_refs": tests,
            "coverage_origin": "deterministic" if status == "COVERED" else "unresolved",
            "status": status,
        })
    result = {
        "schema_version": SCHEMA,
        "prompt_sha256": prompt_hash,
        "prompt_char_length": len(prompt),
        "atom_count": len(atoms),
        "implementation_catalog_sha256": _sha(implementations),
        "acceptance_catalog_sha256": _sha(acceptances),
        "atoms": atoms,
        "unresolved_atom_ids": unresolved,
        "review_policy": {
            "deterministic_first": True,
            "semantic_review_only_when_unresolved": True,
            "reviewer_may_create_implementation": False,
            "release_requires_zero_unresolved": True,
        },
        "ir_sha256": "",
    }
    result["ir_sha256"] = _hash_without(result, "ir_sha256")
    return result


def _extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise AtomicRequirementError("Coverage reviewer did not return JSON.")


def _root_hints(proposal: Any, implementations: Mapping[str, str]) -> list[str]:
    contract = getattr(proposal, "game_design", {}).get("_production_contract")
    if not isinstance(contract, dict):
        return []
    root_refs: set[str] = set()
    root_requirements = {
        str(item.get("requirement_ref"))
        for item in contract.get("requirement_catalog", [])
        if isinstance(item, dict) and item.get("source") == "requested_prompt"
    }
    for group in contract.get("coverage_groups", []):
        if not isinstance(group, dict) or str(group.get("requirement_ref")) not in root_requirements:
            continue
        root_refs.update(str(ref) for ref in group.get("implementation_refs", []) if str(ref) in implementations)
    return sorted(root_refs)


def semantic_review(router: Any, proposal: Any, ir: dict[str, Any]) -> dict[str, Any]:
    unresolved = set(ir["unresolved_atom_ids"])
    if not unresolved:
        return ir
    implementations, acceptances = _implementations(proposal), _acceptances(proposal)
    hints = _root_hints(proposal, implementations)
    review_atoms: list[dict[str, Any]] = []
    for atom in ir["atoms"]:
        if atom["atom_id"] not in unresolved:
            continue
        impl_refs = list(dict.fromkeys([
            *hints,
            *(ref for _score_value, ref in _rank(atom["text"], implementations, _MAX_IMPL_CANDIDATES)),
        ]))[:_MAX_IMPL_CANDIDATES]
        if not impl_refs:
            impl_refs = list(implementations)[:_MAX_IMPL_CANDIDATES]
        acceptance_refs = [ref for _score_value, ref in _rank(atom["text"], acceptances, _MAX_ACCEPTANCE_CANDIDATES)]
        if not acceptance_refs:
            acceptance_refs = list(acceptances)[:_MAX_ACCEPTANCE_CANDIDATES]
        review_atoms.append({
            "atom_id": atom["atom_id"],
            "text": atom["text"],
            "implementation_candidates": {ref: implementations[ref] for ref in impl_refs},
            "acceptance_candidates": {ref: acceptances[ref] for ref in acceptance_refs},
        })
    request = {
        "schema_version": "mmm/atomic-coverage-review-request-v1",
        "rule": (
            "Map each atom only to existing IDs that genuinely satisfy it. Never invent "
            "IDs or features. If the current plan does not implement the atom, supported=false."
        ),
        "atoms": review_atoms,
        "response_contract": {
            "mappings": [{
                "atom_id": "existing atom id",
                "supported": True,
                "implementation_refs": ["existing implementation ref"],
                "acceptance_refs": ["existing acceptance ref"],
            }]
        },
    }
    response = router.generate_text(
        "planner",
        [
            {"role": "system", "content": (
                "You are a narrow coverage verifier. You cannot create features, files, "
                "IDs, or evidence. Unsupported requirements must remain unsupported."
            )},
            {"role": "user", "content": _canonical(request)},
        ],
        response_format="json",
    )
    value = _extract_json(response)
    if set(value) != {"mappings"} or not isinstance(value["mappings"], list):
        raise AtomicRequirementError("Coverage reviewer contract is invalid.")
    atoms = {item["atom_id"]: dict(item) for item in ir["atoms"]}
    seen: set[str] = set()
    for mapping in value["mappings"]:
        if not isinstance(mapping, dict) or set(mapping) != {
            "atom_id","supported","implementation_refs","acceptance_refs"
        }:
            raise AtomicRequirementError("Coverage reviewer mapping fields are invalid.")
        atom_id = str(mapping["atom_id"])
        if atom_id not in unresolved or atom_id in seen:
            raise AtomicRequirementError("Coverage reviewer returned unknown/duplicate atom.")
        seen.add(atom_id)
        if type(mapping["supported"]) is not bool:
            raise AtomicRequirementError("Coverage reviewer supported must be boolean.")
        impl_refs, test_refs = mapping["implementation_refs"], mapping["acceptance_refs"]
        if not isinstance(impl_refs, list) or not isinstance(test_refs, list):
            raise AtomicRequirementError("Coverage reviewer refs must be lists.")
        if any(ref not in implementations for ref in impl_refs):
            raise AtomicRequirementError("Coverage reviewer invented an implementation ref.")
        if any(ref not in acceptances for ref in test_refs):
            raise AtomicRequirementError("Coverage reviewer invented an acceptance ref.")
        atom = atoms[atom_id]
        if mapping["supported"] and impl_refs and test_refs:
            atom["implementation_refs"] = list(dict.fromkeys(str(ref) for ref in impl_refs))
            atom["acceptance_refs"] = list(dict.fromkeys(str(ref) for ref in test_refs))
            atom["status"] = "COVERED"
        else:
            atom["status"] = "UNSUPPORTED"
        atom["coverage_origin"] = "semantic_reviewer"
    ordered = [atoms[item["atom_id"]] for item in ir["atoms"]]
    missing = [item["atom_id"] for item in ordered if item["status"] != "COVERED"]
    updated = {**ir, "atoms": ordered, "unresolved_atom_ids": missing, "ir_sha256": ""}
    updated["ir_sha256"] = _hash_without(updated, "ir_sha256")
    return updated


def validate_ir(proposal: Any) -> dict[str, Any]:
    design = getattr(proposal, "game_design", {})
    ir = design.get("_atomic_requirement_ir") if isinstance(design, dict) else None
    if not isinstance(ir, dict) or ir.get("schema_version") != SCHEMA:
        raise AtomicRequirementError("Complete proposal is missing the code-owned atomic requirement IR; re-plan it.")
    expected = {
        "schema_version","prompt_sha256","prompt_char_length","atom_count",
        "implementation_catalog_sha256","acceptance_catalog_sha256","atoms",
        "unresolved_atom_ids","review_policy","ir_sha256",
    }
    if set(ir) != expected:
        raise AtomicRequirementError("Atomic requirement IR fields are invalid.")
    if ir["prompt_sha256"] != _sha(str(getattr(proposal, "requested_prompt", ""))):
        raise AtomicRequirementError("Atomic requirement IR request binding changed.")
    if ir["implementation_catalog_sha256"] != _sha(_implementations(proposal)):
        raise AtomicRequirementError("Atomic requirement implementation catalog changed.")
    if ir["acceptance_catalog_sha256"] != _sha(_acceptances(proposal)):
        raise AtomicRequirementError("Atomic requirement acceptance catalog changed.")
    if ir["ir_sha256"] != _hash_without(ir, "ir_sha256"):
        raise AtomicRequirementError("Atomic requirement IR hash mismatch.")
    atoms = ir.get("atoms")
    if not isinstance(atoms, list) or not atoms or ir.get("atom_count") != len(atoms):
        raise AtomicRequirementError("Atomic requirement atom catalog is invalid.")
    unresolved = [
        str(item.get("atom_id"))
        for item in atoms
        if not isinstance(item, dict) or item.get("status") != "COVERED"
    ]
    if ir.get("unresolved_atom_ids") != unresolved:
        raise AtomicRequirementError("Atomic requirement unresolved catalog mismatch.")
    if unresolved:
        raise AtomicRequirementError("Atomic requirement coverage is incomplete: " + ", ".join(unresolved[:8]))
    impl_refs, acceptance_refs = set(_implementations(proposal)), set(_acceptances(proposal))
    for atom in atoms:
        if not atom.get("implementation_refs") or not atom.get("acceptance_refs"):
            raise AtomicRequirementError("Covered atom lacks implementation or acceptance refs.")
        if not set(atom["implementation_refs"]) <= impl_refs:
            raise AtomicRequirementError("Atomic requirement uses an unknown implementation ref.")
        if not set(atom["acceptance_refs"]) <= acceptance_refs:
            raise AtomicRequirementError("Atomic requirement uses an unknown acceptance ref.")
    return ir


def install(complete_planner_module: Any, orchestrator_module: Any) -> None:
    planner_cls = complete_planner_module.CompleteGameDesignPlanner
    planner_original = planner_cls.plan
    if not getattr(planner_original, "_mmm_atomic_requirement_ir", False):
        @wraps(planner_original)
        def planned(self: Any, *args: Any, **kwargs: Any):
            proposal = planner_original(self, *args, **kwargs)
            ir = compile_ir(proposal)
            if ir["unresolved_atom_ids"]:
                ir = semantic_review(self.router, proposal, ir)
            if ir["unresolved_atom_ids"]:
                missing = [
                    atom["text"] for atom in ir["atoms"]
                    if atom["atom_id"] in set(ir["unresolved_atom_ids"])
                ]
                raise complete_planner_module.SpecValidationError(
                    "Planner left authoritative request atoms uncovered after bounded review: "
                    + " | ".join(missing[:6])
                )
            game_design = dict(proposal.game_design)
            game_design["_atomic_requirement_ir"] = ir
            return replace(proposal, game_design=game_design, approval_hash="").with_hash()

        planned._mmm_atomic_requirement_ir = True
        planner_cls.plan = planned

    orchestrator_cls = orchestrator_module.CompleteProductionOrchestrator
    execute_original = orchestrator_cls.execute
    if not getattr(execute_original, "_mmm_atomic_release_guard", False):
        @wraps(execute_original)
        def guarded(self: Any, proposal: Any, *args: Any, **kwargs: Any):
            parsed = (
                proposal
                if isinstance(proposal, orchestrator_module.CompleteProposal)
                else orchestrator_module.CompleteProposal.from_dict(proposal)
            )
            try:
                validate_ir(parsed)
            except AtomicRequirementError as exc:
                raise orchestrator_module.CompleteProductionError(str(exc)) from exc
            return execute_original(self, parsed, *args, **kwargs)

        guarded._mmm_atomic_release_guard = True
        orchestrator_cls.execute = guarded
