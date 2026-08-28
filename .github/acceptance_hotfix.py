from pathlib import Path
import re

boundary_path = Path("minecraft_mod_ai/production_boundary_contract.py")
production_path = Path("minecraft_mod_ai/production_contract.py")
test_path = Path("tests/test_acceptance_hotfix_regression.py")

boundary = boundary_path.read_text(encoding="utf-8")
production = production_path.read_text(encoding="utf-8")

helper = """
def _sanitize_evidence_plan_public_acceptance(
    evidence_plan: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(evidence_plan, Mapping):
        return evidence_plan

    plan = dict(evidence_plan)
    acceptance_by_ref: dict[str, list[str]] = {}

    def clean_values(value: Any) -> list[str]:
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple)):
            values = [item for item in value if isinstance(item, str)]
        else:
            values = []

        clean: list[str] = []
        for value_item in values:
            candidate = value_item.strip()
            if not candidate:
                continue
            if _strict_public_acceptance(candidate):
                clean.append(candidate)
                continue
            for chunk in re.split(r"[;\r\n]+", candidate):
                chunk = chunk.strip(" \t-*•")
                if chunk and _strict_public_acceptance(chunk):
                    clean.append(chunk)
        return clean

    for catalog_key in ("request_catalog", "_evidence_request_catalog"):
        catalog = plan.get(catalog_key)
        if not isinstance(catalog, Mapping):
            continue
        clean_catalog = dict(catalog)
        requirements = catalog.get("requirements")
        if not isinstance(requirements, (list, tuple)):
            continue
        clean_requirements = []
        for index, requirement in enumerate(requirements):
            if not isinstance(requirement, Mapping):
                clean_requirements.append(requirement)
                continue
            clean_requirement = dict(requirement)
            raw_acceptance = requirement.get("acceptance")
            clean_acceptance = clean_values(raw_acceptance)
            if not clean_acceptance:
                clean_acceptance = [
                    "The requested behavior is present and observable in the generated Minecraft mod."
                ]
            clean_requirement["acceptance"] = clean_acceptance
            requirement_ref = str(
                requirement.get("requirement_id")
                or requirement.get("requirement_ref")
                or ""
            ).strip()
            print(
                "production boundary: requirement acceptance "
                f"catalog={catalog_key!r} index={index} requirement_ref={requirement_ref!r} "
                f"raw={raw_acceptance!r} sanitized={clean_acceptance!r}"
            )
            if requirement_ref:
                acceptance_by_ref[requirement_ref] = list(clean_acceptance)
            clean_requirements.append(clean_requirement)
        clean_catalog["requirements"] = clean_requirements
        plan[catalog_key] = clean_catalog

    bindings = plan.get("acceptance_release_bindings")
    if isinstance(bindings, (list, tuple)):
        clean_bindings = []
        for index, binding in enumerate(bindings):
            if not isinstance(binding, Mapping):
                clean_bindings.append(binding)
                continue
            clean_binding = dict(binding)
            requirement_ref = str(
                binding.get("requirement_ref")
                or binding.get("requirement_id")
                or ""
            ).strip()
            raw_acceptance = binding.get("acceptance")
            clean_acceptance = acceptance_by_ref.get(requirement_ref)
            if clean_acceptance is None:
                clean_acceptance = clean_values(raw_acceptance)
            if not clean_acceptance:
                clean_acceptance = [
                    "The requested behavior is present and observable in the generated Minecraft mod."
                ]
            clean_binding["acceptance"] = list(clean_acceptance)
            print(
                "production boundary: release binding acceptance "
                f"index={index} requirement_ref={requirement_ref!r} "
                f"task_refs={binding.get('task_refs')!r} raw={raw_acceptance!r} "
                f"sanitized={clean_acceptance!r}"
            )
            clean_bindings.append(clean_binding)
        plan["acceptance_release_bindings"] = clean_bindings

    return plan


"""

if "def _sanitize_evidence_plan_public_acceptance(" not in boundary:
    anchor = "def _filter_evidence_input_acceptance("
    if anchor not in boundary:
        raise SystemExit("production boundary insertion anchor missing")
    boundary = boundary.replace(anchor, helper + anchor, 1)

if "evidence_plan = _sanitize_evidence_plan_public_acceptance(evidence_plan)" not in boundary:
    match = re.search(
        r"(?m)^(?P<i>[ \t]*)effective_acceptance\s*=\s*_filter_evidence_input_acceptance\(",
        boundary,
    )
    if not match:
        raise SystemExit("production boundary compile anchor missing")
    insertion = (
        f"{match.group('i')}evidence_plan = "
        "_sanitize_evidence_plan_public_acceptance(evidence_plan)\n"
    )
    boundary = boundary[:match.start()] + insertion + boundary[match.start():]

generic = "raise ProductionContractError('public acceptance contains internal task or integrity language')"
if generic in production:
    replacement = (
        'matched_markers = []\n'
        '        if "task_" in folded:\n'
        '            matched_markers.append("task_")\n'
        '        matched_markers.extend(\n'
        '            marker\n'
        '            for marker in _PUBLIC_ACCEPTANCE_INTERNAL_MARKERS\n'
        '            if marker in folded and marker not in matched_markers\n'
        '        )\n'
        '        print(\n'
        '            "production contract: public acceptance rejected "\n'
        '            f"markers={matched_markers!r} statement={statement!r}"\n'
        '        )\n'
        '        raise ProductionContractError(\n'
        '            "public acceptance contains internal task or integrity language: "\n'
        '            f"markers={matched_markers!r}; statement={statement!r}"\n'
        '        )'
    )
    production = production.replace(generic, replacement, 1)

for path in Path("minecraft_mod_ai").rglob("*.py"):
    text = path.read_text(encoding="utf-8")
    if "llama structured recovery:" not in text or " details=" in text:
        continue
    pattern = re.compile(r"errors=\{len\((?P<v>[A-Za-z_]\w*)\)\}")
    match = pattern.search(text)
    if not match:
        continue
    var = match.group("v")
    text = (
        text[:match.start()]
        + f"errors={{len({var})}} details={{{var}!r}}"
        + text[match.end():]
    )
    path.write_text(text, encoding="utf-8")

tests = """
from minecraft_mod_ai.production_boundary_contract import (
    _sanitize_evidence_plan_public_acceptance,
)
from minecraft_mod_ai.production_contract import (
    ProductionContractError,
    _validate_public_acceptance,
)


def test_dirty_requirement_acceptance_is_cleaned_before_production_compile():
    plan = {
        "request_catalog": {
            "requirements": [
                {
                    "requirement_id": "req_visible",
                    "acceptance": [
                        "The requested block is visible in game; task_sha256 must match"
                    ],
                }
            ]
        },
        "acceptance_release_bindings": [
            {
                "requirement_ref": "req_visible",
                "task_refs": ["task_internal"],
                "acceptance": ["required_gates must pass"],
            }
        ],
    }
    cleaned = _sanitize_evidence_plan_public_acceptance(plan)
    expected = ["The requested block is visible in game"]
    assert cleaned["request_catalog"]["requirements"][0]["acceptance"] == expected
    assert cleaned["acceptance_release_bindings"][0]["acceptance"] == expected


def test_public_acceptance_error_exposes_marker_and_statement():
    statement = "Visible result and task_sha256 must match"
    try:
        _validate_public_acceptance(statement)
    except ProductionContractError as exc:
        message = str(exc)
    else:
        raise AssertionError("dirty public acceptance unexpectedly passed")
    assert "task_" in message
    assert statement in message
"""

boundary_path.write_text(boundary, encoding="utf-8")
production_path.write_text(production, encoding="utf-8")
test_path.write_text(tests.lstrip(), encoding="utf-8")
