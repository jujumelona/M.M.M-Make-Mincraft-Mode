from pathlib import Path
import re

boundary_path = Path("minecraft_mod_ai/production_boundary_contract.py")
production_path = Path("minecraft_mod_ai/production_contract.py")
test_path = Path("tests/test_public_acceptance_runtime_boundary.py")

boundary = boundary_path.read_text(encoding="utf-8")
production = production_path.read_text(encoding="utf-8")
tests = test_path.read_text(encoding="utf-8")

helper = '''
def _sanitize_public_acceptance_items(value):
    # Remove internal task/integrity clauses before production compilation.
    if not isinstance(value, str):
        return []

    from . import evidence_first_planning as _evidence

    normalize = getattr(_evidence, "_normalize_public_acceptance", None)
    candidate = normalize(value) if callable(normalize) else value.strip()
    candidate = str(candidate or "").strip()
    if not candidate:
        return []
    if _strict_public_acceptance(candidate):
        return [candidate]

    clean = []
    for chunk in candidate.replace("\\r", "\\n").replace("\\n", ";").split(";"):
        chunk = chunk.strip(" \\t-*•")
        if chunk and _strict_public_acceptance(chunk):
            clean.append(chunk)
    return clean


def _sanitize_evidence_plan_public_acceptance(evidence_plan):
    # Make request-catalog public acceptance authoritative before strict compile.
    from collections.abc import Mapping as _Mapping

    if not isinstance(evidence_plan, _Mapping):
        return evidence_plan

    plan = dict(evidence_plan)
    catalog = plan.get("request_catalog")
    if not isinstance(catalog, _Mapping):
        return plan

    requirements = catalog.get("requirements")
    if not isinstance(requirements, (list, tuple)):
        return plan

    clean_requirements = []
    acceptance_by_ref = {}

    for requirement in requirements:
        if not isinstance(requirement, _Mapping):
            clean_requirements.append(requirement)
            continue

        clean_requirement = dict(requirement)
        raw_acceptance = requirement.get("acceptance")
        if isinstance(raw_acceptance, str):
            raw_items = [raw_acceptance]
        elif isinstance(raw_acceptance, (list, tuple)):
            raw_items = list(raw_acceptance)
        else:
            raw_items = []

        clean_items = []
        for item in raw_items:
            clean_items.extend(_sanitize_public_acceptance_items(item))

        if not clean_items:
            source_span = requirement.get("source_span")
            source_text = ""
            if isinstance(source_span, _Mapping):
                source_text = str(source_span.get("text") or "").strip()

            if source_text and _strict_public_acceptance(source_text):
                visible = "The generated Minecraft mod visibly satisfies this request: " + source_text
                if _strict_public_acceptance(visible):
                    clean_items = [visible]

            if not clean_items:
                clean_items = [
                    "The requested behavior is present and observable in the generated Minecraft mod."
                ]

        clean_requirement["acceptance"] = clean_items
        clean_requirements.append(clean_requirement)

        requirement_ref = str(
            requirement.get("requirement_id")
            or requirement.get("requirement_ref")
            or ""
        ).strip()
        if requirement_ref:
            acceptance_by_ref[requirement_ref] = list(clean_items)

    clean_catalog = dict(catalog)
    clean_catalog["requirements"] = clean_requirements
    plan["request_catalog"] = clean_catalog

    bindings = plan.get("acceptance_release_bindings")
    if isinstance(bindings, (list, tuple)):
        clean_bindings = []
        for binding in bindings:
            if not isinstance(binding, _Mapping):
                clean_bindings.append(binding)
                continue

            clean_binding = dict(binding)
            requirement_ref = str(
                binding.get("requirement_ref")
                or binding.get("requirement_id")
                or ""
            ).strip()

            if requirement_ref in acceptance_by_ref:
                clean_binding["acceptance"] = list(acceptance_by_ref[requirement_ref])
            else:
                raw_acceptance = binding.get("acceptance")
                if isinstance(raw_acceptance, str):
                    raw_items = [raw_acceptance]
                elif isinstance(raw_acceptance, (list, tuple)):
                    raw_items = list(raw_acceptance)
                else:
                    raw_items = []

                clean_items = []
                for item in raw_items:
                    clean_items.extend(_sanitize_public_acceptance_items(item))
                if clean_items:
                    clean_binding["acceptance"] = clean_items

            clean_bindings.append(clean_binding)

        plan["acceptance_release_bindings"] = clean_bindings

    return plan


'''

if "def _sanitize_evidence_plan_public_acceptance(" not in boundary:
    anchor = re.search(
        r"(?m)^(?P<indent>[ \t]*)def _filter_evidence_input_acceptance\(",
        boundary,
    )
    if not anchor:
        raise SystemExit("critical hotfix anchor missing: _filter_evidence_input_acceptance")
    boundary = boundary[:anchor.start()] + helper + boundary[anchor.start():]

if "_sanitize_evidence_plan_public_acceptance(evidence_plan)" not in boundary:
    effective_pattern = re.compile(
        r"(?ms)^(?P<indent>[ \t]*)effective_acceptance\s*=\s*"
        r"_filter_evidence_input_acceptance\(\s*"
        r"acceptance_tests\s*,\s*evidence_plan\s*,?\s*\)"
    )
    match = effective_pattern.search(boundary)
    if not match:
        raise SystemExit("critical hotfix anchor missing: multiline effective_acceptance")
    indent = match.group("indent")
    original = match.group(0)
    replacement = (
        indent + "evidence_plan = _sanitize_evidence_plan_public_acceptance(evidence_plan)\n"
        + original
    )
    boundary = boundary[:match.start()] + replacement + boundary[match.end():]

if "marker={matched_marker!r}; value={folded!r}" not in production:
    production_pattern = re.compile(
        r"(?ms)(?P<indent>^[ \t]*)if\s+'task_'\s+in\s+folded\s+or\s+any\("
        r"marker\s+in\s+folded\s+for\s+marker\s+in\s+_PUBLIC_ACCEPTANCE_INTERNAL_MARKERS"
        r"\):\s*"
        r"raise\s+ProductionContractError\(\s*"
        r"['\"]public acceptance contains internal task or integrity language['\"]\s*"
        r"\)"
    )
    match = production_pattern.search(production)
    if match:
        i = match.group("indent")
        block = "\n".join([
            i + "if 'task_' in folded or any(marker in folded for marker in _PUBLIC_ACCEPTANCE_INTERNAL_MARKERS):",
            i + "    matched_marker = (",
            i + "        'task_'",
            i + "        if 'task_' in folded",
            i + "        else next(",
            i + "            (marker for marker in _PUBLIC_ACCEPTANCE_INTERNAL_MARKERS if marker in folded),",
            i + "            'unknown',",
            i + "        )",
            i + "    )",
            i + "    raise ProductionContractError(",
            i + "        'public acceptance contains internal task or integrity language: '",
            i + "        f'marker={matched_marker!r}; value={folded!r}'",
            i + "    )",
        ])
        production = production[:match.start()] + block + production[match.end():]

regression = '''

def test_evidence_plan_public_acceptance_is_sanitized_before_production_compile():
    from minecraft_mod_ai.production_boundary_contract import (
        _sanitize_evidence_plan_public_acceptance,
    )

    plan = {
        "request_catalog": {
            "requirements": [
                {
                    "requirement_id": "req_visible",
                    "source_span": {"text": "Show the requested visible behavior"},
                    "acceptance": [
                        "The requested block is visible in game; task_sha256 must match"
                    ],
                }
            ]
        },
        "acceptance_release_bindings": [
            {
                "requirement_ref": "req_visible",
                "acceptance": ["required_gates must pass"],
            }
        ],
    }

    sanitized = _sanitize_evidence_plan_public_acceptance(plan)
    expected = ["The requested block is visible in game"]

    assert sanitized["request_catalog"]["requirements"][0]["acceptance"] == expected
    assert sanitized["acceptance_release_bindings"][0]["acceptance"] == expected


def test_public_acceptance_sanitizer_falls_back_without_internal_integrity_language():
    from minecraft_mod_ai.production_boundary_contract import (
        _sanitize_evidence_plan_public_acceptance,
    )
    from minecraft_mod_ai.production_contract import _validate_public_acceptance

    plan = {
        "request_catalog": {
            "requirements": [
                {
                    "requirement_id": "req_fallback",
                    "source_span": {"text": "Show a visible block in game"},
                    "acceptance": ["task_sha256 must match required_gates"],
                }
            ]
        },
        "acceptance_release_bindings": [
            {
                "requirement_ref": "req_fallback",
                "acceptance": ["done_predicate must pass"],
            }
        ],
    }

    sanitized = _sanitize_evidence_plan_public_acceptance(plan)
    values = sanitized["request_catalog"]["requirements"][0]["acceptance"]
    assert values
    for value in values:
        _validate_public_acceptance(value)
    assert sanitized["acceptance_release_bindings"][0]["acceptance"] == values
'''

if "test_evidence_plan_public_acceptance_is_sanitized_before_production_compile" not in tests:
    tests = tests.rstrip() + regression + "\n"

boundary_path.write_text(boundary, encoding="utf-8")
production_path.write_text(production, encoding="utf-8")
test_path.write_text(tests, encoding="utf-8")
