from __future__ import annotations

from pathlib import Path


def replace_between(path: str, start: str, end: str, replacement: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    i = text.find(start)
    j = text.find(end, i + len(start))
    if i < 0 or j < 0:
        raise RuntimeError(f"markers not found in {path}: {start!r} .. {end!r}")
    target.write_text(text[:i] + replacement.rstrip() + "\n\n\n" + text[j:], encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"expected block not found in {path}: {old[:160]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_game_design() -> None:
    replacement = r'''def _generate_section(
    router: Any,
    *,
    prompt: str,
    section_id: str,
    fields: Sequence[str],
    host_properties: Mapping[str, Any] | None = None,
    research: Mapping[str, Any],
    media_paths: Sequence[str | Path],
    trace_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compile one section host-side from independent semantic field generations.

    The model never owns section keys, heading presence, or the final structure.  A
    malformed field response is replaced from the frozen requirement ledger rather
    than aborting the entire planner.
    """
    del host_properties
    trace = PlannerStageTrace(
        stage=f"game_design_{section_id}",
        prompt=prompt,
        media_paths=media_paths,
        metadata=dict(trace_metadata or {}),
    )
    ledger = _active_requirement_ledger(prompt)
    requirement_ids = tuple(item["requirement_id"] for item in ledger)
    section: dict[str, Any] = {}
    raw_outputs: dict[str, str] = {}
    fallback_fields: dict[str, str] = {}

    for index, field in enumerate(fields):
        raw = router.generate_text(
            "planner",
            _field_messages(
                prompt=prompt,
                section_id=section_id,
                field=field,
                research=research,
            ),
            media_paths=media_paths if index == 0 else (),
            response_format="text",
            response_schema=None,
            tool_stage="game_design",
            enable_tools=False,
        )
        raw_outputs[field] = str(raw or "")
        try:
            value = _parse_field_output(raw, field)
            if field == "modules":
                value = _ensure_module_coverage(value, ledger)
            _validate_section_types(
                {field: value},
                (field,),
                requirement_ids=requirement_ids,
            )
        except (KeyError, SpecValidationError, ValueError, TypeError) as exc:
            fallback_fields[field] = f"{type(exc).__name__}: {exc}"
            value = _host_field_fallback(field, prompt=prompt, ledger=ledger)
            if field == "modules":
                value = _ensure_module_coverage(value, ledger)
            _validate_section_types(
                {field: value},
                (field,),
                requirement_ids=requirement_ids,
            )
        section[field] = value

    trace.record_attempt(
        raw_output=json.dumps(raw_outputs, ensure_ascii=False, sort_keys=True),
        validation_error=None,
        candidate=section,
        accepted=section,
        context={
            "section_id": section_id,
            "format": "host_owned_field_compiler",
            "fallback_fields": fallback_fields,
        },
    )
    trace.record_success(section)
    return section


def _strip_accidental_field_wrapper(raw: Any, field: str) -> str:
    body = str(raw or "").strip()
    if body.startswith("```") and body.endswith("```"):
        lines = body.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        body = "\n".join(lines).strip()
    lines = body.splitlines()
    if lines:
        heading = re.match(r"^\s*##\s+(.+?)\s*$", lines[0])
        if heading and _normalize_heading(heading.group(1)) == _normalize_heading(field):
            body = "\n".join(lines[1:]).strip()
    return body


def _parse_field_output(raw: Any, field: str) -> Any:
    body = _strip_accidental_field_wrapper(raw, field)
    if field in {"title", "pitch"}:
        value = _plain_text(body)
        if not value:
            raise SpecValidationError(f"Planner left {field} empty")
        return value
    if field in _LIST_FIELDS:
        values = _markdown_list(body)
        if not values:
            raise SpecValidationError(f"Planner left {field} empty")
        return values
    if field in _MAP_FIELDS:
        return _markdown_map(body)
    if field == "modules":
        return _module_rows(body)
    if field == "assets":
        return _asset_rows(body)
    raise SpecValidationError(f"Unsupported host design field: {field}")


def _fallback_requirement_text(item: Mapping[str, Any]) -> str:
    for key in ("semantic_statement", "authored_text"):
        value = " ".join(str(item.get(key) or "").split()).strip()
        if value:
            return value
    acceptance = item.get("acceptance")
    if isinstance(acceptance, list):
        value = " ".join(str(entry).strip() for entry in acceptance if str(entry).strip())
        if value:
            return value
    return str(item.get("requirement_id") or "requested mechanic").strip()


def _fallback_list(prompt: str, ledger: Sequence[Mapping[str, Any]], *, acceptance: bool = False) -> list[str]:
    values: list[str] = []
    for item in ledger:
        if acceptance:
            raw = item.get("acceptance")
            if isinstance(raw, list):
                values.extend(" ".join(str(entry).split()) for entry in raw if str(entry).strip())
                continue
        value = _fallback_requirement_text(item)
        if value:
            values.append(value)
    values = list(dict.fromkeys(value for value in values if value))
    if values:
        return values
    fallback = " ".join(str(prompt or "").split()).strip()
    return [fallback or "Implement and verify the approved Minecraft mod request."]


def _fallback_module(item: Mapping[str, Any], index: int) -> dict[str, Any]:
    requirement_id = str(item.get("requirement_id") or "").strip()
    capability = _normalize_heading(str(item.get("capability") or "")) or f"requirement_{index + 1}"
    acceptance = item.get("acceptance")
    obligations = (
        [" ".join(str(entry).split()) for entry in acceptance if str(entry).strip()]
        if isinstance(acceptance, list)
        else []
    )
    if not obligations:
        obligations = [_fallback_requirement_text(item)]
    return {
        "plugin_id": f"design_{capability}_{index + 1}",
        "status": "custom_required",
        "reason": _fallback_requirement_text(item),
        "requirement_refs": [requirement_id] if requirement_id else [],
        "implementation_obligations": list(dict.fromkeys(obligations)),
    }


def _host_field_fallback(
    field: str,
    *,
    prompt: str,
    ledger: Sequence[Mapping[str, Any]],
) -> Any:
    if field == "title":
        capabilities = [
            str(item.get("capability") or "").replace("_", " ").strip()
            for item in ledger
            if str(item.get("capability") or "").strip()
        ]
        if capabilities:
            return " + ".join(capabilities[:3]).title() + " Minecraft Mod"
        return (" ".join(prompt.split())[:120] or "Minecraft Mod Design")
    if field == "pitch":
        return " ".join(prompt.split()) or "Implement the approved Minecraft mod request."
    if field in {"core_loop", "progression"}:
        return _fallback_list(prompt, ledger)
    if field == "acceptance_tests":
        return _fallback_list(prompt, ledger, acceptance=True)
    if field in {"combat", "mod_context", "art_direction"}:
        return {}
    if field == "modules":
        return [_fallback_module(item, index) for index, item in enumerate(ledger)]
    if field == "assets":
        return []
    raise SpecValidationError(f"No host fallback exists for design field {field}")


def _ensure_module_coverage(
    modules: Any,
    ledger: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = [dict(item) for item in modules if isinstance(item, Mapping)] if isinstance(modules, list) else []
    required = {
        str(item.get("requirement_id") or "").strip(): (index, item)
        for index, item in enumerate(ledger)
        if str(item.get("requirement_id") or "").strip()
    }
    covered: set[str] = set()
    for module in result:
        refs = module.get("requirement_refs")
        if isinstance(refs, list):
            covered.update(str(ref).strip() for ref in refs if str(ref).strip())
    for requirement_id, (index, item) in required.items():
        if requirement_id not in covered:
            result.append(_fallback_module(item, index))
    return result
'''
    replace_between(
        "minecraft_mod_ai/agentic_research_game_design.py",
        "def _generate_section(\n",
        "def _normalize_heading(\n",
        replacement,
    )

    field_messages = r'''def _field_messages(
    *,
    prompt: str,
    section_id: str,
    field: str,
    research: Mapping[str, Any],
) -> list[dict[str, str]]:
    if field in {"title", "pitch"}:
        format_instruction = "Return only the field text."
    elif field in _LIST_FIELDS:
        format_instruction = "Return only one or more concise bullet lines."
    elif field in _MAP_FIELDS:
        format_instruction = "Return 'none' or use ### subgroup headings followed by bullets."
    elif field == "modules":
        format_instruction = (
            "Return module records only. For each module use ### <plugin_id>, then "
            "- status: <value>, - reason: <text>, - requirement_refs: <exact comma-separated approved IDs>, "
            "and - implementation_obligations: followed by one or more nested bullets."
        )
    elif field == "assets":
        format_instruction = (
            "Return asset records only. For each asset use ### <id>, then - kind: <kind> and - brief: <description>. "
            "Return 'none' when no dedicated asset is required."
        )
    else:
        format_instruction = "Return only the requested field content."
    system = (
        "You are a bounded Minecraft mod design worker. The host already owns the field name and final structure. "
        "Generate semantic content for exactly one field. Do not write the field name, a Markdown ## heading, JSON, "
        "code fences, <think>, analysis, or unrelated fields. Preserve exact approved requirement IDs when requested. "
        + format_instruction
        + " No JSON. "
        + _PRODUCTION_DEPTH
    )
    ledger = _active_requirement_ledger(prompt)
    user = (
        "AUTHORITATIVE REQUEST\n"
        + prompt
        + "\n\nSECTION\n"
        + section_id
        + "\n\nFIELD\n"
        + field
        + "\n\n"
        + _render_requirement_ledger(ledger)
        + "\n\nRESEARCH CONTEXT\n"
        + _render_design_research(research)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
'''
    replace_between(
        "minecraft_mod_ai/agentic_research_game_design.py",
        "def _section_messages(\n",
        "def _render_design_research(\n",
        field_messages,
    )


def patch_source_gate() -> None:
    old = '''def _term_overlap(wanted: set[str], available: set[str]) -> bool:\n    for left in wanted:\n        for right in available:\n            if left == right:\n                return True\n            if min(len(left), len(right)) >= 5 and (left.startswith(right) or right.startswith(left)):\n                return True\n    return False\n'''
    new = '''def _term_overlap(wanted: set[str], available: set[str]) -> bool:\n    for left in wanted:\n        for right in available:\n            if left == right:\n                return True\n            if min(len(left), len(right)) >= 5 and (left.startswith(right) or right.startswith(left)):\n                return True\n            # Capability words often differ only by morphology in repository metadata\n            # (seasonal/seasons, colony/colonization, planting/plant).  Accept a long\n            # semantic stem while still requiring Minecraft-ecosystem gating separately.\n            common = 0\n            for lch, rch in zip(left, right):\n                if lch != rch:\n                    break\n                common += 1\n            if common >= 5 and (len(left) - common <= 4 or len(right) - common <= 4):\n                return True\n    return False\n'''
    replace_once("minecraft_mod_ai/pre_design_external_source_contract.py", old, new)


def patch_empty_evidence_skip() -> None:
    old = '''    try:\n        pages = project_rag._read_evidence_pages(working_document)\n    except Exception:\n        pages = []\n\n    claims: list[dict[str, Any]] = []\n    diagnostics: list[str] = []\n'''
    new = '''    projection_is_empty = (\n        "model_unit_count" in working_document\n        and int(working_document.get("model_unit_count") or 0) == 0\n    )\n    if projection_is_empty:\n        pages = []\n    else:\n        try:\n            pages = project_rag._read_evidence_pages(working_document)\n        except Exception:\n            pages = []\n\n    claims: list[dict[str, Any]] = []\n    diagnostics: list[str] = (\n        ["no_claim_bearing_source_bodies"] if projection_is_empty else []\n    )\n'''
    replace_once("minecraft_mod_ai/small_model_predesign_research.py", old, new)


def patch_section_test_router() -> None:
    path = "tests/test_agentic_research_game_design.py"
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    start = text.find("class _SectionRouter:\n")
    end = text.find("class _ResearchRouter:\n", start)
    if start < 0 or end < 0:
        raise RuntimeError("section router block not found")
    router_block = r'''class _SectionRouter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_text(
        self,
        role,
        messages,
        *,
        media_paths=(),
        response_format="text",
        response_schema=None,
        tool_stage=None,
        enable_tools=True,
    ):
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "media_paths": tuple(media_paths),
                "response_format": response_format,
                "response_schema": response_schema,
                "tool_stage": tool_stage,
                "enable_tools": enable_tools,
            }
        )
        text = str(messages[-1]["content"])
        marker = "\n\nFIELD\n"
        field = text.split(marker, 1)[1].split("\n", 1)[0].strip()
        bodies = {
            "title": "연구 기반 모드",
            "pitch": "검색 근거를 바탕으로 설계한다.",
            "core_loop": "- 탐색하고 상호작용한다",
            "progression": "- 기능을 단계적으로 해금한다",
            "combat": "none",
            "mod_context": "none",
            "modules": "none",
            "assets": "none",
            "acceptance_tests": "- 요청한 핵심 루프가 게임 내에서 동작한다",
            "art_direction": "none",
        }
        return bodies[field]

'''
    text = text[:start] + router_block + text[end:]
    old_test_start = text.find("def test_sectioned_game_design_uses_host_parsed_markdown() -> None:\n")
    old_test_end = text.find("def test_research_domain_legacy_facade_is_host_owned() -> None:\n", old_test_start)
    if old_test_start < 0 or old_test_end < 0:
        raise RuntimeError("section design test block not found")
    test_block = r'''def test_sectioned_game_design_uses_host_owned_field_compiler() -> None:
    router = _SectionRouter()
    research = {
        "research_brief": {"domains": []},
        "domain_notes": [],
        "deterministic": {},
        "errors": [],
    }
    result = agentic.generate_sectioned_game_design(
        game_design,
        router,
        "연구를 먼저 하고 모드를 설계해줘",
        research=research,
    )
    expected_fields = [
        field
        for _section_id, fields, _properties in agentic._SECTION_SPECS
        for field in fields
    ]
    assert len(router.calls) == len(expected_fields)
    assert all(call["role"] == "planner" for call in router.calls)
    assert all(call["response_format"] == "text" for call in router.calls)
    assert all(call["response_schema"] is None for call in router.calls)
    assert all(call["tool_stage"] == "game_design" for call in router.calls)
    assert all(call["enable_tools"] is False for call in router.calls)
    for field, call in zip(expected_fields, router.calls, strict=True):
        system = str(call["messages"][0]["content"])
        user = str(call["messages"][1]["content"])
        assert "No JSON" in system
        assert "Do not write the field name, a Markdown ## heading" in system
        assert f"\n\nFIELD\n{field}\n" in user
    assert result["title"] == "연구 기반 모드"
    assert result["core_loop"]
    assert result["acceptance_tests"]
    assert "art_direction" not in result


def test_missing_markdown_headings_can_never_abort_identity_section() -> None:
    class HeadinglessRouter:
        def generate_text(self, role, messages, **kwargs):
            del role, messages, kwargs
            return "계절 작물과 요리를 연결하는 플레이 경험"

    section = agentic._generate_section(
        HeadinglessRouter(),
        prompt="계절마다 다른 작물을 재배하고 요리하는 모드를 만들어줘.",
        section_id="identity_and_loop",
        fields=("title", "pitch", "core_loop"),
        research={},
        media_paths=(),
        trace_metadata=None,
    )
    assert set(section) == {"title", "pitch", "core_loop"}
    assert section["title"]
    assert section["pitch"]
    assert section["core_loop"]

'''
    target.write_text(text[:old_test_start] + test_block + text[old_test_end:], encoding="utf-8")


def write_new_regressions() -> None:
    Path("tests/test_planner_log24_regressions.py").write_text(
        r'''from __future__ import annotations


def test_season_repository_candidate_is_not_false_negative():
    from minecraft_mod_ai import pre_design_external_source_contract as external

    query = "minecraft seasonal crop planting mod"
    repository = {
        "full_name": "lucaargolo/fabric-seasons",
        "description": "A Fabric mod that adds seasons to Minecraft",
        "topics": ["minecraft", "fabric", "mod", "seasons"],
    }
    assert external._repository_candidate_relevant(query, repository)
    assert external._body_relevant(
        query,
        "Fabric Seasons is a mod for Minecraft that adds seasons and seasonal world behavior.",
    )


def test_empty_evidence_projection_never_calls_small_model():
    from minecraft_mod_ai import small_model_predesign_research as research

    class NeverRouter:
        def generate_text(self, *args, **kwargs):
            raise AssertionError("model must not be called for an empty evidence projection")

    class ProjectRag:
        @staticmethod
        def _read_evidence_pages(document):
            raise AssertionError("empty evidence pages must not be read")

        @staticmethod
        def _prompt_document_receipt(document):
            return {"page_count": int(document.get("page_count") or 0)}

    document = {
        "domain_id": "request",
        "model_unit_count": 0,
        "page_count": 1,
        "document_sha256": "sha256:empty",
    }
    result = research.research_document_domain(
        object(),
        ProjectRag(),
        NeverRouter(),
        prompt="계절 작물과 요리",
        domain={"domain_id": "request", "objective": "seasonal crops"},
        document=document,
        trace_metadata=None,
    )
    assert result["claims"] == []
    assert result["research_evidence_status"] == "no_relevant_external_evidence"
    assert "no_claim_bearing_source_bodies" in result["page_local_diagnostics"]
''',
        encoding="utf-8",
    )


def main() -> None:
    patch_game_design()
    patch_source_gate()
    patch_empty_evidence_skip()
    patch_section_test_router()
    write_new_regressions()


if __name__ == "__main__":
    main()
