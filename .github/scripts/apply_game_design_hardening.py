from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {count}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_between(path: str, start: str, end: str, replacement: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    a = text.find(start)
    b = text.find(end, a + len(start)) if a >= 0 else -1
    if a < 0 or b < 0:
        raise SystemExit(f"{path}: section markers not found: {start!r} -> {end!r}")
    target.write_text(text[:a] + replacement.rstrip() + "\n\n" + text[b:], encoding="utf-8")


# ---------------------------------------------------------------------------
# GenerationRequest / router: carry an optional host-owned JSON schema end-to-end.
# ---------------------------------------------------------------------------
replace_once(
    "minecraft_mod_ai/model_adapters/base.py",
    '''    response_format: str = "text"\n    tools: tuple[Mapping[str, Any], ...] = ()\n''',
    '''    response_format: str = "text"\n    response_schema: Mapping[str, Any] | None = None\n    tools: tuple[Mapping[str, Any], ...] = ()\n''',
)

replace_once(
    "minecraft_mod_ai/model_router.py",
    '''        response_format: str = "text",\n        tool_stage: str | None = None,\n''',
    '''        response_format: str = "text",\n        response_schema: Mapping[str, Any] | None = None,\n        tool_stage: str | None = None,\n''',
)
replace_once(
    "minecraft_mod_ai/model_router.py",
    '''                response_format=response_format,\n                tools=tools,\n''',
    '''                response_format=response_format,\n                response_schema=response_schema,\n                tools=tools,\n''',
)
replace_once(
    "minecraft_mod_ai/model_router.py",
    '''                response_format=request.response_format,\n                tools=request.tools,\n                tool_choice=request.tool_choice,\n''',
    '''                response_format=request.response_format,\n                response_schema=request.response_schema,\n                tools=request.tools,\n                tool_choice=request.tool_choice,\n''',
)
replace_once(
    "minecraft_mod_ai/model_router.py",
    '''                    response_format=request.response_format,\n                    tools=(),\n                    tool_choice=None,\n''',
    '''                    response_format=request.response_format,\n                    response_schema=request.response_schema,\n                    tools=(),\n                    tool_choice=None,\n''',
)

# llama.cpp's documented json_object + schema shape converts the schema to a grammar.
replace_once(
    "minecraft_mod_ai/model_adapters/llama_cpp_adapter.py",
    '''            if getattr(request, "response_format", None) == "json":\n                payload["response_format"] = {"type": "json_object"}\n                payload["reasoning_effort"] = "none"\n''',
    '''            if getattr(request, "response_format", None) == "json":\n                if request.response_schema is not None:\n                    payload["response_format"] = {\n                        "type": "json_object",\n                        "schema": dict(request.response_schema),\n                    }\n                else:\n                    payload["response_format"] = {"type": "json_object"}\n                payload["reasoning_effort"] = "none"\n''',
)

# Remote OpenAI-compatible providers get the standard json_schema wrapper. Host-side
# validation remains authoritative, so a provider that merely hints JSON cannot bypass it.
replace_once(
    "minecraft_mod_ai/model_adapters/openai_compatible.py",
    '''            if request.response_format == "json":\n                # Standard OpenAI-compatible structured-output hint. Keep it scoped\n                # to text generation so image and speech retain their contracts.\n                payload["response_format"] = {"type": "json_object"}\n''',
    '''            if request.response_format == "json":\n                # Keep the schema host-owned and explicit. Validation still runs after\n                # generation because compatibility servers can differ in enforcement.\n                if request.response_schema is not None:\n                    payload["response_format"] = {\n                        "type": "json_schema",\n                        "json_schema": {\n                            "name": "mmm_structured_response",\n                            "strict": True,\n                            "schema": dict(request.response_schema),\n                        },\n                    }\n                else:\n                    payload["response_format"] = {"type": "json_object"}\n''',
)

# ---------------------------------------------------------------------------
# game_design: strict schema + progress-driven repair + persisted raw attempts.
# ---------------------------------------------------------------------------
replace_once(
    "minecraft_mod_ai/game_design.py",
    '''from .model_router import ModelRouter\nfrom .planner import HeuristicPlanner, _proposal_from_model_data\n''',
    '''from .model_router import ModelRouter\nfrom .planner import HeuristicPlanner, _proposal_from_model_data\nfrom .planner_stage_trace import PlannerStageTrace\n''',
)

schema_source = '''_GAME_DESIGN_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "game_design": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "minLength": 1},
                "pitch": {"type": "string", "minLength": 1},
                "core_loop": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "progression": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "combat": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "mod_context": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "modules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "plugin_id": {"type": "string", "minLength": 1},
                            "status": {"type": "string", "minLength": 1},
                            "reason": {"type": "string", "minLength": 1},
                        },
                        "required": ["plugin_id", "status", "reason"],
                        "additionalProperties": False,
                    },
                },
                "assets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "kind": {"type": "string", "minLength": 1},
                            "brief": {"type": "string", "minLength": 1},
                        },
                        "required": ["id", "kind", "brief"],
                        "additionalProperties": False,
                    },
                },
                "acceptance_tests": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "art_direction": {"type": "object"},
            },
            "required": [
                "title",
                "pitch",
                "core_loop",
                "progression",
                "combat",
                "mod_context",
                "modules",
                "assets",
                "acceptance_tests",
            ],
            "additionalProperties": False,
        }
    },
    "required": ["game_design"],
    "additionalProperties": False,
}
'''
replace_once(
    "minecraft_mod_ai/game_design.py",
    '''_OPTIONAL_GAME_DESIGN_FIELDS = ("art_direction",)\n\n# A per-call transport budget''',
    '''_OPTIONAL_GAME_DESIGN_FIELDS = ("art_direction",)\n\n''' + schema_source + '''\n# A per-call transport budget''',
)

old_plan = '''        messages = [\n            {"role": "system", "content": _system_prompt()},\n            {"role": "user", "content": prompt},\n        ]\n        text = self.router.generate_text(\n            "planner",\n            messages,\n            media_paths=media_paths,\n            response_format="json",\n        )\n        try:\n            design = _extract_valid_game_design(text)\n        except SpecValidationError as initial_error:\n            repaired_text = self.router.generate_text(\n                "planner",\n                _repair_messages(prompt),\n                media_paths=media_paths,\n                response_format="json",\n            )\n            try:\n                design = _extract_valid_game_design(repaired_text)\n            except SpecValidationError as repair_error:\n                raise SpecValidationError(\n                    "Planner could not return a complete game_design after one "\n                    "automatic repair of that stage. "\n                    f"Initial response: {initial_error} "\n                    f"Repair response: {repair_error}"\n                ) from repair_error\n\n'''
new_plan = '''        messages = [\n            {"role": "system", "content": _system_prompt()},\n            {"role": "user", "content": prompt},\n        ]\n        trace = PlannerStageTrace(\n            stage="game_design",\n            prompt=prompt,\n            media_paths=media_paths,\n        )\n        design = _generate_valid_game_design(\n            self.router,\n            authoritative_prompt=prompt,\n            initial_messages=messages,\n            media_paths=media_paths,\n            trace=trace,\n        )\n        trace.record_success(design)\n\n'''
replace_once("minecraft_mod_ai/game_design.py", old_plan, new_plan)

helper_source = r'''def _repair_candidate_from_text(text: str) -> dict[str, Any] | None:
    """Return the most complete parseable design candidate without accepting it."""

    candidates = tuple(_json_objects(text))
    for candidate in reversed(candidates):
        if "response" in candidate and isinstance(candidate["response"], dict):
            response = candidate["response"]
            if "data" in response and isinstance(response["data"], dict):
                candidate = response["data"]
        nested = candidate.get("game_design")
        possible = nested if isinstance(nested, dict) else candidate
        if isinstance(possible, dict) and set(possible) & set(_GAME_DESIGN_FIELDS):
            return _normalize_model_game_design(possible)
    return None


def _rejected_design_state(
    *,
    error: SpecValidationError,
    candidate: dict[str, Any] | None,
) -> str:
    # Raw prose is not progress. For an unparsable answer, only a changed validator
    # condition can advance the state. Parseable candidates are compared structurally.
    payload: dict[str, Any] = {"validation_error": str(error)}
    if candidate is not None:
        payload["candidate"] = candidate
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _generate_valid_game_design(
    router: ModelRouter,
    *,
    authoritative_prompt: str,
    initial_messages: Sequence[Mapping[str, Any]],
    media_paths: Sequence[str | Path],
    trace: PlannerStageTrace,
    repair_system_prompt: str | None = None,
    trace_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate until valid or a host-proven rejected state repeats.

    There is intentionally no attempt-count ceiling. A new structured candidate is
    allowed to progress no matter how many repairs were needed. The loop terminates
    only if validation succeeds or a previously rejected semantic/schema state recurs.
    """

    seen_rejected_states: set[str] = set()
    messages: Sequence[Mapping[str, Any]] = initial_messages
    validation_error: SpecValidationError | None = None
    candidate: dict[str, Any] | None = None

    while True:
        text = router.generate_text(
            "planner",
            messages,
            media_paths=media_paths,
            response_format="json",
            response_schema=_GAME_DESIGN_RESPONSE_SCHEMA,
        )
        try:
            design = _extract_valid_game_design(text)
        except SpecValidationError as exc:
            candidate = _repair_candidate_from_text(text)
            state = _rejected_design_state(error=exc, candidate=candidate)
            trace.record_attempt(
                raw_output=text,
                validation_error=str(exc),
                candidate=candidate,
                context=trace_context,
            )
            if state in seen_rejected_states:
                raise SpecValidationError(
                    "Planner game_design repair reached an exact no-progress cycle. "
                    f"Repeated validator state: {exc}"
                ) from exc
            seen_rejected_states.add(state)
            validation_error = exc
            messages = _repair_messages(
                authoritative_prompt,
                validation_error=str(validation_error),
                previous_candidate=candidate,
                system_prompt=repair_system_prompt,
            )
            continue

        trace.record_attempt(
            raw_output=text,
            validation_error=None,
            candidate=design,
            accepted=design,
            context=trace_context,
        )
        return design
'''
replace_once(
    "minecraft_mod_ai/game_design.py",
    '''\ndef _request_page_bytes(router: ModelRouter | None = None, role: str = "planner") -> int:\n''',
    "\n\n" + helper_source + '''\n\ndef _request_page_bytes(router: ModelRouter | None = None, role: str = "planner") -> int:\n''',
)

new_sharded = r'''def _generate_sharded_design_page(
    router: ModelRouter,
    *,
    request_text: str,
    media_paths: Sequence[str | Path],
    page_index: int,
    page_count: int,
) -> dict[str, Any]:
    trace = PlannerStageTrace(
        stage="game_design_page",
        prompt=request_text,
        media_paths=media_paths,
        metadata={"page_index": page_index, "page_count": page_count},
    )
    design = _generate_valid_game_design(
        router,
        authoritative_prompt=request_text,
        initial_messages=[
            {"role": "system", "content": _sharded_design_system_prompt()},
            {"role": "user", "content": request_text},
        ],
        media_paths=media_paths,
        trace=trace,
        repair_system_prompt=(
            _sharded_design_system_prompt()
            + "\n\nRepair the same bounded request page only. Preserve every valid field "
            "from the previous candidate and correct the host validator error."
        ),
        trace_context={"page_index": page_index, "page_count": page_count},
    )
    trace.record_success(design)
    return design
'''
replace_between(
    "minecraft_mod_ai/game_design.py",
    "def _generate_sharded_design_page(\n",
    "def _sharded_design_system_prompt() -> str:\n",
    new_sharded,
)

# Dynamic platform ownership: this stage describes the mod, it does not select target versions.
replace_once(
    "minecraft_mod_ai/game_design.py",
    "You are GameDesignPlanner for a Minecraft Java 1.20.1 Fabric production system.\n",
    "You are GameDesignPlanner for a Minecraft mod production system. The host resolves and validates the exact Minecraft version, loader, mappings, and Java target separately; never invent or override those platform choices.\n",
)
replace_once(
    "minecraft_mod_ai/game_design.py",
    "request for a Minecraft Java 1.20.1 Fabric mod. Return exactly one JSON object with\n",
    "request for a Minecraft mod. The host owns the exact platform target. Return exactly one JSON object with\n",
)

# Repair messages carry the exact validator failure and the last parseable candidate,
# never the full untrusted raw model prose.
new_repair_messages = r'''def _repair_messages(
    prompt: str,
    *,
    validation_error: str | None = None,
    previous_candidate: Mapping[str, Any] | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Repair from authoritative input plus bounded structured failure evidence."""

    repair_system = system_prompt or _repair_system_prompt()
    if validation_error:
        repair_system += (
            "\n\nHOST VALIDATOR ERROR (authoritative):\n"
            + validation_error
            + "\nCorrect this exact structural/semantic contract failure. Preserve valid "
            "user-requested content; do not add unrelated systems."
        )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": repair_system},
        {"role": "user", "content": prompt},
    ]
    if previous_candidate is not None:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"game_design": dict(previous_candidate)},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Repair the preceding game_design candidate according to the "
                        "host validator error. Return one complete corrected JSON object only."
                    ),
                },
            ]
        )
    return messages
'''
replace_between(
    "minecraft_mod_ai/game_design.py",
    "def _repair_messages(prompt: str) -> list[dict[str, str]]:\n",
    "def _repair_system_prompt() -> str:\n",
    new_repair_messages,
)

# Safe transport normalization. It changes representation only, never invents a fact.
normalize_insert = '''    normalized = dict(design)\n\n    for field in ("core_loop", "progression", "acceptance_tests"):\n        value = normalized.get(field)\n        if isinstance(value, str) and value.strip():\n            normalized[field] = [value.strip()]\n\n    for field in ("combat", "mod_context"):\n        value = normalized.get(field)\n        if not isinstance(value, dict):\n            continue\n        canonical_map: dict[str, Any] = {}\n        for key, item in value.items():\n            if isinstance(item, str) and item.strip():\n                canonical_map[str(key)] = [item.strip()]\n            elif isinstance(item, list):\n                canonical_map[str(key)] = [\n                    element.strip() if isinstance(element, str) else element\n                    for element in item\n                ]\n            else:\n                canonical_map[str(key)] = item\n        normalized[field] = canonical_map\n\n    modules = normalized.get("modules")\n'''
replace_once(
    "minecraft_mod_ai/game_design.py",
    '''    normalized = dict(design)\n    modules = normalized.get("modules")\n''',
    normalize_insert,
)

# The repair prompt should explicitly prioritize exact validator feedback and current
# candidate preservation rather than generic regeneration.
replace_once(
    "minecraft_mod_ai/game_design.py",
    '''Retry only the compact game-design stage from the unchanged original request. Return\nexactly one JSON object and no analysis or markdown. Preserve every distinct requested\nsystem, grouping large repeated catalogs into families. Do not add unrequested systems.\n''',
    '''Repair only the compact game-design stage from the unchanged authoritative request.\nReturn exactly one JSON object and no analysis or markdown. When a previous candidate\nand HOST VALIDATOR ERROR are supplied, preserve all already-valid fields and change only\nwhat is necessary to satisfy that error. Preserve every distinct requested system,\ngrouping large repeated catalogs into families. Do not add unrequested systems.\n''',
)

# ---------------------------------------------------------------------------
# Tests: align old one-repair assumptions and add the new hardening contracts.
# ---------------------------------------------------------------------------
replace_once(
    "tests/test_game_design_router.py",
    '''def test_multimodal_design_does_not_bootstrap_without_essential_design() -> None:\n    """Format recovery must not replace a missing game design with a template."""\n\n    incomplete_design = _valid_planner_payload()["game_design"]\n    del incomplete_design["acceptance_tests"]\n    truncated = (\n        '{"game_design":'\n        + json.dumps(incomplete_design)\n        + ',"build_slice":'\n    )\n    router = _SequenceTextRouter(truncated, truncated)\n\n    with pytest.raises(SpecValidationError, match="complete game_design"):\n        GameDesignPlanner(router).plan("Create a moon crystal item.")\n\n    assert len(router.calls) == 2\n''',
    '''def test_multimodal_design_stops_only_when_invalid_state_repeats() -> None:\n    """A repeated rejected structured state is a semantic no-progress proof."""\n\n    incomplete_design = _valid_planner_payload()["game_design"]\n    del incomplete_design["acceptance_tests"]\n    truncated = (\n        '{"game_design":'\n        + json.dumps(incomplete_design)\n        + ',"build_slice":'\n    )\n    router = _SequenceTextRouter(truncated, truncated)\n\n    with pytest.raises(SpecValidationError, match="no-progress cycle"):\n        GameDesignPlanner(router).plan("Create a moon crystal item.")\n\n    assert len(router.calls) == 2\n''',
)
replace_once(
    "tests/test_game_design_router.py",
    '''    invalid = _valid_planner_payload()["game_design"]\n    invalid["mod_context"] = {\n        "loader": "Fabric",\n        "minecraft_version": "1.20.1",\n    }\n    with pytest.raises(\n        SpecValidationError,\n        match="game_design.mod_context values must be lists",\n    ):\n        _validate_design(invalid)\n''',
    '''    invalid = _valid_planner_payload()["game_design"]\n    invalid["mod_context"] = {\n        "loader": "Fabric",\n        "minecraft_version": "1.20.1",\n    }\n    recovered = _extract_valid_game_design(\n        json.dumps({"game_design": invalid}, ensure_ascii=False)\n    )\n    assert recovered["mod_context"] == {\n        "loader": ["Fabric"],\n        "minecraft_version": ["1.20.1"],\n    }\n''',
)

append_tests = r'''


def test_game_design_repair_can_progress_beyond_two_model_calls(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MMM_PLANNER_TRACE_DIR", str(tmp_path / "traces"))
    payload = _valid_planner_payload()
    payload.pop("build_slice", None)
    invalid_one = json.loads(json.dumps(payload))
    invalid_one["game_design"]["assets"] = [{"id": "moon", "kind": "item"}]
    invalid_two = json.loads(json.dumps(payload))
    invalid_two["game_design"]["modules"] = ["quest_system"]
    router = _SequenceTextRouter(
        "not json",
        json.dumps(invalid_one),
        json.dumps(invalid_two),
        json.dumps(payload),
    )

    design, _proposal = GameDesignPlanner(router).plan("Create a moon crystal item.")

    assert design["title"] == "Moon Forge"
    assert len(router.calls) == 4
    # The third request contains the preceding structured candidate and exact validator error.
    messages = router.calls[2][0]
    assert any("HOST VALIDATOR ERROR" in message["content"] for message in messages)
    assert any(message["role"] == "assistant" for message in messages)


def test_game_design_trace_persists_raw_failures_and_final_success(tmp_path, monkeypatch) -> None:
    trace_root = tmp_path / "planner-traces"
    monkeypatch.setenv("MMM_PLANNER_TRACE_DIR", str(trace_root))
    payload = _valid_planner_payload()
    payload.pop("build_slice", None)
    router = _SequenceTextRouter("RAW_FAILURE_SENTINEL", json.dumps(payload))

    GameDesignPlanner(router).plan("Create a moon crystal item.")

    runs = list((trace_root / "game_design").iterdir())
    assert len(runs) == 1
    attempts = sorted(runs[0].glob("attempt-*.json"))
    assert len(attempts) == 2
    first = json.loads(attempts[0].read_text(encoding="utf-8"))
    second = json.loads(attempts[1].read_text(encoding="utf-8"))
    accepted = json.loads((runs[0] / "accepted.json").read_text(encoding="utf-8"))
    assert first["raw_output"] == "RAW_FAILURE_SENTINEL"
    assert first["validation_error"]
    assert second["accepted"]["title"] == "Moon Forge"
    assert accepted["accepted"]["title"] == "Moon Forge"


def test_game_design_requests_host_schema_from_router(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MMM_PLANNER_TRACE_DIR", str(tmp_path / "traces"))
    payload = _valid_planner_payload()
    payload.pop("build_slice", None)
    router = _SequenceTextRouter(json.dumps(payload))

    GameDesignPlanner(router).plan("Create a moon crystal item.")

    schema = router.calls[0][1]["response_schema"]
    assert schema["required"] == ["game_design"]
    game_design = schema["properties"]["game_design"]
    assert "mod_context" in game_design["required"]
    assert game_design["properties"]["mod_context"]["additionalProperties"]["type"] == "array"
'''
path = ROOT / "tests/test_game_design_router.py"
text = path.read_text(encoding="utf-8")
if "test_game_design_repair_can_progress_beyond_two_model_calls" not in text:
    path.write_text(text.rstrip() + append_tests + "\n", encoding="utf-8")

# Remote adapter structured schema contract.
path = ROOT / "tests/test_openai_compatible_json.py"
text = path.read_text(encoding="utf-8")
if "test_openai_compatible_json_schema_is_forwarded" not in text:
    text += r'''


def test_openai_compatible_json_schema_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    payload = _capture_payload(
        monkeypatch,
        GenerationRequest(
            messages=({"role": "user", "content": "Return structured JSON."},),
            response_format="json",
            response_schema=schema,
        ),
    )

    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "mmm_structured_response",
            "strict": True,
            "schema": schema,
        },
    }
'''
    path.write_text(text, encoding="utf-8")

print("game-design structured-output hardening patch applied")
