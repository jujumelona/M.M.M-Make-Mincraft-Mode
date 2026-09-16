from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text or text.count(old) != 1:
        raise SystemExit(f"expected unique source block not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


mcp_tools = ROOT / "minecraft_mod_ai" / "mcp_tools.py"
replace_once(
    mcp_tools,
    """    def search_project_rag(self, query: str, minecraft_version: str | None = None, limit: int = 6) -> dict[str, Any]:\n""",
    """    @staticmethod\n    def _host_api_evidence(target: Any, query: str, limit: int) -> dict[str, Any]:\n        try:\n            facts = json.loads(str(getattr(target, 'host_facts_json', '') or '{}'))\n        except json.JSONDecodeError as exc:\n            raise SpecValidationError('RAG_HOST_FACTS_INVALID: target facts are not valid JSON.') from exc\n        if not isinstance(facts, dict):\n            raise SpecValidationError('RAG_HOST_FACTS_INVALID: target facts must be an object.')\n        api_symbols = facts.get('api_symbols')\n        artifact_rules = facts.get('artifact_rules')\n        api_symbols = api_symbols if isinstance(api_symbols, dict) else {}\n        artifact_rules = artifact_rules if isinstance(artifact_rules, dict) else {}\n        terms = set(re.findall(r'[a-z0-9_.-]+', query.casefold()))\n\n        def score(identifier: str, value: Any) -> int:\n            searchable = (\n                identifier.casefold() + ' ' +\n                json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).casefold()\n            )\n            return sum(1 for term in terms if term and term in searchable)\n\n        ranked_rules = sorted(\n            artifact_rules.items(),\n            key=lambda item: (-score(str(item[0]), item[1]), str(item[0])),\n        )\n        selected_rules = {\n            str(identifier): value\n            for identifier, value in ranked_rules\n            if score(str(identifier), value) > 0\n        }\n        selected_rules = dict(list(selected_rules.items())[: min(max(limit, 1), 8)])\n        required_symbols: set[str] = set()\n\n        def collect_required(value: Any) -> None:\n            if isinstance(value, dict):\n                for key, child in value.items():\n                    if key == 'required_symbols' and isinstance(child, (list, tuple)):\n                        required_symbols.update(str(item) for item in child if str(item).strip())\n                    else:\n                        collect_required(child)\n            elif isinstance(value, (list, tuple)):\n                for child in value:\n                    collect_required(child)\n\n        for value in selected_rules.values():\n            collect_required(value)\n        ranked_symbols = sorted(\n            api_symbols.items(),\n            key=lambda item: (\n                0 if str(item[0]) in required_symbols else 1,\n                -score(str(item[0]), item[1]),\n                str(item[0]),\n            ),\n        )\n        selected_symbols: dict[str, Any] = {}\n        for identifier, value in ranked_symbols:\n            symbol_id = str(identifier)\n            if symbol_id in required_symbols or score(symbol_id, value) > 0:\n                selected_symbols[symbol_id] = value\n            if len(selected_symbols) >= min(max(limit * 2, 4), 16):\n                break\n        return {\n            'schema_version': 'mmm/host-api-evidence-v1',\n            'target_version': target.minecraft_version,\n            'loader': target.loader,\n            'api_symbols': selected_symbols,\n            'artifact_rules': selected_rules,\n            'required_symbol_ids': sorted(required_symbols),\n        }\n\n    def search_project_rag(self, query: str, minecraft_version: str | None = None, limit: int = 6) -> dict[str, Any]:\n""",
)
replace_once(
    mcp_tools,
    """        target_context: dict[str, Any] | None = None\n        project_code: dict[str, Any] | None = None\n        if target is not None:\n""",
    """        target_context: dict[str, Any] | None = None\n        host_api_evidence: dict[str, Any] | None = None\n        project_code: dict[str, Any] | None = None\n        if target is not None:\n""",
)
replace_once(
    mcp_tools,
    """            target_context = self._rag_target_context(target)\n            project_code = search_workspace_source(\n""",
    """            target_context = self._rag_target_context(target)\n            host_api_evidence = self._host_api_evidence(target, normalized_query, limit)\n            project_code = search_workspace_source(\n""",
)
replace_once(
    mcp_tools,
    """            'target': target_context,\n            'sources': [source.__dict__ for source in sources],\n""",
    """            'target': target_context,\n            'host_api_evidence': host_api_evidence,\n            'sources': [source.__dict__ for source in sources],\n""",
)

adapter = ROOT / "minecraft_mod_ai" / "model_adapters" / "llama_cpp_adapter.py"
replace_once(
    adapter,
    '_HOST_OWNED_MODEL_ARGUMENTS = frozenset({"workspace_root"})\n',
    '''_HOST_OWNED_MODEL_ARGUMENTS = frozenset({\n    "workspace_root",\n    "minecraft_version",\n    "loader",\n    "java_version",\n    "mappings",\n    "mapping_namespace",\n})\n''',
)

loop = ROOT / "minecraft_mod_ai" / "progress_aware_tool_loop.py"
replace_once(
    loop,
    '''            state.record_no_progress_result({\n                "phase": state.phase.value,\n                "model_tool_rejections": rejection_payloads,\n            })\n            content = (turn.content or "").strip()\n''',
    '''            # Admission rejection was not executed, so it cannot establish a\n            # repeated source/action/result state or a verification-repair fixed point.\n            content = (turn.content or "").strip()\n''',
)

test_path = ROOT / "tests" / "test_generation_rag_target_binding.py"
with test_path.open("a", encoding="utf-8") as handle:
    handle.write(r'''


def test_generation_rag_returns_host_api_repair_facts(monkeypatch, tmp_path):
    import json

    target = _fake_target()
    target.host_facts_json = json.dumps({
        "api_symbols": {
            "builtin_item_registry": {"owner": "net.minecraft.core.registries.BuiltInRegistries", "member": "ITEM"},
            "register_item": {"owner": "net.minecraft.core.Registry", "member": "register"},
            "resource_key_create": {"owner": "net.minecraft.resources.ResourceKey", "member": "create"},
        },
        "artifact_rules": {
            "fabric/item/register_keyed": {
                "description": "item registry registration",
                "required_symbols": ["builtin_item_registry", "register_item", "resource_key_create"],
            }
        },
    })
    monkeypatch.setenv("MMM_MCP_STAGE", "generation")
    monkeypatch.setenv("MMM_MCP_MINECRAFT_VERSION", "26.2")
    monkeypatch.setenv("MMM_MCP_LOADER", "fabric")
    monkeypatch.setattr(mcp_tools, "adapter_from_project", lambda root: target)
    monkeypatch.setattr(mcp_tools.AuthoritativeEvidenceRetriever, "search", lambda *args, **kwargs: ())
    monkeypatch.setattr(mcp_tools, "search_workspace_source", lambda *args, **kwargs: {"hits": [], "receipt": {"status": "EMPTY"}})

    result = MMMToolService(workspace_root=tmp_path).search_project_rag("item registry register", limit=6)
    evidence = result["host_api_evidence"]
    assert evidence["api_symbols"]["builtin_item_registry"]["member"] == "ITEM"
    assert evidence["api_symbols"]["register_item"]["owner"] == "net.minecraft.core.Registry"
    assert "fabric/item/register_keyed" in evidence["artifact_rules"]
    assert set(evidence["required_symbol_ids"]) == {"builtin_item_registry", "register_item", "resource_key_create"}


def test_stale_numeric_target_coordinate_is_host_owned_at_admission():
    from minecraft_mod_ai.model_adapters.base import ToolCall
    from minecraft_mod_ai.model_adapters.llama_cpp_adapter import _normalize_admission_arguments

    call = ToolCall(
        id="call-1",
        name="search_project_rag",
        arguments={"query": "item registry", "minecraft_version": 26.2, "limit": 6},
        raw_arguments='{"query":"item registry","minecraft_version":26.2,"limit":6}',
    )
    normalized, failure = _normalize_admission_arguments(call, {"search_project_rag": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["query"],
        "additionalProperties": False,
    }})
    assert failure is None
    assert normalized.arguments == {"query": "item registry", "limit": 6}


def test_nonexecuted_admission_rejection_cannot_create_semantic_fixed_point():
    source = Path("minecraft_mod_ai/progress_aware_tool_loop.py").read_text(encoding="utf-8")
    start = source.index("rejection_feedback = _model_tool_rejection_feedback")
    end = source.index("        if required_rag_choice:", start)
    rejection_branch = source[start:end]
    assert "record_no_progress_result" not in rejection_branch
    assert "state.record_failure" in rejection_branch
    assert "continue" in rejection_branch
''')

print("generation RAG recovery completion applied")
