from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one match in {path}: found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


corrective = Path("minecraft_mod_ai/pre_design_rag_corrective.py")
replace_once(
    corrective,
    '''    schema = {\n        "type": "object",\n        "properties": {\n            "queries": {\n                "type": "array",\n                "minItems": 1,\n                "maxItems": 4,\n                "items": {"type": "string", "minLength": 4, "maxLength": 180},\n                "uniqueItems": True,\n            }\n        },\n        "required": ["queries"],\n        "additionalProperties": True,\n    }\n''',
    '''    query_array_schema = {\n        "type": "array",\n        "minItems": 1,\n        "maxItems": 4,\n        "items": {"type": "string", "minLength": 4, "maxLength": 180},\n        "uniqueItems": True,\n    }\n    schema = {\n        "type": "object",\n        "properties": {\n            "queries": query_array_schema,\n            # Qwen commonly emits this semantically equivalent key. Accept it at the\n            # host parser boundary instead of discarding a valid retrieval plan before\n            # any external source request can run.\n            "search_queries": query_array_schema,\n        },\n        "additionalProperties": True,\n    }\n''',
)
replace_once(
    corrective,
    '''        queries = value.get("queries") if isinstance(value, Mapping) else None\n        if not isinstance(queries, list):\n            raise agentic_module.SpecValidationError(\n                "corrective query planner omitted queries"\n            )\n''',
    '''        queries = value.get("queries") if isinstance(value, Mapping) else None\n        if not isinstance(queries, list) and isinstance(value, Mapping):\n            queries = value.get("search_queries")\n        if not isinstance(queries, list):\n            raise agentic_module.SpecValidationError(\n                "corrective query planner omitted queries/search_queries"\n            )\n''',
)
replace_once(
    corrective,
    '''                except Exception:\n                    unseen = []\n''',
    '''                except Exception as exc:\n                    failures.append(\n                        {\n                            "unit": f"corrective-query:{round_index}",\n                            "error": f"{type(exc).__name__}: {exc}",\n                        }\n                    )\n                    unseen = []\n''',
)

bootstrap = Path("minecraft_mod_ai/runtime_bootstrap.py")
replace_once(
    bootstrap,
    '''    from .minecraft_mcp_evidence_contract import (\n        install as install_minecraft_mcp_evidence,\n    )\n''',
    '''    from .minecraft_mcp_evidence_contract import (\n        install as install_minecraft_mcp_evidence,\n    )\n    from .pre_design_external_source_contract import (\n        install as install_pre_design_external_source,\n    )\n''',
)
replace_once(
    bootstrap,
    '''    install_agent_security(\n        pre_design_rag_module=agentic_pre_design_rag,\n''',
    '''    # The base pre-design retriever owns the approved query set and local evidence;\n    # this layer performs the missing bounded external search -> source-body acquisition.\n    install_pre_design_external_source(agentic_pre_design_rag)\n    install_agent_security(\n        pre_design_rag_module=agentic_pre_design_rag,\n''',
)

print("RAG source acquisition integration patched")
