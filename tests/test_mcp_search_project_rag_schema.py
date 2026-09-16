from __future__ import annotations

from jsonschema.validators import validator_for

from minecraft_mod_ai import mcp_server


def test_search_project_rag_schema_accepts_numeric_minecraft_version() -> None:
    tool = mcp_server.mcp._tool_manager.get_tool("search_project_rag")
    assert tool is not None

    schema = tool.parameters
    validator_type = validator_for(schema)
    validator_type.check_schema(schema)
    errors = list(
        validator_type(schema).iter_errors(
            {
                "query": "DebugToken.java src/main/java/dev/mmm/debugfixture",
                "minecraft_version": 26.2,
                "limit": 1,
            }
        )
    )

    assert errors == []


def test_search_project_rag_normalizes_numeric_version_before_core(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class FakeCore:
        def search_project_rag(self, query, minecraft_version, limit):
            observed.update(
                query=query,
                minecraft_version=minecraft_version,
                limit=limit,
            )
            return {"ok": True}

    monkeypatch.setattr(mcp_server, "_core", lambda: FakeCore())

    result = mcp_server.search_project_rag(
        "DebugToken.java src/main/java/dev/mmm/debugfixture",
        26.2,
        1,
    )

    assert result == {"ok": True}
    assert observed == {
        "query": "DebugToken.java src/main/java/dev/mmm/debugfixture",
        "minecraft_version": "26.2",
        "limit": 1,
    }
