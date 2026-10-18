"""Do not reinstate external mod search transports in pre-design RAG."""
from minecraft_mod_ai import pre_design_grounded_rag as rag
from minecraft_mod_ai.catalog_first_grounded_rag import _domain_spec_occurrences


def test_external_mod_search_methods_are_absent():
    assert not hasattr(rag, "_search_modrinth")
    assert not hasattr(rag, "_search_curseforge")


def test_only_actual_nonempty_api_queries_are_authorized():
    rows = _domain_spec_occurrences({
        "providers": ["modrinth", "curseforge", "project_rag"],
        "queries": ["", "Minecraft 26.2 EntityType register", "   "],
    })
    assert rows == [
        ("Minecraft 26.2 EntityType register", ("project_rag",))
    ]
