from __future__ import annotations

from minecraft_mod_ai import production_tools


def test_production_service_close_only_closes_materialized_java_owner(tmp_path, monkeypatch) -> None:
    events = []

    class FakeJava:
        def __init__(self):
            events.append("created")

        def close(self):
            events.append("closed")

    monkeypatch.setattr(production_tools, "JavaLanguageService", FakeJava)
    service = production_tools.ProductionToolService(workspace_root=tmp_path)

    service.close()
    assert events == []

    assert service.java is service.java
    assert events == ["created"]
    service.close()
    assert events == ["created", "closed"]
    service.close()
    assert events == ["created", "closed"]
