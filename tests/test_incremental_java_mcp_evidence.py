"""Host-owned MCP fallback must not reclassify untrusted results as APIs."""
from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.incremental_java_mcp_evidence import augment_api_lookup


class FakeRouter:
    def __init__(self, *, success=True):
        self.calls = []
        self.success = success

    def invoke(self, capability, **kwargs):
        self.calls.append((capability, kwargs))
        if not self.success:
            return {"status": "UNAVAILABLE", "evidence": []}
        return {"status": "PASS", "evidence": [{
            "server": "fake-minecraft-docs", "tool": "search_docs",
            "result_sha256": "sha256:abc",
            "result": {"text": ["Signature for example.Registry is for reference only"]},
        }]}


def _builder(router, *, target=None):
    return SimpleNamespace(context={
        "target": target or {"minecraft_version": "26.2", "loader": "fabric",
                             "mappings": "OFFICIAL"},
        "host_mcp_router": router,
    })


def _miss(query="example.Registry", offset=0):
    return {"query": query, "offset": offset, "total": 0,
            "members": [], "declarations": [], "next_offset": None}


def test_only_zero_hit_local_page_invokes_readonly_mcp(monkeypatch):
    monkeypatch.delenv("MMM_HOST_MCP_EVIDENCE", raising=False)
    router = FakeRouter()
    builder = _builder(router)
    page = augment_api_lookup(builder, "example.Registry", _miss())
    data = page["external_mcp_evidence"]
    assert data["status"] == "UNVERIFIED_EXTERNAL_REFERENCE"
    assert data["provider"] == "fake-minecraft-docs"
    assert "members" in page and page["members"] == []
    assert "usage" in data and "not a verified" in data["usage"].lower()
    assert router.calls == [("source_search", {
        "stage": "generation", "max_access": "read",
        "target": builder.context["target"], "arguments": {"query": "example.Registry"},
        "corroborate": 1, "required": False,
    })]
    assert augment_api_lookup(builder, "example.Registry", _miss()) == page
    assert len(router.calls) == 1
    assert builder.context["_incremental_mcp_evidence_metrics"] == {
        "provider_attempts": 1, "provider_successes": 1,
        "cache_misses": 1, "cache_hits": 1,
    }


def test_local_verified_hit_and_nonfirst_page_never_call_mcp():
    router = FakeRouter()
    builder = _builder(router)
    found = {**_miss(), "total": 1, "members": [{"name": "good"}]}
    assert augment_api_lookup(builder, "example.Registry", found) == found
    assert augment_api_lookup(builder, "example.Registry", _miss(offset=10)) == _miss(offset=10)
    assert router.calls == []


def test_unavailable_results_are_cached_and_do_not_forge_symbols():
    router = FakeRouter(success=False)
    builder = _builder(router)
    assert augment_api_lookup(builder, "missing", _miss("missing")) == _miss("missing")
    assert augment_api_lookup(builder, "missing", _miss("missing")) == _miss("missing")
    assert len(router.calls) == 2  # docs first, then exact source
    assert all(kwargs["max_access"] == "read" for _, kwargs in router.calls)


def test_readonly_mcp_needs_known_platform_target_and_is_configurable(monkeypatch):
    router = FakeRouter()
    builder = _builder(router, target={"loader": "fabric"})
    page = _miss()
    assert augment_api_lookup(builder, "example.Registry", page) == page
    monkeypatch.setenv("MMM_HOST_MCP_EVIDENCE", "0")
    assert augment_api_lookup(_builder(router), "example.Registry", page) == page
    assert not router.calls


def test_nonproject_test_context_does_not_start_mcp_automatically(monkeypatch):
    monkeypatch.delenv("MMM_HOST_MCP_EVIDENCE", raising=False)
    builder = SimpleNamespace(context={"target": {"minecraft_version": "26.2"}})
    assert augment_api_lookup(builder, "missing", _miss("missing")) == _miss("missing")
    assert "host_mcp_router" not in builder.context



def test_compiler_missing_external_api_retrieves_evidence_without_model_tool_calls():
    from minecraft_mod_ai.incremental_java_mcp_evidence import recover_compiler_api_evidence

    router = FakeRouter()
    seen = []
    class API:
        def lookup(self, query, offset=0):
            seen.append((query, offset))
            return _miss(query, offset)
    builder = _builder(router)
    builder.api = API()
    builder.spec = {"target_path": "src/main/java/example/Feature.java"}
    builder.latest_api = {}
    recover_compiler_api_evidence(
        builder,
        "Feature.java:2: error: package net.minecraft.world.item does not exist"
    )
    assert seen
    assert builder.latest_api.get("external_mcp_evidence", {}).get("status") == "UNVERIFIED_EXTERNAL_REFERENCE"
    assert len(router.calls) == 1


def test_compiler_local_error_does_not_trigger_remote_evidence():
    from minecraft_mod_ai.incremental_java_mcp_evidence import recover_compiler_api_evidence

    router = FakeRouter()
    builder = _builder(router)
    builder.api = SimpleNamespace(lookup=lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("local compiler faults should not run API discovery")))
    builder.spec = {"target_path": "src/main/java/example/Feature.java"}
    builder.latest_api = {"previous": "preserved"}
    recover_compiler_api_evidence(builder, "Feature.java:3: error: incompatible types: int cannot be converted to boolean")
    assert builder.latest_api == {"previous": "preserved"}
    assert router.calls == []
