from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{path}: expected one replacement, found {count}: {old[:100]!r}"
        )
    file_path.write_text(text.replace(old, new), encoding="utf-8")


# The federation router itself enforces the provider scope, including fallbacks.
path = "minecraft_mod_ai/external_mcp_router.py"
replace_once(path, "from typing import Any, Mapping\n", "from typing import Any, Collection, Mapping\n")
replace_once(
    path,
    "class ExternalMCPError(RuntimeError):\n    pass\n\n\n@dataclass(frozen=True)",
    "class ExternalMCPError(RuntimeError):\n    pass\n\n\ndef _server_scope(values: Collection[str] | None) -> frozenset[str] | None:\n    if values is None:\n        return None\n    return frozenset(\n        value\n        for raw in values\n        if (value := str(raw).strip())\n    )\n\n\n@dataclass(frozen=True)",
)
replace_once(
    path,
    "    def capability_manifest(\n        self,\n        *,\n        stage: str,\n        target: Any = None,\n        max_access: str = \"read\",\n    ) -> dict[str, Any]:\n        resolved = MCPRouteTarget.from_value(target)\n",
    "    def capability_manifest(\n        self,\n        *,\n        stage: str,\n        target: Any = None,\n        max_access: str = \"read\",\n        allowed_server_ids: Collection[str] | None = None,\n    ) -> dict[str, Any]:\n        resolved = MCPRouteTarget.from_value(target)\n        allowed_servers = _server_scope(allowed_server_ids)\n",
)
replace_once(
    path,
    "            routes = self.registry.routes(\n                capability,\n                stage=stage,\n                minecraft_version=resolved.minecraft_version,\n                loader=resolved.loader,\n                max_access=max_access,\n            )\n            if routes:\n",
    "            routes = self.registry.routes(\n                capability,\n                stage=stage,\n                minecraft_version=resolved.minecraft_version,\n                loader=resolved.loader,\n                max_access=max_access,\n            )\n            if allowed_servers is not None:\n                routes = [\n                    route\n                    for route in routes\n                    if str(route[\"server\"]) in allowed_servers\n                ]\n            if routes:\n",
)
replace_once(
    path,
    "        required: bool = False,\n        max_access: str = \"read\",\n        disposable_runtime: bool = False,\n    ) -> dict[str, Any]:\n",
    "        required: bool = False,\n        max_access: str = \"read\",\n        disposable_runtime: bool = False,\n        allowed_server_ids: Collection[str] | None = None,\n    ) -> dict[str, Any]:\n",
)
replace_once(
    path,
    "        resolved = MCPRouteTarget.from_value(target)\n        if stage != \"runtime\" and max_access != \"read\":\n",
    "        resolved = MCPRouteTarget.from_value(target)\n        allowed_servers = _server_scope(allowed_server_ids)\n        if stage != \"runtime\" and max_access != \"read\":\n",
)
replace_once(
    path,
    "        routes = self.registry.routes(\n            capability,\n            stage=stage,\n            minecraft_version=resolved.minecraft_version,\n            loader=resolved.loader,\n            max_access=max_access,\n        )\n        attempts: list[dict[str, Any]] = []\n",
    "        routes = self.registry.routes(\n            capability,\n            stage=stage,\n            minecraft_version=resolved.minecraft_version,\n            loader=resolved.loader,\n            max_access=max_access,\n        )\n        if allowed_servers is not None:\n            routes = [\n                route\n                for route in routes\n                if str(route[\"server\"]) in allowed_servers\n            ]\n        attempts: list[dict[str, Any]] = []\n",
)

# The model bridge carries the scope through discovery, schema caching, and calls.
path = "minecraft_mod_ai/external_agent_bridge.py"
replace_once(path, "from typing import Any, Mapping\n", "from typing import Any, Collection, Mapping\n")
replace_once(
    path,
    "        self._schema_cache: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}\n",
    "        self._schema_cache: dict[\n            tuple[str, str, str, str, str, tuple[str, ...] | None], dict[str, Any]\n        ] = {}\n",
)
replace_once(
    path,
    "    def call(\n        self,\n        stage: str,\n        name: str,\n        payload: Mapping[str, Any],\n    ) -> dict[str, Any]:\n",
    "    def call(\n        self,\n        stage: str,\n        name: str,\n        payload: Mapping[str, Any],\n        *,\n        allowed_server_ids: Collection[str] | None = None,\n    ) -> dict[str, Any]:\n",
)
replace_once(
    path,
    "        router = self._external_router()\n\n        if name == CAPABILITIES_TOOL:\n",
    "        router = self._external_router()\n        allowed_servers = (\n            None\n            if allowed_server_ids is None\n            else frozenset(\n                value\n                for raw in allowed_server_ids\n                if (value := str(raw).strip())\n            )\n        )\n\n        if name == CAPABILITIES_TOOL:\n",
)
replace_once(
    path,
    "                max_access=max_access,\n            )\n\n        capability = str(payload.get(\"capability\", \"\")).strip()\n",
    "                max_access=max_access,\n                allowed_server_ids=allowed_servers,\n            )\n\n        capability = str(payload.get(\"capability\", \"\")).strip()\n",
)
replace_once(
    path,
    "                target[\"mappings\"],\n            )\n",
    "                target[\"mappings\"],\n                None if allowed_servers is None else tuple(sorted(allowed_servers)),\n            )\n",
)
replace_once(
    path,
    "                target,\n                max_access,\n            )\n",
    "                target,\n                max_access,\n                allowed_servers,\n            )\n",
)
replace_once(
    path,
    "            max_access=max_access,\n            disposable_runtime=bool(payload.get(\"disposable_runtime\", False)),\n        )\n",
    "            max_access=max_access,\n            disposable_runtime=bool(payload.get(\"disposable_runtime\", False)),\n            allowed_server_ids=allowed_servers,\n        )\n",
)
replace_once(
    path,
    "        target: Mapping[str, str],\n        max_access: str,\n    ) -> dict[str, Any]:\n",
    "        target: Mapping[str, str],\n        max_access: str,\n        allowed_server_ids: Collection[str] | None,\n    ) -> dict[str, Any]:\n",
)
replace_once(
    path,
    "        routes = router.registry.routes(\n            capability,\n            stage=stage,\n            minecraft_version=resolved.minecraft_version,\n            loader=resolved.loader,\n            max_access=max_access,\n        )\n        attempts: list[dict[str, Any]] = []\n",
    "        routes = router.registry.routes(\n            capability,\n            stage=stage,\n            minecraft_version=resolved.minecraft_version,\n            loader=resolved.loader,\n            max_access=max_access,\n        )\n        if allowed_server_ids is not None:\n            routes = [\n                route\n                for route in routes\n                if str(route[\"server\"]) in allowed_server_ids\n            ]\n        attempts: list[dict[str, Any]] = []\n",
)

# The runtime keeps an unscoped host API but exposes an explicit model-scoped path.
path = "minecraft_mod_ai/agent_tool_runtime.py"
replace_once(path, "from typing import Any, Mapping\n", "from typing import Any, Collection, Mapping\n")
replace_once(
    path,
    '''    def call(\n        self,\n        stage: str,\n        name: str,\n        arguments: Mapping[str, Any] | None = None,\n    ) -> dict[str, Any]:\n        selected = self._stage(stage)\n        tool_name = name.strip()\n        if not tool_name:\n            raise AgentToolRuntimeError("tool name must not be empty")\n        if tool_name in _BLOCKED_MODEL_TOOLS:\n            raise AgentToolRuntimeError(\n                f"Tool {tool_name!r} is intentionally not model-callable."\n            )\n        # Materialize the authoritative stage schema once. Keep the immutable name\n        # set beside it so hot-path calls do not rebuild the same set repeatedly.\n        self.tool_schemas(selected)\n        with self._lock:\n            allowed = self._allowed_tool_cache[selected]\n        if tool_name not in allowed:\n            raise AgentToolRuntimeError(\n                f"Tool {tool_name!r} is not exposed in stage {selected!r}."\n            )\n        payload = dict(arguments or {})\n        if tool_name in EXTERNAL_TOOL_NAMES:\n            return _bounded_result(\n                self._external_bridge.call(selected, tool_name, payload)\n            )\n        result = self._run_async(\n            self._call_tool_async,\n            selected,\n            tool_name,\n            payload,\n        )\n        return _bounded_result(result)\n\n''',
    '''    def call(\n        self,\n        stage: str,\n        name: str,\n        arguments: Mapping[str, Any] | None = None,\n    ) -> dict[str, Any]:\n        """Host-stage call. ModelRouter uses call_scoped for model-owned execution."""\n        return self._call(\n            stage,\n            name,\n            arguments,\n            external_server_ids=None,\n        )\n\n    def call_scoped(\n        self,\n        stage: str,\n        name: str,\n        arguments: Mapping[str, Any] | None = None,\n        *,\n        external_server_ids: Collection[str],\n    ) -> dict[str, Any]:\n        """Execute a model tool while enforcing its reviewed external MCP providers."""\n        return self._call(\n            stage,\n            name,\n            arguments,\n            external_server_ids=frozenset(\n                value\n                for raw in external_server_ids\n                if (value := str(raw).strip())\n            ),\n        )\n\n    def _call(\n        self,\n        stage: str,\n        name: str,\n        arguments: Mapping[str, Any] | None,\n        *,\n        external_server_ids: frozenset[str] | None,\n    ) -> dict[str, Any]:\n        selected = self._stage(stage)\n        tool_name = name.strip()\n        if not tool_name:\n            raise AgentToolRuntimeError("tool name must not be empty")\n        if tool_name in _BLOCKED_MODEL_TOOLS:\n            raise AgentToolRuntimeError(\n                f"Tool {tool_name!r} is intentionally not model-callable."\n            )\n        self.tool_schemas(selected)\n        with self._lock:\n            allowed = self._allowed_tool_cache[selected]\n        if tool_name not in allowed:\n            raise AgentToolRuntimeError(\n                f"Tool {tool_name!r} is not exposed in stage {selected!r}."\n            )\n        payload = dict(arguments or {})\n        if tool_name in EXTERNAL_TOOL_NAMES:\n            return _bounded_result(\n                self._external_bridge.call(\n                    selected,\n                    tool_name,\n                    payload,\n                    allowed_server_ids=external_server_ids,\n                )\n            )\n        result = self._run_async(\n            self._call_tool_async,\n            selected,\n            tool_name,\n            payload,\n        )\n        return _bounded_result(result)\n\n''',
)

# One canonical stage/physical-role resolver supplies the execution allowlist.
path = "minecraft_mod_ai/agent_capability_context.py"
replace_once(
    path,
    "    return selected_role\n\n\n@lru_cache(maxsize=8)\n",
    "    return selected_role\n\n\ndef reviewed_mcp_servers_for_model_role(\n    stage: str, model_role: str\n) -> frozenset[str]:\n    \"\"\"Return reviewed external MCP servers for this logical agent turn.\"\"\"\n    return frozenset(\n        mcp_servers_for_model_role(_policy_model_role(stage, model_role))\n    )\n\n\n@lru_cache(maxsize=8)\n",
)
replace_once(
    path,
    "    reviewed_servers = mcp_servers_for_model_role(policy_role)\n",
    "    reviewed_servers = reviewed_mcp_servers_for_model_role(selected, model_role)\n",
)

# Model execution is fail-closed for external MCP if its runtime lacks role scoping.
path = "minecraft_mod_ai/model_router.py"
replace_once(
    path,
    "        from .agent_capability_context import skills_for_tool\n",
    "        from .agent_capability_context import (\n            reviewed_mcp_servers_for_model_role,\n            skills_for_tool,\n        )\n",
)
replace_once(
    path,
    "        require_rag = bool(\n            self._agent_require_fresh_evidence\n            and role in {\"coder\", \"coder_safe\"}\n            and exposed_tools & _RAG_EVIDENCE_TOOLS\n        )\n\n        while True:\n",
    "        require_rag = bool(\n            self._agent_require_fresh_evidence\n            and role in {\"coder\", \"coder_safe\"}\n            and exposed_tools & _RAG_EVIDENCE_TOOLS\n        )\n        reviewed_external_servers = reviewed_mcp_servers_for_model_role(stage, role)\n\n        while True:\n",
)
replace_once(
    path,
    "                    result = runtime.call(stage, call.name, call.arguments)\n",
    "                    scoped_call = getattr(runtime, \"call_scoped\", None)\n                    if callable(scoped_call):\n                        result = scoped_call(\n                            stage,\n                            call.name,\n                            call.arguments,\n                            external_server_ids=reviewed_external_servers,\n                        )\n                    elif call.name.startswith(\"external_mcp_\"):\n                        raise ModelConfigurationError(\n                            \"External MCP execution requires a role-scoped agent runtime.\"\n                        )\n                    else:\n                        result = runtime.call(stage, call.name, call.arguments)\n",
)

# Real generation-stage MCP subprocess handshake.
path = "tests/test_mcp_stdio.py"
p = Path(path)
text = p.read_text(encoding="utf-8")
if "test_real_stdio_generation_stage_exposes_coder_evidence_tools" not in text:
    p.write_text(
        text.rstrip()
        + '''\n\n\ndef test_real_stdio_generation_stage_exposes_coder_evidence_tools(tmp_path: Path) -> None:\n    tools = asyncio.run(_handshake(tmp_path / "workspace", "generation"))\n    assert "inspect_existing_mod" in tools\n    assert "search_project_rag" in tools\n    assert "search_code_rag" in tools\n    assert "runtime_start_server" not in tools\n    assert "package_release" not in tools\n    assert "plan_complete_game" not in tools\n''',
        encoding="utf-8",
    )

Path("tests/test_external_mcp_role_scope.py").write_text(
    '''from __future__ import annotations\n\nfrom pathlib import Path\nfrom types import SimpleNamespace\n\nfrom minecraft_mod_ai.agent_tool_runtime import AgentToolRuntime\nfrom minecraft_mod_ai.external_agent_bridge import (\n    CALL_TOOL,\n    CAPABILITIES_TOOL,\n    SCHEMA_TOOL,\n    ExternalAgentBridge,\n)\nfrom minecraft_mod_ai.external_mcp import ExternalMCPRegistry\nfrom minecraft_mod_ai.external_mcp_router import ExternalMCPRouter\nfrom minecraft_mod_ai.model_adapters import GenerationResponse, ToolCall\nfrom minecraft_mod_ai.model_router import ModelRouter\n\n\ndef _registry(tmp_path: Path) -> ExternalMCPRegistry:\n    path = tmp_path / "role-scope-mcp.yaml"\n    path.write_text(\n        """schema_version: mmm/external-mcp-registry-v2\nservers:\n  allowed-provider:\n    status: enabled\n    transport: stdio\n    command: [allowed]\n    version_policy: dynamic\n    loaders: [fabric]\n    trust: test\n    capabilities:\n      source_search:\n        tool: allowed_search\n        access: read\n        stages: [generation]\n        priority: 20\n  forbidden-provider:\n    status: enabled\n    transport: stdio\n    command: [forbidden]\n    version_policy: dynamic\n    loaders: [fabric]\n    trust: test\n    capabilities:\n      source_search:\n        tool: forbidden_search\n        access: read\n        stages: [generation]\n        priority: 1\n      forbidden_only:\n        tool: forbidden_only\n        access: read\n        stages: [generation]\n        priority: 1\n""",\n        encoding="utf-8",\n    )\n    return ExternalMCPRegistry(path)\n\n\ndef test_router_filters_manifest_and_invocation_to_role_server_scope(tmp_path, monkeypatch) -> None:\n    router = ExternalMCPRouter(_registry(tmp_path))\n    manifest = router.capability_manifest(\n        stage="generation", allowed_server_ids={"allowed-provider"}\n    )\n    assert set(manifest["capabilities"]) == {"source_search"}\n    assert [row["server"] for row in manifest["capabilities"]["source_search"]] == [\n        "allowed-provider"\n    ]\n    calls: list[str] = []\n\n    def fake_call(server_name, entry, *, tool, arguments):\n        calls.append(server_name)\n        return {\n            "server_info": {"name": server_name},\n            "result": {"structured": {"hits": [server_name]}},\n        }\n\n    monkeypatch.setattr(router, "_call_provider", fake_call)\n    bundle = router.invoke(\n        "source_search",\n        stage="generation",\n        arguments={"query": "Block"},\n        allowed_server_ids={"allowed-provider"},\n    )\n    assert bundle["status"] == "PASS"\n    assert calls == ["allowed-provider"]\n    assert bundle["evidence"][0]["server"] == "allowed-provider"\n    denied = router.invoke(\n        "forbidden_only",\n        stage="generation",\n        allowed_server_ids={"allowed-provider"},\n    )\n    assert denied["status"] == "UNAVAILABLE"\n    assert denied["attempts"] == []\n\n\ndef test_bridge_schema_cache_isolated_by_server_scope(monkeypatch) -> None:\n    bridge = ExternalAgentBridge()\n    seen: list[frozenset[str]] = []\n\n    async def fake_describe(stage, capability, target, max_access, allowed_server_ids):\n        scope = frozenset(allowed_server_ids or ())\n        seen.append(scope)\n        return {"server": sorted(scope)[0], "status": "PASS"}\n\n    monkeypatch.setattr(bridge, "_describe_async", fake_describe)\n    one = bridge.call(\n        "generation",\n        SCHEMA_TOOL,\n        {"capability": "source_search"},\n        allowed_server_ids={"provider-a"},\n    )\n    two = bridge.call(\n        "generation",\n        SCHEMA_TOOL,\n        {"capability": "source_search"},\n        allowed_server_ids={"provider-b"},\n    )\n    again = bridge.call(\n        "generation",\n        SCHEMA_TOOL,\n        {"capability": "source_search"},\n        allowed_server_ids={"provider-a"},\n    )\n    assert one["server"] == "provider-a"\n    assert two["server"] == "provider-b"\n    assert again == one\n    assert seen == [frozenset({"provider-a"}), frozenset({"provider-b"})]\n\n\ndef test_agent_runtime_propagates_exact_external_server_scope(tmp_path, monkeypatch) -> None:\n    runtime = AgentToolRuntime(profile="test", workspace_root=tmp_path)\n    runtime._schema_cache["generation"] = tuple(\n        ExternalAgentBridge.tool_schemas("generation")\n    )\n    runtime._allowed_tool_cache["generation"] = frozenset(\n        {CAPABILITIES_TOOL, SCHEMA_TOOL, CALL_TOOL}\n    )\n    seen: dict[str, object] = {}\n\n    def fake_bridge(stage, name, payload, *, allowed_server_ids=None):\n        seen["stage"] = stage\n        seen["name"] = name\n        seen["scope"] = frozenset(allowed_server_ids or ())\n        return {"status": "PASS"}\n\n    monkeypatch.setattr(runtime._external_bridge, "call", fake_bridge)\n    result = runtime.call_scoped(\n        "generation",\n        CALL_TOOL,\n        {"capability": "source_search", "arguments": {}},\n        external_server_ids={"minecraft-dev", "mcmodding-docs"},\n    )\n    assert result["status"] == "PASS"\n    assert seen == {\n        "stage": "generation",\n        "name": CALL_TOOL,\n        "scope": frozenset({"minecraft-dev", "mcmodding-docs"}),\n    }\n\n\nclass _Registry:\n    def load_profile(self, name):\n        return object()\n\n    def role(self, profile, role):\n        return SimpleNamespace(\n            role=role,\n            provider="local",\n            adapter="llama_cpp",\n            exclusive_gpu=False,\n        )\n\n\nclass _ExternalRuntime:\n    def __init__(self) -> None:\n        self.scopes: list[frozenset[str]] = []\n        self.unscoped_calls = 0\n\n    def tool_schemas(self, stage):\n        return ExternalAgentBridge.tool_schemas(stage)\n\n    def call_scoped(self, stage, name, arguments, *, external_server_ids):\n        self.scopes.append(frozenset(external_server_ids))\n        return {"status": "PASS", "capability": arguments.get("capability")}\n\n    def call(self, stage, name, arguments):\n        self.unscoped_calls += 1\n        raise AssertionError("external MCP model calls must never use the unscoped path")\n\n\nclass _ExternalAdapter:\n    def __init__(self) -> None:\n        self.count = 0\n\n    def generate_turn(self, request):\n        self.count += 1\n        if self.count == 1:\n            return GenerationResponse(\n                tool_calls=(\n                    ToolCall(\n                        id="mcp_1",\n                        name=CALL_TOOL,\n                        arguments={"capability": "source_search", "arguments": {}},\n                        raw_arguments='{\"capability\":\"source_search\",\"arguments\":{}}',\n                    ),\n                )\n            )\n        return GenerationResponse(content="done")\n\n\ndef test_model_router_enforces_minecraftcoder_server_scope(monkeypatch) -> None:\n    runtime = _ExternalRuntime()\n    adapter = _ExternalAdapter()\n    router = ModelRouter(\n        profile="test",\n        registry=_Registry(),\n        agent_tool_runtime_factory=lambda **_: runtime,\n    )\n    monkeypatch.setattr(router, "_new_text_adapter", lambda config, role: adapter)\n    assert router.generate_text(\n        "coder",\n        ({"role": "user", "content": "inspect exact Minecraft API"},),\n        tool_stage="generation",\n    ) == "done"\n    assert runtime.unscoped_calls == 0\n    assert len(runtime.scopes) == 1\n    scope = runtime.scopes[0]\n    assert "minecraft-dev" in scope\n    assert "mcmodding-docs" in scope\n    assert "minecraft-wiki" in scope\n    assert "fabric-game-runtime" not in scope\n    assert "minecraft-player-agent" not in scope\n\n\ndef test_model_router_fails_closed_for_external_runtime_without_scope(monkeypatch) -> None:\n    class UnscopedRuntime:\n        def tool_schemas(self, stage):\n            return ExternalAgentBridge.tool_schemas(stage)\n\n        def call(self, stage, name, arguments):\n            raise AssertionError("unscoped external path must not execute")\n\n    adapter = _ExternalAdapter()\n    router = ModelRouter(\n        profile="test",\n        registry=_Registry(),\n        agent_tool_runtime_factory=lambda **_: UnscopedRuntime(),\n    )\n    monkeypatch.setattr(router, "_new_text_adapter", lambda config, role: adapter)\n    assert router.generate_text(\n        "coder",\n        ({"role": "user", "content": "inspect"},),\n        tool_stage="generation",\n    ) == "done"\n    assert adapter.count == 2\n''',
    encoding="utf-8",
)

# Final tree must not contain this one-shot machinery.
Path(".github/workflows/apply-mcp-role-scope-once.yml").unlink(missing_ok=True)
Path(".github/scripts/apply_mcp_role_scope_once.py").unlink(missing_ok=True)
