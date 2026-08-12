from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Mapping

import anyio


class AgentToolRuntimeError(RuntimeError):
    pass


# These tools either recurse back into the same planning/execution model loop or
# transfer control that must remain owned by the durable host workflow. Tool calls
# themselves never require user approval; the model may call every other tool exposed
# by the active MCP stage directly.
_BLOCKED_MODEL_TOOLS = frozenset(
    {
        "plan_game",
        "plan_complete_game",
        "revise_plan",
        "revise_complete_plan",
        "approve_plan",
        "approve_complete_plan",
        "execute_complete_project",
        "generate_assets",
        "repair_project",
        "run_model_smoke",
    }
)
_VALID_STAGES = frozenset(
    {
        "frontdoor",
        "planning",
        "research",
        "generation",
        "quality",
        "runtime",
        "release",
        "training",
    }
)
_MAX_TOOL_RESULT_BYTES = 128 * 1024


class AgentToolRuntime:
    """Expose the existing stage-scoped MMM MCP server to a local/remote LLM.

    The MCP server remains the single source of truth for tool schemas and execution.
    This host only converts MCP tool schemas to the OpenAI function-tool shape and
    bridges synchronous model generation to the asynchronous MCP Python client.
    """

    def __init__(
        self,
        *,
        profile: str,
        workspace_root: str | Path | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        if not profile.strip():
            raise ValueError("profile must not be empty")
        if not 1.0 <= float(timeout_seconds) <= 600.0:
            raise ValueError("timeout_seconds must be between 1 and 600")
        self.profile = profile.strip()
        configured_workspace = (
            str(workspace_root)
            if workspace_root is not None
            else os.environ.get("MMM_WORKSPACE", "mmm-output")
        )
        self.workspace_root = str(Path(configured_workspace).expanduser().resolve())
        self.timeout_seconds = float(timeout_seconds)
        self._schema_cache: dict[str, tuple[dict[str, Any], ...]] = {}
        self._lock = threading.RLock()

    def tool_schemas(self, stage: str) -> tuple[dict[str, Any], ...]:
        selected = self._stage(stage)
        with self._lock:
            cached = self._schema_cache.get(selected)
            if cached is not None:
                return cached
            listed = self._run_async(self._list_tools_async, selected)
            schemas: list[dict[str, Any]] = []
            for item in listed:
                name = str(item.get("name", "")).strip()
                if not name or name in _BLOCKED_MODEL_TOOLS:
                    continue
                parameters = item.get("input_schema")
                if not isinstance(parameters, Mapping):
                    parameters = {"type": "object", "properties": {}}
                schemas.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": str(item.get("description", "")).strip(),
                            "parameters": dict(parameters),
                        },
                    }
                )
            result = tuple(schemas)
            self._schema_cache[selected] = result
            return result

    def call(
        self,
        stage: str,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        selected = self._stage(stage)
        tool_name = name.strip()
        if not tool_name:
            raise AgentToolRuntimeError("tool name must not be empty")
        if tool_name in _BLOCKED_MODEL_TOOLS:
            raise AgentToolRuntimeError(
                f"Tool {tool_name!r} is intentionally not model-callable."
            )
        allowed = {
            str(item["function"]["name"])
            for item in self.tool_schemas(selected)
        }
        if tool_name not in allowed:
            raise AgentToolRuntimeError(
                f"Tool {tool_name!r} is not exposed in stage {selected!r}."
            )
        payload = dict(arguments or {})
        result = self._run_async(
            self._call_tool_async,
            selected,
            tool_name,
            payload,
        )
        return _bounded_result(result)

    @staticmethod
    def _stage(stage: str) -> str:
        value = stage.strip().lower()
        if value not in _VALID_STAGES:
            raise ValueError(f"Unsupported agent tool stage: {stage!r}")
        return value

    def _child_env(self, stage: str) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "MMM_MCP_STAGE": stage,
                "MMM_MODEL_PROFILE": self.profile,
                "MMM_WORKSPACE": self.workspace_root,
                # If an MCP implementation internally uses a model, do not let that
                # nested model open another agent/MCP loop.
                "MMM_AGENT_TOOL_CHILD": "1",
            }
        )
        return env

    def _run_async(self, function: Any, *args: Any) -> Any:
        async def runner() -> Any:
            return await function(*args)

        with self._lock:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return anyio.run(runner)

            value: dict[str, Any] = {}
            errors: list[BaseException] = []

            def worker() -> None:
                try:
                    value["result"] = anyio.run(runner)
                except BaseException as exc:  # pragma: no cover - event-loop bridge
                    errors.append(exc)

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            thread.join(self.timeout_seconds + 5.0)
            if thread.is_alive():
                raise AgentToolRuntimeError("MCP synchronous bridge timed out")
            if errors:
                raise AgentToolRuntimeError(str(errors[0])) from errors[0]
            return value["result"]

    async def _list_tools_async(self, stage: str) -> list[dict[str, Any]]:
        async with self._session(stage) as session:
            listed = await session.list_tools()
            result: list[dict[str, Any]] = []
            for item in getattr(listed, "tools", ()) or ():
                schema = getattr(item, "inputSchema", None)
                if schema is None:
                    schema = getattr(item, "input_schema", None)
                result.append(
                    {
                        "name": str(getattr(item, "name", "")),
                        "description": str(getattr(item, "description", "") or ""),
                        "input_schema": _jsonable(schema),
                    }
                )
            return result

    async def _call_tool_async(
        self,
        stage: str,
        name: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        async with self._session(stage) as session:
            listed = await session.list_tools()
            available = {
                str(getattr(item, "name", ""))
                for item in getattr(listed, "tools", ()) or ()
            }
            if name not in available:
                raise AgentToolRuntimeError(
                    f"MCP stage {stage!r} does not expose tool {name!r}."
                )
            raw = await session.call_tool(name, arguments=dict(arguments))
            if bool(getattr(raw, "isError", getattr(raw, "is_error", False))):
                raise AgentToolRuntimeError(f"MCP tool {name!r} returned an error")
            return _normalize_tool_result(raw)

    def _session(self, stage: str):
        return _MCPStdioSession(
            stage=stage,
            env=self._child_env(stage),
            timeout_seconds=self.timeout_seconds,
        )


class _MCPStdioSession:
    def __init__(
        self,
        *,
        stage: str,
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> None:
        self.stage = stage
        self.env = dict(env)
        self.timeout_seconds = timeout_seconds
        self._stdio_cm: Any = None
        self._session_cm: Any = None
        self.session: Any = None
        self._timeout_cm: Any = None

    async def __aenter__(self) -> Any:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except Exception as exc:  # pragma: no cover - dependency failure
            raise AgentToolRuntimeError("The pinned MCP Python client is unavailable") from exc

        self._timeout_cm = anyio.fail_after(self.timeout_seconds)
        self._timeout_cm.__enter__()
        try:
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "minecraft_mod_ai.mcp_server"],
                env=self.env,
            )
            self._stdio_cm = stdio_client(params)
            read_stream, write_stream = await self._stdio_cm.__aenter__()
            self._session_cm = ClientSession(read_stream, write_stream)
            self.session = await self._session_cm.__aenter__()
            await self.session.initialize()
            return self.session
        except BaseException:
            await self.__aexit__(*sys.exc_info())
            raise

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if self._session_cm is not None:
                await self._session_cm.__aexit__(exc_type, exc, tb)
        finally:
            try:
                if self._stdio_cm is not None:
                    await self._stdio_cm.__aexit__(exc_type, exc, tb)
            finally:
                if self._timeout_cm is not None:
                    self._timeout_cm.__exit__(exc_type, exc, tb)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump(mode="json"))
    return str(value)


def _normalize_tool_result(raw: Any) -> dict[str, Any]:
    structured = getattr(raw, "structuredContent", None)
    if structured is None:
        structured = getattr(raw, "structured_content", None)
    texts: list[str] = []
    resources: list[Any] = []
    for item in getattr(raw, "content", ()) or ():
        text = getattr(item, "text", None)
        if isinstance(text, str):
            texts.append(text)
        else:
            resources.append(_jsonable(item))
    parsed_text: Any = None
    if len(texts) == 1:
        try:
            parsed_text = json.loads(texts[0])
        except json.JSONDecodeError:
            pass
    return {
        "structured_content": _jsonable(structured),
        "text": texts,
        "parsed_text": _jsonable(parsed_text),
        "resources": resources,
    }


def _bounded_result(result: Mapping[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    if len(encoded) <= _MAX_TOOL_RESULT_BYTES:
        return dict(result)
    preview = encoded[:_MAX_TOOL_RESULT_BYTES].decode("utf-8", errors="ignore")
    return {
        "truncated": True,
        "original_bytes": len(encoded),
        "preview": preview,
        "hint": "Use the tool's cursor/page/limit arguments to request a smaller result.",
    }
