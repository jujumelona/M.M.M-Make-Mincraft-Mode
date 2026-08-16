from __future__ import annotations

"""Cross-cutting latency fixes for the local MMM hot path.

The package already owns correctness contracts for planning, RAG, MCP and llama.cpp.
This module removes transport/work duplication without weakening those contracts:
structured JSON stops once a complete root object exists, MCP sessions are reused,
read-only external MCP calls are single-flight cached for the run, forced RAG queries
are deduplicated and packed, and automatic Best-of-N never serializes candidates on
a one-slot local llama server.
"""

import atexit
import copy
import hashlib
import json
import os
import queue
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from contextlib import AsyncExitStack
from contextvars import ContextVar
from functools import wraps
from typing import Any, Mapping, Sequence

import anyio

_MARKER = "_mmm_bottleneck_elimination_v1"
_MCP_STAGE: ContextVar[str] = ContextVar("mmm_external_mcp_stage", default="")
_POOL_LOCK = threading.RLock()
_WORKERS: dict[str, "_PersistentMCPWorker"] = {}
_READ_LOCK = threading.RLock()
_READ_CACHE: dict[str, dict[str, Any]] = {}
_READ_INFLIGHT: dict[str, Future[dict[str, Any]]] = {}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump(mode="json"))
        except TypeError:
            return _jsonable(model_dump())
    return str(value)


def _normalize_mcp_result(raw: Any) -> dict[str, Any]:
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


class _PersistentMCPWorker:
    """One long-lived MCP session owned by one daemon thread/event loop."""

    def __init__(
        self,
        *,
        transport: str,
        timeout_seconds: float,
        command: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        url: str = "",
    ) -> None:
        self.transport = transport
        self.timeout_seconds = float(timeout_seconds)
        self.command = tuple(str(value) for value in command)
        self.env = dict(env or {})
        self.url = str(url)
        self._requests: queue.Queue[Any] = queue.Queue()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._tools: tuple[dict[str, Any], ...] = ()
        self._tool_names: frozenset[str] = frozenset()
        self._server_info: Any = {}
        self._closed = False
        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"mmm-mcp-{transport}",
            daemon=True,
        )
        self._thread.start()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive() and not self._closed

    @property
    def server_info(self) -> Any:
        self._ensure_ready()
        return copy.deepcopy(self._server_info)

    def list_tools(self) -> tuple[dict[str, Any], ...]:
        self._ensure_ready()
        return tuple(copy.deepcopy(item) for item in self._tools)

    def call_tool(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self._ensure_ready()
        if tool not in self._tool_names:
            raise RuntimeError(
                f"Provider does not expose reviewed tool {tool!r}; "
                f"available={sorted(self._tool_names)}"
            )
        future: Future[dict[str, Any]] = Future()
        self._requests.put((str(tool), dict(arguments), future))
        try:
            return future.result(timeout=self.timeout_seconds + 5.0)
        except FutureTimeoutError as exc:
            raise TimeoutError(f"MCP tool {tool!r} timed out") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._requests.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def _ensure_ready(self) -> None:
        if not self._ready.wait(timeout=self.timeout_seconds + 5.0):
            raise TimeoutError("MCP provider startup timed out")
        if self._startup_error is not None:
            raise RuntimeError(
                "MCP provider startup failed: "
                f"{type(self._startup_error).__name__}: {self._startup_error}"
            ) from self._startup_error
        if not self._thread.is_alive() and not self._closed:
            raise RuntimeError("MCP provider worker exited unexpectedly")

    def _thread_main(self) -> None:
        try:
            anyio.run(self._serve)
        except BaseException as exc:  # pragma: no cover - defensive daemon boundary
            if not self._ready.is_set():
                self._startup_error = exc
                self._ready.set()
            self._fail_pending(exc)

    async def _serve(self) -> None:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamable_http_client
        except Exception as exc:
            self._startup_error = exc
            self._ready.set()
            return

        stack = AsyncExitStack()
        try:
            if self.transport == "stdio":
                if not self.command:
                    raise RuntimeError("MCP stdio command is empty")
                errlog = stack.enter_context(
                    tempfile.TemporaryFile(mode="w+", encoding="utf-8")
                )
                params = StdioServerParameters(
                    command=self.command[0],
                    args=list(self.command[1:]),
                    env=self.env or None,
                )
                read_stream, write_stream = await stack.enter_async_context(
                    stdio_client(params, errlog=errlog)
                )
            elif self.transport == "streamable_http":
                if not self.url:
                    raise RuntimeError("MCP HTTP URL is empty")
                read_stream, write_stream, _ = await stack.enter_async_context(
                    streamable_http_client(self.url)
                )
            else:
                raise RuntimeError(f"Unsupported persistent MCP transport: {self.transport}")

            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            with anyio.fail_after(self.timeout_seconds):
                initialized = await session.initialize()
                listed = await session.list_tools()

            tools: list[dict[str, Any]] = []
            for item in getattr(listed, "tools", ()) or ():
                schema = getattr(item, "inputSchema", None)
                if schema is None:
                    schema = getattr(item, "input_schema", None)
                tools.append(
                    {
                        "name": str(getattr(item, "name", "")),
                        "description": str(getattr(item, "description", "") or ""),
                        "input_schema": _jsonable(schema),
                    }
                )
            self._tools = tuple(tools)
            self._tool_names = frozenset(
                item["name"] for item in tools if item["name"]
            )
            self._server_info = _jsonable(
                getattr(initialized, "serverInfo", getattr(initialized, "server_info", None))
            )
            self._ready.set()

            while True:
                request = await anyio.to_thread.run_sync(self._requests.get)
                if request is None:
                    return
                tool, arguments, future = request
                if future.cancelled():
                    continue
                try:
                    if tool not in self._tool_names:
                        raise RuntimeError(
                            f"Provider does not expose reviewed tool {tool!r}; "
                            f"available={sorted(self._tool_names)}"
                        )
                    with anyio.fail_after(self.timeout_seconds):
                        raw = await session.call_tool(tool, arguments=arguments)
                    if bool(getattr(raw, "isError", getattr(raw, "is_error", False))):
                        raise RuntimeError(f"MCP tool {tool!r} returned an error result")
                    future.set_result(_normalize_mcp_result(raw))
                except BaseException as exc:
                    future.set_exception(exc)
        except BaseException as exc:
            if not self._ready.is_set():
                self._startup_error = exc
                self._ready.set()
            self._fail_pending(exc)
        finally:
            try:
                await stack.aclose()
            except BaseException:
                pass

    def _fail_pending(self, exc: BaseException) -> None:
        while True:
            try:
                request = self._requests.get_nowait()
            except queue.Empty:
                return
            if request is None:
                continue
            _tool, _arguments, future = request
            if not future.done():
                future.set_exception(exc)


def _worker(key: str, **kwargs: Any) -> _PersistentMCPWorker:
    with _POOL_LOCK:
        current = _WORKERS.get(key)
        if current is not None and current.alive:
            return current
        if current is not None:
            current.close()
        value = _PersistentMCPWorker(**kwargs)
        _WORKERS[key] = value
        return value


def _close_workers() -> None:
    with _POOL_LOCK:
        values = tuple(_WORKERS.values())
        _WORKERS.clear()
    for value in values:
        value.close()


atexit.register(_close_workers)


class _JsonObjectTracker:
    """Incrementally detect the end of one root JSON object."""

    def __init__(self) -> None:
        self.started = False
        self.depth = 0
        self.in_string = False
        self.escaped = False
        self.complete = False
        self.invalid = False

    def feed(self, text: str) -> bool:
        if self.invalid or self.complete:
            return self.complete
        for index, char in enumerate(text):
            if not self.started:
                if char.isspace():
                    continue
                if char in ("`", "<", "\n", "\r", ">"):
                    continue
                if char != "{":
                    self.invalid = True
                    return False
                self.started = True
                self.depth = 1
                continue
            if self.in_string:
                if self.escaped:
                    self.escaped = False
                elif char == "\\":
                    self.escaped = True
                elif char == '"':
                    self.in_string = False
                continue
            if char == '"':
                self.in_string = True
            elif char == "{":
                self.depth += 1
            elif char == "}":
                self.depth -= 1
                if self.depth < 0:
                    self.invalid = True
                    return False
                if self.depth == 0:
                    if any(not value.isspace() for value in text[index + 1 :]):
                        self.invalid = True
                        return False
                    self.complete = True
                    return True
        return False


def _install_json_early_stop() -> None:
    from . import llama_server_hardware_policy as hardware
    from . import llama_stream_efficiency_contract as stream_contract
    from .model_adapters import ModelBackendError

    current = hardware._strict_server_generate
    if getattr(current, "_mmm_json_root_early_stop", False):
        return

    @wraps(current)
    def generate(adapter: Any, request: Any, server_url: str) -> str:
        if (
            getattr(request, "response_format", None) != "json"
            or os.environ.get("MMM_LLAMA_JSON_EARLY_STOP", "1").strip().lower()
            in {"0", "false", "no", "off"}
            or os.environ.get("MMM_LLAMA_DETAILED_TELEMETRY", "").strip().lower()
            in {"1", "true", "yes", "on"}
        ):
            return current(adapter, request, server_url)

        try:
            payload = hardware._server_payload(adapter, request)
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
            endpoint = f"{server_url.rstrip('/')}/chat/completions"
            client = stream_contract._client(server_url)
            tracker = _JsonObjectTracker()
            pieces: list[str] = []
            reasoning_chars = 0
            saw_done = False
            host_complete = False
            started = time.monotonic()

            with client.stream("POST", endpoint, json=payload) as response:
                if response.status_code != 200:
                    response.read()
                    body = response.text.strip().replace("\n", " ")
                    if len(body) > 1200:
                        body = body[:1200] + "..."
                    raise RuntimeError(
                        f"llama server returned HTTP {response.status_code}"
                        + (f": {body}" if body else "")
                    )
                stream_contract._report_server_connection(server_url)
                for raw_line in response.iter_lines():
                    line = raw_line.strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        saw_done = True
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("llama server returned malformed SSE JSON") from exc
                    if not isinstance(chunk, dict):
                        continue
                    choices = chunk.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        continue
                    reasoning, content = hardware._stream_delta_parts(choice)
                    reasoning_chars += len(reasoning)
                    if not content:
                        continue
                    pieces.append(content)
                    if tracker.feed(content):
                        host_complete = True
                        break

            content = "".join(pieces).strip()
            if not content:
                if reasoning_chars:
                    raise RuntimeError(
                        "llama server produced reasoning deltas but no visible JSON content"
                    )
                raise RuntimeError("llama server stream produced no JSON content")
            def _parse_root_json_object(text: str) -> dict[str, Any]:
                try:
                    res = json.loads(text)
                    if isinstance(res, dict):
                        return res
                except Exception:
                    pass
                try:
                    res = json.loads(text, strict=False)
                    if isinstance(res, dict):
                        return res
                except Exception:
                    pass
                cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
                if "</think>" in cleaned:
                    cleaned = cleaned.split("</think>")[-1]
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip(), flags=re.MULTILINE)
                cleaned = re.sub(r"```\s*$", "", cleaned.strip(), flags=re.MULTILINE)
                cleaned = cleaned.replace("```", "").strip()
                try:
                    res = json.loads(cleaned, strict=False)
                    if isinstance(res, dict):
                        return res
                except Exception:
                    pass
                repaired = re.sub(
                    r'(?<=: ")(.*?)(?=")',
                    lambda m: m.group(1).replace("\n", "\\n").replace("\r", "").replace("\t", "\\t"),
                    cleaned,
                    flags=re.DOTALL,
                )
                res = json.loads(repaired, strict=False)
                if not isinstance(res, dict):
                    raise RuntimeError("structured response root must be a JSON object")
                return res

            if host_complete:
                _parse_root_json_object(content)
                print(
                    "llama server: structured JSON complete; decode cancelled",
                    f" content_chars={len(content)}",
                    f" elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
                return content
            if not saw_done:
                raise RuntimeError("llama server stream ended before JSON completion")
            _parse_root_json_object(content)
            return content
        except Exception as exc:
            if isinstance(exc, ModelBackendError):
                raise
            raise ModelBackendError(
                role=adapter.config.role,
                model_id=adapter.config.model_id,
                cause=exc,
            ) from exc

    generate._mmm_json_root_early_stop = True
    generate.__wrapped__ = current
    hardware._strict_server_generate = generate


def _active_local_single_slot() -> bool:
    raw = os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "").strip()
    if raw:
        try:
            return int(raw) <= 1
        except ValueError:
            return True
    return bool(os.environ.get("LLAMA_SERVER_URL", "").strip())


def _install_single_slot_search_guard() -> None:
    from . import agentic_optimization_contract as agentic

    planner = agentic._planner_candidate_count
    if not getattr(planner, "_mmm_single_slot_guard", False):
        @wraps(planner)
        def planner_count(request: Any, stage: str) -> int:
            width = int(planner(request, stage))
            if agentic._mode() == "auto" and width > 1 and _active_local_single_slot():
                return 1
            return width

        planner_count._mmm_single_slot_guard = True
        planner_count.__wrapped__ = planner
        agentic._planner_candidate_count = planner_count

    repair = agentic._repair_candidate_count
    if not getattr(repair, "_mmm_single_slot_guard", False):
        @wraps(repair)
        def repair_count(
            self: Any,
            evidence: Mapping[str, Any],
            memory: Sequence[Mapping[str, Any]],
        ) -> int:
            width = int(repair(self, evidence, memory))
            if agentic._mode() == "auto" and width > 1 and _active_local_single_slot():
                return 1
            return width

        repair_count._mmm_single_slot_guard = True
        repair_count.__wrapped__ = repair
        agentic._repair_candidate_count = repair_count


def _install_rag_efficiency() -> None:
    from dataclasses import asdict
    from . import agentic_pre_design_rag as rag

    units = rag._evidence_units
    if not getattr(units, "_mmm_packed_units", False):
        def packed_units(evidence: Mapping[str, Any]):
            batch: list[dict[str, Any]] = []
            batch_chars = 2
            batch_index = 0
            target = max(2048, int(rag._EVIDENCE_PAGE_CHARS * 0.88))

            def flush():
                nonlocal batch, batch_chars, batch_index
                if not batch:
                    return None
                value = (f"packed:{batch_index}", {"units": batch})
                batch_index += 1
                batch = []
                batch_chars = 2
                return value

            for unit_id, value in units(evidence):
                record = {"unit_id": unit_id, "value": value}
                rendered = json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
                size = len(rendered)
                if size > target:
                    pending = flush()
                    if pending is not None:
                        yield pending
                    yield unit_id, value
                    continue
                if batch and batch_chars + size + 1 > target:
                    pending = flush()
                    if pending is not None:
                        yield pending
                batch.append(record)
                batch_chars += size + 1
            pending = flush()
            if pending is not None:
                yield pending

        packed_units._mmm_packed_units = True
        packed_units.__wrapped__ = units
        rag._evidence_units = packed_units

    forced = rag._forced_rag_bundle
    if getattr(forced, "_mmm_deduplicated_queries", False):
        return

    @wraps(forced)
    def deduplicated_bundle(router: Any, research_brief: Mapping[str, Any]) -> dict[str, Any]:
        raw_domains = research_brief.get("domains")
        domains = [item for item in raw_domains or [] if isinstance(item, Mapping)]
        jobs: list[tuple[str, str]] = []
        unique_queries: list[str] = []
        seen_queries: set[str] = set()
        for domain in domains:
            domain_id = str(domain.get("domain_id", "")).strip()
            queries = domain.get("queries")
            if not domain_id or not isinstance(queries, list):
                continue
            for query in queries:
                text = str(query).strip()
                if not text:
                    continue
                jobs.append((domain_id, text))
                if text not in seen_queries:
                    seen_queries.add(text)
                    unique_queries.append(text)

        versions = rag._research_versions(router)
        code_index_path = rag._existing_code_index()
        local_state = threading.local()

        def project_search(query: str) -> dict[str, Any]:
            retriever = getattr(local_state, "retriever", None)
            if retriever is None:
                retriever = rag.AuthoritativeEvidenceRetriever()
                local_state.retriever = retriever
            sources: dict[str, dict[str, Any]] = {}
            errors: list[dict[str, str]] = []
            for version in versions:
                try:
                    catalog = rag.evidence_catalog_for_version(version)
                    limit = min(6, len(catalog))
                    for source in retriever.search(
                        query,
                        minecraft_version=version,
                        limit=limit,
                    ):
                        payload = asdict(source)
                        payload["matched_version"] = version
                        sources.setdefault(source.source_id, payload)
                except Exception as exc:
                    errors.append(
                        {
                            "minecraft_version": version,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            return {
                "schema_version": "mmm/forced-project-rag-query-v1",
                "sources": [sources[key] for key in sorted(sources)],
                "errors": errors,
            }

        def code_search(query: str) -> dict[str, Any]:
            if code_index_path is None:
                return {
                    "schema_version": "mmm/forced-code-rag-query-v1",
                    "status": "not_indexed",
                    "hits": [],
                }
            try:
                index = getattr(local_state, "code_index", None)
                if index is None:
                    index = rag.ProjectRAGIndex(code_index_path)
                    local_state.code_index = index
                result = index.search_with_receipt(
                    query,
                    limit=8,
                    semantic=False,
                    rerank=False,
                )
                return {
                    "schema_version": "mmm/forced-code-rag-query-v1",
                    "status": "searched",
                    "hits": [asdict(hit) for hit in result.hits],
                    "receipt": asdict(result.receipt),
                }
            except Exception as exc:
                return {
                    "schema_version": "mmm/forced-code-rag-query-v1",
                    "status": "error",
                    "hits": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }

        def run(query: str) -> tuple[str, dict[str, Any]]:
            return query, {
                "query": query,
                "query_sha256": rag._sha256_text(query),
                "project_rag": project_search(query),
                "code_rag": code_search(query),
            }

        results: dict[str, dict[str, Any]] = {}
        if unique_queries:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(
                max_workers=max(1, min(8, len(unique_queries))),
                thread_name_prefix="mmm_pre_design_rag",
            ) as pool:
                for query, result in pool.map(run, unique_queries):
                    results[query] = result

        by_domain: dict[str, list[dict[str, Any]]] = {
            str(item.get("domain_id", "")): [] for item in domains
        }
        for domain_id, query in jobs:
            by_domain.setdefault(domain_id, []).append(copy.deepcopy(results[query]))

        payload = {
            "schema_version": "mmm/forced-pre-design-rag-v2",
            "versions": list(versions),
            "domain_count": len(domains),
            "query_count": len(jobs),
            "unique_query_count": len(unique_queries),
            "project_source_count": sum(
                len(item.get("project_rag", {}).get("sources", []))
                for values in by_domain.values()
                for item in values
            ),
            "code_index_status": "available" if code_index_path is not None else "not_indexed",
            "code_index_path": str(code_index_path) if code_index_path is not None else "",
            "domains": [
                {
                    "domain_id": str(domain.get("domain_id", "")),
                    "queries": by_domain.get(str(domain.get("domain_id", "")), []),
                }
                for domain in domains
            ],
        }
        payload["research_sha256"] = rag._sha256(payload)
        return payload

    deduplicated_bundle._mmm_deduplicated_queries = True
    deduplicated_bundle.__wrapped__ = forced
    rag._forced_rag_bundle = deduplicated_bundle


def _mcp_worker_key(prefix: str, payload: Mapping[str, Any]) -> str:
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return prefix + ":" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _external_worker(router: Any, server_name: str, entry: Mapping[str, Any]) -> _PersistentMCPWorker:
    transport = str(entry.get("transport", ""))
    if transport == "stdio":
        command = entry.get("command")
        if not isinstance(command, list) or not command:
            raise RuntimeError(f"{server_name} has no stdio command")
        env = router._child_env(entry)
        key = _mcp_worker_key(
            "external",
            {
                "server": server_name,
                "transport": transport,
                "command": command,
                "env": env,
            },
        )
        return _worker(
            key,
            transport="stdio",
            timeout_seconds=router.timeout_seconds,
            command=command,
            env=env,
        )
    if transport == "streamable_http":
        url = router._server_url(entry)
        if not url:
            raise RuntimeError(f"{server_name} has no configured HTTP MCP URL")
        key = _mcp_worker_key(
            "external",
            {"server": server_name, "transport": transport, "url": url},
        )
        return _worker(
            key,
            transport="streamable_http",
            timeout_seconds=router.timeout_seconds,
            url=url,
        )
    raise RuntimeError(
        f"Federation does not invoke transport {transport!r} for {server_name}"
    )


def _read_only_tool(entry: Mapping[str, Any], tool: str) -> bool:
    matches: list[str] = []
    capabilities = entry.get("capabilities", {})
    if isinstance(capabilities, Mapping):
        for route in capabilities.values():
            if isinstance(route, Mapping) and str(route.get("tool", "")) == tool:
                matches.append(str(route.get("access", "read")))
    return bool(matches) and all(value == "read" for value in matches)


def _install_external_mcp_efficiency() -> None:
    from . import external_mcp_router as external

    invoke = external.ExternalMCPRouter.invoke
    if not getattr(invoke, "_mmm_stage_context", False):
        @wraps(invoke)
        def invoke_with_stage(self: Any, capability: str, **kwargs: Any) -> dict[str, Any]:
            stage = str(kwargs.get("stage", ""))
            token = _MCP_STAGE.set(stage)
            try:
                return invoke(self, capability, **kwargs)
            finally:
                _MCP_STAGE.reset(token)

        invoke_with_stage._mmm_stage_context = True
        invoke_with_stage.__wrapped__ = invoke
        external.ExternalMCPRouter.invoke = invoke_with_stage

    call_provider = external.ExternalMCPRouter._call_provider
    if getattr(call_provider, "_mmm_persistent_provider", False):
        return

    @wraps(call_provider)
    def pooled_call(
        self: Any,
        server_name: str,
        entry: Mapping[str, Any],
        *,
        tool: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            worker = _external_worker(self, server_name, entry)
            cacheable = _MCP_STAGE.get() != "runtime" and _read_only_tool(entry, tool)
            key = _mcp_worker_key(
                "read",
                {
                    "server": server_name,
                    "tool": tool,
                    "arguments": dict(arguments),
                    "worker": id(worker),
                },
            )
            if not cacheable:
                return {
                    "server_info": worker.server_info,
                    "result": worker.call_tool(tool, arguments),
                }

            with _READ_LOCK:
                cached = _READ_CACHE.get(key)
                if cached is not None:
                    return copy.deepcopy(cached)
                future = _READ_INFLIGHT.get(key)
                owner = future is None
                if owner:
                    future = Future()
                    _READ_INFLIGHT[key] = future
            assert future is not None
            if not owner:
                try:
                    return copy.deepcopy(
                        future.result(timeout=self.timeout_seconds + 5.0)
                    )
                except FutureTimeoutError as exc:
                    raise external.ExternalMCPError(
                        f"External MCP {server_name} single-flight timed out"
                    ) from exc

            try:
                value = {
                    "server_info": worker.server_info,
                    "result": worker.call_tool(tool, arguments),
                }
            except BaseException as exc:
                with _READ_LOCK:
                    _READ_INFLIGHT.pop(key, None)
                    if not future.done():
                        future.set_exception(exc)
                raise
            with _READ_LOCK:
                frozen = copy.deepcopy(value)
                _READ_CACHE[key] = frozen
                _READ_INFLIGHT.pop(key, None)
                if not future.done():
                    future.set_result(copy.deepcopy(frozen))
            return value
        except Exception as exc:
            if isinstance(exc, external.ExternalMCPError):
                raise
            raise external.ExternalMCPError(str(exc)) from exc

    pooled_call._mmm_persistent_provider = True
    pooled_call._mmm_parallel_sessions = True
    pooled_call.__wrapped__ = call_provider
    external.ExternalMCPRouter._call_provider = pooled_call


def _first_party_worker(runtime: Any, stage: str) -> _PersistentMCPWorker:
    env = runtime._child_env(stage)
    key = _mcp_worker_key(
        "first-party",
        {
            "profile": runtime.profile,
            "workspace": runtime.workspace_root,
            "stage": stage,
            "python": sys.executable,
        },
    )
    return _worker(
        key,
        transport="stdio",
        timeout_seconds=runtime.timeout_seconds,
        command=(sys.executable, "-m", "minecraft_mod_ai.mcp_server"),
        env=env,
    )


def _install_first_party_mcp_efficiency() -> None:
    from . import agent_tool_runtime as runtime_module

    current = runtime_module.AgentToolRuntime._run_async
    if getattr(current, "_mmm_persistent_first_party_mcp", False):
        return

    @wraps(current)
    def run_async(self: Any, function: Any, *args: Any) -> Any:
        name = str(getattr(function, "__name__", ""))
        if name == "_list_tools_async" and len(args) == 1:
            stage = str(args[0])
            return list(_first_party_worker(self, stage).list_tools())
        if name == "_call_tool_async" and len(args) == 3:
            stage, tool, arguments = args
            return _first_party_worker(self, str(stage)).call_tool(
                str(tool),
                dict(arguments),
            )
        return current(self, function, *args)

    run_async._mmm_persistent_first_party_mcp = True
    run_async.__wrapped__ = current
    runtime_module.AgentToolRuntime._run_async = run_async


def install() -> None:
    """Install every steady-state bottleneck fix exactly once."""
    if getattr(install, _MARKER, False):
        return
    _install_json_early_stop()
    _install_single_slot_search_guard()
    _install_rag_efficiency()
    _install_external_mcp_efficiency()
    _install_first_party_mcp_efficiency()
    setattr(install, _MARKER, True)


__all__ = ["_JsonObjectTracker", "install"]