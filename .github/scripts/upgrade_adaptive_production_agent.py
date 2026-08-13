from __future__ import annotations

import re
from pathlib import Path

ROOT = Path('.')
MODEL = ROOT / 'minecraft_mod_ai/model_router.py'
RUNTIME = ROOT / 'minecraft_mod_ai/agent_tool_runtime.py'
MCP = ROOT / 'minecraft_mod_ai/mcp_server.py'
SKILLS = ROOT / 'minecraft_mod_ai/skill_catalog.py'
CAPS = ROOT / 'minecraft_mod_ai/agent_capability_context.py'
PROD = ROOT / 'minecraft_mod_ai/production_tools.py'
CUSTOM = ROOT / 'minecraft_mod_ai/custom_module_generator.py'
REPAIR = ROOT / 'minecraft_mod_ai/repair_engine.py'
TEST = ROOT / 'tests/test_adaptive_production_agent.py'


def replace_block(source: str, header_prefix: str, replacement: str) -> str:
    lines = source.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.startswith(header_prefix)), -1)
    if start < 0:
        raise SystemExit(f'block not found: {header_prefix!r}')
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            continue
        current = len(line) - len(line.lstrip())
        stripped = line.lstrip()
        if current == indent and (
            stripped.startswith('def ')
            or stripped.startswith('async def ')
            or stripped.startswith('class ')
            or stripped.startswith('@')
        ):
            end = i
            break
        if current < indent:
            end = i
            break
    new_lines = lines[:start] + [replacement.rstrip() + '\n\n'] + lines[end:]
    return ''.join(new_lines)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if new in source:
        return source
    if old not in source:
        raise SystemExit(f'anchor not found: {label}')
    return source.replace(old, new, 1)


def patch_model_router() -> None:
    source = MODEL.read_text(encoding='utf-8')
    source = replace_once(
        source,
        'import threading\n',
        'import threading\nfrom concurrent.futures import ThreadPoolExecutor\n',
        'thread pool import',
    )
    constants_anchor = '_NATIVE_TOOL_ADAPTERS = frozenset({"llama_cpp", "vllm", "openai_compatible"})\n'
    constants = '''_NATIVE_TOOL_ADAPTERS = frozenset({"llama_cpp", "vllm", "openai_compatible"})
_RAG_EVIDENCE_TOOLS = frozenset({"search_code_rag", "search_project_rag"})
_PARALLEL_READ_TOOLS = frozenset(
    {
        "search_code_rag",
        "search_project_rag",
        "discover_ecosystem_resources",
        "inspect_modrinth_project",
        "inspect_github_repository",
        "inspect_huggingface_model",
        "inspect_existing_mod",
        "assess_technology_compatibility",
        "build_technology_radar",
        "java_diagnostics",
        "java_workspace_symbols",
        "read_complete_plan_section",
        "read_quality_contract",
        "quality_status",
        "work_status",
        "work_tasks",
        "external_mcp_capabilities",
        "external_mcp_schema",
    }
)
'''
    source = replace_once(source, constants_anchor, constants, 'agent tool constants')
    init_old = '''        self._agent_tool_runtime_factory = agent_tool_runtime_factory
        self._agent_tool_runtime: Any | None = None
'''
    init_new = '''        self._agent_tool_runtime_factory = agent_tool_runtime_factory
        self._agent_tool_runtime: Any | None = None
        self._agent_workspace_root: Path | None = None
        self._agent_require_fresh_evidence = False
'''
    source = replace_once(source, init_old, init_new, 'agent workspace state')

    marker = '    @contextmanager\n    def generation_session(self, role: str):\n'
    methods = '''    def bind_agent_workspace(
        self,
        workspace_root: str | Path,
        *,
        require_fresh_evidence: bool = False,
    ) -> "ModelRouter":
        """Bind model-callable MCP/RAG tools to one production workspace.

        Binding is explicit so a reusable router cannot accidentally search a stale
        default ``mmm-output`` tree while producing or repairing a different project.
        Changing the workspace invalidates the cached MCP runtime and its tool schemas.
        """

        root = Path(workspace_root).expanduser().resolve()
        if not root.is_dir() or root.is_symlink():
            raise ModelConfigurationError(
                f"Agent workspace must be a regular directory: {root}"
            )
        with self._generation_lock:
            if root != self._agent_workspace_root:
                self._agent_workspace_root = root
                self._agent_tool_runtime = None
            self._agent_require_fresh_evidence = bool(require_fresh_evidence)
        return self

    @contextmanager
    def generation_session(self, role: str):
'''
    if 'def bind_agent_workspace(' not in source:
        source = replace_once(source, marker, methods, 'bind agent workspace')

    old_call = '''                        runtime=runtime,
                        stage=stage,
                    )
'''
    new_call = '''                        runtime=runtime,
                        stage=stage,
                        role=role,
                    )
'''
    source = replace_once(source, old_call, new_call, 'tool loop role forwarding')

    tool_loop = '''    def _generate_with_tools(
        self,
        *,
        adapter: Any,
        request: GenerationRequest,
        runtime: Any,
        stage: str,
        role: str,
    ) -> str:
        """Run an adaptive retrieve/act/observe loop until evidence-backed completion.

        Production coders are not allowed to finalize from parametric memory alone
        when RAG tools are available. Retrieval quality is read from the RAG receipt;
        an empty/zero-evidence receipt causes a query/source correction instead of a
        blind final answer. Independent read-only evidence calls execute concurrently,
        while state-changing or unknown tools remain strictly ordered.
        """
        from .agent_capability_context import skills_for_tool

        messages: list[dict[str, Any]] = [dict(message) for message in request.messages]
        previous_exchange_state: str | None = None
        weak_fixed_point_seen = False
        premature_final_state: str | None = None
        rag_evidence_seen = False
        round_index = 0
        advertised = _tool_names(request.tools)
        require_rag = bool(
            self._agent_require_fresh_evidence
            and role in {"coder", "coder_safe"}
            and advertised & _RAG_EVIDENCE_TOOLS
        )

        while True:
            turn_request = GenerationRequest(
                messages=messages,
                media_paths=request.media_paths if round_index == 0 else (),
                response_format=request.response_format,
                response_schema=request.response_schema,
                tools=request.tools,
                tool_choice=request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
            )
            turn = adapter.generate_turn(turn_request)
            if not turn.tool_calls:
                content = turn.content.strip()
                if not content:
                    raise ModelConfigurationError(
                        "Tool-capable model returned an empty final response."
                    )
                if require_rag and not rag_evidence_seen:
                    state = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    if state == premature_final_state:
                        raise ModelConfigurationError(
                            "Production coder repeated a final answer without gathering "
                            "fresh RAG evidence."
                        )
                    premature_final_state = state
                    messages.extend(
                        [
                            {"role": "assistant", "content": content},
                            {
                                "role": "system",
                                "content": (
                                    "This is a production coding turn. Before finalizing, "
                                    "gather fresh project/API evidence with search_code_rag "
                                    "and/or search_project_rag. Do not guess exact Minecraft, "
                                    "Fabric, mapping, registry, networking, lifecycle or Java "
                                    "API facts from memory when a reviewed evidence route is "
                                    "available. Inspect the retrieval receipt; if evidence is "
                                    "empty or zero-coverage, reformulate the query or switch "
                                    "evidence source before returning the final answer."
                                ),
                            },
                        ]
                    )
                    round_index += 1
                    continue
                return content

            messages.append(
                {
                    "role": "assistant",
                    "content": turn.content or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.raw_arguments
                                or json.dumps(
                                    dict(call.arguments),
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            },
                        }
                        for call in turn.tool_calls
                    ],
                }
            )

            def execute(call: Any) -> tuple[Any, Mapping[str, Any]]:
                route_metadata: dict[str, Any] = {
                    "skills": list(skills_for_tool(stage, call.name)),
                }
                if call.name == "external_mcp_call":
                    capability = str(call.arguments.get("capability", "")).strip()
                    if capability:
                        route_metadata["external_mcp_capability"] = capability
                try:
                    result = runtime.call(stage, call.name, call.arguments)
                    payload: Mapping[str, Any] = {
                        "ok": True,
                        "tool": call.name,
                        **route_metadata,
                        "result": result,
                    }
                except Exception as exc:
                    payload = {
                        "ok": False,
                        "tool": call.name,
                        **route_metadata,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                return call, payload

            calls = tuple(turn.tool_calls)
            if len(calls) > 1 and all(
                call.name in _PARALLEL_READ_TOOLS for call in calls
            ):
                workers = min(len(calls), _parallel_read_workers())
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    executed = tuple(executor.map(execute, calls))
            else:
                executed = tuple(execute(call) for call in calls)

            observations: list[dict[str, Any]] = []
            weak_rag_in_round = False
            for call, payload in executed:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": json.dumps(
                            payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            default=str,
                        ),
                    }
                )
                observations.append(
                    {
                        "name": call.name,
                        "arguments": dict(call.arguments),
                        "observation": payload,
                    }
                )
                if call.name in _RAG_EVIDENCE_TOOLS and bool(payload.get("ok")):
                    if _usable_rag_result(payload.get("result")):
                        rag_evidence_seen = True
                    else:
                        weak_rag_in_round = True

            if require_rag and weak_rag_in_round and not rag_evidence_seen:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "The latest RAG observation did not contain usable fresh "
                            "evidence. Use the receipt/correction fields already returned "
                            "to reformulate the query, or switch between code RAG and "
                            "authoritative project/API RAG. Do not finalize yet and do not "
                            "repeat an identical weak retrieval."
                        ),
                    }
                )

            exchange_state = hashlib.sha256(
                json.dumps(
                    {
                        "assistant_content": turn.content or "",
                        "tool_exchanges": observations,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            if exchange_state == previous_exchange_state:
                if require_rag and not rag_evidence_seen:
                    if weak_fixed_point_seen:
                        raise ModelConfigurationError(
                            "Production RAG converged without usable fresh evidence after "
                            "a corrective retrieval instruction."
                        )
                    weak_fixed_point_seen = True
                    previous_exchange_state = None
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "An identical weak retrieval repeated. Change the query "
                                "substantively or use a different reviewed evidence source "
                                "before attempting a final production patch."
                            ),
                        }
                    )
                    round_index += 1
                    continue

                final_messages = [
                    *messages,
                    {
                        "role": "system",
                        "content": (
                            "Tool use has converged after usable evidence was gathered. "
                            "Do not call any more tools. Return the final answer now using "
                            "only the evidence already present in this conversation. "
                            "Preserve the requested response format and do not mention "
                            "this convergence instruction."
                        ),
                    },
                ]
                final_request = GenerationRequest(
                    messages=final_messages,
                    media_paths=(),
                    response_format=request.response_format,
                    response_schema=request.response_schema,
                    tools=(),
                    tool_choice=None,
                    parallel_tool_calls=False,
                )
                final_turn = adapter.generate_turn(final_request)
                if final_turn.tool_calls:
                    raise ModelConfigurationError(
                        "Agent emitted tool calls after tools were disabled at an exact "
                        "no-progress fixed point."
                    )
                final_content = final_turn.content.strip()
                if not final_content:
                    raise ModelConfigurationError(
                        "Agent returned an empty final response after exact tool "
                        "fixed-point convergence."
                    )
                return final_content
            previous_exchange_state = exchange_state
            round_index += 1
'''
    source = replace_block(source, '    def _generate_with_tools(', tool_loop)

    tool_runtime = '''    def _tool_runtime(self) -> Any:
        if self._agent_tool_runtime is not None:
            return self._agent_tool_runtime
        if self._agent_tool_runtime_factory is not None:
            # Custom factories are primarily test/integration seams and historically
            # accepted only ``profile``. Production uses AgentToolRuntime below, where
            # workspace ownership is explicit.
            self._agent_tool_runtime = self._agent_tool_runtime_factory(
                profile=self.profile
            )
        else:
            from .agent_tool_runtime import AgentToolRuntime

            self._agent_tool_runtime = AgentToolRuntime(
                profile=self.profile,
                workspace_root=self._agent_workspace_root,
            )
        return self._agent_tool_runtime
'''
    source = replace_block(source, '    def _tool_runtime(', tool_runtime)

    helper_anchor = '\n\nclass ModelRouter:'
    helpers = '''

def _tool_names(tools: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    names: set[str] = set()
    for schema in tools:
        function = schema.get("function")
        if isinstance(function, Mapping):
            name = str(function.get("name", "")).strip()
            if name:
                names.add(name)
    return frozenset(names)


def _parallel_read_workers() -> int:
    raw = os.environ.get("MMM_AGENT_PARALLEL_READS", "4").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 4
    return max(1, min(value, 16))


def _usable_rag_result(value: Any) -> bool:
    """Read RAG result/receipt semantics without depending on one MCP wrapper shape."""

    found_receipt = False
    found_nonempty_evidence = False

    def visit(item: Any) -> None:
        nonlocal found_receipt, found_nonempty_evidence
        if isinstance(item, Mapping):
            receipt = item.get("receipt")
            if isinstance(receipt, Mapping):
                found_receipt = True
                try:
                    if int(receipt.get("result_count", 0) or 0) > 0:
                        coverage = float(receipt.get("coverage_score", 0.0) or 0.0)
                        relevance = float(receipt.get("relevance_score", 0.0) or 0.0)
                        if coverage > 0.0 and relevance > 0.0:
                            found_nonempty_evidence = True
                except (TypeError, ValueError):
                    pass
            hits = item.get("hits")
            if isinstance(hits, Sequence) and not isinstance(hits, (str, bytes)) and hits:
                found_nonempty_evidence = True
            for child in item.values():
                visit(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for child in item:
                visit(child)

    visit(value)
    if found_receipt:
        return found_nonempty_evidence
    # Authoritative project RAG can return an evidence pack instead of the code-RAG
    # receipt shape. A successful non-empty structured payload is fresh evidence.
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return bool(value)
    return False
'''
    if 'def _usable_rag_result(' not in source:
        source = source.replace(helper_anchor, helpers + helper_anchor, 1)

    compile(source, str(MODEL), 'exec')
    MODEL.write_text(source, encoding='utf-8')


def patch_runtime() -> None:
    source = RUNTIME.read_text(encoding='utf-8')
    replacement = '''    def _run_async(self, function: Any, *args: Any) -> Any:
        """Bridge one independent MCP stdio session without globally serializing reads."""

        async def runner() -> Any:
            return await function(*args)

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
'''
    source = replace_block(source, '    def _run_async(', replacement)
    compile(source, str(RUNTIME), 'exec')
    RUNTIME.write_text(source, encoding='utf-8')


def expand_stage_map(path: Path) -> None:
    source = path.read_text(encoding='utf-8')
    replacements = {
        'frozenset({"frontdoor", "planning", "research"})': 'frozenset({"frontdoor", "planning", "research", "generation", "quality"})',
        'frozenset({"planning", "research"})': 'frozenset({"planning", "research", "generation", "quality"})',
        'frozenset({"frontdoor", "planning", "research"})': 'frozenset({"frontdoor", "planning", "research", "generation", "quality"})',
        '"search_project_rag": frozenset({"frontdoor", "planning", "research"})': '"search_project_rag": frozenset({"frontdoor", "planning", "research", "generation", "quality"})',
        '"inspect_existing_mod": frozenset({"frontdoor", "planning", "research"})': '"inspect_existing_mod": frozenset({"frontdoor", "planning", "research", "generation", "quality"})',
        '"java_diagnostics": frozenset({"quality"})': '"java_diagnostics": frozenset({"generation", "quality"})',
        '"java_workspace_symbols": frozenset({"quality"})': '"java_workspace_symbols": frozenset({"generation", "quality"})',
    }
    # Apply targeted named entries explicitly after broad stage forms. This expands
    # only read/evidence tools; mutating production tools retain their reviewed stages.
    for name in (
        'discover_ecosystem_resources',
        'inspect_modrinth_project',
        'inspect_github_repository',
        'inspect_huggingface_model',
        'build_technology_radar',
        'assess_technology_compatibility',
        'search_project_rag',
        'inspect_existing_mod',
    ):
        pattern = re.compile(
            rf'("{re.escape(name)}"\s*:\s*frozenset\()(?P<set>\{{[^}}]+\}})(\))',
            re.MULTILINE,
        )
        match = pattern.search(source)
        if not match:
            raise SystemExit(f'stage entry missing in {path}: {name}')
        values = {value.strip().strip('"\'') for value in match.group('set').strip('{}').split(',') if value.strip()}
        values.update({'generation', 'quality'})
        order = ['frontdoor', 'planning', 'research', 'generation', 'quality', 'runtime', 'release', 'training']
        rendered = '{' + ', '.join(f'"{value}"' for value in order if value in values) + '}'
        source = source[:match.start('set')] + rendered + source[match.end('set'):]
    for name in ('java_diagnostics', 'java_workspace_symbols'):
        pattern = re.compile(rf'("{name}"\s*:\s*frozenset\()(?P<set>\{{[^}}]+\}})(\))')
        match = pattern.search(source)
        if not match:
            raise SystemExit(f'stage entry missing in {path}: {name}')
        rendered = '{"generation", "quality"}'
        source = source[:match.start('set')] + rendered + source[match.end('set'):]
    compile(source, str(path), 'exec')
    path.write_text(source, encoding='utf-8')


def patch_capability_context() -> None:
    source = CAPS.read_text(encoding='utf-8')
    old = '''            "Choose every relevant Skill route, not every route indiscriminately. "
            "Use model_tools directly. host_owned_tools belong to the durable host "
            "pipeline and must not be recreated recursively. For an external MCP "
            "capability, use external_mcp_schema when its live arguments are unknown, "
            "then external_mcp_call. Prefer independent relevant evidence in parallel "
            "when it materially improves correctness; skip unrelated tools to avoid "
            "latency and token waste."
'''
    new = '''            "Choose every relevant Skill route, not every route indiscriminately. "
            "During production use an adaptive loop: retrieve fresh project/API evidence; "
            "inspect RAG receipt quality; reformulate or switch evidence source when weak; "
            "generate or repair; consume compiler/JDT/runtime feedback; then retrieve again "
            "when new uncertainty or errors appear. Never guess exact Minecraft/Fabric/API "
            "facts from parametric memory when reviewed evidence tools can resolve them. "
            "Use model_tools directly. host_owned_tools belong to the durable host pipeline "
            "and must not be recreated recursively. For an external MCP capability, use "
            "external_mcp_schema when its live arguments are unknown, then external_mcp_call. "
            "Run independent read-only evidence routes in parallel when useful; keep state "
            "changes ordered and skip unrelated tools."
'''
    source = replace_once(source, old, new, 'adaptive routing policy')
    compile(source, str(CAPS), 'exec')
    CAPS.write_text(source, encoding='utf-8')


def patch_production_tools() -> None:
    source = PROD.read_text(encoding='utf-8')
    source = replace_once(
        source,
        '        target = self._new_file(index_path)\n',
        '        target = self._replaceable_file(index_path)\n',
        'replaceable RAG index',
    )
    marker = '''    def _new_file(self, value: str) -> Path:
        path = self._resolve(value)
        if path.exists():
            raise FileExistsError(path)
        return path
'''
    replacement = '''    def _replaceable_file(self, value: str) -> Path:
        """Return a workspace-contained regular file path suitable for atomic rebuild."""

        path = self._resolve(value)
        if path.exists() and (path.is_symlink() or not path.is_file()):
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _new_file(self, value: str) -> Path:
        path = self._resolve(value)
        if path.exists():
            raise FileExistsError(path)
        return path
'''
    source = replace_once(source, marker, replacement, 'replaceable file helper')
    compile(source, str(PROD), 'exec')
    PROD.write_text(source, encoding='utf-8')


def _insert_custom_rag(source: str) -> str:
    anchor = '''        if self._cached_root == root and self._cached_index is not None:
            index = self._cached_index
        else:
            index = ProjectIndex(root, policy=self.policy)
            self._cached_root = root
            self._cached_index = index
'''
    replacement = anchor + '''
        # Bind MCP to the actual run and rebuild the live code RAG before every
        # model-backed module. The next module therefore sees all previously committed
        # source mutations instead of a stale setup-time index.
        self.router.bind_agent_workspace(root.parent, require_fresh_evidence=True)
        from .production_tools import ProductionToolService

        manifest = index.manifest_receipt()
        mapping_namespace = _mapping_namespace(mappings)
        ProductionToolService(
            workspace_root=root.parent,
            profile=self.router.profile,
        ).index_project_rag(
            [root.name],
            metadata={
                "minecraft_version": minecraft_version,
                "loader": loader,
                "mapping_namespace": mapping_namespace,
                "java_version": "17",
                "license": "project-local",
                "source_commit": str(manifest["sha256"]),
            },
            semantic=False,
        )
'''
    return replace_once(source, anchor, replacement, 'custom live RAG binding')


def patch_custom_generator() -> None:
    source = CUSTOM.read_text(encoding='utf-8')
    source = _insert_custom_rag(source)
    prompt_old = '''                            "is a code-owned commitment to earlier operations. When "
                            "code output for the current page is too large, set "
                            "context_page_complete=false and return a new next_cursor."
'''
    prompt_new = '''                            "is a code-owned commitment to earlier operations. Use the "
                            "live MCP/RAG tools during implementation: code RAG for exact "
                            "repository facts, project/API RAG and reviewed ecosystem tools "
                            "for version-pinned Minecraft/Fabric facts, and JDT symbols or "
                            "diagnostics when exact Java APIs are uncertain. Inspect RAG "
                            "receipts and reformulate/switch source when evidence is weak. "
                            "When code output for the current page is too large, set "
                            "context_page_complete=false and return a new next_cursor."
'''
    source = replace_once(source, prompt_old, prompt_new, 'custom adaptive prompt')

    old_start = source.find('            # Auto-Repair & Feedback Retry Loop (Up to 3 attempts for model self-correction)')
    old_end = source.find('            # Safe defaults if model omitted any expected fields after repair', old_start)
    if old_start < 0 or old_end < 0:
        raise SystemExit('custom repair loop anchors missing')
    loop = '''            # Progress-driven response repair: no arbitrary attempt ceiling. Stop only
            # when the model satisfies the executable contract or repeats the same
            # normalized validation failure without new progress.
            repair_attempts = 0
            repair_signatures: set[str] = set()
            payload: dict[str, Any] = {}
            while True:
                error_reason = ""
                try:
                    payload = _extract_json(text)
                    if "tests" in payload and "runtime_tests" not in payload:
                        payload["runtime_tests"] = payload["tests"]
                    if "cursor" in payload and "next_cursor" not in payload:
                        payload["next_cursor"] = payload["cursor"]
                    if "patch_operations" in payload and "operations" not in payload:
                        payload["operations"] = payload["patch_operations"]
                    if "patches" in payload and "operations" not in payload:
                        payload["operations"] = payload["patches"]

                    ops = payload.get("operations")
                    if (
                        ops is None
                        or not isinstance(ops, list)
                        or (len(ops) == 0 and is_last_page)
                    ):
                        if isinstance(payload, dict) and payload:
                            keys_str = ", ".join(payload.keys())
                            error_reason = (
                                "received object with keys "
                                f"[{keys_str}] but no non-empty 'operations' list "
                                "on final page"
                            )
                        else:
                            error_reason = "response did not contain a valid 'operations' list"
                    else:
                        if ops:
                            self._validate_operations(ops)
                        break
                except Exception as parse_err:
                    error_reason = str(parse_err)

                signature = _normalized_generation_failure(error_reason)
                if signature in repair_signatures:
                    raise CustomModuleGenerationError(
                        "Custom-module response repair stopped because the same "
                        "normalized validation failure repeated without progress: "
                        f"{error_reason}"
                    )
                repair_signatures.add(signature)
                repair_attempts += 1
                print(
                    "🔄 [CustomModule Auto-Repair] 검증 피드백 기반 재시도 "
                    f"({repair_attempts}) - 원인: {error_reason}",
                    flush=True,
                )
                repair_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an evidence-grounded Minecraft Fabric Java repair "
                            "agent. Use live RAG/MCP evidence when regenerating code; "
                            "return exactly one valid JSON object with operations."
                        ),
                    },
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            "Execution & Validation Failure: the previous response "
                            f"failed with reason: {error_reason}. Correct that exact "
                            "failure while preserving the approved module and evidence."
                        ),
                    },
                ]
                text = self.router.generate_text(
                    "coder",
                    repair_messages,
                    response_format="json",
                )

'''
    source = source[:old_start] + loop + source[old_end:]

    if 'def _mapping_namespace(' not in source:
        source += '''\n\ndef _mapping_namespace(value: str) -> str:\n    lowered = value.strip().lower()\n    if "intermediary" in lowered:\n        return "intermediary"\n    if "official" in lowered or "mojang" in lowered:\n        return "official"\n    return "yarn"\n\n\ndef _normalized_generation_failure(value: str) -> str:\n    compact = " ".join(value.lower().split())\n    compact = re.sub(r"0x[0-9a-f]+|[0-9]+", "#", compact)\n    return compact[:2048]\n'''
    compile(source, str(CUSTOM), 'exec')
    CUSTOM.write_text(source, encoding='utf-8')


def patch_repair_engine() -> None:
    source = REPAIR.read_text(encoding='utf-8')
    source = replace_once(source, 'import json\n', 'import json\nimport os\n', 'repair os import')
    bind_anchor = '''        project_index = ProjectIndex(root, policy=self.policy)
        index_token = _ACTIVE_REPAIR_PROJECT_INDEX.set((root, project_index))
'''
    bind_new = '''        project_index = ProjectIndex(root, policy=self.policy)
        self.router.bind_agent_workspace(root.parent, require_fresh_evidence=True)
        index_token = _ACTIVE_REPAIR_PROJECT_INDEX.set((root, project_index))
'''
    source = replace_once(source, bind_anchor, bind_new, 'repair workspace binding')
    request_anchor = '''                context = self._context(root, evidence)
                patch = self._request_patch(evidence, context)
'''
    request_new = '''                context = self._context(root, evidence)
                # Rebuild code RAG from the exact post-previous-patch project before
                # every repair model turn. Compiler/JDT feedback can then trigger a
                # fresh retrieve -> patch -> validate cycle rather than stale lookup.
                from .production_tools import ProductionToolService

                manifest = project_index.manifest_receipt()
                ProductionToolService(
                    workspace_root=root.parent,
                    profile=self.router.profile,
                ).index_project_rag(
                    [root.name],
                    metadata=_repair_rag_metadata(manifest),
                    semantic=False,
                )
                patch = self._request_patch(evidence, context)
'''
    source = replace_once(source, request_anchor, request_new, 'repair live RAG refresh')
    constraint_old = '''                "Use project-index paths; do not assume that omitted content means a file does not exist.",
'''
    constraint_new = '''                "Use project-index paths; do not assume that omitted content means a file does not exist.",
                "Use live code/project RAG and reviewed MCP evidence for unresolved APIs, symbols and version facts; inspect retrieval quality and reformulate weak searches.",
                "Treat JDT/Gradle/GameTest failures as new observations and retrieve again when they introduce new uncertainty.",
'''
    source = replace_once(source, constraint_old, constraint_new, 'repair adaptive constraints')
    if 'def _repair_rag_metadata(' not in source:
        source += '''\n\ndef _repair_rag_metadata(manifest: dict[str, Any]) -> dict[str, Any]:\n    mappings = os.environ.get("MMM_MAPPING_NAMESPACE", os.environ.get("MMM_MAPPINGS", "yarn")).strip().lower()\n    if "intermediary" in mappings:\n        namespace = "intermediary"\n    elif "official" in mappings or "mojang" in mappings:\n        namespace = "official"\n    else:\n        namespace = "yarn"\n    return {\n        "minecraft_version": os.environ.get("MMM_MINECRAFT_VERSION", "1.20.1").strip() or "1.20.1",\n        "loader": os.environ.get("MMM_LOADER", "fabric").strip() or "fabric",\n        "mapping_namespace": namespace,\n        "java_version": os.environ.get("MMM_JAVA_VERSION", "17").strip() or "17",\n        "license": os.environ.get("MMM_PROJECT_LICENSE", "project-local").strip() or "project-local",\n        "source_commit": str(manifest["sha256"]),\n    }\n'''
    compile(source, str(REPAIR), 'exec')
    REPAIR.write_text(source, encoding='utf-8')


def write_tests() -> None:
    TEST.write_text('''from __future__ import annotations\n\nimport json\nimport threading\nimport time\nfrom pathlib import Path\nfrom types import SimpleNamespace\n\nfrom minecraft_mod_ai.model_adapters import GenerationResponse, ToolCall\nfrom minecraft_mod_ai.model_router import ModelRouter\nfrom minecraft_mod_ai.production_tools import ProductionToolService\nfrom minecraft_mod_ai.skill_catalog import REVIEWED_TOOL_STAGES\n\n\nclass _Registry:\n    def __init__(self) -> None:\n        self.config = SimpleNamespace(adapter="llama_cpp", exclusive_gpu=False)\n\n    def load_profile(self, profile: str) -> None:\n        assert profile == "test"\n\n    def role(self, profile: str, role: str):\n        return self.config\n\n\ndef _schema(name: str):\n    return {\n        "type": "function",\n        "function": {\n            "name": name,\n            "description": name,\n            "parameters": {"type": "object", "properties": {}},\n        },\n    }\n\n\ndef test_bound_production_coder_cannot_finalize_before_fresh_rag(monkeypatch, tmp_path: Path) -> None:\n    class Runtime:\n        def tool_schemas(self, stage):\n            return (_schema("search_code_rag"),)\n\n        def call(self, stage, name, arguments):\n            return {\n                "hits": [{"path": "src/main/java/X.java"}],\n                "receipt": {\n                    "result_count": 1,\n                    "coverage_score": 1.0,\n                    "relevance_score": 1.0,\n                },\n            }\n\n    class Adapter:\n        def __init__(self):\n            self.count = 0\n\n        def generate_turn(self, request):\n            self.count += 1\n            if self.count == 1:\n                return GenerationResponse(content="premature")\n            if self.count == 2:\n                return GenerationResponse(tool_calls=(ToolCall(\n                    id="rag", name="search_code_rag",\n                    arguments={"query": "Registry.register"},\n                    raw_arguments='{"query":"Registry.register"}',\n                ),))\n            return GenerationResponse(content="evidence-backed final")\n\n    adapter = Adapter()\n    runtime = Runtime()\n    monkeypatch.setattr(ModelRouter, "_new_text_adapter", staticmethod(lambda config, *, role: adapter))\n    router = ModelRouter(profile="test", registry=_Registry(), agent_tool_runtime_factory=lambda **_: runtime)\n    router.bind_agent_workspace(tmp_path, require_fresh_evidence=True)\n    assert router.generate_text("coder", [{"role": "user", "content": "implement"}]) == "evidence-backed final"\n    assert adapter.count == 3\n\n\ndef test_independent_read_tools_execute_in_parallel(monkeypatch) -> None:\n    lock = threading.Lock()\n    active = 0\n    max_active = 0\n\n    class Runtime:\n        def tool_schemas(self, stage):\n            return (_schema("search_code_rag"), _schema("search_project_rag"))\n\n        def call(self, stage, name, arguments):\n            nonlocal active, max_active\n            with lock:\n                active += 1\n                max_active = max(max_active, active)\n            time.sleep(0.08)\n            with lock:\n                active -= 1\n            return {"evidence": name}\n\n    class Adapter:\n        def __init__(self):\n            self.count = 0\n\n        def generate_turn(self, request):\n            self.count += 1\n            if self.count == 1:\n                return GenerationResponse(tool_calls=(\n                    ToolCall(id="a", name="search_code_rag", arguments={"query":"a"}, raw_arguments='{"query":"a"}'),\n                    ToolCall(id="b", name="search_project_rag", arguments={"query":"b"}, raw_arguments='{"query":"b"}'),\n                ))\n            return GenerationResponse(content="done")\n\n    adapter = Adapter()\n    runtime = Runtime()\n    monkeypatch.setattr(ModelRouter, "_new_text_adapter", staticmethod(lambda config, *, role: adapter))\n    router = ModelRouter(profile="test", registry=_Registry(), agent_tool_runtime_factory=lambda **_: runtime)\n    assert router.generate_text("coder", [{"role":"user","content":"x"}]) == "done"\n    assert max_active == 2\n\n\ndef test_generation_stage_exposes_research_and_java_read_tools() -> None:\n    required = {\n        "search_project_rag", "search_code_rag", "discover_ecosystem_resources",\n        "inspect_github_repository", "inspect_modrinth_project",\n        "java_diagnostics", "java_workspace_symbols",\n    }\n    for name in required:\n        assert "generation" in REVIEWED_TOOL_STAGES[name]\n\n\ndef test_code_rag_index_can_be_refreshed_in_place(tmp_path: Path) -> None:\n    project = tmp_path / "project"\n    source = project / "src/main/java/example/X.java"\n    source.parent.mkdir(parents=True)\n    source.write_text("class X { int oldValue; }", encoding="utf-8")\n    service = ProductionToolService(workspace_root=tmp_path, profile="test")\n    metadata = {\n        "minecraft_version": "1.20.1",\n        "loader": "fabric",\n        "mapping_namespace": "yarn",\n        "java_version": "17",\n        "license": "project-local",\n        "source_commit": "first",\n    }\n    service.index_project_rag(["project"], metadata=metadata)\n    source.write_text("class X { int newValue; }", encoding="utf-8")\n    metadata["source_commit"] = "second"\n    service.index_project_rag(["project"], metadata=metadata)\n    result = service.search_code_rag("newValue")\n    assert result["hits"]\n\n\ndef test_reviewed_stage_map_matches_live_mcp_map(monkeypatch) -> None:\n    monkeypatch.setenv("MMM_MCP_STAGE", "all")\n    from minecraft_mod_ai import mcp_server\n    assert mcp_server._TOOL_STAGES == REVIEWED_TOOL_STAGES\n''', encoding='utf-8')
    compile(TEST.read_text(encoding='utf-8'), str(TEST), 'exec')


def main() -> None:
    patch_model_router()
    patch_runtime()
    expand_stage_map(MCP)
    expand_stage_map(SKILLS)
    patch_capability_context()
    patch_production_tools()
    patch_custom_generator()
    patch_repair_engine()
    write_tests()


if __name__ == '__main__':
    main()
