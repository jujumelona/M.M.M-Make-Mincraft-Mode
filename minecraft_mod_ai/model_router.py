from __future__ import annotations
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from .model_adapters import EmbeddingAdapter, GenerationRequest, ImageDiffusionAdapter, ModelConfigurationError, OpenAICompatibleAdapter, RerankerAdapter, TransformersMultimodalAdapter, TransformersTextAdapter
from .model_registry import ModelRegistry
_GPU_EXCLUSIVE_LOCK = threading.RLock()
_ROLE_TOOL_STAGE = {'planner': 'planning', 'researcher': 'research', 'coder': 'generation', 'coder_safe': 'quality', 'visual_critic': 'quality'}
_NATIVE_TOOL_ADAPTERS = frozenset({'llama_cpp', 'vllm', 'openai_compatible'})
_RAG_EVIDENCE_TOOLS = frozenset({'search_code_rag', 'search_project_rag'})
_EXTERNAL_RAG_CAPABILITIES = frozenset({'mapping_resolution', 'mod_examples', 'mod_jar_analysis', 'official_mod_docs', 'registry_lookup', 'source_search', 'vanilla_knowledge', 'version_diff'})
_PARALLEL_READ_TOOLS = frozenset({'search_code_rag', 'search_project_rag', 'discover_ecosystem_resources', 'inspect_modrinth_project', 'inspect_github_repository', 'inspect_huggingface_model', 'inspect_existing_mod', 'assess_technology_compatibility', 'java_diagnostics', 'java_workspace_symbols', 'read_complete_plan_section', 'read_quality_contract', 'quality_status', 'work_status', 'work_tasks', 'external_mcp_capabilities', 'external_mcp_schema'})
_DEFAULT_AGENT_TOOL_ROUNDS = 12
_MIN_AGENT_TOOL_ROUNDS = 1
_MAX_AGENT_TOOL_ROUNDS = 64

class ModelRouter:
    """Role router with strict profile selection and no silent backend fallback."""

    def __init__(self, *, profile: str='t4_local', registry: ModelRegistry | None=None, agent_tool_runtime_factory: Callable[..., Any] | None=None) -> None:
        self.registry = registry or ModelRegistry()
        self.profile = profile
        self.registry.load_profile(profile)
        self._generation_lock = threading.RLock()
        self._active_generation_role: str | None = None
        self._active_generation_adapter: Any | None = None
        self._agent_tool_runtime_factory = agent_tool_runtime_factory
        self._agent_tool_runtime: Any | None = None
        self._agent_workspace_root: Path | None = None
        self._agent_require_fresh_evidence = False

    def bind_agent_workspace(self, workspace_root: str | Path, *, require_fresh_evidence: bool=False) -> 'ModelRouter':
        """Bind model-callable MCP/RAG tools to the actual production workspace."""
        root = Path(workspace_root).expanduser().resolve()
        if not root.is_dir() or root.is_symlink():
            raise ModelConfigurationError(f'Agent workspace must be a regular directory: {root}')
        with self._generation_lock:
            if root != self._agent_workspace_root:
                self._agent_workspace_root = root
                self._agent_tool_runtime = None
            self._agent_require_fresh_evidence = bool(require_fresh_evidence)
        return self

    @contextmanager
    def generation_session(self, role: str):
        """Keep one text-generation backend alive for a bounded workflow.

        Only one role can be pinned on a router at a time. This avoids an
        unbounded multi-model VRAM cache while allowing a paginated planner to
        reuse the same processor and weights until its complete plan succeeds or
        raises. Direct ``generate_text`` calls outside this context retain their
        existing load-generate-release lifetime.
        """
        config = self.registry.role(self.profile, role)
        adapter = self._new_text_adapter(config, role=role)
        with self._generation_lock:
            if self._active_generation_adapter is not None:
                raise ModelConfigurationError(f'A generation session is already active for role {self._active_generation_role!r}.')
            self._active_generation_role = role
            self._active_generation_adapter = adapter
            session_factory = getattr(adapter, 'generation_session', None)
            try:
                if callable(session_factory):
                    with session_factory():
                        yield self
                else:
                    try:
                        yield self
                    finally:
                        adapter.close()
            finally:
                self._active_generation_adapter = None
                self._active_generation_role = None

    def generate_text(self, role: str, messages: Sequence[Mapping[str, Any]], *, media_paths: Sequence[str | Path]=(), response_format: str='text', response_schema: Mapping[str, Any] | None=None, tool_stage: str | None=None, enable_tools: bool=True) -> str:
        with self._generation_lock:
            config = self.registry.role(self.profile, role)
            if self._active_generation_adapter is not None:
                if role != self._active_generation_role:
                    raise ModelConfigurationError(f'Generation session for role {self._active_generation_role!r} cannot serve role {role!r}.')
                adapter = self._active_generation_adapter
            else:
                adapter = self._new_text_adapter(config, role=role)
            stage, runtime, tools, request = self._prepare_generation_request(role, messages, config=config, media_paths=media_paths, response_format=response_format, response_schema=response_schema, tool_stage=tool_stage, enable_tools=enable_tools)
            with self._gpu_scope(config.exclusive_gpu):
                if runtime is not None and tools:
                    return self._generate_with_tools(adapter=adapter, request=request, runtime=runtime, stage=stage, role=role)
                return adapter.generate(request)

    def generate_tool_decision(self, role: str, messages: Sequence[Mapping[str, Any]], *, tool_name: str, parameters: Mapping[str, Any], description: str='') -> dict[str, Any]:
        """Return one host-validated native function call instead of free-form JSON.

        This is the small-model structured-decision path. It deliberately bypasses
        the general agent tool runtime: the function is a return channel, not an
        executable side effect. Qwen/llama.cpp receive one forced native tool and
        the host accepts only that tool's decoded argument object. Assistant text is
        never reparsed as JSON.
        """
        name = str(tool_name or '').strip()
        if not name:
            raise ModelConfigurationError('Tool-decision name must not be empty.')
        schema = {
            'type': 'function',
            'function': {
                'name': name,
                'description': str(description or '').strip(),
                'parameters': dict(parameters),
            },
        }
        with self._generation_lock:
            config = self.registry.role(self.profile, role)
            if config.adapter not in _NATIVE_TOOL_ADAPTERS:
                raise ModelConfigurationError(
                    f'Role {role!r} adapter {config.adapter!r} does not support native tool decisions.'
                )
            if self._active_generation_adapter is not None:
                if role != self._active_generation_role:
                    raise ModelConfigurationError(
                        f'Generation session for role {self._active_generation_role!r} cannot serve role {role!r}.'
                    )
                adapter = self._active_generation_adapter
            else:
                adapter = self._new_text_adapter(config, role=role)
            base_messages = tuple(dict(message) for message in messages)
            with self._gpu_scope(config.exclusive_gpu):
                for attempt in range(2):
                    request_messages = base_messages
                    if attempt:
                        request_messages = (*base_messages, {
                            'role': 'system',
                            'content': f'Call the required function {name} exactly once. Do not answer in prose.',
                        })
                    request = GenerationRequest(
                        messages=request_messages,
                        media_paths=(),
                        response_format='text',
                        response_schema=None,
                        tools=(schema,),
                        tool_choice={'type': 'function', 'function': {'name': name}},
                        parallel_tool_calls=False,
                    )
                    turn = adapter.generate_turn(request)
                    matches = tuple(call for call in turn.tool_calls if call.name == name)
                    if len(matches) == 1 and len(turn.tool_calls) == 1:
                        return dict(matches[0].arguments)
        raise ModelConfigurationError(
            f'Native structured decision did not return exactly one {name!r} tool call after bounded retry.'
        )

    def _prepare_generation_request(self, role: str, messages: Sequence[Mapping[str, Any]], *, config: Any, media_paths: Sequence[str | Path]=(), response_format: str='text', response_schema: Mapping[str, Any] | None=None, tool_stage: str | None=None, enable_tools: bool=True) -> tuple[str, Any | None, tuple[Mapping[str, Any], ...], GenerationRequest]:
        """Build the canonical model request used by every text execution policy.

        Locking/concurrency contracts may decide *when* generation runs, but tool
        exposure, role filtering, Skill/MCP context and structured-output semantics
        are prepared here once so late runtime wrappers cannot drift from the router.
        """
        stage = (tool_stage or _ROLE_TOOL_STAGE.get(role, '')).strip().lower()
        runtime = None
        tools: tuple[Mapping[str, Any], ...] = ()
        request_messages: Sequence[Mapping[str, Any]] = messages
        if self._tools_enabled(enable_tools=enable_tools, stage=stage, adapter_name=config.adapter):
            runtime = self._tool_runtime()
            raw_tools = tuple(runtime.tool_schemas(stage))
            if raw_tools:
                from .agent_capability_context import build_agent_capability_context, filter_tool_schemas_for_role
                tools = filter_tool_schemas_for_role(stage, role, raw_tools)
                if tools:
                    request_messages = _inject_system_context(messages, build_agent_capability_context(stage, tools, model_role=role))
        request = GenerationRequest(messages=request_messages, media_paths=tuple((Path(path) for path in media_paths)), response_format=response_format, response_schema=response_schema, tools=tools, tool_choice='auto' if tools else None, parallel_tool_calls=True)
        return (stage, runtime, tools, request)

    def _generate_with_tools(self, *, adapter: Any, request: GenerationRequest, runtime: Any, stage: str, role: str) -> str:
        """Run bounded retrieve/act/observe production until semantic convergence."""
        from .agent_capability_context import reviewed_mcp_servers_for_model_role, skills_for_tool
        messages: list[dict[str, Any]] = [dict(message) for message in request.messages]
        exposed_tools = frozenset(_tool_schema_names(request.tools))
        previous_exchange_state: str | None = None
        weak_fixed_point_seen = False
        premature_final_state: str | None = None
        rag_evidence_seen = False
        round_index = 0
        round_limit = _agent_tool_round_limit()
        require_rag = bool(self._agent_require_fresh_evidence and role in {'coder', 'coder_safe'} and exposed_tools & _RAG_EVIDENCE_TOOLS)
        reviewed_external_servers = reviewed_mcp_servers_for_model_role(stage, role)
        while True:
            if round_index >= round_limit:
                if require_rag and not rag_evidence_seen:
                    raise ModelConfigurationError(
                        f'Agent tool budget exhausted after {round_limit} rounds without usable fresh RAG evidence.'
                    )
                final_messages = [*messages, {
                    'role': 'system',
                    'content': (
                        f'The host tool budget is exhausted after {round_limit} rounds. '
                        'Do not call more tools. Return the final answer using only observations already present.'
                    ),
                }]
                final_request = GenerationRequest(messages=final_messages, media_paths=(), response_format=request.response_format, response_schema=request.response_schema, tools=(), tool_choice=None, parallel_tool_calls=False)
                final_turn = adapter.generate_turn(final_request)
                if final_turn.tool_calls:
                    raise ModelConfigurationError('Agent emitted tool calls after the host disabled tools at the hard round budget.')
                final_content = final_turn.content.strip()
                if not final_content:
                    raise ModelConfigurationError('Agent returned an empty final response at the hard tool-round budget.')
                return final_content

            turn_request = GenerationRequest(messages=messages, media_paths=request.media_paths if round_index == 0 else (), response_format=request.response_format, response_schema=request.response_schema, tools=request.tools, tool_choice=request.tool_choice, parallel_tool_calls=request.parallel_tool_calls)
            turn = adapter.generate_turn(turn_request)
            if not turn.tool_calls:
                content = turn.content.strip()
                if not content:
                    raise ModelConfigurationError('Tool-capable model returned an empty final response.')
                if require_rag and (not rag_evidence_seen):
                    state = hashlib.sha256(content.encode('utf-8')).hexdigest()
                    if state == premature_final_state:
                        raise ModelConfigurationError('Production coder repeated a final answer without gathering fresh RAG evidence.')
                    premature_final_state = state
                    messages.extend([{'role': 'assistant', 'content': content}, {'role': 'system', 'content': 'This production coding turn requires fresh evidence before finalization. Use search_code_rag and/or search_project_rag. Inspect the retrieval receipt. If result_count/coverage/relevance is weak or empty, change the query or reviewed evidence source. Do not guess exact Minecraft/Fabric/mapping/dependency/Java API facts from memory.'}])
                    round_index += 1
                    continue
                return content
            messages.append({'role': 'assistant', 'content': turn.content or None, 'tool_calls': [{'id': call.id, 'type': 'function', 'function': {'name': call.name, 'arguments': call.raw_arguments or json.dumps(dict(call.arguments), ensure_ascii=False, separators=(',', ':'))}} for call in turn.tool_calls]})

            def execute(call: Any) -> tuple[Any, Mapping[str, Any]]:
                route_metadata: dict[str, Any] = {'skills': list(skills_for_tool(stage, call.name, model_role=role))}
                if call.name == 'external_mcp_call':
                    capability = str(call.arguments.get('capability', '')).strip()
                    if capability:
                        route_metadata['external_mcp_capability'] = capability
                try:
                    if call.name not in exposed_tools:
                        raise ModelConfigurationError(f'Agent attempted hidden tool {call.name!r} outside its reviewed role routes for {role!r}/{stage!r}.')
                    scoped_call = getattr(runtime, 'call_scoped', None)
                    if callable(scoped_call):
                        result = scoped_call(stage, call.name, call.arguments, external_server_ids=reviewed_external_servers)
                    elif call.name.startswith('external_mcp_'):
                        raise ModelConfigurationError('External MCP execution requires a role-scoped agent runtime.')
                    else:
                        result = runtime.call(stage, call.name, call.arguments)
                    payload: Mapping[str, Any] = {'ok': True, 'tool': call.name, **route_metadata, 'result': result}
                except Exception as exc:
                    payload = {'ok': False, 'tool': call.name, **route_metadata, 'error': f'{type(exc).__name__}: {exc}'}
                return (call, payload)
            calls = tuple(turn.tool_calls)
            executed = _execute_tool_waves(calls, execute)
            observations: list[dict[str, Any]] = []
            weak_rag_in_round = False
            for call, payload in executed:
                messages.append({'role': 'tool', 'tool_call_id': call.id, 'name': call.name, 'content': json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)})
                observations.append({'name': call.name, 'arguments': dict(call.arguments), 'observation': payload})
                if not bool(payload.get('ok')):
                    continue
                if call.name in _RAG_EVIDENCE_TOOLS:
                    usable_rag = _usable_rag_result(payload.get('result'))
                elif call.name == 'external_mcp_call' and _external_rag_capability(call.arguments):
                    usable_rag = _usable_external_rag_result(call.arguments, payload.get('result'))
                else:
                    continue
                if usable_rag:
                    rag_evidence_seen = True
                else:
                    weak_rag_in_round = True
            if require_rag and weak_rag_in_round and (not rag_evidence_seen):
                messages.append({'role': 'system', 'content': 'The latest RAG observation is not usable fresh evidence. Use its receipt/correction fields to reformulate the query, or switch between current code RAG and reviewed exact-version project/API evidence. Do not finalize and do not repeat the identical weak retrieval.'})
            exchange_state = hashlib.sha256(json.dumps({'assistant_content': turn.content or '', 'tool_exchanges': observations}, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')).hexdigest()
            if exchange_state == previous_exchange_state:
                if require_rag and (not rag_evidence_seen):
                    if weak_fixed_point_seen:
                        raise ModelConfigurationError('Production RAG converged without usable fresh evidence after a corrective retrieval instruction.')
                    weak_fixed_point_seen = True
                    previous_exchange_state = None
                    messages.append({'role': 'system', 'content': 'An identical weak retrieval repeated. Change the query substantively or use a different reviewed evidence source before attempting a final production patch.'})
                    round_index += 1
                    continue
                final_messages = [*messages, {'role': 'system', 'content': 'Tool use has converged after usable evidence was gathered. Do not call more tools. Return the final answer using only the evidence already present. Preserve the requested response format and do not mention this convergence instruction.'}]
                final_request = GenerationRequest(messages=final_messages, media_paths=(), response_format=request.response_format, response_schema=request.response_schema, tools=(), tool_choice=None, parallel_tool_calls=False)
                final_turn = adapter.generate_turn(final_request)
                if final_turn.tool_calls:
                    raise ModelConfigurationError('Agent emitted tool calls after tools were disabled at an exact no-progress fixed point.')
                final_content = final_turn.content.strip()
                if not final_content:
                    raise ModelConfigurationError('Agent returned an empty final response after exact tool fixed-point convergence.')
                return final_content
            previous_exchange_state = exchange_state
            round_index += 1

    def _tool_runtime(self) -> Any:
        runtime = self._agent_tool_runtime
        if runtime is not None:
            return runtime
        with self._generation_lock:
            runtime = self._agent_tool_runtime
            if runtime is not None:
                return runtime
            if self._agent_tool_runtime_factory is not None:
                runtime = self._agent_tool_runtime_factory(profile=self.profile)
            else:
                from .agent_tool_runtime import AgentToolRuntime
                runtime = AgentToolRuntime(profile=self.profile, workspace_root=self._agent_workspace_root)
            self._agent_tool_runtime = runtime
            return runtime

    @staticmethod
    def _tools_enabled(*, enable_tools: bool, stage: str, adapter_name: str) -> bool:
        if not enable_tools or not stage:
            return False
        if adapter_name not in _NATIVE_TOOL_ADAPTERS:
            return False
        if os.environ.get('MMM_AGENT_TOOL_CHILD', '').strip() == '1':
            return False
        raw = os.environ.get('MMM_AGENT_TOOLS', '1').strip().lower()
        return raw not in {'0', 'false', 'no', 'off'}

    @staticmethod
    def _new_text_adapter(config, *, role: str):
        if config.adapter == 'transformers_text':
            return TransformersTextAdapter(config)
        if config.adapter == 'transformers_multimodal':
            return TransformersMultimodalAdapter(config)
        if config.adapter in ('llama_cpp', 'vllm'):
            from .model_adapters.llama_cpp_adapter import LlamaCppAdapter
            return LlamaCppAdapter(config)
        if config.adapter == 'openai_compatible':
            return OpenAICompatibleAdapter(config)
        raise ModelConfigurationError(f'Role {role!r} cannot generate text with adapter {config.adapter!r}.')

    def embed(self, texts: Sequence[str], role: str='embedding') -> list[list[float]]:
        config = self.registry.role(self.profile, role)
        if config.adapter != 'embedding':
            raise ModelConfigurationError(f'Role {role!r} does not expose an embedding adapter.')
        return EmbeddingAdapter(config).embed(texts)

    def rerank(self, query: str, documents: Sequence[str], *, role: str='reranker', instruction: str='Retrieve the Minecraft modding evidence that directly answers the query for the caller-selected platform target. Do not prefer or infer a different Minecraft version or mapping namespace.') -> list[float]:
        config = self.registry.role(self.profile, role)
        if config.adapter != 'reranker':
            raise ModelConfigurationError(f'Role {role!r} does not expose a reranker adapter.')
        return RerankerAdapter(config).score(query, documents, instruction=instruction)

    def generate_image(self, role: str, *, prompt: str, output_path: str | Path, width: int=512, height: int=512, seed: int=0) -> Path:
        config = self.registry.role(self.profile, role)
        if config.adapter == 'image_diffusion':
            adapter = ImageDiffusionAdapter(config)
        elif config.adapter == 'openai_compatible':
            adapter = OpenAICompatibleAdapter(config)
        else:
            raise ModelConfigurationError(f'Role {role!r} cannot generate images with adapter {config.adapter!r}.')
        with self._gpu_scope(config.exclusive_gpu):
            return adapter.generate_image(prompt=prompt, output_path=Path(output_path), width=width, height=height, seed=seed)

    def transcribe(self, role: str) -> str:
        config = self.registry.role(self.profile, role)
        if config.adapter == 'openai_compatible':
            adapter = OpenAICompatibleAdapter(config)
        else:
            raise ModelConfigurationError()
        with self._gpu_scope(config.exclusive_gpu):
            return adapter.transcribe(Path(None))

    @staticmethod
    @contextmanager
    def _gpu_scope(exclusive: bool):
        if exclusive:
            with _GPU_EXCLUSIVE_LOCK:
                yield
        else:
            yield

def _agent_tool_round_limit() -> int:
    raw = os.environ.get('MMM_AGENT_TOOL_ROUNDS', str(_DEFAULT_AGENT_TOOL_ROUNDS)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = _DEFAULT_AGENT_TOOL_ROUNDS
    return max(_MIN_AGENT_TOOL_ROUNDS, min(value, _MAX_AGENT_TOOL_ROUNDS))

def _parallel_read_workers() -> int:
    raw = os.environ.get('MMM_AGENT_PARALLEL_READS', '4').strip()
    try:
        value = int(raw)
    except ValueError:
        value = 4
    return max(1, min(value, 16))

def _execute_tool_waves(calls: Sequence[Any], execute: Callable[[Any], tuple[Any, Mapping[str, Any]]]) -> tuple[tuple[Any, Mapping[str, Any]], ...]:
    """Execute maximal read waves concurrently while preserving serial barriers.

    Unclassified or side-effectful tools remain ordered barriers. Independent read
    calls on either side of those barriers still overlap instead of forcing the whole
    model-emitted batch through the previous all-or-nothing serial fallback.
    """
    completed: list[tuple[Any, Mapping[str, Any]]] = []
    pending_reads: list[Any] = []

    def flush_reads() -> None:
        if not pending_reads:
            return
        batch = tuple(pending_reads)
        pending_reads.clear()
        workers = min(len(batch), _parallel_read_workers())
        if workers <= 1:
            completed.extend((execute(call) for call in batch))
            return
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='mmm_agent_read_wave') as executor:
            completed.extend(executor.map(execute, batch))
    for call in calls:
        if call.name in _PARALLEL_READ_TOOLS:
            pending_reads.append(call)
            continue
        flush_reads()
        completed.append(execute(call))
    flush_reads()
    return tuple(completed)

def _external_rag_capability(arguments: Mapping[str, Any]) -> str:
    capability = str(arguments.get('capability', '')).strip()
    return capability if capability in _EXTERNAL_RAG_CAPABILITIES else ''

def _external_mcp_result_has_content(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    for key in ('structured', 'parsed_text'):
        item = value.get(key)
        if item not in (None, '', [], {}):
            return True
    texts = value.get('text')
    if isinstance(texts, Sequence) and not isinstance(texts, (str, bytes)) and any(str(item).strip() for item in texts):
        return True
    other = value.get('other_content')
    return isinstance(other, Sequence) and not isinstance(other, (str, bytes)) and bool(other)

def _usable_external_rag_result(arguments: Mapping[str, Any], value: Any) -> bool:
    """Accept only reviewed external retrieval receipts with real provider content."""
    capability = _external_rag_capability(arguments)
    if not capability or not isinstance(value, Mapping):
        return False
    if str(value.get('schema_version', '')).strip() != 'mmm/external-mcp-evidence-bundle-v1':
        return False
    if str(value.get('capability', '')).strip() != capability or str(value.get('status', '')).strip() != 'PASS':
        return False
    evidence = value.get('evidence')
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)) or not evidence:
        return False
    for receipt in evidence:
        if not isinstance(receipt, Mapping):
            continue
        if str(receipt.get('schema_version', '')).strip() != 'mmm/external-mcp-call-receipt-v1':
            continue
        if str(receipt.get('capability', '')).strip() != capability or str(receipt.get('status', '')).strip() != 'PASS':
            continue
        if str(receipt.get('access', '')).strip() != 'read':
            continue
        if _external_mcp_result_has_content(receipt.get('result')):
            return True
    return False

def _usable_rag_result(value: Any) -> bool:
    """Treat RAG receipts as authoritative and accept other non-empty evidence packs."""
    found_receipt = False
    usable_receipt = False
    found_hits = False

    def visit(item: Any) -> None:
        nonlocal found_receipt, usable_receipt, found_hits
        if isinstance(item, Mapping):
            receipt = item.get('receipt')
            if isinstance(receipt, Mapping):
                found_receipt = True
                try:
                    if int(receipt.get('result_count', 0) or 0) > 0 and float(receipt.get('coverage_score', 0.0) or 0.0) > 0.0 and (float(receipt.get('relevance_score', 0.0) or 0.0) > 0.0):
                        usable_receipt = True
                except (TypeError, ValueError):
                    pass
            hits = item.get('hits')
            if isinstance(hits, Sequence) and (not isinstance(hits, (str, bytes))) and hits:
                found_hits = True
            for child in item.values():
                visit(child)
        elif isinstance(item, Sequence) and (not isinstance(item, (str, bytes))):
            for child in item:
                visit(child)
    visit(value)
    if found_receipt:
        return usable_receipt
    if found_hits:
        return True
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
        return bool(value)
    return False

def _inject_system_context(messages: Sequence[Mapping[str, Any]], content: str) -> tuple[dict[str, Any], ...]:
    copied = [dict(message) for message in messages]
    insert_at = 0
    while insert_at < len(copied) and copied[insert_at].get('role') == 'system':
        insert_at += 1
    copied.insert(insert_at, {'role': 'system', 'content': content})
    return tuple(copied)

def _tool_schema_names(tool_schemas: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    names: set[str] = set()
    for schema in tool_schemas:
        function = schema.get('function')
        if not isinstance(function, Mapping):
            continue
        name = str(function.get('name', '')).strip()
        if name:
            names.add(name)
    return tuple(sorted(names))

def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, '').strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default