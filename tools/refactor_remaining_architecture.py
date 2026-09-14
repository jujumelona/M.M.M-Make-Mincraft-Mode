from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{relative}: expected exactly one replacement target, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_llama_adapter() -> None:
    relative = "minecraft_mod_ai/model_adapters/llama_cpp_adapter.py"
    old_generation = '''def _qwen_tool_generation_response(
    message: Mapping[str, Any],
    request: GenerationRequest,
) -> GenerationResponse:
    # Convert ToolDefinition objects to dict format
    tool_schemas = [
        tool.to_schema() if hasattr(tool, 'to_schema') else tool
        for tool in request.tools
    ]
    schemas = _tool_schema_map(tool_schemas)
    content_value = message.get("content")
    content_raw = content_value if isinstance(content_value, str) else ""
    reasoning_value = message.get("reasoning_content", message.get("reasoning"))
    server_reasoning = reasoning_value if isinstance(reasoning_value, str) else ""
    embedded_reasoning, content_raw = _split_qwen_reasoning_markup(content_raw)
    reasoning_raw = _merge_reasoning(server_reasoning, embedded_reasoning)

    reasoning, reasoning_calls = _parse_qwen_tool_markup(reasoning_raw, schemas)
    content, content_calls = _parse_qwen_tool_markup(content_raw, schemas)
    markup_calls = (*reasoning_calls, *content_calls)
    native_calls = _parse_native_tool_calls(message, schemas)
    if native_calls and markup_calls:
        raise ToolCallValidationError(
            "llama-server returned both structured tool_calls and raw Qwen tool markup"
        )
    calls = native_calls or markup_calls
    if len(calls) > 1 and not request.parallel_tool_calls:
        # Qwen/llama.cpp can ignore the OpenAI parallel-tool hint. Preserve the
        # host's serial execution contract by exposing only the first action now;
        # the next action must be regenerated after this tool result is observed.
        calls = calls[:1]
    _validate_tool_calls_against_host_schema(calls, schemas)
    _validate_tool_choice(request, calls)
    return GenerationResponse(
        content=content.strip(),
        tool_calls=tuple(calls),
        reasoning_content=reasoning.strip(),
    )
'''
    new_generation = '''def _request_tool_schema_map(request: GenerationRequest) -> dict[str, Mapping[str, Any]]:
    tool_schemas = [
        tool.to_schema() if hasattr(tool, "to_schema") else tool
        for tool in request.tools
    ]
    return _tool_schema_map(tool_schemas)


def _qwen_response_text(message: Mapping[str, Any]) -> tuple[str, str]:
    content_value = message.get("content")
    content_raw = content_value if isinstance(content_value, str) else ""
    reasoning_value = message.get("reasoning_content", message.get("reasoning"))
    server_reasoning = reasoning_value if isinstance(reasoning_value, str) else ""
    embedded_reasoning, content_raw = _split_qwen_reasoning_markup(content_raw)
    return _merge_reasoning(server_reasoning, embedded_reasoning), content_raw


def _select_qwen_tool_calls(
    message: Mapping[str, Any],
    markup_calls: Sequence[ToolCall],
    schemas: Mapping[str, Mapping[str, Any]],
    *,
    parallel_tool_calls: bool,
) -> tuple[ToolCall, ...]:
    native_calls = _parse_native_tool_calls(message, schemas)
    if native_calls and markup_calls:
        raise ToolCallValidationError(
            "llama-server returned both structured tool_calls and raw Qwen tool markup"
        )
    calls = tuple(native_calls or markup_calls)
    if len(calls) > 1 and not parallel_tool_calls:
        return calls[:1]
    return calls


def _qwen_tool_generation_response(
    message: Mapping[str, Any],
    request: GenerationRequest,
) -> GenerationResponse:
    schemas = _request_tool_schema_map(request)
    reasoning_raw, content_raw = _qwen_response_text(message)
    reasoning, reasoning_calls = _parse_qwen_tool_markup(reasoning_raw, schemas)
    content, content_calls = _parse_qwen_tool_markup(content_raw, schemas)
    calls = _select_qwen_tool_calls(
        message,
        (*reasoning_calls, *content_calls),
        schemas,
        parallel_tool_calls=request.parallel_tool_calls,
    )
    _validate_tool_calls_against_host_schema(calls, schemas)
    _validate_tool_choice(request, calls)
    return GenerationResponse(
        content=content.strip(),
        tool_calls=calls,
        reasoning_content=reasoning.strip(),
    )
'''
    replace_once(relative, old_generation, new_generation)

    old_choice = '''def _validate_tool_choice(request: GenerationRequest, calls: Sequence[ToolCall]) -> None:
    if not request.parallel_tool_calls and len(calls) > 1:
        raise RuntimeError("model emitted parallel tool calls when they are disabled")
    choice = request.tool_choice
    if choice is None or choice == "auto":
        return
    if choice == "none":
        if calls:
            raise RuntimeError("model emitted a tool call when tool_choice is none")
        return
    if choice == "required":
        if not calls:
            raise RuntimeError("model did not emit a tool call when one is required")
        return
    if isinstance(choice, str):
        # String tool_choice specifies a specific tool name
        expected = choice.strip()
        if len(calls) != 1 or calls[0].name != expected:
            received = ", ".join(call.name for call in calls) or "<none>"
            raise RuntimeError(
                f"model violated named tool_choice {expected!r}; received {received}"
            )
        return
    if isinstance(choice, Mapping):
        function = choice.get("function")
        if not isinstance(function, Mapping):
            raise TypeError("named tool_choice lacks function metadata")
        expected = str(function.get("name", "")).strip()
        if not expected:
            raise RuntimeError("named tool_choice lacks a function name")
        if len(calls) != 1 or calls[0].name != expected:
            received = ", ".join(call.name for call in calls) or "<none>"
            raise RuntimeError(
                f"model violated named tool_choice {expected!r}; received {received}"
            )
        return
    raise RuntimeError(f"unsupported tool_choice contract: {choice!r}")
'''
    new_choice = '''def _named_tool_choice(choice: Any) -> str:
    if isinstance(choice, str):
        return choice.strip()
    if not isinstance(choice, Mapping):
        raise RuntimeError(f"unsupported tool_choice contract: {choice!r}")
    function = choice.get("function")
    if not isinstance(function, Mapping):
        raise TypeError("named tool_choice lacks function metadata")
    expected = str(function.get("name", "")).strip()
    if not expected:
        raise RuntimeError("named tool_choice lacks a function name")
    return expected


def _validate_named_tool_choice(expected: str, calls: Sequence[ToolCall]) -> None:
    if len(calls) == 1 and calls[0].name == expected:
        return
    received = ", ".join(call.name for call in calls) or "<none>"
    raise RuntimeError(
        f"model violated named tool_choice {expected!r}; received {received}"
    )


def _validate_tool_choice(request: GenerationRequest, calls: Sequence[ToolCall]) -> None:
    if not request.parallel_tool_calls and len(calls) > 1:
        raise RuntimeError("model emitted parallel tool calls when they are disabled")
    choice = request.tool_choice
    if choice is None or choice == "auto":
        return
    if choice == "none":
        if calls:
            raise RuntimeError("model emitted a tool call when tool_choice is none")
        return
    if choice == "required":
        if not calls:
            raise RuntimeError("model did not emit a tool call when one is required")
        return
    _validate_named_tool_choice(_named_tool_choice(choice), calls)
'''
    replace_once(relative, old_choice, new_choice)


def patch_complete_orchestrator() -> None:
    relative = "minecraft_mod_ai/complete_orchestrator.py"
    class_marker = '''class CompleteProductionOrchestrator:
    """Approved request -> sharded source -> repair -> runtime -> release."""
'''
    helper = '''def _attested_repair_build(repair_result: Any) -> dict[str, Any] | None:
    if not isinstance(repair_result, dict) or repair_result.get("status") != "PASS":
        return None
    repair_evidence = repair_result.get("evidence")
    if not isinstance(repair_evidence, dict) or repair_evidence.get("passed") is not True:
        return None
    repaired_build = repair_evidence.get("build")
    if not isinstance(repaired_build, dict) or repaired_build.get("status") != "PASS":
        return None
    return repaired_build


class CompleteProductionOrchestrator:
    """Approved request -> sharded source -> repair -> runtime -> release."""
'''
    replace_once(relative, class_marker, helper)

    init_marker = '''        self.policy = policy or ScalePolicy.from_environment()
        self.policy.validate()

    @execution_scoped
'''
    method = '''        self.policy = policy or ScalePolicy.from_environment()
        self.policy.validate()

    def _run_build_with_repair(
        self,
        *,
        project_root: Path,
        cache: Path,
        options: CompleteExecutionOptions,
        router: ModelRouter | None,
    ) -> tuple[dict[str, Any], ModelRouter | None]:
        build_result = GradleRunner(cache).build(
            project_root, run_gametest=options.run_gametest
        ).to_dict()
        repair_result: dict[str, Any] | None = None
        active_router = router
        if build_result.get("status") != "PASS" and options.auto_repair:
            active_router = active_router or self.router_factory()
            repair_result = RepairEngine(
                router=active_router,
                gradle_cache=cache,
                policy=self.policy,
            ).repair(
                project_root,
                run_gametest=options.run_gametest,
                max_attempts=options.max_repair_attempts,
            )
            repaired_build = _attested_repair_build(repair_result)
            if repaired_build is not None:
                build_result = dict(repaired_build)
            else:
                build_result = GradleRunner(cache).build(
                    project_root, run_gametest=options.run_gametest
                ).to_dict()
        return {"build": build_result, "repair": repair_result}, active_router

    @execution_scoped
'''
    replace_once(relative, init_marker, method)

    old_build = '''        def build_with_repair() -> dict[str, Any]:
            nonlocal router
            build_result = GradleRunner(cache).build(project_root, run_gametest=options.run_gametest).to_dict()
            repair_result: dict[str, Any] | None = None
            if build_result.get('status') != 'PASS' and options.auto_repair:
                router = router or self.router_factory()
                repair_result = RepairEngine(router=router, gradle_cache=cache, policy=self.policy).repair(project_root, run_gametest=options.run_gametest, max_attempts=options.max_repair_attempts)
                repair_evidence = repair_result.get('evidence') if isinstance(repair_result, dict) else None
                repaired_build = repair_evidence.get('build') if isinstance(repair_evidence, dict) else None
                if (
                    repair_result.get('status') == 'PASS'
                    and isinstance(repair_evidence, dict)
                    and repair_evidence.get('passed') is True
                    and isinstance(repaired_build, dict)
                    and repaired_build.get('status') == 'PASS'
                ):
                    # RepairEngine already ran the final Gradle/GameTest validation on
                    # the repaired tree. Reuse that attested receipt instead of running
                    # the identical expensive build a third time.
                    build_result = dict(repaired_build)
                else:
                    # Preserve fail-closed compatibility for alternate repair engines
                    # that do not provide a validated build receipt.
                    build_result = GradleRunner(cache).build(project_root, run_gametest=options.run_gametest).to_dict()
            return {'build': build_result, 'repair': repair_result}
'''
    new_build = '''        def build_with_repair() -> dict[str, Any]:
            nonlocal router
            bundle, router = self._run_build_with_repair(
                project_root=project_root,
                cache=cache,
                options=options,
                router=router,
            )
            return bundle
'''
    replace_once(relative, old_build, new_build)


def main() -> None:
    patch_llama_adapter()
    patch_complete_orchestrator()
    for relative in (
        "minecraft_mod_ai/model_adapters/llama_cpp_adapter.py",
        "minecraft_mod_ai/complete_orchestrator.py",
    ):
        compile((ROOT / relative).read_text(encoding="utf-8"), relative, "exec")
    (ROOT / "tools/refactor_remaining_architecture.py").unlink()
    (ROOT / ".github/workflows/apply-architecture-refactor.yml").unlink()


if __name__ == "__main__":
    main()
