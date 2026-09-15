from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import minecraft_mod_ai.authored_design_mutation_scope as scope_contract
import minecraft_mod_ai.direct_task_mutation_authority_contract as direct_authority
from minecraft_mod_ai.authored_design_mutation_scope import (
    AuthoredMutationScope,
    _canonical_scoped_path,
    _resolve_authored_scope,
    _scoped_mutation_error,
)


def _messages(*, task_id: str = "task-1", phase: str = "implement_authored_design"):
    payload = {
        "phase": phase,
        "workspace_project_root": ".",
        "module": {
            "module_id": task_id,
            "authored_plan": {"requested_prompt": "build a complete space mod"},
        },
    }
    return ({"role": "user", "content": json.dumps(payload)},)


def test_authored_scope_requires_matching_host_authority():
    authority = SimpleNamespace(task_id="task-1")
    scope = _resolve_authored_scope(
        _messages(), authority=authority, stage="generation", role="coder"
    )
    assert scope == AuthoredMutationScope(task_id="task-1")

    assert _resolve_authored_scope(
        _messages(), authority=None, stage="generation", role="coder"
    ) is None
    assert _resolve_authored_scope(
        _messages(), authority=SimpleNamespace(task_id="other"), stage="generation", role="coder"
    ) is None


def test_ordinary_implementation_phase_does_not_receive_broad_scope():
    scope = _resolve_authored_scope(
        _messages(phase="implement_module"),
        authority=SimpleNamespace(task_id="task-1"),
        stage="generation",
        role="coder",
    )
    assert scope is None


@pytest.mark.parametrize(
    "path",
    [
        "src/main/java/example/GalacticFrontierMod.java",
        "src/main/resources/assets/galactic_frontier/lang/en_us.json",
        "src/test/java/example/GalacticFrontierTest.java",
        "src/gametest/example/GalacticFrontierGameTest.java",
    ],
)
def test_authored_scope_allows_only_generated_source_roots(path: str):
    assert _canonical_scoped_path(path) == path


@pytest.mark.parametrize(
    "path",
    [
        "build.gradle",
        ".mmm/state.json",
        "../outside.java",
        "/tmp/absolute.java",
        "C:/outside.java",
        "src/main/java/../resources/escape.json",
        "src/main/java",
        "src/main/javaevil/NotAllowed.java",
    ],
)
def test_authored_scope_rejects_paths_outside_bounded_roots(path: str):
    assert _canonical_scoped_path(path) == ""


def test_original_galactic_frontier_create_is_authorized():
    scope = AuthoredMutationScope(task_id="task-1")
    error = _scoped_mutation_error(
        "apply_source_edit",
        {
            "operation": "create_file",
            "path": "src/main/java/ai/minecraft/generated/authored_7bf498f5bbba/GalacticFrontierMod.java",
            "content": "package ai.minecraft.generated.authored_7bf498f5bbba;",
        },
        path_keys=("path", "file_path", "target_path"),
        scope=scope,
    )
    assert error is None


def test_authored_scope_rejects_delete_even_inside_safe_root():
    scope = AuthoredMutationScope(task_id="task-1")
    error = _scoped_mutation_error(
        "apply_source_edit",
        {"operation": "delete_file", "path": "src/main/java/example/Old.java"},
        path_keys=("path",),
        scope=scope,
    )
    assert error is not None
    assert error.startswith("WRITE_SCOPE_DELETE_FORBIDDEN")


def test_authored_scope_reports_outside_write_scope():
    scope = AuthoredMutationScope(task_id="task-1")
    error = _scoped_mutation_error(
        "apply_source_edit",
        {"operation": "create_file", "path": "build.gradle"},
        path_keys=("path",),
        scope=scope,
    )
    assert error is not None
    assert error.startswith("PATH_OUTSIDE_WRITABLE_SET")


@dataclass(frozen=True)
class _FakeRequest:
    messages: tuple[dict[str, str], ...]
    parallel_tool_calls: bool = False


def _installed_fake_loop():
    loop = SimpleNamespace()
    seen: dict[str, object] = {}

    def exact_target_error(tool_name, arguments, context):
        return "ORIGINAL_EXACT_AUTHORITY"

    def original_turn(*args, **kwargs):
        seen["parallel"] = kwargs["parallel_tool_calls"]
        seen["turn_request_parallel"] = kwargs["request"].parallel_tool_calls
        return SimpleNamespace()

    def original_generate(
        router,
        *,
        config,
        adapter,
        request,
        runtime,
        stage,
        role,
    ):
        seen["mutation_error"] = loop._mutation_target_error(
            "apply_source_edit",
            {
                "operation": "create_file",
                "path": (
                    "src/main/java/ai/minecraft/generated/"
                    "authored_7bf498f5bbba/GalacticFrontierMod.java"
                ),
            },
            SimpleNamespace(),
        )
        loop._generate_turn_with_context_recovery(
            router,
            config=config,
            adapter=adapter,
            request=request,
            messages=list(request.messages),
            media_paths=(),
            tool_choice={"type": "function", "function": {"name": "apply_source_edit"}},
            parallel_tool_calls=False,
        )
        seen["messages"] = request.messages
        return "ok"

    loop.generate_with_tools = original_generate
    loop._mutation_target_error = exact_target_error
    loop._generate_turn_with_context_recovery = original_turn
    loop._SOURCE_EDIT_PATH_KEYS = ("path", "file_path", "target_path")
    scope_contract.install(loop)
    return loop, seen


def test_installed_contract_broadens_only_host_proven_authored_run():
    loop, seen = _installed_fake_loop()
    token = direct_authority._CURRENT_AUTHORITY.set(SimpleNamespace(task_id="task-1"))
    try:
        result = loop.generate_with_tools(
            object(),
            config=object(),
            adapter=object(),
            request=_FakeRequest(messages=_messages()),
            runtime=object(),
            stage="generation",
            role="coder",
        )
    finally:
        direct_authority._CURRENT_AUTHORITY.reset(token)

    assert result == "ok"
    assert seen["mutation_error"] is None
    assert seen["parallel"] is True
    assert seen["turn_request_parallel"] is True
    assert any(
        message.get("role") == "developer"
        and "Host-authored design write scope is active" in message.get("content", "")
        for message in seen["messages"]
    )


def test_installed_contract_keeps_ordinary_exact_authority_and_single_tool_mode():
    loop, seen = _installed_fake_loop()
    token = direct_authority._CURRENT_AUTHORITY.set(SimpleNamespace(task_id="task-1"))
    try:
        result = loop.generate_with_tools(
            object(),
            config=object(),
            adapter=object(),
            request=_FakeRequest(messages=_messages(phase="implement_module")),
            runtime=object(),
            stage="generation",
            role="coder",
        )
    finally:
        direct_authority._CURRENT_AUTHORITY.reset(token)

    assert result == "ok"
    assert seen["mutation_error"] == "ORIGINAL_EXACT_AUTHORITY"
    assert seen["parallel"] is False
    assert seen["turn_request_parallel"] is False
