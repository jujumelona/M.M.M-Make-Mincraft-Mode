from __future__ import annotations

from dataclasses import dataclass

from minecraft_mod_ai import generation_evidence_controller as controller


@dataclass(frozen=True)
class _Call:
    id: str
    name: str
    arguments: dict
    raw_arguments: str


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    }


def test_fresh_semantics_do_not_depend_on_scaffold_existence() -> None:
    assert controller.semantic_fresh_java(
        "fresh",
        "src/main/java/dev/mmm/Foo.java",
        materialized_new_file=False,
    )
    assert not controller.semantic_fresh_java(
        "reuse",
        "src/main/java/dev/mmm/Foo.java",
        materialized_new_file=True,
    )


def test_fresh_frontier_is_host_owned_and_single_route() -> None:
    selected = controller.initial_evidence_frontier(
        available={"search_project_rag", "search_code_rag", "java_workspace_symbols"},
        attempted=set(),
        localization_stage="READY",
        semantic_fresh_java_target=True,
    )
    assert selected == ("search_code_rag",)
    selected = controller.initial_evidence_frontier(
        available={"search_project_rag", "search_code_rag", "java_workspace_symbols"},
        attempted={"search_code_rag"},
        localization_stage="READY",
        semantic_fresh_java_target=True,
    )
    assert selected == ("search_project_rag",)


def test_repair_router_classifies_platform_api_failure() -> None:
    route = controller.repair_evidence_route_for_errors(
        (
            {
                "path": "src/main/java/dev/mmm/Foo.java",
                "message": (
                    "package net.minecraft.registry does not exist\n"
                    "import net.minecraft.registry.Registry;"
                ),
            },
        )
    )
    assert route["route"] == "official_api"
    assert controller.repair_route_requires_retrieval(route["route"])


def test_rejected_alternate_retriever_is_rebound_not_consumed() -> None:
    rejected = _Call(
        id="r1",
        name="__mmm_rejected_tool_call__",
        arguments={
            "failure_code": "TOOL_NOT_VISIBLE",
            "original_tool": "search_project_rag",
            "raw_arguments": '{"query":"block registration 26.2","limit":8}',
        },
        raw_arguments="{}",
    )
    normalized = controller.normalize_forced_evidence_rejection_calls(
        (rejected,),
        phase_tools=(_schema("search_code_rag"),),
        forced_evidence_tool="search_code_rag",
    )
    assert normalized is not None
    assert normalized[0].name == "search_code_rag"
    assert normalized[0].arguments == {
        "query": "block registration 26.2",
        "limit": 8,
    }


def test_low_quality_code_rag_is_not_authoritative_java_evidence() -> None:
    value = {
        "schema_version": "mmm/code-rag-result-v1",
        "retrieval_quality_warning": "coverage_or_relevance_below_target",
        "hits": [
            {
                "source_path": "src/main/java/dev/mmm/Other.java",
                "text": "import net.minecraft.world.item.Item; class Other {}",
            }
        ],
    }
    assert not controller.authoritative_java_evidence(value)


def test_current_target_placeholder_cannot_self_authorize_fresh_java() -> None:
    target = "src/main/java/dev/mmm/Foo.java"
    value = {
        "schema_version": "mmm/code-rag-result-v1",
        "hits": [
            {
                "source_path": target,
                "text": "package dev.mmm; public final class Foo {}",
            }
        ],
    }
    assert not controller.authoritative_java_evidence(value, target_path=target)


def test_evidence_obligation_admission_is_controller_owned() -> None:
    assert controller.evidence_obligation_satisfied(
        require_evidence=False,
        semantic_fresh_java_target=True,
        has_fresh_evidence=False,
        has_authoritative_java_evidence=False,
    )
    assert not controller.evidence_obligation_satisfied(
        require_evidence=True,
        semantic_fresh_java_target=True,
        has_fresh_evidence=True,
        has_authoritative_java_evidence=False,
    )
    assert controller.evidence_obligation_satisfied(
        require_evidence=True,
        semantic_fresh_java_target=True,
        has_fresh_evidence=True,
        has_authoritative_java_evidence=True,
    )
    assert controller.evidence_obligation_satisfied(
        require_evidence=True,
        semantic_fresh_java_target=False,
        has_fresh_evidence=True,
        has_authoritative_java_evidence=False,
    )


def test_compile_backed_existing_java_skips_speculative_presearch() -> None:
    assert not controller.initial_evidence_required(
        role="coder",
        host_grounded=False,
        router_requires_fresh_evidence=True,
        implementation_requires_mutation=True,
        host_target_execution_authority=True,
        compile_backed_java=True,
    )


def test_unverified_fresh_java_still_requires_initial_evidence() -> None:
    assert controller.initial_evidence_required(
        role="coder",
        host_grounded=False,
        router_requires_fresh_evidence=True,
        implementation_requires_mutation=True,
        host_target_execution_authority=True,
        compile_backed_java=False,
    )


def test_compile_backed_host_authored_scaffold_skips_speculative_presearch() -> None:
    options = {
        "role": "coder",
        "host_grounded": False,
        "router_requires_fresh_evidence": True,
        "implementation_requires_mutation": True,
        "host_target_execution_authority": True,
        "compile_backed_java": True,
        "semantic_fresh_java_target": True,
        "host_authored_scaffold": True,
    }
    assert not controller.initial_evidence_required(**options)


def test_materialized_fresh_java_honors_explicit_evidence_policy() -> None:
    options = {
        "role": "coder", "host_grounded": False, "router_requires_fresh_evidence": True,
        "implementation_requires_mutation": True, "host_target_execution_authority": True,
        "compile_backed_java": True, "semantic_fresh_java_target": True,
    }
    assert controller.initial_evidence_required(**options)
    assert not controller.initial_evidence_required(**{**options, "host_grounded": True})
    assert not controller.initial_evidence_required(
        **{**options, "router_requires_fresh_evidence": False}
    )


def test_official_api_recovery_does_not_route_through_modrinth_or_jdt() -> None:
    available = {
        "search_project_rag",
        "search_code_rag",
        "java_workspace_symbols",
        "inspect_modrinth_project",
        "external_mcp_capabilities",
        "external_mcp_schema",
        "external_mcp_call",
    }
    selected = controller.recovery_evidence_frontier(
        available=available,
        attempted=set(),
        route="official_api",
    )
    assert selected == ("search_project_rag",)
    selected = controller.recovery_evidence_frontier(
        available=available,
        attempted={"search_project_rag"},
        route="official_api",
    )
    assert selected == ("search_code_rag",)

def test_verifier_recovery_query_prefers_failed_api_over_local_target() -> None:
    errors = (
        {
            "path": "src/main/java/demo/AuthoredFeature002.java",
            "message": (
                "cannot find symbol\n"
                "import net.fabricmc.fabric.api.event.lifecycle.v1.ClientTickEvents;\n"
                "symbol: class ClientTickEvents\n"
                "location: package net.fabricmc.fabric.api.event.lifecycle.v1"
            ),
        },
    )
    query = controller.verifier_recovery_query(
        errors,
        target_path="src/main/java/demo/AuthoredFeature002.java",
    )
    assert "net.fabricmc.fabric.api.event.lifecycle.v1.ClientTickEvents" in query
    assert "ClientTickEvents" in query
    assert "AuthoredFeature002" not in query


def test_recovery_call_query_is_host_bound_to_verifier_diagnostic() -> None:
    call = _Call(
        id="r1",
        name="search_code_rag",
        arguments={"query": "AuthoredFeature002 generated local class"},
        raw_arguments='{"query":"AuthoredFeature002 generated local class"}',
    )
    normalized = controller.normalize_recovery_evidence_calls(
        (call,),
        errors=(
            {
                "message": (
                    "package net.minecraft.registry does not exist\n"
                    "import net.minecraft.registry.Registry;"
                ),
            },
        ),
        target_path="src/main/java/demo/AuthoredFeature002.java",
        repair_route="official_api",
    )
    assert normalized is not None
    query = normalized[0].arguments["query"]
    assert "net.minecraft.registry.Registry" in query
    assert "AuthoredFeature002" not in query


def test_external_source_search_query_is_bound_to_verifier_diagnostic() -> None:
    call = _Call(
        id="r1",
        name="external_mcp_call",
        arguments={
            "capability": "source_search",
            "arguments": {
                "query": "class AuthoredFeature001",
                "searchType": "class",
                "version": "26.2",
                "mapping": "mojmap",
            },
        },
        raw_arguments=(
            '{"capability":"source_search","arguments":'
            '{"query":"class AuthoredFeature001","searchType":"class",'
            '"version":"26.2","mapping":"mojmap"}}'
        ),
    )
    normalized = controller.normalize_recovery_evidence_calls(
        (call,),
        errors=(
            {
                "message": (
                    "cannot find symbol\n"
                    "import net.minecraft.client.MinecraftClient;\n"
                    "symbol: class MinecraftClient"
                ),
            },
        ),
        target_path="src/main/java/demo/AuthoredFeature001.java",
        repair_route="official_api",
    )
    assert normalized is not None
    query = normalized[0].arguments["arguments"]["query"]
    assert query == "net.minecraft.client.MinecraftClient"
    assert "AuthoredFeature001" not in query

def test_malformed_external_source_search_is_rebuilt_from_verifier_diagnostic() -> None:
    call = _Call(
        id="r2",
        name="external_mcp_call",
        arguments={
            "capability": "source_search",
            "arguments": {
                "action": "read",
                "path": "src/main/java/demo/AuthoredFeature001.java",
            },
        },
        raw_arguments=(
            '{"capability":"source_search","arguments":'
            '{"action":"read","path":"src/main/java/demo/AuthoredFeature001.java"}}'
        ),
    )
    normalized = controller.normalize_recovery_evidence_calls(
        (call,),
        errors=(
            {
                "message": (
                    "cannot find symbol\n"
                    "import net.minecraft.client.MinecraftClient;\n"
                    "symbol: class MinecraftClient"
                ),
            },
        ),
        target_path="src/main/java/demo/AuthoredFeature001.java",
        repair_route="official_api",
    )
    assert normalized is not None
    nested = normalized[0].arguments["arguments"]
    assert nested == {
        "query": "net.minecraft.client.MinecraftClient",
        "searchType": "class",
    }


def test_rejected_source_search_is_translated_to_external_mcp_call() -> None:
    rejected_call = _Call(
        id="call_rejected_1",
        name="__mmm_rejected_tool_call__",
        arguments={
            "original_tool": "source_search",
            "raw_arguments": '{"query":"BlockEntity","searchType":"class"}',
            "failure_code": "TOOL_NOT_VISIBLE",
            "error": "model emitted non-visible tool 'source_search'",
        },
        raw_arguments='{"original_tool":"source_search"}',
    )
    phase_tool = {
        "type": "function",
        "function": {
            "name": "external_mcp_call",
            "parameters": {
                "type": "object",
                "properties": {
                    "capability": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["capability", "arguments"],
            },
        },
    }
    normalized = controller.normalize_forced_evidence_rejection_calls(
        (rejected_call,),
        phase_tools=(phase_tool,),
        forced_evidence_tool="external_mcp_call",
    )
    assert normalized is not None
    assert len(normalized) == 1
    assert normalized[0].name == "external_mcp_call"
    assert normalized[0].arguments["capability"] == "source_search"
    assert normalized[0].arguments["arguments"] == {"query": "BlockEntity", "searchType": "class"}



def test_rejected_host_selected_external_capability_is_rebound() -> None:
    rejected = _Call(
        id="r-external",
        name="__mmm_rejected_tool_call__",
        arguments={
            "failure_code": "TOOL_SCHEMA_INVALID",
            "original_tool": "external_mcp_schema",
            "raw_arguments": '{"capability":"minecraft_api_docs"}',
        },
        raw_arguments="{}",
    )
    phase_tool = {
        "type": "function",
        "function": {
            "name": "external_mcp_schema",
            "parameters": {
                "type": "object",
                "properties": {
                    "capability": {
                        "type": "string",
                        "enum": ["source_search"],
                    }
                },
                "required": ["capability"],
            },
        },
    }
    normalized = controller.normalize_forced_evidence_rejection_calls(
        (rejected,),
        phase_tools=(phase_tool,),
        forced_evidence_tool="external_mcp_schema",
    )
    assert normalized is not None
    assert normalized[0].name == "external_mcp_schema"
    assert normalized[0].arguments == {"capability": "source_search"}


def test_cross_tool_external_rebind_uses_host_selected_capability() -> None:
    rejected = _Call(
        id="cross-tool",
        name="__mmm_rejected_tool_call__",
        arguments={
            "failure_code": "TOOL_NOT_VISIBLE",
            "original_tool": "external_mcp_call",
            "raw_arguments": (
                '{"capability":"source_search","arguments":'
                '{"query":"AuthoredFeature002.java","searchType":"file"}}'
            ),
        },
        raw_arguments="{}",
    )
    phase_tool = {
        "type": "function",
        "function": {
            "name": "external_mcp_schema",
            "parameters": {
                "type": "object",
                "properties": {
                    "capability": {
                        "type": "string",
                        "enum": ["registry_lookup"],
                    }
                },
                "required": ["capability"],
            },
        },
    }
    normalized = controller.normalize_forced_evidence_rejection_calls(
        (rejected,),
        phase_tools=(phase_tool,),
        forced_evidence_tool="external_mcp_schema",
    )
    assert normalized is not None
    assert normalized[0].name == "external_mcp_schema"
    assert normalized[0].arguments == {"capability": "registry_lookup"}


def test_authoritative_evidence_diagnostic_explains_self_target_rejection() -> None:
    target = "src/main/java/dev/mmm/AuthoredFeature002.java"
    diagnostic = controller.authoritative_java_evidence_diagnostic(
        {
            "schema_version": "mmm/code-rag-result-v1",
            "hits": [
                {
                    "source_path": target,
                    "text": "package dev.mmm; public final class AuthoredFeature002 {}",
                }
            ],
        },
        target_path=target,
    )
    assert diagnostic["accepted"] is False
    assert diagnostic["reason"] == "CODE_RAG_API_HIT_MISSING"
    assert diagnostic["api_hit"] is False


def test_nonvisible_read_file_is_rebound_to_host_owned_mcp_capabilities() -> None:
    rejected = _Call(
        id="read-file-capabilities",
        name="__mmm_rejected_tool_call__",
        arguments={
            "failure_code": "TOOL_NOT_VISIBLE",
            "original_tool": "read_file",
            "raw_arguments": '{"path":"src/main/java/demo/AuthoredFeature001.java"}',
        },
        raw_arguments="{}",
    )
    phase_tool = {
        "type": "function",
        "function": {
            "name": "external_mcp_capabilities",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }
    normalized = controller.normalize_forced_evidence_rejection_calls(
        (rejected,),
        phase_tools=(phase_tool,),
        forced_evidence_tool="external_mcp_capabilities",
    )
    assert normalized is not None
    assert normalized[0].name == "external_mcp_capabilities"
    assert normalized[0].arguments == {}


def test_nonvisible_read_file_is_rebound_to_host_owned_mcp_schema() -> None:
    rejected = _Call(
        id="read-file-schema",
        name="__mmm_rejected_tool_call__",
        arguments={
            "failure_code": "TOOL_NOT_VISIBLE",
            "original_tool": "read_file",
            "raw_arguments": '{"path":"src/main/java/demo/AuthoredFeature001.java"}',
        },
        raw_arguments="{}",
    )
    phase_tool = {
        "type": "function",
        "function": {
            "name": "external_mcp_schema",
            "parameters": {
                "type": "object",
                "properties": {
                    "capability": {
                        "type": "string",
                        "enum": ["source_search"],
                    }
                },
                "required": ["capability"],
                "additionalProperties": False,
            },
        },
    }
    normalized = controller.normalize_forced_evidence_rejection_calls(
        (rejected,),
        phase_tools=(phase_tool,),
        forced_evidence_tool="external_mcp_schema",
    )
    assert normalized is not None
    assert normalized[0].name == "external_mcp_schema"
    assert normalized[0].arguments == {"capability": "source_search"}
