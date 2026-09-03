from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import agent_tool_runtime
from minecraft_mod_ai import custom_module_generator as generator_module
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.custom_module_generator import (
    _CHECKPOINT_SCHEMA,
    CustomModuleGenerationError,
    CustomModuleGenerator,
    _checkpoint_router_scope,
    _generation_checkpoint_identity,
    _persist_generation_checkpoint,
    _prepare_generation_checkpoint,
    _remove_generation_checkpoint,
)
from minecraft_mod_ai.llama_finish_reason_contract import (
    CONTEXT_PRESSURE,
    OUTPUT_EXHAUSTED,
    LlamaCompletionBoundaryError,
)
from minecraft_mod_ai.model_adapters.base import (
    ModelBackendError,
    ModelConfigurationError,
)
from minecraft_mod_ai.platform_catalog import adapter_for_target
from minecraft_mod_ai.scale_policy import ScalePolicy


def _implement_request(messages) -> dict:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("phase") == "implement_module":
            return payload
    raise AssertionError("No implement_module request was found in the coder message history.")


class _AgenticRouter:
    def __init__(self) -> None:
        self.workspace: Path | None = None
        self.calls: list[dict] = []
        self.messages: list[list[dict[str, str]]] = []

    def bind_agent_workspace(self, workspace_root, **_kwargs):
        self.workspace = Path(workspace_root)
        return self

    def generate_tool_decision(self, *_args, **_kwargs):
        raise AssertionError("custom-module production must not use a file-plan structured return channel")

    def generate_text(self, role, messages, **kwargs):
        assert role == "coder"
        assert kwargs["response_format"] == "text"
        assert kwargs["tool_stage"] == "generation"
        assert kwargs["enable_tools"] is True
        assert self.workspace is not None
        self.calls.append(dict(kwargs))
        self.messages.append([dict(message) for message in messages])
        request = _implement_request(messages)
        assert request["phase"] == "implement_module"
        project = self.workspace / request["workspace_project_root"]
        target = project / "src/main/java/example/Generated.java"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("package example; final class Generated {}\n", encoding="utf-8")
        return "Implemented the approved module."


def test_custom_module_uses_coding_agent_tool_loop_not_file_plan(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    router = _AgenticRouter()
    target = adapter_for_target("1.20.1", "fabric")

    result = CustomModuleGenerator(
        router,
        policy=ScalePolicy(model_context_bytes=4096),
    ).generate(
        root,
        module=ProductionModule("agentic_custom", "custom_java", {"feature": "shape"}),
        minecraft_version=target.minecraft_version,
        loader=target.loader,
        mappings=target.yarn_mappings,
    )

    assert result["status"] == "SOURCE_GENERATED"
    assert result["generation_checkpoint_acknowledged"] is True
    assert result["generation_checkpoint"]["status"] == "CLEANED_AFTER_LIVE_COMMIT"
    assert "cleanup_token" not in result["generation_checkpoint"]
    assert "path" not in result["generation_checkpoint"]
    assert len(router.calls) == 1
    assert (root / "src/main/java/example/Generated.java").is_file()
    assert result["touched_paths"] == ["src/main/java/example/Generated.java"]
    request = _implement_request(router.messages[0])
    assert request["task"].startswith("Implement the approved Minecraft/Fabric mod feature")
    assert any("workspace/RAG/MCP tools" in rule for rule in request["rules"])
    assert all("return_custom_module_file_plan" not in str(message.get("content", "")) for message in router.messages[0])


def test_checkpoint_base_is_hidden_from_model_bound_workspace(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "src/main/java").mkdir(parents=True)
    (root / "settings.gradle").write_text(
        'rootProject.name = "project"\n',
        encoding="utf-8",
    )

    class _WorkspaceIsolationRouter(_AgenticRouter):
        def generate_text(self, role, messages, **kwargs):
            assert self.workspace is not None
            discovered, project_argument = agent_tool_runtime._discover_model_project_root(
                self.workspace
            )
            assert discovered == self.workspace.resolve()
            assert project_argument == "."
            assert (self.workspace.parent / "base").is_dir()
            assert not (self.workspace / "base").exists()
            request = _implement_request(messages)
            assert request["workspace_project_root"] == "."
            return super().generate_text(role, messages, **kwargs)

    platform = adapter_for_target("1.20.1", "fabric")
    result = CustomModuleGenerator(_WorkspaceIsolationRouter()).generate(
        root,
        module=ProductionModule(
            "isolated_checkpoint_workspace",
            "custom_java",
            {"feature": "hide host checkpoint base"},
        ),
        minecraft_version=platform.minecraft_version,
        loader=platform.loader,
        mappings=platform.yarn_mappings,
    )

    assert result["status"] == "SOURCE_GENERATED"
    assert (root / "src/main/java/example/Generated.java").is_file()


def test_checkpoint_cleanup_failure_revokes_token_and_releases_lease(
    tmp_path: Path,
    monkeypatch,
) -> None:
    generator = CustomModuleGenerator(_AgenticRouter())
    token = "a" * 64
    identity = "sha256:" + ("b" * 64)

    class _Lease:
        closed = False

        def close(self) -> None:
            self.closed = True

    lease = _Lease()
    generator._checkpoint_cleanup_tokens[token] = (
        identity,
        tmp_path / ".mmm-custom-checkpoints" / ("b" * 64),
        lease,
    )
    result = {
        "generation_checkpoint": {
            "schema_version": _CHECKPOINT_SCHEMA,
            "status": "AWAITING_LIVE_COMMIT",
            "identity_sha256": identity,
            "cleanup_token": token,
        }
    }
    monkeypatch.setattr(
        generator_module,
        "_remove_generation_checkpoint",
        lambda _path: (_ for _ in ()).throw(OSError("busy")),
    )

    assert generator.acknowledge_generation_checkpoint(result) is False
    assert lease.closed is True
    assert token not in generator._checkpoint_cleanup_tokens
    assert result["generation_checkpoint"]["status"] == (
        "PRESERVED_AFTER_CLEANUP_FAILURE"
    )
    assert "cleanup_token" not in result["generation_checkpoint"]


def test_out_of_scope_agent_edit_is_discarded_without_touching_real_project(tmp_path: Path) -> None:
    root = tmp_path / "project"
    wrapper = root / "gradle/wrapper/gradle-wrapper.properties"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("distributionUrl=original\n", encoding="utf-8")

    class _MixedRouter(_AgenticRouter):
        def generate_text(self, role, messages, **kwargs):
            summary = super().generate_text(role, messages, **kwargs)
            request = _implement_request(messages)
            assert self.workspace is not None
            project = self.workspace / request["workspace_project_root"]
            staged_wrapper = project / "gradle/wrapper/gradle-wrapper.properties"
            staged_wrapper.write_text("distributionUrl=wrong\n", encoding="utf-8")
            return summary

    router = _MixedRouter()
    target = adapter_for_target("1.20.1", "fabric")
    result = CustomModuleGenerator(router, policy=ScalePolicy(model_context_bytes=4096)).generate(
        root,
        module=ProductionModule("safe_scope", "custom_java", {"feature": "shape"}),
        minecraft_version=target.minecraft_version,
        loader=target.loader,
        mappings=target.yarn_mappings,
    )

    assert wrapper.read_text(encoding="utf-8") == "distributionUrl=original\n"
    assert "gradle/wrapper/gradle-wrapper.properties" in result["discarded_out_of_scope_paths"]
    assert (root / "src/main/java/example/Generated.java").is_file()


def test_exhausted_causal_resync_checkpoints_staged_edit_without_touching_real_project(
    tmp_path: Path,
) -> None:
    """Lock the exact Colab failure's workspace-impact boundary.

    The model may already have used the staged edit tool before it repeats a stale
    action. The real project remains untouched, while the hash-bound host checkpoint
    preserves work for an exact-input resume instead of replaying the whole node.
    """

    root = tmp_path / "project"
    root.mkdir()

    class _CausalResyncFailureRouter(_AgenticRouter):
        def generate_text(self, role, messages, **kwargs):
            super().generate_text(role, messages, **kwargs)
            raise ModelConfigurationError(
                "Model failed the single causal-frontier re-synchronization attempt; "
                "forced='search_project_rag' rejected=apply_source_edit "
                "visible=search_project_rag,java_workspace_symbols,search_code_rag"
            )

    router = _CausalResyncFailureRouter()
    target = adapter_for_target("1.20.1", "fabric")
    with pytest.raises(ModelConfigurationError, match="causal-frontier"):
        CustomModuleGenerator(
            router,
            policy=ScalePolicy(model_context_bytes=4096),
        ).generate(
            root,
            module=ProductionModule(
                "causal_retry_guard",
                "custom_java",
                {"feature": "shape"},
            ),
            minecraft_version=target.minecraft_version,
            loader=target.loader,
            mappings=target.yarn_mappings,
        )

    assert not (root / "src/main/java/example/Generated.java").exists()
    assert router.workspace is not None
    assert router.workspace.exists()
    assert (router.workspace / "src/main/java/example/Generated.java").is_file()
    manifest = json.loads(
        (router.workspace.parent / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == _CHECKPOINT_SCHEMA
    assert "Generated.java" not in json.dumps(manifest)


def _output_exhausted_error(
    *,
    partial_message: dict | None = None,
    kind: str = OUTPUT_EXHAUSTED,
) -> ModelBackendError:
    boundary = LlamaCompletionBoundaryError(
        "bounded coder action exhausted",
        kind=kind,
        partial_message=partial_message,
    )
    return ModelBackendError(role="coder", model_id="test", cause=boundary)


def test_output_exhaustion_continues_from_same_staged_workspace_until_success(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()

    class _ChunkedRouter(_AgenticRouter):
        def __init__(self) -> None:
            super().__init__()
            self.bound_workspaces: list[Path] = []

        def bind_agent_workspace(self, workspace_root, **kwargs):
            self.bound_workspaces.append(Path(workspace_root))
            return super().bind_agent_workspace(workspace_root, **kwargs)

        def generate_text(self, role, messages, **kwargs):
            assert role == "coder"
            assert self.workspace is not None
            self.calls.append(dict(kwargs))
            self.messages.append([dict(message) for message in messages])
            request = _implement_request(messages)
            project = self.workspace / request["workspace_project_root"]
            target = project / "src/main/java/example/Chunked.java"
            target.parent.mkdir(parents=True, exist_ok=True)
            call_index = len(self.calls)
            previous = target.read_text(encoding="utf-8") if target.exists() else ""
            target.write_text(previous + f"// chunk {call_index}\n", encoding="utf-8")
            if call_index <= 3:
                raise _output_exhausted_error()
            return "Implemented through bounded continuation."

    router = _ChunkedRouter()
    target = adapter_for_target("1.20.1", "fabric")
    result = CustomModuleGenerator(
        router,
        policy=ScalePolicy(model_context_bytes=4096),
    ).generate(
        root,
        module=ProductionModule(
            "chunked_output",
            "custom_java",
            {"feature": "large bounded source"},
        ),
        minecraft_version=target.minecraft_version,
        loader=target.loader,
        mappings=target.yarn_mappings,
    )

    assert result["output_exhaustion_continuations"] == 3
    assert len(router.calls) == 4
    assert len(set(router.bound_workspaces)) == 1
    assert (root / "src/main/java/example/Chunked.java").read_text(
        encoding="utf-8"
    ) == "// chunk 1\n// chunk 2\n// chunk 3\n// chunk 4\n"
    continuation = _implement_request(router.messages[1])
    assert continuation["continuation"]["continuation_index"] == 1
    assert continuation["continuation"]["preserved_path_count"] == 1
    assert continuation["initial_exact_source_context"]["global_anchors"] == []
    assert continuation["initial_exact_source_context"]["page_observations"] == []
    assert (
        continuation["initial_exact_source_context"]["ledger_receipt"]
        == continuation["source_observation_receipt"]
    )
    assert any("bounded tool actions" in rule for rule in continuation["rules"])


def test_context_pressure_raises_directly_without_masking(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()

    class _ContextPressureRouter(_AgenticRouter):
        def generate_text(self, role, messages, **kwargs):
            self.calls.append(dict(kwargs))
            raise _output_exhausted_error(kind=CONTEXT_PRESSURE)

    router = _ContextPressureRouter()
    platform = adapter_for_target("1.20.1", "fabric")
    with pytest.raises(ModelBackendError):
        CustomModuleGenerator(
            router,
            policy=ScalePolicy(model_context_bytes=4096),
        ).generate(
            root,
            module=ProductionModule(
                "context_pressure_unmasked",
                "custom_java",
                {"feature": "no tool-disabled side channel"},
            ),
            minecraft_version=platform.minecraft_version,
            loader=platform.loader,
            mappings=platform.yarn_mappings,
        )
    assert len(router.calls) == 1


def test_checkpoint_router_scope_binds_candidate_count_and_strategy_epoch() -> None:
    width_two = SimpleNamespace(
        _candidate_index=0,
        _strategy="api_contract_first",
        _count=2,
        _router=None,
    )
    width_three = SimpleNamespace(
        _candidate_index=0,
        _strategy="api_contract_first",
        _count=3,
        _router=None,
    )
    scope_two = _checkpoint_router_scope(width_two)
    scope_three = _checkpoint_router_scope(width_three)
    assert scope_two["candidate_count"] == 2
    assert scope_three["candidate_count"] == 3
    assert scope_two["strategy_epoch"] == scope_three["strategy_epoch"]
    assert scope_two != scope_three


def test_checkpoint_identity_ignores_random_clone_path_but_not_candidate_width(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "random-a/project"
    second_root = tmp_path / "random-b/project"
    for root in (first_root, second_root):
        source = root / "src/main/java/example/Base.java"
        source.parent.mkdir(parents=True)
        source.write_text("final class Base {}\n", encoding="utf-8")

    def identity(_root: Path, count: int) -> str:
        return _generation_checkpoint_identity(
            module_query='{"module_id":"same"}',
            minecraft_version="1.20.1",
            loader="fabric",
            mappings="1.20.1+build.10",
            research_context={"receipt": "same"},
            router=SimpleNamespace(
                _candidate_index=0,
                _strategy="api_contract_first",
                _count=count,
                _router=None,
            ),
        )

    assert identity(first_root, 2) == identity(second_root, 2)
    assert identity(first_root, 2) != identity(second_root, 3)


def test_identical_checkpoint_cannot_be_opened_concurrently(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    configured = tmp_path / "run/.minecraft_ai/.mmm-custom-checkpoints"
    identity = "sha256:" + ("1" * 64)
    checkpoint, _staged, resumed, lease = _prepare_generation_checkpoint(
        root,
        identity_sha256=identity,
        configured_root=configured,
    )
    assert resumed is False
    try:
        with pytest.raises(CustomModuleGenerationError, match="already active"):
            _prepare_generation_checkpoint(
                root,
                identity_sha256=identity,
                configured_root=configured,
            )
    finally:
        lease.close()

    checkpoint_again, _staged_again, resumed_again, lease_again = (
        _prepare_generation_checkpoint(
            root,
            identity_sha256=identity,
            configured_root=configured,
        )
    )
    assert checkpoint_again == checkpoint
    assert resumed_again is True
    lease_again.close()
    _remove_generation_checkpoint(checkpoint_again)


def test_checkpoint_rebases_saved_edit_over_unrelated_live_change(tmp_path: Path) -> None:
    root = tmp_path / "project"
    source_a = root / "src/main/java/example/A.java"
    source_a.parent.mkdir(parents=True)
    source_a.write_text("final class A { int value = 1; }\n", encoding="utf-8")
    configured = tmp_path / "run/.minecraft_ai/.mmm-custom-checkpoints"
    identity = "sha256:" + ("3" * 64)

    checkpoint, staged, resumed, lease = _prepare_generation_checkpoint(
        root,
        identity_sha256=identity,
        configured_root=configured,
    )
    assert resumed is False
    staged_a = staged / "src/main/java/example/A.java"
    staged_a.write_text("final class A { int value = 2; }\n", encoding="utf-8")
    _persist_generation_checkpoint(
        checkpoint,
        staged,
        identity_sha256=identity,
    )
    lease.close()

    source_b = root / "src/main/java/example/B.java"
    source_b.write_text("final class B {}\n", encoding="utf-8")
    checkpoint_again, staged_again, resumed_again, lease_again = (
        _prepare_generation_checkpoint(
            root,
            identity_sha256=identity,
            configured_root=configured,
        )
    )
    try:
        assert checkpoint_again == checkpoint
        assert resumed_again is True
        assert (staged_again / "src/main/java/example/A.java").read_text(
            encoding="utf-8"
        ) == "final class A { int value = 2; }\n"
        assert (staged_again / "src/main/java/example/B.java").read_text(
            encoding="utf-8"
        ) == "final class B {}\n"
    finally:
        lease_again.close()
        _remove_generation_checkpoint(checkpoint_again)


def test_checkpoint_same_file_conflict_reinitializes_instead_of_poisoning(tmp_path: Path) -> None:
    root = tmp_path / "project"
    source = root / "src/main/java/example/Conflict.java"
    source.parent.mkdir(parents=True)
    source.write_text("final class Conflict { int value = 1; }\n", encoding="utf-8")
    configured = tmp_path / "run/.minecraft_ai/.mmm-custom-checkpoints"
    identity = "sha256:" + ("4" * 64)

    checkpoint, staged, _resumed, lease = _prepare_generation_checkpoint(
        root,
        identity_sha256=identity,
        configured_root=configured,
    )
    (staged / "src/main/java/example/Conflict.java").write_text(
        "final class Conflict { int value = 2; }\n",
        encoding="utf-8",
    )
    _persist_generation_checkpoint(
        checkpoint,
        staged,
        identity_sha256=identity,
    )
    lease.close()
    source.write_text("final class Conflict { int value = 3; }\n", encoding="utf-8")

    checkpoint_again, staged_again, resumed_again, lease_again = (
        _prepare_generation_checkpoint(
            root,
            identity_sha256=identity,
            configured_root=configured,
        )
    )
    try:
        assert resumed_again is False
        assert (staged_again / "src/main/java/example/Conflict.java").read_text(
            encoding="utf-8"
        ) == "final class Conflict { int value = 3; }\n"
    finally:
        lease_again.close()
        _remove_generation_checkpoint(checkpoint_again)


def test_checkpoint_persistence_ignores_inert_legacy_temp_symlink(tmp_path: Path) -> None:
    checkpoint = tmp_path / ".mmm-custom-checkpoints" / ("2" * 64)
    (checkpoint / "base").mkdir(parents=True)
    staged = checkpoint / "project"
    staged.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("do not overwrite", encoding="utf-8")
    legacy_temp = checkpoint / ".checkpoint.json.tmp"
    try:
        legacy_temp.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    _persist_generation_checkpoint(
        checkpoint,
        staged,
        identity_sha256="sha256:" + ("2" * 64),
    )
    assert outside.read_text(encoding="utf-8") == "do not overwrite"
    assert legacy_temp.is_symlink()
    assert (checkpoint / "checkpoint.json").is_file()


def test_checkpoint_persistence_rejects_manifest_symlink(tmp_path: Path) -> None:
    checkpoint = tmp_path / ".mmm-custom-checkpoints" / ("3" * 64)
    (checkpoint / "base").mkdir(parents=True)
    staged = checkpoint / "project"
    staged.mkdir()
    outside = tmp_path / "outside-manifest.txt"
    outside.write_text("do not overwrite", encoding="utf-8")
    manifest = checkpoint / "checkpoint.json"
    try:
        manifest.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="manifest may not be a symlink"):
        _persist_generation_checkpoint(
            checkpoint,
            staged,
            identity_sha256="sha256:" + ("3" * 64),
        )
    assert outside.read_text(encoding="utf-8") == "do not overwrite"


def test_exact_input_rerun_resumes_hash_bound_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    checkpoint_root = tmp_path / "run/.minecraft_ai/.mmm-custom-checkpoints"
    module = ProductionModule(
        "durable_chunk",
        "custom_java",
        {"feature": "resume without replay"},
    )
    platform = adapter_for_target("1.20.1", "fabric")

    class _InterruptedRouter(_AgenticRouter):
        def generate_text(self, role, messages, **kwargs):
            assert self.workspace is not None
            request = _implement_request(messages)
            project = self.workspace / request["workspace_project_root"]
            source = project / "src/main/java/example/Durable.java"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("// preserved chunk\n", encoding="utf-8")
            raise ModelConfigurationError("simulated process-safe interruption")

    interrupted = _InterruptedRouter()
    with pytest.raises(ModelConfigurationError, match="process-safe interruption"):
        CustomModuleGenerator(
            interrupted,
            policy=ScalePolicy(model_context_bytes=4096),
            checkpoint_root=checkpoint_root,
        ).generate(
            root,
            module=module,
            minecraft_version=platform.minecraft_version,
            loader=platform.loader,
            mappings=platform.yarn_mappings,
        )
    assert not (root / "src/main/java/example/Durable.java").exists()
    assert len(list(checkpoint_root.iterdir())) == 1

    class _ResumeRouter(_AgenticRouter):
        def generate_text(self, role, messages, **kwargs):
            assert self.workspace is not None
            request = _implement_request(messages)
            assert request["checkpoint"]["resumed"] is True
            project = self.workspace / request["workspace_project_root"]
            source = project / "src/main/java/example/Durable.java"
            assert source.read_text(encoding="utf-8") == "// preserved chunk\n"
            source.write_text(
                "// preserved chunk\nfinal class Durable {}\n",
                encoding="utf-8",
            )
            return "Completed the resumed module."

    resumed = _ResumeRouter()
    result = CustomModuleGenerator(
        resumed,
        policy=ScalePolicy(model_context_bytes=4096),
        checkpoint_root=checkpoint_root,
    ).generate(
        root,
        module=module,
        minecraft_version=platform.minecraft_version,
        loader=platform.loader,
        mappings=platform.yarn_mappings,
    )

    assert result["generation_checkpoint_resumed"] is True
    assert (root / "src/main/java/example/Durable.java").read_text(
        encoding="utf-8"
    ) == "// preserved chunk\nfinal class Durable {}\n"
    assert resumed.workspace is not None
    assert not resumed.workspace.exists()
    assert not checkpoint_root.exists()
