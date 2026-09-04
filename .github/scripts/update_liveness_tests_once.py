from __future__ import annotations

from pathlib import Path
import re


def replace_regex(path: Path, pattern: str, replacement: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    new, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise SystemExit(f"{label}: replacement count={count}")
    path.write_text(new, encoding="utf-8")


output_test = Path("tests/test_custom_module_output_continuation.py")
replace_regex(
    output_test,
    r"def test_output_boundary_loop_is_tool_enabled_checkpointed_and_fixed_point_bounded\(\) -> None:\n.*\Z",
    r'''def test_output_boundary_is_tool_enabled_checkpointed_and_single_owner() -> None:
    source = inspect.getsource(CustomModuleGenerator.generate)
    exception_block = source[source.index("except BaseException as exc:") :]

    assert "enable_tools=True" in source
    assert "enable_tools=False" not in source
    assert exception_block.index("_persist_generation_checkpoint(") < exception_block.index(
        "boundary_kind = completion_boundary_kind(exc)"
    )
    assert "seen_output_states: set[str] = set()" not in source
    assert "_output_exhaustion_continuation_messages(" not in source
    assert "refusing an outer continuation" in source
''',
    "single-owner output contract",
)

structural_test = Path("tests/test_custom_module_structural_repair_efficiency.py")
replace_regex(
    structural_test,
    r"def test_output_exhaustion_continues_from_same_staged_workspace_until_success\(\n    tmp_path: Path,\n\) -> None:\n.*?(?=def test_context_pressure_raises_directly_without_masking)",
    r'''def test_output_exhaustion_preserves_checkpoint_without_outer_state_restart(
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
            target.write_text("// chunk 1\n", encoding="utf-8")
            raise _output_exhausted_error()

    router = _ChunkedRouter()
    target = adapter_for_target("1.20.1", "fabric")
    with pytest.raises(
        CustomModuleGenerationError,
        match="refusing an outer continuation",
    ):
        CustomModuleGenerator(
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

    assert len(router.calls) == 1
    assert len(set(router.bound_workspaces)) == 1
    assert not (root / "src/main/java/example/Chunked.java").exists()
    assert router.workspace is not None
    assert (router.workspace / "src/main/java/example/Chunked.java").read_text(
        encoding="utf-8"
    ) == "// chunk 1\n"


''',
    "outer continuation contract",
)

unified_test = Path("tests/test_unified_host_execution_loop.py")
text = unified_test.read_text(encoding="utf-8")
old = '    assert _filter_tools_for_phase(mixed, LoopPhase.VERIFY, role="coder") == (read_tool, verify_tool)\n'
new = '    assert _filter_tools_for_phase(mixed, LoopPhase.VERIFY, role="coder") == (verify_tool,)\n'
if old not in text:
    raise SystemExit("verify filter expectation missing")
unified_test.write_text(text.replace(old, new, 1), encoding="utf-8")
