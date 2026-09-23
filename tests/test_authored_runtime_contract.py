import hashlib
import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import generation_target_compile as compile_tool
from minecraft_mod_ai import small_model_task_capsule_contract as capsules
from minecraft_mod_ai.final_artifact import _authored_feature_semantic_findings

TARGET = "src/main/java/demo/AuthoredFeature001.java"
BASE = """package demo;
import net.fabricmc.api.Environment;
import net.fabricmc.api.EnvType;
public final class AuthoredFeature001 {
    public static void initialize() { System.out.println("ready"); }
}
"""


@pytest.mark.parametrize(
    "annotation",
    [
        "@Environment(EnvType.CLIENT)",
        "@net.fabricmc.api.Environment(net.fabricmc.api.EnvType.SERVER)",
    ],
)
@pytest.mark.parametrize(
    "declaration", ["public final class", "public static void initialize"]
)
def test_side_stripped_host_surface_fails_compile_and_release_contract(
    tmp_path, monkeypatch, annotation, declaration
):
    source = tmp_path / TARGET
    source.parent.mkdir(parents=True)
    source.write_text(
        BASE.replace(declaration, annotation + "\n" + declaration), encoding="utf-8"
    )
    capsule = SimpleNamespace(
        task_id="authored_feature_001",
        primary_path=TARGET,
        primary_symbol="AuthoredFeature001",
    )
    monkeypatch.setattr(
        capsules, "_CURRENT_CAPSULE", SimpleNamespace(get=lambda: capsule)
    )
    monkeypatch.setattr(
        compile_tool,
        "GradleRunner",
        lambda *_: SimpleNamespace(
            compile_java=lambda *_: SimpleNamespace(
                to_dict=lambda: {"status": "PASS", "commands": []}
            )
        ),
    )
    receipt = compile_tool.run_generation_target_compile(tmp_path, target_path=TARGET)
    assert receipt["status"] == "FAIL"
    assert any(d["code"] == "host:authored-side-only" for d in receipt["diagnostics"])
    findings = _authored_feature_semantic_findings(
        tmp_path,
        [
            {
                "module_id": capsule.task_id,
                "path": TARGET,
                "symbol": capsule.primary_symbol,
            }
        ],
    )
    assert any("both" in finding for finding in findings)


def test_placeholder_guard_does_not_count_as_implemented_feature(tmp_path, monkeypatch):
    source = tmp_path / TARGET
    source.parent.mkdir(parents=True)
    source.write_text(
        BASE.replace(
            'System.out.println("ready");', "// MMM_AUTHORED_FEATURE_BODY_001\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        capsules,
        "_CURRENT_CAPSULE",
        SimpleNamespace(
            get=lambda: SimpleNamespace(
                task_id="authored_feature_001",
                primary_path=TARGET,
                primary_symbol="AuthoredFeature001",
            )
        ),
    )
    monkeypatch.setattr(
        compile_tool,
        "GradleRunner",
        lambda *_: SimpleNamespace(
            compile_java=lambda *_: SimpleNamespace(
                to_dict=lambda: {"status": "PASS", "commands": []}
            )
        ),
    )
    receipt = compile_tool.run_generation_target_compile(tmp_path, target_path=TARGET)
    assert receipt["status"] == "FAIL"
    assert any(d["code"] == "host:authored-placeholder" for d in receipt["diagnostics"])


def test_side_specific_helper_and_annotation_text_are_allowed(tmp_path, monkeypatch):
    source = tmp_path / TARGET
    source.parent.mkdir(parents=True)
    source.write_text(
        BASE.replace(
            'System.out.println("ready");',
            'System.out.println("@Environment(EnvType.CLIENT)");',
        ).replace(
            "\n}", "\n@Environment(EnvType.CLIENT) private static void client() {}\n}"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        capsules,
        "_CURRENT_CAPSULE",
        SimpleNamespace(
            get=lambda: SimpleNamespace(
                task_id="authored_feature_001",
                primary_path=TARGET,
                primary_symbol="AuthoredFeature001",
            )
        ),
    )
    monkeypatch.setattr(
        compile_tool,
        "GradleRunner",
        lambda *_: SimpleNamespace(
            compile_java=lambda *_: SimpleNamespace(
                to_dict=lambda: {"status": "PASS", "commands": []}
            )
        ),
    )
    assert (
        compile_tool.run_generation_target_compile(tmp_path, target_path=TARGET)[
            "status"
        ]
        == "PASS"
    )


def test_side_annotation_is_repaired_by_same_coder_before_completion(
    tmp_path, monkeypatch
):
    from minecraft_mod_ai import progress_aware_tool_loop as loop
    from minecraft_mod_ai.model_adapters import (
        GenerationRequest,
        GenerationResponse,
        ToolCall,
    )
    from minecraft_mod_ai.mutation_authority import (
        CURRENT_MUTATION_AUTHORITY,
        MutationAuthority,
    )

    source = tmp_path / TARGET
    source.parent.mkdir(parents=True)
    source.write_text(BASE, encoding="utf-8", newline="")
    anchor = {
        "kind": "symbol",
        "locator": TARGET + "#AuthoredFeature001",
        "status": "existing",
        "ownership": "host_exact_authored_lowering",
    }
    module = SimpleNamespace(
        module_id="authored_feature_001",
        kind="custom_java",
        config={
            "evidence_task": {
                "task_id": "authored_feature_001",
                "owned_anchors": [anchor],
                "production_bindings": [
                    {
                        "task_ref": "authored_feature_001",
                        "reuse_action": "fresh",
                        "owned_anchors": [anchor],
                    }
                ],
                "required_gates": ["target_compile"],
            }
        },
    )
    capsule = capsules.compile_task_capsule(module)
    monkeypatch.setattr(
        compile_tool,
        "GradleRunner",
        lambda *_: SimpleNamespace(
            compile_java=lambda *_: SimpleNamespace(
                to_dict=lambda: {"status": "PASS", "commands": []}
            )
        ),
    )
    requests = []

    class Adapter:
        def generate_turn(self, request):
            requests.append(request)
            assert len(requests) <= 2
            if len(requests) == 1:
                new = BASE.replace(
                    "public static void",
                    "@Environment(EnvType.CLIENT)\n    public static void",
                )
            else:
                assert "host:authored-side-only" in str(request.messages)
                new = ""
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id=str(len(requests)),
                        name="apply_source_edit",
                        arguments={"new": new},
                    ),
                )
            )

    class Runtime:
        workspace_root = tmp_path

        def call(self, stage, name, args):
            if name == "target_compile":
                return compile_tool.run_generation_target_compile(
                    tmp_path, target_path=TARGET
                )
            assert name == "apply_source_edit"
            before = source.read_bytes()
            current = before.decode()
            assert current.count(args["old"]) == 1
            after = current.replace(args["old"], args["new"], 1).encode()
            source.write_bytes(after)
            return {
                "schema_version": "mmm/source-patch-receipt-v1",
                "status": "APPLIED",
                "operations": [
                    {
                        "operation": "replace",
                        "path": TARGET,
                        "before_sha256": hashlib.sha256(before).hexdigest(),
                        "after_sha256": hashlib.sha256(after).hexdigest(),
                    }
                ],
            }

    capsule_token = capsules._CURRENT_CAPSULE.set(capsule)
    authority_token = CURRENT_MUTATION_AUTHORITY.set(MutationAuthority.exact((TARGET,)))
    try:
        result = loop.generate_with_tools(
            SimpleNamespace(_agent_require_fresh_evidence=False),
            config=SimpleNamespace(
                adapter="test",
                max_context=32768,
                max_input_tokens=0,
                max_new_tokens=512,
            ),
            adapter=Adapter(),
            request=GenerationRequest(
                messages=(
                    {
                        "role": "developer",
                        "content": json.dumps(capsule.to_host_authority_payload()),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "phase": "implement_module",
                                "task": "Initialize the approved feature.",
                            }
                        ),
                    },
                ),
                tools=(
                    {
                        "type": "function",
                        "function": {
                            "name": "apply_source_edit",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    },
                ),
            ),
            runtime=Runtime(),
            stage="generation",
            role="coder",
        )
    finally:
        capsules._CURRENT_CAPSULE.reset(capsule_token)
        CURRENT_MUTATION_AUTHORITY.reset(authority_token)
    assert len(requests) == 2
    assert "passed generation-time" in json.loads(result)["summary"]
    assert "@Environment" not in source.read_text(encoding="utf-8")
