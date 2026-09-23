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


@pytest.mark.parametrize("case", [
    "allowed", "other_path", "compiler_error", "side_only", "untrusted_diagnostic",
    "existing_feature", "missing_baseline", "wrong_authority",
])
def test_full_implementation_recovery_requires_fresh_owned_scaffold(case):
    from minecraft_mod_ai import progress_aware_tool_loop as loop
    from minecraft_mod_ai.mutation_authority import (
        CURRENT_MUTATION_AUTHORITY,
        MutationAuthority,
    )

    scaffold = BASE.replace('System.out.println("ready");', "// MMM_AUTHORED_FEATURE_BODY_001\n")
    diagnostic = {
        "path": TARGET, "source": "host-authored-contract",
        "code": "host:authored-placeholder", "severity": 1, "line": 5,
    }
    if case == "other_path":
        diagnostic["path"] = "src/main/java/demo/Other.java"
    elif case == "compiler_error":
        diagnostic["code"] = "compiler.err.cant.resolve"
    elif case == "side_only":
        diagnostic["code"] = "host:authored-side-only"
    elif case == "untrusted_diagnostic":
        diagnostic["source"] = "model"
    state = loop.HostRunState(
        validation_status="FAIL", latest_verifier_tool="target_compile",
        semantic_fresh_java=case != "existing_feature",
        latest_verifier_errors=(diagnostic,),
        trusted_materialized_baseline=None if case == "missing_baseline" else (TARGET, scaffold, False),
        mutation_context=loop.TargetMutationContext(
            target_path=TARGET, source_body=scaffold, target_pinned=True,
            evidence_source="verifier_workspace_source", writable_paths=(TARGET,),
        ),
    )
    token = CURRENT_MUTATION_AUTHORITY.set(MutationAuthority.exact(
        ("src/main/java/demo/Other.java",) if case == "wrong_authority" else (TARGET,)
    ))
    try:
        assert loop._authored_implementation_recovery(state) is (case == "allowed")
    finally:
        CURRENT_MUTATION_AUTHORITY.reset(token)


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


@pytest.mark.parametrize(
    "failure", [
        "side_annotation", "unimplemented", "unimplemented_retry",
        "unimplemented_fixedpoint", "unimplemented_javac",
    ]
)
def test_authored_contract_is_repaired_by_same_coder_before_completion(
    tmp_path, monkeypatch, failure
):
    import shutil
    import subprocess
    from contextlib import nullcontext

    import jsonschema

    from minecraft_mod_ai import progress_aware_tool_loop as loop
    from minecraft_mod_ai.model_adapters import (
        GenerationRequest,
        GenerationResponse,
        ModelConfigurationError,
        ToolCall,
    )
    from minecraft_mod_ai.mutation_authority import (
        CURRENT_MUTATION_AUTHORITY,
        MutationAuthority,
    )

    source = tmp_path / TARGET
    javac = shutil.which("javac") if failure == "unimplemented_javac" else None
    java = shutil.which("java") if failure == "unimplemented_javac" else None
    if failure == "unimplemented_javac" and not (javac and java):
        pytest.skip("Java compiler/runtime is not installed")
    source.parent.mkdir(parents=True)
    baseline = BASE if failure == "side_annotation" else """package demo;
/** Host-owned authored feature slot 1/13. */
public final class AuthoredFeature001 {
    private AuthoredFeature001() {}
    public static void initialize() {
        // MMM_AUTHORED_FEATURE_BODY_001
    }
}
"""
    implemented = """package demo;
public final class AuthoredFeature001 {
    private AuthoredFeature001() {}
    private static int miningRange;
    public static void initialize() { miningRange = 5; }
    public static boolean canMine(int distance) {
        return distance >= 1 && distance <= miningRange;
    }
    public static void main(String[] args) {
        initialize();
        System.out.print(canMine(5) + ":" + canMine(6));
    }
}
"""
    source.write_text(baseline, encoding="utf-8", newline="")
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
    compile_calls = []

    def compile_java(*_):
        if javac:
            compiled = subprocess.run(
                [javac, "-d", str(tmp_path / "classes"), str(source)],
                capture_output=True, text=True, timeout=30, check=False,
            )
            assert compiled.returncode == 0, compiled.stderr
        compile_calls.append(source.read_text(encoding="utf-8"))
        return SimpleNamespace(to_dict=lambda: {"status": "PASS", "commands": []})

    monkeypatch.setattr(
        compile_tool,
        "GradleRunner",
        lambda *_: SimpleNamespace(compile_java=compile_java),
    )
    requests = []
    receipts = []

    class Adapter:
        def generate_turn(self, request):
            requests.append(request)
            assert len(requests) <= (4 if failure == "unimplemented_fixedpoint" else
                                     3 if failure == "unimplemented_retry" else 2)
            if failure.startswith("unimplemented"):
                new = baseline.rstrip() if len(requests) == 1 else implemented
                if ((failure == "unimplemented_retry" and len(requests) == 2)
                        or (failure == "unimplemented_fixedpoint" and len(requests) > 1)):
                    new = baseline.replace("slot 1/13", "slot 1 of 13")
                if len(requests) > 1:
                    assert "host:authored-placeholder" in str(request.messages)
                    assert "Mining range must be 1 through 5" in str(request.messages)
                    parameters = request.tools[0]["function"]["parameters"]
                    jsonschema.validate({"new": new}, parameters)
                    assert request.metadata["mmm_output_token_ceiling"] == 4096
                    assert "Never regenerate the complete source file" not in str(request.messages)
            elif len(requests) == 1:
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
                receipt = compile_tool.run_generation_target_compile(
                    tmp_path, target_path=TARGET
                )
                receipts.append(receipt["status"])
                return receipt
            assert name == "apply_source_edit"
            assert args["path"] == TARGET
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
    expected = (
        pytest.raises(ModelConfigurationError, match="FIXED_POINT")
        if failure == "unimplemented_fixedpoint" else nullcontext()
    )
    try:
        with expected:
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
                                    "task": "Initialize the approved feature. Mining range must be 1 through 5.",
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
                    metadata={"mmm_output_token_ceiling": 4096},
                ),
                runtime=Runtime(),
                stage="generation",
                role="coder",
            )
    finally:
        capsules._CURRENT_CAPSULE.reset(capsule_token)
        CURRENT_MUTATION_AUTHORITY.reset(authority_token)
    if failure == "unimplemented_fixedpoint":
        assert receipts and set(receipts) == {"FAIL"}
        assert not compile_calls
        assert source.read_text(encoding="utf-8") == baseline.rstrip()
        return
    assert len(requests) == (3 if failure == "unimplemented_retry" else 2)
    assert receipts == ["FAIL"] * (len(requests) - 1) + ["PASS"]
    assert len(compile_calls) == 1
    assert "passed generation-time" in json.loads(result)["summary"]
    assert "@Environment" not in source.read_text(encoding="utf-8")
    if failure.startswith("unimplemented"):
        assert source.read_text(encoding="utf-8") == implemented
    if java:
        executed = subprocess.run(
            [java, "-cp", str(tmp_path / "classes"), "demo.AuthoredFeature001"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        assert executed.returncode == 0, executed.stderr
        assert executed.stdout == "true:false"
