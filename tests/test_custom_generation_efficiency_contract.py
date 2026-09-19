from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import custom_module_generator as generator
from minecraft_mod_ai.custom_generation_search_contract import (
    _ResearchEvidenceRouter,
    _width,
)


def _complex_module():
    return SimpleNamespace(
        kind="custom_java",
        config={
            "network": True,
            "runtime": True,
            "payload": "x" * 2500,
        },
        depends_on=("core", "api"),
        required_gates=("jdt", "runtime"),
    )


def test_auto_custom_search_does_not_duplicate_single_native_lane(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_CUSTOM_SEARCH_WIDTH", "3")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    assert _width(_complex_module()) == 1


def test_explicit_custom_search_remains_user_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "on")
    monkeypatch.setenv("MMM_CUSTOM_SEARCH_WIDTH", "2")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    assert _width(_complex_module()) == 2


def test_custom_generation_receipt_preserves_only_plan_declared_gates() -> None:
    module = SimpleNamespace(required_gates=("target_compile", "target_compile"))
    assert generator._receipt_required_gates(module) == ["target_compile"]


def test_research_router_preserves_fresh_evidence_binding(monkeypatch, tmp_path) -> None:
    class Router:
        def __init__(self) -> None:
            self.binds = []
            self.calls = []

        def bind_agent_workspace(self, root, *, require_fresh_evidence=False):
            self.binds.append((root, require_fresh_evidence))
            return self

        def generate_text(self, role, messages, **kwargs):
            self.calls.append((role, messages, dict(kwargs)))
            return "ok"

    base = Router()
    proxy = _ResearchEvidenceRouter(
        base,
        owner=SimpleNamespace(policy=None, _cached_index=None, _cached_root=None),
        project_root=tmp_path,
        module=_complex_module(),
        minecraft_version="1.20.1",
        loader="fabric",
        mappings="1.20.1+build.1",
    )
    proxy.bind_agent_workspace(tmp_path, require_fresh_evidence=True)
    assert base.binds == [(tmp_path, True)]

    messages = ({"role": "user", "content": "plan"},)
    result = proxy.generate_text(
        "planner",
        messages,
        response_format="json",
        enable_tools=True,
    )
    assert result == "ok"
    assert base.calls[-1][1] == messages
    assert base.calls[-1][2]["enable_tools"] is True


def test_post_generation_stage_snapshot_is_reused_for_checkpoint_and_diff(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    base_root = checkpoint_root / "base"
    staged_root = checkpoint_root / "project"
    base_source = base_root / "src/main/java/example/Feature.java"
    staged_source = staged_root / "src/main/java/example/Feature.java"
    base_source.parent.mkdir(parents=True)
    staged_source.parent.mkdir(parents=True)
    base_source.write_text("final class Feature {}\n", encoding="utf-8")
    staged_source.write_text("final class Feature { int value; }\n", encoding="utf-8")

    real_digest = generator.content_digest
    calls = {"base": 0, "stage": 0}

    def counted_digest(path):
        resolved = Path(path).resolve()
        if resolved.is_relative_to(staged_root.resolve()):
            calls["stage"] += 1
        elif resolved.is_relative_to(base_root.resolve()):
            calls["base"] += 1
        return real_digest(path)

    monkeypatch.setattr(generator, "content_digest", counted_digest)

    stage_tree_sha256, after = generator._stage_tree_snapshot(staged_root)
    assert calls == {"base": 0, "stage": 1}

    generator._persist_generation_checkpoint(
        checkpoint_root,
        staged_root,
        identity_sha256="sha256:" + "a" * 64,
        stage_tree_sha256=stage_tree_sha256,
    )
    assert calls == {"base": 1, "stage": 1}

    operations, touched, discarded = generator._collect_staged_operations(
        base_root,
        staged_root,
        {"src/main/java/example/Feature.java": "sha256:" + "b" * 64},
        after=after,
    )
    assert calls == {"base": 1, "stage": 1}
    assert touched == ["src/main/java/example/Feature.java"]
    assert discarded == []
    assert operations[0]["operation"] == "replace"
