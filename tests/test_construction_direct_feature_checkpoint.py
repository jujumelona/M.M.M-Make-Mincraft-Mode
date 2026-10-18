"""A whole feature is resumable only after exact-target Java symbol validation."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import construction_direct_writer as writer
from minecraft_mod_ai import javac_bridge
from minecraft_mod_ai import compiled_feature_patch


def _context(tmp_path):
    return {
        "project_root": str(tmp_path),
        "target": {"minecraft_version": "26.2", "loader": "fabric", "java_version": 25},
        "java_classpath": ["/pinned/target-26.2.jar"],
        "namespace": "mojang",
        "prerequisite_hashes": {},
        "authoring_contract": {},
        "symbols": {},
        "source_pages": {},
    }


def _inputs():
    return (
        {"target_path": "src/main/java/dev/demo/DemoMod.java"},
        [{"behavior_id": "register_alien",
          "source_quote": "register alien",
          "intent": "Register the alien entity",
          "effects": "Alien exists in the server entity registry",
          "trigger": "startup"}],
    )


def _stubs(monkeypatch):
    class Index:
        def __init__(self, *_args, **_kwargs):
            pass
        def include_source(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(writer, "ConstructionEvidenceIndex", Index)
    monkeypatch.setattr(writer, "_evidence",
                        lambda *_args: {"target_api": [], "source": []})
    monkeypatch.setattr(writer, "_parse", lambda *_args: None)
    monkeypatch.setattr(compiled_feature_patch, "replay_compiled_repair",
                        lambda _root, _path, source: source)


def test_invalid_entire_feature_never_enters_checkpoint(tmp_path, monkeypatch):
    _stubs(monkeypatch)
    monkeypatch.setattr(javac_bridge, "analyze_java",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            ValueError("JAVA_ANALYSIS_FAILED: MissingAlienClass")))

    class Router:
        calls = 0
        def generate_java_source(self, _role, _messages, *, source_frame, **_kwargs):
            self.calls += 1
            return (source_frame["prefix"] +
                    "new MissingAlienClass();" +
                    source_frame["suffix"])

    spec, behaviors = _inputs()
    with pytest.raises(ValueError, match="CONSTRUCTION_DIRECT_FEATURE_TARGET_SYMBOL_INVALID"):
        writer.compose_feature(
            Router(), spec, _context(tmp_path), behaviors,
            package="dev.demo", name="DemoMod", owner="dev.demo.DemoMod",
        )
    assert list((tmp_path / ".mmm" / "construction_direct").glob("*.json")) == []


def test_validated_feature_can_resume_without_new_inference(tmp_path, monkeypatch):
    _stubs(monkeypatch)
    compile_calls = []
    def verified(source, **kwargs):
        compile_calls.append((source, kwargs))
        return []
    monkeypatch.setattr(javac_bridge, "analyze_java", verified)

    class Router:
        calls = 0
        def generate_java_source(self, _role, _messages, *, source_frame, **_kwargs):
            self.calls += 1
            return source_frame["prefix"] + "System.out.println(42);" + source_frame["suffix"]

    router = Router()
    spec, behaviors = _inputs()
    ctx = _context(tmp_path)
    result = writer.compose_feature(router, spec, ctx, behaviors,
                                    package="dev.demo", name="DemoMod",
                                    owner="dev.demo.DemoMod")
    assert router.calls == 1
    assert "System.out.println(42);" in result
    assert len(compile_calls) == 1
    assert compile_calls[0][1]["classpath"] == ["/pinned/target-26.2.jar"]
    assert compile_calls[0][1]["java_version"] == "25"
    receipts = list((tmp_path / ".mmm" / "construction_direct").glob("*.json"))
    assert len(receipts) == 1

    again = writer.compose_feature(router, spec, ctx, behaviors,
                                   package="dev.demo", name="DemoMod",
                                   owner="dev.demo.DemoMod")
    assert again == result
    assert router.calls == 1
    assert len(compile_calls) == 2  # Old bytes must be rechecked on every resume
